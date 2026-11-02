"""Hand-held capture for Gaussian Splatting: the camera in your hand instead of on the arm.

A robot scan knows where the camera was for every picture. A hand-held one does not, so the poses
have to be solved from the pictures themselves (structure from motion, scripts/solve_poses.py).
That only works when the pictures overlap, and a splat is only as good as the views it was given:
the whole fabric from far away, closer views, close-ups, and tilted views from every side.

This module is what tells the person holding the camera how far along that is, while they shoot.
Every kept picture is matched (SIFT, the features the pose solve itself uses) against the earlier
ones. A knit lies flat, so a match is a homography, and a homography back to the first picture -
taken of the whole fabric, straight down - says how far away this one was relative to it, how
tilted it is, and from which side it looks. Those are estimates; the pose solve measures the same
three things properly, and both are judged by the same targets (readiness).
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

import robot_camera

# Distance bands, relative to the first picture (the whole fabric, straight down): label, what to
# shoot, the band's lower and upper bound, and how many pictures it needs. Generous for a flat
# piece of cloth; splatting guidance for small objects is 40-150 views with steady overlap.
DISTANCE_BANDS = (
    ("far",    "the whole fabric in view",                 0.75, math.inf, 10),
    ("middle", "about half the first distance",            0.40, 0.75,     15),
    ("close",  "less than 40% of it - stitches fill the frame", 0.0, 0.40,  15),
)
# Sides: tilted views are counted in this many sectors around the fabric, and this many must have
# one. A view tilted less than MIN_SIDE_TILT_DEG looks straight down and is from no side.
SIDE_SECTORS = 8
SIDES_NEEDED = 6
MIN_SIDE_TILT_DEG = 20.0
# Of all kept pictures, at least this share must match an earlier one: an unmatched picture is one
# the pose solve will most likely drop, and it carries no estimate of where it was taken.
MIN_LINKED_SHARE = 0.9
# After the pose solve: at least this share of the pictures must have been placed, at no more
# than this mean reprojection error.
MIN_REGISTERED_SHARE = 0.9
MAX_REPROJECTION_ERROR_PX = 1.5


# Matching works on a copy at most this wide; SIFT this many features per picture. A match needs
# this many RANSAC inliers. A picture whose best match moved its corners less than DUPLICATE_SHIFT
# of the width adds nothing and is not kept.
WORKING_WIDTH = 800
SIFT_FEATURES = 2000
RATIO_TEST = 0.75
MIN_LINK_INLIERS = 30
RANSAC_PX = 3.0
DUPLICATE_SHIFT = 0.04
# Matched against the first picture and at most this many of the latest - enough to find the
# neighbour of a hand that moved smoothly, without the cost growing with the session.
RECENT_TO_MATCH = 40
# A picture is the sharpest frame of the last BURST_SECONDS: a hand pressing Capture, or simply
# holding a camera, shakes it, and the steadiest moment of a short burst is rarely blurred.
BURST_SECONDS = 0.6
BURST_SHARPNESS_WIDTH = 320
# Blur is judged against the camera itself, not a fixed number: how sharp a frame can be depends on
# the camera, its resolution and the distance. A picture less sharp than BLUR_SHARE of the sharpest
# frame of the last PEAK_SECONDS was taken moving; below MIN_SHARPNESS there is nothing to match.
BLUR_SHARE = 0.6
PEAK_SECONDS = 3.0
MIN_SHARPNESS = 15.0
# While shooting, the live view is matched against the last kept picture this often. Automatic
# pictures are taken when it has moved on by AUTO_MOVE_SHIFT of the frame but still overlaps it:
# far enough to add a view, near enough to be placed next to the last one.
PROBE_INTERVAL = 0.4
AUTO_MOVE_SHIFT = 0.15
# How the live overlap reads: inlier counts from which it is called strong, and good.
OVERLAP_STRONG = 150
OVERLAP_GOOD = 60

# One target: what it asks for, how far along as text and as a fraction, and whether it is met.
Check = namedtuple("Check", "label progress fraction done")


def _count(label, have, need):
    return Check(label, f"{have} / {need}", min(1.0, have / need) if need else 1.0, have >= need)


def readiness(rows, sides_needed: bool = True) -> list[Check]:
    """How far `rows` (one per picture, with relative_distance, tilt_from_vertical_deg, azimuth_deg) are from enough.

    Rows are either the live estimates of a session or what the pose solve measured; the targets are
    the same, which is what makes "enough" before the solve and after it the same claim. Without
    `sides_needed` there is no target for tilted views: a splat of top-down views only is flatter in
    depth, which is a choice the person shooting may make.
    """
    placed = [row for row in rows if row.get("relative_distance") is not None]
    checks = [_count(f"{label} - {hint}", sum(low <= row["relative_distance"] < high for row in placed), need)
              for label, hint, low, high, need in DISTANCE_BANDS]
    sides = set() if not sides_needed else {int((float(row["azimuth_deg"]) % 360.0) // (360.0 / SIDE_SECTORS)) for row in placed
             if row.get("azimuth_deg") is not None and float(row.get("tilt_from_vertical_deg") or 0.0) >= MIN_SIDE_TILT_DEG}
    if sides_needed:
        checks.append(_count(f"sides - tilted {MIN_SIDE_TILT_DEG:.0f}+ deg, from {SIDES_NEEDED} of {SIDE_SECTORS} directions around the fabric",
                             len(sides), SIDES_NEEDED))
    linked = _count("overlap - each picture matches an earlier one", len(placed), max(1, math.ceil(MIN_LINKED_SHARE * len(rows))))
    checks.append(linked._replace(progress=f"{len(placed)} / {len(rows)}"))
    return checks


def solve_checks(report: dict) -> list[Check]:
    """What only the pose solve can say: how many pictures it placed, and how well they agree."""
    total, registered = int(report.get("total", 0)), int(report.get("registered", 0))
    error = report.get("mean_reprojection_error_px")
    placed = _count("placed by the pose solve", registered, max(1, math.ceil(MIN_REGISTERED_SHARE * total)))
    precise = error is not None and float(error) <= MAX_REPROJECTION_ERROR_PX
    return [placed._replace(progress=f"{registered} / {total}"),
            Check(f"precision - reprojection error at most {MAX_REPROJECTION_ERROR_PX} px", "-" if error is None else f"{float(error):.2f} px",
                  1.0 if precise else (0.0 if error is None else MAX_REPROJECTION_ERROR_PX / float(error)), precise)]


# ============================================================================
# Matching and what a homography says about the view
# ============================================================================

def working_intrinsics(width: int, height: int, fov_degrees: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(float(fov_degrees)) / 2.0))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def features(image: Image.Image):
    """SIFT keypoints (in working pixels, at most WORKING_WIDTH across) and descriptors of `image`, and the working size."""
    import cv2

    grey = image.convert("L")
    width = min(WORKING_WIDTH, grey.size[0])
    size = (width, max(1, int(round(grey.size[1] * width / grey.size[0]))))
    pixels = np.asarray(grey.resize(size, Image.BILINEAR), dtype=np.uint8)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(pixels, None)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    return points, descriptors, size


def match(earlier, later):
    """The homography taking `earlier`'s working pixels to `later`'s, and its inlier count; (None, 0) for no match."""
    import cv2

    (points_a, descriptors_a, _), (points_b, descriptors_b, _) = earlier, later
    if descriptors_a is None or descriptors_b is None or len(points_a) < 8 or len(points_b) < 8:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_a, descriptors_b, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
    if len(good) < MIN_LINK_INLIERS:
        return None, 0
    source = points_a[[m.queryIdx for m in good]]
    target = points_b[[m.trainIdx for m in good]]
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PX)
    count = 0 if inliers is None else int(inliers.sum())
    return (homography, count) if homography is not None and count >= MIN_LINK_INLIERS else (None, 0)


def corner_shift(homography, size) -> float:
    """How far `homography` moves the frame's corners, on average, as a share of the width."""
    width, height = size
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=float).T
    moved = homography @ corners
    moved = moved[:2] / moved[2]
    return float(np.linalg.norm(moved - corners[:2], axis=0).mean() / width)


def view_from_first(first_to_this, size, intrinsics) -> dict:
    """Distance, tilt and side of a view, from the homography taking the first picture to it.

    The first picture is taken to look straight down, so its pixels are the fabric plane up to a
    scale. Relative distance is how much fabric one pixel of this view spans at its centre, against
    the first's, with the stretch of the tilt taken out - so it is the distance along the view's
    centre ray, as a share of the first's. The plane's vanishing line in this view (the first's line at infinity, carried over)
    gives the fabric's normal in this camera, hence the tilt; this view's own line at infinity,
    carried back onto the plane, lies on the camera's side, hence the side it looks from.
    """
    width, height = size
    this_to_first = np.linalg.inv(first_to_this)
    centre = np.array([width / 2.0, height / 2.0, 1.0])

    def onto_first(x, y):
        point = this_to_first @ np.array([x, y, 1.0])
        return point[:2] / point[2]

    step = 1.0
    origin = onto_first(*centre[:2])
    jacobian = np.column_stack([onto_first(centre[0] + step, centre[1]) - origin, onto_first(centre[0], centre[1] + step) - origin])
    normal = intrinsics.T @ this_to_first[2]
    cos_tilt = min(1.0, abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1e-12))
    tilt = math.degrees(math.acos(cos_tilt))
    relative_distance = math.sqrt(abs(float(np.linalg.det(jacobian))) * cos_tilt)

    horizon = first_to_this[2]
    towards_camera = -math.copysign(1.0, float(horizon @ np.array([*origin, 1.0]))) * horizon[:2]
    azimuth = math.degrees(math.atan2(float(towards_camera[1]), float(towards_camera[0])))
    return {"relative_distance": relative_distance, "tilt_from_vertical_deg": tilt, "azimuth_deg": azimuth if tilt >= MIN_SIDE_TILT_DEG else None}


def distance_band(relative_distance) -> str:
    if relative_distance is None:
        return ""
    return next((label for label, _, low, high, _ in DISTANCE_BANDS if low <= relative_distance < high), "")



# ============================================================================
# A session
# ============================================================================

# What happened to one attempt at a picture: when, which kind (kept, unlinked - kept but matching
# nothing, skipped), and what to tell the person holding the camera.
Event = namedtuple("Event", "at kind text")


class FreeCaptureSession:
    """One hand-held capture: its folder, its pictures with what was estimated about each, the worker
    that judges a new picture against the rest, and the probe that keeps the live view matched against
    the last picture - so the panel never waits on either."""

    def __init__(self, folder: Path, fov_degrees: float):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fov_degrees = float(fov_degrees)
        self.records: list[dict] = []
        self.events: deque = deque(maxlen=12)
        self.skipped = 0
        # The live view against the last kept picture: (inliers, how far it has moved on), or None
        # until there is a picture and a frame to compare.
        self.overlap = None
        self._features: list = []
        self._from_first: list = []
        self._burst: deque = deque()
        self._recent: deque = deque()
        self._burst_index = -1
        self._newest = None
        self._last_take = 0.0
        # Numbers the picture files and never goes back, so an undone picture's name is not reused.
        self._shots = 0
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._threads = [threading.Thread(target=self._work, daemon=True, name="free-capture"),
                         threading.Thread(target=self._probe, daemon=True, name="free-capture-probe")]
        for thread in self._threads:
            thread.start()

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def busy(self) -> bool:
        return self._jobs.unfinished_tasks > 0

    @property
    def last_event(self):
        return self.events[-1] if self.events else None

    def close(self) -> None:
        """Stop both threads and wait for them: one left inside OpenCV when the app exits aborts the process."""
        self._closed.set()
        for thread in self._threads:
            thread.join(timeout=3.0)

    def watch(self, frame) -> None:
        """Remember the camera's newest frame, for the burst a picture is chosen from and for the probe."""
        if frame is None or frame.index == self._burst_index:
            return
        self._burst_index = frame.index
        self._newest = frame
        now = time.monotonic()
        score = robot_camera.sharpness(frame.image, BURST_SHARPNESS_WIDTH)
        self._burst.append((score, now, frame))
        self._recent.append((score, now))
        while self._burst and now - self._burst[0][1] > BURST_SECONDS:
            self._burst.popleft()
        while now - self._recent[0][1] > PEAK_SECONDS:
            self._recent.popleft()

    def take(self) -> None:
        """Hand the sharpest frame of the last burst to the worker, which keeps it or says why not."""
        if not self._burst:
            self._tell("skipped", "No frame from the camera yet")
            return
        score, _, frame = max(self._burst, key=lambda entry: entry[0])
        try:
            self._jobs.put_nowait((frame, score, max(entry[0] for entry in self._recent)))
        except queue.Full:
            return
        self._last_take = time.monotonic()
        self._burst.clear()

    def auto_ready(self, interval: float) -> tuple[bool, str]:
        """Whether an automatic picture is due now, and if not, what it is waiting for."""
        if not self.records:
            return False, "Take the first picture yourself: the whole fabric, straight down"
        if self.busy:
            return False, "Checking the last picture..."
        wait = interval - (time.monotonic() - self._last_take)
        if wait > 0:
            return False, f"Next picture in {wait:.1f} s"
        if self.overlap is None:
            return False, "Comparing the view with the last picture..."
        inliers, shift = self.overlap
        if inliers == 0:
            return False, "Lost the last picture - move back towards it"
        if shift < AUTO_MOVE_SHIFT:
            return False, f"Move on - the view is {shift / AUTO_MOVE_SHIFT:.0%} of the way to the next picture"
        return True, "Taking a picture"

    def discard_last(self) -> None:
        """Take back the latest picture: its files, and everything matched against it."""
        if self.busy or not self.records:
            return
        with self._lock:
            record = self.records.pop()
            self._features.pop()
            self._from_first.pop()
        self.overlap = None
        for path in (Path(record["path"]), Path(record["path"]).with_suffix(".json")):
            path.unlink(missing_ok=True)
        self._tell("skipped", f"Removed picture {len(self.records) + 1}")

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self.records)

    def _tell(self, kind: str, text: str) -> None:
        self.events.append(Event(time.monotonic(), kind, text))
        if kind == "skipped":
            self.skipped += 1

    def _work(self) -> None:
        while not self._closed.is_set():
            try:
                frame, score, peak = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._judge(frame, score, peak)
            except Exception as exc:
                self._tell("skipped", f"Not kept: {exc}")
            finally:
                self._jobs.task_done()

    def _probe(self) -> None:
        while not self._closed.wait(PROBE_INTERVAL):
            frame = self._newest
            if frame is None or not self._features:
                self.overlap = None
                continue
            try:
                last = self._features[-1]
                found = features(frame.image)
                homography, inliers = match(last, found)
                self.overlap = (inliers, 0.0 if homography is None else corner_shift(homography, found[2]))
            except Exception:
                self.overlap = None

    def _judge(self, frame, score: float = 0.0, peak: float = 0.0) -> None:
        """Keep `frame` or say why not. `score` is its burst sharpness and `peak` the sharpest the camera
        gave of late, both at BURST_SHARPNESS_WIDTH."""
        sharpness = robot_camera.sharpness(frame.image)
        if score < BLUR_SHARE * peak:
            self._tell("skipped", f"Skipped: blurred - {score / peak:.0%} as sharp as the camera was a moment ago. Hold it still for a second")
            return
        if sharpness < MIN_SHARPNESS:
            self._tell("skipped", f"Skipped: blurred or featureless (sharpness {sharpness:.0f}) - hold still, or check the focus")
            return
        found = features(frame.image)
        size = found[2]
        candidates = sorted({0, *range(max(0, len(self._features) - RECENT_TO_MATCH), len(self._features))}) if self._features else []
        best, best_homography, best_inliers = None, None, 0
        for index in candidates:
            homography, inliers = match(self._features[index], found)
            if inliers > best_inliers:
                best, best_homography, best_inliers = index, homography, inliers
        if best is not None and corner_shift(best_homography, size) < DUPLICATE_SHIFT:
            self._tell("skipped", f"Skipped: same view as picture {best + 1} - move to a new spot")
            return

        first_to_this = np.eye(3) if not self._features else (
            None if best is None or self._from_first[best] is None else best_homography @ self._from_first[best])
        view = (view_from_first(first_to_this, size, working_intrinsics(*size, self.fov_degrees)) if first_to_this is not None
                else {"relative_distance": None, "tilt_from_vertical_deg": None, "azimuth_deg": None})

        index = len(self.records)
        path = self.folder / f"free_{self._shots:03d}.png"
        self._shots += 1
        frame.image.save(path)
        record = {
            "path": str(path),
            "free_capture": True,
            "capture_view": f"hand-held {index + 1}",
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sharpness": sharpness,
            "linked_to": None if best is None else best + 1,
            "link_inliers": best_inliers,
            "distance_band": distance_band(view["relative_distance"]),
            **view,
            **frame.metadata(),
        }
        path.with_suffix(".json").write_text(json.dumps(record, indent=2))
        with self._lock:
            self.records.append(record)
            self._features.append(found)
            self._from_first.append(first_to_this)
        self.overlap = None
        if index == 0:
            self._tell("kept", "Picture 1 kept. Now move round the fabric and closer - a little at a time")
        elif first_to_this is None:
            self._tell("unlinked", f"Picture {index + 1} kept, but it matches no earlier picture - "
                                   "move back towards the last one (or Undo it)")
        else:
            tilt = view["tilt_from_vertical_deg"]
            self._tell("kept", f"Picture {index + 1} kept - {record['distance_band'] or 'outside the bands'}, "
                               f"{view['relative_distance']:.2f} of the first distance, tilted {tilt:.0f} deg")
