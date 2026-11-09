"""How much of a capture is fabric, by MobileSAM segmentation.

The capture path already keeps every automatic frame on the taught fabric outline, so this
is the second opinion: it looks at the picture itself. A frame that is mostly table or room
is flagged at capture time and left out of the colour analysis rather than averaged in.

MobileSAM (Zhang et al., MIT licence) runs here as two ONNX graphs on onnxruntime's CPU
provider: an image encoder, run once per frame, and SAM's mask decoder, prompted with one
point. The prompt is the frame's centre, because a planned view is aimed at the cloth - the
plan guarantees the centre is fabric, and the mask says how far the fabric extends from it.
SAM answers a point with nested masks (a stripe, a band of stripes, the whole cloth); the
largest confident one is the fabric, since a knit's stripes are its parts.

The model files are not shipped. ``scripts/fetch_mobilesam.py`` downloads them, pinned and
hash-checked, into ``paths.MOBILESAM_DIR``; until then ``available()`` is False and the scan
runs without the check.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import numpy as np
from PIL import Image

import paths

ENCODER_FILE = "mobile_sam_image_encoder.onnx"
DECODER_FILE = "sam_mask_decoder_multi.onnx"
LABEL_STRIP_HEIGHT = 58
LABEL_STRIP_MIN_IMAGE_HEIGHT = 70

# SAM's working resolution and its ImageNet normalisation.
ENCODER_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# A mask whose own predicted quality is below this is not trusted to be the cloth.
MIN_MASK_CONFIDENCE = 0.6
# Frames are segmented at this width: coverage is a fraction, and the encoder resizes anyway.
WORKING_WIDTH = 640
INFERENCE_THREADS = 2


def missing_files() -> list[str]:
    return [name for name in (ENCODER_FILE, DECODER_FILE) if not (paths.MOBILESAM_DIR / name).is_file()]


def available() -> bool:
    """Whether the check can run: onnxruntime installed and both model files fetched."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return not missing_files()


def describe() -> str:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return "onnxruntime is not installed - run uv sync"
    missing = missing_files()
    if missing:
        return "model files not fetched (" + ", ".join(missing) + ") - run: uv run python scripts/fetch_mobilesam.py"
    return "MobileSAM ready"


@functools.lru_cache(maxsize=1)
def _sessions():
    import onnxruntime

    options = onnxruntime.SessionOptions()
    # Not every core: onnxruntime takes them all by default, and a laptop pinned for a second
    # starves the RTDE stream, which the controller then drops mid-scan.
    options.intra_op_num_threads = INFERENCE_THREADS
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    return (onnxruntime.InferenceSession(str(paths.MOBILESAM_DIR / ENCODER_FILE), options, providers=providers),
            onnxruntime.InferenceSession(str(paths.MOBILESAM_DIR / DECODER_FILE), options, providers=providers))


def _encode(encoder, rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """The image embedding, and the scale from the frame to the encoder's 1024 grid.

    The published encoder either takes the resized frame as HxWx3 pixels and normalises and
    pads it itself, or takes the finished 1x3x1024x1024 tensor. Its input's rank says which,
    so both exports work without anyone having to know which one was fetched.
    """
    height, width = rgb.shape[:2]
    scale = ENCODER_SIZE / float(max(height, width))
    resized = np.asarray(Image.fromarray(rgb).resize((int(round(width * scale)), int(round(height * scale))), Image.BILINEAR),
                         dtype=np.float32)

    spec = encoder.get_inputs()[0]
    if len(spec.shape) == 3:
        tensor = resized
    else:
        padded = np.zeros((ENCODER_SIZE, ENCODER_SIZE, 3), dtype=np.float32)
        padded[:resized.shape[0], :resized.shape[1]] = (resized - PIXEL_MEAN) / PIXEL_STD
        tensor = padded.transpose(2, 0, 1)[None]
    return encoder.run(None, {spec.name: tensor})[0], scale


def fabric_mask(image: Image.Image, sessions=None) -> np.ndarray:
    """The fabric's pixels in `image`, at WORKING_WIDTH, as a boolean mask."""
    encoder, decoder = sessions or _sessions()
    frame = image.convert("RGB")
    if frame.size[0] > WORKING_WIDTH:
        frame = frame.resize((WORKING_WIDTH, max(1, int(round(frame.size[1] * WORKING_WIDTH / frame.size[0])))), Image.BILINEAR)
    rgb = np.asarray(frame, dtype=np.uint8)
    height, width = rgb.shape[:2]

    embedding, scale = _encode(encoder, rgb)
    # One positive point at the centre, and SAM's "no box" padding point.
    centre = np.array([[[0.5 * width * scale, 0.5 * height * scale], [0.0, 0.0]]], dtype=np.float32)
    feeds = {"image_embeddings": embedding,
             "point_coords": centre,
             "point_labels": np.array([[1.0, -1.0]], dtype=np.float32),
             "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
             "has_mask_input": np.zeros(1, dtype=np.float32),
             "orig_im_size": np.array([height, width], dtype=np.float32)}
    masks, confidence = decoder.run(None, {spec.name: feeds[spec.name] for spec in decoder.get_inputs()})[:2]

    masks, confidence = masks[0] > 0.0, np.asarray(confidence[0], dtype=float)
    trusted = [index for index in range(len(confidence)) if confidence[index] >= MIN_MASK_CONFIDENCE]
    chosen = max(trusted, key=lambda index: int(masks[index].sum())) if trusted else int(np.argmax(confidence))
    return masks[chosen]


def coverage(image: Image.Image, sessions=None) -> tuple[float, np.ndarray]:
    """The fraction of `image` that is fabric, and the mask it was measured from."""
    mask = fabric_mask(image, sessions)
    return float(mask.mean()), mask


# The texture test, for finding the cloth when nothing says where it is. A knit is edges at every
# scale a camera resolves; a table, a floor or a wall is mostly smooth. So the fabric is where
# the image's local detail is high - no model, no prompt point, and so usable before the camera
# is known to be over the cloth, which MobileSAM's centre prompt is not.
TEXTURE_WIDTH = 160
TEXTURE_WINDOW = 5
MIN_FABRIC_SHARE = 0.02


def texture_mask(image: Image.Image) -> np.ndarray:
    """Where `image` is textured like a knit, at TEXTURE_WIDTH, as a boolean mask (Otsu on local detail)."""
    grey = image.convert("L")
    grey = grey.resize((TEXTURE_WIDTH, max(1, int(round(grey.size[1] * TEXTURE_WIDTH / grey.size[0])))), Image.BILINEAR)
    pixels = np.asarray(grey, dtype=np.float32)
    laplacian = np.zeros_like(pixels)
    laplacian[1:-1, 1:-1] = np.abs(pixels[1:-1, :-2] + pixels[1:-1, 2:] + pixels[:-2, 1:-1] + pixels[2:, 1:-1] - 4.0 * pixels[1:-1, 1:-1])

    pad = TEXTURE_WINDOW // 2
    padded = np.pad(laplacian, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (TEXTURE_WINDOW, TEXTURE_WINDOW))
    energy = windows.mean(axis=(-1, -2))
    if float(energy.max()) - float(energy.min()) < 1e-6:
        return np.zeros(energy.shape, dtype=bool)

    counts, edges = np.histogram(energy, bins=64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total, weighted = counts.sum(), np.cumsum(counts * centres)
    below = np.cumsum(counts)
    spread = [(below[i] * (total - below[i]) * (weighted[i] / below[i] - (weighted[-1] - weighted[i]) / (total - below[i])) ** 2)
              if 0 < below[i] < total else 0.0 for i in range(len(counts))]
    return energy > centres[int(np.argmax(spread))]


def fabric_offset(image: Image.Image) -> tuple[np.ndarray, float]:
    """Where the fabric's middle is in `image`, as (x right, y down) fractions of the frame from its
    centre, and how much of the frame is fabric. Raises when the frame shows none."""
    mask = texture_mask(image)
    share = float(mask.mean())
    if share < MIN_FABRIC_SHARE:
        raise RuntimeError(f"no fabric in view - {share * 100:.1f}% of the frame looks like a knit")
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean() / mask.shape[1] - 0.5, rows.mean() / mask.shape[0] - 0.5]), share


def annotate(image: Image.Image, mask: np.ndarray, lines=()) -> Image.Image:
    """`image` with what was taken for fabric kept bright and the rest dimmed, a cross at the fabric's
    middle, a ring at the frame's centre, and `lines` of text - the detection made visible."""
    from PIL import ImageDraw

    frame = image.convert("RGB")
    fabric = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").resize(frame.size, Image.NEAREST)
    dimmed = Image.blend(frame, Image.new("RGB", frame.size, (20, 20, 30)), 0.65)
    shown = Image.composite(frame, dimmed, fabric)

    draw = ImageDraw.Draw(shown)
    width, height = shown.size
    ring = max(6, width // 60)
    draw.ellipse([width / 2 - ring, height / 2 - ring, width / 2 + ring, height / 2 + ring], outline=(255, 255, 255), width=2)
    if mask.any():
        rows, cols = np.nonzero(mask)
        x, y = cols.mean() / mask.shape[1] * width, rows.mean() / mask.shape[0] * height
        arm = ring * 2
        draw.line([x - arm, y, x + arm, y], fill=(80, 255, 120), width=3)
        draw.line([x, y - arm, x, y + arm], fill=(80, 255, 120), width=3)
    for index, text in enumerate(lines):
        draw.rectangle([6, 6 + index * 18, 12 + 7 * len(text), 22 + index * 18], fill=(0, 0, 0))
        draw.text((9, 8 + index * 18), text, fill=(255, 255, 255))
    return shown


def fabric_rgb_stats(image, debug_path=None):
    """Average the fabric pixels of one capture."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        return {
            "rgb": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "pixel_count": 0,
            "total_pixels": 0,
            "method": "empty",
        }

    y_offset = LABEL_STRIP_HEIGHT if rgb.shape[0] > LABEL_STRIP_MIN_IMAGE_HEIGHT else 0
    work = rgb[y_offset:, :, :]
    h, w = work.shape[:2]
    total_pixels = int(h * w)

    def finish(mask, method):
        mask = np.asarray(mask, dtype=bool)
        if int(mask.sum()) < 1:
            fabric = work.reshape(-1, 3)
            mask = np.ones((h, w), dtype=bool)
            method = "full frame fallback"
        else:
            fabric = work[mask]

        saved_debug_path = None
        if debug_path is not None and int(mask.sum()) > 0:
            try:
                ys_mask, xs_mask = np.where(mask)
                x0, x1 = int(xs_mask.min()), int(xs_mask.max()) + 1
                y0, y1 = int(ys_mask.min()), int(ys_mask.max()) + 1
                crop_rgb = np.clip(work[y0:y1, x0:x1], 0, 255).astype(np.uint8)
                crop_mask = mask[y0:y1, x0:x1]
                rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = crop_rgb
                rgba[:, :, 3] = np.where(crop_mask, 255, 0).astype(np.uint8)
                debug_out = Path(debug_path)
                debug_out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba, "RGBA").save(debug_out)
                saved_debug_path = str(debug_out)
            except Exception:
                saved_debug_path = None

        return {
            "rgb": fabric.mean(axis=0),
            "pixel_count": int(len(fabric)),
            "total_pixels": total_pixels,
            "method": method,
            "debug_path": saved_debug_path,
        }

    green = (
        (work[:, :, 1] > 165.0)
        & (work[:, :, 0] < 90.0)
        & (work[:, :, 2] < 135.0)
    )
    ys, xs = np.where(green)
    if xs.size >= 24:
        pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(float(len(pts) - 1), 1.0)
        _, vecs = np.linalg.eigh(cov)
        axes = vecs[:, ::-1].astype(np.float32)
        outline_proj = centered @ axes
        lo = outline_proj.min(axis=0)
        hi = outline_proj.max(axis=0)
        margin = 4.0
        if np.all((hi - lo) > margin * 3.0):
            yy, xx = np.mgrid[0:h, 0:w]
            grid = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float32)
            proj = (grid - center) @ axes
            mask = (
                (proj[:, 0] >= lo[0] + margin)
                & (proj[:, 0] <= hi[0] - margin)
                & (proj[:, 1] >= lo[1] + margin)
                & (proj[:, 1] <= hi[1] - margin)
            ).reshape(h, w)
            mask &= ~green
            if int(mask.sum()) >= 32:
                brightness = work.max(axis=2)
                saturation = work.max(axis=2) - work.min(axis=2)
                color_mask = mask & (brightness > 42.0) & (saturation > 16.0)
                if int(color_mask.sum()) >= 32:
                    return finish(color_mask, "fabric-outline color mask")
                else:
                    return finish(mask, "fabric-outline mask")

    edge = np.concatenate([
        work[:8, :, :].reshape(-1, 3),
        work[-8:, :, :].reshape(-1, 3),
        work[:, :8, :].reshape(-1, 3),
        work[:, -8:, :].reshape(-1, 3),
    ])
    bg = np.median(edge, axis=0)
    diff = np.linalg.norm(work - bg[None, None, :], axis=2)
    brightness = work.max(axis=2)
    saturation = work.max(axis=2) - work.min(axis=2)
    mask = (diff > 30.0) & (brightness > 42.0) & (saturation > 16.0)
    mask &= ~green
    if int(mask.sum()) < 32:
        mask = (diff > 22.0) & (brightness > 28.0) & (saturation > 10.0)
    if int(mask.sum()) < 1:
        return finish(np.ones((h, w), dtype=bool), "full frame fallback")
    return finish(mask, "background mask")


def summarize_capture_records(records, estimated_cell_colors=None, *, pattern_signature=""):
    """Group per-capture RGB records into the per-cell / per-angle analysis."""
    grouped = {}
    for record in records:
        key = (int(record.get("row", 0)), int(record.get("col", 0)))
        grouped.setdefault(key, []).append(record)

    estimates_by_pos = {
        (int(item.get("row", 0)), int(item.get("col", 0))): item
        for item in (estimated_cell_colors or [])
    }

    cells = []
    for (row, col), cell_records in sorted(grouped.items()):
        all_rgb = np.asarray([record["rgb"] for record in cell_records], dtype=np.float32)
        overall = all_rgb.mean(axis=0)
        estimate = estimates_by_pos.get((int(row), int(col)), {})
        estimated_rgb = [float(v) for v in estimate.get("rgb", [0.0, 0.0, 0.0])]
        estimate_delta = float(np.linalg.norm(overall - np.asarray(estimated_rgb, dtype=np.float32)))
        angle_results = []
        for angle in sorted({str(record["angle"]) for record in cell_records}):
            angle_rgb = np.asarray(
                [record["rgb"] for record in cell_records if str(record["angle"]) == angle],
                dtype=np.float32,
            )
            angle_results.append({
                "angle": angle,
                "rgb": [float(v) for v in angle_rgb.mean(axis=0)],
                "count": int(len(angle_rgb)),
            })
        cells.append({
            "row": int(row),
            "col": int(col),
            "estimated_rgb": estimated_rgb,
            "estimated_active_ratio": float(estimate.get("active_ratio", 0.0)),
            "estimate_actual_delta_rgb": estimate_delta,
            "overall_rgb": [float(v) for v in overall],
            "count": int(len(cell_records)),
            "angles": angle_results,
            "fabric_pixel_count": int(sum(int(r.get("fabric_pixel_count", 0)) for r in cell_records)),
            "analysis_total_pixels": int(sum(int(r.get("analysis_total_pixels", 0)) for r in cell_records)),
            "analysis_masks": sorted({str(r.get("analysis_mask", "unknown")) for r in cell_records}),
        })

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_count": int(len(records)),
        "pattern_signature": str(pattern_signature),
        "cells": cells,
        "background_ignored": True,
        "analysis_note": (
            "Average RGB is computed from detected fabric pixels only; "
            "scanner background and label areas are ignored."
        ),
    }


def compare_cell_colors(cells):
    """Pairwise colour distances between measured fabric cells."""
    comparisons = []
    for i, first in enumerate(cells):
        for second in cells[i + 1:]:
            a = np.asarray(first.get("overall_rgb", [0.0, 0.0, 0.0]), dtype=np.float32)
            b = np.asarray(second.get("overall_rgb", [0.0, 0.0, 0.0]), dtype=np.float32)
            comparisons.append({
                "a": f"row {int(first.get('row', 0)) + 1}, col {int(first.get('col', 0)) + 1}",
                "b": f"row {int(second.get('row', 0)) + 1}, col {int(second.get('col', 0)) + 1}",
                "a_rgb": [float(v) for v in a],
                "b_rgb": [float(v) for v in b],
                "delta_rgb": float(np.linalg.norm(a - b)),
                "delta_per_channel": [float(v) for v in np.abs(a - b)],
            })
    comparisons.sort(key=lambda item: item["delta_rgb"], reverse=True)
    return comparisons
