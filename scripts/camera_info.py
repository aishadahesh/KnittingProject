"""Report every camera this PC can open: its name, USB id, the largest frame it gives, and whether
its focus can be driven from software - what choosing a camera, and sizing a holder, needs to know.

Run from the project root:  uv run python scripts/camera_info.py
"""

import subprocess
import sys

import cv2

TRY_SIZES = ((4656, 3496), (3840, 2160), (2592, 1944), (2560, 1440), (1920, 1080), (1280, 720))


def usb_ids():
    if not sys.platform.startswith("win"):
        return []
    command = ("Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in 'Camera','Image' } | "
               "ForEach-Object { $_.FriendlyName + ' | ' + $_.InstanceId }")
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=20).stdout.split("\n")
    except Exception as exc:
        return [f"(could not list USB ids: {exc})"]


def names():
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def frame_size(index, backend, width, height, fourcc_first):
    """The size of a frame actually read after asking for `width` x `height` - what the driver reports can lie."""
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        return None
    try:
        requests = [(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*"MJPG"))),
                    (cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height)]
        for prop, value in (requests if fourcc_first else requests[1:] + requests[:1]):
            capture.set(prop, value)
        ok, frame = capture.read()
        return (frame.shape[1], frame.shape[0]) if ok and frame is not None else None
    finally:
        capture.release()


def probe(index, name):
    backends = (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)) if sys.platform.startswith("win") else (("ANY", cv2.CAP_ANY),)
    capture = cv2.VideoCapture(index, backends[0][1])
    if not capture.isOpened():
        return False
    autofocus = capture.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
    focus = capture.set(cv2.CAP_PROP_FOCUS, 30.0)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
    capture.release()
    print(f"[{index}] {name}")
    largest = (0, 0)
    for label, backend in backends:
        sizes = sorted({size for width, height in TRY_SIZES for first in (True, False)
                        if (size := frame_size(index, backend, width, height, first))}, key=lambda s: -s[0] * s[1])
        print(f"     {label}: frames read at " + (", ".join(f"{w}x{h}" for w, h in sizes) or "none"))
        if sizes and sizes[0][0] * sizes[0][1] > largest[0] * largest[1]:
            largest = sizes[0]
    print(f"     largest frame: {largest[0]} x {largest[1]}  ({largest[0] * largest[1] / 1e6:.1f} MP)")
    print(f"     software focus: {'YES - Find best focus will work' if focus else 'no - fixed focus, or the driver hides it'}"
          f"   (autofocus switch {'yes' if autofocus else 'no'})")
    capture.release()
    return True


def main():
    print("USB cameras (name | id - the VID_xxxx&PID_xxxx part identifies the model):")
    for line in usb_ids():
        if line.strip():
            print("   ", line.strip())
    print()
    found = names()
    misses = 0
    for index in range(10):
        if probe(index, found[index] if index < len(found) else f"Camera {index}"):
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break


if __name__ == "__main__":
    main()
