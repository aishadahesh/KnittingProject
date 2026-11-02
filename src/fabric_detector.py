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

import numpy as np
from PIL import Image

import paths

ENCODER_FILE = "mobile_sam_image_encoder.onnx"
DECODER_FILE = "sam_mask_decoder_multi.onnx"

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
