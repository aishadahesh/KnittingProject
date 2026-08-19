"""Fabric colour measurement shared by the simulation scanner and the real UR5.

The RGB numbers stored in the database have to mean the same thing whichever
robot produced the image, so both capture paths measure through this one
function rather than each carrying its own copy.

The measurement isolates fabric pixels from everything else in the frame. Two
strategies, tried in order:

  1. Simulated scan images draw a bright green rectangle around the target
     fabric. When that outline is present it is by far the most reliable signal,
     so an oriented rectangle is fitted to it and only the pixels inside are
     averaged.
  2. Real camera frames have no such outline. There the scanner background is
     estimated from the image border and pixels that differ from it enough to be
     fabric are kept.

A real capture therefore lands on the background mask automatically; nothing
about the caller has to change.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image


# The saved simulation captures print a label strip across the top of the frame.
# It is scanner annotation, not fabric, so it is cropped away before measuring.
LABEL_STRIP_HEIGHT = 58
LABEL_STRIP_MIN_IMAGE_HEIGHT = 70


def fabric_rgb_stats(image, debug_path=None):
    """Average the fabric pixels of one capture.

    Returns a dict with the mean ``rgb`` (0-255 floats), how many pixels went
    into it, the frame's total pixel count, and the ``method`` that selected
    them -- the method string is stored per capture so a surprising colour can
    be traced back to how it was measured.

    ``debug_path`` optionally writes an RGBA cutout of exactly the pixels used,
    which is what the analysis "used pixels" images are.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        return {
            "rgb": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "pixel_count": 0,
            "total_pixels": 0,
            "method": "empty",
        }

    # Ignore the black label strip and scanner annotations. The analysis is
    # the perceived fabric color, not UI text/background color.
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

    # Preferred path: saved scan images draw a bright green rectangle around
    # the target fabric. Use that outline to build an oriented rectangle mask
    # and average only the pixels inside the fabric area.
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
        try:
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
        except Exception:
            pass

    # Fallback: estimate the scanner background from image edges and keep
    # pixels that differ from that background enough to be fabric. This is the
    # path real camera frames take, since they carry no drawn outline.
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
    """Group per-capture RGB records into the per-cell / per-angle analysis.

    Mirrors the shape the simulation analysis produces, so a real scan's stored
    result is directly comparable: overall average RGB per fabric cell, an
    average per camera angle, and the delta against the colour the pattern was
    expected to show.
    """
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
    """Pairwise colour distances between measured fabric cells.

    Answers "how different are these two patterns", which is what the colour
    comparison in the UI reports. Sorted most-different first.
    """
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
