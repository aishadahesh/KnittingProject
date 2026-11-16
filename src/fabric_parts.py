"""The numbered parts of the fabric, found in its top-down photo.

The fabric is two rows of knitted panels sharing one set of column edges. Seen from the robot the far row
(larger y) holds the even numbers and the near row the odd ones, both counted from the +x end: column k is
part 2k in the far row and 2k + 1 in the near row. Label and hem bands at the ends are not panels.

Images are top-down and in the robot base frame: column = (x - x0) / res, row = (y_top - y) / res.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

COLUMNS = 20
EDGE_STEP_M = 0.003
EDGE_MIN_SPACING_M = 0.015
EDGE_PROMINENCE = 4.0
MERGE_M = 0.004
REGULAR_WIDTH = 0.25
SEAM_SEARCH = 0.3


def _colour_change(lab_profile: np.ndarray, step: int) -> np.ndarray:
    """|Lab difference| between samples `step` apart, centred on the sample between them."""
    change = np.zeros(len(lab_profile))
    change[step // 2:step // 2 + len(lab_profile) - step] = np.linalg.norm(lab_profile[step:] - lab_profile[:-step], axis=1)
    return change


def find_seam(lab: np.ndarray, res_m: float, y_top: float, x_cols: slice, y_range: tuple[float, float]) -> float:
    """The y (m) where the two rows meet: the strongest row-wise colour change in the middle of the fabric."""
    rows_y = y_top - np.arange(lab.shape[0]) * res_m
    change = _colour_change(lab[:, x_cols].mean(1), max(2, int(EDGE_STEP_M / res_m)))
    low, high = y_range
    middle = (rows_y > low + SEAM_SEARCH * (high - low)) & (rows_y < high - SEAM_SEARCH * (high - low))
    return float(rows_y[middle][np.argmax(change[middle])])


def column_edges(lab_band: np.ndarray, res_m: float, x0: float, x_range: tuple[float, float]) -> np.ndarray:
    """x (m) of the colour edges across one row band, between the fabric's ends."""
    change = _colour_change(lab_band.mean(0), max(2, int(EDGE_STEP_M / res_m)))
    peaks, _ = find_peaks(change, distance=int(EDGE_MIN_SPACING_M / res_m), prominence=EDGE_PROMINENCE)
    xs = x0 + peaks * res_m
    return xs[(xs > x_range[0]) & (xs < x_range[1])]


def panel_columns(edges: list[np.ndarray]) -> list[tuple[float, float]]:
    """The run of COLUMNS regular-width columns from both rows' edges, ordered from +x; raises otherwise."""
    merged = []
    for x in np.sort(np.concatenate(edges)):
        if merged and x - merged[-1][-1] < MERGE_M:
            merged[-1].append(x)
        else:
            merged.append([x])
    xs = np.array([np.mean(group) for group in merged])
    widths = np.diff(xs)
    regular = np.abs(widths / np.median(widths) - 1.0) < REGULAR_WIDTH
    runs, start = [], None
    for i, ok in enumerate([*regular, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    first, last = max(runs, key=lambda run: run[1] - run[0])
    if last - first != COLUMNS:
        raise ValueError(f"found {last - first} regular columns, expected {COLUMNS}: edges at {np.round(xs, 3).tolist()}")
    return [(float(xs[i]), float(xs[i + 1])) for i in range(last - 1, first - 1, -1)]


def number_parts(columns: list[tuple[float, float]], seam_y: float, y_range: tuple[float, float]) -> list[dict]:
    """Part number, row and rectangle (x0, x1, y0, y1 in m) for every panel; columns ordered from +x."""
    rows = {"far": (seam_y, y_range[1], 0), "near": (y_range[0], seam_y, 1)}
    return sorted(({"number": 2 * k + parity, "row": name, "column": k, "x": [x0, x1], "y": [y0, y1]}
                   for k, (x0, x1) in enumerate(columns) for name, (y0, y1, parity) in rows.items()),
                  key=lambda part: part["number"])


def find_parts(lab: np.ndarray, res_m: float, x0: float, y_top: float, fabric: dict) -> list[dict]:
    """Every numbered part in a top-down Lab image of the fabric (`fabric`: workcell.json's x, y ranges)."""
    def row(y):
        return int(round((y_top - y) / res_m))

    x_range, y_range = tuple(fabric["x"]), tuple(fabric["y"])
    seam = find_seam(lab, res_m, y_top, slice(*(int(round((x - x0) / res_m)) for x in x_range)), y_range)
    inset = 0.05 * (y_range[1] - y_range[0])
    bands = [(seam + inset, y_range[1] - inset), (y_range[0] + inset, seam - inset)]
    edges = [column_edges(lab[row(high):row(low)], res_m, x0, x_range) for low, high in bands]
    return number_parts(panel_columns(edges), seam, y_range)
