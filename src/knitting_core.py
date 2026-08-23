import ctypes
import glob
import os
import sys
import numpy as np
from scipy.interpolate import CubicSpline

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _preload_linux_nvidia_libs():
    if not sys.platform.startswith("linux"):
        return
    for p in sys.path:
        if not p.endswith("site-packages"):
            continue
        root = os.path.join(p, "nvidia")
        if not os.path.isdir(root):
            continue
        for so in sorted(glob.glob(os.path.join(root, "**/*.so*"), recursive=True)):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
        return


_preload_linux_nvidia_libs()

def compute_bitmap_scale_factors(bitmap):
    """Return active spans; zero bitmap cells remain inactive."""
    bitmap_array = np.asarray(bitmap, dtype=np.float32) > 0.5
    n_rows, n_cols = bitmap_array.shape
    scale_factors = np.zeros((n_rows, n_cols), dtype=np.float32)

    for col_idx in range(n_cols):
        active_rows = np.flatnonzero(bitmap_array[:, col_idx])
        for active_index, row_idx in enumerate(active_rows):
            next_row = active_rows[active_index + 1] if active_index + 1 < len(active_rows) else n_rows
            scale_factors[row_idx, col_idx] = float(next_row - row_idx)

    return scale_factors


def unanchored_columns(bitmap):
    """Columns whose bottom cell is cleared, leaving a stitch with no support.

    A cleared cell is normally covered: compute_bitmap_scale_factors gives the
    nearest active cell below it a span reaching up over the gap, and the
    topmost active cell stretches to the top of the fabric. Row 0 is the one
    place that cannot work, because there is no cell below it -- so a cleared
    bottom cell is never stood in for, and the fabric there hangs from nothing.

    Returns the offending column indices; empty means the pattern is legal.
    This also covers an entirely cleared column, which necessarily has its
    bottom cell cleared too.
    """
    bitmap_array = np.asarray(bitmap, dtype=np.float32)
    if bitmap_array.ndim != 2 or bitmap_array.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(bitmap_array[0, :] <= 0.5)


def bitmap_is_anchored(bitmap):
    """True when every column has a stitch in row 0 for the fabric to hang from."""
    return unanchored_columns(bitmap).size == 0


def anchor_bitmap(bitmap):
    """Return a legal copy of `bitmap`, plus the columns that had to be filled.

    Filling row 0 is the only repair available. Clearing the rest of the column
    instead would leave it with no stitches at all, which is worse, and the
    column count is fixed by the pattern size.

    Never mutates the input, and always returns a new array so callers cannot
    end up aliasing each other's buffers.
    """
    bitmap_array = np.asarray(bitmap, dtype=np.float32)
    if bitmap_array.ndim != 2 or bitmap_array.size == 0:
        return bitmap_array.astype(np.float32, copy=True), np.empty(0, dtype=np.int64)
    columns = unanchored_columns(bitmap_array)
    repaired = bitmap_array.astype(np.float32, copy=True)
    if columns.size:
        repaired[0, columns] = 1.0
    return repaired, columns


def _height_grid_from_params(params, bitmap, lh_idx):
    params = np.asarray(params, dtype=np.float32)
    bitmap = np.asarray(bitmap, dtype=np.float32)
    row_heights = np.asarray(params[np.array(lh_idx, dtype=np.int32)], dtype=np.float32)
    if row_heights.size == 0:
        return np.zeros_like(bitmap, dtype=np.float32)
    grid = np.zeros_like(bitmap, dtype=np.float32)
    for row_idx in range(bitmap.shape[0]):
        grid[row_idx, :] = float(row_heights[min(row_idx, row_heights.size - 1)])
    return grid


def build_parametric_control_rows(params, bitmap, pidx, lh_idx, spl=5, loop_heights=None):
    """Build stitch control rows with per-bitmap-cell loop heights."""
    p = np.asarray(params, dtype=np.float32)
    bitmap_array = np.asarray(bitmap, dtype=np.float32)
    stitch_bulge = float(p[pidx["stitch_bulge"]])
    stitch_z = float(p[pidx["stitch_z"]])
    dy = float(p[pidx["dy"]])
    # How wide one stitch is. This used to be hard-coded at one unit per column
    # while loop height stayed free, so raising the height gave tall narrow
    # loops that overlapped their neighbours instead of a proper knit -- which
    # is why a regenerated model looked nothing like the tuned one saved in
    # initial_params.json. That saved geometry is this same formula with x
    # scaled by 3.706, so the width is now a parameter and the automatic build
    # can reproduce it. Older files simply lack the key and load the default.
    stitch_width = float(p[pidx["stitch_width"]]) if "stitch_width" in pidx else 1.0
    if loop_heights is None:
        height_grid = _height_grid_from_params(p, bitmap_array, lh_idx)
    else:
        height_grid = np.asarray(loop_heights, dtype=np.float32)
        if height_grid.shape != bitmap_array.shape:
            fallback = _height_grid_from_params(p, bitmap_array, lh_idx)
            fixed = fallback.copy()
            h_rows = min(fixed.shape[0], height_grid.shape[0])
            h_cols = min(fixed.shape[1], height_grid.shape[1])
            fixed[:h_rows, :h_cols] = height_grid[:h_rows, :h_cols]
            height_grid = fixed
    # What each cell's loop would measure if it were knitted. Clearing a cell
    # also zeroes its stored height, so height_grid cannot say how tall the
    # stitch standing in for it has to reach; the row's own loop_height
    # parameter is that answer. Only ever consulted for cells with no height of
    # their own, so per-cell edits are respected everywhere else.
    natural_heights = height_grid.copy()
    row_defaults = _height_grid_from_params(p, bitmap_array, lh_idx)
    blank = natural_heights <= 0.0
    natural_heights[blank] = row_defaults[blank]

    scale_factors = compute_bitmap_scale_factors(bitmap_array)
    n_rows, n_cols = scale_factors.shape
    base_t_values = np.linspace(0.0, 2.0 * np.pi, int(spl), endpoint=False, dtype=np.float32)
    # How much of a loop's height its highest control point actually reaches.
    # The samples are spread evenly around the loop and never land exactly on
    # its crown, so at the usual five per loop the top control point sits at
    # 0.9045 of the height rather than 1.0. A replacement stitch has to clear
    # the row pitches it crosses in those same units, or it stops fractionally
    # short of the loop it is standing in for.
    peak_factor = float(np.max((1.0 - np.cos(base_t_values)) / 2.0)) if int(spl) > 0 else 1.0
    peak_factor = max(peak_factor, 1e-6)
    rows = []

    for row_idx in range(n_rows):
        row_points = []
        col_indices = range(n_cols) if row_idx % 2 == 0 else range(n_cols - 1, -1, -1)
        t_values = base_t_values if row_idx % 2 == 0 else base_t_values[::-1]

        for col_idx in col_indices:
            # A stitch grows upward to take the place of the switched-off cells
            # above it in its own column, finishing where the topmost one it
            # replaces would have finished. compute_bitmap_scale_factors already
            # reports that reach as a span -- 1 normally, one more per
            # switched-off cell above -- but it was being read as a plain
            # yes/no flag, so a cleared cell just left a hole.
            #
            # The replacement stitch has to end level with the loop it stands in
            # for, not simply be a multiple of its own height: it rises over the
            # (span - 1) row pitches it crosses and then forms the loop the top
            # covered cell would have had. A span of 1 leaves loop_height
            # exactly as it was, so a fully active bitmap is untouched.
            span = int(scale_factors[row_idx, col_idx])
            has_loop = 1.0 if span > 0 else 0.0
            if has_loop:
                top_row = min(row_idx + span - 1, n_rows - 1)
                loop_height = float(natural_heights[top_row, col_idx])
                loop_height += (span - 1) * dy / peak_factor
            else:
                loop_height = 0.0
            for t in t_values:
                # The bulge scales with the stitch too, so its shape stays the
                # same proportion of a stitch at any width.
                x = (
                    col_idx
                    + (stitch_bulge * np.sin(2.0 * t) if has_loop else 0.0)
                    + t / (2.0 * np.pi)
                ) * stitch_width
                y = row_idx * dy - loop_height * (np.cos(t) - 1.0) / 2.0
                z = has_loop * stitch_z * (np.cos(2.0 * t) - 1.0) / 2.0
                row_points.append([x, y, z])

        # Carry the row all the way to the right-hand edge of the fabric.
        #
        # Sampling a stitch stops one step short of its far side, so a row's
        # outermost point sits mid-stitch rather than on the edge. Rows are
        # knitted alternately, and only the left-to-right ones were given this
        # closing point -- so right-to-left rows began at x=6.050 instead of
        # 7.412 and ended up shifted a third of a stitch out of line with their
        # neighbours, their ends landing away from the fabric edge. Whichever
        # way the row runs, the edge belongs at the end that reaches for it:
        # appended when the row finishes on the right, prepended when it starts
        # there. Both then span exactly one period, 0 .. n_cols * stitch_width.
        edge_point = [float(n_cols) * stitch_width, row_idx * dy, 0.0]
        if row_idx % 2 == 0:
            row_points.append(edge_point)
        else:
            row_points.insert(0, edge_point)
        rows.append(np.array(row_points, dtype=float))

    return rows


def row_base_pitch(ctrl_rows):
    """The row lattice pitch: the Y step from one row's base to the next's.

    Rows are built above as ``y = row_idx * dy - loop_height * (cos t - 1) / 2``.
    The second term is zero at ``t = 0`` and non-negative everywhere else, so a
    row's minimum Y is exactly ``row_idx * dy`` no matter how tall its loops
    are. Minimum Y is therefore the only reading of the pitch that loop height
    cannot move -- row centres and the mesh bounding box both shift with it,
    which is how a period built on either ends up wrong by a third or more.

    Measured from the rows rather than read back from ``dy`` because control
    rows can be scaled or dragged after they are built, at which point ``dy``
    no longer describes them.

    Median of the consecutive steps rather than a fitted slope, for the same
    reason the X period next door takes a median: a single hand-dragged point
    must not move the answer.

    Returns None when there is nothing to measure -- fewer than two rows, or a
    degenerate result.
    """
    bases = [
        float(np.asarray(row, dtype=np.float32)[:, 1].min())
        for row in ctrl_rows
        if len(row)
    ]
    if len(bases) < 2:
        return None
    pitch = float(np.median(np.abs(np.diff(bases))))
    return pitch if pitch > 1e-6 else None


def compute_knitting_faces(seg, vl):
    if not vl:
        return []
    n = vl[0][1]
    i, j = np.meshgrid(np.arange(n - 1), np.arange(seg), indexing="ij")
    faces = np.stack((
        i * seg + j,
        i * seg + (j + 1) % seg,
        (i + 1) * seg + (j + 1) % seg,
        (i + 1) * seg + j
    ), axis=-1).reshape(-1, 4)
    return [faces] * len(vl)


def build_surface_fiber_meshes(
    base_vl,
    segments,
    enabled,
    count,
    radius,
    radius_scale,
    lift,
    surface_arc,
    randomness,
    twist,
):
    if not enabled:
        return list(base_vl), [{"row": row_idx} for row_idx in range(len(base_vl))]

    fiber_radius = max(radius * radius_scale, 1e-5)
    lift_val = max(lift, 0.0)
    surface_arc_val = float(np.clip(surface_arc, 0.05, 1.0))
    randomness_val = float(np.clip(randomness, 0.0, 1.0))

    out_vl = []
    meta = []

    for row_idx, (verts, n_points) in enumerate(base_vl):
        verts = np.asarray(verts, dtype=np.float32)
        n_points = int(n_points)
        if n_points < 2 or len(verts) != n_points * segments:
            continue

        rings = verts.reshape(n_points, segments, 3)
        centers = rings.mean(axis=1)
        top_idx = int(np.argmax((rings - centers[:, None, :])[:, :, 2].mean(axis=0)))
        offsets = (
            np.zeros(1, dtype=np.float32)
            if count == 1
            else np.linspace(-0.5, 0.5, count, dtype=np.float32) * surface_arc_val * float(segments)
        )

        for fiber_idx, offset in enumerate(offsets):
            rng = np.random.default_rng(row_idx * 1009 + fiber_idx * 9173)
            phase_jitter = rng.normal(0.0, 0.35 * randomness_val)
            lift_jitter = rng.normal(0.0, 0.20 * randomness_val)
            radius_jitter = float(np.clip(1.0 + rng.normal(0.0, 0.18 * randomness_val), 0.55, 1.45))
            local_radius = max(fiber_radius * radius_jitter, 1e-5)

            sample_idx = np.mod(
                top_idx + offset + phase_jitter + twist * np.linspace(0.0, 1.0, n_points, dtype=np.float32) * segments,
                float(segments),
            )
            lo_float = np.floor(sample_idx)
            lo = lo_float.astype(np.int32) % segments
            hi = (lo + 1) % segments
            frac = (sample_idx - lo_float).astype(np.float32)
            surface = rings[np.arange(n_points), lo] * (1.0 - frac[:, None]) + rings[np.arange(n_points), hi] * frac[:, None]
            radial = surface - centers
            surface_radius = np.linalg.norm(radial, axis=1, keepdims=True)
            radial /= surface_radius + 1e-8
            center_radius = np.maximum(surface_radius - local_radius + local_radius * (lift_val + lift_jitter), local_radius)
            line = centers + radial * center_radius

            tangent = np.gradient(line, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
            side = np.cross(tangent, radial)
            bad = np.linalg.norm(side, axis=1) < 1e-6
            if np.any(bad):
                side[bad] = np.cross(tangent[bad], [1.0, 0.0, 0.0])
            side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-8
            normal = np.cross(side, tangent)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8

            angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
            offsets_ring = (
                normal[:, None, :] * np.cos(angles)[None, :, None]
                + side[:, None, :] * np.sin(angles)[None, :, None]
            ) * local_radius
            out_vl.append(((line[:, None, :] + offsets_ring).reshape(-1, 3).astype(np.float32), n_points))
            meta.append({'row': row_idx})

    return out_vl, meta



def eval_centerline(cp, D, nout, t=None, to=None):
    """Samples one periodic row centreline.

    The row is closed by appending cp[0] + D, then "detrended" by subtracting the
    linear ramp along D before fitting, so a periodic cubic spline is valid; the
    ramp is added back afterwards. That is what makes tiled copies join smoothly
    across the period boundary instead of kinking.

    `t`/`to` may be supplied to reuse a parameterisation across calls -- the
    Jacobian builder relies on that to hold the knot vector fixed while varying
    one control point at a time.
    """
    cp = np.asarray(cp, dtype=float)
    if len(cp) <= 1:
        return np.repeat(cp, nout, axis=0)
    cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
    if t is None or to is None:
        t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
        to = np.linspace(t[0], t[-1], nout)
    cp_detrended = cp_aug - D[None, :] * (t / t[-1])[:, None]
    if len(cp) == 2:
        pts_detrended = np.column_stack([np.interp(to, t, cp_detrended[:, i]) for i in range(3)])
    else:
        pts_detrended = np.column_stack([CubicSpline(t, cp_detrended[:, i], bc_type="periodic")(to) for i in range(3)])
    return pts_detrended + D[None, :] * (to / t[-1])[:, None]


def _centerline_sample_count(period_offset_x, config):
    """Samples per row. Shared so the mesh, the simulation centrelines and the
    Jacobian all agree on nout -- if they disagree the Jacobian silently stops
    matching the geometry it is supposed to differentiate."""
    D = np.asarray(period_offset_x, dtype=float)
    if D.ndim == 0:
        bitmap_width = float(D)
    else:
        bitmap_width = float(np.linalg.norm(D))
    res = config["knit_parameters"]["loop_res"]
    return max(3, res * int(round(bitmap_width)) + 1)


def evaluate_centerlines(ctrl_rows, period_offset_x, config):
    """Centreline vertices + per-row edge topology, as the yarn simulation wants
    them: one flat V array of shape (rows * nout, 3) and the index pairs joining
    consecutive samples within each row."""
    D = np.asarray(period_offset_x, dtype=float)
    nout = _centerline_sample_count(D, config)

    V_list = [eval_centerline(np.asarray(r, dtype=float), D, nout) for r in ctrl_rows]
    if not V_list:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), D, nout

    V = np.vstack(V_list)
    edges_list = []
    row_offset = 0
    for _ in range(len(ctrl_rows)):
        row_edges = np.column_stack((np.arange(nout - 1), np.arange(1, nout))).astype(np.int32) + row_offset
        edges_list.append(row_edges)
        row_offset += nout
    edges = np.vstack(edges_list) if edges_list else np.empty((0, 2), dtype=np.int32)
    return V, edges, D, nout


def build_row_spline_jacobian(cp, D, nout):
    """d(sampled points) / d(control points) for one row.

    The map from control points to samples is linear once the knot vector is
    fixed, so each column is recovered by evaluating a unit impulse control
    point through the same parameterisation. Passing t/to keeps that
    parameterisation identical across columns, which is what makes the columns
    combine into a valid Jacobian.
    """
    cp = np.asarray(cp, dtype=float)
    num_ctrl = len(cp)
    if num_ctrl <= 1:
        return np.ones((nout, 1))
    cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
    t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
    to = np.linspace(t[0], t[-1], nout)
    cols = []
    for k in range(num_ctrl):
        cp_dummy = np.zeros((num_ctrl, 3))
        cp_dummy[k, 0] = 1.0
        pts = eval_centerline(cp_dummy, np.zeros(3), nout, t=t, to=to)
        cols.append(pts[:, 0])
    return np.column_stack(cols)


def build_spline_mesh(
    ctrl_rows,
    params,
    config,
    pidx,
    period_offset_x,
    radius_ctrl_rows=None,
):
    p = np.asarray(params)
    rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
    seg = config["knit_parameters"]["segments"]

    if isinstance(period_offset_x, (int, float, np.integer, np.floating)):
        D = np.array([float(period_offset_x), 0.0, 0.0], dtype=float)
    else:
        D = np.asarray(period_offset_x, dtype=float)

    nout = _centerline_sample_count(period_offset_x, config)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out = []
    for row_idx, r in enumerate(ctrl_rows):
        cp = np.asarray(r, dtype=float)
        if len(cp) == 0:
            continue
        # eval_centerline closes a row periodically by appending cp[0] + D,
        # which assumes the row runs the same way D points. Rows alternate
        # direction, so on a right-to-left row cp[0] is already at the far end
        # and adding D threw the closing point a whole period past it: the row
        # was drawn out to x=13.46 on a model only 7.41 wide, which is the long
        # straight tail those rows trailed off to one side. Point the period the
        # way this row actually travels.
        row_period = D
        if len(cp) > 1 and float(np.dot(cp[-1] - cp[0], D)) < 0.0:
            row_period = -D
        pts = eval_centerline(cp, row_period, nout)
        if len(cp) <= 1:
            ctrl_sample_idx = np.zeros(nout, dtype=float)
        else:
            ctrl_sample_idx = np.linspace(0.0, len(cp), nout, dtype=float)

        if radius_ctrl_rows is not None and row_idx < len(radius_ctrl_rows):
            radius_cp = np.asarray(radius_ctrl_rows[row_idx], dtype=float)
            if radius_cp.shape[0] == len(cp):
                if len(cp) <= 1:
                    radius_line = np.full(nout, float(radius_cp[0]) if len(radius_cp) else float(rad), dtype=float)
                else:
                    radius_cp_aug = np.append(radius_cp, radius_cp[0])
                    radius_line = np.interp(ctrl_sample_idx, np.arange(len(radius_cp_aug), dtype=float), radius_cp_aug)
            else:
                radius_line = np.full(nout, float(rad), dtype=float)
        else:
            radius_line = np.full(nout, float(rad), dtype=float)
        radius_line = np.maximum(radius_line, 1e-6)


        if len(cp) <= 1:
            T = np.gradient(pts, axis=0)
        else:
            T = np.zeros_like(pts)
            T[1:-1] = (pts[2:] - pts[:-2]) / 2.0
            T[0] = (pts[1] - (pts[-2] - D)) / 2.0
            T[-1] = ((pts[1] + D) - pts[-2]) / 2.0
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
        U = np.cross(T, [0, 0, 1])
        b = np.linalg.norm(U, axis=1) < 1e-6
        U[b] = np.cross(T[b], [1, 0, 0])
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        V = np.cross(T, U)
        rline = radius_line[:, None, None]
        out.append(((pts[:, None, :] + U[:, None, :] * ca * rline * rat + V[:, None, :] * sa * rline).reshape(-1, 3), nout))
    return out
