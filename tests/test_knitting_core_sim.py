import numpy as np
import scipy.sparse
from knitting_core import eval_centerline, build_row_spline_jacobian

def test_spline_centerline_and_jacobian():
    cp = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.1], [2.0, -0.2, 0.3]])
    D = np.array([3.0, 0.0, 0.0])
    nout = 20
    pts = eval_centerline(cp, D, nout)
    assert pts.shape == (nout, 3)
    J = build_row_spline_jacobian(cp, D, nout)
    assert J.shape == (nout, len(cp))
    # Verify linearity: J @ Px + bx == Vx
    cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
    t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
    to = np.linspace(t[0], t[-1], nout)
    pts_zero_offset = eval_centerline(cp, np.zeros(3), nout, t=t, to=to)
    for dim in range(3):
        V_dim = pts_zero_offset[:, dim]
        P_dim = cp[:, dim]
        assert np.allclose(J @ P_dim, V_dim, atol=1e-5)


def test_elastic_forces_and_hessian():
    from knitting_core import compute_elastic_forces_and_hessian
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, -0.1], [2.0, -0.05, 0.05]], dtype=float)
    edges = np.array([[0, 1], [1, 2]])
    L0 = 1.0
    L0_array = np.full(len(edges), L0)
    k_s, k_b = 100.0, 10.0
    e, g, H = compute_elastic_forces_and_hessian(V, edges, L0_array, k_s, k_b, np.array([3.0, 0.0, 0.0]), 20)
    assert e > 0
    assert g.shape == (9,)
    assert H.shape == (9, 9)
    # Check symmetry
    assert np.allclose(H.toarray(), H.toarray().T)


def test_collision_forces_and_hessian():
    from knitting_core import compute_collision_forces_and_hessian
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -0.5, 0.01], [0.5, 0.5, 0.01]], dtype=float)
    D = np.array([1.5, 1.5, 0.0])
    edges = np.array([[0, 1], [2, 3]])
    dhat = 0.05
    k_c = 10.0
    e, g, H = compute_collision_forces_and_hessian(V, D, edges, dhat, k_c)
    assert e > 0  # since the skew edges are very close (0.01 < dhat)
    assert g.shape == (12,)
    assert H.shape == (12, 12)


def test_run_simulation_step():
    from knitting_core import run_simulation_step, build_row_spline_jacobian
    ctrl_rows = [np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.0]])]
    D = np.array([3.0, 3.0, 0.0])
    config = {"knit_parameters": {"loop_res": 5, "segments": 4}}
    
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(np.linalg.norm(D))
    nout = res * int(round(bitmap_width)) + 1
    # Build cached Jacobian
    J_base = build_row_spline_jacobian(ctrl_rows[0], D, nout)
    J_cached = scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")
    
    L0_array = np.full(nout - 1, bitmap_width / (nout - 1))
    updated = run_simulation_step(ctrl_rows, D, config, J_cached, L0_array, k_s=100.0, k_b=10.0, k_c=1.0, dhat=0.05)
    assert len(updated) == 1
    assert updated[0].shape == (3, 3)


def test_check_gradients_and_hessians_fd():
    from knitting_core import check_gradients_and_hessians_fd, build_row_spline_jacobian
    ctrl_rows = [np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.0]])]
    D = np.array([3.0, 3.0, 0.0])
    config = {"knit_parameters": {"loop_res": 5, "segments": 4}}
    
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(np.linalg.norm(D))
    nout = res * int(round(bitmap_width)) + 1
    J_base = build_row_spline_jacobian(ctrl_rows[0], D, nout)
    J_cached = scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")
    
    res = check_gradients_and_hessians_fd(ctrl_rows, D, config, J_cached, k_s=10.0, k_b=1.0, k_c=1.0, dhat=0.05, eps=1e-5)
    assert "Gradient difference" in res
    assert "Hessian difference" in res

def test_intersecting_yarns():
    from knitting_core import eval_energy, evaluate_centerlines
    ctrl_rows = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    ]
    
    config = {
        "knit_parameters": {
            "segments": 8,
            "loop_res": 5
        }
    }
    D = np.array([2.0, 0.0, 0.0])
    
    V, edges, _, _ = evaluate_centerlines(ctrl_rows, D, config)
    v0_pts = V[edges[:, 0]]
    v1_pts = V[edges[:, 1]]
    L0_array = np.linalg.norm(v1_pts - v0_pts, axis=1)
    
    k_s, k_b, k_c, dhat = 1000.0, 10.0, 1.0, 0.1
    flat_P = np.concatenate(ctrl_rows)
    e_el, e_b, e_col = eval_energy(flat_P, ctrl_rows, D, config, L0_array, k_s, k_b, k_c, dhat, filter_collisions=True)
    
    assert e_col > 0.0, "Expected collision energy > 0 for crossing yarns"
    
    ctrl_rows_shifted = [ctrl_rows[0].copy(), ctrl_rows[1].copy()]
    ctrl_rows_shifted[1][:, 2] += 0.2
    flat_P_shifted = np.concatenate(ctrl_rows_shifted)
    
    _, _, e_col_new = eval_energy(flat_P_shifted, ctrl_rows_shifted, D, config, L0_array, k_s, k_b, k_c, dhat, filter_collisions=True)
    
    assert e_col_new < e_col, "Expected collision energy to decrease"
    assert np.isclose(e_col_new, 0.0), "Expected zero collision energy when separated"
