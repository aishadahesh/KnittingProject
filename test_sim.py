from OpenGL.GL.ARB import compute_shader
import numpy as np
import scipy.optimize
import ipctk

def build_subdivision_matrix(num_ctrl, num_samples):
    """
    Creates a simple linear interpolation matrix J. 
    Swap this with a Cubic B-spline matrix for your actual tool.
    V = J @ P
    """
    J = np.zeros((num_samples, num_ctrl))
    t = np.linspace(0, num_ctrl - 1, num_samples)
    for i, ti in enumerate(t):
        idx = int(ti)
        frac = ti - idx
        if idx < num_ctrl - 1:
            J[i, idx] = 1.0 - frac
            J[i, idx + 1] = frac
        else:
            J[i, -1] = 1.0
    return J

def compute_energy_and_gradient(P_flat, J, edges, mesh, dhat, L0, k_s, k_b, k_c):
    # 1. Map control points (P) to segment vertices (V)
    P = P_flat.reshape(-1, 3)
    V = J @ P
    
    num_V = V.shape[0]
    grad_V = np.zeros_like(V)
    energy = 0.0
    
    # 2. Stretch Energy: E = 0.5 * k_s * (length - L0)^2
    for i in range(len(edges)):
        v0, v1 = edges[i]
        diff = V[v1] - V[v0]
        l = np.linalg.norm(diff)
        
        energy += 0.5 * k_s * (l - L0)**2
        
        if l > 1e-8:
            g = k_s * (l - L0) * (diff / l)
            grad_V[v1] += g
            grad_V[v0] -= g

    # 3. Bend Energy: E = 0.5 * k_b * ||V_{i-1} - 2V_i + V_{i+1}||^2
    for i in range(1, num_V - 1):
        lap = V[i-1] - 2.0 * V[i] + V[i+1]
        energy += 0.5 * k_b * np.sum(lap**2)
        
        grad_V[i-1] += k_b * lap
        grad_V[i]   -= 2.0 * k_b * lap
        grad_V[i+1] += k_b * lap

# 4. IPC Barrier Energy (Collisions)
    collisions = ipctk.NormalCollisions()
    collisions.build(mesh, V, dhat)
    
    barrier = ipctk.BarrierPotential(dhat, 1.0)
    barrier_E = barrier(collisions, mesh, V)
    barrier_grad = barrier.gradient(collisions, mesh, V)
    # 5. Chain Rule: Map gradients back to the control points
    grad_P = J.T @ grad_V
    
    return energy, grad_P.flatten()

def run_simulation():
    # --- Setup ---
    num_ctrl = 10
    num_samples = 50  # Over-sample for smoother collision detection
    
    # Create initial control points (e.g., a straight line)
    P_init = np.zeros((num_ctrl, 3))
    P_init[:, 0] = np.linspace(0, 1, num_ctrl)
    
    # Add some noise to test relaxation
    np.random.seed(42)
    P_init += np.random.normal(0, 0.05, size=P_init.shape)

    # Build subdivision mapping and topology
    J = build_subdivision_matrix(num_ctrl, num_samples)
    edges = np.array([[i, i+1] for i in range(num_samples - 1)], dtype=np.int32)
    faces = np.empty((0, 3), dtype=np.int32)
    
    # Create the initial geometry for the CollisionMesh
    V_init = J @ P_init
    mesh = ipctk.CollisionMesh(V_init, edges, faces)

    # Physical parameters
    dhat = 0.02          # Barrier activation distance (yarn thickness)
    L0 = 1.0 / (num_samples - 1)  # Target segment rest length
    k_s = 1000.0         # Stretch stiffness
    k_b = 10.0           # Bending stiffness
    k_c = 1.0            # Collision barrier stiffness

    # --- Optimization ---
    print("Starting optimization...")
    
    # SciPy's L-BFGS-B handles gradient-based minimization smoothly
    result = scipy.optimize.minimize(
        fun=compute_energy_and_gradient,
        x0=P_init.flatten(),
        args=(J, edges, mesh, dhat, L0, k_s, k_b, k_c),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 100}
    )
    
    # Print the solver's exit message instead of relying on 'disp'
    print(f"Solver message: {result.message}")
    P_opt = result.x.reshape(-1, 3)
    print("\nOptimization Finished!")
    print(f"Final Energy: {result.fun}")

if __name__ == "__main__":
    run_simulation()