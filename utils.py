"""
Utility functions for quantum thermodynamics simulations.
Includes Lindblad dissipators, heat curvature, Liouvillian gap, and cycle heat calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, solve as linsolve, svd, eig as eigenvalues
from functools import wraps
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='sans-serif', serif=['Helvetica'], size=15)

# Pauli Matrices
sigma0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_up = (sigma1 + 1j * sigma2) / 2


def lindblad_dissipator(L, rho):
    """Lindblad Dissipator"""
    L_dag = np.conjugate(L).T
    term1 = L @ rho @ L_dag
    term2 = 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
    return term1 - term2

def old_heat_curvature(ham_func, h_target, diss_func, ranges, points=20):
    """Numerical solver for heat curvature"""
    # Setup Dimensions
    H_test = ham_func(ranges[0][1], ranges[1][1])
    d = H_test.shape[0]
    d2 = d**2

    # Vectorization
    basis = np.eye(d2, d2).reshape(d2, d, d)
    vec = lambda m: m.flatten()
    unvec = lambda v: v.reshape(d, d)

    def solve_l_response(L_mat, target_vec):
        aug_mat = L_mat.copy()
        aug_mat[0, :] = vec(np.identity(d))
        aug_rhs = target_vec.copy()
        aug_rhs[0] = 0
        return linsolve(aug_mat, aug_rhs)

    def get_curvature_point(p1_val, p2_val):
        """Point-wise Curvature Kernel"""
        p_vals = np.array([p1_val, p2_val])
        step = np.abs(p_vals) * 1e-4
        step[step == 0] = 1e-4

        # Steady State at Center
        H_tot = ham_func(*p_vals)
        L_mat = np.zeros((d2, d2), dtype=complex)
        for i, b in enumerate(basis):
            L_mat[:, i] = vec(-1j * (H_tot @ b - b @ H_tot) + diss_func(b)(*p_vals))
        
        ns = null_space(L_mat)
        rho_vec = ns[:, 0] if ns.shape[1] > 0 else svd(L_mat)[2][-1, :]
        rho = unvec(rho_vec / np.trace(unvec(rho_vec)))
        rho_vec = vec(rho)

        # Compute Derivatives of H and rho
        drho = np.zeros((2, d, d), dtype=complex)
        dH = np.zeros((2, d, d), dtype=complex)

        for i in range(2):
            p_step = p_vals.copy()
            p_step[i] += step[i]
            H_step = ham_func(*p_step)
            
            L_mat_step = np.zeros((d2, d2), dtype=complex)
            for j, b in enumerate(basis):
                L_mat_step[:, j] = vec(-1j * (H_step @ b - b @ H_step) + diss_func(b)(*p_step))

            dl_rho = -(L_mat_step @ rho_vec - L_mat @ rho_vec) / step[i]
            drho[i, :, :] = unvec(solve_l_response(L_mat, dl_rho))
            dH[i, :, :] = (H_step - H_tot) / step[i]

        # Curvature Formula
        return np.real(np.trace(dH[0] @ drho[1]) - np.trace(dH[1] @ drho[0]))

    # Map Generation
    p1_grid = np.linspace(ranges[0][1], ranges[0][2], points)
    p2_grid = np.linspace(ranges[1][1], ranges[1][2], points)
    
    curvature_data = np.zeros((points, points))
    for i, p1 in enumerate(p1_grid):
        for j, p2 in enumerate(p2_grid):
            curvature_data[j, i] = get_curvature_point(p1, p2)
            
    max_val = np.max(np.abs(curvature_data))
    if max_val > 0:
        curvature_data /= max_val

    return p1_grid, p2_grid, curvature_data, max_val

def heat_curvature(H_tot, H_ref, diss, ranges, points=20):
    """
    Numerical computation of heat curvature F^Q_{ij} = d_i A^Q_j - d_j A^Q_i
    Using the deterministic trace-replacement trick for numerical stability.
    """

    def liouvillian(H, rho, diss_op):
        commutator = -1j * (H @ rho - rho @ H)
        return commutator + diss_op(rho)

    def get_steady_state(p_vals):
        """Find steady state by replacing the first row of L with Tr(rho) = 1"""
        H = H_tot(*p_vals)
        d = H.shape[0]
        d2 = d * d
        
        # Build Liouvillian superoperator matrix
        L_mat = np.zeros((d2, d2), dtype=complex)
        for col in range(d2):
            e = np.zeros((d, d), dtype=complex)
            e[col // d, col % d] = 1.0
            L_e = liouvillian(H, e, lambda r: diss(r)(*p_vals))
            L_mat[:, col] = L_e.flatten()
        
        # Trace-replacement: replace row 0 with the vectorized identity matrix
        M = L_mat.copy()
        M[0, :] = 0.0
        for k in range(d):
            M[0, k * d + k] = 1.0  # Summing diagonal elements for Tr(rho)
        
        # Right hand side: trace must equal 1, all other equations equal 0
        b = np.zeros(d2, dtype=complex)
        b[0] = 1.0
        
        # Deterministic solver eliminates SVD gauge jumps
        rho_vec = np.linalg.solve(M, b)
        return rho_vec.reshape(d, d)

    # Determine stable, uniform step sizes based on the parameter ranges
    # Optimal step size for nested second derivatives is h ~ range * epsilon^(1/4)
    h1 = 1e-3 * (ranges[0][2] - ranges[0][1])
    h2 = 1e-3 * (ranges[1][2] - ranges[1][1])
    global_step = np.array([h1, h2])

    def compute_A_Q(p_vals):
        """Compute heat vector A^Q using uniform global step sizes"""
        H_ref_val = H_ref(*p_vals)
        d = H_ref_val.shape[0]
        drho = np.zeros((2, d, d), dtype=complex)
        
        for i in range(2):
            p_plus = p_vals.copy()
            p_plus[i] += global_step[i]
            p_minus = p_vals.copy()
            p_minus[i] -= global_step[i]
            
            rho_ss_plus = get_steady_state(p_plus)
            rho_ss_minus = get_steady_state(p_minus)
            drho[i] = (rho_ss_plus - rho_ss_minus) / (2 * global_step[i])
        
        A_Q = np.array([np.real(np.trace(H_ref_val @ drho[i])) for i in range(2)])
        return A_Q

    def get_curvature_point(p1_val, p2_val):
        p_vals = np.array([p1_val, p2_val])
        dA_Q = np.zeros((2, 2))
        
        for i in range(2):
            p_plus = p_vals.copy()
            p_plus[i] += global_step[i]
            p_minus = p_vals.copy()
            p_minus[i] -= global_step[i]
            
            A_Q_plus = compute_A_Q(p_plus)
            A_Q_minus = compute_A_Q(p_minus)
            dA_Q[i] = (A_Q_plus - A_Q_minus) / (2 * global_step[i])
        
        return dA_Q[0, 1] - dA_Q[1, 0]

    # Generate grid
    p1_grid = np.linspace(ranges[0][1], ranges[0][2], points)
    p2_grid = np.linspace(ranges[1][1], ranges[1][2], points)
    
    curvature_data = np.zeros((points, points))
    for i, p1 in enumerate(p1_grid):
        for j, p2 in enumerate(p2_grid):
            curvature_data[j, i] = get_curvature_point(p1, p2)
    
    max_val = np.max(np.abs(curvature_data))
    if max_val > 0:
        curvature_data /= max_val

    return p1_grid, p2_grid, curvature_data, max_val


def liouvillian_gap(ham_func, diss_func, params, ranges, points=25):
    """Computes and plots the Liouvillian spectral gap for a 2D parameter space."""
    
    H_test = ham_func(ranges[0][1], ranges[1][1])
    d = H_test.shape[0]
    d2 = d * d

    # Grid Setup
    p1_range = np.linspace(ranges[0][1], ranges[0][2], points)
    p2_range = np.linspace(ranges[1][1], ranges[1][2], points)
    gap_data = np.zeros((points, points))

    print(f"Computing Liouvillian Gap for {d}x{d} system...")

    # Iterate over parameter grid and compute gap
    for i, p1 in enumerate(p1_range):
        for j, p2 in enumerate(p2_range):
            H = ham_func(p1, p2)
            
            # Construct the Liouvillian superoperator matrix
            # L_mat @ vec(ρ) = vec(L[ρ])
            L_mat = np.zeros((d2, d2), dtype=complex)
            
            for col in range(d2):
                # Create basis matrix e_ij (only one element = 1)
                e = np.zeros((d, d), dtype=complex)
                e[col // d, col % d] = 1.0
                
                # Apply Liouvillian: L[e] = -i[H, e] + diss(e)
                L_e = -1j * (H @ e - e @ H) + diss_func(e)(p1, p2)
                
                # Vectorize and store as column
                L_mat[:, col] = L_e.flatten()
            
            # Compute eigenvalues
            evals, _ = eigenvalues(L_mat)
            
            # Gap is absolute value of 2nd largest real part
            sorted_reals = np.sort(np.real(evals))[::-1]
            gap = np.abs(sorted_reals[1])
            gap_data[j, i] = gap

    print(f"Liouvillian gap between {min(gap_data.flatten()):.2e} and {max(gap_data.flatten()):.2e}.")

    # Visualization
    plt.figure(figsize=(8, 6))    
    cp = plt.contourf(p1_range, p2_range, gap_data, levels=20, cmap='viridis')
    plt.colorbar(cp, label=r'$\Delta_{\mathcal{L}}$')
    plt.xlabel(f"{params[0]}")
    plt.ylabel(f"{params[1]}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return p1_range, p2_range, gap_data


def compute_cycle_heats(ham_func, ham_ref, diss_func, p_min, p_max, tau=2*np.pi, points=50, clockwise=False):
    """Computes reversible and dissipative heat for a rectangular cycle.
    Uses higher-precision central differences for numerical derivatives.
    
    Parameters
    ----------
    clockwise : bool, optional
        If False (default), traces counter-clockwise. If True, traces clockwise."""

    H_test = ham_func(*p_min)
    d = H_test.shape[0]
    d2 = d * d

    def build_liouvillian_matrix(H, p_vals):
        """Build Liouvillian superoperator as d2 x d2 matrix"""
        L_mat = np.zeros((d2, d2), dtype=complex)
        
        for col in range(d2):
            # Create basis matrix e_ij
            e = np.zeros((d, d), dtype=complex)
            e[col // d, col % d] = 1.0
            
            # Apply Liouvillian: L[e] = -i[H, e] + diss(e)
            L_e = -1j * (H @ e - e @ H) + diss_func(e)(*p_vals)
            
            # Vectorize and store as column
            L_mat[:, col] = L_e.flatten()
        
        return L_mat

    def solve_l_response(L_mat, target_vec):
        """Solve for response in null space of Liouvillian"""
        aug_mat = L_mat.copy()
        aug_mat[0, :] = np.identity(d).flatten()
        aug_rhs = target_vec.copy()
        aug_rhs[0] = 0
        return linsolve(aug_mat, aug_rhs)

    def get_numerical_data(t_val):
        """Numerical Kernel with Rectangular Parameterization"""
        segment = np.floor(t_val)
        local_t = t_val % 1
        if t_val == 4.0:
            segment = 3
            local_t = 1.0

        # Handle boundary
        if segment == 0:
            p_vals = np.array([p_min[0] + local_t * (p_max[0] - p_min[0]), p_min[1]])
            vel = np.array([p_max[0] - p_min[0], 0.0])
        elif segment == 1:
            p_vals = np.array([p_max[0], p_min[1] + local_t * (p_max[1] - p_min[1])])
            vel = np.array([0.0, p_max[1] - p_min[1]])
        elif segment == 2:
            p_vals = np.array([p_max[0] - local_t * (p_max[0] - p_min[0]), p_max[1]])
            vel = np.array([-(p_max[0] - p_min[0]), 0.0])
        else:  # segment == 3
            p_vals = np.array([p_min[0], p_max[1] - local_t * (p_max[1] - p_min[1])])
            vel = np.array([0.0, -(p_max[1] - p_min[1])])
        
        # For clockwise, reverse the direction of traversal
        if clockwise:
            vel = -vel

        # Local Matrices
        H = ham_func(*p_vals)
        H_ref = ham_ref(*p_vals)
        
        # Build Liouvillian matrix
        L_mat = build_liouvillian_matrix(H, p_vals)

        # Robust Steady State extraction
        ns = null_space(L_mat)
        rho_vec = ns[:, 0] if ns.shape[1] > 0 else svd(L_mat)[2][-1, :]
        rho = rho_vec.reshape(d, d)
        rho = rho / np.trace(rho)
        rho_vec = rho.flatten()
        
        # Higher-precision numerical differentiation (adaptive central differences)
        step = np.abs(p_vals) * 1e-6
        step = np.maximum(step, 1e-6)  # Ensure minimum step of 1e-6 for better accuracy
        drho = np.zeros((2, d, d), dtype=complex)
        dH = np.zeros((2, d, d), dtype=complex)
        
        for i in range(2):
            p_plus = p_vals.copy()
            p_plus[i] += step[i]
            p_minus = p_vals.copy()
            p_minus[i] -= step[i]
            
            H_plus = ham_func(*p_plus)
            H_minus = ham_func(*p_minus)
            
            # Build Liouvillian at perturbed points
            L_mat_plus = build_liouvillian_matrix(H_plus, p_plus)
            L_mat_minus = build_liouvillian_matrix(H_minus, p_minus)
            
            # Compute derivatives using central differences
            dl_vec = -(L_mat_plus @ rho_vec - L_mat_minus @ rho_vec) / (2 * step[i])
            drho[i, :, :] = solve_l_response(L_mat, dl_vec).reshape(d, d)
            dH[i, :, :] = (H_plus - H_minus) / (2 * step[i])

        # Compute Metrics
        Ai = np.array([np.real(np.trace(H_ref @ drho[i])) for i in range(2)])
        g_metric = np.array([[np.real(np.trace(solve_l_response(L_mat, drho[i].flatten()).reshape(d, d) @ dH[j])) for j in range(2)] for i in range(2)])
        
        return Ai @ vel, vel @ g_metric @ vel

    # Execution and Integration
    print(f"Starting cycle integration for d={d} system...")
    t_grid = np.linspace(0.0, 4.0, points)
    grid_data = np.array([get_numerical_data(t) for t in t_grid])
    
    t_step = 4.0 / points
    q_geo = t_step * np.sum(grid_data[:, 0])
    q_diss = (-1.0 / tau) * t_step * np.sum(grid_data[:, 1])

    print("--- Results ---")
    print(f"Geometric Heat: {q_geo}")
    print(f"Dissipated Heat: {q_diss}")
    print(f"Net: {q_geo + q_diss}")
    
    return q_geo, q_diss