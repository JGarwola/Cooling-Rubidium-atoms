"""
Utility functions for quantum thermodynamics simulations.
Includes Lindblad dissipators, heat curvature, Liouvillian gap, and cycle heat calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, solve as linsolve, svd, eig as eigenvalues
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


def heat_curvature(H_tot, H_ref, diss, ranges, points=20):
    """
    Numerical computation of heat curvature F^Q_{ij} = d_i A^Q_j - d_j A^Q_i
    where A^Q_i = Tr(H_ref d_i rho_ss)
    """

    def liouvillian(H, rho, diss_op):
        """Compute L[rho] = -i[H, rho] + dissipator(rho)"""
        commutator = -1j * (H @ rho - rho @ H)
        return commutator + diss_op(rho)

    def get_steady_state(p_vals):
        """Find steady state as null space of Liouvillian superoperator"""
        H = H_tot(*p_vals)
        d = H.shape[0]
        d2 = d * d
        
        # Build Liouvillian superoperator matrix: vec(L[ρ]) = L_mat @ vec(ρ)
        L_mat = np.zeros((d2, d2), dtype=complex)
        
        for col in range(d2):
            # Create basis matrix e_ij (only one element = 1)
            e = np.zeros((d, d), dtype=complex)
            e[col // d, col % d] = 1.0
            
            # Apply Liouvillian: L[e]
            L_e = liouvillian(H, e, lambda r: diss(r)(*p_vals))
            
            # Vectorize and store as column
            L_mat[:, col] = L_e.flatten()
        
        # Steady state is in null space of L_mat
        ns = null_space(L_mat)
        if ns.shape[1] > 0:
            rho_vec = ns[:, 0]
        else:
            # Fallback: right singular vector corresponding to smallest singular value
            _, _, Vh = svd(L_mat)
            rho_vec = Vh[-1, :]
        
        # Unvec and normalize
        rho = rho_vec.reshape(d, d)
        trace = np.trace(rho)
        if np.abs(trace) > 1e-10:
            rho = rho / trace
        else:
            # Fallback to maximally mixed state if normalization fails
            rho = np.eye(d) / d
        
        return rho

    def compute_A_Q(p_vals):
        """Compute heat vector A^Q_i = Tr(H_ref d_i rho_ss)"""

        H = H_tot(*p_vals)
        H_ref_val = H_ref(*p_vals)
        rho_ss = get_steady_state(p_vals)
        step = np.abs(p_vals) * 1e-6
        
        # Compute d_i rho_ss using finite differences
        drho = np.zeros((2, H.shape[0], H.shape[0]), dtype=complex)
        for i in range(2):
            p_step = p_vals.copy()
            p_step[i] += step[i]
            rho_ss_step = get_steady_state(p_step)
            drho[i] = (rho_ss_step - rho_ss) / step[i]
        
        # A^Q_i = Tr(H_ref d_i rho_ss)
        A_Q = np.array([np.real(np.trace(H_ref_val @ drho[i])) for i in range(2)])
        return A_Q

    def get_curvature_point(p1_val, p2_val):
        """Compute curvature at a single point"""

        p_vals = np.array([p1_val, p2_val])
        step = np.abs(p_vals) * 1e-6
        
        A_Q_center = compute_A_Q(p_vals)
        
        # Finite difference for d_i A^Q_j
        dA_Q = np.zeros((2, 2))
        for i in range(2):
            p_step = p_vals.copy()
            p_step[i] += step[i]
            A_Q_step = compute_A_Q(p_step)
            dA_Q[i] = (A_Q_step - A_Q_center) / step[i]
        
        # Curvature: F^Q = d_0 A^Q_1 - d_1 A^Q_0
        return dA_Q[0, 1] - dA_Q[1, 0]

    # Generate map
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


def compute_cycle_heats(ham_func, ham_ref, diss_func, p_min, p_max, tau=2*np.pi, points=50):
    """Computes reversible and dissipative heat for a rectangular cycle."""

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
        
        # Finite Difference for d_i(rho) and d_i(H)
        step = 1e-6
        drho = np.zeros((2, d, d), dtype=complex)
        dH = np.zeros((2, d, d), dtype=complex)
        
        for i in range(2):
            p_step = p_vals.copy()
            p_step[i] += step
            H_step = ham_func(*p_step)
            
            # Build Liouvillian at perturbed point
            L_mat_step = build_liouvillian_matrix(H_step, p_step)
            
            dl_vec = -(L_mat_step @ rho_vec - L_mat @ rho_vec) / step
            drho[i, :, :] = solve_l_response(L_mat, dl_vec).reshape(d, d)
            dH[i, :, :] = (H_step - H) / step

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