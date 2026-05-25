#!/usr/bin/env python3
"""
Quantum Thermodynamics Translation Project - Complete Workflow
==============================================================

This script recreates the complete workflow from the conversation:
1. Translation from Mathematica to Python
2. Implementation of core physics functions
3. Example analyses (Rubidium and Motion-TLS systems)
4. Debugging and verification

Author: GitHub Copilot
Date: May 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, solve as linsolve, svd, eig as eigenvalues

print("=" * 70)
print("QUANTUM THERMODYNAMICS SIMULATION - COMPLETE WORKFLOW")
print("=" * 70)
print("\nProject: Mathematica to Python Translation")
print("Purpose: Analyze heat curvature and thermodynamic cycles in quantum systems")
print()

# ============================================================================
# PART 1: CORE UTILITIES
# ============================================================================
print("PART 1: DEFINING CORE UTILITIES")
print("-" * 70)

# Pauli Matrices
sigma0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_up = (sigma1 + 1j * sigma2) / 2

print("✓ Pauli matrices defined")


def lindblad_dissipator(L, rho):
    """
    Lindblad dissipator for open quantum systems.
    
    D[L]ρ = Lρ L† - (1/2){L†L, ρ}
    
    Args:
        L: Lindblad operator
        rho: Density matrix
    
    Returns:
        Dissipated density matrix element
    """
    L_dag = np.conjugate(L).T
    term1 = L @ rho @ L_dag
    term2 = 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
    return term1 - term2

print("✓ Lindblad dissipator implemented")


def heat_curvature(ham_func, h_target, diss_func, params, ranges, rules, points=20):
    """
    Compute Berry curvature (heat curvature) in parameter space.
    
    The heat curvature measures the geometric phase acquired when the
    system is adiabatically evolved around a closed loop in parameter space.
    
    κ = ∂_p1(H_ref @ ∂_p2ρ) - ∂_p2(H_ref @ ∂_p1ρ)
    """
    H_test = ham_func(ranges[0][1], ranges[1][1])
    d = H_test.shape[0]
    d2 = d**2
    
    basis = np.eye(d2, d2).reshape(d2, d, d)
    vec = lambda m: m.flatten()
    unvec = lambda v: v.reshape(d, d)
    
    def solve_l_response(L_mat, target_vec):
        """Solve augmented linear system for response computation"""
        aug_mat = L_mat.copy()
        aug_mat[0, :] = vec(np.identity(d))
        aug_rhs = target_vec.copy()
        aug_rhs[0] = 0
        return linsolve(aug_mat, aug_rhs)
    
    def get_curvature_point(p1_val, p2_val):
        """Compute curvature at a single point"""
        p_vals = np.array([p1_val, p2_val])
        step = np.abs(p_vals) * 1e-4
        step[step == 0] = 1e-4
        
        # Steady state
        H_tot = ham_func(*p_vals)
        L_mat = np.zeros((d2, d2), dtype=complex)
        for i, b in enumerate(basis):
            L_mat[:, i] = vec(-1j * (H_tot @ b - b @ H_tot) + diss_func(b)(*p_vals))
        
        ns = null_space(L_mat)
        rho_vec = ns[:, 0] if ns.shape[1] > 0 else svd(L_mat)[2][-1, :]
        rho = unvec(rho_vec / np.trace(unvec(rho_vec)))
        rho_vec = vec(rho)
        
        # Parameter derivatives
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
        
        return np.real(np.trace(dH[0] @ drho[1]) - np.trace(dH[1] @ drho[0]))
    
    # Map generation
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

print("✓ Heat curvature computation implemented")


def liouvillian_gap(ham_func, diss_func, params, ranges, points=25, verbose=False):
    """
    Compute Liouvillian spectral gap across parameter space.
    
    The gap = |Re(λ_2)| where λ_2 is second-largest eigenvalue of L.
    This determines the decay rate to steady state.
    """
    H_test = ham_func(ranges[0][1], ranges[1][1])
    d = H_test.shape[0]
    d2 = d**2
    basis = np.eye(d2, d2, dtype=complex).reshape(d2, d, d)
    vec = lambda m: m.flatten()
    
    p1_range = np.linspace(ranges[0][1], ranges[0][2], points)
    p2_range = np.linspace(ranges[1][1], ranges[1][2], points)
    gap_data = np.zeros((points, points))
    
    if verbose:
        print(f"  Computing Liouvillian Gap for {d}x{d} system...")
    
    for i, p1 in enumerate(p1_range):
        for j, p2 in enumerate(p2_range):
            H = ham_func(p1, p2)
            
            L_mat = np.zeros((d2, d2), dtype=complex)
            for k, b in enumerate(basis):
                L_of_b = -1j * (H @ b - b @ H) + diss_func(b)(p1, p2)
                L_mat[:, k] = vec(L_of_b)
            
            evals, _ = eigenvalues(L_mat)
            sorted_reals = np.sort(np.real(evals))[::-1]
            gap = np.abs(sorted_reals[1])
            gap_data[j, i] = gap
    
    # Visualization
    plt.figure(figsize=(8, 6))
    p1_plot = p1_range / (2 * np.pi) if 'Δ' in params[0] else p1_range
    p2_plot = p2_range / (2 * np.pi) if 'Ω' in params[1] else p2_range
    
    cp = plt.contourf(p1_plot, p2_plot, gap_data, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Gap')
    plt.xlabel(f"{params[0]}")
    plt.ylabel(f"{params[1]}")
    plt.title('Liouvillian Spectral Gap')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return p1_range, p2_range, gap_data

print("✓ Liouvillian gap computation implemented")


def compute_cycle_heats(ham_func, ham_ref, diss_func, params, rules, 
                        p_min, p_max, tau=2*np.pi, points=50):
    """
    Compute reversible and dissipative heat for a rectangular thermodynamic cycle.
    
    Traces a rectangular path:
    (p_min[0], p_min[1]) → (p_max[0], p_min[1]) → (p_max[0], p_max[1]) 
                        → (p_min[0], p_max[1]) → (p_min[0], p_min[1])
    """
    H_test = ham_func(*p_min)
    d = H_test.shape[0]
    d2 = d**2
    basis = np.eye(d2, d2).reshape(d2, d, d)
    vec = lambda m: m.flatten()
    unvec = lambda v: v.reshape(d, d)
    
    def solve_l_response(L_mat, target_vec):
        aug_mat = L_mat.copy()
        aug_mat[0, :] = vec(np.identity(d))
        aug_rhs = target_vec.copy()
        aug_rhs[0] = 0
        return linsolve(aug_mat, aug_rhs)
    
    def get_numerical_data(t_val):
        """Data point on rectangular cycle (t: 0 to 4)"""
        segment = np.floor(t_val)
        local_t = t_val % 1
        if t_val == 4.0:
            segment = 3
            local_t = 1.0
        
        if segment == 0:
            p_vals = np.array([p_min[0] + local_t * (p_max[0] - p_min[0]), p_min[1]])
            vel = np.array([p_max[0] - p_min[0], 0.0])
        elif segment == 1:
            p_vals = np.array([p_max[0], p_min[1] + local_t * (p_max[1] - p_min[1])])
            vel = np.array([0.0, p_max[1] - p_min[1]])
        elif segment == 2:
            p_vals = np.array([p_max[0] - local_t * (p_max[0] - p_min[0]), p_max[1]])
            vel = np.array([-(p_max[0] - p_min[0]), 0.0])
        else:
            p_vals = np.array([p_min[0], p_max[1] - local_t * (p_max[1] - p_min[1])])
            vel = np.array([0.0, -(p_max[1] - p_min[1])])
        
        H = ham_func(*p_vals)
        H_ref = ham_ref(*p_vals)
        
        L_mat = np.zeros((d2, d2), dtype=complex)
        for i, b in enumerate(basis):
            L_mat[:, i] = vec(-1j * (H @ b - b @ H) + diss_func(b)(*p_vals))
        
        ns = null_space(L_mat)
        rho_vec = ns[:, 0] if ns.shape[1] > 0 else svd(L_mat)[2][-1, :]
        rho = unvec(rho_vec / np.trace(unvec(rho_vec)))
        rho_vec = vec(rho)
        
        step = 1e-6
        drho = np.zeros((2, d, d), dtype=complex)
        for i in range(2):
            p_step = p_vals.copy()
            p_step[i] += step
            H_step = ham_func(*p_step)
            
            L_mat_step = np.zeros((d2, d2), dtype=complex)
            for j, b in enumerate(basis):
                L_mat_step[:, j] = vec(-1j * (H_step @ b - b @ H_step) + diss_func(b)(*p_step))
            
            dl_vec = -(L_mat_step @ rho_vec - L_mat @ rho_vec) / step
            drho[i, :, :] = unvec(solve_l_response(L_mat, dl_vec))
        
        Ai = np.array([np.real(np.trace(H_ref @ drho[i])) for i in range(2)])
        g_metric = np.array([[np.real(np.trace(drho[i] @ unvec(solve_l_response(L_mat, vec(drho[j]))))) 
                              for j in range(2)] for i in range(2)])
        
        return Ai @ vel, vel @ g_metric @ vel
    
    print(f"  Starting rectangular cycle integration for d={d} system...")
    t_grid = np.linspace(0.0, 4.0, points)
    grid_data = np.array([get_numerical_data(t) for t in t_grid])
    
    t_step = 4.0 / points
    q_geo = t_step * np.sum(grid_data[:, 0])
    q_diss = (-1.0 / tau) * t_step * np.sum(grid_data[:, 1])
    
    return q_geo, q_diss

print("✓ Cycle heat computation implemented")
print()

# ============================================================================
# PART 2: EXAMPLE 1 - 3-LEVEL RUBIDIUM ATOM
# ============================================================================
print("PART 2: 3-LEVEL RUBIDIUM ATOM EXAMPLE")
print("-" * 70)


def HR(Delta, Omega):
    """3-level Rubidium atom Hamiltonian"""
    Omega1 = Omega
    Omega2 = Omega
    Delta1 = Delta / 10
    Delta2 = Delta
    H = np.array([
        [0, 0, Omega1 / 2],
        [0, Delta1, Omega2 / 2],
        [Omega1 / 2, Omega2 / 2, Delta2]
    ], dtype=complex)
    return H


def diss_Rb(r):
    """Total dissipator for Rubidium"""
    def LDiss1(rho):
        L = (2 * np.pi * 6.1) * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
        return lindblad_dissipator(L, rho)
    
    def LDiss2(rho):
        L = (2 * np.pi * 6.1) * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
        return lindblad_dissipator(L, rho)
    
    def LD1(rho):
        L = 0.0 * np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=complex)
        return lindblad_dissipator(L, rho)
    
    def LD2(rho):
        L = 0.0 * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
        return lindblad_dissipator(L, rho)
    
    def func(Delta, Omega):
        return LDiss1(r) + LDiss2(r) + LD1(r) + LD2(r)
    
    return func


print("Parameters:")
print("  Detuning Δ: [-2π·100, 2π·100] MHz")
print("  Rabi frequency Ω: [2π·1, 2π·10] MHz")
print("  Dissipation: γ_d = 2π·6.1 MHz")
print()

print("Computing heat curvature (20×20 grid)...")
p1_grid_rb, p2_grid_rb, curv_rb, max_rb = heat_curvature(
    HR, HR, diss_Rb,
    params=['Δ', 'Ω'],
    ranges=[
        ['Δ', -2 * np.pi * 100, 2 * np.pi * 100],
        ['Ω', 2 * np.pi, 2 * np.pi * 10]
    ],
    rules={},
    points=20
)
print(f"✓ Heat curvature: max value = {max_rb:.2e}")

print("Computing Liouvillian gap (12×12 grid)...")
_, _, gap_rb = liouvillian_gap(
    HR, diss_Rb,
    params=['Δ', 'Ω'],
    ranges=[
        ['Δ', -2 * np.pi * 100, 2 * np.pi * 100],
        ['Ω', 2 * np.pi, 2 * np.pi * 10]
    ],
    points=12,
    verbose=True
)
print("✓ Liouvillian gap computed")

print("Computing thermodynamic cycle heats...")
q_geo_rb, q_diss_rb = compute_cycle_heats(
    HR, HR, diss_Rb,
    params=['Δ', 'Ω'],
    rules={},
    p_min=[-2 * np.pi * 100, 2 * np.pi],
    p_max=[2 * np.pi * 100, 2 * np.pi * 10],
    tau=2 * np.pi,
    points=30
)
print(f"✓ Geometric heat: {q_geo_rb:.6f}")
print(f"✓ Dissipated heat: {q_diss_rb:.6f}")
print()

# ============================================================================
# PART 3: EXAMPLE 2 - MOTION-TLS COUPLED SYSTEM
# ============================================================================
print("PART 3: MOTION-TLS COUPLED SYSTEM EXAMPLE")
print("-" * 70)

N_mot = 3
a_diag_vals = np.sqrt(np.arange(1, N_mot))
a_op = np.diag(a_diag_vals, k=1)
adag_op = a_op.T.conj()


def H_mtls(omega_rf, omega_rabi):
    """Motion-TLS Hamiltonian"""
    x0 = 1e-5
    omega_mot = 2 * np.pi * 500
    mu_B = 9.27e-24
    gF = 0.5
    B0 = 1e-3
    dB = 1.0
    hbar = 1.05e-34
    m = 1.44e-25
    
    x_ho = np.sqrt(hbar / (2 * m * omega_mot))
    eta = gF * mu_B * x_ho * dB / hbar
    delta = omega_rf - gF * mu_B * B0 / hbar
    
    H_mot = omega_mot * (np.kron(sigma0, adag_op @ a_op) - x0 * np.kron(sigma0, adag_op + a_op))
    H_spin = 0.5 * np.kron(delta * sigma3 + omega_rabi * sigma1, np.eye(N_mot))
    H_coup = np.kron(sigma3, -eta * (adag_op + a_op))
    
    return H_mot + H_spin + H_coup


def H_mtls_ref(omega_rf, omega_rabi):
    """Reference Hamiltonian (photon number)"""
    return np.kron(np.eye(2), adag_op @ a_op)


def diss_mtls(r):
    """Dissipator for Motion-TLS"""
    def func(omega_rf, omega_rabi):
        gamma_up = 1e3
        gamma_phi = 1.0
        gamma_m = 1.0
        n_th = 100
        
        L_up = np.sqrt(gamma_up) * np.kron(sigma_up, np.eye(N_mot))
        L_phi = np.sqrt(gamma_phi / 2) * np.kron(sigma3, np.eye(N_mot))
        L_cool = np.sqrt(gamma_m * (n_th + 1)) * np.kron(np.eye(2), a_op)
        L_heat = np.sqrt(gamma_m * n_th) * np.kron(np.eye(2), adag_op)
        
        return (lindblad_dissipator(L_up, r) +
                lindblad_dissipator(L_phi, r) +
                lindblad_dissipator(L_cool, r) +
                lindblad_dissipator(L_heat, r))
    return func


print("Parameters:")
print("  RF frequency ω_RF: [10^6, 10^7] Hz")
print("  Rabi frequency Ω: [2π·10, 2π·10^4] Hz")
print("  Motion frequency: 2π·500 Hz")
print("  Dissipation: γ_up = 10³ Hz, γ_φ = 1 Hz, Γ_m = 1 Hz")
print()

print("Computing heat curvature (12×12 grid)...")
p1_grid_mtls, p2_grid_mtls, curv_mtls, max_mtls = heat_curvature(
    H_mtls, H_mtls_ref, diss_mtls,
    params=['ω_RF', 'Ω'],
    ranges=[
        ['ω_RF', 1e6, 1e7],
        ['Ω', 2 * np.pi * 10, 2 * np.pi * 1e4]
    ],
    rules={},
    points=12
)
print(f"✓ Heat curvature: max value = {max_mtls:.2e}")

print("Computing Liouvillian gap (12×12 grid)...")
_, _, gap_mtls = liouvillian_gap(
    H_mtls, diss_mtls,
    params=['ω_RF', 'Ω'],
    ranges=[
        ['ω_RF', 1e6, 1e7],
        ['Ω', 2 * np.pi * 10, 2 * np.pi * 1e4]
    ],
    points=12,
    verbose=True
)
print("✓ Liouvillian gap computed")

print("Computing thermodynamic cycle heats...")
q_geo_mtls, q_diss_mtls = compute_cycle_heats(
    H_mtls, H_mtls_ref, diss_mtls,
    params=['ω_RF', 'Ω'],
    rules={},
    p_min=[1e6, 2 * np.pi * 10],
    p_max=[1e7, 2 * np.pi * 1e4],
    tau=2 * np.pi,
    points=30
)
print(f"✓ Geometric heat: {q_geo_mtls:.6f}")
print(f"✓ Dissipated heat: {q_diss_mtls:.6f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nProject completion status:")
print("✓ Mathematica code translated to Python")
print("✓ Core physics functions implemented")
print("✓ Debugging completed (3 major issues fixed)")
print("✓ Example 1: Rubidium atom analysis completed")
print("✓ Example 2: Motion-TLS system analysis completed")
print()
print("Key debugging fixes:")
print("1. NameError: Added ns = null_space(L_mat) in compute_cycle_heats")
print("2. TypeError: Fixed diss_func calling convention to diss_func(b)(p1, p2)")
print("3. ValueError: Unpacked eigenvalue tuple as evals, _ = eig(L_mat)")
print()
print("Files created:")
print("  • utils.py - All computational functions (exported)")
print("  • SS_work.ipynb - Jupyter notebook with examples (clean)")
print("  • workflow_summary.py - This file (complete documentation)")
print()
print("=" * 70)
