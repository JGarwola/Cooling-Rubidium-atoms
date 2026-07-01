import numpy as np
from utils_GPE import *
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

#==========================================
# Plotting Functions
#==========================================

def plot_steady_state(x_grid, rho_ee, rho_gg, rho_eg):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    # Density plot
    ax1.plot(x_grid, rho_ee.real, label=r'$\rho_{ee}(r)$', color='crimson', lw=2)
    ax1.plot(x_grid, rho_gg.real, label=r'$\rho_{gg}(r)$', color='dodgerblue', lw=2)
    ax1.grid(True, linestyle='--')
    ax1.legend()
    
    # Coherence plot
    ax2.plot(x_grid, rho_eg.real, label=r'$\mathrm{Re}[\rho_{eg}(r)]$', color='darkorange', lw=1.5)
    ax2.plot(x_grid, np.abs(rho_eg), label=r'$|\rho_{eg}(r)|$', color='purple', lw=1.5, linestyle='--')
    ax2.set_xlabel('Spatial coordinate $r$')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "figures/steady_state_GPE.pdf"), dpi=300)



def plot_initial_final(x_real, k_momentum, 
                               rho_ee_hist_real, rho_gg_hist_real, rho_eg_hist_real,
                               rho_ee_fft, rho_gg_fft, rho_eg_fft):
    """
    Plots the initial and final states in both real and momentum space.
    """
    # Real space initial and final
    rho_gg_init_real = np.real(rho_gg_hist_real[0, :])
    rho_ee_init_real = np.real(rho_ee_hist_real[0, :])
    rho_eg_init_real_abs = np.abs(rho_eg_hist_real[0, :])

    rho_gg_final_real = np.real(rho_gg_hist_real[-1, :])
    rho_ee_final_real = np.real(rho_ee_hist_real[-1, :])
    rho_eg_final_real_abs = np.abs(rho_eg_hist_real[-1, :])

    # Momentum space initial and final (using the magnitude of the FFT data)
    rho_gg_init_mom = np.real(rho_gg_fft[0, :])
    rho_ee_init_mom = np.real(rho_ee_fft[0, :])
    rho_eg_init_mom = np.abs(rho_eg_fft[0, :])

    rho_gg_final_mom = np.real(rho_gg_fft[-1, :])
    rho_ee_final_mom = np.real(rho_ee_fft[-1, :])
    rho_eg_final_mom = np.abs(rho_eg_fft[-1, :])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # ------------------------------------------
    # Initial State Plot (Real Space)
    # ------------------------------------------
    axs[0,0].plot(x_real, rho_gg_init_real, label=r'$\tilde{{\rho}}_{{gg}}$ (width = {:.3f})'.format(width(x_real, rho_gg_init_real)), color='tab:blue', lw=2)
    axs[0,0].plot(x_real, rho_ee_init_real, label=r'$\tilde{{\rho}}_{{ee}}$ (width = {:.3f})'.format(width(x_real, rho_ee_init_real)), color='tab:orange', lw=2)
    axs[0,0].plot(x_real, rho_eg_init_real_abs, label=r'$|\tilde{{\rho}}_{{eg}}|$ (width = {:.3f})'.format(width(x_real, rho_eg_init_real_abs)), color='tab:green', lw=2)
    axs[0,0].set_title('Initial State (Real Space)')
    axs[0,0].set_xlabel('$r$')
    axs[0,0].set_ylabel('Density')
    axs[0,0].legend(fontsize=11)

    # ------------------------------------------
    # Final State Plot (Real Space)
    # ------------------------------------------
    axs[0,1].plot(x_real, rho_gg_final_real, label=r'$\tilde{{\rho}}_{{gg}}$ (width = {:.3f})'.format(width(x_real, rho_gg_final_real)), color='tab:blue', lw=2)
    axs[0,1].plot(x_real, rho_ee_final_real, label=r'$\tilde{{\rho}}_{{ee}}$ (width = {:.3f})'.format(width(x_real, rho_ee_final_real)), color='tab:orange', lw=2)
    axs[0,1].plot(x_real, rho_eg_final_real_abs, label=r'$|\tilde{{\rho}}_{{eg}}|$ (width = {:.3f})'.format(width(x_real, rho_eg_final_real_abs)), color='tab:green', lw=2)
    axs[0,1].set_title('Final State (Real Space)')
    axs[0,1].set_xlabel('$r$')
    axs[0,1].set_ylabel('Density')
    axs[0,1].legend(fontsize=11)

    # ------------------------------------------
    # Initial State Plot (Momentum Space)
    # ------------------------------------------
    axs[1,0].plot(k_momentum, rho_gg_init_mom, label=r'$\tilde{{\rho}}_{{gg}}$ (width = {:.3f})'.format(width(k_momentum, rho_gg_init_mom)), color='tab:blue', lw=2)
    axs[1,0].plot(k_momentum, rho_ee_init_mom, label=r'$\tilde{{\rho}}_{{ee}}$ (width = {:.3f})'.format(width(k_momentum, rho_ee_init_mom)), color='tab:orange', lw=2)
    axs[1,0].plot(k_momentum, rho_eg_init_mom, label=r'$|\tilde{{\rho}}_{{eg}}|$ (width = {:.3f})'.format(width(k_momentum, rho_eg_init_mom)), color='tab:green', lw=2)
    axs[1,0].set_title('Initial State (Momentum Space)')
    axs[1,0].set_xlabel('$k$')
    axs[1,0].set_ylabel('Density')
    axs[1,0].set_xlim([np.min(k_momentum)/10, np.max(k_momentum)/10])
    axs[1,0].legend(fontsize=11)

    # ------------------------------------------
    # Final State Plot (Momentum Space)
    # ------------------------------------------
    axs[1,1].plot(k_momentum, rho_gg_final_mom, label=r'$\tilde{{\rho}}_{{gg}}$ (width = {:.3f})'.format(width(k_momentum, rho_gg_final_mom)), color='tab:blue', lw=2)
    axs[1,1].plot(k_momentum, rho_ee_final_mom, label=r'$\tilde{{\rho}}_{{ee}}$ (width = {:.3f})'.format(width(k_momentum, rho_ee_final_mom)), color='tab:orange', lw=2)
    axs[1,1].plot(k_momentum, rho_eg_final_mom, label=r'$|\tilde{{\rho}}_{{eg}}|$ (width = {:.3f})'.format(width(k_momentum, rho_eg_final_mom)), color='tab:green', lw=2)
    axs[1,1].set_title('Final State (Momentum Space)')
    axs[1,1].set_xlabel('$k$')
    axs[1,1].set_ylabel('Density')
    axs[1,1].set_xlim([np.min(k_momentum)/10, np.max(k_momentum)/10])
    axs[1,1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'figures/initial_final_GPE.pdf'), dpi=300)

#==========================================
# Main Execution
#==========================================

# x_grid, rho_ee, rho_gg, rho_eg = simulate_steady_state(1.0, 0.0, 1.0, t_relax=5.0, dt=1e-3, T_obs=10.0)
# plot_steady_state(x_grid, rho_ee, rho_gg, rho_eg)

T = 50.0  # Period for modulation = simulation time -> one cycle
dt = 1e-3
n_traj = 4

def Omega_func(t):
    return 1.0 + 0.5 * np.sin(2 * np.pi * t / T)  # Example: sinusoidal modulation

def Delta_func(t):
    return np.cos(2 * np.pi * t / T)

def omega_func(t):
    return 1.0

times, x, rho_ee_hist, rho_gg_hist, rho_eg_hist, k, rho_ee_fft, rho_gg_fft, rho_eg_fft = simulate_ensemble_dynamics(T, dt, n_traj, Omega_func, Delta_func, omega_func)
plot_initial_final(x, k, rho_ee_hist, rho_gg_hist, rho_eg_hist, rho_ee_fft, rho_gg_fft, rho_eg_fft)