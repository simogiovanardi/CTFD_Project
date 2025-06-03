"""
Numerical solution of a 1D steady advection-diffusion equation with Python
(1D thermal boundary layer)

--- CENTRAL DIFFERENCE SCHEME (0) ---
--- FIRST-ORDER UPWIND SCHEME (1) ---

"""

import numpy as np
import json
import os
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from plot import plot_solution_steady

def solve_advection_diffusion(M, ks, Pe, L=1, return_errors=False):
    """
    Solve the 1D steady advection-diffusion equation.
    
    Parameters:
        M (int): Number of grid points
        ks (int): Scheme type (0 for central difference, 1 for upwind)
        Pe (float): Peclet number
        L (float): Domain length
        
    Returns:
        tuple: (x, T, Texact) - grid points, numerical solution, exact solution
    """
    # Grid setup
    dx = L/(M-1)
    x = np.linspace(start=0, stop=L, num=M)
    Pec = Pe*dx  # Grid Peclet Number
    
    # Boundary conditions and exact T
    T = np.zeros(M)
    T[0] = 0
    T[-1] = 1
    Texact = (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
    
    # Weight and matrix calculation A 
    match ks:
        case 0:
            ww = -(0.5*Pec + 1)
            wc = 2
            we = 0.5*Pec - 1
        case 1:
            ww = - Pec - 1
            wc = 2 + Pec
            we = -1
    
    diagl = ww*np.ones((M-2))  # lower diagonal
    diagp = wc*np.ones((M-2))  # principal diagonal
    diagu = we*np.ones((M-2))  # upper diagonal
    
    A = sps.spdiags([diagl, diagp, diagu], [-1, 0, 1], M-2, M-2, format='csc')
    
    # Resolution
    rhs = np.zeros(M-2)
    rhs[0] = rhs[0] - ww*T[0]
    rhs[-1] = rhs[-1] - we*T[-1]
    Tint = sps.linalg.spsolve(A, rhs)
    
    # Solution
    T[1:M-1] = Tint[0:M-2]
    
    if return_errors:
        L2_error, inf_error = calculate_errors(T, Texact)
        return x, T, Texact, L2_error, inf_error
    else:
        return x, T, Texact

def compare_multiple_M(Pe, ks, high_M=100, M_values=[3, 4, 5, 6], L=1):
    """
    Compare solutions for different M values (displaying results based on Pec).
    
    Parameters:
        Pe (float): Peclet number
        ks (int): Scheme type
        high_M (int): M value for exact solution
        M_values (list): List of M values to compare
        L (float): Domain length
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate exact solution with high M
    x_exact, _, Texact = solve_advection_diffusion(high_M, ks, Pe, L)
    dx_exact = L/(high_M-1)
    Pec_exact = Pe*dx_exact
    plt.plot(x_exact, Texact, 'k-', label=f'Exact (Pec={Pec_exact:.3f})', linewidth=2)
    
    # Calculate and plot solutions for different M values
    markers = ['o', 's', '^', 'D', 'x', '*']
    for i, M in enumerate(M_values):
        x, T, _ = solve_advection_diffusion(M, ks, Pe, L)
        dx = L/(M-1)
        Pec = Pe*dx
        marker = markers[i % len(markers)]
        plt.plot(x, T, marker=marker, linestyle='-', 
                label=f'Pec={Pec:.2f} (M={M})', markevery=max(1, M//10))
    
    # Add plot details
    scheme_name = "Central Difference" if ks == 0 else "Upwind"
    plt.title(f'1D Steady Advection-Diffusion, Pe={Pe}, {scheme_name} Scheme')
    plt.xlabel('x')
    plt.ylabel('T')
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_errors(T_numerical, T_exact):
    """
    Calculate L2 norm error (root mean square) and infinity norm error (L∞)
    
    Parameters:
        T_numerical: Numerical solution array
        T_exact: Exact solution array
        
    Returns:
        tuple: (L2_error, inf_error)
    """
    # Differences
    diff = T_numerical - T_exact
    
    # L2 norm error (root mean square)
    L2_error = np.sqrt(np.mean(np.square(diff)))
    
    # Infinity norm error (L∞)
    inf_error = np.max(np.abs(diff))
    
    return L2_error, inf_error

def plot_error_convergence(Pe, ks, dx_values=None, M_values=None, L=1):
    """
    Plot L2 and L-infinity errors against grid spacing on a bilogarithmic plot.
    
    Parameters:
        Pe (float): Peclet number
        ks (int): Scheme type
        dx_values (list): Specific dx values to use (optional)
        M_values (list): Specific M values to use (optional)
        L (float): Domain length
    """
    # Handle input parameters
    if dx_values is None and M_values is None:
        # Default: use logarithmically spaced M values
        M_values = [3, 4, 6, 11, 21, 41, 81]
    
    # Calculate dx from M or vice versa
    if dx_values is not None:
        M_values = [int(L/dx) + 1 for dx in dx_values]
        # Recalculate exact dx values based on M
        dx_values = [L/(M-1) for M in M_values]
    else:
        dx_values = [L/(M-1) for M in M_values]
    
    # Arrays to store results
    L2_errors = []
    inf_errors = []
    pec_values = []
    
    # Calculate errors for each grid spacing
    for i, M in enumerate(M_values):
        _, T, Texact, L2_error, inf_error = solve_advection_diffusion(M, ks, Pe, L, return_errors=True)
        dx = dx_values[i]
        pec = Pe*dx
        
        L2_errors.append(L2_error)
        inf_errors.append(inf_error)
        pec_values.append(pec)
    
    # Create bilogarithmic plot
    plt.figure(figsize=(10, 6))
    plt.loglog(dx_values, L2_errors, 'o-', label='L2 Norm Error')
    plt.loglog(dx_values, inf_errors, 's-', label='L∞ Norm Error')
    
    # Add reference slopes for first and second order convergence
    x_ref = np.array([min(dx_values), max(dx_values)])
    for order, style, color in zip([1, 2], ['--', '-.'], ['gray', 'black']):
        y_ref = inf_errors[-1]*(x_ref/dx_values[-1])**order
        plt.loglog(x_ref, y_ref, linestyle=style, color=color, label=f'Order {order}')
    
    # Add plot details
    scheme_name = "Central Difference" if ks == 0 else "Upwind"
    plt.title(f'Error Convergence Analysis, Pe={Pe}, {scheme_name} Scheme')
    plt.xlabel('Grid Spacing (dx)')
    plt.ylabel('Error Norm')
    plt.grid(True, which="both")
    plt.legend()
    
    # Print convergence data
    print("\nError Convergence Analysis:")
    print("---------------------------------------------------------------------------------")
    print("  M  |    dx    |   Pec    | L2 Error      | L∞ Error      | L2 Order | L∞ Order")
    print("---------------------------------------------------------------------------------")
    
    for i in range(len(M_values)):
        if i > 0:
            # Calculate orders of convergence
            L2_order = np.log(L2_errors[i-1]/L2_errors[i]) / np.log(dx_values[i-1]/dx_values[i])
            inf_order = np.log(inf_errors[i-1]/inf_errors[i]) / np.log(dx_values[i-1]/dx_values[i])
            print(f"{M_values[i]:4d} | {dx_values[i]:.6f} | {pec_values[i]:.6f} | {L2_errors[i]:.4e} | {inf_errors[i]:.4e} | {L2_order:.4f} | {inf_order:.4f}")
        else:
            print(f"{M_values[i]:4d} | {dx_values[i]:.6f} | {pec_values[i]:.6f} | {L2_errors[i]:.4e} | {inf_errors[i]:.4e} | {'---':^8} | {'---':^10}")
    
    plt.show()
    
    return {
        'M_values': M_values,
        'dx_values': dx_values, 
        'pec_values': pec_values,
        'L2_errors': L2_errors,
        'inf_errors': inf_errors
    }