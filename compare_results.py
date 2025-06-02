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

def solve_advection_diffusion(M, ks, Pe, L=1):
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
                label=f'Pec={Pec:.3f} (M={M})', markevery=max(1, M//10))
    
    # Add plot details
    scheme_name = "Central Difference" if ks == 0 else "Upwind"
    plt.title(f'1D Steady Advection-Diffusion, Pe={Pe}, {scheme_name} Scheme')
    plt.xlabel('x')
    plt.ylabel('T')
    plt.grid(True)
    plt.legend()
    plt.show()


    

script_dir = os.path.dirname(os.path.abspath(__file__))  # path to script
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)
ks = config["scheme"] 
Pe = config["peclet"] 

# Compare multiple M values
compare_multiple_M(Pe, ks, high_M=100, M_values=[21, 11, 6])
