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
from plot import plot_solution_steady
from compare_results import compare_multiple_M
from compare_results import plot_error_convergence



script_dir = os.path.dirname(os.path.abspath(__file__))  # path to script
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)  # read parameters from a config json file
M = config["M"] # number of grid points
ks = config["scheme"] # scheme: 0 --> central difference, 1 --> upwind
Pe = config["peclet"] # Peclet number

L = 1 # total length of the domain


# Grid setup
dx = L/(M-1); # space-step (x direction)
x = np.linspace(start=0, stop=L, num=M) # grid points

Pec = Pe*dx # Grid Peclet Number

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

diagl = ww*np.ones((M-2)) # lower diagonal
diagp = wc*np.ones((M-2)) # principal diagonal
diagu = we*np.ones((M-2)) # upper diagonal

A = sps.spdiags([diagl, diagp, diagu], [-1, 0, 1], M-2, M-2, format='csc') # creating system matrix

# Resolution
rhs= np.zeros(M-2) # right-hand side initialization (A*T = b) 
rhs[0] = rhs[0] - ww*T[0] # -Ww*T1
rhs[-1] = rhs[-1] - we*T[-1] # -We*T(M)
Tint = sps.linalg.spsolve(A,rhs)

def solution(T,Tint,M):
    T[1:M-1] = Tint[0:M-2]
    return T

T = solution(T,Tint,M)

# Plot the solution (fixed M)
plot_solution_steady(x,T,Texact,Pe,dx)

# Compare multiple M values to evaluate Pec
# compare_multiple_M(Pe, ks, high_M=100, M_values=[21, 11, 6, 5, 4, 3])

# Evaluate the accuracy (Errors L2 & L∞)
# plot_error_convergence(Pe, ks, dx_values=[0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0001, 0.00001], L=L)