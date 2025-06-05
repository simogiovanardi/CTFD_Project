# 1D Steady Advection-Diffusion Solver

## Overview
This project implements a numerical solver for the 1D steady advection-diffusion equation, modeling a thermal boundary layer. The solver provides two discretization schemes: central difference and first-order upwind.

## Problem Description
The steady advection-diffusion equation describes heat transfer in a fluid flow with both advection (convection) and diffusion mechanisms. The 1D steady form is:

u * dT/dx = α * d²T/dx²


where:
- `T` is temperature
- `u` is the fluid velocity
- `α` is the thermal diffusivity 
- `x` is the spatial coordinate

## Features
- Central difference scheme (second-order accurate)
- First-order upwind scheme (more stable for high Peclet numbers)
- Sparse matrix implementation for efficient computation
- Multi-resolution analysis with variable grid sizes
- Error convergence analysis with L2 and L∞ norms
- Automated logarithmic spacing for grid refinement studies
- Visualization of numerical solutions compared to exact solution
- Configurable grid size and Peclet number

## Mathematical Background
The dimensionless form of the equation introduces the Peclet number (Pe = u*L/α), which represents the ratio of advective to diffusive transport. The grid Peclet number (Pe*dx) is crucial for numerical stability.

## Usage

### Configuration
Modify the `config.json` file to set:

- `M`: Number of grid points
- `scheme`: Discretization scheme (0: central difference, 1: upwind)
- `peclet`: Peclet number for the simulation

### Running the Solver
```bash
python 1D_steady_advection-diffusion.py
```
### Multi-Resolution Analysis
Compare solutions for different grid resolutions:

```bash
compare_multiple_M(Pe, ks, high_M=100, M_values=[21, 11, 6, 5, 4, 3])
```
### Error Convergence Analysis
Evaluate numerical error trends with grid refinement:

```bash
num_points = 15  # Control number of analysis points
dx_values = np.logspace(np.log10(0.5), -5, num_points)
plot_error_convergence(Pe, ks, dx_values=dx_values)
```

### Visualization
The solution is automatically visualized using the included plotting function, showing:

- Computed vs. exact temperature profiles
- Multi-resolution comparisons with varying grid sizes
- Convergence analysis with L2 and L∞ error norms
- Peclet number and other relevant parameters

### Code Structure
- 1D_steady_advection-diffusion.py: Main solver implementation
- plot.py: Visualization functions
- compare_results.py: Advanced analysis tools for multi-resolution and error convergence
- config.json: Configuration parameters

### Dependencies
- NumPy: For numerical operations
- SciPy: For sparse matrix operations
- Matplotlib: For visualization
- JSON: For configuration handling

### Mathematical Formulation
The discretization schemes transform the continuous differential equation into a tridiagonal linear system:

### Central Difference:

    -(0.5*Pe*dx + 1)*T[i-1] + 2*T[i] + (0.5*Pe*dx - 1)*T[i+1] = 0

### First-Order Upwind:

    -(Pe*dx + 1)*T[i-1] + (2 + Pe*dx)*T[i] - T[i+1] = 0

This system is solved efficiently using sparse matrix techniques.

### Error Analysis
The code provides tools to analyze numerical errors:

- L2 norm error: Root mean square of differences between numerical and exact solutions
- L∞ norm error: Maximum absolute difference between numerical and exact solutions
- Convergence rate: Calculated from log-log slope of error vs. grid spacing

### Example Results
The solution visualizers display:

- Numerical and analytical solutions for comparison
- Multi-resolution comparisons showing the effect of grid size
- Bilogarithmic plots of error convergence with reference slopes
- Detailed convergence tables with calculated error orders
