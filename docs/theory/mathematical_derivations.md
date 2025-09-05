---
layout: default
title: Mathematical Derivations
permalink: /theory/derivations/
---

# Mathematical Derivations

## Heat Equation Finite Difference Scheme

### Derivation of Forward Euler Method

Starting from the heat equation:

$$u_t = \kappa u_{xx}$$

Temporal discretization (Forward Euler):

$$\frac{\partial u}{\partial t} \approx \frac{u_i^{n+1} - u_i^n}{\Delta t}$$

Spatial discretization (Central differences):

$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{(\Delta x)^2}$$

Complete scheme:

$$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \kappa \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{(\Delta x)^2}$$

Explicit update:

$$u_i^{n+1} = u_i^n + \frac{\kappa \Delta t}{(\Delta x)^2}(u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

### CFL Stability Analysis

Define $r = \frac{\kappa \Delta t}{(\Delta x)^2}$. The update becomes:

$$u_i^{n+1} = (1-2r)u_i^n + r(u_{i+1}^n + u_{i-1}^n)$$

Von Neumann stability analysis: Substitute $u_j^n = \xi^n e^{ijk\Delta x}$:

$$\xi = 1 - 2r + 2r\cos(k\Delta x) = 1 - 2r(1 - \cos(k\Delta x))$$

For stability, $|\xi| \leq 1$ for all $k$:

- Maximum occurs when $\cos(k\Delta x) = -1$: $\xi = 1 - 4r$
- Stability requires: $|1 - 4r| \leq 1$

This gives: $0 \leq r \leq \frac{1}{2}$

**CFL Condition:** $\frac{\kappa \Delta t}{(\Delta x)^2} \leq \frac{1}{2}$

## Bayesian Inference Formulation

### Likelihood Function

For Gaussian measurement noise:

$$P(y_i | \kappa) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - u(x_i, t_i; \kappa))^2}{2\sigma^2}\right)$$

Joint likelihood:

$$P(\mathbf{y} | \kappa) = \prod_{i=1}^N P(y_i | \kappa) = (2\pi\sigma^2)^{-N/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - u(x_i, t_i; \kappa))^2\right)$$

Log-likelihood:

$$\ell(\kappa) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - u(x_i, t_i; \kappa))^2$$

### Posterior Distribution

With uniform prior $P(\kappa) = \frac{1}{b-a}$ on $[a,b]$:

$$P(\kappa | \mathbf{y}) \propto P(\mathbf{y} | \kappa) P(\kappa) \propto \exp(\ell(\kappa))$$

Normalized posterior:

$$P(\kappa | \mathbf{y}) = \frac{\exp(\ell(\kappa))}{\int_a^b \exp(\ell(\kappa')) d\kappa'}$$

## Error Analysis

### Truncation Error

- Temporal error: $O(\Delta t)$
- Spatial error: $O((\Delta x)^2)$ 
- Global error: $O(\Delta t + (\Delta x)^2)$

### Convergence Rate

For fixed final time $T$:

- Time steps: $M = T/\Delta t$
- Grid points: $N = L/\Delta x$
- Computational cost: $O(MN) = O\left(\frac{T}{\Delta t} \cdot \frac{L}{\Delta x}\right)$