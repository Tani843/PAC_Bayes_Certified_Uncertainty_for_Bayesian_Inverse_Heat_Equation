---
layout: page
title: Methodology
permalink: /methodology/
---

# Methodology

## Problem Formulation

We consider the 1D heat equation with unknown thermal conductivity:

$$\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\left(k(x) \frac{\partial u}{\partial x}\right) + f(x,t)$$

**Given**: Sparse, noisy temperature observations $\{u(x_i, t_j)\}_{i,j}$  
**Find**: Thermal conductivity $k(x)$ with certified uncertainty bounds

## Forward Model

### Finite Difference Discretization

We discretize the heat equation using implicit finite differences:

$$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \frac{1}{\Delta x^2}\left[k_{i+1/2}(u_{i+1}^{n+1} - u_i^{n+1}) - k_{i-1/2}(u_i^{n+1} - u_{i-1}^{n+1})\right]$$

where $k_{i+1/2}$ represents the harmonic mean of conductivity at cell interfaces:

$$k_{i+1/2} = \frac{2k_i k_{i+1}}{k_i + k_{i+1}}$$

### Implementation Features

- **Implicit scheme**: Unconditionally stable for large time steps
- **Variable conductivity**: Handles discontinuous material properties
- **Flexible boundary conditions**: Dirichlet, Neumann, or mixed
- **Source terms**: Supports heat generation/absorption

## Bayesian Inference

### Prior Distribution

$$P(\kappa) = \text{Uniform}[1, 10]$$

### Likelihood Model

$$P(\text{data}|\kappa) = \prod_{i=1}^{N_{\text{obs}}} \mathcal{N}(y_i | u(x_i, t_i; \kappa), \sigma^2)$$

### Posterior Distribution

$$P(\kappa|\text{data}) \propto P(\text{data}|\kappa) \cdot P(\kappa)$$

### Enhanced MCMC Sampling

We use the **emcee ensemble sampler** with:

- **Ensemble walkers**: Robust exploration of parameter space
- **Adaptive scaling**: Automatic step size tuning
- **Convergence diagnostics**: R-hat, effective sample size, autocorrelation time
- **Parallel computation**: Efficient sampling on multi-core systems

### Convergence Assessment

Key diagnostics include:

- **Gelman-Rubin R-hat**: $\hat{R} < 1.1$ indicates convergence
- **Effective Sample Size**: ESS > 100 per parameter recommended
- **Autocorrelation Time**: $\tau_{auto} < N_{samples}/50$ for efficiency

## PAC-Bayes Bounds

### Theorem

For any prior $P$ and confidence $\delta$:

$$\mathbb{P}\left[\text{KL}(Q \| P) \leq \frac{1}{n}\left(\log \frac{1}{\delta} + \log \mathbb{E}_P[e^{n\ell}]\right)\right] \geq 1-\delta$$

where:
- $Q$ = posterior distribution
- $P$ = prior distribution  
- $n$ = sample size
- $\ell$ = loss function

### Implementation

1. **Compute posterior $Q$** via MCMC
2. **Estimate $\mathbb{E}_P[e^{n\ell}]$** via Monte Carlo
3. **Construct certified interval** from KL bound

### Risk Function

We define the risk as negative log-likelihood:

$$R(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}}[\log p(y|x,\theta)]$$

For Gaussian observation noise:

$$R(\theta) = \frac{1}{2\sigma^2}\mathbb{E}[||y - G(\theta, x)||^2] + \text{const}$$

where $G(\theta, x)$ is the forward model prediction.

### Implementation Details

1. **Empirical Risk**: Computed via forward model evaluation on training data
2. **KL Divergence**: Estimated from MCMC samples using Gaussian approximation
3. **Bound Computation**: Direct evaluation of PAC-Bayes formula
4. **Prediction Intervals**: Monte Carlo sampling from posterior for confidence bounds

## Validation Framework

### Synthetic Data Generation

We test on multiple thermal conductivity profiles:
- **Smooth functions**: Linear, quadratic, sinusoidal, exponential
- **Discontinuous profiles**: Step functions, layered structures
- **Random realizations**: Gaussian process samples

### Performance Metrics

- **Coverage Rate**: Fraction of true parameters within confidence intervals
- **Interval Width**: Mean width of uncertainty bounds
- **Computational Efficiency**: Runtime scaling with grid resolution
- **Noise Robustness**: Performance across noise levels 0.001-0.1

### Cross-Validation

We employ k-fold temporal cross-validation:
1. Split time series into training/testing segments
2. Train on early observations, test on later measurements
3. Validate bounds on unseen data from same physical system

This rigorous methodology ensures that our certified bounds are both theoretically sound and practically reliable.