---
layout: default
title: "Methodology"
permalink: /methodology/
---

# Methodology

This page details the complete technical methodology for applying PAC-Bayes certified bounds to Bayesian inverse problems in partial differential equations.

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Forward Model](#forward-model)
3. [Bayesian Inference Framework](#bayesian-inference-framework)
4. [PAC-Bayes Certified Bounds](#pac-bayes-certified-bounds)
5. [Implementation Details](#implementation-details)
6. [Validation Framework](#validation-framework)

## Problem Formulation

We consider the inverse problem of estimating the thermal conductivity parameter $\kappa$ in the 1D heat equation from noisy sensor observations.

### Mathematical Setup
**Forward Model**: 1D heat equation on domain $[0, L]$
$$\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, L], \; t > 0$$

**Boundary Conditions**: Homogeneous Dirichlet
$$u(0, t) = u(L, t) = 0, \quad t > 0$$

**Initial Condition**: Gaussian pulse
$$u(x, 0) = \exp\left(-50(x - 0.3)^2\right)$$

**Observations**: Noisy sensor measurements
$$y_{i,j} = u(x_j, t_i) + \epsilon_{i,j}, \quad \epsilon_{i,j} \sim \mathcal{N}(0, \sigma^2)$$

where sensors are located at $x_j \in \{0.25L, 0.5L, 0.75L\}$ and observations are taken at times $t_i$.

## Forward Model

### Finite Difference Discretization
We discretize the heat equation using explicit finite differences:

**Spatial discretization**: $x_k = k \Delta x$, $k = 0, 1, \ldots, N_x$
$$\Delta x = \frac{L}{N_x}$$

**Temporal discretization**: $t_n = n \Delta t$, $n = 0, 1, \ldots, N_t$

**Discrete scheme**:
$$u_k^{n+1} = u_k^n + \frac{\kappa \Delta t}{(\Delta x)^2}\left(u_{k+1}^n - 2u_k^n + u_{k-1}^n\right)$$

### Stability Constraint

For numerical stability, we enforce the CFL condition:
$$\frac{\kappa \Delta t}{(\Delta x)^2} \leq 0.25$$

This ensures $\Delta t \leq 0.25 \frac{(\Delta x)^2}{\kappa}$ for stability.

### Implementation Features
- **Adaptive timestep**: Automatically adjusts $\Delta t$ based on $\kappa$ and CFL constraint
- **Boundary handling**: Enforces homogeneous Dirichlet conditions
- **Interpolation**: Extracts sensor values via spatial interpolation
- **Vectorized computation**: Efficient NumPy implementation

## Bayesian Inference Framework

### Prior Distribution

We assume a uniform prior on the thermal conductivity:
$$P(\kappa) = \text{Uniform}[1, 10]$$

This reflects limited prior knowledge about the parameter range.

### Likelihood Model
Given observations $\mathbf{y} = \{y_{i,j}\}$, the likelihood assumes independent Gaussian noise:
$$P(\mathbf{y} | \kappa) = \prod_{i,j} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_{i,j} - u(x_j, t_i; \kappa))^2}{2\sigma^2}\right)$$

**Log-likelihood**:
$$\log P(\mathbf{y} | \kappa) = -\frac{1}{2\sigma^2} \sum_{i,j} (y_{i,j} - u(x_j, t_i; \kappa))^2 - C$$

where $C$ is a normalization constant.

### Posterior Sampling
**MCMC Implementation**: We use the `emcee` ensemble sampler with:
- **Multiple chains**: 3 parallel chains for convergence assessment
- **Burn-in period**: 50% of samples discarded as burn-in
- **Convergence diagnostic**: Gelman-Rubin $\hat{R}$ statistic
- **Target acceptance rate**: 0.5-0.8 for optimal efficiency

**Convergence Criterion**: $\hat{R} < 1.01$ for all parameters

## PAC-Bayes Certified Bounds

### Bounded Loss Function
We define a bounded per-datum loss:
$$\tilde{\ell}_i(\kappa) = \min\left(\frac{(y_i - \hat{y}_i(\kappa))^2}{(c\sigma)^2}, 1\right)$$

where:
- $c = 3.0$ is the bounding parameter
- $\hat{y}_i(\kappa)$ is the predicted observation for parameter $\kappa$
- The loss is bounded in $[0, 1]$

**Aggregate empirical risk**:
$$\hat{R}_Q = \mathbb{E}_{\kappa \sim Q}\left[\frac{1}{n}\sum_{i=1}^n \tilde{\ell}_i(\kappa)\right]$$

### PAC-Bayes-KL Bound
For any posterior distribution $Q$ and confidence level $\delta$, the PAC-Bayes-KL inequality provides:

$$\text{KL}(\hat{R}_Q \| R_Q) \leq \frac{\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta)}{n}$$

where:
- $\hat{R}_Q$ is the empirical risk under $Q$
- $R_Q$ is the true risk under $Q$
- $\text{KL}(Q \| P)$ is the KL divergence between posterior and prior
- $n$ is the number of observations

### Binary KL Divergence

The binary KL divergence is:
$$\text{KL}(p \| q) = p \log\frac{p}{q} + (1-p) \log\frac{1-p}{1-q}$$

### Bound Inversion

We invert the PAC-Bayes inequality using Brent's method to find $R_Q^{\text{upper}}$ such that:
$$\text{KL}(\hat{R}_Q \| R_Q^{\text{upper}}) = \frac{\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta)}{n}$$

### Certified Parameter Interval

The certified interval consists of all parameters with bounded risk below the upper bound:
$$\mathcal{C}_{\delta} = \{\kappa : \tilde{R}(\kappa) \leq R_Q^{\text{upper}}\}$$

where $\tilde{R}(\kappa)$ is the bounded risk for parameter $\kappa$.

## Implementation Details

### Computational Workflow
1. **Data Generation**: Create synthetic observations with known ground truth
2. **Forward Model Caching**: Pre-compute predictions on parameter grid
3. **MCMC Sampling**: Generate posterior samples with convergence monitoring
4. **KL Divergence Estimation**: Histogram-based approximation
5. **Monte Carlo Integration**: Estimate prior expectation term
6. **Bound Computation**: Invert PAC-Bayes inequality
7. **Interval Construction**: Find largest contiguous certified region

### Numerical Considerations
**Grid Resolution**: 181 points on $[1, 10]$ for parameter space discretization

**Histogram KL**: 50 bins for posterior density estimation with numerical stabilization

**Overflow Protection**: Clip extreme log-likelihood ratios for numerical stability

**Convergence Monitoring**: Track Monte Carlo estimates with relative tolerance

### Performance Optimizations

- **Vectorized Operations**: NumPy arrays for efficient computation
- **Cached Predictions**: Avoid redundant PDE solves
- **Interpolation**: Linear interpolation for off-grid parameter values
- **Parallel Processing**: Multi-core MCMC sampling

## Validation Framework

### Experimental Design

**Baseline Validation**: Original dataset with known parameters

**Noise Robustness**: Testing across 5%, 10%, 20% noise levels

**Sensor Sparsity**: Evaluation with 3, 2, and 1 sensor configurations

**Scalability Analysis**: Forward solver performance across grid sizes

### Success Metrics
**Coverage Assessment**: Empirical coverage of true parameter across scenarios

**Efficiency Evaluation**: Width ratio between certified and uncertified intervals

**Reliability Tiers**: Classification based on Effective Sample Size (ESS)
- High: ESS ≥ 1500
- Medium: ESS 800-1500  
- Low: ESS 300-800
- Unreliable: ESS < 300

### Statistical Validation

**Importance Sampling**: Reweight posterior samples for scenario-specific evaluation

**ESS Monitoring**: Track statistical reliability of importance weights

**Convergence Diagnostics**: Ensure robust Monte Carlo estimates

---

This methodology provides a complete framework for applying PAC-Bayes certified bounds to inverse PDE problems, enabling reliable uncertainty quantification with mathematical guarantees.