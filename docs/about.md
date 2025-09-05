---
layout: default
title: About
permalink: /about/
---

# About This Project

## Motivation

Traditional Bayesian approaches to inverse PDE problems provide posterior distributions that represent uncertainty, but these come with **no guarantees**. While a 95% credible interval might seem reliable, it's essentially an educated guess—there's no certificate that the true parameter actually lies within that interval.

### The Problem with Standard Bayesian Inference

Consider estimating thermal conductivity $\kappa$ from temperature measurements:

**Standard approach:** Compute posterior $P(\kappa | \text{data})$  
**Output:** 95% credible interval $[\kappa_{\text{low}}, \kappa_{\text{high}}]$  
**Reality check:** No guarantee the true $\kappa$ is actually inside this interval

### Our Innovation: Certified Uncertainty

We apply **PAC-Bayes bounds** to provide rigorous probabilistic guarantees:

**Input:** Same noisy temperature measurements  
**Output:** Certified interval with provable coverage probability  
**Guarantee:** With probability ≥ 95%, the true $\kappa$ lies in our certified interval

## Novelty vs. Prior Work

### Existing Research
- **Bayesian PDE inversion:** Extensive literature, but no coverage guarantees
- **PAC-Bayes theory:** Well-developed, but mostly applied to machine learning  
- **Uncertainty quantification:** Focus on posterior sampling, not certification

### Our Contribution
- **First application** of PAC-Bayes bounds to PDE inverse problems
- **Rigorous mathematical framework** bridging probability theory and scientific computing
- **Comprehensive validation** under realistic conditions (noise, sparsity, scale)

## Why This Matters

### Scientific Computing
- Trustworthy predictions in engineering applications
- Risk assessment with mathematical guarantees  
- Quality control for inverse problem solutions

### Broader Impact
- Foundation for certified uncertainty in other PDEs
- Bridge between theoretical probability and applied science
- New standard for reliable parameter estimation

## Technical Approach

1. **Forward Problem:** Solve heat equation $u_t = \kappa u_{xx}$ with finite differences
2. **Data Generation:** Add realistic noise and sensor limitations
3. **Bayesian Inference:** Standard MCMC posterior sampling
4. **PAC-Bayes Certification:** Compute guaranteed uncertainty bounds
5. **Validation:** Test coverage, efficiency, and scalability

This project demonstrates that we can move beyond "best guesses" to **mathematically guaranteed uncertainty quantification** in inverse PDE problems.