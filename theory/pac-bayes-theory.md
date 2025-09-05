---
layout: default
title: "PAC-Bayes Theory"
permalink: /theory/pac-bayes-theory/
---

# PAC-Bayes Theory

This page provides the complete theoretical foundation for PAC-Bayes bounds, which form the mathematical basis for our certified uncertainty quantification approach.

## PAC-Bayes Inequalities

### Basic PAC-Bayes Bound

**Theorem 1 (McAllester, 1999)**: For any prior $P$, any $\delta \in (0,1)$, with probability at least $1-\delta$ over the draw of the sample, for all posteriors $Q$:

$$R(Q) \leq \hat{R}(Q) + \sqrt{\frac{\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta)}{2n}}$$

where:
- $R(Q) = \mathbb{E}_{\theta \sim Q}[R(\theta)]$ is the expected true risk
- $\hat{R}(Q) = \mathbb{E}_{\theta \sim Q}[\hat{R}(\theta)]$ is the expected empirical risk

### PAC-Bayes-KL Bound
**Theorem 2 (Seeger, 2002)**: Under the same conditions, the tighter KL-based bound holds:

$$\text{KL}(\hat{R}(Q) \| R(Q)) \leq \frac{\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta)}{n}$$

This bound is generally tighter than the square-root bound and forms the basis of our implementation.

## Proof Sketch of PAC-Bayes-KL

The proof relies on the following key steps:

### Step 1: Change of Measure
Apply Fubini's theorem to exchange expectation and probability:
$$\mathbb{E}_{S \sim \mathcal{D}^n} \mathbb{E}_{\theta \sim Q}[f(\theta, S)] = \mathbb{E}_{\theta \sim Q} \mathbb{E}_{S \sim \mathcal{D}^n}[f(\theta, S)]$$

### Step 2: Exponential Moment Bound
For fixed $\theta$, apply Hoeffding's inequality for bounded losses:
$$\mathbb{E}_{S \sim \mathcal{D}^n}[\exp(\lambda n(\hat{R}(\theta) - R(\theta)))] \leq \exp\left(\frac{\lambda^2 n}{8}\right)$$

### Step 3: Integration over Posterior
Integrate the pointwise bound over the posterior:
$$\mathbb{E}_{\theta \sim Q} \mathbb{E}_{S \sim \mathcal{D}^n}[\exp(\lambda n(\hat{R}(\theta) - R(\theta)))] \leq \exp\left(\frac{\lambda^2 n}{8}\right)$$

### Step 4: Change of Measure Again
Apply change of measure to relate posterior $Q$ to prior $P$:
$$\mathbb{E}_{S \sim \mathcal{D}^n} \mathbb{E}_{\theta \sim P}\left[\frac{dQ}{dP}(\theta) \exp(\lambda n(\hat{R}(\theta) - R(\theta)))\right] \leq \exp\left(\frac{\lambda^2 n}{8}\right)$$

### Step 5: Optimization
Optimize over $\lambda$ to obtain the tightest bound, yielding the KL form.

## Binary KL Divergence

The KL divergence between Bernoulli distributions with parameters $p$ and $q$ is:

$$\text{KL}(p \| q) = p \ln\frac{p}{q} + (1-p) \ln\frac{1-p}{1-q}$$

**Properties**:
- Convex in both arguments
- $\text{KL}(p \| q) \geq 0$ with equality iff $p = q$
- Continuous for $p, q \in (0, 1)$

### Inversion Problem

Given empirical risk $\hat{r}$ and bound $B$, find $r^*$ such that:
$$\text{KL}(\hat{r} \| r^*) = B$$

This is solved numerically using root-finding algorithms like Brent's method.

## Application to Inverse Problems

### Bounded Loss Design
For unbounded losses like squared error, we construct bounded versions:
$$\tilde{\ell}(\theta, z) = \min\left(\frac{\ell(\theta, z)}{M}, 1\right)$$

where $M$ is a problem-specific bound.

### Posterior Construction

The posterior $Q$ comes from Bayesian inference:
$$Q(\theta) \propto P(\theta) \exp\left(-\sum_{i=1}^n \ell(\theta, z_i)\right)$$

### KL Divergence Estimation

For continuous posteriors, estimate $\text{KL}(Q \| P)$ using:
- Histogram-based density approximation
- Monte Carlo sampling
- Importance sampling techniques

## Computational Considerations

### Numerical Stability

- Clip extreme log-likelihood ratios
- Use log-space arithmetic for large exponentials
- Regularize density estimates near boundaries

### Convergence Monitoring

- Track Monte Carlo estimates with confidence intervals
- Monitor relative changes in bound estimates
- Validate using multiple random seeds

## Extensions and Variants

### Other PAC-Bayes Bounds

- **Catoni bound**: Alternative to KL-based bound
- **Empirical Bernstein**: Data-dependent bounds
- **Time-uniform**: Valid at all stopping times

### Recent Developments
- **Fractional posteriors**: Tempered likelihoods
- **Split-kl**: Refined bounds using data splitting
- **Non-vacuous bounds**: Practical bounds for neural networks

## References

1. McAllester, D. (1999). PAC-Bayesian model averaging. COLT.
2. Seeger, M. (2002). PAC-Bayesian generalization error bounds for Gaussian process classification. JMLR.
3. Catoni, O. (2007). PAC-Bayesian supervised classification: The thermodynamics of statistical learning.
4. Alquier, P. (2021). User-friendly introduction to PAC-Bayes bounds.

---

This theoretical foundation enables certified uncertainty quantification with mathematical guarantees, extending beyond traditional statistical approaches.