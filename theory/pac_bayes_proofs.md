---
layout: default
title: PAC-Bayes Proofs
permalink: /theory/proofs/
---

# PAC-Bayes Theory and Proofs

## PAC-Bayes Theorem

**Theorem (McAllester, 1999):** For any prior distribution $P$ over parameters, any $\delta \in (0,1)$, and any loss function $\ell$, with probability at least $1-\delta$ over the sample:

$$\text{KL}(Q \| P) \leq \frac{1}{n}\left(\log \frac{1}{\delta} + \log \mathbb{E}_{P}[e^{n L_S(\theta)}]\right)$$

for any posterior distribution $Q$, where $L_S(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(\theta, z_i)$.

## Application to Inverse Heat Equation

### Setup

- **Parameter**: $\theta = \kappa$ (thermal conductivity)
- **Sample**: $S = \{(x_i, t_i, y_i)\}_{i=1}^n$ (sensor measurements)
- **Loss function**: $\ell(\kappa, (x, t, y)) = \frac{1}{2\sigma^2}(y - u(x,t;\kappa))^2$
- **Prior**: $P(\kappa) = \text{Uniform}[a, b]$
- **Posterior**: $Q(\kappa) = P(\kappa | S)$ (from Bayesian inference)

### Bound Computation

**Step 1:** Compute empirical loss

$$L_S(\kappa) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2\sigma^2}(y_i - u(x_i, t_i; \kappa))^2$$

**Step 2:** Evaluate prior expectation via Monte Carlo

$$\mathbb{E}_{P}[e^{n L_S(\kappa)}] \approx \frac{1}{M}\sum_{j=1}^M e^{n L_S(\kappa_j)}$$

where $\kappa_j \sim P$ are prior samples.

**Step 3:** Compute KL divergence bound

$$\text{KL}(Q \| P) \leq \frac{1}{n}\left(\log \frac{1}{\delta} + \log \widehat{\mathbb{E}}_{P}[e^{n L_S}]\right)$$

**Step 4:** Convert to certified interval

From KL bound, construct prediction set $C_\delta$ such that:

$$\mathbb{P}[\kappa_{\text{true}} \in C_\delta] \geq 1 - \delta$$

## Proof Sketch

### Key Lemma (Change of Measure)

For any measurable function $f$ and distributions $P, Q$:

$$\mathbb{E}_Q[f(\theta)] = \mathbb{E}_P\left[f(\theta) \frac{dQ}{dP}(\theta)\right]$$

### Main Proof Strategy

1. Apply Chernoff bound to $\mathbb{E}_Q[L(\theta)]$ where $L$ is true risk
2. Use change of measure to relate $Q$-expectation to $P$-expectation
3. Invoke concentration inequalities on empirical process
4. Optimize over $\lambda$ in exponential bound

### Detailed Steps

**Step 1:** For any $\lambda > 0$:

$$\mathbb{P}[\mathbb{E}_Q[L(\theta)] \geq \epsilon] \leq e^{-\lambda\epsilon} \mathbb{E}_Q[e^{\lambda L(\theta)}]$$

**Step 2:** Change of measure:

$$\mathbb{E}_Q[e^{\lambda L(\theta)}] = \mathbb{E}_P\left[e^{\lambda L(\theta)} \frac{dQ}{dP}(\theta)\right]$$

**Step 3:** Use data-dependent bound:

$$\mathbb{E}_P\left[e^{\lambda L(\theta)} \frac{dQ}{dP}(\theta)\right] \leq \mathbb{E}_P[e^{\lambda L_S(\theta)}] \cdot e^{\text{KL}(Q \| P)}$$

**Step 4:** Optimize $\lambda = n$ and apply union bound over all $Q$.

## Computational Implementation

### Monte Carlo KL Estimation

For continuous distributions, estimate KL divergence:

$$\text{KL}(Q \| P) \approx \frac{1}{M}\sum_{j=1}^M \log \frac{q(\kappa_j)}{p(\kappa_j)}$$

where $\kappa_j \sim Q$ are posterior samples.

### Certified Interval Construction

From KL bound $\text{KL}(Q \| P) \leq B_\delta$:

1. Construct level sets: $\{\kappa : \log \frac{q(\kappa)}{p(\kappa)} \leq B_\delta\}$
2. Numerical integration: Find interval $[a,b]$ such that:

   $$\int_a^b q(\kappa) d\kappa \geq 1-\delta$$

3. Verification: Check KL constraint is satisfied

## Convergence Properties

### Sample Complexity

**Theorem:** The PAC-Bayes bound is tight with rate $O(1/\sqrt{n})$.

**Proof idea:**
- Upper bound: Direct from concentration inequalities
- Lower bound: Information-theoretic minimax arguments

### Computational Complexity

- **Prior sampling**: $O(M)$ forward solves
- **Posterior sampling**: $O(K)$ MCMC iterations
- **KL estimation**: $O(K \log K)$ density evaluations
- **Total**: $O(M \cdot T_{\text{solve}} + K \cdot T_{\text{MCMC}})$

where $T_{\text{solve}}$ and $T_{\text{MCMC}}$ are forward solve and MCMC step times.