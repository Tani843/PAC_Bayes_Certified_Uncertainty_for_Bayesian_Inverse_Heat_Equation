---
layout: default
title: Home
---

# PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation

## Overview

Most research on Bayesian inverse PDE problems stops at posterior distributions, which are smart guesses but not guaranteed to contain the truth. In this project, we go further: we apply **PAC-Bayes bounds** to provide certified uncertainty guarantees for the inverse heat equation.

Our framework shows provably reliable intervals even under:

✅ **Noisy measurements** (5%, 10%, 20% Gaussian noise)  
✅ **Sparse sensor data** (10 sensors → 3 sensors)  
✅ **Large grid sizes** (20 to 200 spatial points)

## Key Innovation

This is one of the first applications of PAC-Bayes certification to PDE inverse problems, bridging rigorous probability bounds with applied scientific computing.

## Problem Statement

We estimate the unknown thermal conductivity parameter $\kappa$ in the 1D heat equation:

$$u_t = \kappa u_{xx}, \quad x \in [0,L], \, t > 0$$

from noisy, sparse temperature measurements using Bayesian inference with certified uncertainty guarantees.

## Navigation

- **[About](about.html)** - Project motivation and novelty
- **[Methodology](methodology.html)** - Heat equation, Bayesian inference, PAC-Bayes theory  
- **[Results](results.html)** - Experiments, plots, and validation tables
- **[Theory](theory/)** - Mathematical derivations and proofs
- **[Conclusion](conclusion.html)** - Summary and future work
- **[Contact](contact.html)** - Author information

## Quick Start

```bash
git clone https://github.com/tani843/PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation.git
cd PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation
pip install -r requirements.txt
python src/experiments.py
```

## Citation

```bibtex
@misc{pac_bayes_heat_equation,
  title={PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation},
  author={Tanisha Gupta},
  year={2025},
  note={First application of PAC-Bayes bounds to PDE inverse problems},
  url={https://tani843.github.io/PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation}
}
```