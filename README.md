# PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation

![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Validation](https://img.shields.io/badge/Validation-PASS-green)
![Coverage](https://img.shields.io/badge/Certified_Coverage-100%25-blue)

## Overview
This repository contains the complete implementation and documentation for applying PAC-Bayes certified bounds to Bayesian inverse problems in partial differential equations.

## Documentation Website

The complete documentation is available as a Jekyll website. To run locally:

```bash
# Install dependencies
bundle install

# Serve the site locally
bundle exec jekyll serve

# View at http://localhost:4000
```

## Project Structure

```
├── _config.yml              # Jekyll configuration
├── _layouts/                # HTML templates
├── _includes/               # Reusable HTML components
├── assets/                  # CSS, JS, and plot files
├── theory/                  # Mathematical theory pages
├── src/                     # Python implementation
├── results/                 # Experimental results
└── docs/                    # Additional documentation
```

## Quick Start
1. **View Documentation**: Visit the [project website](https://github.com/Tani843/PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation)
2. **Run Code**: See implementation in `src/` directory
3. **Reproduce Results**: Follow methodology in documentation

## Key Results

- **100% certified coverage** across 46 validation scenarios
- **Perfect robustness** under noise and sparse sensor conditions
- **Scalable implementation** for practical inverse problems

## Citation

```bibtex
@software{pac_bayes_inverse_pde_2024,
  title={PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation},
  author={Tanisha Gupta},
  year={2024},
  url={https://github.com/Tani843/PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation}
}
```

## License

MIT License - see LICENSE file for details.
