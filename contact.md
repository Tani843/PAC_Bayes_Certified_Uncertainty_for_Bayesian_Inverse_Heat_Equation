---
layout: default
title: "Contact"
permalink: /contact/
---

# Contact & References

## Author Information

### Primary Contact
**Your Name**  
*Affiliation*  
Email: your.email@example.com  
GitHub: [yourusername](https://github.com/yourusername)  
ORCID: 0000-0000-0000-0000

### Project Repository
**PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation**  
GitHub: [https://github.com/yourusername/pac-bayes-certified-uncertainty](https://github.com/yourusername/pac-bayes-certified-uncertainty)  
License: MIT License  
DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

## Code and Data Availability

### Implementation
- **Language**: Python 3.8+
- **Dependencies**: NumPy, SciPy, Matplotlib, emcee, pathlib
- **Documentation**: Complete API documentation included
- **Tests**: Comprehensive unit and integration tests
- **Examples**: Jupyter notebooks with worked examples

### Reproducibility
All results in this documentation are fully reproducible:
- **Fixed random seeds**: Ensures deterministic results
- **Version control**: Complete git history available
- **Docker container**: Containerized environment for exact reproduction
- **Continuous integration**: Automated testing and validation

### Data Sets
- **Synthetic data**: Generated using documented procedures
- **Validation results**: Complete Phase 6 experimental outputs
- **Figures**: High-resolution plots in PNG and SVG formats
- **Tables**: Machine-readable CSV and JSON formats

## Citation

If you use this work in your research, please cite:

```bibtex
@software{pac_bayes_inverse_pde_2024,
  title={PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pac-bayes-certified-uncertainty},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## References

### PAC-Bayes Theory
1. **McAllester, D. A.** (1999). PAC-Bayesian model averaging. *Proceedings of the 12th Annual Conference on Computational Learning Theory*, 164-170.

2. **Seeger, M.** (2002). PAC-Bayesian generalization error bounds for Gaussian process classification. *Journal of Machine Learning Research*, 3, 233-269.

3. **Catoni, O.** (2007). *PAC-Bayesian supervised classification: The thermodynamics of statistical learning*. Institute of Mathematical Statistics.

4. **Alquier, P.** (2021). User-friendly introduction to PAC-Bayes bounds. *Foundations and Trends in Machine Learning*, 14(3), 174-303.

5. **Guedj, B.** (2019). A primer on PAC-Bayesian learning. *arXiv preprint arXiv:1901.05353*.

### Inverse Problems and PDEs
6. **Kaipio, J., & Somersalo, E.** (2005). *Statistical and computational inverse problems*. Springer.

7. **Stuart, A. M.** (2010). Inverse problems: a Bayesian perspective. *Acta Numerica*, 19, 451-559.

8. **Dashti, M., & Stuart, A. M.** (2017). The Bayesian approach to inverse problems. *Handbook of Uncertainty Quantification*, 311-428.

9. **Iglesias, M. A., Law, K. J., & Stuart, A. M.** (2013). Ensemble Kalman methods for inverse problems. *Inverse Problems*, 29(4), 045001.

### Uncertainty Quantification
10. **Ghanem, R. G., Higdon, D., & Osman, H.** (Eds.). (2017). *Handbook of uncertainty quantification*. Springer.

11. **Le Maître, O., & Knio, O. M.** (2010). *Spectral methods for uncertainty quantification*. Springer.

12. **Smith, R. C.** (2013). *Uncertainty quantification: theory, implementation, and applications*. SIAM.

### Computational Methods
13. **Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J.** (2013). emcee: the MCMC hammer. *Publications of the Astronomical Society of the Pacific*, 125(925), 306.

14. **Gelman, A., & Rubin, D. B.** (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457-472.

15. **Brooks, S., Gelman, A., Jones, G., & Meng, X. L.** (Eds.). (2011). *Handbook of Markov chain Monte Carlo*. CRC Press.

### Related Applications
16. **Bilionis, I., & Zabaras, N.** (2014). Solution of inverse problems with limited forward solver evaluations. *SIAM Journal on Scientific Computing*, 36(1), A44-A71.

17. **Cui, T., Law, K. J., & Marzouk, Y. M.** (2016). Dimension-independent likelihood-informed MCMC. *Journal of Computational Physics*, 304, 109-137.

18. **Petra, N., Martin, J., Stadler, G., & Ghattas, O.** (2014). A computational framework for infinite-dimensional Bayesian inverse problems. *Journal of Computational Physics*, 252, 518-537.

## Acknowledgments

### Theoretical Foundations
We acknowledge the foundational work of the PAC-Bayes community, particularly David McAllester's original framework and Olivier Catoni's comprehensive treatment of PAC-Bayesian learning theory.

### Computational Infrastructure
This work was enabled by:
- **Open source scientific computing**: NumPy, SciPy, Matplotlib ecosystems
- **MCMC sampling**: The emcee package by Foreman-Mackey et al.
- **Version control**: Git and GitHub for reproducible research
- **Documentation**: Jekyll and GitHub Pages for open documentation

### Community Support
Thanks to the broader scientific computing and uncertainty quantification communities for:
- Code review and feedback
- Theoretical discussions and insights
- Computational best practices
- Open access publication models

## License and Usage

### Software License
This project is released under the **MIT License**:

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Documentation License
Documentation is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

### Usage Guidelines
- **Academic use**: Citation required as specified above
- **Commercial use**: Permitted under MIT license terms
- **Modification**: Encouraged with proper attribution
- **Distribution**: Freely allowed with license preservation

## Contributing

### How to Contribute
We welcome contributions to improve this work:
1. **Issues**: Report bugs or suggest enhancements via GitHub issues
2. **Pull requests**: Submit improvements via GitHub pull requests
3. **Documentation**: Help improve documentation and examples
4. **Testing**: Expand test coverage and validation scenarios

### Development Guidelines
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality
- Update documentation for API changes

### Code of Conduct
This project adheres to a code of conduct promoting:
- Respectful and inclusive communication
- Constructive feedback and collaboration
- Recognition of diverse perspectives and contributions

## Future Work and Collaboration

### Research Opportunities
Interested in extending this work? Consider:
- Multi-dimensional PDE applications
- Alternative PAC-Bayes bounds
- Computational efficiency improvements
- Real-world validation studies

### Collaboration Interests
We welcome collaboration on:
- Theoretical developments in certified uncertainty
- Practical applications in engineering and science
- Computational advances and optimization
- Educational materials and outreach

### Contact for Collaboration
Email: your.email@example.com  
Subject line: "[PAC-Bayes] Collaboration Inquiry"

---

*This work represents a collaborative effort to advance trustworthy uncertainty quantification in computational science. We encourage its use, extension, and improvement by the broader research community.*