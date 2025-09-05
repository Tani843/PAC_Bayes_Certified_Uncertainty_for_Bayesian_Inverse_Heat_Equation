"""
PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation

This package provides certified uncertainty quantification for inverse heat
equation parameter estimation using PAC-Bayes bounds.

Modules:
    heat_solver: Forward heat equation solver with CFL stability
    data_generator: Synthetic data generation with noise and sparse sensors
    bayesian_inference: Standard Bayesian posterior inference with MCMC
    enhanced_mcmc: Enhanced MCMC sampling with emcee ensemble sampler
    pac_bayes: PAC-Bayes certified uncertainty bounds implementation
    experiments: Validation experiments and plotting
    utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Tanisha Gupta"
__email__ = "tanisha@example.com"

from .heat_solver import HeatEquationSolver
from .data_generator import SyntheticDataGenerator
from .bayesian_inference import BayesianInference
from .enhanced_mcmc import EnhancedMCMC
from .pac_bayes import PACBayesUncertainty
from .experiments import ExperimentRunner
from . import utils

__all__ = [
    "HeatEquationSolver",
    "SyntheticDataGenerator",
    "BayesianInference",
    "EnhancedMCMC",
    "PACBayesUncertainty",
    "ExperimentRunner",
    "utils"
]