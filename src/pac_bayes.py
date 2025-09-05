"""PAC-Bayes Uncertainty Quantification for Heat Equation Inverse Problems

This module implements PAC-Bayes theory for providing certified uncertainty
bounds on thermal conductivity estimates and temperature predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from .heat_solver import HeatEquationSolver
from .bayesian_inference import BayesianInference


class PACBayesUncertainty:
    """PAC-Bayes uncertainty quantification for inverse heat equation problems."""
    
    def __init__(self, 
                 heat_solver: HeatEquationSolver,
                 bayesian_inference: BayesianInference):
        """Initialize PAC-Bayes uncertainty quantifier.
        
        Args:
            heat_solver: Heat equation solver
            bayesian_inference: Bayesian inference engine
        """
        self.solver = heat_solver
        self.inference = bayesian_inference
        self.nx = heat_solver.nx
        
        # Storage for bounds and certificates
        self.pac_bounds = None
        self.generalization_bounds = None
        self.confidence_certificates = None
    
    def kl_divergence_gaussians(self, 
                               mu1: np.ndarray, 
                               cov1: np.ndarray,
                               mu2: np.ndarray, 
                               cov2: np.ndarray) -> float:
        """Compute KL divergence between two multivariate Gaussians.
        
        Args:
            mu1, cov1: Parameters of first Gaussian
            mu2, cov2: Parameters of second Gaussian
            
        Returns:
            KL divergence D_KL(N(mu1, cov1) || N(mu2, cov2))
        """
        k = len(mu1)
        
        # Compute terms
        cov2_inv = np.linalg.inv(cov2)
        
        term1 = np.trace(cov2_inv @ cov1)
        term2 = (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1)
        term3 = np.linalg.slogdet(cov2)[1] - np.linalg.slogdet(cov1)[1]
        
        return 0.5 * (term1 + term2 - k + term3)
    
    def empirical_risk(self,
                      theta: np.ndarray,
                      observations: np.ndarray,
                      sensor_locations: np.ndarray,
                      observation_times: np.ndarray,
                      noise_variance: float,
                      initial_condition: Union[float, np.ndarray, Callable],
                      boundary_conditions: Tuple[float, float],
                      source_term: Optional[Callable] = None) -> float:
        """Compute empirical risk (negative log likelihood).
        
        Args:
            theta: Log thermal conductivity parameters
            observations: Observed temperatures
            sensor_locations: Sensor locations
            observation_times: Observation times
            noise_variance: Observation noise variance
            initial_condition: Initial temperature distribution
            boundary_conditions: Dirichlet boundary conditions
            source_term: Heat source function
            
        Returns:
            Empirical risk
        """
        log_like = self.inference.log_likelihood(
            theta, observations, sensor_locations, observation_times,
            noise_variance, initial_condition, boundary_conditions, source_term
        )
        return -log_like
    
    def pac_bayes_bound(self,
                       posterior_samples: np.ndarray,
                       observations: np.ndarray,
                       sensor_locations: np.ndarray,
                       observation_times: np.ndarray,
                       noise_variance: float,
                       initial_condition: Union[float, np.ndarray, Callable],
                       boundary_conditions: Tuple[float, float],
                       delta: float = 0.05,
                       source_term: Optional[Callable] = None) -> Dict:
        """Compute PAC-Bayes generalization bound.
        
        This provides a high-probability bound on the true risk given
        the empirical risk and KL divergence to prior.
        
        Args:
            posterior_samples: Samples from approximate posterior
            observations: Training observations
            sensor_locations: Sensor locations
            observation_times: Observation times
            noise_variance: Observation noise variance  
            initial_condition: Initial temperature distribution
            boundary_conditions: Dirichlet boundary conditions
            delta: Confidence parameter (bound holds with prob. 1-delta)
            source_term: Heat source function
            
        Returns:
            Dictionary with PAC-Bayes bounds and components
        """
        n_samples = len(posterior_samples)
        n_obs = observations.size
        
        # Compute empirical risk for each sample
        empirical_risks = np.zeros(n_samples)
        for i, theta in enumerate(posterior_samples):
            empirical_risks[i] = self.empirical_risk(
                theta, observations, sensor_locations, observation_times,
                noise_variance, initial_condition, boundary_conditions, source_term
            )
        
        # Average empirical risk
        avg_empirical_risk = np.mean(empirical_risks)
        
        # Estimate KL divergence between posterior and prior
        posterior_mean = np.mean(posterior_samples, axis=0)
        posterior_cov = np.cov(posterior_samples.T)
        
        # Add regularization for numerical stability
        posterior_cov += 1e-6 * np.eye(self.nx)
        
        kl_div = self.kl_divergence_gaussians(
            posterior_mean, posterior_cov,
            self.inference.prior_mean, self.inference.prior_cov
        )
        
        # PAC-Bayes bound
        pac_bound = avg_empirical_risk + np.sqrt(
            (kl_div + np.log(2 * np.sqrt(n_obs) / delta)) / (2 * n_obs - 1)
        )
        
        # Tighter bound using empirical Bernstein
        variance_term = np.var(empirical_risks)
        bernstein_bound = avg_empirical_risk + np.sqrt(
            2 * variance_term * (kl_div + np.log(2 / delta)) / n_obs
        ) + (kl_div + np.log(2 / delta)) / n_obs
        
        self.pac_bounds = {
            'empirical_risk': avg_empirical_risk,
            'kl_divergence': kl_div,
            'pac_bound': pac_bound,
            'bernstein_bound': bernstein_bound,
            'confidence': 1 - delta,
            'n_observations': n_obs,
            'risk_variance': variance_term
        }
        
        return self.pac_bounds
    
    def prediction_bounds(self,
                         posterior_samples: np.ndarray,
                         initial_condition: Union[float, np.ndarray, Callable],
                         boundary_conditions: Tuple[float, float],
                         prediction_times: np.ndarray,
                         prediction_locations: np.ndarray,
                         confidence: float = 0.95,
                         source_term: Optional[Callable] = None) -> Dict:
        """Compute prediction confidence bounds using posterior samples.
        
        Args:
            posterior_samples: Samples from posterior distribution
            initial_condition: Initial temperature distribution
            boundary_conditions: Dirichlet boundary conditions
            prediction_times: Times for prediction
            prediction_locations: Locations for prediction
            confidence: Confidence level for bounds
            source_term: Heat source function
            
        Returns:
            Dictionary with prediction bounds
        """
        n_samples = len(posterior_samples)
        n_times = len(prediction_times)
        n_locations = len(prediction_locations)
        
        # Storage for predictions
        predictions = np.zeros((n_samples, n_times, n_locations))
        
        # Generate predictions for each sample
        for i, theta in enumerate(posterior_samples):
            k = np.exp(theta)
            
            try:
                solution, times = self.solver.solve(
                    thermal_conductivity=k,
                    initial_condition=initial_condition,
                    boundary_conditions=boundary_conditions,
                    source_term=source_term,
                    t_final=prediction_times[-1],
                    observation_times=prediction_times
                )
                
                predictions[i] = self.solver.compute_observations(
                    solution, prediction_locations, noise_level=0.0
                )
                
            except Exception:
                predictions[i] = np.nan
        
        # Remove failed predictions
        valid_mask = ~np.isnan(predictions).any(axis=(1, 2))
        predictions = predictions[valid_mask]
        
        # Compute bounds
        alpha = 1 - confidence
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        pred_lower = np.percentile(predictions, lower_percentile, axis=0)
        pred_upper = np.percentile(predictions, upper_percentile, axis=0)
        
        # Compute coverage width
        coverage_width = pred_upper - pred_lower
        
        return {
            'predictions': predictions,
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'prediction_lower': pred_lower,
            'prediction_upper': pred_upper,
            'coverage_width': coverage_width,
            'confidence_level': confidence,
            'prediction_times': prediction_times,
            'prediction_locations': prediction_locations,
            'n_valid_samples': len(predictions)
        }
    
    def cross_validation_bounds(self,
                               observations: np.ndarray,
                               sensor_locations: np.ndarray,
                               observation_times: np.ndarray,
                               noise_variance: float,
                               initial_condition: Union[float, np.ndarray, Callable],
                               boundary_conditions: Tuple[float, float],
                               k_folds: int = 5,
                               n_mcmc_samples: int = 1000,
                               source_term: Optional[Callable] = None) -> Dict:
        """Compute cross-validation based uncertainty bounds.
        
        Args:
            observations: Observed temperatures
            sensor_locations: Sensor locations
            observation_times: Observation times
            noise_variance: Observation noise variance
            initial_condition: Initial temperature distribution
            boundary_conditions: Dirichlet boundary conditions
            k_folds: Number of CV folds
            n_mcmc_samples: MCMC samples per fold
            source_term: Heat source function
            
        Returns:
            Dictionary with CV bounds and diagnostics
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Storage for fold results
        fold_risks = []
        fold_samples = []
        
        # Flatten observations for CV splitting
        obs_flat = observations.reshape(-1)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(obs_flat)):
            print(f"Processing fold {fold_idx + 1}/{k_folds}")
            
            # Split data (simple temporal split for now)
            n_times = len(observation_times)
            split_time = int(0.8 * n_times)
            
            train_times = observation_times[:split_time]
            test_times = observation_times[split_time:]
            train_obs = observations[:split_time]
            test_obs = observations[split_time:]
            
            # Run MCMC on training data
            mcmc_result = self.inference.mcmc_sample(
                train_obs, sensor_locations, train_times, noise_variance,
                initial_condition, boundary_conditions, 
                n_samples=n_mcmc_samples, n_burn=200
            )
            
            # Compute test risk
            test_risks = []
            for theta in mcmc_result['samples']:
                test_risk = self.empirical_risk(
                    theta, test_obs, sensor_locations, test_times,
                    noise_variance, initial_condition, boundary_conditions, source_term
                )
                test_risks.append(test_risk)
            
            fold_risks.extend(test_risks)
            fold_samples.append(mcmc_result['samples'])
        
        # Combine results
        all_samples = np.vstack(fold_samples)
        
        # Compute statistics
        cv_risk_mean = np.mean(fold_risks)
        cv_risk_std = np.std(fold_risks)
        cv_risk_upper = cv_risk_mean + 1.96 * cv_risk_std  # 95% confidence
        
        return {
            'cv_risk_mean': cv_risk_mean,
            'cv_risk_std': cv_risk_std,
            'cv_risk_upper': cv_risk_upper,
            'fold_risks': fold_risks,
            'combined_samples': all_samples,
            'n_folds': k_folds
        }
    
    def concentration_bounds(self,
                           posterior_samples: np.ndarray,
                           true_theta: Optional[np.ndarray] = None) -> Dict:
        """Compute concentration bounds for posterior distribution.
        
        Args:
            posterior_samples: Samples from posterior
            true_theta: True parameter values (if known)
            
        Returns:
            Dictionary with concentration measures
        """
        n_samples, n_params = posterior_samples.shape
        
        # Posterior statistics
        posterior_mean = np.mean(posterior_samples, axis=0)
        posterior_cov = np.cov(posterior_samples.T)
        
        # Distance to prior
        prior_mean = self.inference.prior_mean
        dist_to_prior = np.linalg.norm(posterior_mean - prior_mean)
        
        # Posterior concentration (trace of covariance)
        concentration = np.trace(posterior_cov)
        
        # Effective dimensionality (participation ratio)
        eigenvals = np.linalg.eigvals(posterior_cov)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
        participation_ratio = (np.sum(eigenvals)**2) / np.sum(eigenvals**2)
        
        results = {
            'posterior_concentration': concentration,
            'distance_to_prior': dist_to_prior,
            'effective_dimensionality': participation_ratio,
            'posterior_determinant': np.linalg.det(posterior_cov),
            'condition_number': np.linalg.cond(posterior_cov)
        }
        
        # If true parameters are known, compute additional metrics
        if true_theta is not None:
            # Distance to truth
            dist_to_truth = np.linalg.norm(posterior_mean - true_theta)
            
            # Coverage probability (approximate)
            diffs = posterior_samples - true_theta[np.newaxis, :]
            mahalanobis_dists = np.array([
                diff.T @ np.linalg.inv(posterior_cov) @ diff 
                for diff in diffs
            ])
            
            # Chi-square test for coverage
            chi2_stat = np.mean(mahalanobis_dists)
            expected_chi2 = n_params  # Expected value for chi-square(n_params)
            
            results.update({
                'distance_to_truth': dist_to_truth,
                'mahalanobis_distances': mahalanobis_dists,
                'chi2_statistic': chi2_stat,
                'expected_chi2': expected_chi2,
                'coverage_calibration': chi2_stat / expected_chi2
            })
        
        return results
    
    def certified_bounds(self,
                        posterior_samples: np.ndarray,
                        observations: np.ndarray,
                        sensor_locations: np.ndarray,
                        observation_times: np.ndarray,
                        noise_variance: float,
                        initial_condition: Union[float, np.ndarray, Callable],
                        boundary_conditions: Tuple[float, float],
                        delta: float = 0.05,
                        source_term: Optional[Callable] = None) -> Dict:
        """Compute comprehensive certified uncertainty bounds.
        
        Args:
            posterior_samples: Posterior samples
            observations: Training observations
            sensor_locations: Sensor locations
            observation_times: Observation times
            noise_variance: Observation noise variance
            initial_condition: Initial temperature distribution
            boundary_conditions: Dirichlet boundary conditions
            delta: Confidence parameter
            source_term: Heat source function
            
        Returns:
            Comprehensive uncertainty certification
        """
        print("Computing PAC-Bayes bounds...")
        pac_bounds = self.pac_bayes_bound(
            posterior_samples, observations, sensor_locations, observation_times,
            noise_variance, initial_condition, boundary_conditions, delta, source_term
        )
        
        print("Computing concentration bounds...")
        concentration_bounds = self.concentration_bounds(posterior_samples)
        
        print("Computing prediction bounds...")
        # Use same time/location grid for predictions
        prediction_bounds = self.prediction_bounds(
            posterior_samples[:500],  # Subsample for efficiency
            initial_condition, boundary_conditions,
            observation_times, sensor_locations,
            confidence=1-delta, source_term=source_term
        )
        
        # Combine all bounds
        self.confidence_certificates = {
            'pac_bayes': pac_bounds,
            'concentration': concentration_bounds,
            'prediction': prediction_bounds,
            'confidence_level': 1 - delta,
            'certification_timestamp': np.datetime64('now')
        }
        
        return self.confidence_certificates
    
    def validate_bounds(self,
                       test_observations: np.ndarray,
                       test_sensor_locations: np.ndarray,
                       test_observation_times: np.ndarray,
                       noise_variance: float,
                       initial_condition: Union[float, np.ndarray, Callable],
                       boundary_conditions: Tuple[float, float],
                       source_term: Optional[Callable] = None) -> Dict:
        """Validate computed bounds on test data.
        
        Args:
            test_observations: Test observations
            test_sensor_locations: Test sensor locations
            test_observation_times: Test observation times
            noise_variance: Test noise variance
            initial_condition: Initial condition
            boundary_conditions: Boundary conditions
            source_term: Source term
            
        Returns:
            Validation results
        """
        if self.confidence_certificates is None:
            raise ValueError("No bounds computed. Run certified_bounds first.")
        
        # Get prediction bounds
        pred_bounds = self.confidence_certificates['prediction']
        predicted_mean = pred_bounds['prediction_mean']
        predicted_lower = pred_bounds['prediction_lower']
        predicted_upper = pred_bounds['prediction_upper']
        
        # Interpolate to test locations/times if different
        # (For simplicity, assuming same grid here)
        
        # Check coverage
        coverage_mask = ((test_observations >= predicted_lower) & 
                        (test_observations <= predicted_upper))
        empirical_coverage = np.mean(coverage_mask)
        
        # Compute prediction errors
        prediction_errors = np.abs(test_observations - predicted_mean)
        mean_absolute_error = np.mean(prediction_errors)
        rmse = np.sqrt(np.mean(prediction_errors**2))
        
        # Width of confidence intervals
        interval_widths = predicted_upper - predicted_lower
        mean_interval_width = np.mean(interval_widths)
        
        validation_results = {
            'empirical_coverage': empirical_coverage,
            'expected_coverage': pred_bounds['confidence_level'],
            'coverage_gap': abs(empirical_coverage - pred_bounds['confidence_level']),
            'mean_absolute_error': mean_absolute_error,
            'rmse': rmse,
            'mean_interval_width': mean_interval_width,
            'prediction_errors': prediction_errors,
            'interval_widths': interval_widths,
            'coverage_mask': coverage_mask
        }
        
        return validation_results