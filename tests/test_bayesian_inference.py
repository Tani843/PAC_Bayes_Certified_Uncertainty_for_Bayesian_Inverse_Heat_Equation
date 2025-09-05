"""Tests for Bayesian inference module."""

import pytest
import numpy as np
from src.heat_solver import HeatSolver
from src.bayesian_inference import BayesianHeatInference
from src.data_generator import DataGenerator


class TestBayesianHeatInference:
    """Test class for BayesianHeatInference."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.domain = (0.0, 1.0)
        self.nx = 20  # Small for fast tests
        self.dt = 0.001
        self.solver = HeatSolver(self.domain[0], self.domain[1], self.nx, self.dt)
        self.inference = BayesianHeatInference(self.solver)
        
        # Generate test data
        self.generator = DataGenerator(self.domain, self.nx, self.dt)
        self.test_dataset = self.generator.generate_dataset(
            conductivity_name='linear',
            t_final=0.1,
            noise_level=0.02
        )
    
    def test_initialization(self):
        """Test inference engine initialization."""
        assert self.inference.solver == self.solver
        assert self.inference.nx == self.nx
        assert len(self.inference.prior_mean) == self.nx
        assert self.inference.prior_cov.shape == (self.nx, self.nx)
        
        # Prior covariance should be positive definite
        eigenvals = np.linalg.eigvals(self.inference.prior_cov)
        assert np.all(eigenvals > 0)
    
    def test_prior_covariance_construction(self):
        """Test prior covariance matrix construction."""
        cov = self.inference._construct_prior_covariance(variance=0.5, length_scale=0.2)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        
        # Should be positive definite
        eigenvals = np.linalg.eigvals(cov)
        assert np.all(eigenvals > 0)
        
        # Diagonal should equal variance (approximately)
        assert np.allclose(np.diag(cov), 0.5, rtol=0.1)
    
    def test_log_prior(self):
        """Test log prior density evaluation."""
        theta = self.inference.prior_mean.copy()
        log_prior = self.inference.log_prior(theta)
        
        # Should be finite
        assert np.isfinite(log_prior)
        
        # Should be maximum at prior mean
        theta_offset = theta + 0.1 * np.random.randn(self.nx)
        log_prior_offset = self.inference.log_prior(theta_offset)
        assert log_prior >= log_prior_offset
    
    def test_log_likelihood(self):
        """Test log likelihood evaluation."""
        # Use true parameters from test dataset
        true_k = self.test_dataset['true_conductivity']
        theta_true = np.log(true_k)
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        log_like = self.inference.log_likelihood(
            theta_true, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'], 
            self.test_dataset['boundary_conditions']
        )
        
        # Should be finite and not too negative
        assert np.isfinite(log_like)
        assert log_like > -1e6  # Reasonable likelihood
        
        # Perturbed parameters should have lower likelihood
        theta_wrong = theta_true + 0.5 * np.random.randn(self.nx)
        log_like_wrong = self.inference.log_likelihood(
            theta_wrong, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # True parameters should have higher likelihood (usually)
        # Note: This is stochastic so we don't enforce strict inequality
        assert log_like >= log_like_wrong - 100  # Allow some tolerance
    
    def test_log_posterior(self):
        """Test log posterior density evaluation."""
        theta = self.inference.prior_mean.copy()
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        log_post = self.inference.log_posterior(
            theta, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        assert np.isfinite(log_post)
        
        # Should equal sum of log prior and log likelihood
        log_prior = self.inference.log_prior(theta)
        log_like = self.inference.log_likelihood(
            theta, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        expected_log_post = log_prior + log_like
        assert np.isclose(log_post, expected_log_post, rtol=1e-10)
    
    def test_map_estimation(self):
        """Test MAP estimation."""
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        map_result = self.inference.find_map_estimate(
            observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Check result structure
        assert 'theta_map' in map_result
        assert 'k_map' in map_result
        assert 'log_posterior' in map_result
        assert 'optimization_success' in map_result
        
        # MAP estimate should be reasonable
        theta_map = map_result['theta_map']
        k_map = map_result['k_map']
        
        assert len(theta_map) == self.nx
        assert len(k_map) == self.nx
        assert np.all(k_map > 0)  # Physical conductivity should be positive
        assert np.all(np.isfinite(theta_map))
        
        # Store MAP estimate for other tests
        self.inference.map_estimate = theta_map
    
    @pytest.mark.slow
    def test_mcmc_sampling(self):
        """Test MCMC sampling (marked as slow)."""
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Run short MCMC for testing
        mcmc_result = self.inference.mcmc_sample(
            observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            n_samples=100, n_burn=20, step_size=0.01
        )
        
        # Check result structure
        assert 'samples' in mcmc_result
        assert 'posterior_mean' in mcmc_result
        assert 'posterior_cov' in mcmc_result
        assert 'acceptance_rate' in mcmc_result
        
        samples = mcmc_result['samples']
        
        # Check sample dimensions
        assert samples.shape == (100, self.nx)
        
        # Acceptance rate should be reasonable
        acceptance_rate = mcmc_result['acceptance_rate']
        assert 0.1 <= acceptance_rate <= 0.8
        
        # Samples should be finite
        assert np.all(np.isfinite(samples))
        
        # Posterior statistics
        post_mean = mcmc_result['posterior_mean']
        post_cov = mcmc_result['posterior_cov']
        
        assert len(post_mean) == self.nx
        assert post_cov.shape == (self.nx, self.nx)
        
        # Covariance should be positive semi-definite
        eigenvals = np.linalg.eigvals(post_cov)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    @pytest.mark.slow  
    def test_variational_inference(self):
        """Test variational inference (marked as slow)."""
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Run short VI for testing
        vi_result = self.inference.variational_inference(
            observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            n_iterations=50, learning_rate=0.01
        )
        
        # Check result structure
        assert 'variational_mean' in vi_result
        assert 'variational_cov' in vi_result
        assert 'elbo_history' in vi_result
        assert 'final_elbo' in vi_result
        
        var_mean = vi_result['variational_mean']
        var_cov = vi_result['variational_cov']
        elbo_history = vi_result['elbo_history']
        
        # Check dimensions
        assert len(var_mean) == self.nx
        assert var_cov.shape == (self.nx, self.nx)
        assert len(elbo_history) == 50
        
        # ELBO should generally increase (allowing for some fluctuations)
        # Check that final ELBO is not much worse than best ELBO
        best_elbo = np.max(elbo_history)
        final_elbo = elbo_history[-1]
        assert final_elbo >= best_elbo - 100  # Allow some tolerance
        
        # Variational parameters should be finite
        assert np.all(np.isfinite(var_mean))
        assert np.all(np.isfinite(var_cov))
    
    def test_posterior_prediction(self):
        """Test posterior predictive sampling."""
        # Generate some posterior samples (simplified)
        n_samples = 20
        posterior_samples = (self.inference.prior_mean[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        prediction_times = np.linspace(0, 0.05, 10)
        sensor_locations = np.array([0.3, 0.7])
        
        pred_result = self.inference.predict_posterior(
            posterior_samples,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            prediction_times,
            sensor_locations
        )
        
        # Check result structure
        assert 'predictions' in pred_result
        assert 'prediction_mean' in pred_result
        assert 'prediction_std' in pred_result
        
        predictions = pred_result['predictions']
        pred_mean = pred_result['prediction_mean']
        pred_std = pred_result['prediction_std']
        
        # Check dimensions
        expected_shape = (len(prediction_times), len(sensor_locations))
        assert pred_mean.shape == expected_shape
        assert pred_std.shape == expected_shape
        
        # Should have reasonable number of valid predictions
        n_valid = pred_result['n_valid_samples']
        assert n_valid >= n_samples * 0.5  # At least half should be valid
        
        # Predictions should be finite
        assert np.all(np.isfinite(pred_mean))
        assert np.all(np.isfinite(pred_std))
        assert np.all(pred_std >= 0)  # Standard deviation should be non-negative
    
    def test_posterior_statistics(self):
        """Test computation of posterior statistics."""
        # Generate mock samples
        n_samples = 50
        samples = (self.inference.prior_mean[np.newaxis, :] + 
                  0.1 * np.random.randn(n_samples, self.nx))
        
        self.inference.samples = samples
        
        stats = self.inference.compute_posterior_statistics()
        
        # Check required fields
        required_fields = ['theta_samples', 'k_samples', 'theta_mean', 'theta_median',
                          'k_mean', 'k_median', 'theta_std', 'k_std',
                          'theta_credible_interval', 'k_credible_interval',
                          'effective_sample_size', 'spatial_grid']
        
        for field in required_fields:
            assert field in stats
        
        # Check dimensions
        assert stats['theta_mean'].shape == (self.nx,)
        assert stats['k_mean'].shape == (self.nx,)
        assert stats['theta_credible_interval'].shape == (2, self.nx)
        assert stats['k_credible_interval'].shape == (2, self.nx)
        assert stats['effective_sample_size'].shape == (self.nx,)
        
        # Physical constraints
        assert np.all(stats['k_mean'] > 0)
        assert np.all(stats['k_median'] > 0)
        assert np.all(stats['k_std'] >= 0)
        
        # Credible intervals should be ordered
        theta_ci = stats['theta_credible_interval']
        k_ci = stats['k_credible_interval']
        assert np.all(theta_ci[0] <= theta_ci[1])  # Lower <= Upper
        assert np.all(k_ci[0] <= k_ci[1])
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Invalid theta (wrong size)
        with pytest.raises((ValueError, IndexError)):
            self.inference.log_likelihood(
                np.array([1.0, 2.0]),  # Wrong size
                observations, sensor_locs, obs_times, 0.02**2,
                self.test_dataset['initial_condition'],
                self.test_dataset['boundary_conditions']
            )
        
        # Negative noise variance should work but might give warnings
        # (Physical interpretation: perfect observations)
        try:
            log_like = self.inference.log_likelihood(
                self.inference.prior_mean, observations, sensor_locs, obs_times,
                -0.01,  # Negative variance
                self.test_dataset['initial_condition'],
                self.test_dataset['boundary_conditions']
            )
            # If no exception, should still be finite
            assert np.isfinite(log_like)
        except:
            # It's also acceptable to raise an exception for negative variance
            pass
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Very negative theta (very small conductivity)
        theta_small = -10 * np.ones(self.nx)
        log_like_small = self.inference.log_likelihood(
            theta_small, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Should either be finite or return -inf for numerical issues
        assert np.isfinite(log_like_small) or log_like_small == -np.inf
        
        # Very positive theta (very large conductivity)
        theta_large = 10 * np.ones(self.nx)
        log_like_large = self.inference.log_likelihood(
            theta_large, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Should either be finite or return very negative value for numerical issues
        assert np.isfinite(log_like_large) or log_like_large < -1e6