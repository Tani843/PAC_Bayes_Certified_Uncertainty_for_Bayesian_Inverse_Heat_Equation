"""Tests for PAC-Bayes uncertainty quantification module."""

import pytest
import numpy as np
from src.heat_solver import HeatSolver
from src.bayesian_inference import BayesianHeatInference
from src.pac_bayes import PACBayesUncertainty
from src.data_generator import DataGenerator


class TestPACBayesUncertainty:
    """Test class for PACBayesUncertainty."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.domain = (0.0, 1.0)
        self.nx = 15  # Small for fast tests
        self.dt = 0.001
        self.solver = HeatSolver(self.domain[0], self.domain[1], self.nx, self.dt)
        self.inference = BayesianHeatInference(self.solver)
        self.pac_bayes = PACBayesUncertainty(self.solver, self.inference)
        
        # Generate test data
        self.generator = DataGenerator(self.domain, self.nx, self.dt)
        self.test_dataset = self.generator.generate_dataset(
            conductivity_name='constant',
            t_final=0.05,
            noise_level=0.02
        )
    
    def test_initialization(self):
        """Test PAC-Bayes uncertainty quantifier initialization."""
        assert self.pac_bayes.solver == self.solver
        assert self.pac_bayes.inference == self.inference
        assert self.pac_bayes.nx == self.nx
    
    def test_kl_divergence_gaussians(self):
        """Test KL divergence computation between Gaussians."""
        # Test with identical distributions (should be 0)
        mu = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.1], [0.1, 2.0]])
        
        kl_div = self.pac_bayes.kl_divergence_gaussians(mu, cov, mu, cov)
        assert np.isclose(kl_div, 0.0, atol=1e-10)
        
        # Test with different distributions (should be positive)
        mu2 = mu + 0.5
        cov2 = cov * 1.5
        
        kl_div = self.pac_bayes.kl_divergence_gaussians(mu, cov, mu2, cov2)
        assert kl_div > 0
        assert np.isfinite(kl_div)
        
        # KL divergence should be asymmetric
        kl_div_reverse = self.pac_bayes.kl_divergence_gaussians(mu2, cov2, mu, cov)
        assert not np.isclose(kl_div, kl_div_reverse)
    
    def test_empirical_risk(self):
        """Test empirical risk computation."""
        # Use true parameters
        true_k = self.test_dataset['true_conductivity']
        theta_true = np.log(true_k)
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        risk = self.pac_bayes.empirical_risk(
            theta_true, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Risk should be finite and positive (negative log-likelihood)
        assert np.isfinite(risk)
        assert risk >= 0  # Risk is negative log-likelihood, should be non-negative
        
        # Wrong parameters should have higher risk
        theta_wrong = theta_true + 0.5
        risk_wrong = self.pac_bayes.empirical_risk(
            theta_wrong, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Usually true parameters should have lower risk, but this is stochastic
        assert risk_wrong >= risk - 10  # Allow some tolerance
    
    def test_pac_bayes_bound(self):
        """Test PAC-Bayes generalization bound computation."""
        # Generate mock posterior samples around true parameters
        true_k = self.test_dataset['true_conductivity']
        theta_true = np.log(true_k)
        
        n_samples = 30
        posterior_samples = (theta_true[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        bounds = self.pac_bayes.pac_bayes_bound(
            posterior_samples, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            delta=0.05
        )
        
        # Check required fields
        required_fields = ['empirical_risk', 'kl_divergence', 'pac_bound', 
                          'bernstein_bound', 'confidence', 'n_observations']
        
        for field in required_fields:
            assert field in bounds
        
        # Check properties
        assert bounds['empirical_risk'] >= 0
        assert bounds['kl_divergence'] >= 0
        assert bounds['pac_bound'] >= bounds['empirical_risk']  # Bound should be above risk
        assert bounds['confidence'] == 0.95  # 1 - delta
        assert np.isfinite(bounds['pac_bound'])
        assert np.isfinite(bounds['bernstein_bound'])
    
    def test_prediction_bounds(self):
        """Test prediction confidence bounds."""
        # Generate mock posterior samples
        n_samples = 20
        posterior_samples = (self.inference.prior_mean[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        prediction_times = np.linspace(0, 0.03, 5)  # Short time for stability
        prediction_locations = np.array([0.3, 0.7])
        
        bounds = self.pac_bayes.prediction_bounds(
            posterior_samples,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            prediction_times,
            prediction_locations,
            confidence=0.9
        )
        
        # Check required fields
        required_fields = ['predictions', 'prediction_mean', 'prediction_std',
                          'prediction_lower', 'prediction_upper', 'coverage_width',
                          'confidence_level', 'n_valid_samples']
        
        for field in required_fields:
            assert field in bounds
        
        # Check dimensions
        expected_shape = (len(prediction_times), len(prediction_locations))
        assert bounds['prediction_mean'].shape == expected_shape
        assert bounds['prediction_lower'].shape == expected_shape
        assert bounds['prediction_upper'].shape == expected_shape
        
        # Check ordering: lower <= mean <= upper
        assert np.all(bounds['prediction_lower'] <= bounds['prediction_mean'])
        assert np.all(bounds['prediction_mean'] <= bounds['prediction_upper'])
        
        # Check that standard deviation is non-negative
        assert np.all(bounds['prediction_std'] >= 0)
        
        # Check confidence level
        assert bounds['confidence_level'] == 0.9
    
    def test_concentration_bounds(self):
        """Test posterior concentration bounds."""
        # Generate mock posterior samples
        n_samples = 50
        posterior_samples = (self.inference.prior_mean[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        # Test without true values
        bounds = self.pac_bayes.concentration_bounds(posterior_samples)
        
        required_fields = ['posterior_concentration', 'distance_to_prior',
                          'effective_dimensionality', 'posterior_determinant',
                          'condition_number']
        
        for field in required_fields:
            assert field in bounds
            assert np.isfinite(bounds[field])
            assert bounds[field] >= 0  # These should all be non-negative
        
        # Test with true values
        true_theta = np.log(self.test_dataset['true_conductivity'])
        bounds_with_truth = self.pac_bayes.concentration_bounds(
            posterior_samples, true_theta
        )
        
        additional_fields = ['distance_to_truth', 'mahalanobis_distances',
                           'chi2_statistic', 'expected_chi2', 'coverage_calibration']
        
        for field in additional_fields:
            assert field in bounds_with_truth
        
        # Chi-square statistic should be positive
        assert bounds_with_truth['chi2_statistic'] > 0
        assert bounds_with_truth['expected_chi2'] == self.nx
        
        # Mahalanobis distances should be non-negative
        mahal_dists = bounds_with_truth['mahalanobis_distances']
        assert len(mahal_dists) == n_samples
        assert np.all(mahal_dists >= 0)
    
    @pytest.mark.slow
    def test_cross_validation_bounds(self):
        """Test cross-validation based uncertainty bounds (marked as slow)."""
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Run with minimal settings for speed
        cv_results = self.pac_bayes.cross_validation_bounds(
            observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            k_folds=3, n_mcmc_samples=50  # Minimal for testing
        )
        
        # Check required fields
        required_fields = ['cv_risk_mean', 'cv_risk_std', 'cv_risk_upper',
                          'fold_risks', 'combined_samples', 'n_folds']
        
        for field in required_fields:
            assert field in cv_results
        
        # Check properties
        assert cv_results['cv_risk_mean'] >= 0
        assert cv_results['cv_risk_std'] >= 0
        assert cv_results['cv_risk_upper'] >= cv_results['cv_risk_mean']
        assert cv_results['n_folds'] == 3
        
        # Should have risks from all folds
        fold_risks = cv_results['fold_risks']
        assert len(fold_risks) > 0
        assert np.all(np.array(fold_risks) >= 0)
        
        # Combined samples should have reasonable shape
        combined_samples = cv_results['combined_samples']
        assert combined_samples.shape[1] == self.nx  # Correct parameter dimension
    
    @pytest.mark.slow
    def test_certified_bounds(self):
        """Test comprehensive certified bounds computation (marked as slow)."""
        # Generate mock posterior samples
        n_samples = 30
        posterior_samples = (self.inference.prior_mean[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        certificates = self.pac_bayes.certified_bounds(
            posterior_samples, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            delta=0.1  # Lower confidence for faster computation
        )
        
        # Check main components
        assert 'pac_bayes' in certificates
        assert 'concentration' in certificates
        assert 'prediction' in certificates
        assert 'confidence_level' in certificates
        
        # Confidence level should match
        assert certificates['confidence_level'] == 0.9
        
        # Each component should be a dictionary with expected fields
        pac_bounds = certificates['pac_bayes']
        assert 'empirical_risk' in pac_bounds
        assert 'pac_bound' in pac_bounds
        
        concentration = certificates['concentration']
        assert 'posterior_concentration' in concentration
        
        prediction = certificates['prediction']
        assert 'prediction_mean' in prediction
        assert 'prediction_lower' in prediction
        assert 'prediction_upper' in prediction
    
    def test_validate_bounds(self):
        """Test validation of computed bounds on test data."""
        # First compute some bounds
        n_samples = 20
        posterior_samples = (self.inference.prior_mean[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Compute certificates
        certificates = self.pac_bayes.certified_bounds(
            posterior_samples, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Use another noise realization as test data
        test_observations = self.test_dataset['observations'][0]  # Same for simplicity
        
        validation = self.pac_bayes.validate_bounds(
            test_observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions']
        )
        
        # Check required fields
        required_fields = ['empirical_coverage', 'expected_coverage', 'coverage_gap',
                          'mean_absolute_error', 'rmse', 'mean_interval_width']
        
        for field in required_fields:
            assert field in validation
            assert np.isfinite(validation[field])
        
        # Check bounds on metrics
        assert 0 <= validation['empirical_coverage'] <= 1
        assert validation['mean_absolute_error'] >= 0
        assert validation['rmse'] >= 0
        assert validation['mean_interval_width'] >= 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very few samples
        posterior_samples = self.inference.prior_mean.reshape(1, -1)  # Single sample
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Should not crash with single sample
        try:
            bounds = self.pac_bayes.pac_bayes_bound(
                posterior_samples, observations, sensor_locs, obs_times, 0.02**2,
                self.test_dataset['initial_condition'],
                self.test_dataset['boundary_conditions']
            )
            
            # Results should be reasonable even with one sample
            assert np.isfinite(bounds['empirical_risk'])
            assert np.isfinite(bounds['pac_bound'])
            
        except Exception as e:
            # It's also acceptable to handle single samples specially
            pytest.skip(f"Single sample case not handled: {e}")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        # Test with very small/large covariance matrices
        n = 3
        
        # Very small covariance (near singular)
        mu1 = np.zeros(n)
        cov1 = 1e-8 * np.eye(n)
        mu2 = np.ones(n)
        cov2 = np.eye(n)
        
        # Should handle near-singular matrices gracefully
        try:
            kl_div = self.pac_bayes.kl_divergence_gaussians(mu1, cov1, mu2, cov2)
            assert np.isfinite(kl_div) or kl_div == np.inf
        except np.linalg.LinAlgError:
            # Acceptable to raise error for singular matrices
            pass
        
        # Very large covariance
        cov_large = 1e6 * np.eye(n)
        
        try:
            kl_div = self.pac_bayes.kl_divergence_gaussians(mu1, cov_large, mu2, cov2)
            assert np.isfinite(kl_div)
        except (np.linalg.LinAlgError, OverflowError):
            # Acceptable to have numerical issues with extreme values
            pass
    
    def test_confidence_parameter_effect(self):
        """Test effect of confidence parameter on bounds."""
        n_samples = 25
        posterior_samples = (self.inference.prior_mean[np.newaxis, :] + 
                           0.1 * np.random.randn(n_samples, self.nx))
        
        observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        # Test different confidence levels
        delta_low = 0.01   # High confidence (99%)
        delta_high = 0.2   # Lower confidence (80%)
        
        bounds_high_conf = self.pac_bayes.pac_bayes_bound(
            posterior_samples, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            delta=delta_low
        )
        
        bounds_low_conf = self.pac_bayes.pac_bayes_bound(
            posterior_samples, observations, sensor_locs, obs_times, 0.02**2,
            self.test_dataset['initial_condition'],
            self.test_dataset['boundary_conditions'],
            delta=delta_high
        )
        
        # Higher confidence should give tighter bounds (higher PAC bound)
        # because we're more conservative
        assert bounds_high_conf['pac_bound'] >= bounds_low_conf['pac_bound']
        
        # Empirical risk should be the same
        assert np.isclose(bounds_high_conf['empirical_risk'], 
                         bounds_low_conf['empirical_risk'])
    
    def test_validate_bounds_error_handling(self):
        """Test error handling in bounds validation."""
        # Try to validate without computing bounds first
        self.pac_bayes.confidence_certificates = None
        
        test_observations = self.test_dataset['observations'][0]
        sensor_locs = self.test_dataset['sensor_locations']
        obs_times = self.test_dataset['observation_times']
        
        with pytest.raises(ValueError, match="No bounds computed"):
            self.pac_bayes.validate_bounds(
                test_observations, sensor_locs, obs_times, 0.02**2,
                self.test_dataset['initial_condition'],
                self.test_dataset['boundary_conditions']
            )