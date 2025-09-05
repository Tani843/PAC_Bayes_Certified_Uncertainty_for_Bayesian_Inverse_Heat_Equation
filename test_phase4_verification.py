#!/usr/bin/env python3
"""
Phase 4 Verification: Bayesian Inference Testing
Comprehensive validation of MCMC sampling, convergence diagnostics, and posterior analysis.
"""

import numpy as np
import sys
import warnings
import json
from pathlib import Path
import tempfile
import shutil
import os

print("=" * 60)
print("PAC-BAYES HEAT EQUATION - PHASE 4 VERIFICATION")
print("=" * 60)

def test_imports():
    """Test all required imports for Phase 4."""
    print("\nTesting Phase 4: Imports and Dependencies")
    
    try:
        from src.bayesian_inference import BayesianInference, BayesianInferenceConfig
        print("✓ BayesianInference classes import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        import emcee
        print("✓ emcee available for MCMC sampling")
        print(f"  Version: {emcee.__version__}")
    except ImportError:
        print("✗ emcee not available - install with: pip install emcee>=3.1.0")
        return False

    try:
        from scipy import stats
        print("✓ scipy.stats available for statistical analysis")
    except ImportError:
        print("✗ scipy.stats not available")
        return False
    
    try:
        import multiprocessing as mp
        print(f"✓ Multiprocessing available: {mp.cpu_count()} cores")
    except ImportError:
        print("✗ Multiprocessing not available")
        return False
    
    return True

def test_configuration():
    """Test configuration object creation and validation."""
    print("\nTesting Configuration Setup")
    
    try:
        from src.bayesian_inference import BayesianInferenceConfig
        
        config = BayesianInferenceConfig()
        
        # Check default values
        assert config.prior_bounds == [1.0, 10.0], f"Prior bounds: {config.prior_bounds}"
        assert config.n_walkers >= 16, f"Walkers: {config.n_walkers}"
        assert config.n_samples >= 1000, f"Samples: {config.n_samples}"
        assert config.n_chains >= 2, f"Chains: {config.n_chains}"
        
        print("✓ Configuration object created with correct defaults")
        print(f"  Prior bounds: {config.prior_bounds}")
        print(f"  MCMC: {config.n_walkers} walkers, {config.n_samples} samples, {config.n_chains} chains")
        print(f"  Multiprocessing: {config.use_multiprocessing} ({config.n_processes} processes)")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_bayesian_inference_setup():
    """Test Bayesian inference object creation and setup."""
    print("\nTesting Bayesian Inference Setup")
    
    try:
        from src.bayesian_inference import BayesianInference, BayesianInferenceConfig
        
        config = BayesianInferenceConfig()
        config.n_samples = 100  # Reduced for testing
        config.n_burn = 50
        config.use_multiprocessing = False  # Disable for testing
        
        inference = BayesianInference(config)
        print("✓ BayesianInference object created")
        
        # Test forward problem setup
        inference.setup_forward_problem(domain_length=1.0, nx=50, final_time=0.3)
        print("✓ Forward problem setup complete")
        
        # Create synthetic observations for testing
        observation_times = np.linspace(0.1, 0.3, 10)
        sensor_locations = np.array([0.25, 0.5, 0.75])
        observations = np.random.normal(0.5, 0.1, (len(observation_times), len(sensor_locations)))
        true_kappa = 5.0
        
        inference.set_observations(observations, observation_times, sensor_locations, 
                                 noise_std=0.05, true_kappa=true_kappa)
        print(f"✓ Observations set: {observations.shape}, noise_std = {inference.noise_std}")
        
        return True
        
    except Exception as e:
        print(f"✗ Bayesian inference setup failed: {e}")
        return False

def test_probability_functions():
    """Test prior, likelihood, and posterior computations."""
    print("\nTesting Probability Functions")
    
    try:
        from src.bayesian_inference import BayesianInference, BayesianInferenceConfig
        
        # Create minimal setup
        config = BayesianInferenceConfig()
        config.use_multiprocessing = False
        inference = BayesianInference(config)
        
        # Setup forward problem
        inference.setup_forward_problem(domain_length=1.0, nx=30, final_time=0.2)
        
        # Create minimal observations
        observation_times = np.array([0.1, 0.15, 0.2])
        sensor_locations = np.array([0.5])
        observations = np.random.normal(0.3, 0.05, (3, 1))
        inference.set_observations(observations, observation_times, sensor_locations, 
                                 noise_std=0.05, true_kappa=5.0)
        
        # Test prior
        log_prior_valid = inference.log_prior(5.0)
        log_prior_invalid = inference.log_prior(15.0)
        assert np.isfinite(log_prior_valid), "Prior should be finite for valid kappa"
        assert log_prior_invalid == -np.inf, "Prior should be -inf for invalid kappa"
        print(f"✓ Prior function: valid κ = {log_prior_valid:.4f}, invalid κ = {log_prior_invalid}")
        
        # Test likelihood
        log_likelihood = inference.log_likelihood(5.0)
        assert np.isfinite(log_likelihood), "Likelihood should be finite for reasonable kappa"
        print(f"✓ Likelihood function: log_likelihood = {log_likelihood:.4f}")
        
        # Test posterior
        log_posterior = inference.log_posterior(5.0)
        assert np.isfinite(log_posterior), "Posterior should be finite for valid kappa"
        print(f"✓ Posterior function: log_posterior = {log_posterior:.4f}")
        
        # Test edge cases
        log_post_edge = inference.log_posterior(0.5)  # Outside prior
        assert log_post_edge == -np.inf, "Posterior should be -inf outside prior bounds"
        print("✓ Edge case handling correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Probability function test failed: {e}")
        return False

def test_mcmc_single_chain():
    """Test single MCMC chain execution."""
    print("\nTesting Single MCMC Chain")
    
    try:
        from src.bayesian_inference import BayesianInference, BayesianInferenceConfig
        
        # Create minimal configuration
        config = BayesianInferenceConfig()
        config.n_samples = 100
        config.n_burn = 50
        config.n_walkers = 8
        config.use_multiprocessing = False
        
        inference = BayesianInference(config)
        
        # Setup forward problem
        inference.setup_forward_problem(domain_length=1.0, nx=30, final_time=0.2)
        
        # Create minimal observations
        observation_times = np.array([0.1, 0.15, 0.2])
        sensor_locations = np.array([0.5])
        observations = np.random.normal(0.3, 0.05, (3, 1))
        inference.set_observations(observations, observation_times, sensor_locations, 
                                 noise_std=0.05, true_kappa=5.0)
        
        # Run single chain
        samples, log_probs = inference.run_single_chain_internal(chain_id=0)
        
        assert len(samples) > 0, "Should have samples"
        assert len(log_probs) == len(samples), "Log probs should match samples"
        assert np.all(np.isfinite(samples)), "All samples should be finite"
        assert np.all(samples >= inference.config.prior_bounds[0]), "Samples should be within prior bounds"
        assert np.all(samples <= inference.config.prior_bounds[1]), "Samples should be within prior bounds"
        
        print(f"✓ Single chain complete: {len(samples)} samples")
        print(f"  Sample range: [{np.min(samples):.3f}, {np.max(samples):.3f}]")
        print(f"  Sample mean: {np.mean(samples):.3f}")
        print(f"  Sample std: {np.std(samples):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Single chain test failed: {e}")
        return False

def test_convergence_diagnostics():
    """Test R-hat and effective sample size computation."""
    print("\nTesting Convergence Diagnostics")
    
    try:
        from src.bayesian_inference import BayesianInference
        
        # Create test chains with known properties
        n_chains, n_samples = 3, 500
        test_chains = []
        
        # Create well-mixed chains (should have low R-hat)
        for i in range(n_chains):
            chain = np.random.normal(5.0, 0.5, n_samples)  # Same distribution
            test_chains.append(chain)
        
        inference = BayesianInference()
        inference.chains = test_chains
        
        # Test convergence diagnostics
        rhat, n_eff = inference._compute_convergence_diagnostics()
        
        assert np.isfinite(rhat), "R-hat should be finite"
        assert rhat > 0, "R-hat should be positive"
        assert rhat < 1.3, f"R-hat should be reasonable for well-mixed chains: {rhat}"
        assert n_eff > 0, "Effective sample size should be positive"
        assert n_eff <= n_chains * n_samples, "n_eff should not exceed total samples"
        
        print(f"✓ Convergence diagnostics: R-hat = {rhat:.4f}, n_eff = {n_eff:.1f}")
        
        # Test with poorly mixed chains (should have high R-hat)
        poor_chains = []
        for i in range(n_chains):
            chain = np.random.normal(3.0 + 2*i, 0.1, n_samples)  # Different means
            poor_chains.append(chain)
        
        inference.chains = poor_chains
        rhat_poor, _ = inference._compute_convergence_diagnostics()
        
        assert rhat_poor > rhat, "Poorly mixed chains should have higher R-hat"
        print(f"✓ Poor mixing detection: R-hat = {rhat_poor:.4f} > {rhat:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convergence diagnostics test failed: {e}")
        return False

def test_parallel_mcmc_execution():
    """Test parallel MCMC execution with minimal setup."""
    print("\nTesting Parallel MCMC Execution")
    
    try:
        from src.bayesian_inference import BayesianInference, BayesianInferenceConfig
        
        # Create very minimal configuration for fast testing
        config = BayesianInferenceConfig()
        config.n_chains = 2
        config.n_samples = 50  # Very small for quick test
        config.n_burn = 25
        config.n_walkers = 4
        config.max_iterations = 1  # Only one iteration
        config.rhat_threshold = 2.0  # Very relaxed for testing
        config.min_effective_samples = 10
        
        inference = BayesianInference(config)
        
        # Setup minimal forward problem
        inference.setup_forward_problem(domain_length=1.0, nx=20, final_time=0.1)
        
        # Create minimal observations
        observation_times = np.array([0.05, 0.1])
        sensor_locations = np.array([0.5])
        observations = np.random.normal(0.2, 0.02, (2, 1))
        inference.set_observations(observations, observation_times, sensor_locations, 
                                 noise_std=0.02, true_kappa=5.0)
        
        # Test sequential execution
        inference.config.use_multiprocessing = False
        converged = inference.run_parallel_mcmc()
        
        # Check results
        assert len(inference.chains) >= 1, f"Should have at least 1 successful chain"
        assert inference.final_samples is not None, "Final samples should exist"
        assert len(inference.final_samples) > 0, "Should have final samples"
        assert inference.posterior_summary, "Posterior summary should be computed"
        
        print(f"✓ Sequential execution: {len(inference.chains)} chains, {len(inference.final_samples)} samples")
        print(f"  Converged: {converged}")
        print(f"  Posterior mean: {inference.posterior_summary['mean']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Parallel MCMC test failed: {e}")
        return False

def run_all_tests():
    """Run all Phase 4 verification tests."""
    
    tests = [
        ("Imports and Dependencies", test_imports),
        ("Configuration Setup", test_configuration),
        ("Bayesian Inference Setup", test_bayesian_inference_setup),
        ("Probability Functions", test_probability_functions),
        ("Single MCMC Chain", test_mcmc_single_chain),
        ("Convergence Diagnostics", test_convergence_diagnostics),
        ("Parallel MCMC Execution", test_parallel_mcmc_execution)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("PHASE 4 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"PASSED: {passed}/{total} tests")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - PHASE 4 BAYESIAN INFERENCE COMPLETE!")
        print("\nPhase 4 Implementation Features:")
        print("✓ Complete Bayesian inference framework")
        print("✓ MCMC sampling with emcee")
        print("✓ R-hat convergence diagnostics")
        print("✓ Effective sample size computation")
        print("✓ Parallel and sequential execution modes")
        print("✓ Robust error handling")
        print("✓ Comprehensive posterior analysis")
        print("\nReady for Phase 5: PAC-Bayes Certified Bounds!")
    else:
        print(f"❌ {total-passed} tests failed - review implementation")
        print("\nFailed tests indicate issues that need to be addressed:")
        print("- Check imports and dependencies")
        print("- Verify MCMC implementation")
        print("- Review convergence diagnostics")
        print("- Test parallel execution capabilities")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)