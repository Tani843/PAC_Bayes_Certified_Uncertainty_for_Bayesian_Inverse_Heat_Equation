#!/usr/bin/env python3
"""
Phase 4: Simplified Bayesian Inference Test
Test with reduced data and debugging
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.append('src')

from src.bayesian_inference import BayesianInference, BayesianInferenceConfig

def test_simple_inference():
    """Test with simplified setup and debugging."""
    
    print("="*60)
    print("Phase 4: Simplified Bayesian Inference Test")
    print("="*60)
    
    # Create simplified configuration
    config = BayesianInferenceConfig()
    config.n_chains = 2
    config.n_samples = 100  # Much smaller for testing
    config.n_burn = 50
    config.n_walkers = 16
    config.rhat_threshold = 1.2  # More relaxed
    
    inference = BayesianInference(config)
    
    # Create very simple synthetic data
    observation_times = np.linspace(0.1, 0.3, 5)  # Only 5 time points
    sensor_locations = np.array([0.5])  # Only 1 sensor
    true_kappa = 5.0
    
    print(f"Test setup:")
    print(f"  Observation times: {len(observation_times)} points")
    print(f"  Sensor locations: {sensor_locations}")
    print(f"  True κ: {true_kappa}")
    
    # Generate simple observations manually
    from src.heat_solver import HeatEquationSolver
    
    solver = HeatEquationSolver(domain_length=1.0)
    def initial_condition(x):
        return np.exp(-50 * (x - 0.3)**2)
    
    try:
        solution, x_grid, t_grid = solver.solve(
            kappa=true_kappa,
            initial_condition=initial_condition,
            boundary_conditions={'left': 0.0, 'right': 0.0},
            nx=50,  # Smaller grid
            final_time=0.3,  # Shorter time
            auto_timestep=True,
            cfl_factor=0.3
        )
        
        print(f"Forward solver successful:")
        print(f"  Solution shape: {solution.shape}")
        print(f"  X grid: {len(x_grid)} points")
        print(f"  T grid: {len(t_grid)} points")
        
        # Extract observations
        observations = np.zeros((len(observation_times), len(sensor_locations)))
        for i, t_obs in enumerate(observation_times):
            t_idx = np.argmin(np.abs(t_grid - t_obs))
            for j, x_sensor in enumerate(sensor_locations):
                x_idx = np.argmin(np.abs(x_grid - x_sensor))
                observations[i, j] = solution[t_idx, x_idx]
        
        # Add noise
        noise_std = 0.01  # Larger noise for easier sampling
        observations += np.random.normal(0, noise_std, observations.shape)
        
        print(f"Observations created:")
        print(f"  Shape: {observations.shape}")
        print(f"  Range: [{np.min(observations):.4f}, {np.max(observations):.4f}]")
        print(f"  Noise std: {noise_std:.4f}")
        
        # Setup inference
        inference.setup_forward_problem(domain_length=1.0, nx=50, final_time=0.3)
        inference.set_observations(observations, observation_times, sensor_locations, 
                                 noise_std=noise_std, true_kappa=true_kappa)
        
        # Test likelihood function at true value
        print(f"\nTesting likelihood function:")
        test_kappas = [1.0, 3.0, 5.0, 7.0, 10.0]
        for test_kappa in test_kappas:
            log_like = inference.log_likelihood(test_kappa)
            log_prior = inference.log_prior(test_kappa)
            log_post = inference.log_posterior(test_kappa)
            print(f"  κ={test_kappa}: log_like={log_like:.2f}, log_prior={log_prior:.2f}, log_post={log_post:.2f}")
        
        # Run a single short chain for testing
        print(f"\nRunning single test chain...")
        config.n_chains = 1
        config.n_samples = 50
        config.n_burn = 20
        config.max_iterations = 1
        
        start_time = time.time()
        converged = inference.run_parallel_mcmc()
        end_time = time.time()
        
        print(f"\nTest completed in {end_time - start_time:.1f} seconds")
        print(f"Converged: {converged}")
        
        if inference.chains and len(inference.chains) > 0:
            samples = inference.chains[0]
            if len(samples) > 0:
                print(f"Chain results:")
                print(f"  Valid samples: {len(samples)}")
                print(f"  Sample range: [{np.min(samples):.3f}, {np.max(samples):.3f}]")
                print(f"  Sample mean: {np.mean(samples):.3f}")
                print(f"  True value: {true_kappa:.3f}")
            else:
                print("No valid samples generated")
        
        return inference, converged
        
    except Exception as e:
        print(f"Error in simple test: {e}")
        import traceback
        traceback.print_exc()
        return None, False


if __name__ == "__main__":
    test_simple_inference()