#!/usr/bin/env python3
"""Basic Example of PAC-Bayes Framework

This script provides a minimal working example of the PAC-Bayes
uncertainty quantification framework for inverse heat problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from heat_solver import HeatSolver
from data_generator import DataGenerator
from bayesian_inference import BayesianHeatInference
from pac_bayes import PACBayesUncertainty


def main():
    """Run basic example."""
    print("=" * 50)
    print("PAC-BAYES BASIC EXAMPLE")
    print("=" * 50)
    
    # 1. Setup
    print("\n1. Setting up problem...")
    
    # Create heat solver
    solver = HeatSolver(x_min=0.0, x_max=1.0, nx=30, dt=0.001)
    
    # Generate synthetic data
    generator = DataGenerator(domain=(0.0, 1.0), nx=30)
    dataset = generator.generate_dataset(
        conductivity_name='linear',
        noise_level=0.02,
        t_final=0.2
    )
    
    print(f"   Generated data with {len(dataset['sensor_locations'])} sensors")
    
    # 2. Bayesian Inference
    print("\n2. Running Bayesian inference...")
    
    # Set up inference engine
    inference = BayesianHeatInference(solver)
    
    # Find MAP estimate
    map_result = inference.find_map_estimate(
        dataset['observations'][0],
        dataset['sensor_locations'],
        dataset['observation_times'],
        0.02**2,
        dataset['initial_condition'],
        dataset['boundary_conditions']
    )
    
    print(f"   MAP optimization successful: {map_result['optimization_success']}")
    
    # Run MCMC (short run for demo)
    mcmc_result = inference.mcmc_sample(
        dataset['observations'][0],
        dataset['sensor_locations'], 
        dataset['observation_times'],
        0.02**2,
        dataset['initial_condition'],
        dataset['boundary_conditions'],
        n_samples=1000,
        n_burn=200
    )
    
    print(f"   MCMC acceptance rate: {mcmc_result['acceptance_rate']:.3f}")
    
    # 3. PAC-Bayes Bounds
    print("\n3. Computing PAC-Bayes bounds...")
    
    # Initialize PAC-Bayes framework
    pac_bayes = PACBayesUncertainty(solver, inference)
    
    # Compute certified bounds
    bounds = pac_bayes.certified_bounds(
        mcmc_result['samples'],
        dataset['observations'][0],
        dataset['sensor_locations'],
        dataset['observation_times'],
        0.02**2,
        dataset['initial_condition'],
        dataset['boundary_conditions']
    )
    
    print(f"   PAC-Bayes bound: {bounds['pac_bayes']['pac_bound']:.6f}")
    print(f"   Confidence level: {bounds['confidence_level']:.1%}")
    
    # 4. Visualization
    print("\n4. Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True vs estimated conductivity
    x_grid = solver.x
    k_true = dataset['true_conductivity']
    k_samples = np.exp(mcmc_result['samples'])
    k_mean = np.mean(k_samples, axis=0)
    k_lower = np.percentile(k_samples, 2.5, axis=0)
    k_upper = np.percentile(k_samples, 97.5, axis=0)
    
    axes[0].plot(x_grid, k_true, 'k-', linewidth=2, label='True')
    axes[0].plot(x_grid, k_mean, 'r--', linewidth=2, label='Posterior mean')
    axes[0].fill_between(x_grid, k_lower, k_upper, alpha=0.3, color='red', label='95% CI')
    axes[0].set_xlabel('Position x')
    axes[0].set_ylabel('Thermal conductivity')
    axes[0].set_title('Conductivity Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Temperature predictions
    pred_bounds = bounds['prediction']
    sensor_idx = len(dataset['sensor_locations']) // 2  # Middle sensor
    
    axes[1].plot(dataset['observation_times'], dataset['observations'][0][:, sensor_idx], 
                'b-', linewidth=2, label='Observations')
    axes[1].plot(dataset['observation_times'], pred_bounds['prediction_mean'][:, sensor_idx], 
                'r--', linewidth=2, label='Prediction')
    axes[1].fill_between(dataset['observation_times'],
                        pred_bounds['prediction_lower'][:, sensor_idx],
                        pred_bounds['prediction_upper'][:, sensor_idx],
                        alpha=0.3, color='red', label='95% bounds')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title(f'Predictions at x = {dataset["sensor_locations"][sensor_idx]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_example_results.png', dpi=300, bbox_inches='tight')
    print("   Results saved to 'basic_example_results.png'")
    
    # 5. Summary
    print("\n" + "=" * 50)
    print("BASIC EXAMPLE COMPLETE")
    print("=" * 50)
    print("\nKey Results:")
    print(f"• MAP estimate found: {map_result['optimization_success']}")
    print(f"• MCMC acceptance rate: {mcmc_result['acceptance_rate']:.1%}")
    print(f"• PAC-Bayes bound: {bounds['pac_bayes']['pac_bound']:.6f}")
    print(f"• 95% confidence bounds computed")
    
    print("\nThis basic example demonstrated:")
    print("• Synthetic data generation")
    print("• MAP estimation and MCMC sampling")
    print("• PAC-Bayes certified uncertainty bounds")
    print("• Results visualization")
    
    return {
        'dataset': dataset,
        'mcmc_result': mcmc_result,
        'bounds': bounds
    }


if __name__ == "__main__":
    results = main()