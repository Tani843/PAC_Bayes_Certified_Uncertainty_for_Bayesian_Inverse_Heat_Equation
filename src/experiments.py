"""Experimental Validation Framework

This module provides comprehensive experiments to validate the PAC-Bayes
uncertainty quantification framework for inverse heat equation problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
import pickle
from pathlib import Path
import pandas as pd

from .heat_solver import HeatEquationSolver
from .data_generator import SyntheticDataGenerator
from .bayesian_inference import BayesianInference
from .pac_bayes import PACBayesUncertainty


class ExperimentRunner:
    """Comprehensive experimental validation of PAC-Bayes uncertainty quantification."""
    
    def __init__(self, 
                 output_dir: str = "results",
                 random_seed: int = 42):
        """Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Storage for results
        self.experiment_results = {}
        
        # Configure plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def experiment_1_synthetic_validation(self,
                                        domain: Tuple[float, float] = (0.0, 1.0),
                                        nx: int = 50,
                                        dt: float = 0.001) -> Dict:
        """Experiment 1: Validation on synthetic data with known ground truth.
        
        Args:
            domain: Spatial domain
            nx: Number of spatial grid points
            dt: Time step
            
        Returns:
            Experimental results dictionary
        """
        print("=" * 60)
        print("EXPERIMENT 1: SYNTHETIC DATA VALIDATION")
        print("=" * 60)
        
        # Initialize components
        solver = HeatEquationSolver(domain[1] - domain[0])
        config = DataGenerationConfig(
            domain_length=domain[1] - domain[0],
            nx=nx,
            dt=dt
        )
        generator = SyntheticDataGenerator(config)
        inference = BayesianInference(solver)
        pac_bayes = PACBayesUncertainty(solver, inference)
        
        results = {}
        
        # Test multiple conductivity profiles
        test_profiles = ['linear', 'quadratic', 'sinusoidal', 'step', 'gaussian']
        
        for profile_name in test_profiles:
            print(f"\nTesting profile: {profile_name}")
            
            # Generate synthetic dataset
            dataset = generator.generate_dataset(
                conductivity_name=profile_name,
                initial_condition='gaussian',
                t_final=0.5,
                noise_level=0.02,
                n_realizations=3
            )
            
            # Extract data for inference
            obs = dataset['observations'][0]  # First realization
            sensor_locs = dataset['sensor_locations']
            obs_times = dataset['observation_times']
            
            # Run Bayesian inference
            print("  Running MAP estimation...")
            map_result = inference.find_map_estimate(
                obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions']
            )
            
            print("  Running MCMC sampling...")
            mcmc_result = inference.mcmc_sample(
                obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions'],
                n_samples=2000, n_burn=500
            )
            
            # Compute PAC-Bayes bounds
            print("  Computing PAC-Bayes bounds...")
            cert_bounds = pac_bayes.certified_bounds(
                mcmc_result['samples'], obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions'],
                delta=0.05
            )
            
            # Validate on test data (use different noise realization)
            test_obs = dataset['observations'][1]
            validation = pac_bayes.validate_bounds(
                test_obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions']
            )
            
            # Store results
            profile_results = {
                'dataset': dataset,
                'map_result': map_result,
                'mcmc_result': mcmc_result,
                'pac_bounds': cert_bounds,
                'validation': validation,
                'true_conductivity': dataset['true_conductivity']
            }
            
            results[profile_name] = profile_results
            
            # Quick diagnostic print
            print(f"    MAP success: {map_result['optimization_success']}")
            print(f"    MCMC acceptance rate: {mcmc_result['acceptance_rate']:.3f}")
            print(f"    PAC-Bayes bound: {cert_bounds['pac_bayes']['pac_bound']:.4f}")
            print(f"    Coverage: {validation['empirical_coverage']:.3f}")
        
        self.experiment_results['synthetic_validation'] = results
        return results
    
    def experiment_2_noise_robustness(self,
                                    domain: Tuple[float, float] = (0.0, 1.0),
                                    nx: int = 50) -> Dict:
        """Experiment 2: Robustness to different noise levels.
        
        Args:
            domain: Spatial domain
            nx: Number of spatial grid points
            
        Returns:
            Noise robustness results
        """
        print("=" * 60)
        print("EXPERIMENT 2: NOISE ROBUSTNESS ANALYSIS")
        print("=" * 60)
        
        # Initialize components
        solver = HeatEquationSolver(domain[1] - domain[0])
        generator = DataGenerator(domain, nx, 0.001)
        inference = BayesianInference(solver)
        pac_bayes = PACBayesUncertainty(solver, inference)
        
        # Test different noise levels
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        results = {}
        
        for noise_level in noise_levels:
            print(f"\nTesting noise level: {noise_level}")
            
            # Generate dataset with specific noise level
            dataset = generator.generate_dataset(
                conductivity_name='sinusoidal',
                noise_level=noise_level,
                n_realizations=5
            )
            
            # Use first realization for training
            obs = dataset['observations'][0]
            sensor_locs = dataset['sensor_locations']
            obs_times = dataset['observation_times']
            
            # Run inference
            mcmc_result = inference.mcmc_sample(
                obs, sensor_locs, obs_times, noise_level**2,
                dataset['initial_condition'], dataset['boundary_conditions'],
                n_samples=1500, n_burn=300
            )
            
            # Compute bounds
            cert_bounds = pac_bayes.certified_bounds(
                mcmc_result['samples'], obs, sensor_locs, obs_times, noise_level**2,
                dataset['initial_condition'], dataset['boundary_conditions']
            )
            
            # Test on remaining realizations
            coverage_rates = []
            for i in range(1, 5):
                test_obs = dataset['observations'][i]
                validation = pac_bayes.validate_bounds(
                    test_obs, sensor_locs, obs_times, noise_level**2,
                    dataset['initial_condition'], dataset['boundary_conditions']
                )
                coverage_rates.append(validation['empirical_coverage'])
            
            results[noise_level] = {
                'mcmc_result': mcmc_result,
                'cert_bounds': cert_bounds,
                'coverage_rates': coverage_rates,
                'mean_coverage': np.mean(coverage_rates),
                'std_coverage': np.std(coverage_rates),
                'dataset': dataset
            }
            
            print(f"    Mean coverage: {np.mean(coverage_rates):.3f} ± {np.std(coverage_rates):.3f}")
            print(f"    PAC-Bayes bound: {cert_bounds['pac_bayes']['pac_bound']:.4f}")
        
        self.experiment_results['noise_robustness'] = results
        return results
    
    def experiment_3_sensor_configuration(self,
                                        domain: Tuple[float, float] = (0.0, 1.0),
                                        nx: int = 50) -> Dict:
        """Experiment 3: Effect of sensor number and placement.
        
        Args:
            domain: Spatial domain
            nx: Number of spatial grid points
            
        Returns:
            Sensor configuration results
        """
        print("=" * 60)
        print("EXPERIMENT 3: SENSOR CONFIGURATION STUDY")
        print("=" * 60)
        
        # Initialize components
        solver = HeatEquationSolver(domain[1] - domain[0])
        generator = DataGenerator(domain, nx, 0.001)
        inference = BayesianInference(solver)
        pac_bayes = PACBayesUncertainty(solver, inference)
        
        results = {}
        
        # Test different numbers of sensors
        n_sensors_list = [5, 10, 15, 20, 25]
        
        for n_sensors in n_sensors_list:
            print(f"\nTesting {n_sensors} sensors")
            
            # Different sensor configurations
            configurations = {
                'uniform': np.linspace(domain[0] + 0.1, domain[1] - 0.1, n_sensors),
                'clustered_left': np.linspace(domain[0] + 0.05, domain[0] + 0.4, n_sensors),
                'clustered_right': np.linspace(domain[1] - 0.4, domain[1] - 0.05, n_sensors),
                'random': np.sort(np.random.uniform(domain[0] + 0.1, domain[1] - 0.1, n_sensors))
            }
            
            sensor_results = {}
            
            for config_name, sensor_locs in configurations.items():
                print(f"  Configuration: {config_name}")
                
                # Generate base dataset
                base_dataset = generator.generate_dataset(
                    conductivity_name='gaussian',
                    noise_level=0.02
                )
                
                # Extract observations at specific sensor locations
                full_solution = base_dataset['solution']
                obs = solver.compute_observations(
                    full_solution, sensor_locs, noise_level=0.02
                )
                
                # Run inference
                mcmc_result = inference.mcmc_sample(
                    obs, sensor_locs, base_dataset['observation_times'], 0.02**2,
                    base_dataset['initial_condition'], base_dataset['boundary_conditions'],
                    n_samples=1000, n_burn=200
                )
                
                # Compute bounds
                cert_bounds = pac_bayes.certified_bounds(
                    mcmc_result['samples'], obs, sensor_locs, 
                    base_dataset['observation_times'], 0.02**2,
                    base_dataset['initial_condition'], base_dataset['boundary_conditions']
                )
                
                sensor_results[config_name] = {
                    'sensor_locations': sensor_locs,
                    'mcmc_result': mcmc_result,
                    'cert_bounds': cert_bounds,
                    'observations': obs
                }
                
                print(f"      PAC bound: {cert_bounds['pac_bayes']['pac_bound']:.4f}")
                print(f"      KL div: {cert_bounds['pac_bayes']['kl_divergence']:.4f}")
            
            results[n_sensors] = sensor_results
        
        self.experiment_results['sensor_configuration'] = results
        return results
    
    def experiment_4_computational_efficiency(self,
                                            domain: Tuple[float, float] = (0.0, 1.0)) -> Dict:
        """Experiment 4: Computational efficiency analysis.
        
        Args:
            domain: Spatial domain
            
        Returns:
            Efficiency analysis results
        """
        print("=" * 60)
        print("EXPERIMENT 4: COMPUTATIONAL EFFICIENCY")
        print("=" * 60)
        
        results = {}
        grid_sizes = [25, 50, 75, 100]
        
        for nx in grid_sizes:
            print(f"\nTesting grid size: {nx}")
            
            # Initialize components
            solver = HeatEquationSolver(domain[1] - domain[0])
            generator = DataGenerator(domain, nx, 0.001)
            inference = BayesianInference(solver)
            pac_bayes = PACBayesUncertainty(solver, inference)
            
            # Generate dataset
            dataset = generator.generate_dataset(
                conductivity_name='linear',
                noise_level=0.02
            )
            
            obs = dataset['observations'][0]
            sensor_locs = dataset['sensor_locations']
            obs_times = dataset['observation_times']
            
            # Time MAP estimation
            start_time = time.time()
            map_result = inference.find_map_estimate(
                obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions']
            )
            map_time = time.time() - start_time
            
            # Time MCMC sampling
            start_time = time.time()
            mcmc_result = inference.mcmc_sample(
                obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions'],
                n_samples=1000, n_burn=200
            )
            mcmc_time = time.time() - start_time
            
            # Time PAC-Bayes bounds
            start_time = time.time()
            cert_bounds = pac_bayes.certified_bounds(
                mcmc_result['samples'], obs, sensor_locs, obs_times, 0.02**2,
                dataset['initial_condition'], dataset['boundary_conditions']
            )
            pac_time = time.time() - start_time
            
            results[nx] = {
                'map_time': map_time,
                'mcmc_time': mcmc_time,
                'pac_time': pac_time,
                'total_time': map_time + mcmc_time + pac_time,
                'map_result': map_result,
                'mcmc_result': mcmc_result,
                'cert_bounds': cert_bounds
            }
            
            print(f"    MAP time: {map_time:.2f}s")
            print(f"    MCMC time: {mcmc_time:.2f}s") 
            print(f"    PAC-Bayes time: {pac_time:.2f}s")
            print(f"    Total time: {map_time + mcmc_time + pac_time:.2f}s")
        
        self.experiment_results['computational_efficiency'] = results
        return results
    
    def visualize_results(self, experiment_name: str, save_plots: bool = True):
        """Visualize experimental results.
        
        Args:
            experiment_name: Name of experiment to visualize
            save_plots: Whether to save plots to disk
        """
        if experiment_name not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        results = self.experiment_results[experiment_name]
        
        if experiment_name == 'synthetic_validation':
            self._plot_synthetic_validation(results, save_plots)
        elif experiment_name == 'noise_robustness':
            self._plot_noise_robustness(results, save_plots)
        elif experiment_name == 'sensor_configuration':
            self._plot_sensor_configuration(results, save_plots)
        elif experiment_name == 'computational_efficiency':
            self._plot_computational_efficiency(results, save_plots)
    
    def _plot_synthetic_validation(self, results: Dict, save_plots: bool):
        """Plot synthetic validation results."""
        n_profiles = len(results)
        fig, axes = plt.subplots(2, n_profiles, figsize=(4*n_profiles, 8))
        
        if n_profiles == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (profile_name, data) in enumerate(results.items()):
            # True vs estimated conductivity
            x_grid = data['dataset']['spatial_grid']
            k_true = data['true_conductivity']
            k_post_mean = np.exp(data['mcmc_result']['posterior_mean'])
            k_post_std = np.exp(data['mcmc_result']['posterior_mean']) * data['mcmc_result']['posterior_cov'].diagonal()**0.5
            
            axes[0, i].plot(x_grid, k_true, 'k-', linewidth=2, label='True')
            axes[0, i].plot(x_grid, k_post_mean, 'r--', linewidth=2, label='Posterior mean')
            axes[0, i].fill_between(x_grid, 
                                  k_post_mean - 2*k_post_std,
                                  k_post_mean + 2*k_post_std,
                                  alpha=0.3, color='red', label='95% CI')
            axes[0, i].set_title(f'{profile_name.title()} Profile')
            axes[0, i].set_xlabel('Position x')
            axes[0, i].set_ylabel('Thermal conductivity')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Coverage validation
            validation = data['validation']
            coverage_data = validation['coverage_mask'].astype(float)
            im = axes[1, i].imshow(coverage_data.T, aspect='auto', cmap='RdYlGn', 
                                 vmin=0, vmax=1, origin='lower')
            axes[1, i].set_title(f'Coverage: {validation["empirical_coverage"]:.3f}')
            axes[1, i].set_xlabel('Time step')
            axes[1, i].set_ylabel('Sensor')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.output_dir / 'synthetic_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_noise_robustness(self, results: Dict, save_plots: bool):
        """Plot noise robustness results."""
        noise_levels = list(results.keys())
        mean_coverages = [results[nl]['mean_coverage'] for nl in noise_levels]
        std_coverages = [results[nl]['std_coverage'] for nl in noise_levels]
        pac_bounds = [results[nl]['cert_bounds']['pac_bayes']['pac_bound'] for nl in noise_levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage vs noise
        ax1.errorbar(noise_levels, mean_coverages, yerr=std_coverages, 
                    marker='o', capsize=5, linewidth=2)
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Target coverage')
        ax1.set_xlabel('Noise level')
        ax1.set_ylabel('Empirical coverage')
        ax1.set_title('Coverage vs Noise Level')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # PAC-Bayes bounds vs noise
        ax2.plot(noise_levels, pac_bounds, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Noise level')
        ax2.set_ylabel('PAC-Bayes bound')
        ax2.set_title('PAC-Bayes Bound vs Noise Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.output_dir / 'noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sensor_configuration(self, results: Dict, save_plots: bool):
        """Plot sensor configuration results."""
        n_sensors_list = list(results.keys())
        configurations = list(results[n_sensors_list[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, config in enumerate(configurations):
            pac_bounds = [results[ns][config]['cert_bounds']['pac_bayes']['pac_bound'] 
                         for ns in n_sensors_list]
            kl_divs = [results[ns][config]['cert_bounds']['pac_bayes']['kl_divergence']
                      for ns in n_sensors_list]
            
            axes[i].plot(n_sensors_list, pac_bounds, 'o-', linewidth=2, 
                        label='PAC-Bayes bound')
            ax2 = axes[i].twinx()
            ax2.plot(n_sensors_list, kl_divs, 's--', color='red', linewidth=2,
                    label='KL divergence')
            
            axes[i].set_xlabel('Number of sensors')
            axes[i].set_ylabel('PAC-Bayes bound')
            ax2.set_ylabel('KL divergence')
            axes[i].set_title(f'{config.title()} Configuration')
            axes[i].grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = axes[i].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[i].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.output_dir / 'sensor_configuration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_computational_efficiency(self, results: Dict, save_plots: bool):
        """Plot computational efficiency results."""
        grid_sizes = list(results.keys())
        map_times = [results[nx]['map_time'] for nx in grid_sizes]
        mcmc_times = [results[nx]['mcmc_time'] for nx in grid_sizes]
        pac_times = [results[nx]['pac_time'] for nx in grid_sizes]
        total_times = [results[nx]['total_time'] for nx in grid_sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stacked bar chart of computation times
        width = 3
        ax1.bar(grid_sizes, map_times, width, label='MAP', alpha=0.8)
        ax1.bar(grid_sizes, mcmc_times, width, bottom=map_times, label='MCMC', alpha=0.8)
        bottom = np.array(map_times) + np.array(mcmc_times)
        ax1.bar(grid_sizes, pac_times, width, bottom=bottom, label='PAC-Bayes', alpha=0.8)
        
        ax1.set_xlabel('Grid size (nx)')
        ax1.set_ylabel('Computation time (s)')
        ax1.set_title('Computation Time Breakdown')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scaling analysis
        ax2.loglog(grid_sizes, total_times, 'o-', linewidth=2, markersize=8)
        
        # Fit scaling law
        log_nx = np.log(grid_sizes)
        log_times = np.log(total_times)
        slope = np.polyfit(log_nx, log_times, 1)[0]
        
        ax2.loglog(grid_sizes, total_times[0] * (np.array(grid_sizes)/grid_sizes[0])**slope,
                  '--', alpha=0.7, label=f'~O(n^{slope:.1f})')
        
        ax2.set_xlabel('Grid size (nx)')
        ax2.set_ylabel('Total time (s)')
        ax2.set_title('Computational Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.output_dir / 'computational_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, filename: str = "experiment_report.txt"):
        """Generate comprehensive experimental report.
        
        Args:
            filename: Output filename for report
        """
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("PAC-BAYES UNCERTAINTY QUANTIFICATION EXPERIMENTAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")
            
            for exp_name, results in self.experiment_results.items():
                f.write(f"\n{exp_name.upper().replace('_', ' ')}\n")
                f.write("-" * 40 + "\n")
                
                if exp_name == 'synthetic_validation':
                    f.write("Validation on synthetic datasets with known ground truth.\n\n")
                    for profile, data in results.items():
                        val = data['validation']
                        f.write(f"Profile: {profile}\n")
                        f.write(f"  Coverage: {val['empirical_coverage']:.3f}\n")
                        f.write(f"  RMSE: {val['rmse']:.4f}\n")
                        f.write(f"  Mean interval width: {val['mean_interval_width']:.4f}\n\n")
                
                elif exp_name == 'noise_robustness':
                    f.write("Robustness analysis across different noise levels.\n\n")
                    f.write("Noise Level | Coverage | PAC Bound\n")
                    f.write("-" * 35 + "\n")
                    for noise, data in results.items():
                        f.write(f"{noise:10.3f} | {data['mean_coverage']:8.3f} | "
                               f"{data['cert_bounds']['pac_bayes']['pac_bound']:8.4f}\n")
                
                elif exp_name == 'sensor_configuration':
                    f.write("Impact of sensor number and placement.\n\n")
                    # Add summary statistics
                    
                elif exp_name == 'computational_efficiency':
                    f.write("Computational efficiency analysis.\n\n")
                    f.write("Grid Size | MAP Time | MCMC Time | PAC Time | Total Time\n")
                    f.write("-" * 55 + "\n")
                    for nx, data in results.items():
                        f.write(f"{nx:8d} | {data['map_time']:8.2f} | "
                               f"{data['mcmc_time']:9.2f} | {data['pac_time']:8.2f} | "
                               f"{data['total_time']:10.2f}\n")
        
        print(f"Report saved to: {report_path}")
    
    def save_results(self, filename: str = "experiment_results.pkl"):
        """Save all experimental results to file.
        
        Args:
            filename: Output filename
        """
        save_path = self.output_dir / filename
        with open(save_path, 'wb') as f:
            pickle.dump(self.experiment_results, f)
        print(f"Results saved to: {save_path}")
    
    def load_results(self, filename: str = "experiment_results.pkl"):
        """Load experimental results from file.
        
        Args:
            filename: Input filename
        """
        load_path = self.output_dir / filename
        with open(load_path, 'rb') as f:
            self.experiment_results = pickle.load(f)
        print(f"Results loaded from: {load_path}")
    
    def run_all_experiments(self):
        """Run complete experimental suite."""
        print("RUNNING COMPLETE EXPERIMENTAL SUITE")
        print("=" * 60)
        
        # Run all experiments
        self.experiment_1_synthetic_validation()
        self.experiment_2_noise_robustness()
        self.experiment_3_sensor_configuration()
        self.experiment_4_computational_efficiency()
        
        # Generate visualizations
        for exp_name in self.experiment_results.keys():
            self.visualize_results(exp_name, save_plots=True)
        
        # Generate report and save results
        self.generate_report()
        self.save_results()
        
        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)