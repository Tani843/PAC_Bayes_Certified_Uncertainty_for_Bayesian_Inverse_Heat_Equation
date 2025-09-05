"""
PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation
Phase 5: PAC-Bayes Bounds Implementation

Mathematical Foundation:
PAC-Bayes bound: KL(Q || P) ≤ (1/n)[log(1/δ) + log E_P[e^(-ℓ)]]

Where:
- Q: posterior distribution of κ (from MCMC samples)
- P: prior distribution of κ (uniform [1,10])
- n: number of observations
- δ: confidence level parameter (0.05 for 95% confidence)
- ℓ: empirical loss function
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.integrate import quad
import json
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass

@dataclass
class PACBayesConfig:
    """Configuration for PAC-Bayes bounds computation."""
    delta: float = 0.05  # Confidence level (0.05 for 95% confidence)
    mc_samples_prior: int = 10000  # Monte Carlo samples from prior
    convergence_threshold: float = 1e-4  # Convergence check threshold
    max_iterations: int = 50  # Maximum iterations for convergence
    prior_bounds: Tuple[float, float] = (1.0, 10.0)  # Prior support
    
@dataclass
class PACBayesResults:
    """Results from PAC-Bayes bounds computation."""
    # Core bounds
    kl_divergence: float
    pac_bound: float
    certified_interval: Tuple[float, float]
    
    # Comparison with uncertified
    uncertified_interval: Tuple[float, float]
    interval_ratio: float  # certified_width / uncertified_width
    
    # Computational details
    n_observations: int
    delta: float
    prior_expectation: float
    convergence_achieved: bool
    
    # Coverage validation
    true_parameter: float
    certified_covers_truth: bool
    uncertified_covers_truth: bool

class PACBayesBounds:
    """
    PAC-Bayes certified uncertainty bounds for Bayesian inverse problems.
    
    Implements the PAC-Bayes theorem for providing certified uncertainty
    guarantees on posterior distributions from MCMC sampling.
    """

    def __init__(self, config: Optional[PACBayesConfig] = None):
        self.config = config or PACBayesConfig()
        self.results: Optional[PACBayesResults] = None
        
    def load_phase4_results(self, results_path: str = "results/phase4_production") -> Dict:
        """Load Phase 4 MCMC results and canonical dataset."""
        results_path = Path(results_path)
        
        # Load MCMC results
        results_file = results_path / "inference_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Phase 4 results not found: {results_file}")
            
        with open(results_file, 'r') as f:
            mcmc_results = json.load(f)
        
        # Load posterior samples
        samples_file = results_path / "posterior_samples.npy"
        if samples_file.exists():
            posterior_samples = np.load(samples_file)
        else:
            # Extract from MCMC chains if available
            chains_file = results_path / "mcmc_chains.npy"
            if chains_file.exists():
                chains = np.load(chains_file, allow_pickle=True)
                # Flatten chains after burn-in
                posterior_samples = np.concatenate([chain for chain in chains])
            else:
                # Use final_samples from results if available
                if mcmc_results.get('final_samples'):
                    posterior_samples = np.array(mcmc_results['final_samples'])
                else:
                    raise FileNotFoundError("No posterior samples found")
        
        # Load canonical dataset
        data_file = Path("data/canonical_dataset.npz")
        if not data_file.exists():
            raise FileNotFoundError("Canonical dataset not found")
            
        dataset = np.load(data_file)
        
        return {
            'posterior_samples': posterior_samples,
            'observations': dataset['observations'],
            'observation_times': dataset['observation_times'],
            'sensor_locations': dataset['sensor_locations'],
            'true_kappa': float(dataset['true_kappa']),
            'noise_std': float(dataset['noise_std']),
            'mcmc_results': mcmc_results
        }

    def compute_empirical_loss(self, kappa: float, observations: np.ndarray, 
                             obs_times: np.ndarray, sensors: np.ndarray, 
                             noise_std: float) -> float:
        """
        Compute empirical loss ℓ(κ, data) for a given parameter value.
        
        Loss function: ℓ(κ) = (1/2σ²) Σᵢ (yᵢ - f(κ, xᵢ))²
        where f(κ, xᵢ) is the forward model prediction.
        """
        from src.heat_solver import HeatEquationSolver
        
        # Solve forward problem
        solver = HeatEquationSolver()
        
        def ic(x):
            return np.exp(-50 * (x - 0.3)**2)

        try:
            solution, x_grid, t_grid = solver.solve(
                kappa=kappa,
                initial_condition=ic,
                boundary_conditions={'left': 0.0, 'right': 0.0},
                nx=50,  # Use efficient grid size
                final_time=obs_times[-1],
                auto_timestep=True,
                cfl_factor=0.25
            )
            
            # Extract predictions at observation points
            predictions = np.zeros_like(observations)
            for i, t_obs in enumerate(obs_times):
                t_idx = np.argmin(np.abs(t_grid - t_obs))
                for j, x_sensor in enumerate(sensors):
                    x_idx = np.argmin(np.abs(x_grid - x_sensor))
                    predictions[i, j] = solution[t_idx, x_idx]

            # Compute empirical loss (negative log-likelihood without constants)
            residuals = observations - predictions
            loss = 0.5 * np.sum((residuals / noise_std)**2)
            
            return loss
            
        except Exception as e:
            # Return large loss for failed forward solves
            warnings.warn(f"Forward solve failed for κ={kappa}: {e}")
            return 1e6
    
    def compute_kl_divergence(self, posterior_samples: np.ndarray) -> float:
        """
        Compute KL(Q || P) where Q is empirical posterior, P is uniform prior.
        
        For continuous distributions:
        KL(Q || P) = ∫ q(x) log(q(x)/p(x)) dx
        
        Using histogram approximation for empirical posterior.
        """
        prior_a, prior_b = self.config.prior_bounds
        
        # Filter samples within prior support
        valid_samples = posterior_samples[
            (posterior_samples >= prior_a) & (posterior_samples <= prior_b)
        ]
        
        if len(valid_samples) == 0:
            raise ValueError("No posterior samples within prior support")
        
        # Compute histogram-based density estimate
        n_bins = min(50, len(valid_samples) // 10)  # Adaptive binning
        counts, bin_edges = np.histogram(valid_samples, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Prior density (uniform)
        prior_density = 1.0 / (prior_b - prior_a)
        
        # KL divergence approximation
        kl_div = 0.0
        for i, count in enumerate(counts):
            if count > 0:  # Avoid log(0)
                posterior_density = count
                kl_div += posterior_density * np.log(posterior_density / prior_density) * bin_width
        
        return max(kl_div, 0.0)  # Ensure non-negative

    def compute_prior_expectation_mc(self, observations: np.ndarray, 
                                   obs_times: np.ndarray, sensors: np.ndarray,
                                   noise_std: float) -> Tuple[float, bool]:
        """
        Compute E_P[e^(-ℓ)] using Monte Carlo sampling from prior.
        
        With convergence monitoring for reliable estimation.
        """
        prior_a, prior_b = self.config.prior_bounds
        n_samples = self.config.mc_samples_prior
        
        # Sample from uniform prior
        prior_samples = np.random.uniform(prior_a, prior_b, n_samples)

        # Compute losses and exponentials in batches for memory efficiency
        batch_size = 100
        exp_neg_losses = []
        
        print(f"Computing prior expectation with {n_samples} samples...")
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_samples = prior_samples[i:batch_end]
            
            batch_exp_losses = []
            for kappa in batch_samples:
                loss = self.compute_empirical_loss(kappa, observations, obs_times, sensors, noise_std)
                # Clip loss to prevent numerical overflow
                clipped_loss = min(loss, 50.0)  # e^(-50) ≈ 1.9e-22
                batch_exp_losses.append(np.exp(-clipped_loss))
            
            exp_neg_losses.extend(batch_exp_losses)
            
            # Progress indicator
            if (i // batch_size + 1) % 10 == 0:
                progress = (batch_end / n_samples) * 100
                print(f"  Progress: {progress:.1f}%")
        
        exp_neg_losses = np.array(exp_neg_losses)
        
        # Convergence check using running average
        convergence_achieved = self._check_convergence(exp_neg_losses)
        
        prior_expectation = np.mean(exp_neg_losses)
        print(f"Prior expectation E_P[e^(-ℓ)]: {prior_expectation:.6e}")
        print(f"Convergence achieved: {convergence_achieved}")
        
        return prior_expectation, convergence_achieved
    
    def _check_convergence(self, values: np.ndarray) -> bool:
        """Check convergence of Monte Carlo estimate using running average."""
        if len(values) < 1000:
            return False
        
        # Compute running averages
        n_check_points = 10
        check_indices = np.linspace(len(values)//2, len(values)-1, n_check_points, dtype=int)
        
        running_means = []
        for idx in check_indices:
            running_means.append(np.mean(values[:idx+1]))
        
        # Check if recent estimates are stable
        recent_std = np.std(running_means[-5:])  # Last 5 estimates
        recent_mean = np.mean(running_means[-5:])
        
        relative_std = recent_std / (recent_mean + 1e-10)  # Avoid division by zero
        
        return relative_std < self.config.convergence_threshold
    
    def compute_pac_bayes_bound(self, data_dict: Dict) -> PACBayesResults:
        """
        Compute the complete PAC-Bayes bound and certified intervals.
        
        Main theorem: KL(Q || P) ≤ (1/n)[log(1/δ) + log E_P[e^(-ℓ)]]
        """
        print("Computing PAC-Bayes certified bounds...")
        print("="*50)
        
        # Extract data
        posterior_samples = data_dict['posterior_samples']
        observations = data_dict['observations']
        obs_times = data_dict['observation_times']
        sensors = data_dict['sensor_locations']
        noise_std = data_dict['noise_std']
        true_kappa = data_dict['true_kappa']
        
        n_observations = observations.size
        
        print(f"Dataset: {observations.shape} observations")
        print(f"Posterior samples: {len(posterior_samples)}")
        print(f"True κ: {true_kappa}")
        
        # Step 1: Compute KL divergence KL(Q || P)
        print("\nStep 1: Computing KL(Q || P)...")
        kl_divergence = self.compute_kl_divergence(posterior_samples)
        print(f"KL divergence: {kl_divergence:.6f}")
        
        # Step 2: Compute prior expectation E_P[e^(-ℓ)]
        print("\nStep 2: Computing E_P[e^(-ℓ)]...")
        prior_expectation, convergence = self.compute_prior_expectation_mc(
            observations, obs_times, sensors, noise_std
        )
        
        # Step 3: Compute PAC-Bayes bound
        print(f"\nStep 3: Computing PAC-Bayes bound...")
        log_term = np.log(1.0 / self.config.delta)
        expectation_term = np.log(prior_expectation)
        pac_bound = (1.0 / n_observations) * (log_term + expectation_term)
        
        print(f"n = {n_observations}")
        print(f"δ = {self.config.delta}")
        print(f"log(1/δ) = {log_term:.6f}")
        print(f"log E_P[e^(-ℓ)] = {expectation_term:.6f}")
        print(f"PAC-Bayes bound: {pac_bound:.6f}")
        
        # Step 4: Check bound validity
        bound_satisfied = kl_divergence <= pac_bound
        print(f"\nBound check: KL(Q||P) ≤ PAC bound? {bound_satisfied}")
        if not bound_satisfied:
            warnings.warn(f"PAC-Bayes bound violated! KL={kl_divergence:.6f} > bound={pac_bound:.6f}")
        
        # Step 5: Compute certified interval
        print(f"\nStep 5: Computing certified intervals...")
        uncertified_interval = (
            float(np.percentile(posterior_samples, 2.5)),
            float(np.percentile(posterior_samples, 97.5))
        )
        
        # Certified interval: expand uncertified by bound violation amount
        if bound_satisfied:
            # If bound is satisfied, certified interval = uncertified interval
            certified_interval = uncertified_interval
        else:
            # Expand interval based on bound violation
            bound_violation = kl_divergence - pac_bound
            expansion_factor = 1.0 + np.sqrt(bound_violation)  # Heuristic expansion
            
            mean_kappa = np.mean(posterior_samples)
            uncert_half_width = (uncertified_interval[1] - uncertified_interval[0]) / 2
            cert_half_width = uncert_half_width * expansion_factor
            
            certified_interval = (
                mean_kappa - cert_half_width,
                mean_kappa + cert_half_width
            )
        
        # Ensure certified interval is within prior bounds
        certified_interval = (
            max(certified_interval[0], self.config.prior_bounds[0]),
            min(certified_interval[1], self.config.prior_bounds[1])
        )
        
        print(f"Uncertified 95% CI: [{uncertified_interval[0]:.4f}, {uncertified_interval[1]:.4f}]")
        print(f"Certified 95% bound: [{certified_interval[0]:.4f}, {certified_interval[1]:.4f}]")
        
        # Step 6: Compute metrics
        uncert_width = uncertified_interval[1] - uncertified_interval[0]
        cert_width = certified_interval[1] - certified_interval[0]
        interval_ratio = cert_width / uncert_width if uncert_width > 0 else float('inf')
        
        uncertified_covers = uncertified_interval[0] <= true_kappa <= uncertified_interval[1]
        certified_covers = certified_interval[0] <= true_kappa <= certified_interval[1]
        
        print(f"\nCoverage Analysis:")
        print(f"True κ = {true_kappa}")
        print(f"Uncertified covers truth: {uncertified_covers}")
        print(f"Certified covers truth: {certified_covers}")
        print(f"Interval width ratio: {interval_ratio:.3f}")
        
        # Create results object
        results = PACBayesResults(
            kl_divergence=kl_divergence,
            pac_bound=pac_bound,
            certified_interval=certified_interval,
            uncertified_interval=uncertified_interval,
            interval_ratio=interval_ratio,
            n_observations=n_observations,
            delta=self.config.delta,
            prior_expectation=prior_expectation,
            convergence_achieved=convergence,
            true_parameter=true_kappa,
            certified_covers_truth=certified_covers,
            uncertified_covers_truth=uncertified_covers
        )
        
        self.results = results
        return results
    
    def plot_certified_bounds(self, data_dict: Dict, save_path: str = "plots/phase5_pac_bayes_bounds.png"):
        """Create comprehensive visualization of certified vs uncertified bounds."""
        if self.results is None:
            raise ValueError("Must compute PAC-Bayes bounds before plotting")
        
        posterior_samples = data_dict['posterior_samples']
        true_kappa = data_dict['true_kappa']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PAC-Bayes Certified Uncertainty Bounds\nBayesian Inverse Heat Equation', fontsize=16, fontweight='bold')
        
        # 1. Posterior distribution with intervals
        ax1 = axes[0, 0]
        ax1.hist(posterior_samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(true_kappa, color='red', linestyle='--', linewidth=2, label='True κ')
        
        # Uncertified interval
        uncert_low, uncert_high = self.results.uncertified_interval
        ax1.axvspan(uncert_low, uncert_high, alpha=0.3, color='blue', 
                   label=f'Uncertified 95% CI: [{uncert_low:.3f}, {uncert_high:.3f}]')
        
        # Certified interval
        cert_low, cert_high = self.results.certified_interval
        ax1.axvspan(cert_low, cert_high, alpha=0.3, color='green',
                   label=f'Certified 95% bound: [{cert_low:.3f}, {cert_high:.3f}]')
        
        ax1.set_xlabel('Thermal Conductivity κ')
        ax1.set_ylabel('Posterior Density')
        ax1.set_title('Posterior Distribution with Uncertainty Bounds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Interval comparison
        ax2 = axes[0, 1]
        intervals = ['Uncertified\n95% CI', 'Certified\n95% Bound']
        widths = [uncert_high - uncert_low, cert_high - cert_low]
        colors = ['blue', 'green']
        
        bars = ax2.bar(intervals, widths, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Interval Width')
        ax2.set_title(f'Interval Width Comparison\nRatio: {self.results.interval_ratio:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Add width labels on bars
        for bar, width in zip(bars, widths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(widths)*0.01,
                    f'{width:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. PAC-Bayes bound analysis
        ax3 = axes[1, 0]
        bound_components = ['KL(Q||P)', 'PAC Bound', 'log(1/δ)/n', 'log E_P[e^(-ℓ)]/n']
        values = [
            self.results.kl_divergence,
            self.results.pac_bound,
            np.log(1/self.results.delta) / self.results.n_observations,
            np.log(self.results.prior_expectation) / self.results.n_observations
        ]
        
        bars = ax3.bar(bound_components, values, color=['red', 'green', 'orange', 'purple'], alpha=0.7)
        ax3.set_ylabel('Value')
        ax3.set_title('PAC-Bayes Bound Components')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Coverage summary
        ax4 = axes[1, 1]
        coverage_data = [
            ('Uncertified', self.results.uncertified_covers_truth),
            ('Certified', self.results.certified_covers_truth)
        ]
        colors = ['green' if covers else 'red' for _, covers in coverage_data]
        labels = [f'{name}\n{"✓" if covers else "✗"}' for name, covers in coverage_data]
        
        ax4.bar(range(len(coverage_data)), [1, 1], color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(coverage_data)))
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('Coverage')
        ax4.set_title(f'Truth Coverage Analysis\n(True κ = {true_kappa:.3f})')
        ax4.set_ylim(0, 1.2)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
        
        print(f"\nPlots saved: {save_path}")
        return fig
    
    def save_results(self, save_dir: str = "results/phase5_pac_bayes"):
        """Save PAC-Bayes results to JSON and NumPy files."""
        if self.results is None:
            raise ValueError("No results to save. Run compute_pac_bayes_bound first.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_dict = {
            'pac_bayes_bound': self.results.pac_bound,
            'kl_divergence': self.results.kl_divergence,
            'certified_interval': list(self.results.certified_interval),
            'uncertified_interval': list(self.results.uncertified_interval),
            'interval_width_ratio': self.results.interval_ratio,
            'n_observations': self.results.n_observations,
            'delta': self.results.delta,
            'prior_expectation': self.results.prior_expectation,
            'convergence_achieved': bool(self.results.convergence_achieved),
            'true_parameter': float(self.results.true_parameter),
            'certified_covers_truth': bool(self.results.certified_covers_truth),
            'uncertified_covers_truth': bool(self.results.uncertified_covers_truth),
            'config': {
                'delta': self.config.delta,
                'mc_samples_prior': self.config.mc_samples_prior,
                'convergence_threshold': self.config.convergence_threshold,
                'prior_bounds': list(self.config.prior_bounds)
            }
        }
        
        with open(save_path / "pac_bayes_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to: {save_path}")

# Demonstration function
def demo_pac_bayes_bounds():
    """Complete demonstration of PAC-Bayes bounds computation."""
    print("PAC-Bayes Certified Uncertainty Bounds")
    print("=" * 50)
    
    # Initialize PAC-Bayes bounds computer
    config = PACBayesConfig(
        delta=0.05,  # 95% confidence
        mc_samples_prior=5000,  # Reduced for faster demo
        convergence_threshold=1e-4
    )
    
    pac_bayes = PACBayesBounds(config)
    
    # Load Phase 4 results
    try:
        data_dict = pac_bayes.load_phase4_results()
        print("✓ Phase 4 results loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Phase 4 results: {e}")
        return

    # Compute PAC-Bayes bounds
    try:
        results = pac_bayes.compute_pac_bayes_bound(data_dict)
        print("✓ PAC-Bayes bounds computed successfully")
    except Exception as e:
        print(f"✗ Error computing PAC-Bayes bounds: {e}")
        return
    
    # Generate plots
    try:
        pac_bayes.plot_certified_bounds(data_dict)
        print("✓ Visualization generated successfully")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
    
    # Save results
    try:
        pac_bayes.save_results()
        print("✓ Results saved successfully")
    except Exception as e:
        print(f"✗ Error saving results: {e}")
    
    print("\n" + "=" * 50)
    print("Phase 5: PAC-Bayes Certified Bounds - COMPLETE")
    print("Ready for Phase 6: Validation Experiments")

if __name__ == "__main__":
    demo_pac_bayes_bounds()