"""
Phase 6: Validation Experiments - Complete Implementation
Reuses Phase 4 posterior and Phase 5 PAC-Bayes-kl methodology with importance reweighting
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.heat_solver import HeatEquationSolver

# Configuration
@dataclass
class Phase6Config:
    """Configuration for Phase 6 validation experiments"""
    # PAC-Bayes settings (same as Phase 5)
    c_parameter: float = 3.0
    delta: float = 0.05
    # Grid and sampling
    kappa_grid_points: int = 181
    target_subsample_min: int = 2000
    target_subsample_max: int = 5000
    max_subsample: int = 10000
    
    # ESS policy
    ess_instability_threshold: float = 30.0  # log-likelihood ratio std
    weight_clip_percentile: float = 95.0
    weight_clip_max_diff: float = 50.0
    tempering_alphas: List[float] = None
    
    # Pilot settings
    pilot_repeats: int = 3
    noise_levels: List[float] = None
    sensor_subsets: List[str] = None
    
    # Full run repeats
    noise_full_repeats: int = 20
    sensor_full_repeats: int = 10
    # Scalability
    scalability_nx_values: List[int] = None
    
    # Runtime
    global_time_budget: Optional[float] = None  # seconds
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.tempering_alphas is None:
            self.tempering_alphas = [0.5, 0.7, 0.85]
        if self.noise_levels is None:
            self.noise_levels = [0.05, 0.10, 0.20]
        if self.sensor_subsets is None:
            self.sensor_subsets = ['all', 'two', 'one']
        if self.scalability_nx_values is None:
            self.scalability_nx_values = [20, 50, 100, 200]

@dataclass
class ScenarioResult:
    """Results from a single scenario repeat"""
    scenario_id: str
    repeat: int
    uncertified_ci: Tuple[float, float]
    certified_interval: Tuple[float, float]
    uncertified_width: float
    certified_width: float
    efficiency: float  # certified_width / uncertified_width
    uncertified_covers: bool
    certified_covers: bool
    raw_ess: float
    final_ess: float
    reliability_tier: str
    mitigations_applied: List[str]
    runtime_seconds: float
    n_pde_solves: int

class Phase6ValidationExperiments:
    """Complete Phase 6 validation experiments implementation"""
    
    def __init__(self, config: Phase6Config = None):
        self.config = config or Phase6Config()
        np.random.seed(self.config.random_seed)
        
        # Cached forward predictions
        self.kappa_grid = None
        self.grid_predictions_cache = {}
        
        # Loaded data
        self.canonical_data = None
        self.phase4_posterior = None
        self.base_noise_std = None
        
    def load_phase4_data(self) -> Dict:
        """Load Phase 4 posterior and canonical dataset"""
        print("Loading Phase 4 posterior and canonical dataset...")
        # Load canonical dataset
        data_file = Path("data/canonical_dataset.npz")
        if not data_file.exists():
            raise FileNotFoundError(f"Canonical dataset not found: {data_file}")
        
        dataset = np.load(data_file, allow_pickle=False)
        
        # Load Phase 4 posterior samples
        phase4_results_dir = Path("results/phase4_production")
        samples_files = [
            phase4_results_dir / "posterior_samples.npy",
            phase4_results_dir / "mcmc_chains.npy"
        ]
        posterior_samples = None
        for samples_file in samples_files:
            if samples_file.exists():
                try:
                    samples = np.load(samples_file, allow_pickle=True)
                    if samples.ndim > 1:
                        posterior_samples = samples.flatten()
                    else:
                        posterior_samples = samples
                    break
                except Exception as e:
                    print(f"  Warning: Could not load {samples_file}: {e}")
                    continue
        
        if posterior_samples is None:
            raise FileNotFoundError("No posterior samples found in Phase 4 results")
        
        # Sort sensors and validate
        sensor_locs = dataset['sensor_locations']
        sorted_indices = np.argsort(sensor_locs)
        sensor_locs = sensor_locs[sorted_indices]
        observations = dataset['observations'][:, sorted_indices]
        
        canonical_data = {
            'observations': observations,
            'observation_times': dataset['observation_times'],
            'sensor_locations': sensor_locs,
            'noise_std': float(dataset['noise_std']),
            'true_kappa': float(dataset['true_kappa']),
            'posterior_samples': posterior_samples
        }
        
        self.canonical_data = canonical_data
        self.phase4_posterior = posterior_samples
        self.base_noise_std = canonical_data['noise_std']
        
        print(f"  Posterior samples: {len(posterior_samples):,}")
        print(f"  Canonical observations: {observations.shape}")
        print(f"  True κ: {canonical_data['true_kappa']}")
        print(f"  Sensors: {sensor_locs}")
        
        return canonical_data
    
    def cache_forward_predictions(self):
        """Cache forward predictions on κ grid"""
        print(f"Caching forward predictions on {self.config.kappa_grid_points}-point grid...")
        
        self.kappa_grid = np.linspace(1.0, 10.0, self.config.kappa_grid_points)
        solver = HeatEquationSolver(domain_length=1.0)
        
        def initial_condition(x):
            return np.exp(-50 * (x - 0.3)**2)
        
        obs_times = self.canonical_data['observation_times']
        sensor_locs = self.canonical_data['sensor_locations']
        
        n_successful = 0
        for i, kappa in enumerate(self.kappa_grid):
            try:
                solution, x_grid, t_grid = solver.solve(
                    kappa=kappa,
                    initial_condition=initial_condition,
                    boundary_conditions={'left': 0.0, 'right': 0.0},
                    nx=50,  # Same as Phase 5
                    final_time=obs_times[-1],
                    auto_timestep=True,
                    cfl_factor=0.25
                )
                
                # Correct time interpolation (same as Phase 5)
                predictions = np.zeros((len(obs_times), len(sensor_locs)))
                for j, t_obs in enumerate(obs_times):
                    k = np.searchsorted(t_grid, t_obs)
                    kL = max(0, k - 1)
                    kR = min(len(t_grid) - 1, k)
                    
                    if kL == kR:
                        time_slice = solution[kL, :]
                    else:
                        w = (t_obs - t_grid[kL]) / (t_grid[kR] - t_grid[kL])
                        time_slice = (1 - w) * solution[kL, :] + w * solution[kR, :]
                    
                    for l, x_sensor in enumerate(sensor_locs):
                        predictions[j, l] = np.interp(x_sensor, x_grid, time_slice)
                
                self.grid_predictions_cache[kappa] = predictions
                n_successful += 1
            except Exception as e:
                print(f"  Warning: Failed to solve for κ={kappa:.3f}: {e}")
                self.grid_predictions_cache[kappa] = None
        
        print(f"  Cached {n_successful}/{len(self.kappa_grid)} predictions")
        return n_successful
    
    def interpolate_predictions(self, kappa: float) -> Optional[np.ndarray]:
        """Interpolate predictions for arbitrary κ using cached grid"""
        if kappa <= self.kappa_grid[0]:
            return self.grid_predictions_cache[self.kappa_grid[0]]
        elif kappa >= self.kappa_grid[-1]:
            return self.grid_predictions_cache[self.kappa_grid[-1]]
        else:
            # Linear interpolation between nearest grid points
            idx = np.searchsorted(self.kappa_grid, kappa)
            kL, kR = idx - 1, idx
            kappa_L, kappa_R = self.kappa_grid[kL], self.kappa_grid[kR]
            pred_L, pred_R = self.grid_predictions_cache[kappa_L], self.grid_predictions_cache[kappa_R]
            
            if pred_L is not None and pred_R is not None:
                w = (kappa - kappa_L) / (kappa_R - kappa_L)
                return (1 - w) * pred_L + w * pred_R
            elif pred_L is not None:
                return pred_L
            elif pred_R is not None:
                return pred_R
            else:
                return None

    def compute_bounded_loss(self, observations: np.ndarray, predictions: np.ndarray, 
                           noise_std: float) -> float:
        """Compute bounded loss exactly as Phase 5"""
        if predictions is None:
            return 1.0
        
        residuals = observations - predictions
        squared_standardized = (residuals / (self.config.c_parameter * noise_std))**2
        bounded_losses = np.minimum(squared_standardized, 1.0)
        return np.mean(bounded_losses)

    def compute_importance_weights(self, scenario_obs: np.ndarray, scenario_noise_std: float,
                                 base_obs: np.ndarray, base_noise_std: float,
                                 kappa_samples: np.ndarray, 
                                 sensor_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
        """Compute importance weights with stability checking"""
        log_ratios = []
        
        for kappa in kappa_samples:
            predictions = self.interpolate_predictions(kappa)
            if predictions is None:
                log_ratios.append(-np.inf)
                continue
            
            # Apply sensor subset if specified
            if sensor_indices is not None:
                predictions = predictions[:, sensor_indices]
            
            # Scenario likelihood (log)
            scenario_residuals = scenario_obs - predictions
            scenario_logL = -0.5 * np.sum((scenario_residuals / scenario_noise_std)**2)
            
            # Base likelihood (log)
            base_residuals = base_obs - predictions
            base_logL = -0.5 * np.sum((base_residuals / base_noise_std)**2)
            
            log_ratios.append(scenario_logL - base_logL)
        
        log_ratios = np.array(log_ratios)
        
        # Stability check
        log_ratio_std = np.std(log_ratios[np.isfinite(log_ratios)])
        unstable = log_ratio_std > self.config.ess_instability_threshold
        
        if unstable:
            return None, {'unstable': True, 'log_ratio_std': log_ratio_std}
        
        # Convert to weights
        max_log_ratio = np.max(log_ratios[np.isfinite(log_ratios)])
        weights = np.exp(log_ratios - max_log_ratio)
        weights[~np.isfinite(weights)] = 0
        weights = weights / np.sum(weights)
        
        return weights, {'unstable': False, 'log_ratio_std': log_ratio_std}
    
    def apply_ess_mitigations(self, weights: np.ndarray) -> Tuple[np.ndarray, List[str], float]:
        """Apply ESS mitigations: clipping, tempering, expansion"""
        mitigations = []
        # Convert to log weights for clipping
        eps = 1e-12
        log_weights = np.log(np.clip(weights, eps, 1-eps))
        
        # Clipping
        clip_threshold = np.percentile(log_weights, self.config.weight_clip_percentile)
        max_allowed = np.max(log_weights) - self.config.weight_clip_max_diff
        clip_value = min(clip_threshold, max_allowed)
        
        if np.any(log_weights > clip_value):
            log_weights = np.minimum(log_weights, clip_value)
            mitigations.append('clip')

        # Convert back to weights and normalize
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        ess = 1.0 / np.sum(weights**2)
        
        # Tempering if needed
        for alpha in self.config.tempering_alphas:
            if ess < self.get_ess_threshold(len(weights), 'low')[0]:
                tempered_weights = weights ** alpha
                tempered_weights = tempered_weights / np.sum(tempered_weights)
                tempered_ess = 1.0 / np.sum(tempered_weights**2)
                
                if tempered_ess > ess:
                    weights = tempered_weights
                    ess = tempered_ess
                    mitigations.append(f'temper_{alpha}')
                    break

        return weights, mitigations, ess
    
    def get_ess_threshold(self, subsample_size: int, tier: str) -> Tuple[float, str]:
        """Get ESS threshold for reliability tier"""
        if tier == 'high':
            return max(1500, 0.25 * subsample_size), 'High'
        elif tier == 'medium':
            return max(800, 0.15 * subsample_size), 'Medium'
        elif tier == 'low':
            return max(300, 0.07 * subsample_size), 'Low'
        else:
            return 300, 'Unreliable'

    def determine_reliability_tier(self, ess: float, subsample_size: int) -> str:
        """Determine reliability tier from ESS"""
        high_thresh, _ = self.get_ess_threshold(subsample_size, 'high')
        med_thresh, _ = self.get_ess_threshold(subsample_size, 'medium')
        low_thresh, _ = self.get_ess_threshold(subsample_size, 'low')
        
        if ess >= high_thresh:
            return 'High'
        elif ess >= med_thresh:
            return 'Medium'
        elif ess >= low_thresh:
            return 'Low'
        else:
            return 'Unreliable'

    def compute_uncertified_ci(self, weights: np.ndarray, kappa_samples: np.ndarray) -> Tuple[float, float]:
        """Compute weighted 95% credible interval"""
        # Resample according to weights
        n_samples = len(weights)
        indices = np.random.choice(n_samples, size=min(n_samples, 5000), p=weights)
        resampled_kappa = kappa_samples[indices]
        return (np.percentile(resampled_kappa, 2.5), np.percentile(resampled_kappa, 97.5))
    
    def compute_pac_bayes_kl_bound(self, observations: np.ndarray, noise_std: float,
                                  weights: np.ndarray, kappa_samples: np.ndarray,
                                  sensor_indices: Optional[List[int]] = None) -> Tuple[float, Dict]:
        """Compute PAC-Bayes-kl certified bound exactly as Phase 5"""
        n_obs = observations.size
        
        # Compute empirical bounded risk
        bounded_risks = []
        for kappa in kappa_samples:
            predictions = self.interpolate_predictions(kappa)
            if sensor_indices is not None and predictions is not None:
                predictions = predictions[:, sensor_indices]
            risk = self.compute_bounded_loss(observations, predictions, noise_std)
            bounded_risks.append(risk)

        bounded_risks = np.array(bounded_risks)
        empirical_risk = np.sum(weights * bounded_risks)  # Weighted average
        
        # Compute KL(Q||P) from weighted samples
        # Use histogram approximation
        valid_mask = (kappa_samples >= 1.0) & (kappa_samples <= 10.0)
        if not np.any(valid_mask):
            return 1.0, {'error': 'no_valid_samples'}
        
        valid_kappa = kappa_samples[valid_mask]
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # Histogram-based KL
        n_bins = min(50, len(valid_kappa) // 20)
        if n_bins < 10:
            n_bins = 10

        hist, bin_edges = np.histogram(valid_kappa, bins=n_bins, weights=valid_weights, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        prior_density = 1.0 / 9.0  # Uniform[1,10]
        
        kl_div = 0.0
        eps = 1e-12
        for h in hist:
            if h > eps:
                kl_div += h * np.log(h / prior_density) * bin_width
        
        kl_div = max(kl_div, 0.0)
        
        # PAC-Bayes-kl inequality
        log_term = np.log(2 * np.sqrt(n_obs) / self.config.delta)
        rhs_bound = (kl_div + log_term) / n_obs

        # Invert binary-kl
        if rhs_bound <= 0:
            risk_upper_bound = empirical_risk
        else:
            risk_upper_bound = self.invert_kl_divergence(empirical_risk, rhs_bound)
        
        # Find certified κ-interval
        grid_risks = []
        for kappa in self.kappa_grid:
            predictions = self.grid_predictions_cache[kappa]
            if predictions is not None:
                if sensor_indices is not None:
                    predictions = predictions[:, sensor_indices]
                risk = self.compute_bounded_loss(observations, predictions, noise_std)
                grid_risks.append(risk)
            else:
                grid_risks.append(1.0)

        grid_risks = np.array(grid_risks)
        valid_indices = np.where(grid_risks <= risk_upper_bound)[0]
        
        if len(valid_indices) == 0:
            certified_interval = (1.0, 1.0)  # Empty
        else:
            # Largest contiguous interval around posterior mean
            posterior_mean = np.sum(weights * kappa_samples)
            mean_idx = np.argmin(np.abs(self.kappa_grid - posterior_mean))
            
            if mean_idx in valid_indices:
                left_idx = mean_idx
                right_idx = mean_idx

                while left_idx > 0 and (left_idx - 1) in valid_indices:
                    left_idx -= 1
                while right_idx < len(self.kappa_grid) - 1 and (right_idx + 1) in valid_indices:
                    right_idx += 1
                
                certified_interval = (self.kappa_grid[left_idx], self.kappa_grid[right_idx])
            else:
                certified_interval = (self.kappa_grid[valid_indices[0]], self.kappa_grid[valid_indices[-1]])
        
        components = {
            'empirical_risk': empirical_risk,
            'kl_divergence': kl_div,
            'log_term': log_term,
            'rhs_bound': rhs_bound,
            'risk_upper_bound': risk_upper_bound,
            'n_observations': n_obs
        }
        
        return certified_interval, components
    
    def kl_divergence_binary(self, p: float, q: float) -> float:
        """Binary KL divergence"""
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)
        q = np.clip(q, eps, 1-eps)

        if abs(p) < eps:
            return -np.log(1-q)
        elif abs(1-p) < eps:
            return -np.log(q)
        else:
            return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))
    
    def invert_kl_divergence(self, empirical_risk: float, rhs_bound: float, max_risk: float = 0.999) -> float:
        """Invert binary-kl divergence via Brent's method"""
        if empirical_risk >= max_risk:
            return max_risk

        def kl_equation(r):
            return self.kl_divergence_binary(empirical_risk, r) - rhs_bound
        
        try:
            lower = empirical_risk + 1e-10
            upper = max_risk
            
            # Check bracket
            f_lower = kl_equation(lower)
            f_upper = kl_equation(upper)
            
            if f_lower * f_upper > 0:
                # Expand bracket
                for _ in range(5):
                    upper = min(upper + 0.1, max_risk)
                    f_upper = kl_equation(upper)
                    if f_lower * f_upper <= 0:
                        break
                else:
                    return min(empirical_risk * 2.0, max_risk)

            result = brentq(kl_equation, lower, upper, xtol=1e-10)
            return min(result, max_risk)
            
        except (ValueError, RuntimeError):
            return min(empirical_risk * 2.0, max_risk)
    
    def run_single_scenario(self, scenario_obs: np.ndarray, scenario_noise_std: float,
                          scenario_id: str, repeat: int, sensor_indices: Optional[List[int]] = None) -> Optional[ScenarioResult]:
        """Run a single scenario repeat"""
        start_time = time.time()

        # Subsample posterior
        n_posterior = len(self.phase4_posterior)
        n_subsample = min(self.config.target_subsample_max, n_posterior)
        indices = np.random.choice(n_posterior, size=n_subsample, replace=False)
        kappa_samples = self.phase4_posterior[indices]
        
        # Compute importance weights
        base_obs = self.canonical_data['observations']
        if sensor_indices is not None:
            base_obs = base_obs[:, sensor_indices]
            
        weights, weight_info = self.compute_importance_weights(
            scenario_obs, scenario_noise_std,
            base_obs, self.base_noise_std,
            kappa_samples, sensor_indices
        )

        if weight_info['unstable']:
            print(f"  {scenario_id} repeat {repeat}: Unstable (log ratio std: {weight_info['log_ratio_std']:.1f})")
            return None
        
        raw_ess = 1.0 / np.sum(weights**2)
        
        # Apply mitigations
        weights, mitigations, final_ess = self.apply_ess_mitigations(weights)
        
        # Determine reliability
        reliability_tier = self.determine_reliability_tier(final_ess, len(weights))
        
        if reliability_tier == 'Unreliable':
            print(f"  {scenario_id} repeat {repeat}: Unreliable ESS ({final_ess:.0f})")
            return None
        
        # Compute intervals
        uncertified_ci = self.compute_uncertified_ci(weights, kappa_samples)
        certified_interval, pac_components = self.compute_pac_bayes_kl_bound(
            scenario_obs, scenario_noise_std, weights, kappa_samples, sensor_indices
        )
        
        # Metrics
        uncert_width = uncertified_ci[1] - uncertified_ci[0]
        cert_width = certified_interval[1] - certified_interval[0]
        efficiency = cert_width / uncert_width if uncert_width > 0 else float('inf')
        
        true_kappa = self.canonical_data['true_kappa']
        uncert_covers = uncertified_ci[0] <= true_kappa <= uncertified_ci[1]
        cert_covers = certified_interval[0] <= true_kappa <= certified_interval[1]
        
        runtime = time.time() - start_time
        
        return ScenarioResult(
            scenario_id=scenario_id,
            repeat=repeat,
            uncertified_ci=uncertified_ci,
            certified_interval=certified_interval,
            uncertified_width=uncert_width,
            certified_width=cert_width,
            efficiency=efficiency,
            uncertified_covers=uncert_covers,
            certified_covers=cert_covers,
            raw_ess=raw_ess,
            final_ess=final_ess,
            reliability_tier=reliability_tier,
            mitigations_applied=mitigations,
            runtime_seconds=runtime,
            n_pde_solves=0  # Using cached predictions
        )
    
    def run_pilot_study(self) -> Dict:
        """Run pilot study to determine adaptive parameters"""
        print("\nRunning pilot study (3 repeats per condition)...")
        pilot_results = {}

        # Test noise levels
        for noise_pct in self.config.noise_levels:
            print(f"  Pilot: {noise_pct*100:.0f}% noise...")
            results = []
            
            # Generate synthetic noisy data
            true_kappa = self.canonical_data['true_kappa']
            clean_predictions = self.interpolate_predictions(true_kappa)
            noise_std = noise_pct * np.std(clean_predictions)

            for repeat in range(self.config.pilot_repeats):
                noisy_obs = clean_predictions + np.random.normal(0, noise_std, clean_predictions.shape)
                result = self.run_single_scenario(noisy_obs, noise_std, f"noise_{noise_pct:.0%}", repeat)
                if result is not None:
                    results.append(result)
            
            pilot_results[f"noise_{noise_pct:.0%}"] = results
        
        # Test sensor subsets
        for subset_name in self.config.sensor_subsets:
            print(f"  Pilot: {subset_name} sensors...")
            results = []

            # Create sensor subset
            if subset_name == 'all':
                sensor_indices = [0, 1, 2]
            elif subset_name == 'two':
                sensor_indices = [1, 2]  # Drop smallest-x
            else:  # 'one'
                sensor_indices = [1]     # Central only
            
            subset_obs = self.canonical_data['observations'][:, sensor_indices]
            noise_std = 0.10 * np.std(subset_obs)

            for repeat in range(self.config.pilot_repeats):
                np.random.seed(self.config.random_seed + repeat)  # Fixed seed per subset
                noisy_obs = subset_obs + np.random.normal(0, noise_std, subset_obs.shape)
                result = self.run_single_scenario(noisy_obs, noise_std, f"sensors_{subset_name}", repeat, sensor_indices)
                if result is not None:
                    results.append(result)
            
            pilot_results[f"sensors_{subset_name}"] = results
        
        # Analyze pilot results
        pilot_analysis = {}
        for condition, results in pilot_results.items():
            if results:
                ess_values = [r.final_ess for r in results]
                reliability_tiers = [r.reliability_tier for r in results]
                
                pilot_analysis[condition] = {
                    'n_results': len(results),
                    'median_ess': np.median(ess_values),
                    'min_ess': np.min(ess_values),
                    'reliability_tiers': reliability_tiers,
                    'high_frac': sum(1 for t in reliability_tiers if t == 'High') / len(reliability_tiers)
                }
            else:
                pilot_analysis[condition] = {
                    'n_results': 0,
                    'median_ess': 0,
                    'min_ess': 0,
                    'reliability_tiers': [],
                    'high_frac': 0
                }
        
        print("  Pilot analysis complete.")
        return pilot_analysis
    
    def determine_repeats_from_pilot(self, pilot_analysis: Dict, condition_type: str) -> int:
        """Determine number of repeats based on pilot results"""
        if condition_type == 'noise':
            base_repeats = self.config.noise_full_repeats
        else:
            base_repeats = self.config.sensor_full_repeats
        
        # Average ESS across relevant conditions
        relevant_conditions = [k for k in pilot_analysis.keys() if condition_type in k]
        if not relevant_conditions:
            return base_repeats
        
        median_ess_values = [pilot_analysis[k]['median_ess'] for k in relevant_conditions]
        avg_median_ess = np.mean(median_ess_values)
        
        high_thresh, _ = self.get_ess_threshold(5000, 'high')  # Typical subsample size
        med_thresh, _ = self.get_ess_threshold(5000, 'medium')

        if avg_median_ess >= high_thresh:
            return base_repeats
        elif avg_median_ess >= med_thresh:
            return max(base_repeats // 2, 5)
        else:
            return 5
    
    def run_baseline_experiment(self) -> Dict:
        """Experiment 1: Baseline uncertified vs certified"""
        print("\nExperiment 1: Baseline (canonical dataset)")
        
        canonical_obs = self.canonical_data['observations']
        canonical_noise_std = self.canonical_data['noise_std']

        result = self.run_single_scenario(canonical_obs, canonical_noise_std, "baseline", 0)
        if result is None:
            print("  Baseline failed!")
            return {}
        
        print(f"  Uncertified CI: [{result.uncertified_ci[0]:.4f}, {result.uncertified_ci[1]:.4f}]")
        print(f"  Certified interval: [{result.certified_interval[0]:.4f}, {result.certified_interval[1]:.4f}]")
        print(f"  Coverage (uncert/cert): {result.uncertified_covers}/{result.certified_covers}")
        print(f"  Efficiency: {result.efficiency:.2f}x")

        print(f"  ESS: {result.final_ess:.0f} ({result.reliability_tier})")
        
        return {'baseline_result': result}
    
    def run_noise_experiment(self, pilot_analysis: Dict) -> Dict:
        """Experiment 2: Noise robustness"""
        print("\nExperiment 2: Noise robustness")
        
        n_repeats = self.determine_repeats_from_pilot(pilot_analysis, 'noise')
        print(f"  Using {n_repeats} repeats based on pilot analysis")
        
        noise_results = {}
        
        for noise_pct in self.config.noise_levels:
            print(f"  Testing {noise_pct*100:.0f}% noise...")
            results = []

            # Generate clean predictions at true kappa
            true_kappa = self.canonical_data['true_kappa']
            clean_predictions = self.interpolate_predictions(true_kappa)
            noise_std = noise_pct * np.std(clean_predictions)
            
            for repeat in range(n_repeats):
                np.random.seed(self.config.random_seed + repeat)
                noisy_obs = clean_predictions + np.random.normal(0, noise_std, clean_predictions.shape)
                
                result = self.run_single_scenario(noisy_obs, noise_std, f"noise_{noise_pct:.0%}", repeat)
                if result is not None:
                    results.append(result)

            noise_results[noise_pct] = results
            print(f"    Completed {len(results)}/{n_repeats} runs")
        
        return noise_results
    
    def run_sensor_experiment(self, pilot_analysis: Dict) -> Dict:
        """Experiment 3: Sparse sensor robustness"""
        print("\nExperiment 3: Sparse sensor robustness")
        
        n_repeats = self.determine_repeats_from_pilot(pilot_analysis, 'sensors')
        print(f"  Using {n_repeats} repeats based on pilot analysis")
        
        sensor_results = {}

        for subset_name in self.config.sensor_subsets:
            print(f"  Testing {subset_name} sensors...")
            results = []
            
            # Create sensor subset
            if subset_name == 'all':
                sensor_indices = [0, 1, 2]
            elif subset_name == 'two':
                sensor_indices = [1, 2]  # Drop smallest-x
            else:  # 'one'
                sensor_indices = [1]     # Central only
            
            subset_obs = self.canonical_data['observations'][:, sensor_indices]
            noise_std = 0.10 * np.std(subset_obs)

            for repeat in range(n_repeats):
                np.random.seed(self.config.random_seed + repeat)
                noisy_obs = subset_obs + np.random.normal(0, noise_std, subset_obs.shape)
                
                result = self.run_single_scenario(noisy_obs, noise_std, f"sensors_{subset_name}", repeat, sensor_indices)
                if result is not None:
                    results.append(result)
            
            sensor_results[subset_name] = results
            print(f"    Completed {len(results)}/{n_repeats} runs")
        
        return sensor_results

    def run_scalability_experiment(self) -> Dict:
        """Experiment 4: Forward solver scalability"""
        print("\nExperiment 4: Forward solver scalability")
        
        scalability_results = {}
        solver = HeatEquationSolver(domain_length=1.0)
        
        def initial_condition(x):
            return np.exp(-50 * (x - 0.3)**2)
        
        obs_times = self.canonical_data['observation_times']
        sensor_locs = self.canonical_data['sensor_locations']
        true_kappa = self.canonical_data['true_kappa']

        # Reference solution (highest resolution)
        ref_nx = max(self.config.scalability_nx_values)
        print(f"  Computing reference solution (nx={ref_nx})...")
        
        try:
            ref_solution, ref_x_grid, ref_t_grid = solver.solve(
                kappa=true_kappa,
                initial_condition=initial_condition,
                boundary_conditions={'left': 0.0, 'right': 0.0},
                nx=ref_nx,
                final_time=obs_times[-1],
                auto_timestep=True,
                cfl_factor=0.25
            )

            # Extract reference predictions
            ref_predictions = np.zeros((len(obs_times), len(sensor_locs)))
            for i, t_obs in enumerate(obs_times):
                k = np.searchsorted(ref_t_grid, t_obs)
                kL = max(0, k - 1)
                kR = min(len(ref_t_grid) - 1, k)
                
                if kL == kR:
                    time_slice = ref_solution[kL, :]
                else:
                    w = (t_obs - ref_t_grid[kL]) / (ref_t_grid[kR] - ref_t_grid[kL])
                    time_slice = (1 - w) * ref_solution[kL, :] + w * ref_solution[kR, :]
                
                for j, x_sensor in enumerate(sensor_locs):
                    ref_predictions[i, j] = np.interp(x_sensor, ref_x_grid, time_slice)
            
        except Exception as e:
            print(f"  Error computing reference: {e}")
            ref_predictions = None

        # Test different grid sizes
        for nx in self.config.scalability_nx_values:
            print(f"  Testing nx={nx}...")
            
            start_time = time.time()
            try:
                solution, x_grid, t_grid = solver.solve(
                    kappa=true_kappa,
                    initial_condition=initial_condition,
                    boundary_conditions={'left': 0.0, 'right': 0.0},
                    nx=nx,
                    final_time=obs_times[-1],
                    auto_timestep=True,
                    cfl_factor=0.25
                )

                # Extract predictions
                predictions = np.zeros((len(obs_times), len(sensor_locs)))
                for i, t_obs in enumerate(obs_times):
                    k = np.searchsorted(t_grid, t_obs)
                    kL = max(0, k - 1)
                    kR = min(len(t_grid) - 1, k)
                    
                    if kL == kR:
                        time_slice = solution[kL, :]
                    else:
                        w = (t_obs - t_grid[kL]) / (t_grid[kR] - t_grid[kL])
                        time_slice = (1 - w) * solution[kL, :] + w * solution[kR, :]
                    
                    for j, x_sensor in enumerate(sensor_locs):
                        predictions[i, j] = np.interp(x_sensor, x_grid, time_slice)
                
                runtime = time.time() - start_time
                
                # Compute RMSE vs reference
                if ref_predictions is not None and nx != ref_nx:
                    rmse = np.sqrt(np.mean((predictions - ref_predictions)**2))
                else:
                    rmse = 0.0  # Reference case
                
                scalability_results[nx] = {
                    'runtime': runtime,
                    'rmse': rmse,
                    'success': True
                }

                print(f"    Runtime: {runtime:.2f}s, RMSE: {rmse:.2e}")
                
            except Exception as e:
                print(f"    Failed: {e}")
                scalability_results[nx] = {
                    'runtime': float('inf'),
                    'rmse': float('inf'),
                    'success': False
                }
        
        return scalability_results

    def generate_plots(self, baseline_result: ScenarioResult, noise_results: Dict,
                      sensor_results: Dict, scalability_results: Dict) -> Dict:
        """Generate all validation plots"""
        plots = {}
        
        # Plot 1: Baseline comparison
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Posterior with intervals
        posterior_samples = self.phase4_posterior
        true_kappa = self.canonical_data['true_kappa']

        ax1.hist(posterior_samples, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='navy')
        ax1.axvline(true_kappa, color='red', linestyle='--', linewidth=2, label='True κ=5.0')
        ax1.axvspan(baseline_result.uncertified_ci[0], baseline_result.uncertified_ci[1], 
                   alpha=0.3, color='blue', label='Uncertified 95% CI')
        ax1.axvspan(baseline_result.certified_interval[0], baseline_result.certified_interval[1],
                   alpha=0.3, color='green', label='Certified interval')
        ax1.set_xlabel('Thermal Conductivity κ')
        ax1.set_ylabel('Posterior Density')
        ax1.set_title('Baseline: Uncertified vs Certified')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Width comparison
        widths = [baseline_result.uncertified_width, baseline_result.certified_width]
        labels = ['Uncertified', 'Certified']
        colors = ['blue', 'green']
        
        bars = ax2.bar(labels, widths, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Interval Width')
        ax2.set_title(f'Width Comparison (Efficiency: {baseline_result.efficiency:.2f}×)')
        ax2.grid(True, alpha=0.3)

        for bar, width in zip(bars, widths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(widths)*0.01,
                    f'{width:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plots['baseline'] = fig1
        
        # Plot 2: Noise robustness
        if noise_results:
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
            
            noise_levels = sorted(noise_results.keys())

            # Coverage vs noise
            uncert_coverage = []
            cert_coverage = []
            
            for noise_pct in noise_levels:
                results = noise_results[noise_pct]
                if results:
                    uncert_cov = sum(r.uncertified_covers for r in results) / len(results)
                    cert_cov = sum(r.certified_covers for r in results) / len(results)
                else:
                    uncert_cov = cert_cov = 0
                
                uncert_coverage.append(uncert_cov)
                cert_coverage.append(cert_cov)

            noise_pcts = [n*100 for n in noise_levels]
            axes2[0].plot(noise_pcts, uncert_coverage, 'bo-', label='Uncertified', linewidth=2)
            axes2[0].plot(noise_pcts, cert_coverage, 'go-', label='Certified', linewidth=2)
            axes2[0].axhline(0.95, color='red', linestyle='--', alpha=0.7, label='Target 95%')
            axes2[0].set_xlabel('Noise Level (%)')
            axes2[0].set_ylabel('Empirical Coverage')
            axes2[0].set_title('Coverage vs Noise Level')
            axes2[0].legend()
            axes2[0].grid(True, alpha=0.3)
            axes2[0].set_ylim(0, 1.1)

            # Width vs noise
            uncert_widths = []
            cert_widths = []
            
            for noise_pct in noise_levels:
                results = noise_results[noise_pct]
                if results:
                    uncert_w = np.mean([r.uncertified_width for r in results])
                    cert_w = np.mean([r.certified_width for r in results])
                else:
                    uncert_w = cert_w = 0
                
                uncert_widths.append(uncert_w)
                cert_widths.append(cert_w)

            axes2[1].plot(noise_pcts, uncert_widths, 'bo-', label='Uncertified', linewidth=2)
            axes2[1].plot(noise_pcts, cert_widths, 'go-', label='Certified', linewidth=2)
            axes2[1].set_xlabel('Noise Level (%)')
            axes2[1].set_ylabel('Average Width')
            axes2[1].set_title('Interval Width vs Noise')
            axes2[1].legend()
            axes2[1].grid(True, alpha=0.3)
            
            # Efficiency vs noise
            efficiencies = []
            for noise_pct in noise_levels:
                results = noise_results[noise_pct]
                if results:
                    eff = np.mean([r.efficiency for r in results if np.isfinite(r.efficiency)])
                else:
                    eff = 1.0
                efficiencies.append(eff)
            
            axes2[2].plot(noise_pcts, efficiencies, 'ro-', linewidth=2)

            axes2[2].axhline(1.0, color='black', linestyle='--', alpha=0.7, label='Equal width')
            axes2[2].set_xlabel('Noise Level (%)')
            axes2[2].set_ylabel('Efficiency (Cert/Uncert)')
            axes2[2].set_title('Efficiency vs Noise')
            axes2[2].legend()
            axes2[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['noise'] = fig2
        
        # Plot 3: Sensor robustness
        if sensor_results:
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

            sensor_names = ['all', 'two', 'one']
            sensor_counts = [3, 2, 1]
            
            # Coverage vs sensor count
            uncert_coverage = []
            cert_coverage = []
            
            for subset_name in sensor_names:
                if subset_name in sensor_results:
                    results = sensor_results[subset_name]
                    if results:
                        uncert_cov = sum(r.uncertified_covers for r in results) / len(results)
                        cert_cov = sum(r.certified_covers for r in results) / len(results)
                    else:
                        uncert_cov = cert_cov = 0
                else:
                    uncert_cov = cert_cov = 0
                
                uncert_coverage.append(uncert_cov)
                cert_coverage.append(cert_cov)

            axes3[0].plot(sensor_counts, uncert_coverage, 'bo-', label='Uncertified', linewidth=2)
            axes3[0].plot(sensor_counts, cert_coverage, 'go-', label='Certified', linewidth=2)
            axes3[0].axhline(0.95, color='red', linestyle='--', alpha=0.7, label='Target 95%')
            axes3[0].set_xlabel('Number of Sensors')
            axes3[0].set_ylabel('Empirical Coverage')
            axes3[0].set_title('Coverage vs Sensor Count')
            axes3[0].legend()
            axes3[0].grid(True, alpha=0.3)
            axes3[0].set_ylim(0, 1.1)
            axes3[0].set_xticks(sensor_counts)

            # Width vs sensor count
            uncert_widths = []
            cert_widths = []
            
            for subset_name in sensor_names:
                if subset_name in sensor_results:
                    results = sensor_results[subset_name]
                    if results:
                        uncert_w = np.mean([r.uncertified_width for r in results])
                        cert_w = np.mean([r.certified_width for r in results])
                    else:
                        uncert_w = cert_w = 0
                else:
                    uncert_w = cert_w = 0

                uncert_widths.append(uncert_w)
                cert_widths.append(cert_w)
            
            axes3[1].plot(sensor_counts, uncert_widths, 'bo-', label='Uncertified', linewidth=2)
            axes3[1].plot(sensor_counts, cert_widths, 'go-', label='Certified', linewidth=2)
            axes3[1].set_xlabel('Number of Sensors')
            axes3[1].set_ylabel('Average Width')

            axes3[1].set_title('Interval Width vs Sensor Count')
            axes3[1].legend()
            axes3[1].grid(True, alpha=0.3)
            axes3[1].set_xticks(sensor_counts)
            
            plt.tight_layout()
            plots['sensors'] = fig3
        
        # Plot 4: Scalability
        if scalability_results:
            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
            
            nx_values = sorted([nx for nx in scalability_results.keys() if scalability_results[nx]['success']])
            runtimes = [scalability_results[nx]['runtime'] for nx in nx_values]
            rmses = [scalability_results[nx]['rmse'] for nx in nx_values]
            
            # Runtime vs nx
            ax4a.plot(nx_values, runtimes, 'bo-', linewidth=2, markersize=8)

            ax4a.set_xlabel('Grid Size (nx)')
            ax4a.set_ylabel('Runtime (seconds)')
            ax4a.set_title('Solver Runtime vs Grid Size')
            ax4a.grid(True, alpha=0.3)
            ax4a.set_xscale('log')
            ax4a.set_yscale('log')
            
            # RMSE vs nx
            if any(r > 0 for r in rmses):

                ax4b.plot(nx_values[:-1], rmses[:-1], 'ro-', linewidth=2, markersize=8)
                ax4b.set_xlabel('Grid Size (nx)')
                ax4b.set_ylabel('RMSE vs Reference')
                ax4b.set_title('Accuracy vs Grid Size')
                ax4b.grid(True, alpha=0.3)
                ax4b.set_xscale('log')
                ax4b.set_yscale('log')
            
            plt.tight_layout()
            plots['scalability'] = fig4
        
        return plots

    def generate_summary(self, baseline_result: ScenarioResult, noise_results: Dict,
                        sensor_results: Dict, pilot_analysis: Dict) -> str:
        """Generate compact text summary for paper"""
        summary_lines = []
        
        # Baseline
        summary_lines.append("PHASE 6 VALIDATION SUMMARY")
        summary_lines.append("=" * 40)
        
        # Baseline results

        summary_lines.append(f"Baseline (canonical dataset):")
        summary_lines.append(f"  Uncertified 95% CI: [{baseline_result.uncertified_ci[0]:.4f}, {baseline_result.uncertified_ci[1]:.4f}] (covers: {baseline_result.uncertified_covers})")
        summary_lines.append(f"  Certified interval: [{baseline_result.certified_interval[0]:.4f}, {baseline_result.certified_interval[1]:.4f}] (covers: {baseline_result.certified_covers})")
        summary_lines.append(f"  Width increase: {baseline_result.efficiency:.2f}×")
        
        # Noise robustness
        if noise_results:

            total_uncert = sum(sum(r.uncertified_covers for r in results) for results in noise_results.values())
            total_cert = sum(sum(r.certified_covers for r in results) for results in noise_results.values())
            total_runs = sum(len(results) for results in noise_results.values())
            
            summary_lines.append(f"\nNoise robustness ({total_runs} total runs):")

            summary_lines.append(f"  Uncertified coverage: {total_uncert}/{total_runs} ({100*total_uncert/total_runs:.1f}%)")
            summary_lines.append(f"  Certified coverage: {total_cert}/{total_runs} ({100*total_cert/total_runs:.1f}%)")
            
            avg_efficiency = np.mean([r.efficiency for results in noise_results.values() 
                                    for r in results if np.isfinite(r.efficiency)])
            summary_lines.append(f"  Average width increase: {avg_efficiency:.2f}×")

        # Sensor robustness
        if sensor_results:
            total_uncert = sum(sum(r.uncertified_covers for r in results) for results in sensor_results.values())
            total_cert = sum(sum(r.certified_covers for r in results) for results in sensor_results.values())
            total_runs = sum(len(results) for results in sensor_results.values())
            
            summary_lines.append(f"\nSensor sparsity ({total_runs} total runs):")
            summary_lines.append(f"  Uncertified coverage: {total_uncert}/{total_runs} ({100*total_uncert/total_runs:.1f}%)")
            summary_lines.append(f"  Certified coverage: {total_cert}/{total_runs} ({100*total_cert/total_runs:.1f}%)")

        # Reliability tiers
        all_results = [baseline_result]
        for results in noise_results.values():
            all_results.extend(results)
        for results in sensor_results.values():
            all_results.extend(results)
        
        tier_counts = {}
        for result in all_results:
            tier = result.reliability_tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        total_results = len(all_results)
        summary_lines.append(f"\nReliability tiers ({total_results} results):")
        for tier in ['High', 'Medium', 'Low', 'Unreliable']:
            if tier in tier_counts:
                count = tier_counts[tier]
                pct = 100 * count / total_results
                summary_lines.append(f"  {tier}: {count} ({pct:.1f}%)")
        
        return "\n".join(summary_lines)

    def save_results(self, all_results: Dict, output_dir: Optional[str] = None):
        """Save results to JSON and plots"""
        if output_dir is None:
            output_dir = "results/phase6_validation"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Custom JSON encoder for numpy types
        class NumpyJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Prepare serializable results
        serializable_results = {}

        for exp_name, exp_results in all_results.items():
            if exp_name == 'plots':
                continue
            elif exp_name == 'baseline_result':
                serializable_results[exp_name] = asdict(exp_results)
            elif exp_name in ['noise_results', 'sensor_results']:
                serializable_results[exp_name] = {}
                for condition, results_list in exp_results.items():

                    serializable_results[exp_name][condition] = [asdict(r) for r in results_list]
            else:
                serializable_results[exp_name] = exp_results
        
        # Save JSON
        with open(output_path / "phase6_validation_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyJSONEncoder)
        
        # Save plots
        if 'plots' in all_results:
            plots_dir = output_path / "plots"
            plots_dir.mkdir(exist_ok=True)

            for plot_name, fig in all_results['plots'].items():
                fig.savefig(plots_dir / f"{plot_name}.png", dpi=300, bbox_inches='tight')
                fig.savefig(plots_dir / f"{plot_name}.svg", bbox_inches='tight')
        
        print(f"\nResults saved to: {output_path}")
    
    def run_all_experiments(self, output_dir: Optional[str] = None) -> Dict:
        """Main entry point: run all Phase 6 validation experiments"""
        print("PHASE 6 VALIDATION EXPERIMENTS")
        print("=" * 50)

        start_time = time.time()
        
        # Load data and cache
        self.load_phase4_data()
        n_cached = self.cache_forward_predictions()
        
        if n_cached < 100:
            print(f"Warning: Only {n_cached} cached predictions, results may be unreliable")
        
        # Run pilot study
        pilot_analysis = self.run_pilot_study()

        # Run main experiments
        baseline_result = self.run_baseline_experiment()['baseline_result']
        noise_results = self.run_noise_experiment(pilot_analysis)
        sensor_results = self.run_sensor_experiment(pilot_analysis)
        scalability_results = self.run_scalability_experiment()
        
        # Generate plots
        plots = self.generate_plots(baseline_result, noise_results, sensor_results, scalability_results)
        
        # Generate summary

        summary = self.generate_summary(baseline_result, noise_results, sensor_results, pilot_analysis)
        
        total_runtime = time.time() - start_time
        
        # Compile results
        all_results = {
            'baseline_result': baseline_result,
            'noise_results': noise_results,
            'sensor_results': sensor_results,
            'scalability_results': scalability_results,
            'pilot_analysis': pilot_analysis,
            'plots': plots,
            'summary': summary,
            'total_runtime_minutes': total_runtime / 60,
            'config': asdict(self.config)
        }

        # Save results
        if output_dir is not None:
            self.save_results(all_results, output_dir)
        
        # Print summary
        print(f"\n{summary}")
        print(f"\nTotal runtime: {total_runtime/60:.1f} minutes")
        print(f"Phase 6 validation experiments complete!")
        
        return all_results

def demo_phase6_validation():
    """Demo function to run Phase 6 validation"""
    config = Phase6Config(

        # Reduced parameters for demo
        noise_full_repeats=10,
        sensor_full_repeats=5,
        target_subsample_max=3000
    )
    
    validator = Phase6ValidationExperiments(config)
    results = validator.run_all_experiments("results/phase6_validation")
    
    return results

if __name__ == "__main__":
    demo_phase6_validation()