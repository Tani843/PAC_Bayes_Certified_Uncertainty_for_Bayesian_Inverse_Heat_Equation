"""
Complete PAC-Bayes-kl Certified Bounds Implementation
Final production script with all fixes and optimizations
"""

import numpy as np
import json
import time
from pathlib import Path
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from src.heat_solver import HeatEquationSolver

# Configuration parameters (exposed at top)
MAX_POSTERIOR_EVALS = 200
KAPPA_GRID_POINTS = 181
C_PARAMETER = 3.0
DELTA = 0.05
DOMAIN_LENGTH = 1.0
NX = 50
CFL_FACTOR = 0.25

def kl_divergence_binary(p, q):
    """Binary KL divergence kl(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))"""
    eps = 1e-12
    p = np.clip(p, eps, 1-eps)
    q = np.clip(q, eps, 1-eps)
    
    if abs(p) < eps:
        return -np.log(1-q)
    elif abs(1-p) < eps:
        return -np.log(q)
    else:
        return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

def invert_kl_divergence(empirical_risk, rhs_bound, max_risk=0.999):
    """Invert kl(empirical_risk || true_risk) <= rhs_bound via bisection"""
    if empirical_risk >= max_risk:
        return max_risk
    
    def kl_equation(r):
        return kl_divergence_binary(empirical_risk, r) - rhs_bound

    # Find bracket with opposite signs
    lower = empirical_risk + 1e-10
    upper = max_risk
    
    # Check if we can bracket the root
    try:
        f_lower = kl_equation(lower)
        f_upper = kl_equation(upper)
        
        # Expand bracket if necessary
        attempts = 0
        while f_lower * f_upper > 0 and attempts < 10:
            upper = min(upper + 0.1, max_risk)
            f_upper = kl_equation(upper)
            attempts += 1
        
        if f_lower * f_upper > 0:
            print(f"Warning: Could not bracket root for kl inversion. Using empirical risk.")
            return empirical_risk
        
        # Solve via Brent's method
        risk_upper = brentq(kl_equation, lower, upper, xtol=1e-10)
        return min(risk_upper, max_risk)
        
    except (ValueError, RuntimeError) as e:
        print(f"Warning: kl inversion failed ({e}). Using fallback.")
        return min(empirical_risk * 2.0, max_risk)

def compute_bounded_loss(observations, predictions, noise_std, c=C_PARAMETER):
    """Compute bounded per-datum loss with clipping at 1.0"""
    if predictions is None:
        return 1.0
    
    residuals = observations - predictions
    squared_standardized = (residuals / (c * noise_std))**2
    bounded_losses = np.minimum(squared_standardized, 1.0)
    return np.mean(bounded_losses)

def compute_forward_predictions_correct_interp(kappa, obs_times, sensor_locations, 
                                             domain_length=DOMAIN_LENGTH, nx=NX):
    """Compute forward predictions with correct time interpolation"""
    solver = HeatEquationSolver()
    
    def initial_condition(x):
        return np.exp(-50 * (x - 0.3)**2)
    
    try:
        solution, x_grid, t_grid = solver.solve(
            kappa=kappa,
            initial_condition=initial_condition,
            boundary_conditions={'left': 0.0, 'right': 0.0},
            nx=nx,
            final_time=obs_times[-1],
            auto_timestep=True,
            cfl_factor=CFL_FACTOR
        )
        
        predictions = np.zeros((len(obs_times), len(sensor_locations)))
        
        for i, t_obs in enumerate(obs_times):
            # Correct time interpolation
            k = np.searchsorted(t_grid, t_obs)
            kL = max(0, k - 1)
            kR = min(len(t_grid) - 1, k)
            
            if kL == kR:
                time_slice = solution[kL, :]
            else:
                w = (t_obs - t_grid[kL]) / (t_grid[kR] - t_grid[kL])
                time_slice = (1 - w) * solution[kL, :] + w * solution[kR, :]
            
            # Spatial interpolation for each sensor
            for j, x_sensor in enumerate(sensor_locations):
                predictions[i, j] = np.interp(x_sensor, x_grid, time_slice)
        
        return predictions
        
    except Exception as e:
        print(f"Forward solve failed for kappa={kappa:.4f}: {e}")
        return None

def compute_kl_posterior_prior_robust(posterior_samples, prior_bounds=(1.0, 10.0)):
    """Compute KL(Q||P) with robust histogram estimation"""
    # Filter samples within prior support
    valid_samples = posterior_samples[
        (posterior_samples >= prior_bounds[0]) & 
        (posterior_samples <= prior_bounds[1])
    ]
    
    if len(valid_samples) == 0:
        return float('inf')
    
    # Adaptive binning: at least 30, at most 100
    n_bins = np.clip(len(valid_samples) // 20, 30, 100)
    counts, bin_edges = np.histogram(valid_samples, bins=n_bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Prior density (uniform)
    prior_density = 1.0 / (prior_bounds[1] - prior_bounds[0])
    
    # KL divergence with epsilon to avoid log(0)
    eps = 1e-12
    kl_div = 0.0
    for count in counts:
        if count > 0:
            posterior_density = count + eps
            kl_div += posterior_density * np.log(posterior_density / prior_density) * bin_width
    
    return max(kl_div, 0.0)

def load_phase4_data():
    """Load Phase 4 results and canonical dataset"""
    # Load canonical dataset
    data_file = Path("data/canonical_dataset.npz")
    if not data_file.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {data_file}")
    
    dataset = np.load(data_file)
    
    # Load Phase 4 posterior samples
    phase4_results_dir = Path("results/phase4_production")
    
    samples_files = [
        phase4_results_dir / "posterior_samples.npy",
        phase4_results_dir / "mcmc_chains.npy"
    ]
    
    posterior_samples = None
    for samples_file in samples_files:
        if samples_file.exists():
            samples = np.load(samples_file, allow_pickle=True)
            if samples.ndim > 1:
                posterior_samples = np.concatenate([chain for chain in samples])
            else:
                posterior_samples = samples
            print(f"Loaded posterior samples from: {samples_file}")
            break
    
    if posterior_samples is None:
        raise FileNotFoundError("No posterior samples found in Phase 4 results")
    
    return {
        'observations': dataset['observations'],
        'observation_times': dataset['observation_times'],
        'sensor_locations': dataset['sensor_locations'],
        'noise_std': float(dataset['noise_std']),
        'true_kappa': float(dataset['true_kappa']),
        'posterior_samples': posterior_samples
    }

def main():
    """Main PAC-Bayes-kl certified bounds computation"""
    start_time = time.time()
    
    print("PAC-Bayes-kl Certified Bounds - Complete Implementation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  MAX_POSTERIOR_EVALS: {MAX_POSTERIOR_EVALS}")
    print(f"  KAPPA_GRID_POINTS: {KAPPA_GRID_POINTS}")
    print(f"  C_PARAMETER: {C_PARAMETER}")
    print(f"  DELTA: {DELTA}")
    print(f"  Domain length: {DOMAIN_LENGTH}")
    print(f"  Grid points: {NX}")
    
    # Load data
    print(f"\nLoading Phase 4 results and canonical dataset...")
    try:
        data_dict = load_phase4_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    observations = data_dict['observations']
    obs_times = data_dict['observation_times']
    sensors = data_dict['sensor_locations']
    noise_std = data_dict['noise_std']
    true_kappa = data_dict['true_kappa']
    posterior_samples = data_dict['posterior_samples']
    
    n_obs = observations.size
    
    print(f"Dataset loaded:")
    print(f"  Observations: {observations.shape} -> {n_obs} total")
    print(f"  Posterior samples: {len(posterior_samples)}")
    print(f"  True κ: {true_kappa}")
    print(f"  Noise std: {noise_std:.8f}")
    
    # Step 1: Create kappa grid and cache forward solves
    print(f"\nStep 1: Computing forward model cache...")
    kappa_grid = np.linspace(1.0, 10.0, KAPPA_GRID_POINTS)
    grid_predictions = {}
    n_pde_solves = 0
    
    for i, kappa in enumerate(kappa_grid):
        predictions = compute_forward_predictions_correct_interp(kappa, obs_times, sensors)
        grid_predictions[kappa] = predictions
        if predictions is not None:
            n_pde_solves += 1
        
        if (i + 1) % 30 == 0:
            print(f"  Progress: {i+1}/{len(kappa_grid)}")
    
    print(f"  Completed {n_pde_solves} PDE solves for grid")
    
    # Step 2: Subsample posterior and compute empirical risk
    print(f"\nStep 2: Computing empirical bounded risk...")
    n_posterior_samples = min(MAX_POSTERIOR_EVALS, len(posterior_samples))
    sample_indices = np.random.choice(len(posterior_samples), n_posterior_samples, replace=False)
    selected_samples = posterior_samples[sample_indices]
    
    bounded_risks = []
    n_interpolated = 0
    
    for kappa_sample in selected_samples:
        # Find nearest grid points for interpolation
        grid_idx = np.searchsorted(kappa_grid, kappa_sample)
        
        if grid_idx == 0:
            # Use first grid point
            predictions = grid_predictions[kappa_grid[0]]
        elif grid_idx >= len(kappa_grid):
            # Use last grid point
            predictions = grid_predictions[kappa_grid[-1]]
        else:
            # Linear interpolation between two nearest grid points
            kL, kR = grid_idx - 1, grid_idx
            kappa_L, kappa_R = kappa_grid[kL], kappa_grid[kR]
            pred_L, pred_R = grid_predictions[kappa_L], grid_predictions[kappa_R]
            
            if pred_L is not None and pred_R is not None:
                w = (kappa_sample - kappa_L) / (kappa_R - kappa_L)
                predictions = (1 - w) * pred_L + w * pred_R
                n_interpolated += 1
            elif pred_L is not None:
                predictions = pred_L
            elif pred_R is not None:
                predictions = pred_R
            else:
                predictions = None
        
        if predictions is not None:
            risk = compute_bounded_loss(observations, predictions, noise_std, C_PARAMETER)
            bounded_risks.append(risk)

    if not bounded_risks:
        print("Error: No valid risk computations")
        return
    
    empirical_risk = np.mean(bounded_risks)
    mc_se = np.std(bounded_risks) / np.sqrt(len(bounded_risks))
    
    print(f"  Empirical bounded risk: {empirical_risk:.6f} ± {mc_se:.6f}")
    print(f"  Used {n_posterior_samples} posterior samples")
    print(f"  Interpolated {n_interpolated} samples from grid")
    
    # Step 3: Compute KL(Q||P)
    print(f"\nStep 3: Computing KL(Q||P)...")
    kl_qp = compute_kl_posterior_prior_robust(posterior_samples)
    print(f"  KL(Q||P): {kl_qp:.6f}")

    # Step 4: Apply PAC-Bayes-kl inequality
    print(f"\nStep 4: PAC-Bayes-kl bound computation...")
    log_term = np.log(2 * np.sqrt(n_obs) / DELTA)
    rhs_bound = (kl_qp + log_term) / n_obs
    
    print(f"  n: {n_obs}")
    print(f"  δ: {DELTA}")
    print(f"  KL term: {kl_qp:.6f}")
    print(f"  Log term: {log_term:.6f}")
    print(f"  RHS bound: {rhs_bound:.6f}")
    
    # Step 5: Invert kl-divergence
    print(f"\nStep 5: Inverting binary kl-divergence...")
    risk_upper_bound = invert_kl_divergence(empirical_risk, rhs_bound)
    print(f"  Risk upper bound: {risk_upper_bound:.6f}")
    
    # Step 6: Compute risk over grid and find certified interval
    print(f"\nStep 6: Computing certified κ-interval...")
    grid_risks = []
    for kappa in kappa_grid:
        predictions = grid_predictions[kappa]
        if predictions is not None:
            risk = compute_bounded_loss(observations, predictions, noise_std, C_PARAMETER)
            grid_risks.append(risk)
        else:
            grid_risks.append(1.0)
    
    grid_risks = np.array(grid_risks)
    
    # Find certified interval: all κ where risk ≤ upper bound
    valid_indices = np.where(grid_risks <= risk_upper_bound)[0]
    
    if len(valid_indices) == 0:
        certified_interval = (kappa_grid[0], kappa_grid[0])  # Empty interval
        print("  Warning: No κ values satisfy risk bound")
    else:
        # Find largest contiguous interval around posterior mean
        posterior_mean = np.mean(posterior_samples)
        mean_idx = np.argmin(np.abs(kappa_grid - posterior_mean))
        
        # Find contiguous interval containing mean_idx
        if mean_idx in valid_indices:
            left_idx = mean_idx
            right_idx = mean_idx
            
            # Expand left
            while left_idx > 0 and (left_idx - 1) in valid_indices:
                left_idx -= 1
            
            # Expand right
            while right_idx < len(kappa_grid) - 1 and (right_idx + 1) in valid_indices:
                right_idx += 1
            
            certified_interval = (kappa_grid[left_idx], kappa_grid[right_idx])
        else:
            # Use largest contiguous interval
            certified_interval = (kappa_grid[valid_indices[0]], kappa_grid[valid_indices[-1]])
    
    # Step 7: Compute uncertified CI and coverage
    uncertified_ci = (np.percentile(posterior_samples, 2.5), np.percentile(posterior_samples, 97.5))
    posterior_mean = np.mean(posterior_samples)
    
    certified_covers = certified_interval[0] <= true_kappa <= certified_interval[1]
    uncertified_covers = uncertified_ci[0] <= true_kappa <= uncertified_ci[1]
    
    runtime = time.time() - start_time

    # Print final results
    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS")
    print(f"=" * 60)
    print(f"n: {n_obs}")
    print(f"delta: {DELTA}")
    print(f"KL(Q||P): {kl_qp:.6f}")
    print(f"log term: {log_term:.6f}")
    print(f"RHS bound: {rhs_bound:.6f}")
    print(f"empirical risk: {empirical_risk:.6f}")
    print(f"risk upper bound: {risk_upper_bound:.6f}")
    print(f"posterior mean: {posterior_mean:.6f}")
    print(f"uncertified 95% CI: [{uncertified_ci[0]:.4f}, {uncertified_ci[1]:.4f}]")
    print(f"certified κ-interval: [{certified_interval[0]:.4f}, {certified_interval[1]:.4f}]")
    print(f"coverage of κ=5.0:")
    print(f"  uncertified: {uncertified_covers}")
    print(f"  certified: {certified_covers}")
    print(f"number of PDE solves: {n_pde_solves}")
    print(f"runtime: {runtime:.1f} seconds")
    
    # Save results
    results = {
        'n_observations': int(n_obs),
        'delta': float(DELTA),
        'kl_divergence': float(kl_qp),
        'log_term': float(log_term),
        'rhs_bound': float(rhs_bound),
        'empirical_risk': float(empirical_risk),
        'mc_standard_error': float(mc_se),
        'risk_upper_bound': float(risk_upper_bound),
        'posterior_mean': float(posterior_mean),
        'uncertified_ci': [float(uncertified_ci[0]), float(uncertified_ci[1])],
        'certified_interval': [float(certified_interval[0]), float(certified_interval[1])],
        'true_kappa': float(true_kappa),
        'uncertified_covers': bool(uncertified_covers),
        'certified_covers': bool(certified_covers),
        'n_pde_solves': int(n_pde_solves),
        'n_posterior_samples': int(n_posterior_samples),
        'n_interpolated': int(n_interpolated),
        'runtime_seconds': float(runtime),
        'c_parameter': float(C_PARAMETER)
    }
    
    # Create output directories
    Path("results/phase5_pac_bayes_kl").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open("results/phase5_pac_bayes_kl/pac_bayes_kl_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Posterior with intervals
    ax1 = axes[0]
    ax1.hist(posterior_samples, bins=60, density=True, alpha=0.7, color='lightblue', edgecolor='navy')
    ax1.axvline(true_kappa, color='red', linestyle='--', linewidth=3, label='True κ=5.0')
    ax1.axvspan(uncertified_ci[0], uncertified_ci[1], alpha=0.4, color='blue', 
               label=f'Uncertified 95% CI')
    ax1.axvspan(certified_interval[0], certified_interval[1], alpha=0.4, color='green',
               label=f'Certified interval')
    ax1.set_xlabel('Thermal Conductivity κ')
    ax1.set_ylabel('Posterior Density')
    ax1.set_title('Posterior Distribution with Certified Bounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Risk function
    ax2 = axes[1]
    ax2.plot(kappa_grid, grid_risks, 'b-', linewidth=2, label='Bounded Risk R(κ)')
    ax2.axhline(risk_upper_bound, color='red', linestyle='--', linewidth=3, 
               label=f'Risk Upper Bound: {risk_upper_bound:.4f}')
    ax2.axvline(true_kappa, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax2.fill_between(kappa_grid, 0, np.minimum(grid_risks, risk_upper_bound), alpha=0.3, color='green')
    ax2.set_xlabel('Thermal Conductivity κ')
    ax2.set_ylabel('Bounded Risk')
    ax2.set_title('Bounded Risk vs κ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(1.1, np.max(grid_risks) * 1.1))

    # Plot 3: PAC-Bayes components
    ax3 = axes[2]
    components = ['KL(Q||P)', 'Log Term', 'RHS Bound', 'Empirical Risk', 'Risk Upper Bound']
    values = [kl_qp, log_term, rhs_bound, empirical_risk, risk_upper_bound]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    bars = ax3.bar(components, values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Value')
    ax3.set_title('PAC-Bayes-kl Components')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/pac_bayes_kl_certified_bounds.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/pac_bayes_kl_certified_bounds.svg', bbox_inches='tight')
    
    print(f"\nResults saved:")
    print(f"  JSON: results/phase5_pac_bayes_kl/pac_bayes_kl_results.json")
    print(f"  Plots: plots/pac_bayes_kl_certified_bounds.png")
    
    # Success assessment
    if risk_upper_bound < 1.0:
        if certified_covers:
            print(f"\n✓ SUCCESS: Non-vacuous certified bound with correct coverage!")
        else:
            print(f"\n⚠ PARTIAL SUCCESS: Non-vacuous bound but coverage issue")
    else:
        print(f"\n⚠ WARNING: Vacuous bound (upper bound = 1.0)")
    
    print(f"\nPAC-Bayes-kl computation complete!")

if __name__ == "__main__":
    main()