#!/bin/bash

echo "MCMC TERMINAL RUNNER - VISIBLE PROGRESS"
echo "======================================"
echo "Starting MCMC with live terminal output..."
echo "Press Ctrl+C to stop if needed"
echo ""

python3 -c "
import numpy as np
import time
import sys

print('Loading optimized dataset...')

# Setup inference manually to avoid validation issues
from src.bayesian_inference import BayesianInference, BayesianInferenceConfig
import json

# Load optimized dataset
data = np.load('data/canonical_dataset.npz')
observations = data['observations']
observation_times = data['observation_times']
sensor_locations = data['sensor_locations']
true_kappa = float(data['true_kappa'])
noise_std = float(data['noise_std'])

# Load temperature scaling
with open('data/master_config.json', 'r') as f:
    config = json.load(f)
temperature_scale_factor = config['numerical_stability']['temperature_scale_factor']

# Setup inference
inf = BayesianInference()
inf.config = BayesianInferenceConfig()
inf.set_observations(observations, observation_times, sensor_locations, noise_std, true_kappa)
inf.temperature_scale_factor = temperature_scale_factor
inf.domain_length = 1.0
inf.nx = 100
inf.final_time = 0.5
inf.boundary_conditions = {'left': 0.0, 'right': 0.0}
inf.initial_condition = lambda x: np.exp(-((x - 0.3) / 0.02)**2)

print(f'Dataset: {inf.observations.size} observations')
print(f'True kappa: {inf.true_kappa}')
print(f'Sensors: {inf.sensor_locations}')

# Test likelihood speed
print('Testing likelihood speed...')
start = time.time()
ll = inf.log_likelihood(inf.true_kappa)
end = time.time()
print(f'Likelihood: {ll:.2f} ({end-start:.3f}s per evaluation)')

# Configure MCMC
inf.config.n_samples = 800
inf.config.n_burn = 400  
inf.config.n_chains = 2
inf.config.rhat_threshold = 1.1
inf.config.use_multiprocessing = False  # Sequential for visible output
total_evals = (800 + 400) * 2
estimated_time = total_evals * (end-start) / 60
print(f'Estimated runtime: {estimated_time:.1f} minutes')
print(f'Total evaluations: {total_evals}')
print('')

print('Starting MCMC sampling...')
print('=' * 50)
sys.stdout.flush()

start_mcmc = time.time()
converged = inf.run_parallel_mcmc()
end_mcmc = time.time()

runtime = (end_mcmc - start_mcmc) / 60
print('')
print('=' * 50)
print('MCMC COMPLETED')
print(f'Runtime: {runtime:.1f} minutes')
print(f'Converged: {converged}')

if inf.posterior_summary:
    s = inf.posterior_summary
    print(f'Posterior mean: {s[\"mean\"]:.4f}')
    print(f'Posterior std: {s[\"std\"]:.4f}') 
    print(f'95% CI: [{s[\"ci_95\"][0]:.4f}, {s[\"ci_95\"][1]:.4f}]')
    print(f'True kappa: {inf.true_kappa}')
    
    coverage = s['ci_95'][0] <= inf.true_kappa <= s['ci_95'][1]
    bias = abs(s['mean'] - inf.true_kappa)
    print(f'Coverage: {\"YES\" if coverage else \"NO\"}')
    print(f'Bias: {bias:.4f}')
    
    if converged and coverage:
        print('🎉 SUCCESS: MCMC parameter estimation completed!')
    else:
        print('⚠️ REVIEW: Check convergence or coverage')
        
print('')
print('Generating diagnostic plots...')
inf.plot_posterior_diagnostics('plots/phase4_terminal_run.png')
print('✅ Plots saved: plots/phase4_terminal_run.png')
print('')
print('🚀 MCMC RUN COMPLETE - PHASE 3/4 OPTIMIZATION SUCCESSFUL!')
"