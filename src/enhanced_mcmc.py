"""Enhanced MCMC Implementation using emcee

This module provides enhanced MCMC sampling capabilities using the emcee
ensemble sampler, which is more robust and efficient than basic Metropolis-Hastings.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union
import emcee
import corner
import arviz as az
from multiprocessing import Pool
import warnings

from .heat_solver import HeatEquationSolver
from .bayesian_inference import BayesianInference


class EnhancedMCMC:
    """Enhanced MCMC sampling using emcee ensemble sampler."""
    
    def __init__(self, bayesian_inference: BayesianInference):
        """Initialize enhanced MCMC sampler.
        
        Args:
            bayesian_inference: Bayesian inference engine
        """
        self.inference = bayesian_inference
        self.solver = bayesian_inference.solver
        self.nx = bayesian_inference.nx
        
        # Storage for samples and diagnostics
        self.sampler = None
        self.samples = None
        self.log_probs = None
        self.acceptance_fractions = None
        
    def log_probability_function(self, theta, *args):
        """Log probability function for emcee sampler.
        
        Args:
            theta: Parameters (thermal conductivity in log space)
            *args: Additional arguments (observations, etc.)
            
        Returns:
            Log posterior probability
        """
        observations, sensor_locs, obs_times, noise_var, ic, bc, source = args
        
        # Check for finite parameters
        if not np.all(np.isfinite(theta)):
            return -np.inf
        
        # Compute log posterior
        try:
            log_post = self.inference.log_posterior(
                theta, observations, sensor_locs, obs_times,
                noise_var, ic, bc, source
            )
            return log_post if np.isfinite(log_post) else -np.inf
        except Exception:
            return -np.inf
    
    def sample_ensemble(self,
                       observations: np.ndarray,
                       sensor_locations: np.ndarray,
                       observation_times: np.ndarray,
                       noise_variance: float,
                       initial_condition: Union[float, np.ndarray, Callable],
                       boundary_conditions: Tuple[float, float],
                       n_walkers: int = None,
                       n_samples: int = 5000,
                       n_burn: int = 1000,
                       source_term: Optional[Callable] = None,
                       initial_state: Optional[np.ndarray] = None,
                       progress: bool = True,
                       parallel: bool = False) -> Dict:
        """Sample using emcee ensemble sampler.
        
        Args:
            observations: Observed temperatures
            sensor_locations: Sensor x-coordinates
            observation_times: Time points of observations
            noise_variance: Observation noise variance
            initial_condition: Initial temperature distribution
            boundary_conditions: Dirichlet boundary conditions
            n_walkers: Number of ensemble walkers (default: 2 * n_params)
            n_samples: Number of samples per walker
            n_burn: Number of burn-in samples
            source_term: Heat source function
            initial_state: Initial state for walkers
            progress: Show progress bar
            parallel: Use multiprocessing
            
        Returns:
            Dictionary with samples and diagnostics
        """
        if n_walkers is None:
            n_walkers = max(2 * self.nx, 20)  # At least 2*ndim, minimum 20
        
        if n_walkers < 2 * self.nx:
            warnings.warn(f"n_walkers ({n_walkers}) should be at least 2*ndim ({2*self.nx})")
            n_walkers = 2 * self.nx
        
        # Initialize walkers
        if initial_state is None:
            # Start from prior with small perturbations
            initial_state = self.inference.prior_mean
        
        # Create initial positions for all walkers
        pos = initial_state + 0.1 * np.random.randn(n_walkers, self.nx)
        
        # Set up sampler arguments
        args = (observations, sensor_locations, observation_times, 
                noise_variance, initial_condition, boundary_conditions, source_term)
        
        # Create sampler
        if parallel:
            with Pool() as pool:
                self.sampler = emcee.EnsembleSampler(
                    n_walkers, self.nx, self.log_probability_function,
                    args=args, pool=pool
                )
                
                # Run burn-in
                print(f"Running burn-in with {n_burn} steps...")
                pos, _, _ = self.sampler.run_mcmc(pos, n_burn, progress=progress)
                self.sampler.reset()
                
                # Production run
                print(f"Running production with {n_samples} steps...")
                self.sampler.run_mcmc(pos, n_samples, progress=progress)
        else:
            self.sampler = emcee.EnsembleSampler(
                n_walkers, self.nx, self.log_probability_function, args=args
            )
            
            # Run burn-in
            print(f"Running burn-in with {n_burn} steps...")
            pos, _, _ = self.sampler.run_mcmc(pos, n_burn, progress=progress)
            self.sampler.reset()
            
            # Production run
            print(f"Running production with {n_samples} steps...")
            self.sampler.run_mcmc(pos, n_samples, progress=progress)
        
        # Extract results
        self.samples = self.sampler.get_chain(flat=True)
        self.log_probs = self.sampler.get_log_prob(flat=True)
        self.acceptance_fractions = self.sampler.acceptance_fraction
        
        # Compute diagnostics
        diagnostics = self._compute_diagnostics()
        
        # Compute posterior statistics
        posterior_mean = np.mean(self.samples, axis=0)
        posterior_cov = np.cov(self.samples.T)
        
        return {
            'samples': self.samples,
            'log_probabilities': self.log_probs,
            'posterior_mean': posterior_mean,
            'posterior_cov': posterior_cov,
            'acceptance_fractions': self.acceptance_fractions,
            'mean_acceptance': np.mean(self.acceptance_fractions),
            'n_walkers': n_walkers,
            'n_samples': n_samples,
            'n_burn': n_burn,
            'diagnostics': diagnostics,
            'sampler': self.sampler  # For further analysis
        }
    
    def _compute_diagnostics(self) -> Dict:
        """Compute MCMC diagnostics."""
        if self.sampler is None:
            raise ValueError("No sampler available. Run sampling first.")
        
        diagnostics = {}
        
        # Autocorrelation time
        try:
            tau = self.sampler.get_autocorr_time(quiet=True)
            diagnostics['autocorr_time'] = tau
            diagnostics['mean_autocorr_time'] = np.mean(tau)
            diagnostics['max_autocorr_time'] = np.max(tau)
        except emcee.autocorr.AutocorrError as e:
            diagnostics['autocorr_time'] = None
            diagnostics['autocorr_error'] = str(e)
        
        # Effective sample size
        if 'autocorr_time' in diagnostics and diagnostics['autocorr_time'] is not None:
            n_samples = self.samples.shape[0] // self.sampler.nwalkers
            ess = n_samples / (2 * diagnostics['autocorr_time'])
            diagnostics['effective_sample_size'] = ess
            diagnostics['min_ess'] = np.min(ess)
        
        # Gelman-Rubin statistic (R-hat)
        chain = self.sampler.get_chain()  # Shape: (n_steps, n_walkers, n_params)
        if chain.shape[0] > 100:  # Need sufficient samples
            r_hat = self._compute_gelman_rubin(chain)
            diagnostics['r_hat'] = r_hat
            diagnostics['max_r_hat'] = np.max(r_hat)
        
        # Acceptance rate statistics
        diagnostics['acceptance_stats'] = {
            'mean': np.mean(self.acceptance_fractions),
            'std': np.std(self.acceptance_fractions),
            'min': np.min(self.acceptance_fractions),
            'max': np.max(self.acceptance_fractions)
        }
        
        return diagnostics
    
    def _compute_gelman_rubin(self, chain: np.ndarray) -> np.ndarray:
        """Compute Gelman-Rubin R-hat statistic.
        
        Args:
            chain: MCMC chain with shape (n_steps, n_walkers, n_params)
            
        Returns:
            R-hat for each parameter
        """
        n_steps, n_walkers, n_params = chain.shape
        
        # Use second half of chain for R-hat calculation
        chain_half = chain[n_steps//2:]
        n_half = chain_half.shape[0]
        
        r_hat = np.zeros(n_params)
        
        for i in range(n_params):
            # Between-chain variance
            chain_means = np.mean(chain_half[:, :, i], axis=0)
            B = n_half * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            chain_vars = np.var(chain_half[:, :, i], axis=0, ddof=1)
            W = np.mean(chain_vars)
            
            # Marginal posterior variance
            var_plus = ((n_half - 1) / n_half) * W + (1 / n_half) * B
            
            # R-hat
            r_hat[i] = np.sqrt(var_plus / W) if W > 0 else 1.0
        
        return r_hat
    
    def plot_traces(self, param_names: Optional[List[str]] = None, 
                   figsize: Tuple[int, int] = (12, 8)):
        """Plot MCMC trace plots.
        
        Args:
            param_names: Parameter names for labeling
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.sampler is None:
            raise ValueError("No samples available. Run sampling first.")
        
        import matplotlib.pyplot as plt
        
        chain = self.sampler.get_chain()  # Shape: (n_steps, n_walkers, n_params)
        n_steps, n_walkers, n_params = chain.shape
        
        if param_names is None:
            param_names = [f'θ_{i+1}' for i in range(n_params)]
        
        # Select subset of parameters if too many
        n_plot = min(n_params, 6)
        indices = np.linspace(0, n_params-1, n_plot, dtype=int)
        
        fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=True)
        if n_plot == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            # Plot all walkers
            for walker in range(n_walkers):
                axes[i].plot(chain[:, walker, idx], alpha=0.3, linewidth=0.5)
            
            axes[i].set_ylabel(param_names[idx])
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Step')
        plt.tight_layout()
        return fig
    
    def plot_corner(self, param_names: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (10, 10)):
        """Create corner plot of posterior samples.
        
        Args:
            param_names: Parameter names for labeling
            figsize: Figure size
            
        Returns:
            Corner plot figure
        """
        if self.samples is None:
            raise ValueError("No samples available. Run sampling first.")
        
        if param_names is None:
            param_names = [f'θ_{i+1}' for i in range(self.nx)]
        
        # Subsample for plotting if too many parameters
        if self.nx > 10:
            indices = np.linspace(0, self.nx-1, 10, dtype=int)
            samples_plot = self.samples[:, indices]
            labels_plot = [param_names[i] for i in indices]
        else:
            samples_plot = self.samples
            labels_plot = param_names
        
        fig = corner.corner(samples_plot, labels=labels_plot, 
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_kwargs={"fontsize": 12})
        fig.set_size_inches(figsize)
        return fig
    
    def to_arviz(self, param_names: Optional[List[str]] = None) -> az.InferenceData:
        """Convert samples to ArviZ InferenceData format.
        
        Args:
            param_names: Parameter names
            
        Returns:
            ArviZ InferenceData object
        """
        if self.sampler is None:
            raise ValueError("No samples available. Run sampling first.")
        
        if param_names is None:
            param_names = [f'theta_{i}' for i in range(self.nx)]
        
        # Get chain with shape (n_steps, n_walkers, n_params)
        chain = self.sampler.get_chain()
        log_prob = self.sampler.get_log_prob()
        
        # Create coordinate and dimension dictionaries
        coords = {"theta_dim": param_names}
        dims = {"theta": ["theta_dim"]}
        
        # Create posterior samples dictionary
        posterior = {
            "theta": (["chain", "draw", "theta_dim"], chain.transpose(1, 0, 2))
        }
        
        # Create sample stats
        sample_stats = {
            "log_likelihood": (["chain", "draw"], log_prob.T)
        }
        
        # Create InferenceData
        idata = az.from_dict(
            posterior=posterior,
            sample_stats=sample_stats,
            coords=coords,
            dims=dims
        )
        
        return idata
    
    def summary_statistics(self) -> Dict:
        """Compute comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.samples is None:
            raise ValueError("No samples available. Run sampling first.")
        
        # Convert to physical space (conductivity)
        k_samples = np.exp(self.samples)
        
        # Compute quantiles
        theta_quantiles = np.percentile(self.samples, [2.5, 25, 50, 75, 97.5], axis=0)
        k_quantiles = np.percentile(k_samples, [2.5, 25, 50, 75, 97.5], axis=0)
        
        summary = {
            'theta': {
                'mean': np.mean(self.samples, axis=0),
                'std': np.std(self.samples, axis=0),
                'quantiles': {
                    '2.5%': theta_quantiles[0],
                    '25%': theta_quantiles[1], 
                    '50%': theta_quantiles[2],
                    '75%': theta_quantiles[3],
                    '97.5%': theta_quantiles[4]
                }
            },
            'conductivity': {
                'mean': np.mean(k_samples, axis=0),
                'std': np.std(k_samples, axis=0),
                'quantiles': {
                    '2.5%': k_quantiles[0],
                    '25%': k_quantiles[1],
                    '50%': k_quantiles[2], 
                    '75%': k_quantiles[3],
                    '97.5%': k_quantiles[4]
                }
            },
            'diagnostics': self._compute_diagnostics() if self.sampler else {},
            'n_samples': len(self.samples)
        }
        
        return summary
    
    def save_results(self, filename: str):
        """Save MCMC results to HDF5 file.
        
        Args:
            filename: Output filename
        """
        import h5py
        
        if self.samples is None:
            raise ValueError("No samples to save. Run sampling first.")
        
        with h5py.File(filename, 'w') as f:
            # Save samples and basic results
            f.create_dataset('samples', data=self.samples)
            f.create_dataset('log_probabilities', data=self.log_probs)
            f.create_dataset('acceptance_fractions', data=self.acceptance_fractions)
            
            # Save metadata
            f.attrs['n_walkers'] = self.sampler.nwalkers
            f.attrs['n_samples'] = self.samples.shape[0]
            f.attrs['n_params'] = self.nx
            f.attrs['mean_acceptance'] = np.mean(self.acceptance_fractions)
            
            # Save diagnostics if available
            diagnostics = self._compute_diagnostics()
            if 'autocorr_time' in diagnostics and diagnostics['autocorr_time'] is not None:
                f.create_dataset('autocorr_time', data=diagnostics['autocorr_time'])
            if 'r_hat' in diagnostics:
                f.create_dataset('r_hat', data=diagnostics['r_hat'])
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename: str):
        """Load MCMC results from HDF5 file.
        
        Args:
            filename: Input filename
        """
        import h5py
        
        with h5py.File(filename, 'r') as f:
            self.samples = f['samples'][:]
            self.log_probs = f['log_probabilities'][:]
            self.acceptance_fractions = f['acceptance_fractions'][:]
        
        print(f"Results loaded from {filename}")
        
    def convergence_diagnostics(self) -> Dict:
        """Comprehensive convergence diagnostics.
        
        Returns:
            Dictionary with convergence assessment
        """
        diagnostics = self._compute_diagnostics()
        
        # Convergence assessment
        convergence = {
            'converged': True,
            'issues': []
        }
        
        # Check R-hat
        if 'r_hat' in diagnostics:
            max_r_hat = diagnostics['max_r_hat']
            if max_r_hat > 1.1:
                convergence['converged'] = False
                convergence['issues'].append(f"High R-hat: {max_r_hat:.3f} > 1.1")
        
        # Check acceptance rate
        mean_acceptance = diagnostics['acceptance_stats']['mean']
        if mean_acceptance < 0.2 or mean_acceptance > 0.7:
            convergence['issues'].append(
                f"Acceptance rate outside optimal range: {mean_acceptance:.3f}"
            )
        
        # Check effective sample size
        if 'min_ess' in diagnostics:
            min_ess = diagnostics['min_ess']
            if min_ess < 100:
                convergence['issues'].append(f"Low ESS: {min_ess:.0f} < 100")
        
        diagnostics['convergence'] = convergence
        return diagnostics