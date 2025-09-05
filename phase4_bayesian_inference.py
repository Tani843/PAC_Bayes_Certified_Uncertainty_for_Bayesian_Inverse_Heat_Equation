#!/usr/bin/env python3
"""
Phase 4: Bayesian Inference Execution Script
Complete MCMC-based parameter estimation with convergence diagnostics
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.bayesian_inference import demo_bayesian_inference, create_specification_inference

def main():
    """Execute comprehensive Phase 4 Bayesian inference."""
    
    print("="*80)
    print("PHASE 4: BAYESIAN INFERENCE FOR THERMAL CONDUCTIVITY ESTIMATION")
    print("MCMC sampling with R-hat convergence diagnostics")
    print("="*80)
    
    try:
        # Execute the complete Bayesian inference pipeline
        inference, converged = demo_bayesian_inference()
        
        if inference is None:
            print("❌ INFERENCE FAILED")
            return None
        
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        
        # Display inference summary
        if hasattr(inference, 'posterior_summary') and inference.posterior_summary:
            summary = inference.posterior_summary
            print("\n✓ POSTERIOR SUMMARY:")
            print(f"  - Posterior mean: {summary['mean']:.4f} ± {summary['std']:.4f}")
            print(f"  - Posterior median: {summary['median']:.4f}")
            print(f"  - 68% CI: [{summary['ci_68'][0]:.4f}, {summary['ci_68'][1]:.4f}]")
            print(f"  - 95% CI: [{summary['ci_95'][0]:.4f}, {summary['ci_95'][1]:.4f}]")
            print(f"  - Total samples: {summary['n_samples']}")
            print(f"  - Number of chains: {summary['n_chains']}")
            
            if inference.true_kappa is not None:
                true_val = inference.true_kappa
                in_68 = summary['ci_68'][0] <= true_val <= summary['ci_68'][1]
                in_95 = summary['ci_95'][0] <= true_val <= summary['ci_95'][1]
                print(f"  - True κ: {true_val:.4f}")
                print(f"  - Coverage: 68% CI {'✓' if in_68 else '✗'}, 95% CI {'✓' if in_95 else '✗'}")
        
        # Display convergence information
        print("\n✓ CONVERGENCE DIAGNOSTICS:")
        print(f"  - Convergence achieved: {'✓' if converged else '✗'}")
        if hasattr(inference, 'convergence_history') and inference.convergence_history:
            last_diag = inference.convergence_history[-1]
            print(f"  - Final R-hat: {last_diag['rhat']:.4f}")
            print(f"  - Effective sample size: {last_diag['n_eff']:.1f}")
            print(f"  - R-hat threshold: {inference.config.rhat_threshold}")
            print(f"  - Min. effective samples: {inference.config.min_effective_samples}")
        
        # Display computational information
        print("\n✓ COMPUTATIONAL DETAILS:")
        print(f"  - MCMC configuration:")
        print(f"    • Chains: {inference.config.n_chains}")
        print(f"    • Walkers per chain: {inference.config.n_walkers}")
        print(f"    • Burn-in samples: {inference.config.n_burn}")
        print(f"    • Production samples: {inference.config.n_samples}")
        print(f"  - Prior bounds: [{inference.config.prior_bounds[0]}, {inference.config.prior_bounds[1]}]")
        print(f"  - Estimated noise std: {inference.noise_std:.4f}")
        
        # Display data information
        print("\n✓ DATA CONFIGURATION:")
        if inference.observations is not None:
            print(f"  - Observations shape: {inference.observations.shape}")
            print(f"  - Sensor locations: {inference.sensor_locations}")
            print(f"  - Time range: [{inference.observation_times[0]:.3f}, {inference.observation_times[-1]:.3f}]")
        
        print("\n✓ OUTPUT FILES:")
        print("  - plots/phase4_bayesian_diagnostics.png")
        print("  - results/inference_config.json")
        print("  - results/inference_results.json")
        if inference.config.save_chains:
            print("  - results/mcmc_chains.npy")
            print("  - results/log_prob_chains.npy")
        
        # Quality assessment
        quality_score = 0
        quality_factors = []
        
        if converged:
            quality_score += 40
            quality_factors.append("Convergence achieved")
        else:
            quality_factors.append("Convergence not achieved")
        
        if hasattr(inference, 'posterior_summary') and inference.posterior_summary:
            if inference.posterior_summary['n_samples'] > 1000:
                quality_score += 20
                quality_factors.append("Sufficient samples")
            
            if inference.true_kappa is not None:
                summary = inference.posterior_summary
                in_95 = summary['ci_95'][0] <= inference.true_kappa <= summary['ci_95'][1]
                if in_95:
                    quality_score += 20
                    quality_factors.append("True value in 95% CI")
        
        if hasattr(inference, 'chains') and len(inference.chains) >= 2:
            quality_score += 20
            quality_factors.append("Multiple chains successful")
        
        print(f"\n✓ QUALITY ASSESSMENT: {quality_score}/100")
        for factor in quality_factors:
            print(f"  - {factor}")
        
        if quality_score >= 80:
            quality_level = "EXCELLENT"
        elif quality_score >= 60:
            quality_level = "GOOD"
        elif quality_score >= 40:
            quality_level = "ACCEPTABLE"
        else:
            quality_level = "NEEDS IMPROVEMENT"
        
        print(f"\n  Overall Quality: {quality_level}")
        
        print("\n" + "="*80)
        if converged and quality_score >= 60:
            print("✅ PHASE 4 COMPLETE - BAYESIAN INFERENCE SUCCESSFUL")
        else:
            print("⚠️ PHASE 4 COMPLETE - RESULTS MAY NEED REVIEW")
        print("="*80)
        
        return inference, converged
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None, False


if __name__ == "__main__":
    results = main()