"""
Phase 5 PAC-Bayes Issues - Comprehensive Diagnostic Script

This script systematically diagnoses the three main issues:
1. PAC-Bayes bound violations (KL > bound)
2. Uncertified interval failure (doesn't cover true κ)
3. Negative PAC bounds

Each issue is analyzed with specific fixes proposed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from src.pac_bayes_bounds import PACBayesBounds, PACBayesConfig
from src.bayesian_inference import create_specification_inference

def diagnose_posterior_bias():
    """Diagnose Issue 1: Why posterior mean ≈ 5.012 instead of 5.0"""
    print("=" * 60)
    print("DIAGNOSTIC 1: POSTERIOR BIAS ANALYSIS")
    print("=" * 60)
    
    try:
        # Load Phase 4 results
        inf = create_specification_inference()
        
        print(f"Dataset properties:")
        print(f"  True κ: {inf.true_kappa}")
        print(f"  Observations shape: {inf.observations.shape}")
        print(f"  Noise std: {inf.noise_std:.6f}")
        
        # Test likelihood at true parameter
        true_logL = inf.log_likelihood(5.0)
        test_logL = inf.log_likelihood(5.012)
        
        print(f"\nLikelihood comparison:")
        print(f"  logL(κ=5.000): {true_logL:.4f}")
        print(f"  logL(κ=5.012): {test_logL:.4f}")
        print(f"  Difference: {test_logL - true_logL:.4f}")
        
        if test_logL > true_logL:
            print("  → ISSUE: Likelihood prefers κ=5.012 over true κ=5.0")
            print("  → This indicates systematic model bias")
        else:
            print("  → Likelihood correctly prefers true parameter")
        
        # Test around true parameter
        kappa_test = np.linspace(4.99, 5.03, 21)
        likelihoods = [inf.log_likelihood(k) for k in kappa_test]
        
        max_idx = np.argmax(likelihoods)
        optimal_kappa = kappa_test[max_idx]
        
        print(f"\nGrid search around truth:")
        print(f"  Optimal κ: {optimal_kappa:.4f}")
        print(f"  Error from truth: {abs(optimal_kappa - 5.0):.4f}")
        
        # Check if there's a systematic offset
        if abs(optimal_kappa - 5.0) > 0.005:
            print(f"  → CONFIRMED BIAS: Model systematically prefers κ={optimal_kappa:.4f}")
            return optimal_kappa - 5.0  # Return bias amount
        else:
            print(f"  → No significant bias detected")
            return 0.0
            
    except Exception as e:
        print(f"Error in bias diagnosis: {e}")
        return None

def diagnose_kl_divergence():
    """Diagnose Issue 2: Why KL divergence is so high (6.024)"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: KL DIVERGENCE ANALYSIS")
    print("=" * 60)
    
    try:
        pac_bayes = PACBayesBounds()
        data_dict = pac_bayes.load_phase4_results()
        
        posterior_samples = data_dict['posterior_samples']
        
        print(f"Posterior statistics:")
        print(f"  Samples: {len(posterior_samples)}")
        print(f"  Mean: {np.mean(posterior_samples):.6f}")
        print(f"  Std: {np.std(posterior_samples):.6f}")
        print(f"  Range: [{np.min(posterior_samples):.6f}, {np.max(posterior_samples):.6f}]")
        
        # Prior statistics for comparison
        prior_mean = 5.5  # Uniform [1,10] has mean 5.5
        prior_std = np.sqrt((10-1)**2/12)  # Uniform variance formula
        
        print(f"\nPrior statistics:")
        print(f"  Mean: {prior_mean:.6f}")
        print(f"  Std: {prior_std:.6f}")
        print(f"  Range: [1.0, 10.0]")
        
        # Concentration analysis
        posterior_std = np.std(posterior_samples)
        concentration_ratio = prior_std / posterior_std
        
        print(f"\nConcentration analysis:")
        print(f"  Posterior std: {posterior_std:.6f}")
        print(f"  Prior std: {prior_std:.6f}")
        print(f"  Concentration ratio: {concentration_ratio:.2f}x")
        
        if concentration_ratio > 100:
            print(f"  → ISSUE: Posterior extremely concentrated (>{concentration_ratio:.0f}x)")
            print(f"  → This causes high KL divergence")
        elif concentration_ratio > 10:
            print(f"  → WARNING: Posterior very concentrated ({concentration_ratio:.1f}x)")
        else:
            print(f"  → Concentration level reasonable")
        
        # Compute actual KL divergence
        kl_div = pac_bayes.compute_kl_divergence(posterior_samples)
        print(f"\nKL divergence: {kl_div:.6f}")
        
        # KL interpretation
        if kl_div > 5.0:
            print(f"  → VERY HIGH: Posterior radically different from prior")
        elif kl_div > 1.0:
            print(f"  → HIGH: Posterior significantly different from prior")
        else:
            print(f"  → REASONABLE: Modest difference from prior")
        
        return {
            'kl_divergence': kl_div,
            'concentration_ratio': concentration_ratio,
            'posterior_std': posterior_std
        }
        
    except Exception as e:
        print(f"Error in KL diagnosis: {e}")
        return None

def diagnose_prior_expectation():
    """Diagnose Issue 3: Why prior expectation is causing negative bounds"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: PRIOR EXPECTATION ANALYSIS")
    print("=" * 60)
    
    try:
        pac_bayes = PACBayesBounds()
        data_dict = pac_bayes.load_phase4_results()
        
        observations = data_dict['observations']
        obs_times = data_dict['observation_times']
        sensors = data_dict['sensor_locations']
        noise_std = data_dict['noise_std']
        
        print(f"Computing prior expectation with reduced samples for diagnosis...")
        
        # Test with small sample for diagnosis
        n_test_samples = 100
        prior_samples = np.random.uniform(1.0, 10.0, n_test_samples)
        
        losses = []
        exp_neg_losses = []
        
        for i, kappa in enumerate(prior_samples[:20]):  # Just first 20 for speed
            loss = pac_bayes.compute_empirical_loss(kappa, observations, obs_times, sensors, noise_std)
            exp_neg_loss = np.exp(-loss)
            
            losses.append(loss)
            exp_neg_losses.append(exp_neg_loss)
            
            if i < 5:  # Show first few
                print(f"  κ={kappa:.3f}: loss={loss:.2f}, e^(-loss)={exp_neg_loss:.2e}")
        
        avg_loss = np.mean(losses)
        avg_exp_neg_loss = np.mean(exp_neg_losses)
        
        print(f"\nSample statistics:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average e^(-loss): {avg_exp_neg_loss:.2e}")
        print(f"  log E[e^(-loss)]: {np.log(avg_exp_neg_loss):.4f}")
        
        # Analyze what makes the expectation small
        if avg_exp_neg_loss < 1e-6:
            print(f"  → ISSUE: Prior expectation extremely small")
            print(f"  → This indicates very poor fit for most prior samples")
            print(f"  → Causes negative PAC bounds")
        elif avg_exp_neg_loss < 1e-3:
            print(f"  → WARNING: Prior expectation quite small")
        else:
            print(f"  → Prior expectation reasonable")
        
        # Check loss distribution
        loss_std = np.std(losses)
        print(f"\nLoss distribution:")
        print(f"  Loss std: {loss_std:.4f}")
        print(f"  Loss range: [{min(losses):.2f}, {max(losses):.2f}]")
        
        if loss_std > 10:
            print(f"  → High variance in losses across prior")
        
        return {
            'avg_loss': avg_loss,
            'prior_expectation': avg_exp_neg_loss,
            'log_expectation': np.log(avg_exp_neg_loss)
        }
        
    except Exception as e:
        print(f"Error in prior expectation diagnosis: {e}")
        return None

def diagnose_pac_bound_computation():
    """Diagnose Issue 4: Overall PAC bound computation"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: PAC BOUND COMPUTATION ANALYSIS")
    print("=" * 60)
    
    try:
        config = PACBayesConfig(delta=0.05, mc_samples_prior=500)  # Fast for diagnosis
        pac_bayes = PACBayesBounds(config)
        data_dict = pac_bayes.load_phase4_results()
        n_obs = data_dict['observations'].size
        
        # Simulate the bound computation components
        print(f"PAC-Bayes bound components:")
        print(f"  n (observations): {n_obs}")
        print(f"  δ (confidence): {config.delta}")
        print(f"  log(1/δ): {np.log(1/config.delta):.4f}")
        print(f"  log(1/δ)/n: {np.log(1/config.delta)/n_obs:.6f}")
        
        # Estimate prior expectation (quick)
        prior_exp_result = diagnose_prior_expectation()
        if prior_exp_result:
            log_exp = prior_exp_result['log_expectation']
            log_exp_term = log_exp / n_obs
            
            print(f"  log E_P[e^(-ℓ)]: {log_exp:.4f}")
            print(f"  log E_P[e^(-ℓ)]/n: {log_exp_term:.6f}")
            
            # Total bound
            pac_bound = (np.log(1/config.delta) + log_exp) / n_obs
            print(f"  PAC bound: {pac_bound:.6f}")
            
            if pac_bound < 0:
                print(f"  → ISSUE: Negative PAC bound")
                print(f"  → This happens when log E_P[e^(-ℓ)] << -log(1/δ)")
                print(f"  → Prior expectation dominates the bound")
                
                # Suggest fixes
                print(f"\n  Potential fixes:")
                print(f"    1. Increase δ (reduce confidence level)")
                print(f"    2. Use different loss function")
                print(f"    3. Modify prior distribution")
                print(f"    4. Increase MC samples for better estimation")
        
        return pac_bound if 'pac_bound' in locals() else None
        
    except Exception as e:
        print(f"Error in PAC bound diagnosis: {e}")
        return None

def propose_fixes():
    """Propose specific fixes based on diagnostic results"""
    print("\n" + "=" * 60)
    print("PROPOSED FIXES")
    print("=" * 60)
    
    print("Fix 1: Address Posterior Bias")
    print("  - Re-examine forward model alignment")
    print("  - Check if scaling correction is still appropriate")
    print("  - Validate canonical dataset generation")
    
    print("\nFix 2: Handle High KL Divergence")
    print("  - Use larger prior uncertainty (e.g., uniform [0.1, 15])")
    print("  - Consider hierarchical prior structure")
    print("  - Accept high KL as indication of informative data")
    
    print("\nFix 3: Improve Prior Expectation Estimation")
    print("  - Increase MC samples to 50,000+")
    print("  - Use importance sampling for better estimation")
    print("  - Consider different loss function scaling")
    
    print("\nFix 4: Alternative PAC-Bayes Approaches")
    print("  - Use PAC-Bayes-λ bounds (different form)")
    print("  - Try concentration inequalities instead")
    print("  - Consider empirical Bernstein bounds")
    
    print("\n" + "=" * 60)

def run_comprehensive_diagnosis():
    """Run all diagnostic tests"""
    print("PAC-BAYES PHASE 5 - COMPREHENSIVE ISSUE DIAGNOSIS")
    print("=" * 80)
    
    results = {}
    
    # Run diagnostics
    bias_result = diagnose_posterior_bias()
    kl_result = diagnose_kl_divergence()
    prior_result = diagnose_prior_expectation()
    bound_result = diagnose_pac_bound_computation()
    
    # Store results
    results['bias'] = bias_result
    results['kl_analysis'] = kl_result
    results['prior_analysis'] = prior_result
    results['bound_value'] = bound_result
    
    # Propose fixes
    propose_fixes()
    
    print(f"\nDiagnosis complete. Issues identified and fixes proposed.")
    return results

if __name__ == "__main__":
    run_comprehensive_diagnosis()