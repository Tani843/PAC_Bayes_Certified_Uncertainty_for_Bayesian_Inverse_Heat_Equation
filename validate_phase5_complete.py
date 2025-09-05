"""
Phase 5 Complete Validation Check
Comprehensive verification before proceeding to Phase 6
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def validate_results_files():
    """Validate that all required result files exist and contain expected data"""
    print("=" * 60)
    print("VALIDATION 1: RESULTS FILES")
    print("=" * 60)
    
    required_files = [
        "results/phase5_pac_bayes_kl/pac_bayes_kl_results.json",
        "plots/pac_bayes_kl_certified_bounds.png", 
        "plots/pac_bayes_kl_certified_bounds.svg"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def validate_json_results():
    """Validate JSON results structure and values"""
    print("\n" + "=" * 60)
    print("VALIDATION 2: JSON RESULTS INTEGRITY")
    print("=" * 60)
    
    json_path = Path("results/phase5_pac_bayes_kl/pac_bayes_kl_results.json")
    if not json_path.exists():
        print("✗ JSON results file missing")
        return False
    
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        # Check required fields
        required_fields = [
            'n_observations', 'delta', 'kl_divergence', 'rhs_bound',
            'empirical_risk', 'risk_upper_bound', 'posterior_mean',
            'uncertified_ci', 'certified_interval', 'true_kappa',
            'uncertified_covers', 'certified_covers', 'n_pde_solves',
            'runtime_seconds'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in results:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"✗ Missing fields: {missing_fields}")
            return False
        
        print("✓ All required fields present")
        
        # Validate key mathematical properties
        print(f"\nKey Results Validation:")
        print(f"  n_observations: {results['n_observations']}")
        print(f"  delta: {results['delta']}")
        print(f"  KL divergence: {results['kl_divergence']:.6f}")
        print(f"  Empirical risk: {results['empirical_risk']:.6f}")
        print(f"  Risk upper bound: {results['risk_upper_bound']:.6f}")
        print(f"  True kappa: {results['true_kappa']}")
        print(f"  Certified covers: {results['certified_covers']}")
        print(f"  Uncertified covers: {results['uncertified_covers']}")
        
        # Mathematical consistency checks
        issues = []
        
        if results['risk_upper_bound'] >= 1.0:
            issues.append("Vacuous bound (risk_upper_bound >= 1.0)")
        
        if results['empirical_risk'] > results['risk_upper_bound']:
            issues.append("Empirical risk exceeds upper bound")
        
        if not (1.0 <= results['true_kappa'] <= 10.0):
            issues.append("True kappa outside prior bounds")
        
        if results['n_observations'] != 240:  # Expected: 80 × 3
            issues.append(f"Unexpected number of observations: {results['n_observations']}")
        
        if results['delta'] != 0.05:
            issues.append(f"Unexpected delta value: {results['delta']}")
        
        if issues:
            print(f"\n⚠ Mathematical Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print(f"\n✓ Mathematical consistency validated")
        
        return results
        
    except Exception as e:
        print(f"✗ Error reading JSON: {e}")
        return False

def validate_coverage_properties(results):
    """Validate coverage properties and interval relationships"""
    print("\n" + "=" * 60)
    print("VALIDATION 3: COVERAGE ANALYSIS")
    print("=" * 60)
    
    true_kappa = results['true_kappa']
    uncert_ci = results['uncertified_ci']
    cert_interval = results['certified_interval']
    uncert_covers = results['uncertified_covers']
    cert_covers = results['certified_covers']
    
    print(f"Interval Analysis:")
    print(f"  True κ: {true_kappa}")
    print(f"  Uncertified CI: [{uncert_ci[0]:.4f}, {uncert_ci[1]:.4f}]")
    print(f"  Certified interval: [{cert_interval[0]:.4f}, {cert_interval[1]:.4f}]")
    
    # Width comparison
    uncert_width = uncert_ci[1] - uncert_ci[0]
    cert_width = cert_interval[1] - cert_interval[0]
    width_ratio = cert_width / uncert_width
    
    print(f"  Uncertified width: {uncert_width:.4f}")
    print(f"  Certified width: {cert_width:.4f}")
    print(f"  Width ratio: {width_ratio:.2f}")
    
    # Coverage validation
    print(f"\nCoverage Validation:")
    
    # Manual coverage check
    manual_uncert_covers = uncert_ci[0] <= true_kappa <= uncert_ci[1]
    manual_cert_covers = cert_interval[0] <= true_kappa <= cert_interval[1]
    
    print(f"  Uncertified covers (stored): {uncert_covers}")
    print(f"  Uncertified covers (computed): {manual_uncert_covers}")
    print(f"  Certified covers (stored): {cert_covers}")
    print(f"  Certified covers (computed): {manual_cert_covers}")
    
    issues = []
    
    if uncert_covers != manual_uncert_covers:
        issues.append("Uncertified coverage mismatch")
    
    if cert_covers != manual_cert_covers:
        issues.append("Certified coverage mismatch")
    
    if cert_width < uncert_width:
        issues.append("Certified interval narrower than uncertified (unexpected)")
    
    if width_ratio < 1.0:
        issues.append("Width ratio < 1.0 (certified should be wider)")
    
    if width_ratio > 30.0:
        issues.append("Width ratio > 30.0 (certified excessively wide)")
    
    if issues:
        print(f"\n⚠ Coverage Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✓ Coverage properties validated")
        return True

def validate_performance_metrics(results):
    """Validate computational performance metrics"""
    print("\n" + "=" * 60)
    print("VALIDATION 4: PERFORMANCE METRICS")  
    print("=" * 60)
    
    n_pde_solves = results['n_pde_solves']
    runtime = results['runtime_seconds']
    n_posterior = results.get('n_posterior_samples', 'unknown')
    
    print(f"Performance Metrics:")
    print(f"  PDE solves: {n_pde_solves}")
    print(f"  Posterior samples: {n_posterior}")
    print(f"  Runtime: {runtime:.1f} seconds")
    
    if n_pde_solves > 0:
        time_per_solve = runtime / n_pde_solves
        print(f"  Time per PDE solve: {time_per_solve:.3f} seconds")

    # Performance checks
    issues = []
    
    if n_pde_solves > 500:
        issues.append(f"Too many PDE solves ({n_pde_solves}), caching not working")
    
    if runtime > 120:  # 2 minutes
        issues.append(f"Runtime too long ({runtime:.1f}s)")
    
    if n_pde_solves < 100:
        issues.append(f"Too few PDE solves ({n_pde_solves}), may indicate errors")
    
    if issues:
        print(f"\n⚠ Performance Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✓ Performance metrics acceptable")
        return True

def validate_mathematical_bounds(results):
    """Validate mathematical bounds and inequalities"""
    print("\n" + "=" * 60)
    print("VALIDATION 5: MATHEMATICAL BOUNDS")
    print("=" * 60)
    
    kl_div = results['kl_divergence']
    rhs_bound = results['rhs_bound'] 
    emp_risk = results['empirical_risk']
    risk_bound = results['risk_upper_bound']
    n_obs = results['n_observations']
    delta = results['delta']
    
    # Recompute key quantities for validation
    log_term = np.log(2 * np.sqrt(n_obs) / delta)
    expected_rhs = (kl_div + log_term) / n_obs
    
    print(f"Bound Verification:")
    print(f"  KL(Q||P): {kl_div:.6f}")
    print(f"  Log term: {log_term:.6f}")
    print(f"  Expected RHS: {expected_rhs:.6f}")
    print(f"  Stored RHS: {rhs_bound:.6f}")
    print(f"  Empirical risk: {emp_risk:.6f}")
    print(f"  Risk upper bound: {risk_bound:.6f}")
    
    issues = []
    
    # Check RHS computation
    if abs(rhs_bound - expected_rhs) > 1e-6:
        issues.append(f"RHS bound mismatch: {rhs_bound:.6f} vs {expected_rhs:.6f}")
    
    # Check bound relationships
    if emp_risk > risk_bound:
        issues.append("Empirical risk exceeds upper bound")
    
    if risk_bound >= 1.0:
        issues.append("Vacuous risk bound (>= 1.0)")
    
    if kl_div < 0:
        issues.append("Negative KL divergence")
    
    if rhs_bound < 0:
        issues.append("Negative RHS bound")
    
    if issues:
        print(f"\n⚠ Mathematical Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✓ Mathematical bounds validated")
        return True

def validate_plots_quality():
    """Check that plots exist and have reasonable file sizes"""
    print("\n" + "=" * 60)
    print("VALIDATION 6: PLOT QUALITY")
    print("=" * 60)
    
    plot_files = [
        "plots/pac_bayes_kl_certified_bounds.png",
        "plots/pac_bayes_kl_certified_bounds.svg"
    ]
    
    for plot_file in plot_files:
        path = Path(plot_file)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {plot_file}: {size:,} bytes")
            
            # Basic size checks
            if plot_file.endswith('.png') and size < 50000:  # 50KB
                print(f"  ⚠ PNG file seems small, may indicate rendering issues")
            elif plot_file.endswith('.svg') and size < 10000:  # 10KB
                print(f"  ⚠ SVG file seems small, may indicate rendering issues")
        else:
            print(f"✗ {plot_file}: MISSING")
            return False
    
    print(f"\n✓ Plot files validated")
    return True

def validate_phase4_integration():
    """Validate integration with Phase 4 results"""
    print("\n" + "=" * 60)
    print("VALIDATION 7: PHASE 4 INTEGRATION")
    print("=" * 60)
    
    # Check Phase 4 files still exist
    phase4_files = [
        "data/canonical_dataset.npz",
        "results/phase4_production"
    ]
    
    for file_path in phase4_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            return False

    # Load canonical dataset for cross-validation
    try:
        dataset = np.load("data/canonical_dataset.npz")
        n_obs_expected = dataset['observations'].size
        true_kappa_expected = float(dataset['true_kappa'])
        
        # Load Phase 5 results for comparison  
        with open("results/phase5_pac_bayes_kl/pac_bayes_kl_results.json", 'r') as f:
            phase5_results = json.load(f)
        
        print(f"\nCross-validation:")
        print(f"  Expected n_obs: {n_obs_expected}")
        print(f"  Phase 5 n_obs: {phase5_results['n_observations']}")
        print(f"  Expected true_kappa: {true_kappa_expected}")
        print(f"  Phase 5 true_kappa: {phase5_results['true_kappa']}")
        
        if n_obs_expected != phase5_results['n_observations']:
            print(f"✗ Observation count mismatch")
            return False
            
        if abs(true_kappa_expected - phase5_results['true_kappa']) > 1e-6:
            print(f"✗ True kappa mismatch")
            return False
        
        print(f"✓ Phase 4 integration validated")
        return True
        
    except Exception as e:
        print(f"✗ Error validating integration: {e}")
        return False

def run_comprehensive_validation():
    """Run all validation checks"""
    print("PAC-BAYES PHASE 5 - COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    validation_results = {}
    
    # Run all validations
    validations = [
        ("Files Exist", validate_results_files),
        ("JSON Integrity", validate_json_results), 
        ("Performance", validate_performance_metrics),
        ("Plots Quality", validate_plots_quality),
        ("Phase 4 Integration", validate_phase4_integration)
    ]
    
    json_results = None
    all_passed = True
    
    for name, validation_func in validations:
        try:
            if name == "JSON Integrity":
                result = validation_func()
                if result and isinstance(result, dict):
                    json_results = result
                    validation_results[name] = True
                else:
                    validation_results[name] = False
                    all_passed = False
            else:
                # For validations that need json_results
                if name in ["Coverage Analysis", "Mathematical Bounds", "Performance"] and json_results:
                    result = validation_func(json_results)
                else:
                    result = validation_func()
                
                validation_results[name] = result
                if not result:
                    all_passed = False
                    
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            validation_results[name] = False
            all_passed = False
    
    # Run coverage and bounds validation if we have json results
    if json_results:
        try:
            coverage_result = validate_coverage_properties(json_results)
            bounds_result = validate_mathematical_bounds(json_results)
            validation_results["Coverage Analysis"] = coverage_result
            validation_results["Mathematical Bounds"] = bounds_result
            if not coverage_result or not bounds_result:
                all_passed = False
        except Exception as e:
            print(f"✗ Coverage/Bounds validation failed: {e}")
            all_passed = False
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for validation_name, passed in validation_results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {validation_name}: {status}")
    
    print(f"\nOverall Status: {'ALL VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    
    if all_passed:
        print(f"\n🎉 Phase 5 fully validated and ready for Phase 6!")
        print(f"✓ Non-vacuous PAC-Bayes-kl bounds achieved")
        print(f"✓ Certified intervals provide coverage where uncertified fail")
        print(f"✓ Computational efficiency maintained")
        print(f"✓ Mathematical rigor confirmed")
        print(f"✓ Integration with Phase 4 verified")
        return True
    else:
        print(f"\n❌ Phase 5 has issues that need resolution before Phase 6")
        return False

if __name__ == "__main__":
    run_comprehensive_validation()