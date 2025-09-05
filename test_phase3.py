#!/usr/bin/env python3
"""
Phase 3 Validation Tests - Quick validation of synthetic data generation
"""

import numpy as np
import sys
import traceback
from src.data_generator import SyntheticDataGenerator

def test_phase3_imports():
    """Test Phase 3: Data generator imports"""
    print("Testing Phase 3: Data Generator Imports")
    print("-" * 40)
    
    try:
        from src.data_generator import SyntheticDataGenerator
        print("✅ SyntheticDataGenerator imports successfully")
        
        from src.heat_solver import HeatEquationSolver
        print("✅ HeatEquationSolver dependency works")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_phase3_initialization():
    """Test Phase 3: Generator initialization"""
    print("Testing Phase 3: Generator Initialization")
    print("-" * 40)
    
    try:
        # Test with default config
        config = DataGenerationConfig(n_realizations=2, nx=20, n_time_points=10)
        generator = SyntheticDataGenerator(config)
        
        print("✅ Generator initializes with custom config")
        
        # Check configuration values
        if generator.config.n_sensors == 10:
            print("✅ Standard sensor count correct: 10")
        else:
            print(f"❌ Wrong sensor count: {generator.config.n_sensors}")
            return False
            
        if generator.config.sparse_sensors == 3:
            print("✅ Sparse sensor count correct: 3")
        else:
            print(f"❌ Wrong sparse sensor count: {generator.config.sparse_sensors}")
            return False
            
        # Check noise levels specification
        expected_noise = (0.05, 0.10, 0.20)
        if generator.config.noise_levels == expected_noise:
            print("✅ Noise levels correct: 5%, 10%, 20%")
        else:
            print(f"❌ Wrong noise levels: {generator.config.noise_levels}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_sensor_placement():
    """Test Phase 3: Sensor placement specification compliance"""
    print("Testing Phase 3: Sensor Placement")
    print("-" * 35)
    
    try:
        config = DataGenerationConfig(domain_length=1.0)
        generator = SyntheticDataGenerator(config)
        
        standard_sensors, sparse_sensors = generator.place_sensors()
        
        # Test standard sensors
        if len(standard_sensors) == 10:
            print("✅ Standard sensor count: 10")
        else:
            print(f"❌ Wrong standard sensor count: {len(standard_sensors)}")
            return False
            
        # Test sparse sensors - EXACT specification
        expected_sparse = np.array([0.25, 0.5, 0.75])
        if np.allclose(sparse_sensors, expected_sparse, atol=1e-10):
            print("✅ Sparse sensors at exact positions: 0.25L, 0.5L, 0.75L")
        else:
            print(f"❌ Wrong sparse sensor positions: {sparse_sensors}")
            print(f"   Expected: {expected_sparse}")
            return False
            
        if len(sparse_sensors) == 3:
            print("✅ Sparse sensor count: 3")
        else:
            print(f"❌ Wrong sparse sensor count: {len(sparse_sensors)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Sensor placement test failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_clean_solution():
    """Test Phase 3: Clean solution generation"""
    print("Testing Phase 3: Clean Solution Generation")
    print("-" * 42)
    
    try:
        config = DataGenerationConfig(nx=20, n_time_points=10, t_final=0.1)
        generator = SyntheticDataGenerator(config)
        
        # Generate clean solution with smaller parameters
        clean_data = generator.generate_clean_solution(
            conductivity_profile="constant",
            initial_condition="gaussian"
        )
        
        # Verify required fields
        required_fields = [
            'solution', 'x_grid', 't_grid', 'true_conductivity',
            'initial_condition', 'conductivity_profile', 
            'initial_condition_type'
        ]
        
        for field in required_fields:
            if field in clean_data:
                print(f"✅ Has field: {field}")
            else:
                print(f"❌ Missing field: {field}")
                return False
        
        # Check solution properties
        solution = clean_data['solution']
        if solution.shape[1] == config.nx:
            print(f"✅ Solution spatial dimension correct: {solution.shape[1]}")
        else:
            print(f"❌ Wrong spatial dimension: {solution.shape[1]}")
            return False
            
        if np.all(np.isfinite(solution)):
            print("✅ Solution is finite and stable")
        else:
            print("❌ Solution contains NaN or Inf")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Clean solution test failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_noise_generation():
    """Test Phase 3: Noise generation at multiple levels"""
    print("Testing Phase 3: Multi-Level Noise Generation")
    print("-" * 45)
    
    try:
        config = DataGenerationConfig(n_realizations=3, nx=20, n_time_points=10)
        generator = SyntheticDataGenerator(config)
        
        # Create fake clean observations
        clean_obs = np.random.randn(10, 10) * 0.5  # 10 time points, 10 sensors
        
        # Generate noisy data
        noisy_data = generator.add_noise_multiple_levels(clean_obs, n_realizations=3)
        
        # Check noise levels
        expected_levels = set([0.05, 0.10, 0.20])
        actual_levels = set(noisy_data.keys())
        
        if actual_levels == expected_levels:
            print("✅ All noise levels present: 5%, 10%, 20%")
        else:
            print(f"❌ Wrong noise levels: {actual_levels} vs {expected_levels}")
            return False
        
        # Check shapes
        for noise_level, realizations in noisy_data.items():
            expected_shape = (3, 10, 10)  # n_realizations × time × sensors
            if realizations.shape == expected_shape:
                print(f"✅ Noise level {noise_level*100}%: correct shape {realizations.shape}")
            else:
                print(f"❌ Wrong shape for {noise_level*100}%: {realizations.shape} vs {expected_shape}")
                return False
        
        # Verify noise is actually added
        for noise_level in noisy_data:
            realization = noisy_data[noise_level][0]
            if not np.allclose(realization, clean_obs, atol=1e-10):
                print(f"✅ Noise level {noise_level*100}%: noise successfully added")
            else:
                print(f"❌ No noise added for {noise_level*100}%")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Noise generation test failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_specification_compliance():
    """Test Phase 3: Overall specification compliance"""
    print("Testing Phase 3: Specification Compliance")
    print("-" * 40)
    
    try:
        config = DataGenerationConfig(
            nx=20, 
            n_time_points=20, 
            t_final=0.05,  # Smaller time to avoid CFL issues
            n_realizations=2
        )
        generator = SyntheticDataGenerator(config)
        
        # Generate small dataset
        dataset = generator.generate_complete_dataset(
            conductivity_profile="constant",
            initial_condition="gaussian"
        )
        
        # Run validation
        validation = generator.validate_dataset(dataset)
        
        if validation['specification_compliant']:
            print("✅ Dataset passes specification compliance")
        else:
            print("❌ Dataset fails specification compliance:")
            for issue in validation['issues']:
                print(f"   - {issue}")
            return False
            
        # Check key compliance points
        if len(dataset['sparse_sensors']) == 3:
            print("✅ Sparse sensor count compliant: 3")
        else:
            print(f"❌ Wrong sparse sensor count: {len(dataset['sparse_sensors'])}")
            return False
            
        noise_levels = set(dataset['standard_noisy_observations'].keys())
        if noise_levels == {0.05, 0.10, 0.20}:
            print("✅ Noise levels compliant: 5%, 10%, 20%")
        else:
            print(f"❌ Wrong noise levels: {noise_levels}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Specification compliance test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run Phase 3 validation tests"""
    print("=" * 60)
    print("PAC-BAYES HEAT EQUATION - PHASE 3 VALIDATION")
    print("=" * 60)
    print()
    
    tests = [
        ("Phase 3: Imports", test_phase3_imports),
        ("Phase 3: Initialization", test_phase3_initialization),
        ("Phase 3: Sensor Placement", test_phase3_sensor_placement),
        ("Phase 3: Clean Solution", test_phase3_clean_solution),
        ("Phase 3: Noise Generation", test_phase3_noise_generation),
        ("Phase 3: Specification Compliance", test_phase3_specification_compliance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
            print()
    
    # Summary
    print("=" * 60)
    print("PHASE 3 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")
            
    print()
    print(f"RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PHASE 3 TESTS PASSED!")
        print("✅ Synthetic Data Generation is working correctly")
        print("✅ Full specification compliance achieved")
        print("Ready to proceed to Phase 4: Bayesian Inference")
    else:
        print("⚠️  Some Phase 3 tests failed - check the errors above")
        print("Fix the issues before proceeding to Phase 4")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)