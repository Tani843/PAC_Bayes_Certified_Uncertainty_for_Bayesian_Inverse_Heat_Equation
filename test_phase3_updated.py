#!/usr/bin/env python3
"""
Phase 3 Validation Tests - Updated for new implementation
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
        # Test with specification parameters
        generator = SyntheticDataGenerator(
            true_kappa=5.0,
            domain_length=1.0,
            final_time=0.1,  # Smaller for testing
            nx=20
        )
        
        print("✅ Generator initializes with specification parameters")
        
        # Check specification values
        if generator.true_kappa == 5.0:
            print("✅ True kappa correct: 5.0")
        else:
            print(f"❌ Wrong true kappa: {generator.true_kappa}")
            return False
            
        if np.allclose(generator.sensor_locations, [0.25, 0.5, 0.75]):
            print("✅ Sensor locations correct: [0.25, 0.5, 0.75]")
        else:
            print(f"❌ Wrong sensor locations: {generator.sensor_locations}")
            return False
            
        # Check noise levels specification
        expected_noise = [0.05, 0.10, 0.20]
        if generator.noise_levels == expected_noise:
            print("✅ Noise levels correct: 5%, 10%, 20%")
        else:
            print(f"❌ Wrong noise levels: {generator.noise_levels}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_clean_solution():
    """Test Phase 3: Clean solution generation"""
    print("Testing Phase 3: Clean Solution Generation")
    print("-" * 42)
    
    try:
        generator = SyntheticDataGenerator(
            true_kappa=5.0,
            nx=20, 
            final_time=0.1
        )
        
        # Generate clean solution
        solution, x_grid, t_grid = generator.generate_clean_solution()
        
        # Verify solution properties
        if solution.shape[1] == 20:
            print(f"✅ Solution spatial dimension correct: {solution.shape[1]}")
        else:
            print(f"❌ Wrong spatial dimension: {solution.shape[1]}")
            return False
            
        if np.all(np.isfinite(solution)):
            print("✅ Solution is finite and stable")
        else:
            print("❌ Solution contains NaN or Inf")
            return False
            
        if len(x_grid) == 20:
            print(f"✅ X grid has correct size: {len(x_grid)}")
        else:
            print(f"❌ Wrong x grid size: {len(x_grid)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Clean solution test failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_observations():
    """Test Phase 3: Observation extraction at exact sensor positions"""
    print("Testing Phase 3: Observation Extraction")
    print("-" * 38)
    
    try:
        generator = SyntheticDataGenerator(
            true_kappa=5.0,
            nx=20,
            final_time=0.1
        )
        
        # Generate solution and extract observations
        generator.generate_clean_solution()
        observations = generator.extract_observations()
        
        # Check observation properties
        if observations.shape[1] == 3:
            print(f"✅ Observations have 3 sensors: {observations.shape[1]}")
        else:
            print(f"❌ Wrong number of sensors: {observations.shape[1]}")
            return False
            
        # Verify sensor locations are exact specification
        expected_sensors = np.array([0.25, 0.5, 0.75])
        if np.allclose(generator.sensor_locations, expected_sensors):
            print(f"✅ Exact sensor positions: {generator.sensor_locations}")
        else:
            print(f"❌ Wrong sensor positions: {generator.sensor_locations}")
            return False
            
        if np.all(np.isfinite(observations)):
            print("✅ Observations are finite")
        else:
            print("❌ Observations contain NaN or Inf")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Observation extraction test failed: {e}")
        traceback.print_exc()
        return False

def test_phase3_noise_generation():
    """Test Phase 3: Noise generation at specified levels"""
    print("Testing Phase 3: Multi-Level Noise Generation")
    print("-" * 45)
    
    try:
        generator = SyntheticDataGenerator(
            true_kappa=5.0,
            nx=20,
            final_time=0.1
        )
        
        # Generate solution and observations
        generator.generate_clean_solution()
        generator.extract_observations()
        
        # Add noise levels
        noisy_datasets = generator.add_noise_levels()
        
        # Check noise levels
        expected_keys = {"5pct", "10pct", "20pct"}
        actual_keys = set(noisy_datasets.keys())
        
        if actual_keys == expected_keys:
            print("✅ All noise levels present: 5%, 10%, 20%")
        else:
            print(f"❌ Wrong noise levels: {actual_keys} vs {expected_keys}")
            return False
        
        # Check that noise was actually added
        clean_obs = generator.clean_observations
        for noise_key, noisy_data in noisy_datasets.items():
            if not np.allclose(noisy_data, clean_obs, atol=1e-10):
                print(f"✅ Noise level {noise_key}: noise successfully added")
            else:
                print(f"❌ No noise added for {noise_key}")
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
        generator = SyntheticDataGenerator(
            true_kappa=5.0,      # Exact specification
            domain_length=1.0,   # Exact specification
            final_time=0.1,      # Smaller for testing
            nx=20                # Smaller for testing
        )
        
        # Generate complete dataset (smaller version for testing)
        generator.generate_clean_solution()
        generator.extract_observations()
        generator.add_noise_levels()
        generator.create_sparse_variant()
        
        # Check specification compliance
        if generator.true_kappa == 5.0:
            print("✅ True kappa specification: 5.0")
        else:
            print(f"❌ Wrong true kappa: {generator.true_kappa}")
            return False
            
        if np.allclose(generator.sensor_locations, [0.25, 0.5, 0.75]):
            print("✅ Sensor positions specification: [0.25L, 0.5L, 0.75L]")
        else:
            print(f"❌ Wrong sensor positions: {generator.sensor_locations}")
            return False
            
        if generator.noise_levels == [0.05, 0.10, 0.20]:
            print("✅ Noise levels specification: [5%, 10%, 20%]")
        else:
            print(f"❌ Wrong noise levels: {generator.noise_levels}")
            return False
            
        if generator.sparse_observations.shape[1] == 3:
            print("✅ Sparse mode: 3 sensors")
        else:
            print(f"❌ Wrong sparse sensor count: {generator.sparse_observations.shape[1]}")
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
    print("Updated Implementation")
    print("=" * 60)
    print()
    
    tests = [
        ("Phase 3: Imports", test_phase3_imports),
        ("Phase 3: Initialization", test_phase3_initialization),
        ("Phase 3: Clean Solution", test_phase3_clean_solution),
        ("Phase 3: Observations", test_phase3_observations),
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
        print("✅ Updated synthetic data generation is working correctly")
        print("✅ Full specification compliance achieved")
        print("Ready to proceed to Phase 4: Bayesian Inference")
    else:
        print("⚠️  Some Phase 3 tests failed - check the errors above")
        print("Fix the issues before proceeding to Phase 4")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)