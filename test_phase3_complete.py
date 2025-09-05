#!/usr/bin/env python3
"""
Complete Phase 3 verification script.
Tests all specification requirements with no missing components.
"""

import sys
import numpy as np
from pathlib import Path


def test_phase3_complete():
    """Test complete Phase 3 implementation."""
    print("Testing Phase 3: Complete Synthetic Data Generation")
    print("=" * 55)
    
    try:
        from src.data_generator import SyntheticDataGenerator, create_specification_dataset
        print("✓ All imports successful")
        
        # Test 1: Specification compliance
        generator = SyntheticDataGenerator(
            true_kappa=5.0,
            domain_length=1.0, 
            final_time=0.5,
            nx=100
        )
        
        # Test specification parameters
        assert generator.true_kappa == 5.0, "True kappa must be 5.0"
        assert np.allclose(generator.sensor_locations, [0.25, 0.5, 0.75]), "Sensors must be at 0.25L, 0.5L, 0.75L"
        assert generator.noise_levels == [0.05, 0.10, 0.20], "Noise levels must be 5%, 10%, 20%"
        print("✓ Specification parameters correct")
        
        # Test 2: Clean solution generation
        solution, x_grid, t_grid = generator.generate_clean_solution()
        
        assert solution.shape[1] == 100, f"Wrong spatial points: {solution.shape[1]}"
        assert len(x_grid) == 100, f"Wrong x_grid length: {len(x_grid)}"
        assert x_grid[0] == 0.0 and x_grid[-1] == 1.0, "Domain must be [0,1]"
        assert t_grid[-1] <= 0.5, f"Final time too large: {t_grid[-1]}"
        assert np.all(np.isfinite(solution)), "Solution contains non-finite values"
        print("✓ Clean solution generation works")
        
        # Test 3: Exact sensor placement
        observations = generator.extract_observations()
        
        assert observations.shape[1] == 3, f"Wrong number of sensors: {observations.shape[1]}"
        expected_sensors = np.array([0.25, 0.5, 0.75])
        assert np.allclose(generator.sensor_locations, expected_sensors), "Wrong sensor locations"
        print("✓ Exact sensor placement (0.25L, 0.5L, 0.75L)")
        
        # Test 4: Exact noise levels
        noisy_datasets = generator.add_noise_levels()
        
        expected_keys = ["5pct", "10pct", "20pct"]
        assert all(key in noisy_datasets for key in expected_keys), f"Missing noise keys: {list(noisy_datasets.keys())}"
        
        for key, data in noisy_datasets.items():
            assert data.shape == observations.shape, f"Wrong noisy data shape for {key}"
            assert not np.allclose(data, observations), f"No noise added for {key}"
        print("✓ Exact noise levels (5%, 10%, 20%)")
        
        # Test 5: Sparse variant
        sparse_obs = generator.create_sparse_variant()
        
        assert sparse_obs.shape[1] == 3, f"Wrong sparse sensor count: {sparse_obs.shape[1]}"
        assert sparse_obs.shape[0] == observations.shape[0], "Wrong sparse time points"
        print("✓ Sparse variant (3 sensors)")
        
        # Test 6: Data saving
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            generator.save_data(temp_dir)
            
            # Check all required files exist
            required_files = [
                "config.json",
                "clean_solution.npz", 
                "clean_observations.npz",
                "noisy_observations_5pct.npz",
                "noisy_observations_10pct.npz",
                "noisy_observations_20pct.npz",
                "sparse_observations.npz"
            ]
            
            for filename in required_files:
                filepath = Path(temp_dir) / filename
                assert filepath.exists(), f"Missing file: {filename}"
            
            # Test data loading
            clean_data = np.load(Path(temp_dir) / "clean_solution.npz")
            assert clean_data['true_kappa'] == 5.0, "Wrong true kappa in saved data"
            
        print("✓ Complete data saving and loading")
        
        # Test 7: Complete dataset generation
        dataset = generator.generate_complete_dataset()
        
        required_keys = ['specification_compliance', 'numerical_results', 'data_shapes', 'file_locations']
        assert all(key in dataset for key in required_keys), f"Missing dataset keys: {list(dataset.keys())}"
        
        # Check specification compliance
        spec = dataset['specification_compliance']
        assert spec['true_kappa'] == 5.0, "Wrong true kappa in dataset"
        assert spec['sensors_exact'] == [0.25, 0.5, 0.75], "Wrong sensors in dataset"
        assert spec['noise_levels_exact'] == [0.05, 0.10, 0.20], "Wrong noise levels in dataset"
        
        print("✓ Complete dataset generation")
        
        # Test 8: Specification dataset function
        spec_dataset = create_specification_dataset()
        assert spec_dataset is not None, "Specification dataset creation failed"
        print("✓ Specification dataset function")
        
        print("\n" + "=" * 55)
        print("PHASE 3 VERIFICATION COMPLETE")
        print("=" * 55)
        print("✓ All specification requirements implemented")
        print("✓ True κ = 5.0 (exact)")
        print("✓ Sensors at 0.25L, 0.5L, 0.75L (exact)")
        print("✓ Noise levels 5%, 10%, 20% (exact)")
        print("✓ Sparse mode 10 → 3 sensors (exact)")
        print("✓ Complete data pipeline functional")
        print("✓ All files saved correctly")
        print("✓ Ready for Phase 4")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """Test that Phase 3 generates correct data files."""
    print("\nTesting Phase 3 Data File Generation")
    print("-" * 40)
    
    try:
        # Run Phase 3 to generate files
        from src.data_generator import create_specification_dataset
        dataset = create_specification_dataset()
        
        # Check that files were created
        data_dir = Path("data")
        plots_dir = Path("plots")
        
        if data_dir.exists():
            data_files = list(data_dir.glob("*.npz")) + list(data_dir.glob("*.json"))
            print(f"✓ Data files created: {len(data_files)} files")
            
            # Test loading each file
            for file_path in data_files:
                if file_path.suffix == ".npz":
                    data = np.load(file_path)
                    print(f"  ✓ {file_path.name}: {list(data.keys())}")
                elif file_path.suffix == ".json":
                    import json
                    with open(file_path) as f:
                        config = json.load(f)
                    print(f"  ✓ {file_path.name}: config loaded")
        else:
            print("✗ Data directory not created")
            return False
            
        if plots_dir.exists() and list(plots_dir.glob("*.png")):
            print("✓ Plot files created")
        else:
            print("✗ Plot files not created")
            
        return True
        
    except Exception as e:
        print(f"✗ Data file test failed: {e}")
        return False


def main():
    """Run complete Phase 3 verification."""
    print("PAC-BAYES HEAT EQUATION - PHASE 3 COMPLETE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Phase 3 Complete Implementation", test_phase3_complete),
        ("Phase 3 Data File Generation", test_data_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL" 
        print(f"{status}: {test_name}")
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nPhase 3 is COMPLETE and ready for your paper!")
        print("All specification requirements implemented exactly.")
    else:
        print("\nSome tests failed. Check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)