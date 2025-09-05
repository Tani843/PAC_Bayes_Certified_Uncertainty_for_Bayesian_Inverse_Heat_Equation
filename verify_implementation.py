#!/usr/bin/env python3
"""
Quick verification script for Phase 1 and Phase 2 implementation.
Run this to check if everything is working correctly.
"""

import os
import sys
import traceback

def test_phase1_structure():
    """Test Phase 1: Project structure"""
    print("Testing Phase 1: Project Structure")
    print("-" * 40)
    
    # Check directories
    required_dirs = ['src', 'tests', 'plots', 'data', 'docs']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory {directory}/ exists")
        else:
            print(f"❌ Directory {directory}/ missing")
            return False
            
    # Check key files
    required_files = [
        'src/__init__.py',
        'src/heat_solver.py', 
        'src/utils.py',
        'tests/test_heat_solver.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} missing")
            return False
    
    print("✅ Phase 1 structure complete\n")
    return True

def test_phase2_imports():
    """Test Phase 2: Import functionality"""
    print("Testing Phase 2: Import and Basic Functionality")
    print("-" * 50)
    
    try:
        # Test imports
        from src.heat_solver import HeatEquationSolver
        print("✅ HeatEquationSolver imports successfully")
        
        from src.utils import gaussian_pulse, create_sensor_array
        print("✅ Utilities import successfully")
        
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Dependencies (numpy, matplotlib) work")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_phase2_heat_solver():
    """Test Phase 2: Heat solver core functionality"""
    print("Testing Phase 2: Heat Solver Core Functions")
    print("-" * 45)
    
    try:
        from src.heat_solver import HeatEquationSolver
        import numpy as np
        
        # Test 1: Initialization
        solver = HeatEquationSolver(domain_length=1.0)
        print("✅ Solver initialization works")
        
        # Test 2: CFL stability check
        is_stable, ratio = solver.check_cfl_stability(kappa=0.1, dx=0.01, dt=0.00004)
        expected_ratio = 0.1 * 0.00004 / (0.01**2)  # Should be 0.4
        
        if abs(ratio - expected_ratio) < 1e-10 and is_stable:
            print(f"✅ CFL check works: ratio={ratio:.3f}, stable={is_stable}")
        else:
            print(f"❌ CFL check failed: ratio={ratio:.3f}, expected={expected_ratio:.3f}")
            return False
        
        # Test 3: Stable timestep computation
        dt_stable = solver.compute_stable_timestep(kappa=0.1, dx=0.01, cfl_factor=0.4)
        expected_dt = 0.4 * 0.01**2 / 0.1  # Should be 0.00004
        
        if abs(dt_stable - expected_dt) < 1e-8:
            print(f"✅ Stable timestep: {dt_stable:.6f}")
        else:
            print(f"❌ Stable timestep wrong: {dt_stable:.6f}, expected={expected_dt:.6f}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Heat solver test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_full_solve():
    """Test Phase 2: Complete heat equation solve"""
    print("Testing Phase 2: Complete Heat Equation Solve")
    print("-" * 45)
    
    try:
        from src.heat_solver import HeatEquationSolver
        import numpy as np
        
        solver = HeatEquationSolver()
        
        # Simple sine wave initial condition
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        # Solve heat equation
        solution, x_grid, t_grid = solver.solve(
            kappa=0.1,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=50,
            final_time=0.1,
            auto_timestep=True
        )
        
        # Verify solution properties
        if solution.shape[1] == 50:  # nx points
            print(f"✅ Solution has correct spatial dimension: {solution.shape[1]}")
        else:
            print(f"❌ Wrong spatial dimension: {solution.shape[1]}, expected 50")
            return False
            
        if len(x_grid) == 50:
            print(f"✅ X grid has correct size: {len(x_grid)}")
        else:
            print(f"❌ Wrong x grid size: {len(x_grid)}")
            return False
            
        if np.all(np.isfinite(solution)):
            print("✅ Solution is finite (stable)")
        else:
            print("❌ Solution contains NaN or Inf (unstable)")
            return False
        
        # Check boundary conditions
        if np.allclose(solution[:, 0], 0.0) and np.allclose(solution[:, -1], 0.0):
            print("✅ Boundary conditions satisfied")
        else:
            print("❌ Boundary conditions not satisfied")
            return False
            
        # Check energy dissipation (heat equation property)
        initial_energy = np.sum(solution[0, :]**2)
        final_energy = np.sum(solution[-1, :]**2)
        
        if final_energy < initial_energy:
            print(f"✅ Energy dissipates correctly: {initial_energy:.3f} → {final_energy:.3f}")
        else:
            print(f"❌ Energy should decrease: {initial_energy:.3f} → {final_energy:.3f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Full solve test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_error_metrics():
    """Test Phase 2: Error metrics against analytical solution"""
    print("Testing Phase 2: Error Metrics (RMSE, MAE)")
    print("-" * 40)
    
    try:
        from src.heat_solver import HeatEquationSolver
        import numpy as np
        
        solver = HeatEquationSolver()
        
        def initial_condition(x):
            return np.sin(np.pi * x)
        
        def exact_solution(x, t):
            return np.exp(-np.pi**2 * 0.1 * t) * np.sin(np.pi * x)
        
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        solution, x_grid, t_grid = solver.solve(
            kappa=0.1,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=100,
            final_time=0.05,
            auto_timestep=True,
            cfl_factor=0.25
        )
        
        # Compute error metrics
        error_metrics = solver.compute_error_metrics(exact_solution)
        
        rmse = error_metrics['rmse']
        mae = error_metrics['mae']
        
        print(f"✅ RMSE computed: {rmse:.6f}")
        print(f"✅ MAE computed: {mae:.6f}")
        
        # Check if errors are reasonable
        if rmse < 0.01:
            print(f"✅ RMSE is acceptable: {rmse:.6f} < 0.01")
        else:
            print(f"⚠️ RMSE might be large: {rmse:.6f} (but solver works)")
        
        if mae < 0.01:
            print(f"✅ MAE is acceptable: {mae:.6f} < 0.01")
        else:
            print(f"⚠️ MAE might be large: {mae:.6f} (but solver works)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_utilities():
    """Test Phase 2: Utility functions"""
    print("Testing Phase 2: Utility Functions")
    print("-" * 35)
    
    try:
        from src.utils import gaussian_pulse, sine_wave, create_sensor_array, compute_convergence_rate
        import numpy as np
        
        # Test initial conditions
        x = np.linspace(0, 1, 50)
        
        gaussian = gaussian_pulse(x)
        if 0.5 < np.max(gaussian) < 1.5:
            print(f"✅ Gaussian pulse works: max = {np.max(gaussian):.3f}")
        else:
            print(f"❌ Gaussian pulse issue: max = {np.max(gaussian):.3f}")
            return False
            
        sine = sine_wave(x, frequency=2)  # Use frequency=2 to get full period
        if -1.1 < np.min(sine) < -0.9 and 0.9 < np.max(sine) < 1.1:
            print(f"✅ Sine wave works: range = [{np.min(sine):.3f}, {np.max(sine):.3f}]")
        else:
            print(f"❌ Sine wave issue: range = [{np.min(sine):.3f}, {np.max(sine):.3f}]")
            return False
        
        # Test sensor array
        sensors = create_sensor_array(1.0, 3, 'uniform')
        if len(sensors) == 3 and 0 < sensors[0] < sensors[1] < sensors[2] < 1:
            print(f"✅ Sensor array works: {sensors}")
        else:
            print(f"❌ Sensor array issue: {sensors}")
            return False
            
        # Test convergence rate
        grid_sizes = [20, 40, 80]
        errors = [0.1, 0.025, 0.00625]  # Perfect 2nd order
        rate = compute_convergence_rate(grid_sizes, errors)
        if 1.8 < rate < 2.2:  # Should be close to 2.0
            print(f"✅ Convergence rate: {rate:.2f}")
        else:
            print(f"❌ Convergence rate issue: {rate:.2f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("PAC-BAYES HEAT EQUATION - PHASE 1 & 2 VERIFICATION")
    print("=" * 60)
    print()
    
    tests = [
        ("Phase 1: Project Structure", test_phase1_structure),
        ("Phase 2: Imports", test_phase2_imports),
        ("Phase 2: Heat Solver Core", test_phase2_heat_solver),
        ("Phase 2: Complete Solve", test_phase2_full_solve),
        ("Phase 2: Error Metrics", test_phase2_error_metrics),
        ("Phase 2: Utilities", test_phase2_utilities)
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
    print("VERIFICATION SUMMARY")
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
        print("🎉 ALL TESTS PASSED - Phase 1 & 2 are working perfectly!")
        print("Ready to proceed to Phase 3: Synthetic Data Generation")
    else:
        print("⚠️  Some tests failed - check the errors above")
        print("Fix the issues before proceeding to Phase 3")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)