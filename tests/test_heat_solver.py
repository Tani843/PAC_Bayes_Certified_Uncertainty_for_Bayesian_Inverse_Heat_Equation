"""
Unit tests for heat equation solver with CFL stability checks.
Phase 2: Complete test implementation with RMSE and MAE validation.
"""
import pytest
import numpy as np
import warnings
from src.heat_solver import HeatEquationSolver


class TestHeatEquationSolver:
    """Test suite for HeatEquationSolver class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = HeatEquationSolver(domain_length=1.0)
        self.kappa = 0.1
        self.nx = 50
        self.final_time = 0.5
        
    def test_solver_initialization(self):
        """Test solver initializes correctly."""
        assert self.solver is not None
        assert self.solver.L == 1.0
        assert self.solver.solution is None
        
    def test_cfl_stability_check(self):
        """Test CFL stability condition checking."""
        dx = 0.01
        
        # Stable case
        dt_stable = 0.00004  # κdt/(dx)² = 0.1 * 0.00004 / 0.01² = 0.4 < 0.5
        is_stable, ratio = self.solver.check_cfl_stability(self.kappa, dx, dt_stable)
        assert is_stable
        assert ratio == pytest.approx(0.4, rel=1e-10)
        
        # Unstable case  
        dt_unstable = 0.0001  # κdt/(dx)² = 0.1 * 0.0001 / 0.01² = 1.0 > 0.5
        with warnings.catch_warnings(record=True) as w:
            is_stable, ratio = self.solver.check_cfl_stability(self.kappa, dx, dt_unstable)
            assert not is_stable
            assert ratio == pytest.approx(1.0, rel=1e-10)
            assert len(w) == 1
            assert "CFL condition violated" in str(w[0].message)
            
    def test_cfl_input_validation(self):
        """Test CFL check input validation."""
        with pytest.raises(ValueError, match="Step sizes must be positive"):
            self.solver.check_cfl_stability(self.kappa, -0.01, 0.0001)
            
        with pytest.raises(ValueError, match="Thermal conductivity must be positive"):
            self.solver.check_cfl_stability(-0.1, 0.01, 0.0001)
    
    def test_compute_stable_timestep(self):
        """Test stable timestep computation."""
        dx = 0.01
        cfl_factor = 0.4
        
        dt_expected = cfl_factor * dx**2 / self.kappa  # 0.4 * 0.0001 / 0.1 = 0.0004
        dt_computed = self.solver.compute_stable_timestep(self.kappa, dx, cfl_factor)
        assert dt_computed == pytest.approx(dt_expected, rel=1e-12)
        
        # Test invalid CFL factor
        with pytest.raises(ValueError, match="CFL factor must be in"):
            self.solver.compute_stable_timestep(self.kappa, dx, 0.6)  # > 0.5
    
    def test_forward_solve_shape(self):
        """Test forward solve returns correct shape."""
        # Simple initial condition
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        solution, x_grid, t_grid = self.solver.solve(
            kappa=self.kappa,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=self.nx,
            final_time=self.final_time,
            auto_timestep=True
        )
        
        # Check shapes
        assert x_grid.shape == (self.nx,)
        assert solution.shape[1] == self.nx  # Spatial dimension
        assert len(t_grid) == solution.shape[0]  # Time dimension
        
        # Check grid properties
        assert x_grid[0] == 0.0
        assert x_grid[-1] == pytest.approx(1.0, rel=1e-10)
        assert t_grid[0] == 0.0
        assert t_grid[-1] == pytest.approx(self.final_time, rel=1e-10)
    
    def test_forward_solve_stability(self):
        """Test numerical stability of forward solve."""
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        solution, _, _ = self.solver.solve(
            kappa=self.kappa,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=self.nx,
            final_time=self.final_time,
            auto_timestep=True,
            cfl_factor=0.4
        )
        
        # Solution should not blow up (basic stability check)
        assert np.all(np.isfinite(solution))
        assert np.max(np.abs(solution)) < 1000  # Reasonable bound
        
        # Check energy dissipation (heat equation should decrease energy)
        initial_energy = np.sum(solution[0, :]**2)
        final_energy = np.sum(solution[-1, :]**2)
        assert final_energy <= initial_energy  # Energy should decrease
        
    def test_boundary_conditions(self):
        """Test boundary condition implementation."""
        def initial_condition(x):
            return np.ones_like(x)  # Constant initial condition
            
        # Test constant boundary conditions
        boundary_conditions = {'left': 2.0, 'right': 3.0}
        
        solution, _, _ = self.solver.solve(
            kappa=self.kappa,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=self.nx,
            final_time=0.1,
            auto_timestep=True
        )
        
        # Check boundary values are maintained
        assert np.all(solution[:, 0] == 2.0)   # Left boundary
        assert np.all(solution[:, -1] == 3.0)  # Right boundary
        
        # Test time-dependent boundary conditions
        def bc_left(t):
            return t
        def bc_right(t):
            return 2 * t
            
        boundary_conditions_time = {'left': bc_left, 'right': bc_right}
        
        solution_time, _, t_grid = self.solver.solve(
            kappa=self.kappa,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions_time,
            nx=self.nx,
            final_time=0.1,
            auto_timestep=True
        )
        
        # Check time-dependent boundaries
        for i, t in enumerate(t_grid):
            assert solution_time[i, 0] == pytest.approx(t, rel=1e-12)      # Left
            assert solution_time[i, -1] == pytest.approx(2*t, rel=1e-12)   # Right
            
    def test_analytical_solution_validation(self):
        """Test against known analytical solution."""
        # Use simple analytical solution: u(x,t) = exp(-π²κt) * sin(πx)
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        def exact_solution(x, t):
            return np.exp(-np.pi**2 * self.kappa * t) * np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        solution, x_grid, t_grid = self.solver.solve(
            kappa=self.kappa,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=100,  # Higher resolution for accuracy
            final_time=0.1,
            auto_timestep=True,
            cfl_factor=0.25  # More conservative for accuracy
        )
        
        # Compute error metrics
        error_metrics = self.solver.compute_error_metrics(exact_solution)
        
        # Check error metrics are reasonable
        assert error_metrics['rmse'] < 0.01  # RMSE should be small
        assert error_metrics['mae'] < 0.01   # MAE should be small
        assert error_metrics['relative_rmse'] < 0.05  # Relative error < 5%
        
        print(f"Error metrics: {error_metrics}")
    
    def test_sensor_observations(self):
        """Test sensor observation extraction."""
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        self.solver.solve(
            kappa=self.kappa,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=self.nx,
            final_time=self.final_time,
            auto_timestep=True
        )
        
        # Test sensor locations
        sensor_locations = np.array([0.25, 0.5, 0.75])
        observations = self.solver.get_observations(sensor_locations)
        
        # Check observation shape and properties
        assert observations.shape[1] == len(sensor_locations)
        assert observations.shape[0] > 0  # Should have time observations
        assert np.all(np.isfinite(observations))
    
    def test_input_validation(self):
        """Test input validation for solve method."""
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        # Test negative kappa
        with pytest.raises(ValueError, match="Thermal conductivity must be positive"):
            self.solver.solve(-0.1, initial_condition, boundary_conditions)
            
        # Test insufficient grid points
        with pytest.raises(ValueError, match="Need at least 3 spatial points"):
            self.solver.solve(self.kappa, initial_condition, boundary_conditions, nx=2)
            
        # Test negative final time
        with pytest.raises(ValueError, match="Final time must be positive"):
            self.solver.solve(self.kappa, initial_condition, boundary_conditions, 
                            final_time=-1.0)
            
    def test_unstable_manual_timestep(self):
        """Test that unstable manual timestep raises error."""
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        # Force unstable timestep with auto_timestep=False
        with pytest.raises(ValueError, match="Unstable time step"):
            self.solver.solve(
                kappa=self.kappa,
                initial_condition=initial_condition,
                boundary_conditions=boundary_conditions,
                nx=10,
                nt=10,  # This will create large dt
                final_time=1.0,  # Large final time with few steps
                auto_timestep=False
            )
    
    @pytest.mark.slow
    def test_convergence_study(self):
        """Test numerical convergence with grid refinement."""
        def initial_condition(x):
            return np.sin(np.pi * x)
            
        def exact_solution(x, t):
            return np.exp(-np.pi**2 * self.kappa * t) * np.sin(np.pi * x)
            
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        # Test different grid sizes
        grid_sizes = [20, 40, 80]
        errors = []
        
        for nx in grid_sizes:
            solver_conv = HeatEquationSolver()
            solution, _, _ = solver_conv.solve(
                kappa=self.kappa,
                initial_condition=initial_condition,
                boundary_conditions=boundary_conditions,
                nx=nx,
                final_time=0.05,  # Short time for accuracy
                auto_timestep=True,
                cfl_factor=0.2
            )
            
            error_metrics = solver_conv.compute_error_metrics(exact_solution)
            errors.append(error_metrics['rmse'])
            
        # Check convergence (error should decrease with finer grid)
        assert errors[1] < errors[0]  # Error decreases with refinement
        assert errors[2] < errors[1]  # Continues to decrease
        
        print(f"Convergence study - Grid sizes: {grid_sizes}, Errors: {errors}")


# Integration test for complete pipeline
class TestHeatSolverIntegration:
    """Integration tests for complete heat solver pipeline."""
    
    def test_complete_pipeline(self):
        """Test complete pipeline from setup to observation extraction."""
        solver = HeatEquationSolver(domain_length=2.0)
        
        # Gaussian pulse initial condition
        def initial_condition(x):
            return np.exp(-10 * (x - 1.0)**2)
            
        # Zero Dirichlet boundary conditions  
        boundary_conditions = {'left': 0.0, 'right': 0.0}
        
        # Solve heat equation
        solution, x_grid, t_grid = solver.solve(
            kappa=0.05,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            nx=60,
            final_time=1.0,
            auto_timestep=True,
            cfl_factor=0.3
        )
        
        # Verify solution properties
        assert solution.shape == (len(t_grid), len(x_grid))
        assert np.all(np.isfinite(solution))
        
        # Extract sensor observations
        sensor_locations = np.array([0.5, 1.0, 1.5])
        observations = solver.get_observations(sensor_locations)
        
        assert observations.shape == (len(t_grid), len(sensor_locations))
        
        # Check boundary conditions are satisfied
        assert np.allclose(solution[:, 0], 0.0)   # Left boundary
        assert np.allclose(solution[:, -1], 0.0)  # Right boundary
        
        # Check energy dissipation
        initial_energy = np.sum(solution[0, :]**2 * solver.dx)
        final_energy = np.sum(solution[-1, :]**2 * solver.dx)
        assert final_energy < initial_energy
        
        print("✅ Complete pipeline test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])