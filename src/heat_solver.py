"""
Forward Heat Equation Solver with CFL Stability Check

Solves: u_t = κ u_xx, x ∈ [0,L], t > 0
Method: Forward Euler finite differences with CFL stability constraint
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional, Dict, Any
import warnings


class HeatEquationSolver:
    """
    1D Heat Equation Forward Solver with CFL Stability Check.
    
    Solves u_t = κ u_xx with forward Euler finite differences.
    Implements CFL stability constraint: κΔt/(Δx)² ≤ 1/2
    """
    
    def __init__(self, domain_length: float = 1.0):
        """
        Initialize heat equation solver.

        Args:
            domain_length: Spatial domain length L
        """
        self.L = domain_length
        self.solution = None
        self.x_grid = None
        self.t_grid = None
        self.dx = None
        self.dt = None
        self.nx = None
        self.nt = None
        
    def check_cfl_stability(
        self, 
        kappa: float, 
        dx: float, 
        dt: float,
        tolerance: float = 1e-10
    ) -> Tuple[bool, float]:
        """
        Check CFL stability condition: κΔt/(Δx)² ≤ 1/2

        Args:
            kappa: Thermal conductivity parameter
            dx: Spatial step size
            dt: Time step size
            tolerance: Numerical tolerance for stability check
            
        Returns:
            Tuple of (is_stable, stability_ratio)
            - is_stable: True if CFL condition satisfied
            - stability_ratio: κΔt/(Δx)² (should be ≤ 0.5)
        """
        if dx <= 0 or dt <= 0:
            raise ValueError("Step sizes must be positive")
        if kappa <= 0:
            raise ValueError("Thermal conductivity must be positive")
            
        stability_ratio = kappa * dt / (dx**2)
        is_stable = stability_ratio <= (0.5 + tolerance)
        
        if not is_stable:
            warnings.warn(
                f"CFL condition violated: κΔt/(Δx)² = {stability_ratio:.6f} > 0.5. "
                f"Reduce time step to dt ≤ {0.5 * dx**2 / kappa:.6f} for stability.",
                UserWarning
            )
            
        return is_stable, stability_ratio
    
    def compute_stable_timestep(
        self, 
        kappa: float, 
        dx: float, 
        cfl_factor: float = 0.4
    ) -> float:
        """
        Compute stable time step from CFL condition.
        
        Args:
            kappa: Thermal conductivity
            dx: Spatial step size
            cfl_factor: Safety factor (< 0.5 for stability)
            
        Returns:
            Maximum stable time step size
        """
        if not (0 < cfl_factor < 0.5):
            raise ValueError("CFL factor must be in (0, 0.5)")
            
        dt_max = cfl_factor * dx**2 / kappa
        return dt_max
    
    def solve(
        self,
        kappa: float,
        initial_condition: Callable[[np.ndarray], np.ndarray],
        boundary_conditions: Dict[str, Any],
        nx: int = 100,
        nt: int = 1000,
        final_time: float = 1.0,
        auto_timestep: bool = True,
        cfl_factor: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve 1D heat equation u_t = κ u_xx.
        
        Args:
            kappa: Thermal conductivity parameter
            initial_condition: Function u(x, 0) = initial_condition(x)
            boundary_conditions: Dict with keys 'left', 'right' and boundary values
            nx: Number of spatial grid points
            nt: Number of time steps (used if auto_timestep=False)
            final_time: Final simulation time T
            auto_timestep: If True, compute dt from CFL condition
            cfl_factor: CFL safety factor when auto_timestep=True
            
        Returns:
            Tuple of (solution, x_grid, t_grid)
            - solution: Array of shape (nt+1, nx) with u(x,t)
            - x_grid: Spatial grid points of shape (nx,)
            - t_grid: Time grid points of shape (nt+1,)
        """
        # Input validation
        if kappa <= 0:
            raise ValueError("Thermal conductivity must be positive")
        if nx < 3:
            raise ValueError("Need at least 3 spatial points")
        if final_time <= 0:
            raise ValueError("Final time must be positive")
        
        # Setup spatial grid
        self.nx = nx
        self.dx = self.L / (nx - 1)
        self.x_grid = np.linspace(0, self.L, nx)
        
        # Setup temporal grid
        if auto_timestep:
            self.dt = self.compute_stable_timestep(kappa, self.dx, cfl_factor)
            self.nt = int(np.ceil(final_time / self.dt))
            self.dt = final_time / self.nt  # Adjust to hit final_time exactly
        else:
            self.nt = nt
            self.dt = final_time / nt
            
        self.t_grid = np.linspace(0, final_time, self.nt + 1)
        
        # CFL stability check
        is_stable, stability_ratio = self.check_cfl_stability(kappa, self.dx, self.dt)
        if not is_stable and not auto_timestep:
            raise ValueError(
                f"Unstable time step. CFL ratio = {stability_ratio:.6f} > 0.5"
            )
            
        # Initialize solution array
        self.solution = np.zeros((self.nt + 1, self.nx))
        
        # Set initial condition
        self.solution[0, :] = initial_condition(self.x_grid)
        
        # Set boundary conditions for all time steps
        self._apply_boundary_conditions(boundary_conditions)
        
        # Time stepping with forward Euler
        r = kappa * self.dt / (self.dx**2)  # CFL number
        
        for n in range(self.nt):
            # Interior points: u_i^{n+1} = u_i^n + r(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
            self.solution[n+1, 1:-1] = (
                self.solution[n, 1:-1] + 
                r * (self.solution[n, 2:] - 2*self.solution[n, 1:-1] + self.solution[n, :-2])
            )
            
            # Reapply boundary conditions (for time-dependent BCs)
            self._apply_boundary_conditions_at_time(boundary_conditions, n+1)
            
        return self.solution, self.x_grid, self.t_grid
    
    def _apply_boundary_conditions(self, boundary_conditions: Dict[str, Any]) -> None:
        """Apply boundary conditions to all time levels."""
        for n in range(self.nt + 1):
            self._apply_boundary_conditions_at_time(boundary_conditions, n)
            
    def _apply_boundary_conditions_at_time(
        self, 
        boundary_conditions: Dict[str, Any], 
        time_idx: int
    ) -> None:
        """
        Apply boundary conditions at specific time index.
        
        Args:
            boundary_conditions: Dict with 'left' and 'right' boundary specs
            time_idx: Time index to apply BCs
        """
        # Left boundary (x = 0)
        if 'left' in boundary_conditions:
            bc_left = boundary_conditions['left']
            if callable(bc_left):
                self.solution[time_idx, 0] = bc_left(self.t_grid[time_idx])
            else:
                self.solution[time_idx, 0] = bc_left
                
        # Right boundary (x = L)
        if 'right' in boundary_conditions:
            bc_right = boundary_conditions['right']
            if callable(bc_right):
                self.solution[time_idx, -1] = bc_right(self.t_grid[time_idx])
            else:
                self.solution[time_idx, -1] = bc_right
    
    def get_observations(
        self, 
        sensor_locations: np.ndarray,
        time_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract sensor observations from solution.
        
        Args:
            sensor_locations: Array of x-coordinates for sensors
            time_points: Array of t-coordinates (default: all times)
            
        Returns:
            Array of observations with shape (nt, n_sensors)
        """
        if self.solution is None:
            raise RuntimeError("Must call solve() first")
            
        if time_points is None:
            time_points = self.t_grid
            
        # Find nearest grid points for sensors
        sensor_indices = []
        for x_sensor in sensor_locations:
            idx = np.argmin(np.abs(self.x_grid - x_sensor))
            sensor_indices.append(idx)
            
        # Find nearest grid points for time points
        time_indices = []
        for t_obs in time_points:
            idx = np.argmin(np.abs(self.t_grid - t_obs))
            time_indices.append(idx)
            
        # Extract observations
        observations = np.zeros((len(time_points), len(sensor_locations)))
        for i, t_idx in enumerate(time_indices):
            for j, x_idx in enumerate(sensor_indices):
                observations[i, j] = self.solution[t_idx, x_idx]
                
        return observations
    
    def compute_error_metrics(
        self, 
        exact_solution: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute error metrics against exact solution.
        
        Args:
            exact_solution: Function u_exact(x, t) returning exact solution
            
        Returns:
            Dict with RMSE and MAE error metrics
        """
        if self.solution is None:
            raise RuntimeError("Must call solve() first")
            
        # Compute exact solution on grid
        X, T = np.meshgrid(self.x_grid, self.t_grid)
        exact = exact_solution(X, T)
        
        # Compute errors
        error = self.solution - exact
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        return {
            'rmse': rmse,
            'mae': mae, 
            'max_error': max_error,
            'relative_rmse': rmse / np.mean(np.abs(exact))
        }
    
    def plot_solution(
        self, 
        time_snapshots: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot solution at different time snapshots.
        
        Args:
            time_snapshots: Times to plot (default: [0, T/4, T/2, 3T/4, T])
            save_path: Path to save plot (optional)
        """
        if self.solution is None:
            raise RuntimeError("Must call solve() first")
            
        if time_snapshots is None:
            time_snapshots = np.linspace(0, self.t_grid[-1], 5)
            
        plt.figure(figsize=(10, 6))
        
        for t_snap in time_snapshots:
            t_idx = np.argmin(np.abs(self.t_grid - t_snap))
            plt.plot(self.x_grid, self.solution[t_idx, :], 
                    label=f't = {t_snap:.3f}', linewidth=2)
                    
        plt.xlabel('Position x')
        plt.ylabel('Temperature u(x,t)')
        plt.title('Heat Equation Solution Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def demo_heat_solver():
    """Demonstration of heat equation solver with CFL checks."""
    print("🔥 Heat Equation Solver Demo")
    print("=" * 50)
    
    # Initialize solver
    solver = HeatEquationSolver(domain_length=1.0)
    
    # Problem setup
    kappa = 0.1  # Thermal conductivity
    
    # Initial condition: Gaussian pulse
    def initial_condition(x):
        return np.exp(-50 * (x - 0.5)**2)
    
    # Boundary conditions: Dirichlet (fixed temperature)
    boundary_conditions = {
        'left': 0.0,
        'right': 0.0
    }
    
    # Solve with automatic time stepping
    print(f"Solving heat equation with κ = {kappa}")
    print(f"Domain: [0, {solver.L}]")
    print("Boundary conditions: u(0,t) = u(L,t) = 0")
    print("Initial condition: Gaussian pulse at x = 0.5")
    
    solution, x_grid, t_grid = solver.solve(
        kappa=kappa,
        initial_condition=initial_condition,
        boundary_conditions=boundary_conditions,
        nx=100,
        final_time=2.0,
        auto_timestep=True,
        cfl_factor=0.4
    )
    
    # Display solver info
    print(f"\nSolver details:")
    print(f"  Grid points: {len(x_grid)} (Δx = {solver.dx:.4f})")
    print(f"  Time steps: {len(t_grid)-1} (Δt = {solver.dt:.6f})")
    print(f"  CFL number: {kappa * solver.dt / solver.dx**2:.4f}")
    print(f"  Final time: {t_grid[-1]:.3f}")
    
    # Test sensor observations
    sensor_locations = np.array([0.25, 0.5, 0.75])
    observations = solver.get_observations(sensor_locations)
    print(f"\nSensor observations shape: {observations.shape}")
    print(f"Final sensor readings: {observations[-1, :]}")
    
    # Plot solution
    solver.plot_solution()
    
    return solver


if __name__ == "__main__":
    demo_heat_solver()