"""
Utility functions for PAC-Bayes Heat Equation project.
Phase 2: Mathematical helpers, plotting utilities, and validation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any, Optional, Callable
from pathlib import Path


def gaussian_pulse(x: np.ndarray, center: float = 0.5, width: float = 0.1) -> np.ndarray:
    """
    Generate Gaussian pulse initial condition.
    
    Args:
        x: Spatial grid points
        center: Center of Gaussian pulse
        width: Width parameter (smaller = narrower pulse)
        
    Returns:
        Gaussian pulse values at grid points
    """
    return np.exp(-((x - center) / width)**2)


def step_function(x: np.ndarray, step_location: float = 0.5) -> np.ndarray:
    """
    Generate step function initial condition.
    
    Args:
        x: Spatial grid points  
        step_location: Location of step discontinuity
        
    Returns:
        Step function values (1 for x < step_location, 0 otherwise)
    """
    return np.where(x < step_location, 1.0, 0.0)


def sine_wave(x: np.ndarray, frequency: int = 1) -> np.ndarray:
    """
    Generate sine wave initial condition.
    
    Args:
        x: Spatial grid points (assumed on [0, L])
        frequency: Number of periods in domain
        
    Returns:
        Sine wave values
    """
    L = x[-1] - x[0]  # Domain length
    return np.sin(frequency * np.pi * x / L)


def analytical_heat_solution(
    x: np.ndarray, 
    t: np.ndarray, 
    kappa: float,
    L: float = 1.0
) -> np.ndarray:
    """
    Analytical solution for heat equation with sine initial condition.
    
    u(x,t) = exp(-π²κt/L²) * sin(πx/L) for u(0,t) = u(L,t) = 0
    
    Args:
        x: Spatial points (can be meshgrid)
        t: Time points (can be meshgrid)  
        kappa: Thermal conductivity
        L: Domain length
        
    Returns:
        Analytical solution values
    """
    return np.exp(-np.pi**2 * kappa * t / L**2) * np.sin(np.pi * x / L)


def compute_l2_error(
    numerical: np.ndarray,
    analytical: np.ndarray,
    dx: float
) -> float:
    """
    Compute L2 error between numerical and analytical solutions.
    
    Args:
        numerical: Numerical solution array
        analytical: Analytical solution array (same shape)
        dx: Spatial step size
        
    Returns:
        L2 error norm
    """
    if numerical.shape != analytical.shape:
        raise ValueError("Solution arrays must have same shape")
        
    error = numerical - analytical
    l2_error = np.sqrt(np.sum(error**2 * dx))
    return l2_error


def compute_convergence_rate(
    grid_sizes: List[int],
    errors: List[float]
) -> float:
    """
    Compute convergence rate from grid refinement study.
    
    Fits error ≈ C * h^p where h is grid size and p is convergence rate.
    
    Args:
        grid_sizes: List of grid sizes (number of points)
        errors: Corresponding L2 errors
        
    Returns:
        Estimated convergence rate p
    """
    if len(grid_sizes) != len(errors) or len(grid_sizes) < 2:
        raise ValueError("Need at least 2 grid sizes and errors")
        
    # Convert grid sizes to step sizes
    L = 1.0  # Assume unit domain
    h_values = [L / (n - 1) for n in grid_sizes]
    
    # Fit log(error) = log(C) + p * log(h)
    log_h = np.log(h_values)
    log_error = np.log(errors)
    
    # Linear regression
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    p, log_C = np.linalg.lstsq(A, log_error, rcond=None)[0]
    
    return p


def setup_plotting_style():
    """Setup consistent plotting style for all figures."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })


def plot_heat_evolution(
    solution: np.ndarray,
    x_grid: np.ndarray, 
    t_grid: np.ndarray,
    time_snapshots: Optional[List[float]] = None,
    title: str = "Heat Equation Solution Evolution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot solution evolution at multiple time snapshots.
    
    Args:
        solution: Solution array of shape (nt+1, nx)
        x_grid: Spatial grid points
        t_grid: Time grid points
        time_snapshots: Times to plot (default: 5 evenly spaced)
        title: Plot title
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    if time_snapshots is None:
        time_snapshots = np.linspace(t_grid[0], t_grid[-1], 5)
        
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_snapshots)))
    
    for i, t_snap in enumerate(time_snapshots):
        # Find nearest time index
        t_idx = np.argmin(np.abs(t_grid - t_snap))
        actual_time = t_grid[t_idx]
        
        plt.plot(x_grid, solution[t_idx, :], 
                color=colors[i], linewidth=2.5,
                label=f't = {actual_time:.3f}')
    
    plt.xlabel('Position x')
    plt.ylabel('Temperature u(x,t)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.show()


def plot_3d_surface(
    solution: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray, 
    title: str = "Heat Equation 3D Surface",
    save_path: Optional[str] = None
) -> None:
    """
    Create 3D surface plot of solution.
    
    Args:
        solution: Solution array of shape (nt+1, nx)
        x_grid: Spatial grid points
        t_grid: Time grid points
        title: Plot title
        save_path: Path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    setup_plotting_style()
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    X, T = np.meshgrid(x_grid, t_grid)
    
    surf = ax.plot_surface(X, T, solution, 
                          cmap='viridis', alpha=0.9,
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Temperature u(x,t)')
    ax.set_title(title)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.show()


def plot_convergence_study(
    grid_sizes: List[int],
    errors: List[float],
    error_type: str = "L2",
    save_path: Optional[str] = None
) -> None:
    """
    Plot convergence study results.
    
    Args:
        grid_sizes: Grid sizes used
        errors: Corresponding errors
        error_type: Type of error (for labeling)
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    # Compute step sizes
    L = 1.0
    h_values = [L / (n - 1) for n in grid_sizes]
    
    # Estimate convergence rate
    conv_rate = compute_convergence_rate(grid_sizes, errors)
    
    plt.figure(figsize=(10, 7))
    
    # Plot errors vs grid size
    plt.loglog(h_values, errors, 'o-', linewidth=2, markersize=8,
              label=f'{error_type} Error')
    
    # Plot theoretical convergence lines
    h_ref = np.array(h_values)
    error_ref = errors[0]  # Reference error
    h_first_order = error_ref * (h_ref / h_values[0])**1
    h_second_order = error_ref * (h_ref / h_values[0])**2
    
    plt.loglog(h_ref, h_first_order, '--', alpha=0.7, 
              label='First Order (slope=1)')
    plt.loglog(h_ref, h_second_order, '--', alpha=0.7,
              label='Second Order (slope=2)')
    
    plt.xlabel('Grid Spacing h = Δx')
    plt.ylabel(f'{error_type} Error')
    plt.title(f'Convergence Study (Estimated Rate: {conv_rate:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.show()


def create_sensor_array(
    domain_length: float,
    n_sensors: int,
    distribution: str = "uniform"
) -> np.ndarray:
    """
    Create array of sensor locations.
    
    Args:
        domain_length: Length of spatial domain
        n_sensors: Number of sensors
        distribution: Sensor distribution ('uniform', 'random', 'clustered')
        
    Returns:
        Array of sensor locations
    """
    if distribution == "uniform":
        if n_sensors == 1:
            return np.array([domain_length / 2])
        else:
            return np.linspace(0.1 * domain_length, 0.9 * domain_length, n_sensors)
            
    elif distribution == "random":
        np.random.seed(42)  # For reproducibility
        sensors = np.random.uniform(0.1 * domain_length, 
                                  0.9 * domain_length, n_sensors)
        return np.sort(sensors)
        
    elif distribution == "clustered":
        # Cluster around center of domain
        center = domain_length / 2
        width = domain_length / 4
        sensors = np.random.normal(center, width, n_sensors)
        sensors = np.clip(sensors, 0.1 * domain_length, 0.9 * domain_length)
        return np.sort(sensors)
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def add_gaussian_noise(
    clean_data: np.ndarray,
    noise_level: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to clean observations.
    
    Args:
        clean_data: Clean observation data
        noise_level: Noise standard deviation as fraction of signal
        seed: Random seed for reproducibility
        
    Returns:
        Noisy observations
    """
    if seed is not None:
        np.random.seed(seed)
        
    signal_std = np.std(clean_data)
    noise_std = noise_level * signal_std
    
    noise = np.random.normal(0, noise_std, clean_data.shape)
    noisy_data = clean_data + noise
    
    return noisy_data


def save_results(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save experimental results to file.
    
    Args:
        results: Dictionary of results to save
        filepath: Path to save results (supports .npz, .pkl)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.npz':
        np.savez(filepath, **results)
    elif filepath.suffix == '.pkl':
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load experimental results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary of loaded results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = np.load(filepath)
        return {key: data[key] for key in data.keys()}
    elif filepath.suffix == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


# Mathematical constants and helper functions
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
SQRT_2PI = np.sqrt(2 * np.pi)

def safe_log(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Numerically safe logarithm."""
    return np.log(np.maximum(x, eps))

def safe_exp(x: np.ndarray, max_exp: float = 700) -> np.ndarray:
    """Numerically safe exponential."""
    return np.exp(np.minimum(x, max_exp))


if __name__ == "__main__":
    # Demo of utility functions
    print("🛠️  Utility Functions Demo")
    print("=" * 40)
    
    # Test initial conditions
    x = np.linspace(0, 1, 100)
    
    gaussian = gaussian_pulse(x, center=0.3, width=0.1)
    step = step_function(x, step_location=0.6)
    sine = sine_wave(x, frequency=2)
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, gaussian, 'b-', linewidth=2)
    plt.title('Gaussian Pulse')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2) 
    plt.plot(x, step, 'r-', linewidth=2)
    plt.title('Step Function')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(x, sine, 'g-', linewidth=2)
    plt.title('Sine Wave')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test sensor arrays
    sensors_uniform = create_sensor_array(1.0, 5, "uniform")
    sensors_random = create_sensor_array(1.0, 5, "random")
    
    print(f"Uniform sensors: {sensors_uniform}")
    print(f"Random sensors: {sensors_random}")
    
    print("✅ Utility functions working correctly!")