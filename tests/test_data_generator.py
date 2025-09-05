"""Tests for data generator module."""

import pytest
import numpy as np
from src.data_generator import DataGenerator
from src.heat_solver import HeatSolver


class TestDataGenerator:
    """Test class for DataGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.domain = (0.0, 1.0)
        self.nx = 25
        self.dt = 0.001
        self.generator = DataGenerator(self.domain, self.nx, self.dt)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.domain == self.domain
        assert self.generator.nx == self.nx
        assert self.generator.dt == self.dt
        assert isinstance(self.generator.solver, HeatSolver)
    
    def test_conductivity_profiles(self):
        """Test generation of thermal conductivity profiles."""
        profiles = self.generator.generate_conductivity_profiles()
        
        # Check that all expected profiles are present
        expected_profiles = ['constant', 'linear', 'quadratic', 'sinusoidal',
                           'exponential', 'step', 'gaussian', 'layered', 'random_smooth']
        
        for profile_name in expected_profiles:
            assert profile_name in profiles
            
        # Test that profiles are callable and return correct shape
        x = np.linspace(0, 1, 25)
        for name, profile_func in profiles.items():
            result = profile_func(x)
            assert len(result) == len(x)
            assert np.all(result > 0)  # Conductivity should be positive
    
    def test_initial_conditions(self):
        """Test generation of initial conditions."""
        conditions = self.generator.generate_initial_conditions()
        
        expected_conditions = ['zero', 'constant', 'gaussian', 'sine', 'step', 'ramp', 'triangle']
        
        for condition_name in expected_conditions:
            assert condition_name in conditions
        
        # Test that conditions are callable and return correct shape
        x = np.linspace(0, 1, 25)
        for name, condition_func in conditions.items():
            result = condition_func(x)
            assert len(result) == len(x)
            assert np.all(np.isfinite(result))
    
    def test_source_terms(self):
        """Test generation of source terms."""
        sources = self.generator.generate_source_terms()
        
        expected_sources = ['none', 'constant', 'time_decay', 'gaussian', 'moving', 'oscillating']
        
        for source_name in expected_sources:
            assert source_name in sources
        
        # Test callable sources (excluding 'none')
        x = np.linspace(0, 1, 25)
        t = 0.5
        
        for name, source_func in sources.items():
            if source_func is not None:
                result = source_func(x, t)
                assert len(result) == len(x)
                assert np.all(np.isfinite(result))
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        dataset = self.generator.generate_dataset(
            conductivity_name='linear',
            initial_condition='gaussian',
            source_term='none',
            t_final=0.1,
            noise_level=0.02,
            n_realizations=3
        )
        
        # Check required fields
        required_fields = ['solution', 'times', 'observations', 'sensor_locations',
                          'observation_times', 'spatial_grid', 'true_conductivity',
                          'conductivity_function', 'initial_condition', 'boundary_conditions',
                          'parameters']
        
        for field in required_fields:
            assert field in dataset
        
        # Check shapes
        solution = dataset['solution']
        observations = dataset['observations']
        
        assert solution.ndim == 2
        assert observations.ndim == 3  # (n_realizations, n_times, n_sensors)
        assert observations.shape[0] == 3  # n_realizations
        assert observations.shape[1] == solution.shape[0]  # n_times
        
        # Check that conductivity is positive
        assert np.all(dataset['true_conductivity'] > 0)
        
        # Check parameter storage
        params = dataset['parameters']
        assert params['conductivity_name'] == 'linear'
        assert params['initial_condition'] == 'gaussian'
        assert params['noise_level'] == 0.02
    
    def test_multiple_scenarios(self):
        """Test generation of multiple scenarios."""
        scenarios = [
            {'conductivity_name': 'constant', 'noise_level': 0.01},
            {'conductivity_name': 'quadratic', 'noise_level': 0.02},
        ]
        
        base_config = {'t_final': 0.05, 'n_realizations': 2}
        
        datasets = self.generator.generate_multiple_scenarios(scenarios, base_config)
        
        assert len(datasets) == 2
        
        # Check that base config is applied
        for i, dataset in enumerate(datasets):
            assert dataset['parameters']['t_final'] == 0.05
            assert dataset['observations'].shape[0] == 2
            assert dataset['scenario_id'] == i
        
        # Check that scenario-specific config is applied
        assert datasets[0]['parameters']['noise_level'] == 0.01
        assert datasets[1]['parameters']['noise_level'] == 0.02
    
    def test_visualization_no_error(self):
        """Test that visualization runs without error."""
        dataset = self.generator.generate_dataset(
            conductivity_name='sinusoidal',
            t_final=0.05,
            n_realizations=2
        )
        
        # This should not raise an exception
        try:
            fig = self.generator.visualize_dataset(dataset, figsize=(10, 6))
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")
    
    def test_save_load_dataset(self, tmp_path):
        """Test saving and loading datasets."""
        dataset = self.generator.generate_dataset(
            conductivity_name='exponential',
            t_final=0.05
        )
        
        # Save dataset
        filename = tmp_path / "test_dataset.npz"
        self.generator.save_dataset(dataset, str(filename))
        
        # Check that file was created
        assert filename.exists()
        
        # Load dataset
        loaded_dataset = self.generator.load_dataset(str(filename))
        
        # Check that key arrays are preserved
        np.testing.assert_array_equal(dataset['solution'], loaded_dataset['solution'])
        np.testing.assert_array_equal(dataset['true_conductivity'], loaded_dataset['true_conductivity'])
        
        # Parameters should be preserved
        original_params = dataset['parameters']
        loaded_params = loaded_dataset['parameters'].item()  # Convert from numpy scalar
        
        assert loaded_params['conductivity_name'] == original_params['conductivity_name']
        assert loaded_params['noise_level'] == original_params['noise_level']
    
    def test_sensor_location_defaults(self):
        """Test default sensor location generation."""
        dataset = self.generator.generate_dataset(
            conductivity_name='constant',
            sensor_locations=None  # Use defaults
        )
        
        sensor_locs = dataset['sensor_locations']
        
        # Should have 10 sensors by default
        assert len(sensor_locs) == 10
        
        # Should be within domain boundaries with margin
        domain_min, domain_max = self.domain
        margin = 0.1 * (domain_max - domain_min)
        
        assert np.all(sensor_locs >= domain_min + margin)
        assert np.all(sensor_locs <= domain_max - margin)
        
        # Should be sorted
        assert np.all(sensor_locs[1:] >= sensor_locs[:-1])
    
    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.05])
    def test_noise_levels(self, noise_level):
        """Test different noise levels."""
        dataset = self.generator.generate_dataset(
            conductivity_name='constant',
            t_final=0.05,
            noise_level=noise_level,
            n_realizations=5
        )
        
        observations = dataset['observations']
        
        if noise_level == 0.0:
            # All realizations should be identical with no noise
            for i in range(1, observations.shape[0]):
                np.testing.assert_array_almost_equal(
                    observations[0], observations[i], decimal=10
                )
        else:
            # With noise, realizations should differ
            std_across_realizations = np.std(observations, axis=0)
            assert np.mean(std_across_realizations) > 0
    
    def test_boundary_conditions_effect(self):
        """Test effect of different boundary conditions."""
        bc_zero = (0.0, 0.0)
        bc_nonzero = (1.0, 2.0)
        
        dataset_zero = self.generator.generate_dataset(
            boundary_conditions=bc_zero,
            t_final=0.05
        )
        
        dataset_nonzero = self.generator.generate_dataset(
            boundary_conditions=bc_nonzero,
            t_final=0.05
        )
        
        # Solutions should be different
        solution_zero = dataset_zero['solution']
        solution_nonzero = dataset_nonzero['solution']
        
        # Boundary values should match prescribed conditions
        assert np.allclose(solution_zero[:, 0], 0.0)
        assert np.allclose(solution_zero[:, -1], 0.0)
        assert np.allclose(solution_nonzero[:, 0], 1.0)
        assert np.allclose(solution_nonzero[:, -1], 2.0)