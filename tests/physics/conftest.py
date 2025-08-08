"""
Physics Test Configuration

Provides fixtures, mock data, and configuration for physics testing.
This conftest.py file contains shared test configuration that is automatically
loaded by pytest for all physics tests.
"""

import pytest
import numpy as np
import tempfile
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


@pytest.fixture
def physics_test_config() -> Dict[str, Any]:
    """
    Provide comprehensive physics test configuration.
    
    Returns:
        Dictionary containing test configurations for all physics domains
    """
    return {
        'quantum_physics': {
            'test_hamiltonians': [
                {'name': 'harmonic_oscillator', 'omega': 1.0, 'mass': 1.0},
                {'name': 'hydrogen_atom', 'Z': 1, 'rydberg': 13.6},
                {'name': 'infinite_square_well', 'L': 1.0, 'n_states': 10}
            ],
            'expected_energies': {
                'harmonic_oscillator': [0.5, 1.5, 2.5, 3.5, 4.5],
                'hydrogen_atom': [-13.6, -3.4, -1.51, -0.85, -0.54],
                'infinite_square_well': [9.87, 39.48, 88.83, 157.91, 246.74]
            },
            'test_particles': [
                {'name': 'electron', 'mass': 9.109e-31, 'charge': -1.602e-19},
                {'name': 'proton', 'mass': 1.673e-27, 'charge': 1.602e-19},
                {'name': 'neutron', 'mass': 1.675e-27, 'charge': 0.0}
            ],
            'quantum_states': {
                'spin_half': [[1, 0], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)]],
                'entangled_states': [[1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]]
            }
        },
        'computational_physics': {
            'numerical_methods': [
                'euler', 'runge_kutta_4', 'verlet', 'leapfrog'
            ],
            'test_systems': [
                {
                    'name': 'simple_harmonic_oscillator',
                    'initial_conditions': [1.0, 0.0],
                    'parameters': {'omega': 1.0},
                    'analytical_solution': lambda t: np.cos(t)
                },
                {
                    'name': 'pendulum',
                    'initial_conditions': [np.pi/4, 0.0],
                    'parameters': {'g': 9.81, 'L': 1.0},
                    'analytical_solution': None  # Nonlinear, numerical only
                }
            ],
            'benchmark_results': {
                'monte_carlo_pi': {'trials': 1000000, 'expected': np.pi, 'tolerance': 0.01},
                'random_walk': {'steps': 10000, 'expected_displacement': 0.0, 'tolerance': 10.0}
            },
            'algorithms': [
                'molecular_dynamics', 'monte_carlo', 'finite_element', 'spectral_methods'
            ]
        },
        'materials_physics': {
            'crystal_structures': [
                {'name': 'fcc', 'lattice_parameter': 4.05, 'coordination': 12},
                {'name': 'bcc', 'lattice_parameter': 2.87, 'coordination': 8},
                {'name': 'hcp', 'lattice_parameter': 2.95, 'coordination': 12}
            ],
            'test_materials': [
                {
                    'name': 'aluminum',
                    'density': 2700,  # kg/m³
                    'youngs_modulus': 70e9,  # Pa
                    'crystal_structure': 'fcc'
                },
                {
                    'name': 'iron',
                    'density': 7874,  # kg/m³
                    'youngs_modulus': 211e9,  # Pa
                    'crystal_structure': 'bcc'
                }
            ],
            'mechanical_properties': {
                'stress_strain_curves': [
                    # [strain, stress] pairs
                    [[0.0, 0.001, 0.002, 0.003], [0.0, 70e6, 140e6, 210e6]]
                ],
                'thermal_properties': {
                    'thermal_conductivity': [237, 80.2],  # W/m·K for Al, Fe
                    'specific_heat': [897, 449]  # J/kg·K for Al, Fe
                }
            }
        },
        'astrophysics': {
            'celestial_bodies': [
                {
                    'name': 'earth',
                    'mass': 5.972e24,  # kg
                    'radius': 6.371e6,  # m
                    'orbital_period': 365.25 * 24 * 3600  # s
                },
                {
                    'name': 'sun',
                    'mass': 1.989e30,  # kg
                    'radius': 6.96e8,  # m
                    'luminosity': 3.828e26  # W
                }
            ],
            'stellar_evolution': {
                'main_sequence_lifetimes': {
                    'solar_masses': [0.5, 1.0, 2.0, 10.0],
                    'lifetimes_years': [5.6e10, 1.0e10, 1.0e9, 3.2e7]
                }
            },
            'cosmological_parameters': {
                'hubble_constant': 70.0,  # km/s/Mpc
                'dark_matter_fraction': 0.26,
                'dark_energy_fraction': 0.69,
                'critical_density': 9.47e-27  # kg/m³
            },
            'simulation_parameters': {
                'n_body_test': {
                    'n_particles': [100, 1000, 10000],
                    'time_steps': [1e-3, 1e-4, 1e-5],
                    'box_size': 1.0
                }
            }
        },
        'experimental_physics': {
            'measurement_uncertainties': {
                'systematic_errors': [0.01, 0.02, 0.05],  # Fractional errors
                'statistical_errors': [0.001, 0.005, 0.01]  # Fractional errors
            },
            'calibration_standards': [
                {'quantity': 'length', 'standard': 'meter', 'uncertainty': 1e-12},
                {'quantity': 'time', 'standard': 'second', 'uncertainty': 1e-15},
                {'quantity': 'mass', 'standard': 'kilogram', 'uncertainty': 1e-8}
            ],
            'detector_properties': {
                'efficiency': [0.9, 0.85, 0.95],
                'resolution': [0.1, 0.05, 0.2],  # Fractional resolution
                'noise_level': [0.01, 0.005, 0.02]  # RMS noise
            },
            'data_analysis': {
                'fitting_functions': ['linear', 'quadratic', 'exponential', 'gaussian'],
                'statistical_tests': ['chi_squared', 'kolmogorov_smirnov', 't_test'],
                'sample_sizes': [100, 1000, 10000]
            }
        },
        'performance_benchmarks': {
            'computation_times': {
                'matrix_multiplication': {'size': [100, 1000, 10000], 'max_time': [0.1, 10.0, 1000.0]},
                'fft': {'size': [1024, 8192, 65536], 'max_time': [0.01, 0.1, 1.0]},
                'integration': {'points': [1000, 10000, 100000], 'max_time': [0.1, 1.0, 10.0]}
            },
            'memory_usage': {
                'max_memory_mb': 1000,
                'efficiency_threshold': 0.8
            },
            'accuracy_thresholds': {
                'relative_error': 1e-6,
                'absolute_error': 1e-12
            }
        }
    }


@pytest.fixture
def temp_physics_data_dir():
    """
    Create a temporary directory for physics test data.
    
    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_quantum_hamiltonian():
    """
    Create a mock quantum mechanical Hamiltonian matrix.
    
    Returns:
        2D numpy array representing a simple Hamiltonian
    """
    # Simple 2x2 Hamiltonian (Pauli-Z matrix)
    return np.array([[1.0, 0.0], [0.0, -1.0]])


@pytest.fixture
def mock_molecular_structure():
    """
    Create a mock molecular structure for testing.
    
    Returns:
        Dictionary containing atomic positions and types
    """
    return {
        'atoms': ['H', 'H'],
        'positions': np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),  # H2 molecule
        'bonds': [(0, 1)],
        'total_energy': -1.17,  # Hartree
        'forces': np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    }


@pytest.fixture
def mock_experimental_data():
    """
    Create mock experimental data with realistic noise.
    
    Returns:
        Dictionary containing experimental measurements
    """
    # Generate some sample data with noise
    x = np.linspace(0, 10, 100)
    y_true = 2.0 * x + 1.0  # Linear relationship
    noise = np.random.normal(0, 0.1, len(x))
    y_measured = y_true + noise
    
    return {
        'x': x,
        'y_measured': y_measured,
        'y_true': y_true,
        'uncertainty': np.full_like(x, 0.1),
        'metadata': {
            'measurement_date': datetime.now().isoformat(),
            'instrument': 'mock_detector',
            'calibration_date': '2024-01-01'
        }
    }


@pytest.fixture
def mock_physics_simulation_results():
    """
    Create mock physics simulation results for testing.
    
    Returns:
        Dictionary containing simulation outputs
    """
    return {
        'time_series': {
            'time': np.linspace(0, 10, 1000),
            'position': np.sin(np.linspace(0, 10, 1000)),
            'velocity': np.cos(np.linspace(0, 10, 1000)),
            'energy': np.full(1000, 0.5)  # Conserved energy
        },
        'final_state': {
            'position': 0.0,
            'velocity': 1.0,
            'energy': 0.5
        },
        'convergence': {
            'iterations': 1000,
            'residual': 1e-8,
            'converged': True
        }
    }


@pytest.fixture(scope="session")
def physics_test_database():
    """
    Create a temporary database for physics test results.
    
    Returns:
        Path to temporary SQLite database
    """
    import sqlite3
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Initialize database with physics test tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE physics_test_results (
            id TEXT PRIMARY KEY,
            test_category TEXT NOT NULL,
            test_name TEXT NOT NULL,
            parameters TEXT,
            results TEXT,
            status TEXT,
            created_at TEXT,
            execution_time REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE physics_benchmarks (
            id TEXT PRIMARY KEY,
            algorithm TEXT NOT NULL,
            input_size INTEGER,
            execution_time REAL,
            memory_usage REAL,
            accuracy REAL,
            created_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def physics_tolerance_config():
    """
    Provide tolerance configuration for physics calculations.
    
    Returns:
        Dictionary with tolerance values for different physics domains
    """
    return {
        'quantum_mechanics': {
            'energy_tolerance': 1e-6,
            'wavefunction_tolerance': 1e-8,
            'overlap_tolerance': 1e-10
        },
        'classical_mechanics': {
            'position_tolerance': 1e-6,
            'velocity_tolerance': 1e-6,
            'energy_conservation': 1e-8
        },
        'thermodynamics': {
            'temperature_tolerance': 1e-3,
            'pressure_tolerance': 1e-2,
            'entropy_tolerance': 1e-6
        },
        'electromagnetism': {
            'field_tolerance': 1e-6,
            'potential_tolerance': 1e-8,
            'current_tolerance': 1e-9
        },
        'numerical_methods': {
            'integration_tolerance': 1e-8,
            'convergence_tolerance': 1e-10,
            'residual_tolerance': 1e-12
        }
    }


# Physics test utilities
class PhysicsTestUtils:
    """Utility class for physics testing operations."""
    
    @staticmethod
    def assert_energy_conservation(initial_energy: float, final_energy: float, 
                                 tolerance: float = 1e-8) -> bool:
        """Assert that energy is conserved within tolerance."""
        return abs(initial_energy - final_energy) < tolerance
    
    @staticmethod
    def assert_wavefunction_normalized(wavefunction: np.ndarray, 
                                     tolerance: float = 1e-8) -> bool:
        """Assert that wavefunction is properly normalized."""
        norm = np.sum(np.abs(wavefunction)**2)
        return abs(norm - 1.0) < tolerance
    
    @staticmethod
    def generate_test_matrix(size: int, matrix_type: str = 'random') -> np.ndarray:
        """Generate test matrices for physics calculations."""
        if matrix_type == 'random':
            return np.random.random((size, size))
        elif matrix_type == 'hermitian':
            A = np.random.random((size, size)) + 1j * np.random.random((size, size))
            return A + A.conj().T
        elif matrix_type == 'unitary':
            Q, _ = np.linalg.qr(np.random.random((size, size)))
            return Q
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
    
    @staticmethod
    def calculate_relative_error(computed: float, analytical: float) -> float:
        """Calculate relative error between computed and analytical values."""
        if analytical == 0:
            return abs(computed)
        return abs(computed - analytical) / abs(analytical)


@pytest.fixture
def physics_utils():
    """Provide physics test utilities."""
    return PhysicsTestUtils()


# Async fixtures for concurrent physics simulations
@pytest.fixture
async def async_physics_simulator():
    """Provide an async physics simulator for testing."""
    
    class AsyncPhysicsSimulator:
        def __init__(self):
            self.running = False
        
        async def run_simulation(self, duration: float, dt: float = 0.01):
            """Run a mock physics simulation asynchronously."""
            self.running = True
            steps = int(duration / dt)
            
            for step in range(steps):
                # Simulate some computation time
                await asyncio.sleep(0.001)
                
                # Yield control periodically
                if step % 100 == 0:
                    await asyncio.sleep(0)
            
            self.running = False
            return {
                'steps': steps,
                'final_time': duration,
                'status': 'completed'
            }
        
        async def stop_simulation(self):
            """Stop the simulation."""
            self.running = False
    
    import asyncio
    return AsyncPhysicsSimulator()


# Custom pytest markers for physics tests
def pytest_configure(config):
    """Configure custom pytest markers for physics tests."""
    config.addinivalue_line(
        "markers", "quantum: mark test as quantum physics test"
    )
    config.addinivalue_line(
        "markers", "computational: mark test as computational physics test"
    )
    config.addinivalue_line(
        "markers", "materials: mark test as materials physics test"
    )
    config.addinivalue_line(
        "markers", "astrophysics: mark test as astrophysics test"
    )
    config.addinivalue_line(
        "markers", "experimental: mark test as experimental physics test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )