"""
Physics Test Fixtures

Provides test data, mock datasets, and reference results for physics testing.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Base directory for fixtures
FIXTURES_DIR = Path(__file__).parent


def create_quantum_test_data():
    """Create quantum physics test data."""
    data = {
        'hamiltonians': {
            'pauli_x': [[0, 1], [1, 0]],
            'pauli_y': [[0, -1j], [1j, 0]],
            'pauli_z': [[1, 0], [0, -1]],
            'harmonic_oscillator_2level': [[0.5, 0], [0, 1.5]],
            'hydrogen_2level': [[-13.6, 0.1], [0.1, -3.4]]
        },
        'quantum_states': {
            'spin_up': [1, 0],
            'spin_down': [0, 1],
            'plus_state': [0.7071067811865476, 0.7071067811865476],
            'minus_state': [0.7071067811865476, -0.7071067811865476]
        },
        'expected_energies': {
            'harmonic_oscillator': [0.5, 1.5, 2.5, 3.5, 4.5],
            'hydrogen_levels': [-13.6, -3.4, -1.51, -0.85, -0.54]
        },
        'entangled_states': {
            'bell_state_phi_plus': [0.7071067811865476, 0, 0, 0.7071067811865476],
            'bell_state_phi_minus': [0.7071067811865476, 0, 0, -0.7071067811865476],
            'bell_state_psi_plus': [0, 0.7071067811865476, 0.7071067811865476, 0],
            'bell_state_psi_minus': [0, 0.7071067811865476, -0.7071067811865476, 0]
        }
    }
    
    # Save as JSON (convert complex numbers to real for JSON compatibility)
    json_data = {}
    for key, value in data.items():
        if key == 'hamiltonians':
            json_data[key] = {}
            for ham_name, ham_matrix in value.items():
                # Convert complex to [real, imag] format
                json_matrix = []
                for row in ham_matrix:
                    json_row = []
                    for element in row:
                        if isinstance(element, complex):
                            json_row.append([element.real, element.imag])
                        else:
                            json_row.append([float(element), 0.0])
                    json_matrix.append(json_row)
                json_data[key][ham_name] = json_matrix
        else:
            json_data[key] = value
    
    with open(FIXTURES_DIR / 'quantum_test_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return data


def create_materials_test_data():
    """Create materials physics test data."""
    data = {
        'crystal_structures': {
            'fcc_aluminum': {
                'lattice_parameters': {'a': 4.05, 'b': 4.05, 'c': 4.05, 'alpha': 90, 'beta': 90, 'gamma': 90},
                'space_group': 'Fm3m',
                'atoms': ['Al'],
                'positions': [[0, 0, 0]],
                'density': 2700,  # kg/m³
                'coordination_number': 12
            },
            'bcc_iron': {
                'lattice_parameters': {'a': 2.87, 'b': 2.87, 'c': 2.87, 'alpha': 90, 'beta': 90, 'gamma': 90},
                'space_group': 'Im3m',
                'atoms': ['Fe'],
                'positions': [[0, 0, 0], [0.5, 0.5, 0.5]],
                'density': 7874,  # kg/m³
                'coordination_number': 8
            }
        },
        'elastic_constants': {
            'aluminum': {
                'C11': 108e9,  # Pa
                'C12': 62e9,   # Pa
                'C44': 28e9    # Pa
            },
            'iron': {
                'C11': 230e9,  # Pa
                'C12': 135e9,  # Pa
                'C44': 116e9   # Pa
            }
        },
        'thermal_properties': {
            'aluminum': {
                'thermal_conductivity': 237,  # W/m·K
                'specific_heat': 897,         # J/kg·K
                'thermal_expansion': 23.1e-6  # 1/K
            },
            'iron': {
                'thermal_conductivity': 80.2,  # W/m·K
                'specific_heat': 449,          # J/kg·K
                'thermal_expansion': 11.8e-6   # 1/K
            }
        },
        'phase_diagrams': {
            'iron_carbon': {
                'phases': ['austenite', 'ferrite', 'cementite'],
                'transition_temperatures': [727, 910, 1147],  # °C
                'carbon_concentrations': [0.0, 0.77, 6.67]   # wt%
            }
        }
    }
    
    with open(FIXTURES_DIR / 'materials_test_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def create_astrophysics_test_data():
    """Create astrophysics test data."""
    data = {
        'celestial_bodies': {
            'earth': {
                'mass_kg': 5.972e24,
                'radius_m': 6.371e6,
                'orbital_period_s': 365.25 * 24 * 3600,
                'orbital_radius_m': 1.496e11,
                'surface_gravity': 9.81  # m/s²
            },
            'sun': {
                'mass_kg': 1.989e30,
                'radius_m': 6.96e8,
                'luminosity_W': 3.828e26,
                'surface_temperature_K': 5778,
                'core_temperature_K': 15.7e6
            },
            'jupiter': {
                'mass_kg': 1.898e27,
                'radius_m': 6.9911e7,
                'orbital_period_s': 11.862 * 365.25 * 24 * 3600,
                'orbital_radius_m': 7.785e11
            }
        },
        'stellar_evolution': {
            'main_sequence_lifetimes': {
                'masses_solar': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
                'lifetimes_years': [1e13, 5.6e10, 1e10, 1e9, 6.5e7, 3.2e7, 7e6]
            },
            'stellar_types': {
                'O': {'mass_range': [15, 90], 'temperature_range': [30000, 50000], 'color': 'blue'},
                'B': {'mass_range': [2.1, 16], 'temperature_range': [10000, 30000], 'color': 'blue-white'},
                'A': {'mass_range': [1.4, 2.1], 'temperature_range': [7500, 10000], 'color': 'white'},
                'F': {'mass_range': [1.04, 1.4], 'temperature_range': [6000, 7500], 'color': 'yellow-white'},
                'G': {'mass_range': [0.8, 1.04], 'temperature_range': [5200, 6000], 'color': 'yellow'},
                'K': {'mass_range': [0.45, 0.8], 'temperature_range': [3700, 5200], 'color': 'orange'},
                'M': {'mass_range': [0.08, 0.45], 'temperature_range': [2400, 3700], 'color': 'red'}
            }
        },
        'cosmological_parameters': {
            'planck_2018': {
                'H0': 67.4,        # km/s/Mpc
                'Omega_m': 0.315,
                'Omega_lambda': 0.685,
                'Omega_b': 0.049,
                'sigma_8': 0.811,
                'n_s': 0.965
            },
            'wmap_9year': {
                'H0': 69.3,        # km/s/Mpc
                'Omega_m': 0.287,
                'Omega_lambda': 0.713,
                'Omega_b': 0.0463,
                'sigma_8': 0.821,
                'n_s': 0.972
            }
        },
        'galaxy_properties': {
            'milky_way': {
                'mass_kg': 1.5e42,     # Total mass including dark matter
                'stellar_mass_kg': 6e10 * 1.989e30,  # 6e10 solar masses
                'disk_radius_pc': 26000,
                'disk_thickness_pc': 1000,
                'rotation_velocity_km_s': 220
            },
            'andromeda': {
                'mass_kg': 1.5e42,
                'stellar_mass_kg': 1e11 * 1.989e30,  # 1e11 solar masses
                'disk_radius_pc': 35000,
                'rotation_velocity_km_s': 250
            }
        }
    }
    
    with open(FIXTURES_DIR / 'astrophysics_test_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def create_experimental_test_data():
    """Create experimental physics test data."""
    # Generate sample experimental datasets
    np.random.seed(42)  # For reproducibility
    
    # Linear relationship dataset
    n_points = 100
    x_linear = np.linspace(0, 10, n_points)
    y_linear_true = 2.5 * x_linear + 1.2
    y_linear_measured = y_linear_true + np.random.normal(0, 0.1, n_points)
    
    # Exponential decay dataset
    x_exp = np.linspace(0, 5, n_points)
    y_exp_true = 10 * np.exp(-x_exp / 2.0)
    y_exp_measured = y_exp_true * (1 + np.random.normal(0, 0.05, n_points))
    
    # Gaussian peak dataset
    x_gauss = np.linspace(-5, 5, n_points)
    y_gauss_true = 5 * np.exp(-0.5 * ((x_gauss - 0.5) / 1.2)**2)
    y_gauss_measured = y_gauss_true + np.random.normal(0, 0.1, n_points)
    
    # Power law dataset
    x_power = np.linspace(1, 10, n_points)
    y_power_true = 3 * x_power**(-1.5)
    y_power_measured = y_power_true * (1 + np.random.normal(0, 0.1, n_points))
    
    data = {
        'calibration_standards': {
            'voltage_standards': [
                {'nominal_V': 1.000, 'uncertainty_V': 0.001, 'measured_V': 1.002},
                {'nominal_V': 5.000, 'uncertainty_V': 0.002, 'measured_V': 4.998},
                {'nominal_V': 10.000, 'uncertainty_V': 0.005, 'measured_V': 10.003}
            ],
            'temperature_standards': [
                {'nominal_K': 273.15, 'uncertainty_K': 0.01, 'measured_K': 273.16},
                {'nominal_K': 373.15, 'uncertainty_K': 0.02, 'measured_K': 373.13},
                {'nominal_K': 473.15, 'uncertainty_K': 0.05, 'measured_K': 473.18}
            ]
        },
        'sample_datasets': {
            'linear_relationship': {
                'x_values': x_linear.tolist(),
                'y_measured': y_linear_measured.tolist(),
                'y_true': y_linear_true.tolist(),
                'fit_type': 'linear',
                'true_parameters': [2.5, 1.2]  # slope, intercept
            },
            'exponential_decay': {
                'x_values': x_exp.tolist(),
                'y_measured': y_exp_measured.tolist(),
                'y_true': y_exp_true.tolist(),
                'fit_type': 'exponential',
                'true_parameters': [10.0, -0.5, 0.0]  # amplitude, decay constant, offset
            },
            'gaussian_peak': {
                'x_values': x_gauss.tolist(),
                'y_measured': y_gauss_measured.tolist(),
                'y_true': y_gauss_true.tolist(),
                'fit_type': 'gaussian',
                'true_parameters': [5.0, 0.5, 1.2]  # amplitude, center, width
            },
            'power_law': {
                'x_values': x_power.tolist(),
                'y_measured': y_power_measured.tolist(),
                'y_true': y_power_true.tolist(),
                'fit_type': 'power_law',
                'true_parameters': [3.0, -1.5]  # amplitude, exponent
            }
        },
        'detector_configurations': {
            'high_precision': {
                'efficiency': 0.95,
                'noise_level': 0.001,
                'bandwidth_Hz': 10000,
                'sampling_rate_Hz': 50000,
                'deadtime_s': 1e-6
            },
            'standard': {
                'efficiency': 0.85,
                'noise_level': 0.01,
                'bandwidth_Hz': 1000,
                'sampling_rate_Hz': 10000,
                'deadtime_s': 1e-5
            },
            'low_cost': {
                'efficiency': 0.70,
                'noise_level': 0.05,
                'bandwidth_Hz': 100,
                'sampling_rate_Hz': 1000,
                'deadtime_s': 1e-4
            }
        },
        'uncertainty_budgets': {
            'voltage_measurement': {
                'type_a_sources': ['statistical_fluctuation', 'readout_noise'],
                'type_b_sources': ['calibration_uncertainty', 'temperature_drift', 'EMI'],
                'type_a_values': [0.001, 0.0005],  # V
                'type_b_values': [0.002, 0.001, 0.0005]  # V
            },
            'temperature_measurement': {
                'type_a_sources': ['sensor_noise', 'averaging_uncertainty'],
                'type_b_sources': ['calibration_uncertainty', 'self_heating', 'radiation'],
                'type_a_values': [0.01, 0.005],  # K
                'type_b_values': [0.02, 0.01, 0.005]  # K
            }
        }
    }
    
    with open(FIXTURES_DIR / 'experimental_test_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def create_computational_test_data():
    """Create computational physics test data."""
    # Generate numerical test cases
    data = {
        'ode_test_cases': {
            'harmonic_oscillator': {
                'equation': 'd²x/dt² = -ω²x',
                'omega': 1.0,
                'initial_conditions': [1.0, 0.0],  # [x(0), dx/dt(0)]
                'analytical_solution': 'x(t) = cos(ωt)',
                'time_span': [0, 6.283185307179586],  # 2π
                'expected_final_state': [1.0, 0.0]
            },
            'exponential_decay': {
                'equation': 'dy/dt = -λy',
                'lambda': 0.5,
                'initial_conditions': [1.0],
                'analytical_solution': 'y(t) = exp(-λt)',
                'time_span': [0, 2.0],
                'expected_final_state': [0.36787944117144233]  # exp(-1)
            },
            'pendulum': {
                'equation': 'd²θ/dt² = -(g/L)sin(θ)',
                'g': 9.81,
                'L': 1.0,
                'initial_conditions': [0.1, 0.0],  # Small angle approximation
                'analytical_solution': 'θ(t) ≈ θ₀cos(√(g/L)t)',
                'time_span': [0, 2.0],
                'expected_period': 2.006  # ≈ 2π√(L/g)
            }
        },
        'integration_test_cases': {
            'polynomial': {
                'function': 'x³',
                'bounds': [0, 2],
                'analytical_result': 4.0,  # ∫₀² x³ dx = x⁴/4|₀² = 4
                'tolerance': 1e-6
            },
            'exponential': {
                'function': 'exp(-x²)',
                'bounds': [-3, 3],
                'analytical_result': 1.7724538509055159,  # √π
                'tolerance': 1e-3
            },
            'oscillatory': {
                'function': 'sin(10x)',
                'bounds': [0, 3.141592653589793],  # π
                'analytical_result': 0.0,  # ∫₀^π sin(10x) dx
                'tolerance': 1e-6
            }
        },
        'linear_algebra_test_cases': {
            'eigenvalue_problems': [
                {
                    'matrix': [[2, 1], [1, 2]],
                    'expected_eigenvalues': [3, 1],
                    'expected_eigenvectors': [[0.7071067811865476, 0.7071067811865476], 
                                           [0.7071067811865476, -0.7071067811865476]]
                },
                {
                    'matrix': [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                    'expected_eigenvalues': [1, 2, 3],
                    'expected_eigenvectors': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                }
            ],
            'matrix_operations': {
                'matrix_mult': {
                    'A': [[1, 2], [3, 4]],
                    'B': [[5, 6], [7, 8]],
                    'expected_result': [[19, 22], [43, 50]]
                },
                'matrix_inv': {
                    'A': [[4, 7], [2, 6]],
                    'expected_inverse': [[0.6, -0.7], [-0.2, 0.4]]
                }
            }
        },
        'fft_test_cases': {
            'pure_sine_wave': {
                'frequency_Hz': 10,
                'sampling_rate_Hz': 100,
                'duration_s': 1.0,
                'amplitude': 1.0,
                'expected_peak_frequency': 10
            },
            'two_tone_signal': {
                'frequencies_Hz': [5, 25],
                'amplitudes': [1.0, 0.5],
                'sampling_rate_Hz': 100,
                'duration_s': 2.0,
                'expected_peak_frequencies': [5, 25]
            }
        },
        'monte_carlo_test_cases': {
            'pi_estimation': {
                'method': 'quarter_circle',
                'bounds': [[0, 1], [0, 1]],
                'analytical_result': 0.7853981633974483,  # π/4
                'min_samples': 10000,
                'expected_accuracy': 0.01
            },
            'gaussian_integral': {
                'method': 'normal_distribution',
                'bounds': [[-3, 3]],
                'analytical_result': 1.0,  # ∫_{-∞}^∞ (1/√(2π))e^(-x²/2) dx
                'min_samples': 50000,
                'expected_accuracy': 0.05
            }
        }
    }
    
    with open(FIXTURES_DIR / 'computational_test_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def load_test_data(data_type: str) -> Dict[str, Any]:
    """Load test data of specified type."""
    filename = f"{data_type}_test_data.json"
    filepath = FIXTURES_DIR / filename
    
    if not filepath.exists():
        # Create the data if it doesn't exist
        if data_type == 'quantum':
            return create_quantum_test_data()
        elif data_type == 'materials':
            return create_materials_test_data()
        elif data_type == 'astrophysics':
            return create_astrophysics_test_data()
        elif data_type == 'experimental':
            return create_experimental_test_data()
        elif data_type == 'computational':
            return create_computational_test_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_mock_time_series(n_points: int = 1000, 
                            sampling_rate: float = 1000.0,
                            components: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Generate mock time series data for testing."""
    if components is None:
        components = [(10.0, 1.0), (25.0, 0.5)]  # [(frequency, amplitude), ...]
    
    dt = 1.0 / sampling_rate
    t = np.arange(n_points) * dt
    
    signal = np.zeros(n_points)
    for freq, amp in components:
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    noise_level = 0.1 * np.std(signal)
    noise = np.random.normal(0, noise_level, n_points)
    noisy_signal = signal + noise
    
    return {
        'time': t,
        'clean_signal': signal,
        'noisy_signal': noisy_signal,
        'noise': noise,
        'sampling_rate': sampling_rate,
        'components': components,
        'snr_db': 20 * np.log10(np.std(signal) / np.std(noise))
    }


def generate_mock_spectrum(n_points: int = 1024,
                         frequency_range: Tuple[float, float] = (0, 500)) -> Dict[str, Any]:
    """Generate mock frequency spectrum for testing."""
    frequencies = np.linspace(frequency_range[0], frequency_range[1], n_points)
    
    # Create spectrum with multiple peaks
    spectrum = np.zeros(n_points)
    
    # Add peaks at specific frequencies
    peak_frequencies = [50, 150, 300]
    peak_amplitudes = [1.0, 0.5, 0.3]
    peak_widths = [5, 10, 15]
    
    for freq, amp, width in zip(peak_frequencies, peak_amplitudes, peak_widths):
        if frequency_range[0] <= freq <= frequency_range[1]:
            peak_idx = np.argmin(np.abs(frequencies - freq))
            spectrum += amp * np.exp(-0.5 * ((frequencies - freq) / width)**2)
    
    # Add background noise
    background = 0.01 * np.random.random(n_points)
    spectrum += background
    
    return {
        'frequencies': frequencies,
        'spectrum': spectrum,
        'peak_frequencies': peak_frequencies,
        'peak_amplitudes': peak_amplitudes,
        'background_level': 0.01
    }


if __name__ == '__main__':
    """Create all test data files."""
    print("Creating physics test fixtures...")
    
    # Create all test data
    quantum_data = create_quantum_test_data()
    materials_data = create_materials_test_data()
    astrophysics_data = create_astrophysics_test_data()
    experimental_data = create_experimental_test_data()
    computational_data = create_computational_test_data()
    
    print(f"Created quantum test data with {len(quantum_data)} categories")
    print(f"Created materials test data with {len(materials_data)} categories")
    print(f"Created astrophysics test data with {len(astrophysics_data)} categories")
    print(f"Created experimental test data with {len(experimental_data)} categories")
    print(f"Created computational test data with {len(computational_data)} categories")
    
    print("Physics test fixtures created successfully!")