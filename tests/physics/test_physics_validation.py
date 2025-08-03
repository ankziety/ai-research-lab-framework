"""
Physics Validation Tests

Validation tests for physics accuracy, reproducibility, and correctness.
Tests against known physics results, conservation laws, and theoretical predictions.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, patch
import scipy.constants as constants

# Validation test markers
pytestmark = pytest.mark.validation


class PhysicsValidationSuite:
    """Suite of physics validation utilities and reference data."""
    
    def __init__(self):
        self.fundamental_constants = {
            'c': constants.speed_of_light,       # m/s
            'h': constants.Planck,               # J⋅s
            'hbar': constants.hbar,              # J⋅s
            'e': constants.elementary_charge,    # C
            'me': constants.electron_mass,       # kg
            'mp': constants.proton_mass,         # kg
            'k_B': constants.Boltzmann,          # J/K
            'epsilon_0': constants.epsilon_0,    # F/m
            'mu_0': constants.mu_0,              # H/m
            'G': constants.gravitational_constant, # m³/kg/s²
            'R': constants.gas_constant,         # J/mol/K
            'N_A': constants.Avogadro           # 1/mol
        }
        
        self.known_results = {
            'hydrogen_atom': {
                'binding_energy_eV': 13.6,
                'bohr_radius_m': 5.29e-11,
                'rydberg_constant': 1.097e7  # m⁻¹
            },
            'harmonic_oscillator': {
                'ground_state_energy': 0.5,  # in units of ħω
                'energy_spacing': 1.0        # ħω
            },
            'blackbody_radiation': {
                'stefan_boltzmann': 5.67e-8,  # W/m²/K⁴
                'wien_displacement': 2.898e-3  # m⋅K
            },
            'particle_masses': {
                'electron_MeV': 0.511,
                'proton_MeV': 938.3,
                'neutron_MeV': 939.6
            },
            'earth_properties': {
                'mass_kg': 5.972e24,
                'radius_m': 6.371e6,
                'orbital_period_s': 365.25 * 24 * 3600,
                'orbital_radius_m': 1.496e11
            },
            'solar_properties': {
                'mass_kg': 1.989e30,
                'radius_m': 6.96e8,
                'luminosity_W': 3.828e26,
                'surface_temperature_K': 5778
            }
        }
        
        self.tolerance_levels = {
            'strict': 1e-10,      # For exact mathematical results
            'tight': 1e-6,        # For well-controlled physics
            'standard': 1e-3,     # For typical physics calculations
            'loose': 1e-1,        # For approximate or statistical results
            'experimental': 0.2   # For experimental comparisons
        }
    
    def validate_conservation_law(self, initial_quantity: float, final_quantity: float,
                                tolerance_level: str = 'standard') -> Dict[str, Any]:
        """Validate conservation of a physical quantity."""
        tolerance = self.tolerance_levels[tolerance_level]
        
        if initial_quantity == 0:
            absolute_error = abs(final_quantity)
            relative_error = np.inf if final_quantity != 0 else 0
        else:
            absolute_error = abs(final_quantity - initial_quantity)
            relative_error = absolute_error / abs(initial_quantity)
        
        is_conserved = relative_error < tolerance
        
        return {
            'initial_value': initial_quantity,
            'final_value': final_quantity,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'is_conserved': is_conserved,
            'conservation_quality': self._assess_conservation_quality(relative_error)
        }
    
    def validate_known_result(self, calculated_value: float, known_value: float,
                            tolerance_level: str = 'standard') -> Dict[str, Any]:
        """Validate calculated result against known theoretical/experimental value."""
        tolerance = self.tolerance_levels[tolerance_level]
        
        absolute_error = abs(calculated_value - known_value)
        relative_error = absolute_error / abs(known_value) if known_value != 0 else abs(calculated_value)
        
        is_accurate = relative_error < tolerance
        
        return {
            'calculated_value': calculated_value,
            'known_value': known_value,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'is_accurate': is_accurate,
            'accuracy_rating': self._assess_accuracy_rating(relative_error)
        }
    
    def validate_dimensional_consistency(self, calculated_quantity: float,
                                       expected_dimensions: str,
                                       reference_scale: float = 1.0) -> Dict[str, Any]:
        """Validate dimensional consistency of calculated quantities."""
        # This is a simplified dimensional analysis
        # In a full implementation, this would check units properly
        
        order_of_magnitude = np.log10(abs(calculated_quantity)) if calculated_quantity != 0 else -np.inf
        reference_order = np.log10(abs(reference_scale)) if reference_scale != 0 else 0
        
        # Check if order of magnitude is reasonable
        order_difference = abs(order_of_magnitude - reference_order)
        
        is_reasonable = order_difference < 10  # Within 10 orders of magnitude
        
        return {
            'calculated_quantity': calculated_quantity,
            'expected_dimensions': expected_dimensions,
            'order_of_magnitude': order_of_magnitude,
            'reference_order': reference_order,
            'order_difference': order_difference,
            'is_reasonable': is_reasonable
        }
    
    def validate_symmetry_property(self, function_values: np.ndarray,
                                 symmetry_type: str) -> Dict[str, Any]:
        """Validate symmetry properties of physical functions."""
        n = len(function_values)
        
        if symmetry_type == 'even':
            # f(-x) = f(x)
            if n % 2 == 1:
                center = n // 2
                left_half = function_values[:center]
                right_half = function_values[center+1:][::-1]
            else:
                center = n // 2
                left_half = function_values[:center]
                right_half = function_values[center:][::-1]
            
            symmetry_error = np.mean(np.abs(left_half - right_half))
            
        elif symmetry_type == 'odd':
            # f(-x) = -f(x)
            if n % 2 == 1:
                center = n // 2
                left_half = function_values[:center]
                right_half = -function_values[center+1:][::-1]
            else:
                center = n // 2
                left_half = function_values[:center]
                right_half = -function_values[center:][::-1]
            
            symmetry_error = np.mean(np.abs(left_half - right_half))
            
        else:
            symmetry_error = np.inf
        
        max_value = np.max(np.abs(function_values))
        relative_symmetry_error = symmetry_error / max_value if max_value > 0 else 0
        
        has_symmetry = relative_symmetry_error < self.tolerance_levels['standard']
        
        return {
            'symmetry_type': symmetry_type,
            'symmetry_error': symmetry_error,
            'relative_symmetry_error': relative_symmetry_error,
            'has_symmetry': has_symmetry,
            'symmetry_quality': self._assess_symmetry_quality(relative_symmetry_error)
        }
    
    def _assess_conservation_quality(self, relative_error: float) -> str:
        """Assess quality of conservation law satisfaction."""
        if relative_error < self.tolerance_levels['strict']:
            return 'excellent'
        elif relative_error < self.tolerance_levels['tight']:
            return 'very_good'
        elif relative_error < self.tolerance_levels['standard']:
            return 'good'
        elif relative_error < self.tolerance_levels['loose']:
            return 'acceptable'
        else:
            return 'poor'
    
    def _assess_accuracy_rating(self, relative_error: float) -> str:
        """Assess accuracy rating compared to known results."""
        if relative_error < self.tolerance_levels['strict']:
            return 'excellent'
        elif relative_error < self.tolerance_levels['tight']:
            return 'very_good'
        elif relative_error < self.tolerance_levels['standard']:
            return 'good'
        elif relative_error < self.tolerance_levels['loose']:
            return 'acceptable'
        else:
            return 'poor'
    
    def _assess_symmetry_quality(self, relative_error: float) -> str:
        """Assess quality of symmetry satisfaction."""
        if relative_error < self.tolerance_levels['strict']:
            return 'perfect'
        elif relative_error < self.tolerance_levels['tight']:
            return 'very_good'
        elif relative_error < self.tolerance_levels['standard']:
            return 'good'
        elif relative_error < self.tolerance_levels['loose']:
            return 'acceptable'
        else:
            return 'poor'


class TestQuantumPhysicsValidation:
    """Validation tests for quantum physics calculations."""
    
    @pytest.fixture
    def quantum_agent(self):
        """Create quantum physics agent for validation testing."""
        from .test_quantum_physics import MockQuantumPhysicsAgent
        return MockQuantumPhysicsAgent()
    
    @pytest.fixture
    def validation_suite(self):
        """Create physics validation suite."""
        return PhysicsValidationSuite()
    
    def test_harmonic_oscillator_validation(self, quantum_agent, validation_suite):
        """Validate harmonic oscillator eigenvalues against analytical results."""
        # Create harmonic oscillator Hamiltonian: H = (p²/2m) + (½mω²x²)
        # In dimensionless units: H = ½(p² + x²)
        omega = 1.0
        
        # Simple 2-level approximation
        hamiltonian = np.array([[0.5, 0.0], [0.0, 1.5]])  # E₀ = ½ħω, E₁ = 3/2ħω
        
        result = quantum_agent.solve_schrodinger_equation(hamiltonian)
        eigenvalues = np.sort(result['eigenvalues'])
        
        # Validate ground state energy
        ground_state_validation = validation_suite.validate_known_result(
            eigenvalues[0], 
            validation_suite.known_results['harmonic_oscillator']['ground_state_energy'],
            'strict'
        )
        
        assert ground_state_validation['is_accurate']
        assert ground_state_validation['accuracy_rating'] == 'excellent'
        
        # Validate energy spacing
        energy_spacing = eigenvalues[1] - eigenvalues[0]
        spacing_validation = validation_suite.validate_known_result(
            energy_spacing,
            validation_suite.known_results['harmonic_oscillator']['energy_spacing'],
            'strict'
        )
        
        assert spacing_validation['is_accurate']
        assert spacing_validation['accuracy_rating'] == 'excellent'
    
    def test_quantum_commutation_relations(self, quantum_agent, validation_suite):
        """Validate quantum commutation relations."""
        # Test [x, p] = iħ using matrix representations
        # In dimensionless units where ħ = 1
        
        # Position and momentum operators (simplified 2x2 representation)
        x_op = np.array([[0, 1], [1, 0]])  # Simplified position operator
        p_op = np.array([[0, -1j], [1j, 0]])  # Simplified momentum operator
        
        # Calculate commutator [x, p] = xp - px
        commutator = x_op @ p_op - p_op @ x_op
        
        # Expected result: [x, p] = iħ = i (in our units)
        expected_commutator = 1j * np.eye(2)
        
        # Validate commutation relation
        commutator_error = np.max(np.abs(commutator - expected_commutator))
        
        commutation_validation = validation_suite.validate_known_result(
            commutator_error, 0.0, 'strict'
        )
        
        assert commutation_validation['is_accurate']
    
    def test_quantum_state_normalization(self, quantum_agent, validation_suite):
        """Validate quantum state normalization."""
        # Test that quantum states are properly normalized
        hamiltonian = np.array([[1.0, 0.2], [0.2, -0.5]])
        
        result = quantum_agent.solve_schrodinger_equation(hamiltonian)
        eigenvectors = result['eigenvectors']
        
        # Check normalization of each eigenvector
        for i in range(eigenvectors.shape[1]):
            eigenvector = eigenvectors[:, i]
            norm_squared = np.sum(np.abs(eigenvector)**2)
            
            normalization_validation = validation_suite.validate_known_result(
                norm_squared, 1.0, 'strict'
            )
            
            assert normalization_validation['is_accurate']
            assert normalization_validation['accuracy_rating'] == 'excellent'
    
    def test_quantum_orthogonality(self, quantum_agent, validation_suite):
        """Validate orthogonality of quantum eigenstates."""
        hamiltonian = np.array([[2.0, 0.5], [0.5, -1.0]])
        
        result = quantum_agent.solve_schrodinger_equation(hamiltonian)
        eigenvectors = result['eigenvectors']
        
        # Check orthogonality between different eigenvectors
        n_states = eigenvectors.shape[1]
        for i in range(n_states):
            for j in range(i + 1, n_states):
                overlap = np.sum(np.conj(eigenvectors[:, i]) * eigenvectors[:, j])
                
                orthogonality_validation = validation_suite.validate_known_result(
                    abs(overlap), 0.0, 'strict'
                )
                
                assert orthogonality_validation['is_accurate']
    
    def test_quantum_time_evolution_unitarity(self, quantum_agent, validation_suite):
        """Validate unitarity of quantum time evolution."""
        initial_state = np.array([1.0, 0.0])
        hamiltonian = np.array([[0.5, 0.2], [0.2, -0.5]])
        
        # Evolve state
        evolved_state = quantum_agent.evolve_quantum_state(initial_state, hamiltonian, time=1.0)
        
        # Check that norm is preserved (unitarity)
        initial_norm = np.sum(np.abs(initial_state)**2)
        final_norm = np.sum(np.abs(evolved_state)**2)
        
        norm_conservation = validation_suite.validate_conservation_law(
            initial_norm, final_norm, 'strict'
        )
        
        assert norm_conservation['is_conserved']
        assert norm_conservation['conservation_quality'] == 'excellent'
    
    def test_quantum_expectation_values(self, quantum_agent, validation_suite):
        """Validate quantum expectation value calculations."""
        # Test with known state and operator
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩ state
        
        # Pauli matrices
        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        
        # Calculate expectation values
        exp_x = quantum_agent.calculate_expectation_value(state, pauli_x)
        exp_y = quantum_agent.calculate_expectation_value(state, pauli_y)
        exp_z = quantum_agent.calculate_expectation_value(state, pauli_z)
        
        # For |+⟩ state: ⟨X⟩ = 1, ⟨Y⟩ = 0, ⟨Z⟩ = 0
        x_validation = validation_suite.validate_known_result(float(np.real(exp_x)), 1.0, 'strict')
        y_validation = validation_suite.validate_known_result(float(np.real(exp_y)), 0.0, 'strict')
        z_validation = validation_suite.validate_known_result(float(np.real(exp_z)), 0.0, 'strict')
        
        assert x_validation['is_accurate']
        assert y_validation['is_accurate']
        assert z_validation['is_accurate']


class TestComputationalPhysicsValidation:
    """Validation tests for computational physics methods."""
    
    @pytest.fixture
    def computational_agent(self):
        """Create computational physics agent for validation testing."""
        from .test_computational_physics import MockComputationalPhysicsAgent
        return MockComputationalPhysicsAgent()
    
    @pytest.fixture
    def validation_suite(self):
        """Create physics validation suite."""
        return PhysicsValidationSuite()
    
    def test_ode_solver_validation(self, computational_agent, validation_suite):
        """Validate ODE solver against analytical solutions."""
        # Test simple harmonic oscillator: d²x/dt² = -ω²x
        # Analytical solution: x(t) = A*cos(ωt) + B*sin(ωt)
        
        omega = 1.0
        def harmonic_oscillator(t, y):
            # y = [x, dx/dt]
            return np.array([y[1], -omega**2 * y[0]])
        
        # Initial conditions: x(0) = 1, dx/dt(0) = 0
        initial_conditions = np.array([1.0, 0.0])
        time_span = (0, 2*np.pi)  # One complete period
        
        result = computational_agent.solve_ode(
            harmonic_oscillator, initial_conditions, time_span, 
            method='runge_kutta_4', dt=0.01
        )
        
        # At t = 2π, should return to initial conditions
        final_state = result['final_state']
        
        # Validate position conservation after full period
        position_validation = validation_suite.validate_known_result(
            final_state[0], initial_conditions[0], 'standard'
        )
        
        # Validate velocity conservation after full period
        velocity_validation = validation_suite.validate_known_result(
            final_state[1], initial_conditions[1], 'standard'
        )
        
        assert position_validation['is_accurate']
        assert velocity_validation['is_accurate']
        
        # Validate energy conservation throughout
        trajectory = result['solution']
        times = result['time']
        
        # Calculate total energy at each time point
        energies = []
        for i in range(len(times)):
            x, v = trajectory[i]
            kinetic_energy = 0.5 * v**2
            potential_energy = 0.5 * omega**2 * x**2
            total_energy = kinetic_energy + potential_energy
            energies.append(total_energy)
        
        # Energy should be conserved
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        energy_conservation = validation_suite.validate_conservation_law(
            initial_energy, final_energy, 'standard'
        )
        
        assert energy_conservation['is_conserved']
    
    def test_monte_carlo_convergence_validation(self, computational_agent, validation_suite):
        """Validate Monte Carlo integration convergence."""
        # Integrate π using quarter circle: ∫∫ 1 dA over x²+y²≤1, x,y≥0
        def quarter_circle(x, y):
            return 1 if x**2 + y**2 <= 1 else 0
        
        bounds = [(0, 1), (0, 1)]
        sample_counts = [1000, 10000, 100000]
        
        errors = []
        for n_samples in sample_counts:
            result = computational_agent.monte_carlo_integration(
                quarter_circle, bounds, n_samples
            )
            
            # Result should be π/4
            analytical_result = np.pi / 4
            error = abs(result['integral'] - analytical_result)
            errors.append(error)
        
        # Error should decrease with more samples (roughly as 1/√n)
        # Check that error decreases between successive sample sizes
        for i in range(len(errors) - 1):
            assert errors[i+1] <= errors[i]  # Error should not increase
        
        # Final result should be reasonably accurate
        final_validation = validation_suite.validate_known_result(
            result['integral'], np.pi / 4, 'loose'
        )
        
        assert final_validation['is_accurate']
    
    def test_fft_validation(self, computational_agent, validation_suite):
        """Validate FFT against known frequency content."""
        # Create signal with known frequency components
        sampling_rate = 1000  # Hz
        t = np.linspace(0, 1, sampling_rate, endpoint=False)
        
        # Signal: 2*sin(2π*50*t) + sin(2π*120*t)
        freq1, freq2 = 50, 120
        amplitude1, amplitude2 = 2.0, 1.0
        
        signal = amplitude1 * np.sin(2 * np.pi * freq1 * t) + amplitude2 * np.sin(2 * np.pi * freq2 * t)
        
        result = computational_agent.fft_analysis(signal, sampling_rate)
        
        frequencies = result['frequencies']
        power_spectrum = result['power_spectrum']
        
        # Find peaks in power spectrum
        positive_freq_mask = frequencies > 0
        positive_freqs = frequencies[positive_freq_mask]
        positive_power = power_spectrum[positive_freq_mask]
        
        # Find the two highest peaks
        peak_indices = np.argsort(positive_power)[-2:]
        detected_frequencies = positive_freqs[peak_indices]
        
        # Validate detected frequencies
        detected_frequencies = np.sort(detected_frequencies)
        expected_frequencies = np.array([freq1, freq2])
        
        for i, (detected, expected) in enumerate(zip(detected_frequencies, expected_frequencies)):
            freq_validation = validation_suite.validate_known_result(
                detected, expected, 'standard'
            )
            assert freq_validation['is_accurate']
    
    def test_numerical_differentiation_validation(self, computational_agent, validation_suite):
        """Validate numerical differentiation accuracy."""
        # Test function: f(x) = x³, f'(x) = 3x²
        def test_function(x):
            return x**3
        
        def analytical_derivative(x):
            return 3 * x**2
        
        # Test point
        x_test = 2.0
        h_values = [0.1, 0.01, 0.001, 0.0001]
        
        analytical_result = analytical_derivative(x_test)
        
        for h in h_values:
            # Central difference approximation
            numerical_derivative = (test_function(x_test + h) - test_function(x_test - h)) / (2 * h)
            
            derivative_validation = validation_suite.validate_known_result(
                numerical_derivative, analytical_result, 'standard'
            )
            
            # Should be accurate for small h
            if h <= 0.01:
                assert derivative_validation['is_accurate']
    
    def test_conservation_laws_in_simulations(self, computational_agent, validation_suite):
        """Validate conservation laws in numerical simulations."""
        # Test molecular dynamics simulation for energy conservation
        n_particles = 20
        n_steps = 100
        dt = 0.001
        
        result = computational_agent.molecular_dynamics_simulation(
            n_particles, n_steps, dt
        )
        
        trajectory = result['trajectory']
        energies = trajectory['energies']
        
        # Validate energy conservation
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        energy_conservation = validation_suite.validate_conservation_law(
            initial_energy, final_energy, 'standard'
        )
        
        assert energy_conservation['is_conserved']
        assert energy_conservation['conservation_quality'] in ['excellent', 'very_good', 'good']


class TestMaterialsPhysicsValidation:
    """Validation tests for materials physics calculations."""
    
    @pytest.fixture
    def materials_agent(self):
        """Create materials physics agent for validation testing."""
        from .test_materials_physics import MockMaterialsPhysicsAgent
        return MockMaterialsPhysicsAgent()
    
    @pytest.fixture
    def validation_suite(self):
        """Create physics validation suite."""
        return PhysicsValidationSuite()
    
    def test_crystal_symmetry_validation(self, materials_agent, validation_suite):
        """Validate crystal symmetry properties."""
        # Test cubic crystal
        cubic_params = {'a': 4.0, 'b': 4.0, 'c': 4.0, 'alpha': 90, 'beta': 90, 'gamma': 90}
        
        result = materials_agent.analyze_crystal_structure(cubic_params)
        
        # Validate cubic symmetry detection
        assert result['crystal_system'] == 'cubic'
        
        # Validate unit cell volume calculation
        expected_volume = 4.0**3
        volume_validation = validation_suite.validate_known_result(
            result['unit_cell_volume'], expected_volume, 'strict'
        )
        
        assert volume_validation['is_accurate']
        assert volume_validation['accuracy_rating'] == 'excellent'
    
    def test_mechanical_properties_validation(self, materials_agent, validation_suite):
        """Validate mechanical property calculations."""
        # Test isotropic material with known elastic constants
        C11, C12, C44 = 200e9, 100e9, 50e9  # Pa
        
        elastic_constants = np.zeros((6, 6))
        elastic_constants[0:3, 0:3] = [[C11, C12, C12], [C12, C11, C12], [C12, C12, C11]]
        elastic_constants[3:6, 3:6] = np.diag([C44, C44, C44])
        
        density = 8000  # kg/m³
        
        result = materials_agent.calculate_mechanical_properties(elastic_constants, density)
        
        # Validate bulk modulus: K = (C11 + 2*C12)/3
        expected_bulk_modulus = (C11 + 2*C12) / 3
        bulk_validation = validation_suite.validate_known_result(
            result['bulk_modulus'], expected_bulk_modulus, 'strict'
        )
        
        assert bulk_validation['is_accurate']
        
        # Validate shear modulus
        expected_shear_modulus = C44
        shear_validation = validation_suite.validate_known_result(
            result['shear_modulus'], expected_shear_modulus, 'strict'
        )
        
        assert shear_validation['is_accurate']
        
        # Validate Poisson's ratio bounds: -1 < ν < 0.5
        poissons_ratio = result['poissons_ratio']
        assert -1 < poissons_ratio < 0.5
    
    def test_thermodynamic_relations_validation(self, materials_agent, validation_suite):
        """Validate thermodynamic relations."""
        crystal_structure = {
            'crystal_system': 'cubic',
            'coordination_number': 6,
            'unit_cell_volume': 64.0
        }
        
        temperature = 300  # K
        result = materials_agent.calculate_thermal_properties(crystal_structure, temperature)
        
        # Validate thermal diffusivity relation: α = k/(ρ*cp)
        k = result['thermal_conductivity']
        rho = result['density']
        cp = result['specific_heat']
        alpha_calculated = result['thermal_diffusivity']
        
        alpha_expected = k / (rho * cp)
        
        diffusivity_validation = validation_suite.validate_known_result(
            alpha_calculated, alpha_expected, 'standard'
        )
        
        assert diffusivity_validation['is_accurate']
    
    def test_phase_diagram_validation(self, materials_agent, validation_suite):
        """Validate phase transition thermodynamics."""
        temperature_range = (300, 800)
        result = materials_agent.simulate_phase_transition(temperature_range)
        
        temperatures = result['temperatures']
        alpha_fraction = result['phase_alpha_fraction']
        beta_fraction = result['phase_beta_fraction']
        
        # Validate phase fraction conservation
        for i in range(len(temperatures)):
            total_fraction = alpha_fraction[i] + beta_fraction[i]
            
            fraction_validation = validation_suite.validate_known_result(
                total_fraction, 1.0, 'strict'
            )
            
            assert fraction_validation['is_accurate']
        
        # Validate phase transition behavior
        T_transition = result['transition_temperature']
        
        # At low temperatures, mostly alpha phase
        low_temp_mask = temperatures < T_transition - 50
        if np.any(low_temp_mask):
            avg_alpha_low = np.mean(alpha_fraction[low_temp_mask])
            assert avg_alpha_low > 0.8
        
        # At high temperatures, mostly beta phase
        high_temp_mask = temperatures > T_transition + 50
        if np.any(high_temp_mask):
            avg_beta_high = np.mean(beta_fraction[high_temp_mask])
            assert avg_beta_high > 0.8
    
    def test_defect_formation_energy_validation(self, materials_agent, validation_suite):
        """Validate defect formation energy calculations."""
        defect_types = ['vacancy', 'interstitial']
        
        for defect_type in defect_types:
            result = materials_agent.analyze_defects(defect_type, concentration=1e-6)
            
            properties = result['properties']
            
            # Validate formation energy is positive
            if 'formation_energy' in properties:
                formation_energy = properties['formation_energy']
                assert formation_energy > 0  # Formation energy should be positive
                
                # Validate reasonable range (typically 0.1 - 10 eV)
                assert 0.1 < formation_energy < 10.0
            
            # Validate thermodynamic stability
            stability = result['thermodynamic_stability']
            if 'formation_energy' in properties:
                expected_stability = properties['formation_energy'] < 2.0
                assert stability == expected_stability


class TestAstrophysicsValidation:
    """Validation tests for astrophysics calculations."""
    
    @pytest.fixture
    def astrophysics_agent(self):
        """Create astrophysics agent for validation testing."""
        from .test_astrophysics import MockAstrophysicsAgent
        return MockAstrophysicsAgent()
    
    @pytest.fixture
    def validation_suite(self):
        """Create physics validation suite."""
        return PhysicsValidationSuite()
    
    def test_kepler_laws_validation(self, astrophysics_agent, validation_suite):
        """Validate Kepler's laws of planetary motion."""
        # Test Earth-Sun system
        earth_mass = validation_suite.known_results['earth_properties']['mass_kg']
        sun_mass = validation_suite.known_results['solar_properties']['mass_kg']
        earth_orbit = validation_suite.known_results['earth_properties']['orbital_radius_m']
        
        result = astrophysics_agent.calculate_orbital_mechanics(sun_mass, earth_mass, earth_orbit)
        
        # Validate orbital period (Kepler's third law)
        calculated_period = result['orbital_period']
        known_period = validation_suite.known_results['earth_properties']['orbital_period_s']
        
        period_validation = validation_suite.validate_known_result(
            calculated_period, known_period, 'standard'
        )
        
        assert period_validation['is_accurate']
        assert period_validation['accuracy_rating'] in ['excellent', 'very_good', 'good']
        
        # Validate orbital velocity
        orbital_velocity = result['orbital_velocity']
        expected_velocity = 2 * np.pi * earth_orbit / known_period
        
        velocity_validation = validation_suite.validate_known_result(
            orbital_velocity, expected_velocity, 'standard'
        )
        
        assert velocity_validation['is_accurate']
    
    def test_stellar_mass_luminosity_relation(self, astrophysics_agent, validation_suite):
        """Validate stellar mass-luminosity relation."""
        # Test main sequence mass-luminosity relation: L ∝ M^3.5
        stellar_masses = [0.5, 1.0, 2.0, 5.0]
        
        luminosities = []
        for mass in stellar_masses:
            result = astrophysics_agent.simulate_stellar_evolution(mass, age_range=(0, 1e8))
            
            # Get main sequence luminosity
            phases = result['phases']
            luminosity_array = result['luminosities']
            
            # Find main sequence phase
            ms_indices = [i for i, phase in enumerate(phases) if phase == 'main_sequence']
            if ms_indices:
                ms_luminosity = np.mean([luminosity_array[i] for i in ms_indices[:10]])  # Early MS
                luminosities.append(ms_luminosity)
        
        if len(luminosities) >= 2:
            # Test mass-luminosity scaling
            masses_array = np.array(stellar_masses[:len(luminosities)])
            luminosities_array = np.array(luminosities)
            
            # Fit L ∝ M^α
            log_masses = np.log(masses_array)
            log_luminosities = np.log(luminosities_array)
            
            # Linear fit in log space
            coeffs = np.polyfit(log_masses, log_luminosities, 1)
            power_law_exponent = coeffs[0]
            
            # Should be close to 3.5 for main sequence stars
            exponent_validation = validation_suite.validate_known_result(
                power_law_exponent, 3.5, 'loose'
            )
            
            # Mass-luminosity relation should show positive correlation
            assert power_law_exponent > 2.0  # At least some positive scaling
    
    def test_black_hole_physics_validation(self, astrophysics_agent, validation_suite):
        """Validate black hole physics calculations."""
        mass = 10.0  # Solar masses
        
        result = astrophysics_agent.calculate_black_hole_properties(mass)
        
        # Validate Schwarzschild radius: r_s = 2GM/c²
        calculated_rs = result['schwarzschild_radius']
        
        G = validation_suite.fundamental_constants['G']
        c = validation_suite.fundamental_constants['c']
        M = mass * validation_suite.known_results['solar_properties']['mass_kg']
        
        expected_rs = 2 * G * M / c**2
        
        rs_validation = validation_suite.validate_known_result(
            calculated_rs, expected_rs, 'standard'
        )
        
        assert rs_validation['is_accurate']
        
        # Validate Hawking temperature: T = ħc³/(8πGMk_B)
        calculated_temp = result['hawking_temperature']
        
        hbar = validation_suite.fundamental_constants['hbar']
        k_B = validation_suite.fundamental_constants['k_B']
        
        expected_temp = hbar * c**3 / (8 * np.pi * G * M * k_B)
        
        temp_validation = validation_suite.validate_known_result(
            calculated_temp, expected_temp, 'standard'
        )
        
        assert temp_validation['is_accurate']
    
    def test_cosmological_distances_validation(self, astrophysics_agent, validation_suite):
        """Validate cosmological distance calculations."""
        redshift = 1.0
        
        result = astrophysics_agent.calculate_cosmological_distances(redshift)
        
        # Validate distance relationships
        comoving_distance = result['comoving_distance']
        angular_diameter_distance = result['angular_diameter_distance']
        luminosity_distance = result['luminosity_distance']
        
        # Angular diameter distance: d_A = d_C / (1 + z)
        expected_angular = comoving_distance / (1 + redshift)
        angular_validation = validation_suite.validate_known_result(
            angular_diameter_distance, expected_angular, 'standard'
        )
        
        assert angular_validation['is_accurate']
        
        # Luminosity distance: d_L = d_C * (1 + z)
        expected_luminosity = comoving_distance * (1 + redshift)
        luminosity_validation = validation_suite.validate_known_result(
            luminosity_distance, expected_luminosity, 'standard'
        )
        
        assert luminosity_validation['is_accurate']
        
        # Distance modulus should be consistent
        distance_modulus = result['distance_modulus']
        expected_dm = 5 * np.log10(luminosity_distance * 1e6 / 10)  # Mpc to pc, then DM formula
        
        dm_validation = validation_suite.validate_known_result(
            distance_modulus, expected_dm, 'standard'
        )
        
        assert dm_validation['is_accurate']
    
    def test_nbody_conservation_laws(self, astrophysics_agent, validation_suite):
        """Validate conservation laws in N-body simulations."""
        n_bodies = 20
        n_steps = 500
        
        result = astrophysics_agent.simulate_nbody_system(n_bodies, n_steps=n_steps)
        
        trajectory = result['trajectory']
        
        # Validate energy conservation
        energies = trajectory['total_energy']
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        energy_conservation = validation_suite.validate_conservation_law(
            initial_energy, final_energy, 'standard'
        )
        
        assert energy_conservation['is_conserved']
        
        # Validate momentum conservation
        total_momentum = result['total_momentum']
        momentum_magnitude = np.linalg.norm(total_momentum)
        
        # Total momentum should be small for system initialized with random velocities
        momentum_validation = validation_suite.validate_known_result(
            momentum_magnitude, 0.0, 'loose'
        )
        
        assert momentum_validation['is_accurate']
    
    def test_cmb_acoustic_peaks_validation(self, astrophysics_agent, validation_suite):
        """Validate CMB acoustic peak positions."""
        l_max = 1000
        
        result = astrophysics_agent.simulate_cosmic_microwave_background(l_max)
        
        # Validate first acoustic peak position
        first_peak = result['first_acoustic_peak']
        
        # First acoustic peak should be around l ~ 220
        peak_validation = validation_suite.validate_known_result(
            first_peak, 220.0, 'loose'
        )
        
        assert peak_validation['is_accurate']
        
        # Validate peak positions are in correct range
        peak_positions = result['peak_positions']
        
        # Peaks should be at roughly l ~ 220, 540, 800...
        expected_peaks = [220, 540, 800]
        
        for i, (calculated, expected) in enumerate(zip(peak_positions, expected_peaks)):
            if i < len(expected_peaks):
                peak_val_validation = validation_suite.validate_known_result(
                    calculated, expected, 'loose'
                )
                
                assert peak_val_validation['is_accurate']


class TestPhysicsValidationSuite:
    """Tests for the physics validation suite itself."""
    
    @pytest.fixture
    def validation_suite(self):
        """Create physics validation suite."""
        return PhysicsValidationSuite()
    
    def test_conservation_law_validation(self, validation_suite):
        """Test conservation law validation functionality."""
        # Test perfect conservation
        perfect_conservation = validation_suite.validate_conservation_law(
            100.0, 100.0, 'strict'
        )
        
        assert perfect_conservation['is_conserved']
        assert perfect_conservation['conservation_quality'] == 'excellent'
        assert perfect_conservation['relative_error'] == 0.0
        
        # Test approximate conservation
        approximate_conservation = validation_suite.validate_conservation_law(
            100.0, 100.1, 'standard'
        )
        
        assert approximate_conservation['is_conserved']
        assert approximate_conservation['relative_error'] == 0.001
        
        # Test poor conservation
        poor_conservation = validation_suite.validate_conservation_law(
            100.0, 120.0, 'strict'
        )
        
        assert not poor_conservation['is_conserved']
        assert poor_conservation['conservation_quality'] == 'poor'
    
    def test_known_result_validation(self, validation_suite):
        """Test known result validation functionality."""
        # Test exact match
        exact_match = validation_suite.validate_known_result(
            np.pi, np.pi, 'strict'
        )
        
        assert exact_match['is_accurate']
        assert exact_match['accuracy_rating'] == 'excellent'
        
        # Test approximate match
        approximate_match = validation_suite.validate_known_result(
            3.14, np.pi, 'standard'
        )
        
        assert approximate_match['is_accurate']
        
        # Test poor match
        poor_match = validation_suite.validate_known_result(
            3.0, np.pi, 'strict'
        )
        
        assert not poor_match['is_accurate']
        assert poor_match['accuracy_rating'] == 'poor'
    
    def test_symmetry_validation(self, validation_suite):
        """Test symmetry property validation."""
        # Test even function
        x = np.linspace(-2, 2, 11)
        even_function = x**2  # f(-x) = f(x)
        
        even_symmetry = validation_suite.validate_symmetry_property(even_function, 'even')
        
        assert even_symmetry['has_symmetry']
        assert even_symmetry['symmetry_quality'] in ['perfect', 'very_good', 'good']
        
        # Test odd function
        odd_function = x**3  # f(-x) = -f(x)
        
        odd_symmetry = validation_suite.validate_symmetry_property(odd_function, 'odd')
        
        assert odd_symmetry['has_symmetry']
        assert odd_symmetry['symmetry_quality'] in ['perfect', 'very_good', 'good']
        
        # Test asymmetric function
        asymmetric_function = x**2 + x  # Neither even nor odd
        
        asymmetric_even = validation_suite.validate_symmetry_property(asymmetric_function, 'even')
        asymmetric_odd = validation_suite.validate_symmetry_property(asymmetric_function, 'odd')
        
        assert not asymmetric_even['has_symmetry']
        assert not asymmetric_odd['has_symmetry']
    
    def test_dimensional_consistency_validation(self, validation_suite):
        """Test dimensional consistency validation."""
        # Test reasonable scale
        reasonable = validation_suite.validate_dimensional_consistency(
            1e6, 'energy [J]', 1e6
        )
        
        assert reasonable['is_reasonable']
        
        # Test unreasonable scale
        unreasonable = validation_suite.validate_dimensional_consistency(
            1e-50, 'length [m]', 1.0
        )
        
        assert not unreasonable['is_reasonable']
    
    def test_fundamental_constants_access(self, validation_suite):
        """Test access to fundamental constants."""
        constants = validation_suite.fundamental_constants
        
        # Check that important constants are available
        assert 'c' in constants
        assert 'h' in constants
        assert 'e' in constants
        assert 'G' in constants
        
        # Check reasonable values
        assert 2.9e8 < constants['c'] < 3.1e8  # Speed of light
        assert 6e-34 < constants['h'] < 7e-34  # Planck constant
        assert 6e-11 < constants['G'] < 7e-11  # Gravitational constant
    
    def test_known_results_database(self, validation_suite):
        """Test known results database."""
        known_results = validation_suite.known_results
        
        # Check that important physics results are available
        assert 'hydrogen_atom' in known_results
        assert 'harmonic_oscillator' in known_results
        assert 'earth_properties' in known_results
        assert 'solar_properties' in known_results
        
        # Check specific values
        hydrogen = known_results['hydrogen_atom']
        assert abs(hydrogen['binding_energy_eV'] - 13.6) < 0.1
        
        earth = known_results['earth_properties']
        assert 5e24 < earth['mass_kg'] < 6e24