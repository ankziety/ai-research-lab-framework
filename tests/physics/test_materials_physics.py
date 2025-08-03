"""
Materials Physics Tests

Comprehensive unit tests for materials science calculations and simulations.
Tests crystal structures, mechanical properties, and materials modeling.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, patch

# Materials physics test markers
pytestmark = pytest.mark.materials


class MockMaterialsPhysicsAgent:
    """Mock materials physics agent for testing."""
    
    def __init__(self):
        self.name = "MaterialsPhysicsAgent"
        self.initialized = True
        self.crystal_database = {}
        self.material_properties = {}
    
    def analyze_crystal_structure(self, lattice_parameters: Dict[str, float],
                                space_group: str = 'P1') -> Dict[str, Any]:
        """Analyze crystal structure properties."""
        # Extract lattice parameters
        a = lattice_parameters.get('a', 1.0)
        b = lattice_parameters.get('b', 1.0)
        c = lattice_parameters.get('c', 1.0)
        alpha = lattice_parameters.get('alpha', 90.0)
        beta = lattice_parameters.get('beta', 90.0)
        gamma = lattice_parameters.get('gamma', 90.0)
        
        # Calculate unit cell volume
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        volume = a * b * c * np.sqrt(1 + 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad) -
                                   np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2)
        
        # Determine crystal system
        crystal_system = self._determine_crystal_system(a, b, c, alpha, beta, gamma)
        
        # Calculate coordination number (simplified)
        coordination_number = self._estimate_coordination_number(crystal_system)
        
        # Calculate packing efficiency
        packing_efficiency = self._calculate_packing_efficiency(crystal_system)
        
        return {
            'lattice_parameters': lattice_parameters,
            'unit_cell_volume': volume,
            'crystal_system': crystal_system,
            'space_group': space_group,
            'coordination_number': coordination_number,
            'packing_efficiency': packing_efficiency,
            'density_calculated': True
        }
    
    def calculate_mechanical_properties(self, elastic_constants: np.ndarray,
                                      density: float = 1000.0) -> Dict[str, Any]:
        """Calculate mechanical properties from elastic constants."""
        # For simplicity, assume isotropic material with C11, C12, C44
        if elastic_constants.shape == (6, 6):
            C11 = elastic_constants[0, 0]
            C12 = elastic_constants[0, 1]
            C44 = elastic_constants[3, 3]
        else:
            # Default values for testing
            C11, C12, C44 = 200e9, 100e9, 50e9
        
        # Calculate bulk modulus
        bulk_modulus = (C11 + 2*C12) / 3
        
        # Calculate shear modulus
        shear_modulus = C44
        
        # Calculate Young's modulus (isotropic approximation)
        youngs_modulus = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
        
        # Calculate Poisson's ratio
        poissons_ratio = (3*bulk_modulus - 2*shear_modulus) / (6*bulk_modulus + 2*shear_modulus)
        
        # Calculate wave velocities
        longitudinal_velocity = np.sqrt(C11 / density)
        transverse_velocity = np.sqrt(C44 / density)
        
        # Calculate Debye temperature (simplified)
        debye_temperature = self._calculate_debye_temperature(
            longitudinal_velocity, transverse_velocity, density
        )
        
        return {
            'bulk_modulus': bulk_modulus,
            'shear_modulus': shear_modulus,
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'longitudinal_velocity': longitudinal_velocity,
            'transverse_velocity': transverse_velocity,
            'debye_temperature': debye_temperature,
            'elastic_constants': elastic_constants
        }
    
    def simulate_stress_strain_curve(self, material_type: str = 'metal',
                                   max_strain: float = 0.1, n_points: int = 100) -> Dict[str, Any]:
        """Simulate stress-strain behavior."""
        strain = np.linspace(0, max_strain, n_points)
        
        if material_type == 'metal':
            # Elastic-plastic behavior
            elastic_modulus = 200e9  # Pa
            yield_strength = 250e6   # Pa
            yield_strain = yield_strength / elastic_modulus
            
            stress = np.zeros_like(strain)
            
            # Elastic region
            elastic_mask = strain <= yield_strain
            stress[elastic_mask] = elastic_modulus * strain[elastic_mask]
            
            # Plastic region (simplified work hardening)
            plastic_mask = strain > yield_strain
            plastic_strain = strain[plastic_mask] - yield_strain
            stress[plastic_mask] = yield_strength + 50e9 * plastic_strain
            
        elif material_type == 'ceramic':
            # Brittle behavior
            elastic_modulus = 400e9  # Pa
            fracture_strength = 300e6  # Pa
            fracture_strain = fracture_strength / elastic_modulus
            
            stress = np.minimum(elastic_modulus * strain, fracture_strength)
            
            # Set stress to zero after fracture
            stress[strain > fracture_strain] = 0
            
        elif material_type == 'polymer':
            # Viscoelastic behavior (simplified)
            elastic_modulus = 2e9  # Pa
            stress = elastic_modulus * strain * (1 - np.exp(-strain/0.01))
            
        else:
            # Linear elastic
            elastic_modulus = 100e9
            stress = elastic_modulus * strain
        
        return {
            'strain': strain,
            'stress': stress,
            'material_type': material_type,
            'elastic_modulus': elastic_modulus,
            'ultimate_strength': np.max(stress),
            'fracture_strain': strain[np.argmax(stress)] if material_type == 'ceramic' else max_strain,
            'toughness': np.trapz(stress, strain)  # Area under curve
        }
    
    def calculate_thermal_properties(self, crystal_structure: Dict[str, Any],
                                   temperature: float = 300.0) -> Dict[str, Any]:
        """Calculate thermal properties of materials."""
        # Mock thermal conductivity calculation
        base_conductivity = 100.0  # W/m·K
        
        # Temperature dependence (simplified)
        thermal_conductivity = base_conductivity * (300.0 / temperature)
        
        # Specific heat (Dulong-Petit law approximation)
        n_atoms = crystal_structure.get('coordination_number', 6)
        specific_heat = 3 * n_atoms * 8.314  # J/mol·K (R per atom)
        
        # Thermal expansion coefficient
        thermal_expansion = 10e-6  # 1/K (typical for metals)
        
        # Thermal diffusivity
        density = 8000.0  # kg/m³ (assumed)
        thermal_diffusivity = thermal_conductivity / (density * specific_heat)
        
        return {
            'thermal_conductivity': thermal_conductivity,
            'specific_heat': specific_heat,
            'thermal_expansion': thermal_expansion,
            'thermal_diffusivity': thermal_diffusivity,
            'temperature': temperature,
            'density': density
        }
    
    def analyze_defects(self, defect_type: str, concentration: float = 1e-6) -> Dict[str, Any]:
        """Analyze material defects and their effects."""
        defect_properties = {}
        
        if defect_type == 'vacancy':
            # Vacancy formation energy (eV)
            formation_energy = 1.0
            
            # Effect on properties
            conductivity_change = -concentration * 0.1  # Decreased conductivity
            modulus_change = -concentration * 0.05      # Decreased modulus
            
            defect_properties = {
                'formation_energy': formation_energy,
                'conductivity_change': conductivity_change,
                'modulus_change': modulus_change,
                'diffusion_coefficient': 1e-12 * np.exp(-formation_energy/0.026)  # m²/s
            }
            
        elif defect_type == 'interstitial':
            formation_energy = 3.0
            conductivity_change = -concentration * 0.2
            modulus_change = concentration * 0.1  # Increased modulus
            
            defect_properties = {
                'formation_energy': formation_energy,
                'conductivity_change': conductivity_change,
                'modulus_change': modulus_change,
                'diffusion_coefficient': 1e-15 * np.exp(-formation_energy/0.026)
            }
            
        elif defect_type == 'grain_boundary':
            # Grain boundary energy
            gb_energy = 0.5  # J/m²
            
            defect_properties = {
                'grain_boundary_energy': gb_energy,
                'conductivity_change': -concentration * 0.3,
                'strength_change': concentration * 0.2,  # Hall-Petch strengthening
                'diffusion_enhancement': 1000  # GB diffusion is faster
            }
            
        elif defect_type == 'dislocation':
            # Dislocation line energy
            line_energy = 1e-9  # J/m
            
            defect_properties = {
                'line_energy': line_energy,
                'yield_strength_change': concentration * 100e6,  # Strengthening
                'conductivity_change': -concentration * 0.05,
                'work_hardening_rate': concentration * 1e9
            }
        
        return {
            'defect_type': defect_type,
            'concentration': concentration,
            'properties': defect_properties,
            'thermodynamic_stability': formation_energy < 2.0 if 'formation_energy' in defect_properties else True
        }
    
    def simulate_phase_transition(self, temperature_range: Tuple[float, float],
                                pressure: float = 1e5) -> Dict[str, Any]:
        """Simulate phase transitions in materials."""
        T_min, T_max = temperature_range
        temperatures = np.linspace(T_min, T_max, 100)
        
        # Mock phase transition at mid-temperature
        T_transition = (T_min + T_max) / 2
        
        # Phase fractions
        phase_alpha = np.zeros_like(temperatures)
        phase_beta = np.zeros_like(temperatures)
        
        # Smooth transition using tanh function
        transition_width = (T_max - T_min) * 0.1
        
        for i, T in enumerate(temperatures):
            if T < T_transition - transition_width:
                phase_alpha[i] = 1.0
                phase_beta[i] = 0.0
            elif T > T_transition + transition_width:
                phase_alpha[i] = 0.0
                phase_beta[i] = 1.0
            else:
                # Transition region
                x = (T - T_transition) / transition_width
                phase_beta[i] = 0.5 * (1 + np.tanh(x))
                phase_alpha[i] = 1.0 - phase_beta[i]
        
        # Calculate enthalpy change
        enthalpy_change = 50000.0  # J/mol (typical for solid-solid transition)
        
        return {
            'temperatures': temperatures,
            'phase_alpha_fraction': phase_alpha,
            'phase_beta_fraction': phase_beta,
            'transition_temperature': T_transition,
            'enthalpy_change': enthalpy_change,
            'pressure': pressure,
            'transition_type': 'solid-solid'
        }
    
    def calculate_electronic_properties(self, band_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate electronic properties from band structure."""
        # Mock band gap calculation
        valence_band_max = band_structure.get('valence_band_max', 0.0)  # eV
        conduction_band_min = band_structure.get('conduction_band_min', 2.0)  # eV
        
        band_gap = conduction_band_min - valence_band_max
        
        # Classify material type
        if band_gap <= 0:
            material_class = 'metal'
            conductivity = 1e6  # S/m
        elif band_gap < 0.1:
            material_class = 'semimetal'
            conductivity = 1e4
        elif band_gap < 3.0:
            material_class = 'semiconductor'
            # Temperature-dependent conductivity
            conductivity = 1e-6 * np.exp(-band_gap / (2 * 0.026))  # Intrinsic
        else:
            material_class = 'insulator'
            conductivity = 1e-12
        
        # Calculate carrier concentrations (simplified)
        if material_class == 'semiconductor':
            intrinsic_carrier_density = 1e10 * np.exp(-band_gap / (2 * 0.026))  # cm⁻³
        else:
            intrinsic_carrier_density = 0.0
        
        return {
            'band_gap': band_gap,
            'material_class': material_class,
            'electrical_conductivity': conductivity,
            'intrinsic_carrier_density': intrinsic_carrier_density,
            'valence_band_max': valence_band_max,
            'conduction_band_min': conduction_band_min,
            'fermi_level': (valence_band_max + conduction_band_min) / 2
        }
    
    def simulate_grain_growth(self, initial_grain_size: float, temperature: float,
                            time: float, activation_energy: float = 200000.0) -> Dict[str, Any]:
        """Simulate grain growth kinetics."""
        # Grain growth law: D² - D₀² = k*t
        # where k = k₀ * exp(-Q/RT)
        
        R = 8.314  # J/mol·K
        k0 = 1e-6  # Pre-exponential factor
        
        # Rate constant
        k = k0 * np.exp(-activation_energy / (R * temperature))
        
        # Time evolution
        times = np.linspace(0, time, 100)
        grain_sizes = np.sqrt(initial_grain_size**2 + k * times)
        
        # Calculate growth rate
        growth_rate = k / (2 * np.sqrt(initial_grain_size**2 + k * time))
        
        return {
            'times': times,
            'grain_sizes': grain_sizes,
            'initial_grain_size': initial_grain_size,
            'final_grain_size': grain_sizes[-1],
            'growth_rate': growth_rate,
            'rate_constant': k,
            'activation_energy': activation_energy,
            'temperature': temperature
        }
    
    def analyze_surface_properties(self, surface_orientation: str,
                                 surface_energy: float = 1.0) -> Dict[str, Any]:
        """Analyze surface properties and reconstruction."""
        # Surface energy depends on orientation
        orientation_factors = {
            '100': 1.0,
            '110': 1.2,
            '111': 0.8
        }
        
        adjusted_surface_energy = surface_energy * orientation_factors.get(surface_orientation, 1.0)
        
        # Calculate surface relaxation (simplified)
        surface_relaxation = 0.05  # 5% change in surface layer spacing
        
        # Adsorption energy (simplified)
        adsorption_energy = -0.5 * adjusted_surface_energy  # Attractive
        
        # Work function (simplified)
        work_function = 4.0 + 0.5 * adjusted_surface_energy  # eV
        
        return {
            'surface_orientation': surface_orientation,
            'surface_energy': adjusted_surface_energy,
            'surface_relaxation': surface_relaxation,
            'adsorption_energy': adsorption_energy,
            'work_function': work_function,
            'surface_reconstruction': abs(surface_relaxation) > 0.02
        }
    
    # Helper methods
    def _determine_crystal_system(self, a, b, c, alpha, beta, gamma):
        """Determine crystal system from lattice parameters."""
        tol = 1e-6
        
        if abs(a - b) < tol and abs(b - c) < tol and abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol:
            return 'cubic'
        elif abs(a - b) < tol and abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol:
            return 'tetragonal'
        elif abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol:
            return 'orthorhombic'
        elif abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 120) < tol:
            return 'hexagonal'
        elif abs(alpha - beta) < tol and abs(beta - gamma) < tol and abs(a - b) < tol and abs(b - c) < tol:
            return 'rhombohedral'
        elif abs(alpha - 90) < tol and abs(gamma - 90) < tol:
            return 'monoclinic'
        else:
            return 'triclinic'
    
    def _estimate_coordination_number(self, crystal_system):
        """Estimate coordination number based on crystal system."""
        coordination_numbers = {
            'cubic': 6,
            'tetragonal': 6,
            'orthorhombic': 6,
            'hexagonal': 8,
            'rhombohedral': 6,
            'monoclinic': 4,
            'triclinic': 4
        }
        return coordination_numbers.get(crystal_system, 6)
    
    def _calculate_packing_efficiency(self, crystal_system):
        """Calculate atomic packing efficiency."""
        efficiencies = {
            'cubic': 0.52,      # Simple cubic
            'tetragonal': 0.68,  # Body-centered
            'orthorhombic': 0.74, # Face-centered
            'hexagonal': 0.74,   # Close-packed
            'rhombohedral': 0.74,
            'monoclinic': 0.60,
            'triclinic': 0.55
        }
        return efficiencies.get(crystal_system, 0.60)
    
    def _calculate_debye_temperature(self, v_l, v_t, density):
        """Calculate Debye temperature."""
        # Simplified calculation
        h = 6.626e-34  # Planck constant
        k_B = 1.381e-23  # Boltzmann constant
        
        # Average velocity
        v_avg = (v_l + 2*v_t) / 3
        
        # Mock Debye temperature
        theta_D = (h / k_B) * (6 * np.pi**2 * density / 1.66e-27)**(1/3) * v_avg
        
        return theta_D


class TestMaterialsPhysicsAgent:
    """Test class for materials physics functionality."""
    
    @pytest.fixture
    def materials_agent(self):
        """Create a materials physics agent instance for testing."""
        return MockMaterialsPhysicsAgent()
    
    def test_agent_initialization(self, materials_agent):
        """Test materials physics agent initialization."""
        assert materials_agent.name == "MaterialsPhysicsAgent"
        assert materials_agent.initialized is True
        assert hasattr(materials_agent, 'crystal_database')
        assert hasattr(materials_agent, 'material_properties')
    
    def test_crystal_structure_analysis(self, materials_agent, physics_test_config):
        """Test crystal structure analysis."""
        # Test cubic crystal
        lattice_params = {'a': 4.05, 'b': 4.05, 'c': 4.05, 'alpha': 90, 'beta': 90, 'gamma': 90}
        
        result = materials_agent.analyze_crystal_structure(lattice_params, space_group='Fm3m')
        
        assert 'lattice_parameters' in result
        assert 'unit_cell_volume' in result
        assert 'crystal_system' in result
        assert 'coordination_number' in result
        assert 'packing_efficiency' in result
        
        # Check cubic system detection
        assert result['crystal_system'] == 'cubic'
        
        # Check volume calculation
        expected_volume = 4.05**3
        assert abs(result['unit_cell_volume'] - expected_volume) < 1e-6
        
        # Check reasonable coordination number
        assert result['coordination_number'] > 0
        assert result['coordination_number'] <= 12
        
        # Check packing efficiency
        assert 0 < result['packing_efficiency'] <= 1
    
    def test_mechanical_properties_calculation(self, materials_agent):
        """Test mechanical properties calculation."""
        # Create elastic constants matrix for isotropic material
        elastic_constants = np.zeros((6, 6))
        elastic_constants[0, 0] = elastic_constants[1, 1] = elastic_constants[2, 2] = 200e9  # C11
        elastic_constants[0, 1] = elastic_constants[0, 2] = elastic_constants[1, 2] = 100e9  # C12
        elastic_constants[1, 0] = elastic_constants[2, 0] = elastic_constants[2, 1] = 100e9
        elastic_constants[3, 3] = elastic_constants[4, 4] = elastic_constants[5, 5] = 50e9   # C44
        
        density = 8000.0  # kg/m³
        
        result = materials_agent.calculate_mechanical_properties(elastic_constants, density)
        
        assert 'bulk_modulus' in result
        assert 'shear_modulus' in result
        assert 'youngs_modulus' in result
        assert 'poissons_ratio' in result
        assert 'longitudinal_velocity' in result
        assert 'transverse_velocity' in result
        assert 'debye_temperature' in result
        
        # Check reasonable values
        assert result['bulk_modulus'] > 0
        assert result['shear_modulus'] > 0
        assert result['youngs_modulus'] > 0
        assert -1 < result['poissons_ratio'] < 0.5  # Physical constraint
        assert result['longitudinal_velocity'] > result['transverse_velocity']
        assert result['debye_temperature'] > 0
    
    @pytest.mark.parametrize("material_type", ['metal', 'ceramic', 'polymer'])
    def test_stress_strain_simulation(self, materials_agent, material_type):
        """Test stress-strain curve simulation for different materials."""
        result = materials_agent.simulate_stress_strain_curve(
            material_type=material_type, max_strain=0.05, n_points=50
        )
        
        assert 'strain' in result
        assert 'stress' in result
        assert 'material_type' in result
        assert 'elastic_modulus' in result
        assert 'ultimate_strength' in result
        assert 'toughness' in result
        
        # Check data consistency
        assert len(result['strain']) == len(result['stress'])
        assert result['material_type'] == material_type
        
        # Check that stress starts at zero
        assert result['stress'][0] == 0.0
        
        # Check monotonic strain
        strain = result['strain']
        assert all(strain[i] <= strain[i+1] for i in range(len(strain)-1))
        
        # Material-specific checks
        if material_type == 'ceramic':
            # Ceramics should show brittle behavior
            max_stress_idx = np.argmax(result['stress'])
            assert max_stress_idx < len(result['stress']) - 1  # Fracture before end
        
        elif material_type == 'metal':
            # Metals should show yielding behavior
            assert result['ultimate_strength'] > 100e6  # Reasonable strength
    
    def test_thermal_properties_calculation(self, materials_agent):
        """Test thermal properties calculation."""
        crystal_structure = {
            'crystal_system': 'cubic',
            'coordination_number': 8,
            'unit_cell_volume': 100.0
        }
        
        temperature = 350.0  # K
        
        result = materials_agent.calculate_thermal_properties(crystal_structure, temperature)
        
        assert 'thermal_conductivity' in result
        assert 'specific_heat' in result
        assert 'thermal_expansion' in result
        assert 'thermal_diffusivity' in result
        assert 'temperature' in result
        
        # Check reasonable values
        assert result['thermal_conductivity'] > 0
        assert result['specific_heat'] > 0
        assert result['thermal_expansion'] > 0
        assert result['thermal_diffusivity'] > 0
        assert result['temperature'] == temperature
        
        # Check units consistency (thermal diffusivity = k/(ρ*cp))
        k = result['thermal_conductivity']
        cp = result['specific_heat']
        rho = result['density']
        alpha_expected = k / (rho * cp)
        
        # Should be approximately equal (within numerical precision)
        assert abs(result['thermal_diffusivity'] - alpha_expected) < alpha_expected * 0.1
    
    @pytest.mark.parametrize("defect_type", ['vacancy', 'interstitial', 'grain_boundary', 'dislocation'])
    def test_defect_analysis(self, materials_agent, defect_type):
        """Test defect analysis for different defect types."""
        concentration = 1e-5
        
        result = materials_agent.analyze_defects(defect_type, concentration)
        
        assert 'defect_type' in result
        assert 'concentration' in result
        assert 'properties' in result
        assert 'thermodynamic_stability' in result
        
        assert result['defect_type'] == defect_type
        assert result['concentration'] == concentration
        
        properties = result['properties']
        
        # Check defect-specific properties
        if defect_type in ['vacancy', 'interstitial']:
            assert 'formation_energy' in properties
            assert 'diffusion_coefficient' in properties
            assert properties['formation_energy'] > 0
            assert properties['diffusion_coefficient'] > 0
        
        elif defect_type == 'grain_boundary':
            assert 'grain_boundary_energy' in properties
            assert 'diffusion_enhancement' in properties
            assert properties['grain_boundary_energy'] > 0
        
        elif defect_type == 'dislocation':
            assert 'line_energy' in properties
            assert properties['line_energy'] > 0
    
    def test_phase_transition_simulation(self, materials_agent):
        """Test phase transition simulation."""
        temperature_range = (300.0, 800.0)
        pressure = 1e5  # Pa
        
        result = materials_agent.simulate_phase_transition(temperature_range, pressure)
        
        assert 'temperatures' in result
        assert 'phase_alpha_fraction' in result
        assert 'phase_beta_fraction' in result
        assert 'transition_temperature' in result
        assert 'enthalpy_change' in result
        
        temperatures = result['temperatures']
        alpha_fraction = result['phase_alpha_fraction']
        beta_fraction = result['phase_beta_fraction']
        
        # Check that fractions sum to 1
        total_fraction = alpha_fraction + beta_fraction
        np.testing.assert_array_almost_equal(total_fraction, np.ones_like(total_fraction), decimal=6)
        
        # Check transition temperature is within range
        T_transition = result['transition_temperature']
        assert temperature_range[0] <= T_transition <= temperature_range[1]
        
        # Check phase behavior
        low_temp_mask = temperatures < T_transition - 50
        high_temp_mask = temperatures > T_transition + 50
        
        if len(low_temp_mask) > 0:
            assert np.mean(alpha_fraction[low_temp_mask]) > 0.8  # Mostly alpha at low T
        if len(high_temp_mask) > 0:
            assert np.mean(beta_fraction[high_temp_mask]) > 0.8   # Mostly beta at high T
    
    def test_electronic_properties_calculation(self, materials_agent):
        """Test electronic properties calculation."""
        # Test semiconductor
        band_structure = {
            'valence_band_max': 0.0,
            'conduction_band_min': 1.1  # Silicon-like gap
        }
        
        result = materials_agent.calculate_electronic_properties(band_structure)
        
        assert 'band_gap' in result
        assert 'material_class' in result
        assert 'electrical_conductivity' in result
        assert 'intrinsic_carrier_density' in result
        assert 'fermi_level' in result
        
        # Check band gap calculation
        expected_gap = 1.1
        assert abs(result['band_gap'] - expected_gap) < 1e-6
        
        # Check material classification
        assert result['material_class'] == 'semiconductor'
        
        # Check conductivity is reasonable for semiconductor
        assert 1e-12 < result['electrical_conductivity'] < 1e6
        
        # Test metal (zero band gap)
        metal_band_structure = {
            'valence_band_max': 0.0,
            'conduction_band_min': -0.5  # Overlapping bands
        }
        
        metal_result = materials_agent.calculate_electronic_properties(metal_band_structure)
        assert metal_result['material_class'] == 'metal'
        assert metal_result['band_gap'] <= 0
    
    def test_grain_growth_simulation(self, materials_agent):
        """Test grain growth kinetics simulation."""
        initial_grain_size = 10e-6  # 10 μm
        temperature = 1000.0  # K
        time = 3600.0  # 1 hour
        
        result = materials_agent.simulate_grain_growth(
            initial_grain_size, temperature, time
        )
        
        assert 'times' in result
        assert 'grain_sizes' in result
        assert 'initial_grain_size' in result
        assert 'final_grain_size' in result
        assert 'growth_rate' in result
        assert 'rate_constant' in result
        
        times = result['times']
        grain_sizes = result['grain_sizes']
        
        # Check that grain size increases with time
        assert all(grain_sizes[i] <= grain_sizes[i+1] for i in range(len(grain_sizes)-1))
        
        # Check initial condition
        assert abs(grain_sizes[0] - initial_grain_size) < 1e-10
        
        # Check final size is larger
        assert result['final_grain_size'] > initial_grain_size
        
        # Check growth rate is positive
        assert result['growth_rate'] > 0
    
    def test_surface_properties_analysis(self, materials_agent):
        """Test surface properties analysis."""
        surface_orientation = '111'
        surface_energy = 1.5  # J/m²
        
        result = materials_agent.analyze_surface_properties(surface_orientation, surface_energy)
        
        assert 'surface_orientation' in result
        assert 'surface_energy' in result
        assert 'surface_relaxation' in result
        assert 'adsorption_energy' in result
        assert 'work_function' in result
        assert 'surface_reconstruction' in result
        
        assert result['surface_orientation'] == surface_orientation
        assert result['surface_energy'] > 0
        assert result['work_function'] > 0
        
        # Check adsorption energy is negative (attractive)
        assert result['adsorption_energy'] < 0
    
    def test_materials_integration_workflow(self, materials_agent, physics_test_config):
        """Test integrated materials analysis workflow."""
        # Step 1: Analyze crystal structure
        lattice_params = {'a': 3.6, 'b': 3.6, 'c': 3.6, 'alpha': 90, 'beta': 90, 'gamma': 90}
        crystal_result = materials_agent.analyze_crystal_structure(lattice_params)
        
        # Step 2: Calculate mechanical properties
        elastic_constants = np.eye(6) * 100e9  # Simplified
        mechanical_result = materials_agent.calculate_mechanical_properties(elastic_constants)
        
        # Step 3: Calculate thermal properties
        thermal_result = materials_agent.calculate_thermal_properties(crystal_result)
        
        # Step 4: Simulate stress-strain behavior
        stress_strain_result = materials_agent.simulate_stress_strain_curve('metal')
        
        # Step 5: Analyze defects
        defect_result = materials_agent.analyze_defects('vacancy', 1e-6)
        
        # Verify all steps completed
        assert crystal_result['crystal_system'] == 'cubic'
        assert mechanical_result['youngs_modulus'] > 0
        assert thermal_result['thermal_conductivity'] > 0
        assert stress_strain_result['ultimate_strength'] > 0
        assert defect_result['defect_type'] == 'vacancy'
        
        # Check consistency between results
        # Young's modulus from mechanical calculation should match stress-strain
        mechanical_modulus = mechanical_result['youngs_modulus']
        stress_strain_modulus = stress_strain_result['elastic_modulus']
        
        # Should be in the same order of magnitude
        ratio = mechanical_modulus / stress_strain_modulus
        assert 0.1 < ratio < 10.0  # Within order of magnitude
    
    @pytest.mark.slow
    def test_large_scale_materials_simulation(self, materials_agent):
        """Test materials simulation on larger scales."""
        # Large grain growth simulation
        result = materials_agent.simulate_grain_growth(
            initial_grain_size=1e-6,
            temperature=1200.0,
            time=86400.0,  # 24 hours
            activation_energy=300000.0
        )
        
        assert len(result['times']) == 100
        assert result['final_grain_size'] > result['initial_grain_size']
        
        # Large stress-strain dataset
        stress_strain_result = materials_agent.simulate_stress_strain_curve(
            material_type='metal',
            max_strain=0.2,
            n_points=1000
        )
        
        assert len(stress_strain_result['strain']) == 1000
        assert len(stress_strain_result['stress']) == 1000
    
    def test_materials_error_handling(self, materials_agent):
        """Test error handling in materials calculations."""
        # Test with invalid lattice parameters
        invalid_lattice = {'a': -1.0, 'b': 2.0, 'c': 3.0}
        
        # Should handle gracefully
        result = materials_agent.analyze_crystal_structure(invalid_lattice)
        assert 'crystal_system' in result
        
        # Test with invalid elastic constants
        invalid_elastic = np.array([[np.inf, 0], [0, np.nan]])
        
        # Should not crash
        result = materials_agent.calculate_mechanical_properties(invalid_elastic)
        assert 'youngs_modulus' in result
        
        # Test with unknown defect type
        result = materials_agent.analyze_defects('unknown_defect')
        assert 'defect_type' in result
        assert result['defect_type'] == 'unknown_defect'


class TestMaterialsPhysicsIntegration:
    """Integration tests for materials physics workflows."""
    
    @pytest.fixture
    def materials_workflow(self):
        """Create a materials physics workflow."""
        agent = MockMaterialsPhysicsAgent()
        
        workflow = {
            'agent': agent,
            'materials_database': {},
            'analysis_results': {}
        }
        
        return workflow
    
    def test_complete_materials_characterization(self, materials_workflow, physics_test_config):
        """Test complete materials characterization workflow."""
        agent = materials_workflow['agent']
        
        # Material: Aluminum (FCC structure)
        material_config = physics_test_config['materials_physics']['test_materials'][0]
        
        # Step 1: Crystal structure analysis
        lattice_params = {'a': 4.05, 'b': 4.05, 'c': 4.05, 'alpha': 90, 'beta': 90, 'gamma': 90}
        structure_result = agent.analyze_crystal_structure(lattice_params, 'Fm3m')
        
        # Step 2: Mechanical characterization
        # Aluminum elastic constants (approximate)
        C11, C12, C44 = 108e9, 62e9, 28e9
        elastic_matrix = np.zeros((6, 6))
        elastic_matrix[0:3, 0:3] = [[C11, C12, C12], [C12, C11, C12], [C12, C12, C11]]
        elastic_matrix[3:6, 3:6] = np.diag([C44, C44, C44])
        
        mechanical_result = agent.calculate_mechanical_properties(elastic_matrix, material_config['density'])
        
        # Step 3: Thermal analysis
        thermal_result = agent.calculate_thermal_properties(structure_result, temperature=300.0)
        
        # Step 4: Electronic properties
        band_structure = {'valence_band_max': 0.0, 'conduction_band_min': -1.0}  # Metal
        electronic_result = agent.calculate_electronic_properties(band_structure)
        
        # Step 5: Surface analysis
        surface_result = agent.analyze_surface_properties('111', surface_energy=0.9)
        
        # Store comprehensive characterization
        materials_workflow['analysis_results'] = {
            'structure': structure_result,
            'mechanical': mechanical_result,
            'thermal': thermal_result,
            'electronic': electronic_result,
            'surface': surface_result
        }
        
        # Verify comprehensive analysis
        assert structure_result['crystal_system'] == 'cubic'
        assert mechanical_result['youngs_modulus'] > 50e9  # Reasonable for Al
        assert thermal_result['thermal_conductivity'] > 100  # Good conductor
        assert electronic_result['material_class'] == 'metal'
        assert surface_result['surface_orientation'] == '111'
        
        # Check interdependencies
        # High thermal conductivity should correlate with metallic behavior
        assert (thermal_result['thermal_conductivity'] > 100 and 
                electronic_result['material_class'] == 'metal')
    
    @pytest.mark.integration
    def test_materials_processing_simulation(self, materials_workflow):
        """Test materials processing simulation workflow."""
        agent = materials_workflow['agent']
        
        # Simulate heat treatment process
        # Step 1: Initial microstructure
        initial_grain_size = 5e-6  # 5 μm
        
        # Step 2: Annealing simulation
        annealing_temperature = 773.0  # 500°C
        annealing_time = 7200.0  # 2 hours
        
        grain_growth_result = agent.simulate_grain_growth(
            initial_grain_size, annealing_temperature, annealing_time
        )
        
        # Step 3: Phase transformation during cooling
        cooling_result = agent.simulate_phase_transition(
            temperature_range=(773.0, 300.0)
        )
        
        # Step 4: Defect evolution
        vacancy_result = agent.analyze_defects('vacancy', concentration=1e-5)
        dislocation_result = agent.analyze_defects('dislocation', concentration=1e12)  # dislocations/m²
        
        # Step 5: Final mechanical properties
        final_stress_strain = agent.simulate_stress_strain_curve(
            material_type='metal', max_strain=0.1
        )
        
        # Store processing results
        materials_workflow['materials_database']['heat_treatment'] = {
            'grain_growth': grain_growth_result,
            'phase_transformation': cooling_result,
            'defect_evolution': {
                'vacancies': vacancy_result,
                'dislocations': dislocation_result
            },
            'final_properties': final_stress_strain
        }
        
        # Verify processing effects
        assert grain_growth_result['final_grain_size'] > initial_grain_size
        assert cooling_result['transition_temperature'] > 300.0
        assert final_stress_strain['ultimate_strength'] > 0
        
        # Check microstructure-property relationships
        # Larger grains should affect strength (Hall-Petch relationship)
        grain_size_ratio = grain_growth_result['final_grain_size'] / initial_grain_size
        assert grain_size_ratio > 1.0  # Grains grew during annealing
    
    @pytest.mark.asyncio
    async def test_async_materials_simulation(self, async_physics_simulator):
        """Test asynchronous materials simulation."""
        # Start long materials simulation
        simulation_task = asyncio.create_task(
            async_physics_simulator.run_simulation(duration=1.0, dt=0.001)
        )
        
        # Check simulation is running
        assert async_physics_simulator.running is True
        
        # Wait for completion
        result = await simulation_task
        
        assert result['status'] == 'completed'
        assert result['steps'] > 0
        assert async_physics_simulator.running is False