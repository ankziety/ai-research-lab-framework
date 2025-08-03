"""
Astrophysics Agent - Specialized agent for astrophysics and cosmology.

This agent provides expertise in astrophysics, cosmology, stellar physics,
and gravitational phenomena across cosmic scales.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import integrate, constants
import matplotlib.pyplot as plt

from .base_physics_agent import BasePhysicsAgent, PhysicsScale, PhysicsMethodology

logger = logging.getLogger(__name__)


class AstrophysicsAgent(BasePhysicsAgent):
    """
    Specialized agent for astrophysics and cosmological research.
    
    Expertise includes:
    - Stellar physics and evolution
    - Galactic dynamics and structure
    - Cosmology and dark matter/energy
    - Gravitational physics and relativity
    - High-energy astrophysics
    - Planetary systems and exoplanets
    - Observational astronomy techniques
    """
    
    def __init__(self, agent_id: str, role: str = None, expertise: List[str] = None,
                 model_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        """
        Initialize astrophysics agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent role (defaults to "Astrophysics Expert")
            expertise: List of expertise areas (uses defaults if None)
            model_config: Configuration for the underlying LLM
            cost_manager: Optional cost manager for tracking API usage
        """
        if role is None:
            role = "Astrophysics Expert"
        
        if expertise is None:
            expertise = [
                "Stellar Physics",
                "Galactic Dynamics",
                "Cosmology",
                "General Relativity",
                "High-Energy Astrophysics",
                "Planetary Science",
                "Dark Matter and Dark Energy",
                "Gravitational Waves",
                "Observational Astronomy",
                "Computational Astrophysics"
            ]
        
        super().__init__(agent_id, role, expertise, model_config, cost_manager)
        
        # Fundamental astrophysical constants
        self.astro_constants = {
            'G': 6.67430e-11,           # Gravitational constant (m³/kg·s²)
            'c': 299792458,             # Speed of light (m/s)
            'h': 6.62607015e-34,        # Planck constant (J·s)
            'k_B': 1.380649e-23,        # Boltzmann constant (J/K)
            'sigma_SB': 5.670374419e-8, # Stefan-Boltzmann constant (W/m²·K⁴)
            'M_sun': 1.98847e30,        # Solar mass (kg)
            'R_sun': 6.96e8,            # Solar radius (m)
            'L_sun': 3.828e26,          # Solar luminosity (W)
            'au': 1.49597871e11,        # Astronomical unit (m)
            'pc': 3.0857e16,            # Parsec (m)
            'H0': 70.0,                 # Hubble constant (km/s/Mpc)
            'Omega_m': 0.31,            # Matter density parameter
            'Omega_lambda': 0.69        # Dark energy density parameter
        }
        
        # Stellar classification and properties
        self.stellar_types = {
            'O': {'temp': 30000, 'mass': 15, 'color': 'blue', 'lifetime': 1e6},
            'B': {'temp': 10000, 'mass': 3, 'color': 'blue-white', 'lifetime': 1e8},
            'A': {'temp': 7500, 'mass': 1.4, 'color': 'white', 'lifetime': 1e9},
            'F': {'temp': 6000, 'mass': 1.04, 'color': 'yellow-white', 'lifetime': 3e9},
            'G': {'temp': 5200, 'mass': 0.8, 'color': 'yellow', 'lifetime': 1e10},
            'K': {'temp': 3700, 'mass': 0.45, 'color': 'orange', 'lifetime': 4e10},
            'M': {'temp': 2400, 'mass': 0.08, 'color': 'red', 'lifetime': 1e12}
        }
        
        # Cosmological models and parameters
        self.cosmological_models = {
            'lambda_cdm': {
                'description': 'Standard cosmological model',
                'parameters': ['H0', 'Omega_m', 'Omega_lambda', 'Omega_b'],
                'dark_matter': True,
                'dark_energy': True
            },
            'einstein_de_sitter': {
                'description': 'Critical density universe',
                'parameters': ['H0'],
                'dark_matter': False,
                'dark_energy': False
            },
            'steady_state': {
                'description': 'Steady state cosmology',
                'parameters': ['H0', 'creation_rate'],
                'dark_matter': False,
                'dark_energy': False
            }
        }
        
        # Observational techniques
        self.observational_techniques = {
            'photometry': {
                'description': 'Measurement of brightness',
                'wavelengths': ['optical', 'infrared', 'ultraviolet'],
                'applications': ['stellar_classification', 'distance_measurement', 'variability_studies']
            },
            'spectroscopy': {
                'description': 'Analysis of electromagnetic spectra',
                'types': ['emission', 'absorption', 'continuous'],
                'applications': ['composition_analysis', 'velocity_measurement', 'temperature_determination']
            },
            'astrometry': {
                'description': 'Precise position measurements',
                'techniques': ['parallax', 'proper_motion', 'radial_velocity'],
                'applications': ['distance_measurement', 'stellar_kinematics', 'exoplanet_detection']
            },
            'interferometry': {
                'description': 'High-resolution imaging technique',
                'types': ['radio', 'optical', 'x-ray'],
                'applications': ['high_resolution_imaging', 'gravitational_wave_detection']
            }
        }
        
        logger.info(f"Astrophysics Agent {self.agent_id} initialized")
    
    def _get_physics_domain(self) -> str:
        """Get the physics domain for astrophysics."""
        return "astrophysics"
    
    def _get_relevant_scales(self) -> List[PhysicsScale]:
        """Get physical scales relevant to astrophysics."""
        return [
            PhysicsScale.PLANETARY,
            PhysicsScale.STELLAR,
            PhysicsScale.GALACTIC,
            PhysicsScale.COSMIC
        ]
    
    def _get_preferred_methodologies(self) -> List[PhysicsMethodology]:
        """Get preferred methodologies for astrophysics."""
        return [
            PhysicsMethodology.THEORETICAL,
            PhysicsMethodology.OBSERVATIONAL,
            PhysicsMethodology.COMPUTATIONAL
        ]
    
    def analyze_stellar_evolution(self, stellar_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stellar evolution and lifecycle.
        
        Args:
            stellar_config: Stellar configuration including mass, composition, and age
            
        Returns:
            Comprehensive stellar evolution analysis
        """
        evolution_result = {
            'success': False,
            'initial_parameters': {},
            'current_state': {},
            'evolution_phases': [],
            'future_evolution': [],
            'lifetime_estimates': {},
            'nucleosynthesis': {},
            'final_fate': 'unknown'
        }
        
        try:
            # Extract stellar parameters
            mass = stellar_config.get('mass', 1.0)  # Solar masses
            metallicity = stellar_config.get('metallicity', 0.02)  # Z/Z_sun
            age = stellar_config.get('age', 0.0)  # Years
            
            # Determine current evolutionary phase
            current_state = self._determine_stellar_phase(mass, age)
            
            # Calculate lifetime estimates
            lifetime_estimates = self._calculate_stellar_lifetimes(mass, metallicity)
            
            # Predict evolution phases
            evolution_phases = self._predict_evolution_phases(mass, metallicity)
            
            # Analyze nucleosynthesis
            nucleosynthesis = self._analyze_stellar_nucleosynthesis(mass, current_state)
            
            # Determine final fate
            final_fate = self._determine_stellar_fate(mass)
            
            # Calculate stellar structure
            stellar_structure = self._calculate_stellar_structure(mass, age)
            
            evolution_result.update({
                'success': True,
                'initial_parameters': {
                    'mass_solar': mass,
                    'metallicity': metallicity,
                    'age_years': age
                },
                'current_state': current_state,
                'evolution_phases': evolution_phases,
                'lifetime_estimates': lifetime_estimates,
                'nucleosynthesis': nucleosynthesis,
                'final_fate': final_fate,
                'stellar_structure': stellar_structure
            })
            
        except Exception as e:
            evolution_result['error'] = str(e)
            logger.error(f"Stellar evolution analysis failed: {e}")
        
        return evolution_result
    
    def model_galactic_dynamics(self, galaxy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model galactic dynamics and structure.
        
        Args:
            galaxy_config: Galaxy configuration including mass, size, and rotation
            
        Returns:
            Galactic dynamics analysis including rotation curves and dark matter
        """
        dynamics_result = {
            'success': False,
            'galaxy_type': 'unknown',
            'mass_distribution': {},
            'rotation_curve': {},
            'dark_matter_profile': {},
            'stability_analysis': {},
            'spiral_structure': {},
            'central_black_hole': {}
        }
        
        try:
            # Determine galaxy type
            galaxy_type = galaxy_config.get('type', 'spiral')
            total_mass = galaxy_config.get('total_mass', 1e12)  # Solar masses
            scale_radius = galaxy_config.get('scale_radius', 3.0)  # kpc
            
            # Calculate mass distribution
            mass_distribution = self._calculate_galactic_mass_distribution(
                galaxy_type, total_mass, scale_radius
            )
            
            # Model rotation curve
            rotation_curve = self._model_rotation_curve(mass_distribution, galaxy_config)
            
            # Analyze dark matter profile
            dark_matter_profile = self._analyze_dark_matter_profile(rotation_curve, mass_distribution)
            
            # Assess stability
            stability_analysis = self._analyze_galactic_stability(mass_distribution, rotation_curve)
            
            # Model spiral structure (if applicable)
            spiral_structure = self._model_spiral_structure(galaxy_type, galaxy_config)
            
            # Analyze central black hole
            central_bh = self._analyze_central_black_hole(total_mass, galaxy_type)
            
            dynamics_result.update({
                'success': True,
                'galaxy_type': galaxy_type,
                'mass_distribution': mass_distribution,
                'rotation_curve': rotation_curve,
                'dark_matter_profile': dark_matter_profile,
                'stability_analysis': stability_analysis,
                'spiral_structure': spiral_structure,
                'central_black_hole': central_bh
            })
            
        except Exception as e:
            dynamics_result['error'] = str(e)
            logger.error(f"Galactic dynamics modeling failed: {e}")
        
        return dynamics_result
    
    def analyze_cosmological_model(self, cosmology_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cosmological models and evolution.
        
        Args:
            cosmology_config: Cosmological parameters and model configuration
            
        Returns:
            Cosmological analysis including universe evolution and predictions
        """
        cosmology_result = {
            'success': False,
            'model_type': 'lambda_cdm',
            'parameters': {},
            'age_of_universe': 0.0,
            'critical_density': 0.0,
            'expansion_history': {},
            'distance_measures': {},
            'structure_formation': {},
            'future_evolution': {}
        }
        
        try:
            # Extract cosmological parameters
            H0 = cosmology_config.get('hubble_constant', self.astro_constants['H0'])
            Omega_m = cosmology_config.get('omega_matter', self.astro_constants['Omega_m'])
            Omega_lambda = cosmology_config.get('omega_lambda', self.astro_constants['Omega_lambda'])
            
            # Calculate derived parameters
            critical_density = self._calculate_critical_density(H0)
            age_of_universe = self._calculate_universe_age(H0, Omega_m, Omega_lambda)
            
            # Model expansion history
            expansion_history = self._model_expansion_history(H0, Omega_m, Omega_lambda)
            
            # Calculate distance measures
            distance_measures = self._calculate_cosmological_distances(H0, Omega_m, Omega_lambda)
            
            # Analyze structure formation
            structure_formation = self._analyze_structure_formation(Omega_m, Omega_lambda)
            
            # Predict future evolution
            future_evolution = self._predict_universe_future(Omega_m, Omega_lambda)
            
            cosmology_result.update({
                'success': True,
                'parameters': {
                    'H0_km_s_Mpc': H0,
                    'Omega_matter': Omega_m,
                    'Omega_lambda': Omega_lambda,
                    'Omega_total': Omega_m + Omega_lambda
                },
                'age_of_universe': age_of_universe,
                'critical_density': critical_density,
                'expansion_history': expansion_history,
                'distance_measures': distance_measures,
                'structure_formation': structure_formation,
                'future_evolution': future_evolution
            })
            
        except Exception as e:
            cosmology_result['error'] = str(e)
            logger.error(f"Cosmological analysis failed: {e}")
        
        return cosmology_result
    
    def design_observational_strategy(self, research_target: Dict[str, Any], 
                                    observing_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design observational strategy for astrophysical research.
        
        Args:
            research_target: Target object and research goals
            observing_constraints: Observational constraints and available resources
            
        Returns:
            Comprehensive observational strategy and plan
        """
        observing_strategy = {
            'success': False,
            'target_analysis': {},
            'optimal_techniques': [],
            'observing_schedule': {},
            'instrumental_requirements': {},
            'data_analysis_plan': {},
            'expected_results': {},
            'feasibility_assessment': {}
        }
        
        try:
            # Analyze target properties
            target_type = research_target.get('type', 'star')
            research_goals = research_target.get('goals', ['photometry'])
            
            # Select optimal observing techniques
            optimal_techniques = self._select_observing_techniques(target_type, research_goals)
            
            # Plan observing schedule
            observing_schedule = self._plan_observing_schedule(
                research_target, optimal_techniques, observing_constraints
            )
            
            # Determine instrumental requirements
            instrumental_requirements = self._determine_instrumental_requirements(
                optimal_techniques, research_target
            )
            
            # Plan data analysis
            data_analysis_plan = self._plan_astrophysical_data_analysis(
                optimal_techniques, research_goals
            )
            
            # Predict expected results
            expected_results = self._predict_observational_results(
                research_target, optimal_techniques
            )
            
            # Assess feasibility
            feasibility = self._assess_observational_feasibility(
                research_target, optimal_techniques, observing_constraints
            )
            
            observing_strategy.update({
                'success': True,
                'target_analysis': {
                    'type': target_type,
                    'goals': research_goals,
                    'coordinates': research_target.get('coordinates', 'unknown'),
                    'magnitude': research_target.get('magnitude', 'unknown')
                },
                'optimal_techniques': optimal_techniques,
                'observing_schedule': observing_schedule,
                'instrumental_requirements': instrumental_requirements,
                'data_analysis_plan': data_analysis_plan,
                'expected_results': expected_results,
                'feasibility_assessment': feasibility
            })
            
        except Exception as e:
            observing_strategy['error'] = str(e)
            logger.error(f"Observational strategy design failed: {e}")
        
        return observing_strategy
    
    def _discover_physics_specific_tools(self, research_question: str) -> List[Dict[str, Any]]:
        """Discover astrophysics specific tools."""
        astro_tools = []
        question_lower = research_question.lower()
        
        # Stellar evolution tools
        if any(keyword in question_lower for keyword in 
               ['stellar', 'star', 'evolution', 'nucleosynthesis']):
            astro_tools.append({
                'tool_id': 'stellar_evolution_calculator',
                'name': 'Stellar Evolution Calculator',
                'description': 'Model stellar evolution and nucleosynthesis',
                'capabilities': ['stellar_structure', 'evolution_tracks', 'nucleosynthesis', 'lifetime_calculation'],
                'confidence': 0.95,
                'physics_specific': True,
                'scales': ['stellar'],
                'methodologies': ['theoretical', 'computational']
            })
        
        # Galactic dynamics tools
        if any(keyword in question_lower for keyword in 
               ['galactic', 'galaxy', 'rotation', 'dark matter']):
            astro_tools.append({
                'tool_id': 'galaxy_modeler',
                'name': 'Galactic Dynamics Modeler',
                'description': 'Model galaxy structure and dynamics',
                'capabilities': ['rotation_curves', 'mass_distribution', 'dark_matter_analysis', 'stability_analysis'],
                'confidence': 0.9,
                'physics_specific': True,
                'scales': ['galactic'],
                'methodologies': ['theoretical', 'computational']
            })
        
        # Cosmological analysis tools
        if any(keyword in question_lower for keyword in 
               ['cosmology', 'universe', 'expansion', 'big bang']):
            astro_tools.append({
                'tool_id': 'cosmology_calculator',
                'name': 'Cosmological Analysis Tool',
                'description': 'Analyze cosmological models and universe evolution',
                'capabilities': ['expansion_modeling', 'distance_calculation', 'age_determination', 'structure_formation'],
                'confidence': 0.88,
                'physics_specific': True,
                'scales': ['cosmic'],
                'methodologies': ['theoretical', 'observational']
            })
        
        # Observational planning tools
        if any(keyword in question_lower for keyword in 
               ['observe', 'telescope', 'photometry', 'spectroscopy']):
            astro_tools.append({
                'tool_id': 'observation_planner',
                'name': 'Astronomical Observation Planner',
                'description': 'Plan and optimize astronomical observations',
                'capabilities': ['target_analysis', 'schedule_optimization', 'instrument_selection', 'data_reduction'],
                'confidence': 0.85,
                'physics_specific': True,
                'scales': ['stellar', 'galactic'],
                'methodologies': ['observational']
            })
        
        return astro_tools
    
    # Stellar evolution methods
    
    def _determine_stellar_phase(self, mass: float, age: float) -> Dict[str, Any]:
        """Determine current stellar evolutionary phase."""
        # Simplified stellar evolution phases
        main_sequence_lifetime = self._calculate_main_sequence_lifetime(mass)
        
        if age < main_sequence_lifetime:
            phase = 'main_sequence'
            fraction_ms = age / main_sequence_lifetime
        elif mass < 0.5:
            phase = 'main_sequence'  # Low-mass stars stay on MS for very long
            fraction_ms = age / main_sequence_lifetime
        elif mass < 8.0:
            if age < main_sequence_lifetime * 1.1:
                phase = 'subgiant'
            elif age < main_sequence_lifetime * 1.2:
                phase = 'red_giant'
            else:
                phase = 'white_dwarf'
        else:
            if age < main_sequence_lifetime * 1.05:
                phase = 'supergiant'
            else:
                phase = 'neutron_star_or_black_hole'
        
        return {
            'phase': phase,
            'main_sequence_lifetime': main_sequence_lifetime,
            'fractional_age': min(1.0, age / main_sequence_lifetime),
            'time_remaining': max(0, main_sequence_lifetime - age)
        }
    
    def _calculate_main_sequence_lifetime(self, mass: float) -> float:
        """Calculate main sequence lifetime in years."""
        # Simplified mass-lifetime relation: t ∝ M^(-2.5)
        solar_ms_lifetime = 1e10  # years
        return solar_ms_lifetime * (mass ** -2.5)
    
    def _calculate_stellar_lifetimes(self, mass: float, metallicity: float) -> Dict[str, float]:
        """Calculate lifetimes for different stellar phases."""
        ms_lifetime = self._calculate_main_sequence_lifetime(mass)
        
        lifetimes = {
            'main_sequence': ms_lifetime,
            'hydrogen_burning': ms_lifetime,
        }
        
        if mass > 0.5:
            lifetimes['subgiant'] = ms_lifetime * 0.1
            lifetimes['red_giant'] = ms_lifetime * 0.1
        
        if mass > 8.0:
            lifetimes['helium_burning'] = ms_lifetime * 0.01
            lifetimes['carbon_burning'] = ms_lifetime * 0.001
            lifetimes['neon_burning'] = ms_lifetime * 0.0001
            lifetimes['silicon_burning'] = ms_lifetime * 0.00001
        
        return lifetimes
    
    def _predict_evolution_phases(self, mass: float, metallicity: float) -> List[Dict[str, Any]]:
        """Predict stellar evolution phases."""
        phases = []
        
        # Main sequence
        phases.append({
            'phase': 'main_sequence',
            'duration': self._calculate_main_sequence_lifetime(mass),
            'burning': 'hydrogen',
            'core_temperature': 1.5e7,  # K
            'luminosity_class': 'V'
        })
        
        if mass > 0.5:
            # Post-main sequence evolution
            phases.append({
                'phase': 'subgiant',
                'duration': self._calculate_main_sequence_lifetime(mass) * 0.1,
                'burning': 'hydrogen_shell',
                'core_temperature': 1e8,
                'luminosity_class': 'IV'
            })
            
            phases.append({
                'phase': 'red_giant',
                'duration': self._calculate_main_sequence_lifetime(mass) * 0.1,
                'burning': 'helium_core',
                'core_temperature': 1e8,
                'luminosity_class': 'III'
            })
        
        if mass > 8.0:
            # Massive star evolution
            phases.extend([
                {
                    'phase': 'supergiant',
                    'duration': self._calculate_main_sequence_lifetime(mass) * 0.01,
                    'burning': 'multiple_shells',
                    'core_temperature': 3e9,
                    'luminosity_class': 'I'
                },
                {
                    'phase': 'core_collapse',
                    'duration': 1,  # seconds
                    'burning': 'silicon',
                    'core_temperature': 1e10,
                    'end_state': 'supernova'
                }
            ])
        
        return phases
    
    def _analyze_stellar_nucleosynthesis(self, mass: float, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nucleosynthesis in stellar interiors."""
        nucleosynthesis = {
            'active_burning': [],
            'elements_produced': [],
            'energy_generation': 0.0,
            'neutrino_luminosity': 0.0
        }
        
        phase = current_state.get('phase', 'main_sequence')
        
        if phase == 'main_sequence':
            nucleosynthesis['active_burning'] = ['hydrogen']
            nucleosynthesis['elements_produced'] = ['helium-4']
            # PP chain or CNO cycle
            if mass < 1.3:
                nucleosynthesis['primary_process'] = 'pp_chain'
            else:
                nucleosynthesis['primary_process'] = 'cno_cycle'
        
        elif phase in ['red_giant', 'supergiant']:
            if mass > 0.5:
                nucleosynthesis['active_burning'] = ['helium']
                nucleosynthesis['elements_produced'] = ['carbon-12', 'oxygen-16']
                nucleosynthesis['primary_process'] = 'triple_alpha'
        
        if mass > 8.0 and phase == 'supergiant':
            nucleosynthesis['active_burning'].extend(['carbon', 'neon', 'oxygen', 'silicon'])
            nucleosynthesis['elements_produced'].extend([
                'neon-20', 'magnesium-24', 'silicon-28', 'iron-56'
            ])
        
        return nucleosynthesis
    
    def _determine_stellar_fate(self, mass: float) -> str:
        """Determine stellar final fate."""
        if mass < 0.08:
            return 'brown_dwarf'
        elif mass < 8.0:
            return 'white_dwarf'
        elif mass < 25.0:
            return 'neutron_star'
        else:
            return 'black_hole'
    
    def _calculate_stellar_structure(self, mass: float, age: float) -> Dict[str, Any]:
        """Calculate stellar structure parameters."""
        # Simplified stellar structure using scaling relations
        
        # Mass-luminosity relation: L ∝ M^3.5
        luminosity = (mass ** 3.5) * self.astro_constants['L_sun']
        
        # Mass-radius relation: R ∝ M^0.8 (main sequence)
        radius = (mass ** 0.8) * self.astro_constants['R_sun']
        
        # Effective temperature from Stefan-Boltzmann law
        effective_temp = (luminosity / (4 * np.pi * radius**2 * self.astro_constants['sigma_SB'])) ** 0.25
        
        # Central temperature (rough estimate)
        central_temp = 1.5e7 * (mass ** 0.5)  # K
        
        # Central density (rough estimate)
        central_density = 1.6e5 * (mass ** 2) / (radius / self.astro_constants['R_sun'])**3  # kg/m³
        
        return {
            'luminosity_watts': luminosity,
            'luminosity_solar': luminosity / self.astro_constants['L_sun'],
            'radius_meters': radius,
            'radius_solar': radius / self.astro_constants['R_sun'],
            'effective_temperature': effective_temp,
            'central_temperature': central_temp,
            'central_density': central_density,
            'surface_gravity': self.astro_constants['G'] * mass * self.astro_constants['M_sun'] / radius**2
        }
    
    # Galactic dynamics methods
    
    def _calculate_galactic_mass_distribution(self, galaxy_type: str, total_mass: float, 
                                            scale_radius: float) -> Dict[str, Any]:
        """Calculate galactic mass distribution."""
        mass_distribution = {
            'total_mass': total_mass,
            'scale_radius': scale_radius,
            'profile_type': 'unknown',
            'central_density': 0.0,
            'mass_components': {}
        }
        
        if galaxy_type == 'spiral':
            # Exponential disk + dark matter halo
            mass_distribution.update({
                'profile_type': 'exponential_disk_plus_halo',
                'mass_components': {
                    'disk': total_mass * 0.05,
                    'bulge': total_mass * 0.01,
                    'dark_halo': total_mass * 0.94
                },
                'disk_scale_length': scale_radius,
                'halo_scale_radius': scale_radius * 10
            })
        
        elif galaxy_type == 'elliptical':
            # de Vaucouleurs profile
            mass_distribution.update({
                'profile_type': 'de_vaucouleurs',
                'mass_components': {
                    'stellar': total_mass * 0.1,
                    'dark_halo': total_mass * 0.9
                },
                'effective_radius': scale_radius
            })
        
        # Calculate central density
        mass_distribution['central_density'] = total_mass / (4 * np.pi * scale_radius**3)
        
        return mass_distribution
    
    def _model_rotation_curve(self, mass_distribution: Dict[str, Any], 
                            galaxy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Model galactic rotation curve."""
        # Generate radii from 0.1 to 100 kpc
        radii = np.logspace(-1, 2, 50)  # kpc
        velocities = []
        
        scale_radius = mass_distribution['scale_radius']
        total_mass = mass_distribution['total_mass']
        
        for r in radii:
            # Simplified rotation curve calculation
            if r < scale_radius:
                # Inner region - solid body rotation approximation
                v_circ = np.sqrt(self.astro_constants['G'] * total_mass * 
                               self.astro_constants['M_sun'] * r / scale_radius**3)
            else:
                # Outer region - Keplerian decline modified by dark matter
                enclosed_mass = total_mass * (1 - np.exp(-r/scale_radius))
                v_circ = np.sqrt(self.astro_constants['G'] * enclosed_mass * 
                               self.astro_constants['M_sun'] / (r * 1000 * self.astro_constants['pc']))
            
            velocities.append(v_circ / 1000)  # Convert to km/s
        
        return {
            'radii_kpc': radii.tolist(),
            'velocities_km_s': velocities,
            'flat_rotation_regime': True if max(velocities) > min(velocities[-10:]) * 1.1 else False,
            'maximum_velocity': max(velocities)
        }
    
    def _analyze_dark_matter_profile(self, rotation_curve: Dict[str, Any], 
                                   mass_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dark matter density profile."""
        dark_matter_profile = {
            'profile_type': 'NFW',
            'characteristic_density': 0.0,
            'scale_radius': 0.0,
            'concentration_parameter': 0.0,
            'dark_matter_fraction': 0.0
        }
        
        # Simplified NFW profile analysis
        total_mass = mass_distribution['total_mass']
        scale_radius = mass_distribution['scale_radius']
        
        # Estimate NFW parameters
        dark_matter_profile.update({
            'characteristic_density': 1e7,  # M_sun/kpc³
            'scale_radius': scale_radius * 5,  # kpc
            'concentration_parameter': 10.0,
            'dark_matter_fraction': 0.85
        })
        
        return dark_matter_profile
    
    def _analyze_galactic_stability(self, mass_distribution: Dict[str, Any], 
                                  rotation_curve: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze galactic stability."""
        stability_analysis = {
            'toomre_q_parameter': 1.5,
            'spiral_arm_stability': 'stable',
            'bar_instability': 'stable',
            'overall_stability': 'stable'
        }
        
        # Simplified stability analysis
        max_velocity = rotation_curve.get('maximum_velocity', 200)
        
        if max_velocity > 300:
            stability_analysis['spiral_arm_stability'] = 'unstable'
            stability_analysis['overall_stability'] = 'marginally_stable'
        
        return stability_analysis
    
    def _model_spiral_structure(self, galaxy_type: str, galaxy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Model spiral structure if applicable."""
        spiral_structure = {
            'spiral_arms': 0,
            'pitch_angle': 0.0,
            'arm_strength': 0.0,
            'pattern_speed': 0.0
        }
        
        if galaxy_type == 'spiral':
            spiral_structure.update({
                'spiral_arms': 2,
                'pitch_angle': 15.0,  # degrees
                'arm_strength': 0.1,
                'pattern_speed': 25.0,  # km/s/kpc
                'corotation_radius': 8.0  # kpc
            })
        
        return spiral_structure
    
    def _analyze_central_black_hole(self, total_mass: float, galaxy_type: str) -> Dict[str, Any]:
        """Analyze central supermassive black hole."""
        # M_BH - M_bulge relation: M_BH ≈ 0.001 * M_bulge
        bulge_fraction = 0.2 if galaxy_type == 'spiral' else 0.8
        bulge_mass = total_mass * bulge_fraction
        bh_mass = bulge_mass * 0.001
        
        central_bh = {
            'mass_solar_masses': bh_mass,
            'schwarzschild_radius': 3e5 * bh_mass,  # km
            'sphere_of_influence': 100 * (bh_mass / 1e8)**0.5,  # pc
            'accretion_rate': 1e-3 * bh_mass,  # M_sun/year (rough estimate)
            'eddington_luminosity': 1.3e38 * bh_mass  # watts
        }
        
        return central_bh
    
    # Cosmological methods
    
    def _calculate_critical_density(self, H0: float) -> float:
        """Calculate critical density of the universe."""
        # ρ_c = 3H²/(8πG)
        H0_si = H0 * 1000 / (1e6 * self.astro_constants['pc'])  # Convert to SI
        critical_density = 3 * H0_si**2 / (8 * np.pi * self.astro_constants['G'])
        return critical_density  # kg/m³
    
    def _calculate_universe_age(self, H0: float, Omega_m: float, Omega_lambda: float) -> float:
        """Calculate age of the universe."""
        # Simplified age calculation for flat ΛCDM
        h = H0 / 100.0
        
        if abs(Omega_m + Omega_lambda - 1.0) < 0.01:  # Flat universe
            # Approximate formula for flat ΛCDM
            age = (2.0 / (3.0 * H0 * 1000 / (1e6 * self.astro_constants['pc']))) * \
                  (1.0 / np.sqrt(Omega_lambda)) * \
                  np.arcsinh(np.sqrt(Omega_lambda / Omega_m))
        else:
            # Simple estimate
            age = 1.0 / (H0 * 1000 / (1e6 * self.astro_constants['pc']))
        
        return age / (365.25 * 24 * 3600)  # Convert to years
    
    def _model_expansion_history(self, H0: float, Omega_m: float, Omega_lambda: float) -> Dict[str, Any]:
        """Model expansion history of the universe."""
        # Redshift range
        z_values = np.logspace(-3, 2, 50)
        a_values = 1 / (1 + z_values)
        
        # Hubble parameter as function of redshift
        H_z = []
        for z in z_values:
            H_z_val = H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)
            H_z.append(H_z_val)
        
        expansion_history = {
            'redshifts': z_values.tolist(),
            'scale_factors': a_values.tolist(),
            'hubble_parameter': H_z,
            'deceleration_parameter': Omega_m / 2 - Omega_lambda,
            'matter_radiation_equality_z': 3400 * Omega_m * (H0 / 70)**2
        }
        
        return expansion_history
    
    def _calculate_cosmological_distances(self, H0: float, Omega_m: float, Omega_lambda: float) -> Dict[str, Any]:
        """Calculate cosmological distance measures."""
        # Sample redshifts
        z_sample = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        distance_measures = {
            'redshifts': z_sample,
            'comoving_distances': [],
            'angular_diameter_distances': [],
            'luminosity_distances': [],
            'lookback_times': []
        }
        
        c_H0 = self.astro_constants['c'] / (H0 * 1000 / (1e6 * self.astro_constants['pc']))
        
        for z in z_sample:
            # Simplified distance calculations
            # Comoving distance (simplified)
            d_c = c_H0 * z * (1 + z/2)  # First-order approximation
            
            # Angular diameter distance
            d_a = d_c / (1 + z)
            
            # Luminosity distance
            d_l = d_c * (1 + z)
            
            # Lookback time (simplified)
            t_lookback = (2.0 / (3.0 * H0 * 1000 / (1e6 * self.astro_constants['pc']))) * \
                        (1 - 1/np.sqrt(1 + z))
            
            distance_measures['comoving_distances'].append(d_c / (1e6 * self.astro_constants['pc']))  # Mpc
            distance_measures['angular_diameter_distances'].append(d_a / (1e6 * self.astro_constants['pc']))
            distance_measures['luminosity_distances'].append(d_l / (1e6 * self.astro_constants['pc']))
            distance_measures['lookback_times'].append(t_lookback / (365.25 * 24 * 3600 * 1e9))  # Gyr
        
        return distance_measures
    
    def _analyze_structure_formation(self, Omega_m: float, Omega_lambda: float) -> Dict[str, Any]:
        """Analyze cosmic structure formation."""
        structure_formation = {
            'linear_growth_factor': 1.0,
            'matter_power_spectrum': {},
            'collapse_redshift': 0.0,
            'nonlinear_scale': 8.0,  # Mpc/h
            'sigma_8': 0.8
        }
        
        # Growth factor approximation
        growth_factor = Omega_m**0.6 + (Omega_lambda/70) * (1 + Omega_m/2)
        structure_formation['linear_growth_factor'] = growth_factor
        
        # Characteristic scales
        structure_formation.update({
            'jeans_length': 150,  # Mpc (matter-radiation equality)
            'silk_damping_scale': 1,  # Mpc
            'first_star_formation_z': 20,
            'reionization_z': 7
        })
        
        return structure_formation
    
    def _predict_universe_future(self, Omega_m: float, Omega_lambda: float) -> Dict[str, Any]:
        """Predict future evolution of the universe."""
        future_evolution = {
            'ultimate_fate': 'unknown',
            'heat_death_time': 0.0,
            'proton_decay_time': 0.0,
            'black_hole_evaporation_time': 0.0
        }
        
        if Omega_lambda > Omega_m:
            future_evolution.update({
                'ultimate_fate': 'heat_death',
                'accelerated_expansion': True,
                'big_rip': False,
                'heat_death_time': 1e100,  # years
                'proton_decay_time': 1e36,  # years (if it occurs)
                'black_hole_evaporation_time': 1e67  # years
            })
        elif Omega_m > 1:
            future_evolution.update({
                'ultimate_fate': 'big_crunch',
                'accelerated_expansion': False,
                'time_to_big_crunch': 1e11  # years (rough estimate)
            })
        else:
            future_evolution['ultimate_fate'] = 'eternal_expansion'
        
        return future_evolution
    
    # Observational methods
    
    def _select_observing_techniques(self, target_type: str, research_goals: List[str]) -> List[Dict[str, Any]]:
        """Select optimal observing techniques."""
        techniques = []
        
        # Photometry
        if any(goal in ['brightness', 'variability', 'color', 'magnitude'] for goal in research_goals):
            techniques.append({
                'technique': 'photometry',
                'filters': ['U', 'B', 'V', 'R', 'I'],
                'precision_required': 0.01,  # magnitudes
                'time_resolution': '1 minute'
            })
        
        # Spectroscopy
        if any(goal in ['composition', 'velocity', 'temperature', 'redshift'] for goal in research_goals):
            techniques.append({
                'technique': 'spectroscopy',
                'resolution': 'R=1000-50000',
                'wavelength_range': '400-900 nm',
                'exposure_time': '1800 seconds'
            })
        
        # Astrometry
        if any(goal in ['position', 'parallax', 'proper_motion'] for goal in research_goals):
            techniques.append({
                'technique': 'astrometry',
                'precision_required': '0.001 arcsec',
                'reference_catalog': 'Gaia',
                'observation_epochs': 'multiple'
            })
        
        # Time-series analysis
        if 'variability' in research_goals:
            techniques.append({
                'technique': 'time_series_photometry',
                'cadence': '1 hour',
                'duration': '30 days',
                'filters': ['V', 'R']
            })
        
        return techniques
    
    def _plan_observing_schedule(self, target: Dict[str, Any], techniques: List[Dict[str, Any]], 
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Plan optimal observing schedule."""
        observing_schedule = {
            'total_nights': 0,
            'optimal_observing_windows': [],
            'seasonal_constraints': {},
            'moon_phase_preferences': {},
            'weather_requirements': {}
        }
        
        # Calculate observing requirements
        target_coords = target.get('coordinates', {'ra': 180, 'dec': 0})
        magnitude = target.get('magnitude', 10)
        
        # Estimate total observing time needed
        total_time_hours = 0
        for technique in techniques:
            if technique['technique'] == 'photometry':
                total_time_hours += 2
            elif technique['technique'] == 'spectroscopy':
                total_time_hours += 4
            elif technique['technique'] == 'time_series_photometry':
                total_time_hours += 30 * 8  # 30 nights, 8 hours per night
        
        observing_schedule.update({
            'total_nights': int(total_time_hours / 8) + 1,
            'total_hours': total_time_hours,
            'optimal_observing_windows': [
                {'start_date': '2024-01-01', 'end_date': '2024-01-31', 'quality': 'excellent'},
                {'start_date': '2024-11-01', 'end_date': '2024-11-30', 'quality': 'good'}
            ],
            'moon_phase_preferences': {
                'photometry': 'new_moon',
                'spectroscopy': 'any',
                'astrometry': 'new_moon'
            }
        })
        
        return observing_schedule
    
    def _determine_instrumental_requirements(self, techniques: List[Dict[str, Any]], 
                                           target: Dict[str, Any]) -> Dict[str, Any]:
        """Determine instrumental requirements."""
        requirements = {
            'telescope_aperture': '1 meter',
            'instruments_needed': [],
            'filters_required': [],
            'detector_specifications': {},
            'auxiliary_equipment': []
        }
        
        magnitude = target.get('magnitude', 10)
        
        # Telescope size based on target magnitude
        if magnitude > 15:
            requirements['telescope_aperture'] = '4+ meters'
        elif magnitude > 10:
            requirements['telescope_aperture'] = '2-4 meters'
        else:
            requirements['telescope_aperture'] = '1-2 meters'
        
        # Instruments based on techniques
        for technique in techniques:
            if technique['technique'] == 'photometry':
                requirements['instruments_needed'].append('CCD_camera')
                requirements['filters_required'].extend(['U', 'B', 'V', 'R', 'I'])
            elif technique['technique'] == 'spectroscopy':
                requirements['instruments_needed'].append('spectrograph')
                requirements['detector_specifications']['spectral_resolution'] = 'R=1000-50000'
            elif technique['technique'] == 'astrometry':
                requirements['instruments_needed'].append('precision_CCD')
                requirements['auxiliary_equipment'].append('field_derotator')
        
        return requirements
    
    def _plan_astrophysical_data_analysis(self, techniques: List[Dict[str, Any]], 
                                        goals: List[str]) -> Dict[str, Any]:
        """Plan data analysis methodology."""
        analysis_plan = {
            'data_reduction_steps': [],
            'calibration_procedures': [],
            'analysis_methods': [],
            'statistical_approaches': [],
            'software_tools': []
        }
        
        # Standard data reduction
        analysis_plan['data_reduction_steps'] = [
            'bias_subtraction',
            'dark_current_removal',
            'flat_field_correction',
            'cosmic_ray_removal'
        ]
        
        # Technique-specific analysis
        for technique in techniques:
            if technique['technique'] == 'photometry':
                analysis_plan['analysis_methods'].extend([
                    'aperture_photometry',
                    'PSF_photometry',
                    'differential_photometry'
                ])
                analysis_plan['software_tools'].append('IRAF/PyRAF')
            
            elif technique['technique'] == 'spectroscopy':
                analysis_plan['analysis_methods'].extend([
                    'wavelength_calibration',
                    'flux_calibration',
                    'line_identification',
                    'equivalent_width_measurement'
                ])
                analysis_plan['software_tools'].append('IRAF/PyRAF')
            
            elif technique['technique'] == 'astrometry':
                analysis_plan['analysis_methods'].extend([
                    'plate_solution',
                    'proper_motion_analysis',
                    'parallax_measurement'
                ])
                analysis_plan['software_tools'].append('Gaia_tools')
        
        return analysis_plan
    
    def _predict_observational_results(self, target: Dict[str, Any], 
                                     techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict expected observational results."""
        expected_results = {
            'measurement_precision': {},
            'expected_uncertainties': {},
            'detection_limits': {},
            'success_probability': 0.8
        }
        
        magnitude = target.get('magnitude', 10)
        
        for technique in techniques:
            technique_name = technique['technique']
            
            if technique_name == 'photometry':
                # Photometric precision depends on magnitude
                if magnitude < 12:
                    precision = 0.01  # 1% precision
                elif magnitude < 15:
                    precision = 0.03  # 3% precision
                else:
                    precision = 0.1   # 10% precision
                
                expected_results['measurement_precision'][technique_name] = precision
                expected_results['detection_limits'][technique_name] = 'V = 20 mag'
            
            elif technique_name == 'spectroscopy':
                # Spectroscopic S/N depends on magnitude and exposure time
                if magnitude < 10:
                    snr = 100
                elif magnitude < 15:
                    snr = 50
                else:
                    snr = 10
                
                expected_results['measurement_precision'][technique_name] = f'S/N = {snr}'
                expected_results['expected_uncertainties'][technique_name] = {
                    'radial_velocity': '1 km/s',
                    'equivalent_width': '10%'
                }
        
        return expected_results
    
    def _assess_observational_feasibility(self, target: Dict[str, Any], 
                                        techniques: List[Dict[str, Any]], 
                                        constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Assess observational feasibility."""
        feasibility = {
            'overall_feasibility': 'high',
            'limiting_factors': [],
            'risk_assessment': {},
            'alternative_strategies': [],
            'estimated_success_rate': 0.8
        }
        
        magnitude = target.get('magnitude', 10)
        budget = constraints.get('budget', 50000)
        
        # Check magnitude limits
        if magnitude > 18:
            feasibility['limiting_factors'].append('target_too_faint')
            feasibility['overall_feasibility'] = 'low'
            feasibility['estimated_success_rate'] = 0.3
        
        # Check budget constraints
        estimated_cost = len(techniques) * 5000  # Simplified cost model
        if estimated_cost > budget:
            feasibility['limiting_factors'].append('insufficient_budget')
            feasibility['alternative_strategies'].append('reduce_number_of_techniques')
        
        # Weather risks
        feasibility['risk_assessment'] = {
            'weather_risk': 'medium',
            'equipment_failure_risk': 'low',
            'scheduling_conflicts': 'medium'
        }
        
        return feasibility