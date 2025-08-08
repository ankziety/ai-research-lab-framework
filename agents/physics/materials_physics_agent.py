"""
Materials Physics Agent - Specialized agent for materials science and condensed matter physics.

This agent provides expertise in materials physics, condensed matter theory,
crystal structures, and materials characterization.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import optimize
import matplotlib.pyplot as plt

from .base_physics_agent import BasePhysicsAgent, PhysicsScale, PhysicsMethodology

logger = logging.getLogger(__name__)


class MaterialsPhysicsAgent(BasePhysicsAgent):
    """
    Specialized agent for materials physics and condensed matter research.
    
    Expertise includes:
    - Crystal structures and crystallography
    - Electronic band structure calculations
    - Mechanical properties and elasticity
    - Thermal and electrical transport
    - Phase transitions and critical phenomena
    - Surface and interface physics
    - Defects and disorder in materials
    """
    
    def __init__(self, agent_id: str, role: str = None, expertise: List[str] = None,
                 model_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        """
        Initialize materials physics agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent role (defaults to "Materials Physics Expert")
            expertise: List of expertise areas (uses defaults if None)
            model_config: Configuration for the underlying LLM
            cost_manager: Optional cost manager for tracking API usage
        """
        if role is None:
            role = "Materials Physics Expert"
        
        if expertise is None:
            expertise = [
                "Condensed Matter Physics",
                "Crystal Structures",
                "Electronic Properties",
                "Mechanical Properties",
                "Thermal Properties",
                "Materials Characterization",
                "Phase Transitions",
                "Surface Physics",
                "Defects and Disorder",
                "Materials Synthesis"
            ]
        
        super().__init__(agent_id, role, expertise, model_config, cost_manager)
        
        # Crystal systems and space groups
        self.crystal_systems = {
            'cubic': {
                'lattice_parameters': ['a'],
                'angles': [90, 90, 90],
                'constraints': 'a = b = c, α = β = γ = 90°',
                'examples': ['diamond', 'NaCl', 'CsCl']
            },
            'tetragonal': {
                'lattice_parameters': ['a', 'c'],
                'angles': [90, 90, 90],
                'constraints': 'a = b ≠ c, α = β = γ = 90°',
                'examples': ['TiO2', 'SnO2']
            },
            'orthorhombic': {
                'lattice_parameters': ['a', 'b', 'c'],
                'angles': [90, 90, 90],
                'constraints': 'a ≠ b ≠ c, α = β = γ = 90°',
                'examples': ['MgSO4', 'KNO3']
            },
            'hexagonal': {
                'lattice_parameters': ['a', 'c'],
                'angles': [90, 90, 120],
                'constraints': 'a = b ≠ c, α = β = 90°, γ = 120°',
                'examples': ['graphite', 'ZnO', 'SiC']
            },
            'trigonal': {
                'lattice_parameters': ['a', 'α'],
                'angles': 'variable',
                'constraints': 'a = b = c, α = β = γ ≠ 90°',
                'examples': ['quartz', 'calcite']
            },
            'monoclinic': {
                'lattice_parameters': ['a', 'b', 'c', 'β'],
                'angles': [90, 'variable', 90],
                'constraints': 'a ≠ b ≠ c, α = γ = 90°, β ≠ 90°',
                'examples': ['gypsum', 'orthoclase']
            },
            'triclinic': {
                'lattice_parameters': ['a', 'b', 'c', 'α', 'β', 'γ'],
                'angles': 'all variable',
                'constraints': 'a ≠ b ≠ c, α ≠ β ≠ γ ≠ 90°',
                'examples': ['CuSO4·5H2O']
            }
        }
        
        # Material properties database
        self.material_properties = {
            'mechanical': {
                'elastic_constants': ['C11', 'C12', 'C44'],
                'moduli': ['young', 'bulk', 'shear'],
                'hardness': ['vickers', 'brinell', 'rockwell'],
                'strength': ['tensile', 'compressive', 'yield']
            },
            'thermal': {
                'conductivity': 'thermal_conductivity',
                'expansion': 'thermal_expansion_coefficient',
                'capacity': 'specific_heat_capacity',
                'diffusivity': 'thermal_diffusivity'
            },
            'electrical': {
                'conductivity': 'electrical_conductivity',
                'resistivity': 'electrical_resistivity',
                'permittivity': 'dielectric_constant',
                'band_gap': 'electronic_band_gap'
            },
            'magnetic': {
                'susceptibility': 'magnetic_susceptibility',
                'permeability': 'magnetic_permeability',
                'moment': 'magnetic_moment',
                'anisotropy': 'magnetic_anisotropy'
            }
        }
        
        # Characterization techniques
        self.characterization_techniques = {
            'structural': {
                'xrd': 'X-ray diffraction',
                'neutron_diffraction': 'Neutron diffraction',
                'electron_diffraction': 'Electron diffraction',
                'tem': 'Transmission electron microscopy',
                'sem': 'Scanning electron microscopy'
            },
            'spectroscopic': {
                'xps': 'X-ray photoelectron spectroscopy',
                'raman': 'Raman spectroscopy',
                'ftir': 'Fourier transform infrared spectroscopy',
                'uv_vis': 'UV-visible spectroscopy',
                'nmr': 'Nuclear magnetic resonance'
            },
            'surface': {
                'afm': 'Atomic force microscopy',
                'stm': 'Scanning tunneling microscopy',
                'xrf': 'X-ray fluorescence',
                'auger': 'Auger electron spectroscopy'
            }
        }
        
        logger.info(f"Materials Physics Agent {self.agent_id} initialized")
    
    def _get_physics_domain(self) -> str:
        """Get the physics domain for materials physics."""
        return "materials_physics"
    
    def _get_relevant_scales(self) -> List[PhysicsScale]:
        """Get physical scales relevant to materials physics."""
        return [
            PhysicsScale.ATOMIC,
            PhysicsScale.NANO,
            PhysicsScale.MICRO,
            PhysicsScale.MESO,
            PhysicsScale.MACRO
        ]
    
    def _get_preferred_methodologies(self) -> List[PhysicsMethodology]:
        """Get preferred methodologies for materials physics."""
        return [
            PhysicsMethodology.THEORETICAL,
            PhysicsMethodology.COMPUTATIONAL,
            PhysicsMethodology.EXPERIMENTAL
        ]
    
    def analyze_crystal_structure(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze crystal structure and determine properties.
        
        Args:
            structure_data: Crystal structure data including lattice parameters and atomic positions
            
        Returns:
            Comprehensive crystal structure analysis
        """
        analysis_result = {
            'success': False,
            'crystal_system': 'unknown',
            'space_group': 'unknown',
            'lattice_parameters': {},
            'unit_cell_volume': 0.0,
            'atomic_positions': [],
            'symmetry_analysis': {},
            'structure_factor': {},
            'density': 0.0,
            'coordination_analysis': {}
        }
        
        try:
            # Extract lattice parameters
            lattice_params = structure_data.get('lattice_parameters', {})
            
            # Determine crystal system
            crystal_system = self._determine_crystal_system(lattice_params)
            
            # Calculate unit cell volume
            volume = self._calculate_unit_cell_volume(lattice_params, crystal_system)
            
            # Analyze atomic positions
            atomic_positions = structure_data.get('atomic_positions', [])
            
            # Perform symmetry analysis
            symmetry_analysis = self._analyze_crystal_symmetry(lattice_params, atomic_positions)
            
            # Calculate structure factors
            structure_factors = self._calculate_structure_factors(atomic_positions, lattice_params)
            
            # Analyze coordination
            coordination_analysis = self._analyze_coordination(atomic_positions, lattice_params)
            
            # Calculate density
            density = self._calculate_crystal_density(structure_data, volume)
            
            analysis_result.update({
                'success': True,
                'crystal_system': crystal_system,
                'lattice_parameters': lattice_params,
                'unit_cell_volume': volume,
                'atomic_positions': atomic_positions,
                'symmetry_analysis': symmetry_analysis,
                'structure_factor': structure_factors,
                'density': density,
                'coordination_analysis': coordination_analysis
            })
            
        except Exception as e:
            analysis_result['error'] = str(e)
            logger.error(f"Crystal structure analysis failed: {e}")
        
        return analysis_result
    
    def calculate_electronic_properties(self, material_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate electronic properties of materials.
        
        Args:
            material_config: Material configuration including structure and composition
            
        Returns:
            Electronic properties including band structure and DOS
        """
        electronic_result = {
            'success': False,
            'band_structure': {},
            'density_of_states': {},
            'band_gap': 0.0,
            'fermi_level': 0.0,
            'conductivity_type': 'unknown',
            'effective_masses': {},
            'optical_properties': {}
        }
        
        try:
            # Simplified electronic structure calculation
            band_gap = self._calculate_band_gap(material_config)
            
            # Determine conductivity type
            conductivity_type = self._determine_conductivity_type(band_gap, material_config)
            
            # Calculate density of states
            dos = self._calculate_density_of_states(material_config, band_gap)
            
            # Calculate effective masses
            effective_masses = self._calculate_effective_masses(material_config)
            
            # Calculate optical properties
            optical_props = self._calculate_optical_properties(band_gap, material_config)
            
            electronic_result.update({
                'success': True,
                'band_gap': band_gap,
                'conductivity_type': conductivity_type,
                'density_of_states': dos,
                'effective_masses': effective_masses,
                'optical_properties': optical_props
            })
            
        except Exception as e:
            electronic_result['error'] = str(e)
            logger.error(f"Electronic properties calculation failed: {e}")
        
        return electronic_result
    
    def analyze_mechanical_properties(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze mechanical properties of materials.
        
        Args:
            material_data: Material data including structure and composition
            
        Returns:
            Comprehensive mechanical properties analysis
        """
        mechanical_result = {
            'success': False,
            'elastic_constants': {},
            'elastic_moduli': {},
            'poisson_ratios': {},
            'hardness_estimates': {},
            'strength_properties': {},
            'anisotropy_analysis': {}
        }
        
        try:
            # Calculate elastic constants
            elastic_constants = self._calculate_elastic_constants(material_data)
            
            # Calculate elastic moduli
            elastic_moduli = self._calculate_elastic_moduli(elastic_constants)
            
            # Calculate Poisson's ratios
            poisson_ratios = self._calculate_poisson_ratios(elastic_constants)
            
            # Estimate hardness
            hardness_estimates = self._estimate_hardness(elastic_moduli, material_data)
            
            # Analyze anisotropy
            anisotropy_analysis = self._analyze_elastic_anisotropy(elastic_constants)
            
            mechanical_result.update({
                'success': True,
                'elastic_constants': elastic_constants,
                'elastic_moduli': elastic_moduli,
                'poisson_ratios': poisson_ratios,
                'hardness_estimates': hardness_estimates,
                'anisotropy_analysis': anisotropy_analysis
            })
            
        except Exception as e:
            mechanical_result['error'] = str(e)
            logger.error(f"Mechanical properties analysis failed: {e}")
        
        return mechanical_result
    
    def predict_phase_diagram(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict phase diagram for material systems.
        
        Args:
            system_config: System configuration including components and conditions
            
        Returns:
            Phase diagram prediction with stability regions
        """
        phase_diagram_result = {
            'success': False,
            'phases': [],
            'phase_boundaries': [],
            'critical_points': [],
            'stability_analysis': {},
            'thermodynamic_properties': {}
        }
        
        try:
            # Identify potential phases
            phases = self._identify_phases(system_config)
            
            # Calculate phase boundaries
            phase_boundaries = self._calculate_phase_boundaries(phases, system_config)
            
            # Find critical points
            critical_points = self._find_critical_points(phase_boundaries, system_config)
            
            # Analyze phase stability
            stability_analysis = self._analyze_phase_stability(phases, system_config)
            
            # Calculate thermodynamic properties
            thermo_props = self._calculate_thermodynamic_properties(phases, system_config)
            
            phase_diagram_result.update({
                'success': True,
                'phases': phases,
                'phase_boundaries': phase_boundaries,
                'critical_points': critical_points,
                'stability_analysis': stability_analysis,
                'thermodynamic_properties': thermo_props
            })
            
        except Exception as e:
            phase_diagram_result['error'] = str(e)
            logger.error(f"Phase diagram prediction failed: {e}")
        
        return phase_diagram_result
    
    def design_characterization_experiment(self, material_info: Dict[str, Any], 
                                         research_goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design characterization experiments for materials.
        
        Args:
            material_info: Information about the material to be characterized
            research_goals: Research objectives and questions
            
        Returns:
            Comprehensive characterization experimental plan
        """
        experiment_design = {
            'success': False,
            'characterization_techniques': [],
            'experimental_protocols': {},
            'sample_preparation': {},
            'measurement_sequence': [],
            'expected_information': {},
            'resource_requirements': {}
        }
        
        try:
            # Analyze material properties to determine appropriate techniques
            material_type = material_info.get('type', 'unknown')
            research_objectives = research_goals.get('objectives', [])
            
            # Select characterization techniques
            techniques = self._select_characterization_techniques(material_info, research_objectives)
            
            # Design experimental protocols
            protocols = self._design_characterization_protocols(techniques, material_info)
            
            # Plan sample preparation
            sample_prep = self._plan_sample_preparation(material_info, techniques)
            
            # Sequence measurements
            measurement_sequence = self._sequence_measurements(techniques, protocols)
            
            # Predict expected information
            expected_info = self._predict_expected_information(techniques, material_info)
            
            # Estimate resources
            resources = self._estimate_characterization_resources(techniques, protocols)
            
            experiment_design.update({
                'success': True,
                'characterization_techniques': techniques,
                'experimental_protocols': protocols,
                'sample_preparation': sample_prep,
                'measurement_sequence': measurement_sequence,
                'expected_information': expected_info,
                'resource_requirements': resources
            })
            
        except Exception as e:
            experiment_design['error'] = str(e)
            logger.error(f"Characterization experiment design failed: {e}")
        
        return experiment_design
    
    def _discover_physics_specific_tools(self, research_question: str) -> List[Dict[str, Any]]:
        """Discover materials physics specific tools."""
        materials_tools = []
        question_lower = research_question.lower()
        
        # Crystal structure analysis tools
        if any(keyword in question_lower for keyword in 
               ['crystal', 'structure', 'lattice', 'diffraction']):
            materials_tools.append({
                'tool_id': 'crystal_analyzer',
                'name': 'Crystal Structure Analysis Tool',
                'description': 'Comprehensive crystal structure analysis and visualization',
                'capabilities': ['structure_analysis', 'symmetry_detection', 'lattice_calculation'],
                'confidence': 0.95,
                'physics_specific': True,
                'scales': ['atomic', 'nano'],
                'methodologies': ['theoretical', 'computational']
            })
        
        # Electronic properties tools
        if any(keyword in question_lower for keyword in 
               ['electronic', 'band', 'conductivity', 'semiconductor']):
            materials_tools.append({
                'tool_id': 'electronic_calculator',
                'name': 'Electronic Properties Calculator',
                'description': 'Calculate electronic band structure and properties',
                'capabilities': ['band_structure', 'dos_calculation', 'conductivity_analysis'],
                'confidence': 0.9,
                'physics_specific': True,
                'scales': ['atomic', 'nano', 'micro'],
                'methodologies': ['theoretical', 'computational']
            })
        
        # Mechanical properties tools
        if any(keyword in question_lower for keyword in 
               ['mechanical', 'elastic', 'strength', 'hardness']):
            materials_tools.append({
                'tool_id': 'mechanical_analyzer',
                'name': 'Mechanical Properties Analyzer',
                'description': 'Analyze and predict mechanical properties',
                'capabilities': ['elastic_constants', 'strength_prediction', 'hardness_estimation'],
                'confidence': 0.88,
                'physics_specific': True,
                'scales': ['nano', 'micro', 'macro'],
                'methodologies': ['theoretical', 'computational', 'experimental']
            })
        
        return materials_tools
    
    # Private helper methods for crystal structure analysis
    
    def _determine_crystal_system(self, lattice_params: Dict[str, float]) -> str:
        """Determine crystal system from lattice parameters."""
        a = lattice_params.get('a', 0)
        b = lattice_params.get('b', 0)
        c = lattice_params.get('c', 0)
        alpha = lattice_params.get('alpha', 90)
        beta = lattice_params.get('beta', 90)
        gamma = lattice_params.get('gamma', 90)
        
        tolerance = 1e-3
        
        # Check angles
        angles_90 = abs(alpha - 90) < tolerance and abs(beta - 90) < tolerance and abs(gamma - 90) < tolerance
        alpha_beta_90 = abs(alpha - 90) < tolerance and abs(beta - 90) < tolerance
        gamma_120 = abs(gamma - 120) < tolerance
        
        # Check lattice parameters
        a_eq_b = abs(a - b) < tolerance * a
        b_eq_c = abs(b - c) < tolerance * b
        a_eq_c = abs(a - c) < tolerance * a
        all_equal = a_eq_b and b_eq_c
        
        if angles_90:
            if all_equal:
                return 'cubic'
            elif a_eq_b and not b_eq_c:
                return 'tetragonal'
            elif not a_eq_b and not b_eq_c and not a_eq_c:
                return 'orthorhombic'
        elif alpha_beta_90 and gamma_120:
            if a_eq_b:
                return 'hexagonal'
        elif all_equal:
            return 'trigonal'
        elif alpha_beta_90 and abs(gamma - 90) > tolerance:
            return 'monoclinic'
        else:
            return 'triclinic'
        
        return 'unknown'
    
    def _calculate_unit_cell_volume(self, lattice_params: Dict[str, float], 
                                   crystal_system: str) -> float:
        """Calculate unit cell volume."""
        a = lattice_params.get('a', 1.0)
        b = lattice_params.get('b', a)
        c = lattice_params.get('c', a)
        alpha = np.radians(lattice_params.get('alpha', 90))
        beta = np.radians(lattice_params.get('beta', 90))
        gamma = np.radians(lattice_params.get('gamma', 90))
        
        # General formula for unit cell volume
        volume = a * b * c * np.sqrt(
            1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
        )
        
        return volume
    
    def _analyze_crystal_symmetry(self, lattice_params: Dict[str, float], 
                                 atomic_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze crystal symmetry."""
        symmetry_analysis = {
            'point_group': 'unknown',
            'space_group': 'unknown',
            'symmetry_operations': [],
            'symmetry_elements': []
        }
        
        # Simplified symmetry analysis based on crystal system
        crystal_system = self._determine_crystal_system(lattice_params)
        
        if crystal_system == 'cubic':
            symmetry_analysis.update({
                'point_group': 'm-3m',
                'symmetry_operations': ['identity', 'rotations', 'reflections', 'inversions'],
                'symmetry_elements': ['rotation_axes', 'mirror_planes', 'inversion_center']
            })
        elif crystal_system == 'hexagonal':
            symmetry_analysis.update({
                'point_group': '6/mmm',
                'symmetry_operations': ['identity', '6-fold_rotation', 'reflections'],
                'symmetry_elements': ['6-fold_axis', 'mirror_planes']
            })
        # Add more crystal systems as needed
        
        return symmetry_analysis
    
    def _calculate_structure_factors(self, atomic_positions: List[Dict[str, Any]], 
                                   lattice_params: Dict[str, float]) -> Dict[str, Any]:
        """Calculate structure factors for diffraction."""
        structure_factors = {
            'calculated_reflections': [],
            'systematic_absences': [],
            'intensity_ratios': {}
        }
        
        # Simplified structure factor calculation
        if atomic_positions:
            # Generate some example reflections
            reflections = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0)]
            
            for h, k, l in reflections:
                # Simplified structure factor calculation
                f_hkl = complex(0, 0)
                
                for atom in atomic_positions:
                    x = atom.get('x', 0)
                    y = atom.get('y', 0)
                    z = atom.get('z', 0)
                    f_atom = atom.get('form_factor', 1.0)  # Atomic form factor
                    
                    phase = 2 * np.pi * (h * x + k * y + l * z)
                    f_hkl += f_atom * complex(np.cos(phase), np.sin(phase))
                
                intensity = abs(f_hkl)**2
                structure_factors['calculated_reflections'].append({
                    'hkl': (h, k, l),
                    'structure_factor': abs(f_hkl),
                    'intensity': intensity
                })
        
        return structure_factors
    
    def _analyze_coordination(self, atomic_positions: List[Dict[str, Any]], 
                            lattice_params: Dict[str, float]) -> Dict[str, Any]:
        """Analyze atomic coordination."""
        coordination_analysis = {
            'coordination_numbers': {},
            'bond_lengths': {},
            'bond_angles': {},
            'coordination_polyhedra': {}
        }
        
        # Simplified coordination analysis
        if len(atomic_positions) > 1:
            for i, atom1 in enumerate(atomic_positions):
                atom1_type = atom1.get('element', f'atom_{i}')
                coordination_analysis['coordination_numbers'][atom1_type] = 0
                coordination_analysis['bond_lengths'][atom1_type] = []
                
                x1, y1, z1 = atom1.get('x', 0), atom1.get('y', 0), atom1.get('z', 0)
                
                for j, atom2 in enumerate(atomic_positions):
                    if i != j:
                        x2, y2, z2 = atom2.get('x', 0), atom2.get('y', 0), atom2.get('z', 0)
                        
                        # Calculate distance
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                        
                        # Check if within coordination sphere (simplified)
                        if distance < 3.0:  # Arbitrary cutoff
                            coordination_analysis['coordination_numbers'][atom1_type] += 1
                            coordination_analysis['bond_lengths'][atom1_type].append(distance)
        
        return coordination_analysis
    
    def _calculate_crystal_density(self, structure_data: Dict[str, Any], volume: float) -> float:
        """Calculate crystal density."""
        atomic_positions = structure_data.get('atomic_positions', [])
        
        if not atomic_positions or volume == 0:
            return 0.0
        
        # Calculate total mass in unit cell
        total_mass = 0.0
        atomic_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'Si': 28.085, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38
        }  # Simplified atomic mass database
        
        for atom in atomic_positions:
            element = atom.get('element', 'C')
            mass = atomic_masses.get(element, 12.0)  # Default to carbon
            total_mass += mass
        
        # Convert to kg/m³
        avogadro = 6.022e23
        volume_m3 = volume * 1e-30  # Convert from Ų to m³
        density = (total_mass / avogadro) / volume_m3
        
        return density
    
    # Electronic properties methods
    
    def _calculate_band_gap(self, material_config: Dict[str, Any]) -> float:
        """Calculate electronic band gap."""
        material_type = material_config.get('type', 'semiconductor')
        elements = material_config.get('elements', ['Si'])
        
        # Simplified band gap estimation
        band_gaps = {
            'Si': 1.12, 'Ge': 0.67, 'GaAs': 1.42, 'GaN': 3.4,
            'ZnO': 3.37, 'TiO2': 3.2, 'SiO2': 9.0
        }
        
        if len(elements) == 1:
            return band_gaps.get(elements[0], 1.0)
        elif len(elements) == 2:
            # Binary compound estimation
            compound = ''.join(sorted(elements))
            return band_gaps.get(compound, 2.0)
        else:
            return 2.0  # Default value
    
    def _determine_conductivity_type(self, band_gap: float, 
                                   material_config: Dict[str, Any]) -> str:
        """Determine electrical conductivity type."""
        if band_gap < 0.1:
            return 'metal'
        elif band_gap < 3.0:
            return 'semiconductor'
        else:
            return 'insulator'
    
    def _calculate_density_of_states(self, material_config: Dict[str, Any], 
                                   band_gap: float) -> Dict[str, Any]:
        """Calculate electronic density of states."""
        dos = {
            'valence_band_maximum': 0.0,
            'conduction_band_minimum': band_gap,
            'dos_at_fermi': 0.0,
            'effective_dos_valence': 1e19,
            'effective_dos_conduction': 1e19
        }
        
        # Simplified DOS calculation
        dos['dos_at_fermi'] = 0.0 if band_gap > 0.1 else 1e22
        
        return dos
    
    def _calculate_effective_masses(self, material_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effective masses."""
        # Simplified effective mass estimation
        effective_masses = {
            'electron_effective_mass': 0.5,  # In units of free electron mass
            'hole_effective_mass': 0.8,
            'density_of_states_effective_mass_electrons': 0.6,
            'density_of_states_effective_mass_holes': 0.9
        }
        
        return effective_masses
    
    def _calculate_optical_properties(self, band_gap: float, 
                                    material_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optical properties."""
        optical_props = {
            'absorption_edge': band_gap,  # eV
            'refractive_index': 2.0,
            'dielectric_constant': 10.0,
            'optical_conductivity': 1e-6
        }
        
        # Estimate refractive index from band gap
        if band_gap > 0:
            optical_props['refractive_index'] = 1.5 + 2.0 / band_gap
        
        return optical_props
    
    # Mechanical properties methods
    
    def _calculate_elastic_constants(self, material_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate elastic constants."""
        # Simplified elastic constants estimation
        crystal_system = material_data.get('crystal_system', 'cubic')
        
        if crystal_system == 'cubic':
            # Cubic system has 3 independent elastic constants
            elastic_constants = {
                'C11': 200.0,  # GPa
                'C12': 50.0,   # GPa
                'C44': 75.0    # GPa
            }
        elif crystal_system == 'hexagonal':
            # Hexagonal system has 5 independent elastic constants
            elastic_constants = {
                'C11': 200.0, 'C33': 220.0, 'C12': 50.0,
                'C13': 40.0, 'C44': 75.0
            }
        else:
            # Default cubic values
            elastic_constants = {
                'C11': 200.0, 'C12': 50.0, 'C44': 75.0
            }
        
        return elastic_constants
    
    def _calculate_elastic_moduli(self, elastic_constants: Dict[str, float]) -> Dict[str, float]:
        """Calculate elastic moduli from elastic constants."""
        C11 = elastic_constants.get('C11', 200.0)
        C12 = elastic_constants.get('C12', 50.0)
        C44 = elastic_constants.get('C44', 75.0)
        
        # For cubic crystals
        bulk_modulus = (C11 + 2*C12) / 3
        shear_modulus = C44
        
        # Young's modulus (Voigt average)
        youngs_modulus = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
        
        elastic_moduli = {
            'bulk_modulus': bulk_modulus,
            'shear_modulus': shear_modulus,
            'youngs_modulus': youngs_modulus
        }
        
        return elastic_moduli
    
    def _calculate_poisson_ratios(self, elastic_constants: Dict[str, float]) -> Dict[str, float]:
        """Calculate Poisson's ratios."""
        C11 = elastic_constants.get('C11', 200.0)
        C12 = elastic_constants.get('C12', 50.0)
        
        # For cubic crystals
        poisson_ratio = C12 / (C11 + C12)
        
        poisson_ratios = {
            'poisson_ratio_xy': poisson_ratio,
            'poisson_ratio_xz': poisson_ratio,
            'poisson_ratio_yz': poisson_ratio
        }
        
        return poisson_ratios
    
    def _estimate_hardness(self, elastic_moduli: Dict[str, float], 
                          material_data: Dict[str, Any]) -> Dict[str, float]:
        """Estimate material hardness."""
        bulk_modulus = elastic_moduli.get('bulk_modulus', 100.0)
        shear_modulus = elastic_moduli.get('shear_modulus', 50.0)
        
        # Empirical hardness estimation
        vickers_hardness = 0.1 * (bulk_modulus + shear_modulus)  # GPa
        
        hardness_estimates = {
            'vickers_hardness': vickers_hardness,
            'brinell_hardness': vickers_hardness * 0.9,
            'estimated_mohs_hardness': min(10, vickers_hardness / 2)
        }
        
        return hardness_estimates
    
    def _analyze_elastic_anisotropy(self, elastic_constants: Dict[str, float]) -> Dict[str, Any]:
        """Analyze elastic anisotropy."""
        anisotropy_analysis = {
            'anisotropy_factor': 1.0,
            'anisotropy_type': 'isotropic',
            'direction_dependent_properties': {}
        }
        
        C11 = elastic_constants.get('C11', 200.0)
        C12 = elastic_constants.get('C12', 50.0)
        C44 = elastic_constants.get('C44', 75.0)
        
        # Zener anisotropy ratio for cubic crystals
        if C11 != C12:
            anisotropy_factor = 2 * C44 / (C11 - C12)
            anisotropy_analysis['anisotropy_factor'] = anisotropy_factor
            
            if abs(anisotropy_factor - 1.0) < 0.1:
                anisotropy_analysis['anisotropy_type'] = 'nearly_isotropic'
            elif anisotropy_factor > 1.5 or anisotropy_factor < 0.5:
                anisotropy_analysis['anisotropy_type'] = 'highly_anisotropic'
            else:
                anisotropy_analysis['anisotropy_type'] = 'moderately_anisotropic'
        
        return anisotropy_analysis
    
    # Phase diagram methods
    
    def _identify_phases(self, system_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential phases in the system."""
        components = system_config.get('components', ['A', 'B'])
        temperature_range = system_config.get('temperature_range', [300, 1500])
        
        phases = []
        
        # Pure component phases
        for component in components:
            phases.append({
                'name': f'{component}_solid',
                'composition': {component: 1.0},
                'structure': 'crystal',
                'stability_range': {'temperature': temperature_range}
            })
            
            phases.append({
                'name': f'{component}_liquid',
                'composition': {component: 1.0},
                'structure': 'liquid',
                'stability_range': {'temperature': [800, temperature_range[1]]}
            })
        
        # Binary compounds (simplified)
        if len(components) == 2:
            phases.append({
                'name': f'{components[0]}{components[1]}',
                'composition': {components[0]: 0.5, components[1]: 0.5},
                'structure': 'intermetallic',
                'stability_range': {'temperature': [400, 1200]}
            })
        
        return phases
    
    def _calculate_phase_boundaries(self, phases: List[Dict[str, Any]], 
                                  system_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate phase boundaries."""
        boundaries = []
        
        # Simplified phase boundary calculation
        for i, phase1 in enumerate(phases):
            for j, phase2 in enumerate(phases[i+1:], i+1):
                # Simple temperature-based boundary
                boundary_temp = 600 + 100 * i + 50 * j
                
                boundaries.append({
                    'phase1': phase1['name'],
                    'phase2': phase2['name'],
                    'boundary_type': 'equilibrium',
                    'temperature': boundary_temp,
                    'composition_range': [0.0, 1.0]
                })
        
        return boundaries
    
    def _find_critical_points(self, phase_boundaries: List[Dict[str, Any]], 
                            system_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find critical points in phase diagram."""
        critical_points = []
        
        # Simplified critical point identification
        if len(phase_boundaries) > 2:
            critical_points.append({
                'type': 'eutectic',
                'temperature': 500,
                'composition': {'A': 0.3, 'B': 0.7},
                'phases_involved': ['A_solid', 'B_solid', 'liquid']
            })
        
        return critical_points
    
    def _analyze_phase_stability(self, phases: List[Dict[str, Any]], 
                               system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze phase stability."""
        stability_analysis = {
            'most_stable_phase': 'unknown',
            'metastable_phases': [],
            'phase_competition': {},
            'thermodynamic_stability': {}
        }
        
        # Simplified stability analysis
        if phases:
            # Assume first phase is most stable
            stability_analysis['most_stable_phase'] = phases[0]['name']
            
            # Others are metastable
            if len(phases) > 1:
                stability_analysis['metastable_phases'] = [p['name'] for p in phases[1:]]
        
        return stability_analysis
    
    def _calculate_thermodynamic_properties(self, phases: List[Dict[str, Any]], 
                                          system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate thermodynamic properties."""
        thermo_props = {
            'formation_energies': {},
            'heat_capacities': {},
            'thermal_expansion': {},
            'compressibilities': {}
        }
        
        # Simplified thermodynamic property calculation
        for phase in phases:
            phase_name = phase['name']
            
            # Estimated formation energy (eV/atom)
            thermo_props['formation_energies'][phase_name] = -0.1 * hash(phase_name) % 100 / 100
            
            # Estimated heat capacity (J/mol·K)
            thermo_props['heat_capacities'][phase_name] = 25.0 + 5.0 * hash(phase_name) % 10
            
            # Estimated thermal expansion coefficient (1/K)
            thermo_props['thermal_expansion'][phase_name] = 1e-5 + 1e-6 * hash(phase_name) % 10
        
        return thermo_props
    
    # Characterization methods
    
    def _select_characterization_techniques(self, material_info: Dict[str, Any], 
                                          objectives: List[str]) -> List[Dict[str, Any]]:
        """Select appropriate characterization techniques."""
        techniques = []
        
        # Structure characterization
        if any('structure' in obj.lower() for obj in objectives):
            techniques.append({
                'technique': 'xrd',
                'full_name': 'X-ray Diffraction',
                'information_provided': ['crystal_structure', 'phase_identification', 'lattice_parameters'],
                'sample_requirements': ['powder or single crystal'],
                'measurement_time': '1-2 hours'
            })
        
        # Electronic properties
        if any('electronic' in obj.lower() or 'electrical' in obj.lower() for obj in objectives):
            techniques.append({
                'technique': 'xps',
                'full_name': 'X-ray Photoelectron Spectroscopy',
                'information_provided': ['electronic_structure', 'chemical_composition', 'oxidation_states'],
                'sample_requirements': ['flat surface', 'UHV compatible'],
                'measurement_time': '2-4 hours'
            })
        
        # Mechanical properties
        if any('mechanical' in obj.lower() for obj in objectives):
            techniques.append({
                'technique': 'nanoindentation',
                'full_name': 'Nanoindentation',
                'information_provided': ['hardness', 'elastic_modulus', 'mechanical_response'],
                'sample_requirements': ['polished surface'],
                'measurement_time': '1-3 hours'
            })
        
        # Surface analysis
        if any('surface' in obj.lower() for obj in objectives):
            techniques.append({
                'technique': 'afm',
                'full_name': 'Atomic Force Microscopy',
                'information_provided': ['surface_topography', 'roughness', 'local_properties'],
                'sample_requirements': ['clean surface'],
                'measurement_time': '30 minutes - 2 hours'
            })
        
        return techniques
    
    def _design_characterization_protocols(self, techniques: List[Dict[str, Any]], 
                                         material_info: Dict[str, Any]) -> Dict[str, Any]:
        """Design experimental protocols for characterization."""
        protocols = {}
        
        for technique in techniques:
            technique_name = technique['technique']
            
            if technique_name == 'xrd':
                protocols[technique_name] = {
                    'sample_preparation': ['grinding to powder', 'mounting on sample holder'],
                    'measurement_parameters': {
                        'scan_range': '10-80 degrees 2theta',
                        'step_size': '0.02 degrees',
                        'counting_time': '1 second per step'
                    },
                    'data_analysis': ['phase_identification', 'lattice_parameter_refinement']
                }
            
            elif technique_name == 'xps':
                protocols[technique_name] = {
                    'sample_preparation': ['cleaning', 'mounting on sample stub'],
                    'measurement_parameters': {
                        'x_ray_source': 'Al Ka',
                        'pass_energy': '20 eV',
                        'energy_step': '0.1 eV'
                    },
                    'data_analysis': ['peak_identification', 'quantitative_analysis', 'chemical_state_analysis']
                }
            
            elif technique_name == 'nanoindentation':
                protocols[technique_name] = {
                    'sample_preparation': ['polishing to mirror finish', 'cleaning'],
                    'measurement_parameters': {
                        'indenter_type': 'Berkovich',
                        'maximum_load': '1 mN',
                        'loading_rate': '0.1 mN/s'
                    },
                    'data_analysis': ['hardness_calculation', 'modulus_calculation', 'load_displacement_analysis']
                }
            
            elif technique_name == 'afm':
                protocols[technique_name] = {
                    'sample_preparation': ['cleaning', 'mounting on sample disk'],
                    'measurement_parameters': {
                        'scan_mode': 'tapping mode',
                        'scan_size': '5x5 um',
                        'resolution': '512x512 pixels'
                    },
                    'data_analysis': ['topography_analysis', 'roughness_calculation', 'grain_size_analysis']
                }
        
        return protocols
    
    def _plan_sample_preparation(self, material_info: Dict[str, Any], 
                               techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan sample preparation procedures."""
        sample_prep = {
            'general_preparation': [],
            'technique_specific': {},
            'safety_considerations': [],
            'estimated_time': {}
        }
        
        # General preparation steps
        sample_prep['general_preparation'] = [
            'receive_and_catalog_sample',
            'initial_visual_inspection',
            'photograph_sample',
            'measure_dimensions'
        ]
        
        # Technique-specific preparation
        for technique in techniques:
            technique_name = technique['technique']
            
            if technique_name in ['xrd', 'xps']:
                sample_prep['technique_specific'][technique_name] = [
                    'cut_appropriate_size',
                    'clean_surface',
                    'mount_on_holder'
                ]
                sample_prep['estimated_time'][technique_name] = '30-60 minutes'
            
            elif technique_name in ['nanoindentation', 'afm']:
                sample_prep['technique_specific'][technique_name] = [
                    'mount_in_resin',
                    'polish_to_mirror_finish',
                    'clean_with_solvents',
                    'dry_in_clean_environment'
                ]
                sample_prep['estimated_time'][technique_name] = '2-4 hours'
        
        # Safety considerations
        sample_prep['safety_considerations'] = [
            'wear_appropriate_PPE',
            'work_in_ventilated_area',
            'follow_chemical_safety_protocols',
            'dispose_of_waste_properly'
        ]
        
        return sample_prep
    
    def _sequence_measurements(self, techniques: List[Dict[str, Any]], 
                             protocols: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sequence measurements for optimal workflow."""
        # Sort techniques by destructiveness and sample requirements
        non_destructive = ['xrd', 'xps', 'raman', 'ftir']
        surface_sensitive = ['xps', 'afm', 'stm']
        mechanical_testing = ['nanoindentation', 'tensile_testing']
        
        sequence = []
        
        # First: Non-destructive bulk characterization
        for technique in techniques:
            if technique['technique'] in non_destructive and technique['technique'] not in surface_sensitive:
                sequence.append({
                    'order': len(sequence) + 1,
                    'technique': technique['technique'],
                    'justification': 'Non-destructive bulk characterization first'
                })
        
        # Second: Surface-sensitive techniques
        for technique in techniques:
            if technique['technique'] in surface_sensitive:
                sequence.append({
                    'order': len(sequence) + 1,
                    'technique': technique['technique'],
                    'justification': 'Surface analysis before mechanical testing'
                })
        
        # Third: Mechanical testing (potentially destructive)
        for technique in techniques:
            if technique['technique'] in mechanical_testing:
                sequence.append({
                    'order': len(sequence) + 1,
                    'technique': technique['technique'],
                    'justification': 'Mechanical testing last due to potential sample damage'
                })
        
        return sequence
    
    def _predict_expected_information(self, techniques: List[Dict[str, Any]], 
                                    material_info: Dict[str, Any]) -> Dict[str, Any]:
        """Predict expected information from characterization."""
        expected_info = {
            'structural_information': [],
            'electronic_properties': [],
            'mechanical_properties': [],
            'surface_properties': [],
            'compositional_information': []
        }
        
        for technique in techniques:
            info_provided = technique.get('information_provided', [])
            
            for info in info_provided:
                if 'structure' in info or 'crystal' in info:
                    expected_info['structural_information'].append(f'{info} from {technique["technique"]}')
                elif 'electronic' in info or 'band' in info:
                    expected_info['electronic_properties'].append(f'{info} from {technique["technique"]}')
                elif 'mechanical' in info or 'hardness' in info or 'modulus' in info:
                    expected_info['mechanical_properties'].append(f'{info} from {technique["technique"]}')
                elif 'surface' in info or 'topography' in info:
                    expected_info['surface_properties'].append(f'{info} from {technique["technique"]}')
                elif 'composition' in info or 'chemical' in info:
                    expected_info['compositional_information'].append(f'{info} from {technique["technique"]}')
        
        return expected_info
    
    def _estimate_characterization_resources(self, techniques: List[Dict[str, Any]], 
                                           protocols: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resources required for characterization."""
        resources = {
            'equipment_access': [],
            'estimated_cost': 0.0,
            'total_time': 0.0,
            'personnel_requirements': [],
            'consumables': []
        }
        
        # Equipment and cost estimation
        technique_costs = {
            'xrd': 100, 'xps': 300, 'sem': 150, 'tem': 400,
            'afm': 200, 'nanoindentation': 250, 'raman': 120
        }
        
        for technique in techniques:
            technique_name = technique['technique']
            
            # Equipment access
            resources['equipment_access'].append(f'{technique["full_name"]} system')
            
            # Cost estimation
            cost = technique_costs.get(technique_name, 150)
            resources['estimated_cost'] += cost
            
            # Time estimation
            time_str = technique.get('measurement_time', '2 hours')
            # Extract hours (simplified)
            if 'hour' in time_str:
                hours = float(time_str.split('-')[0]) if '-' in time_str else 2.0
                resources['total_time'] += hours
        
        # Personnel requirements
        resources['personnel_requirements'] = [
            'trained_instrument_operator',
            'materials_characterization_specialist',
            'data_analysis_expert'
        ]
        
        # Consumables
        resources['consumables'] = [
            'sample_preparation_materials',
            'mounting_media',
            'polishing_supplies',
            'cleaning_solvents'
        ]
        
        return resources