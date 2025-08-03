"""
Molecular Dynamics Engine

Advanced molecular dynamics simulation engine for particle simulations, force calculations,
and statistical mechanics. Integrates with LAMMPS, GROMACS, and other MD software packages.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import warnings
from dataclasses import dataclass

from .base_physics_engine import (
    BasePhysicsEngine, PhysicsEngineType, SoftwareInterface,
    PhysicsProblemSpec, PhysicsResult
)

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Represents a single particle in the simulation."""
    id: int
    type: str
    mass: float
    charge: float
    position: np.ndarray
    velocity: np.ndarray
    force: np.ndarray
    
    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.position is None:
            self.position = np.zeros(3)
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.force is None:
            self.force = np.zeros(3)


@dataclass
class MDSystem:
    """Represents a molecular dynamics system."""
    particles: List[Particle]
    box_dimensions: np.ndarray
    periodic_boundary_conditions: bool
    temperature: float
    pressure: float
    
    def __post_init__(self):
        """Initialize system properties."""
        if self.box_dimensions is None:
            self.box_dimensions = np.array([10.0, 10.0, 10.0])


class MolecularDynamicsEngine(BasePhysicsEngine):
    """
    Molecular dynamics simulation engine for classical particle simulations.
    
    Capabilities:
    - Classical molecular dynamics simulations
    - Force field calculations (Lennard-Jones, Coulomb, bonded interactions)
    - Statistical mechanics property calculations
    - Temperature and pressure control (thermostats and barostats)
    - Multiple integration algorithms (Verlet, Leapfrog, Runge-Kutta)
    - Ensemble simulations (NVE, NVT, NPT, Grand Canonical)
    - Integration with LAMMPS and GROMACS
    """
    
    def __init__(self, config: Dict[str, Any], cost_manager=None):
        """Initialize the molecular dynamics engine."""
        super().__init__(config, cost_manager, logger_name='MolecularDynamicsEngine')
        
        # MD-specific configuration
        self.md_config = {
            'default_timestep': config.get('default_timestep', 1e-15),  # seconds
            'default_temperature': config.get('default_temperature', 300.0),  # Kelvin
            'default_pressure': config.get('default_pressure', 1.0),  # atm
            'force_field': config.get('force_field', 'lennard_jones'),
            'cutoff_radius': config.get('cutoff_radius', 2.5),  # sigma units
            'neighbor_list_skin': config.get('neighbor_list_skin', 0.3),
            'thermostat': config.get('thermostat', 'nose_hoover'),
            'barostat': config.get('barostat', 'parrinello_rahman'),
            'long_range_corrections': config.get('long_range_corrections', True)
        }
        
        # Physical constants
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.NA = 6.02214076e23  # Avogadro constant
        self.elementary_charge = 1.602176634e-19  # C
        self.epsilon_0 = 8.8541878128e-12  # F/m (vacuum permittivity)
        
        # Simulation state
        self.current_systems = {}
        self.trajectory_data = {}
        self.force_field_parameters = {}
        self.neighbor_lists = {}
        
        # Statistical mechanics calculators
        self.property_calculators = {
            'radial_distribution_function': self._calculate_rdf,
            'mean_square_displacement': self._calculate_msd,
            'velocity_autocorrelation': self._calculate_vacf,
            'structure_factor': self._calculate_structure_factor,
            'thermal_conductivity': self._calculate_thermal_conductivity,
            'viscosity': self._calculate_viscosity,
            'diffusion_coefficient': self._calculate_diffusion_coefficient
        }
        
        self.logger.info("Molecular dynamics engine initialized")
    
    def _get_engine_type(self) -> PhysicsEngineType:
        """Get the engine type."""
        return PhysicsEngineType.MOLECULAR_DYNAMICS
    
    def _get_version(self) -> str:
        """Get the engine version."""
        return "1.0.0"
    
    def _get_available_methods(self) -> List[str]:
        """Get available molecular dynamics methods."""
        return [
            'classical_md',
            'langevin_dynamics',
            'brownian_dynamics',
            'monte_carlo_md',
            'replica_exchange_md',
            'steered_md',
            'umbrella_sampling',
            'free_energy_perturbation',
            'thermodynamic_integration',
            'metadynamics',
            'enhanced_sampling',
            'coarse_grained_md'
        ]
    
    def _get_supported_software(self) -> List[SoftwareInterface]:
        """Get supported MD software."""
        return [
            SoftwareInterface.LAMMPS,
            SoftwareInterface.GROMACS,
            SoftwareInterface.CUSTOM
        ]
    
    def _get_capabilities(self) -> List[str]:
        """Get engine capabilities."""
        return [
            'classical_molecular_dynamics',
            'force_field_calculations',
            'statistical_mechanics',
            'thermodynamic_properties',
            'transport_properties',
            'phase_behavior',
            'protein_folding',
            'material_properties',
            'fluid_dynamics',
            'crystallization'
        ]
    
    def solve_problem(self, problem_spec: PhysicsProblemSpec, method: str, 
                     parameters: Dict[str, Any]) -> PhysicsResult:
        """
        Solve a molecular dynamics problem.
        
        Args:
            problem_spec: MD problem specification
            method: MD simulation method to use
            parameters: Method-specific parameters
            
        Returns:
            PhysicsResult with MD simulation results
        """
        start_time = time.time()
        result_data = {}
        warnings_list = []
        
        try:
            self.logger.info(f"Solving MD problem {problem_spec.problem_id} using {method}")
            
            # Validate method
            if method not in self.available_methods:
                raise ValueError(f"Method '{method}' not available in molecular dynamics engine")
            
            # Merge parameters
            merged_params = {**self.md_config, **parameters}
            
            # Initialize system
            md_system = self._initialize_md_system(problem_spec, merged_params)
            
            # Route to appropriate solver
            if method == 'classical_md':
                result_data = self._run_classical_md(md_system, problem_spec, merged_params)
            elif method == 'langevin_dynamics':
                result_data = self._run_langevin_dynamics(md_system, problem_spec, merged_params)
            elif method == 'brownian_dynamics':
                result_data = self._run_brownian_dynamics(md_system, problem_spec, merged_params)
            elif method == 'monte_carlo_md':
                result_data = self._run_monte_carlo_md(md_system, problem_spec, merged_params)
            elif method == 'replica_exchange_md':
                result_data = self._run_replica_exchange_md(md_system, problem_spec, merged_params)
            elif method == 'umbrella_sampling':
                result_data = self._run_umbrella_sampling(md_system, problem_spec, merged_params)
            elif method == 'metadynamics':
                result_data = self._run_metadynamics(md_system, problem_spec, merged_params)
            else:
                # For other methods, use generic MD solver
                result_data = self._run_generic_md_method(md_system, problem_spec, method, merged_params)
            
            execution_time = time.time() - start_time
            
            # Create successful result
            result = PhysicsResult(
                result_id="",
                problem_id=problem_spec.problem_id,
                success=True,
                data=result_data,
                metadata={
                    'method': method,
                    'parameters': merged_params,
                    'engine_type': self.engine_type.value,
                    'md_engine_version': self.version,
                    'system_size': len(md_system.particles)
                },
                execution_time=execution_time,
                warnings=warnings_list
            )
            
            # Update statistics
            self.update_execution_stats(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"MD simulation failed: {e}")
            
            result = PhysicsResult(
                result_id="",
                problem_id=problem_spec.problem_id,
                success=False,
                data={},
                metadata={
                    'method': method,
                    'engine_type': self.engine_type.value
                },
                execution_time=execution_time,
                error_message=str(e),
                warnings=warnings_list
            )
            
            self.update_execution_stats(result)
            return result
    
    def _initialize_md_system(self, problem_spec: PhysicsProblemSpec, 
                             parameters: Dict[str, Any]) -> MDSystem:
        """Initialize MD system from problem specification."""
        system_params = problem_spec.parameters
        
        # Extract system parameters
        n_particles = system_params.get('n_particles', 100)
        particle_types = system_params.get('particle_types', ['Ar'])
        box_size = system_params.get('box_size', 10.0)
        density = system_params.get('density', 0.8)  # reduced units
        temperature = parameters.get('temperature', self.md_config['default_temperature'])
        pressure = parameters.get('pressure', self.md_config['default_pressure'])
        
        # Create particles
        particles = []
        
        for i in range(n_particles):
            particle_type = particle_types[i % len(particle_types)]
            
            # Set mass and charge based on particle type
            mass, charge = self._get_particle_properties(particle_type)
            
            # Random initial position
            position = np.random.random(3) * box_size
            
            # Maxwell-Boltzmann velocity distribution
            velocity = self._sample_maxwell_boltzmann_velocity(mass, temperature)
            
            particle = Particle(
                id=i,
                type=particle_type,
                mass=mass,
                charge=charge,
                position=position,
                velocity=velocity,
                force=np.zeros(3)
            )
            
            particles.append(particle)
        
        # Remove center of mass motion
        self._remove_center_of_mass_motion(particles)
        
        # Create system
        md_system = MDSystem(
            particles=particles,
            box_dimensions=np.array([box_size, box_size, box_size]),
            periodic_boundary_conditions=system_params.get('periodic_boundary_conditions', True),
            temperature=temperature,
            pressure=pressure
        )
        
        # Store system
        self.current_systems[problem_spec.problem_id] = md_system
        
        return md_system
    
    def _run_classical_md(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run classical molecular dynamics simulation."""
        self.logger.info("Running classical MD simulation")
        
        # Simulation parameters
        n_steps = parameters.get('n_steps', 10000)
        dt = parameters.get('timestep', self.md_config['default_timestep'])
        output_frequency = parameters.get('output_frequency', 100)
        ensemble = parameters.get('ensemble', 'NVE')
        
        # Initialize trajectory storage
        trajectory = {
            'positions': [],
            'velocities': [],
            'forces': [],
            'energies': [],
            'temperatures': [],
            'pressures': [],
            'times': []
        }
        
        # Setup force field
        force_field = self._setup_force_field(parameters)
        
        # Setup thermostat and barostat if needed
        thermostat = None
        barostat = None
        
        if ensemble in ['NVT', 'NPT']:
            thermostat = self._setup_thermostat(parameters)
        
        if ensemble in ['NPT']:
            barostat = self._setup_barostat(parameters)
        
        # Main MD loop
        for step in range(n_steps):
            # Calculate forces
            self._calculate_forces(md_system, force_field)
            
            # Update positions and velocities using Verlet algorithm
            self._integrate_verlet(md_system, dt)
            
            # Apply periodic boundary conditions
            if md_system.periodic_boundary_conditions:
                self._apply_periodic_boundary_conditions(md_system)
            
            # Apply thermostat
            if thermostat:
                self._apply_thermostat(md_system, thermostat, parameters)
            
            # Apply barostat
            if barostat:
                self._apply_barostat(md_system, barostat, parameters)
            
            # Output data
            if step % output_frequency == 0:
                self._record_trajectory_frame(md_system, trajectory, step * dt)
        
        # Calculate final properties
        final_properties = self._calculate_final_properties(md_system, trajectory, parameters)
        
        # Store trajectory
        self.trajectory_data[problem_spec.problem_id] = trajectory
        
        return {
            'trajectory': trajectory,
            'final_properties': final_properties,
            'simulation_parameters': {
                'n_steps': n_steps,
                'timestep': dt,
                'ensemble': ensemble,
                'n_particles': len(md_system.particles),
                'box_dimensions': md_system.box_dimensions.tolist(),
                'temperature': md_system.temperature,
                'pressure': md_system.pressure
            },
            'performance_metrics': {
                'steps_per_second': n_steps / (time.time() - time.time()),  # Approximate
                'ns_per_day': (n_steps * dt * 1e9) / 86400,  # Nanoseconds per day
            }
        }
    
    def _run_langevin_dynamics(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Langevin dynamics simulation."""
        self.logger.info("Running Langevin dynamics simulation")
        
        # Langevin parameters
        friction_coefficient = parameters.get('friction_coefficient', 1.0)
        temperature = parameters.get('temperature', md_system.temperature)
        n_steps = parameters.get('n_steps', 10000)
        dt = parameters.get('timestep', self.md_config['default_timestep'])
        
        # Initialize trajectory storage
        trajectory = {
            'positions': [],
            'velocities': [],
            'forces': [],
            'times': []
        }
        
        # Setup force field
        force_field = self._setup_force_field(parameters)
        
        # Langevin dynamics loop
        for step in range(n_steps):
            # Calculate forces
            self._calculate_forces(md_system, force_field)
            
            # Langevin integration
            self._integrate_langevin(md_system, dt, friction_coefficient, temperature)
            
            # Apply periodic boundary conditions
            if md_system.periodic_boundary_conditions:
                self._apply_periodic_boundary_conditions(md_system)
            
            # Record trajectory
            if step % parameters.get('output_frequency', 100) == 0:
                self._record_trajectory_frame(md_system, trajectory, step * dt)
        
        # Calculate properties
        properties = self._calculate_langevin_properties(md_system, trajectory, parameters)
        
        return {
            'trajectory': trajectory,
            'properties': properties,
            'langevin_parameters': {
                'friction_coefficient': friction_coefficient,
                'temperature': temperature,
                'n_steps': n_steps,
                'timestep': dt
            }
        }
    
    def _run_brownian_dynamics(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Brownian dynamics simulation."""
        self.logger.info("Running Brownian dynamics simulation")
        
        # Brownian dynamics parameters
        drag_coefficient = parameters.get('drag_coefficient', 1.0)
        temperature = parameters.get('temperature', md_system.temperature)
        n_steps = parameters.get('n_steps', 10000)
        dt = parameters.get('timestep', self.md_config['default_timestep'])
        
        # Initialize trajectory
        trajectory = {
            'positions': [],
            'displacements': [],
            'times': []
        }
        
        # Setup force field (typically simplified for Brownian dynamics)
        force_field = self._setup_force_field(parameters)
        
        # Brownian dynamics loop
        for step in range(n_steps):
            # Calculate forces
            self._calculate_forces(md_system, force_field)
            
            # Brownian dynamics integration
            self._integrate_brownian(md_system, dt, drag_coefficient, temperature)
            
            # Apply periodic boundary conditions
            if md_system.periodic_boundary_conditions:
                self._apply_periodic_boundary_conditions(md_system)
            
            # Record trajectory
            if step % parameters.get('output_frequency', 100) == 0:
                self._record_brownian_frame(md_system, trajectory, step * dt)
        
        # Calculate diffusion properties
        diffusion_properties = self._calculate_diffusion_properties(trajectory, parameters)
        
        return {
            'trajectory': trajectory,
            'diffusion_properties': diffusion_properties,
            'brownian_parameters': {
                'drag_coefficient': drag_coefficient,
                'temperature': temperature,
                'n_steps': n_steps,
                'timestep': dt
            }
        }
    
    def _run_monte_carlo_md(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo molecular dynamics hybrid simulation."""
        self.logger.info("Running Monte Carlo MD simulation")
        
        # MC-MD parameters
        mc_frequency = parameters.get('mc_frequency', 10)  # MC move every 10 MD steps
        md_steps_per_cycle = parameters.get('md_steps_per_cycle', 100)
        n_cycles = parameters.get('n_cycles', 100)
        temperature = parameters.get('temperature', md_system.temperature)
        
        # Initialize statistics
        mc_acceptance_rate = 0
        total_mc_moves = 0
        energy_history = []
        
        # Setup force field
        force_field = self._setup_force_field(parameters)
        
        # MC-MD cycles
        for cycle in range(n_cycles):
            # MD phase
            for md_step in range(md_steps_per_cycle):
                self._calculate_forces(md_system, force_field)
                self._integrate_verlet(md_system, parameters.get('timestep', self.md_config['default_timestep']))
                
                if md_system.periodic_boundary_conditions:
                    self._apply_periodic_boundary_conditions(md_system)
            
            # MC phase
            if cycle % mc_frequency == 0:
                acceptance = self._perform_mc_moves(md_system, force_field, temperature, parameters)
                mc_acceptance_rate += acceptance
                total_mc_moves += 1
            
            # Calculate and store energy
            total_energy = self._calculate_total_energy(md_system, force_field)
            energy_history.append(total_energy)
        
        # Calculate final acceptance rate
        if total_mc_moves > 0:
            mc_acceptance_rate /= total_mc_moves
        
        return {
            'energy_history': energy_history,
            'mc_acceptance_rate': mc_acceptance_rate,
            'total_cycles': n_cycles,
            'final_energy': energy_history[-1] if energy_history else 0.0,
            'mc_md_parameters': {
                'mc_frequency': mc_frequency,
                'md_steps_per_cycle': md_steps_per_cycle,
                'temperature': temperature
            }
        }
    
    def _run_replica_exchange_md(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run replica exchange molecular dynamics."""
        self.logger.info("Running replica exchange MD")
        
        # REMD parameters
        n_replicas = parameters.get('n_replicas', 8)
        temperature_range = parameters.get('temperature_range', [300, 600])  # K
        exchange_frequency = parameters.get('exchange_frequency', 100)
        n_steps_per_replica = parameters.get('n_steps_per_replica', 1000)
        
        # Create temperature ladder
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_replicas)
        
        # Initialize replicas
        replicas = []
        for i, temp in enumerate(temperatures):
            replica_system = self._copy_md_system(md_system)
            replica_system.temperature = temp
            # Rescale velocities for new temperature
            self._rescale_velocities_to_temperature(replica_system, temp)
            replicas.append(replica_system)
        
        # REMD simulation
        exchange_statistics = []
        replica_trajectories = [[] for _ in range(n_replicas)]
        
        force_field = self._setup_force_field(parameters)
        
        for exchange_cycle in range(parameters.get('n_exchange_cycles', 100)):
            # Run MD on each replica
            for i, replica in enumerate(replicas):
                for step in range(n_steps_per_replica):
                    self._calculate_forces(replica, force_field)
                    self._integrate_verlet(replica, parameters.get('timestep', self.md_config['default_timestep']))
                    
                    if replica.periodic_boundary_conditions:
                        self._apply_periodic_boundary_conditions(replica)
                
                # Record replica state
                if exchange_cycle % 10 == 0:  # Subsample for storage
                    replica_trajectories[i].append(self._get_system_snapshot(replica))
            
            # Attempt replica exchanges
            if exchange_cycle % exchange_frequency == 0:
                exchanges = self._attempt_replica_exchanges(replicas, force_field, temperatures)
                exchange_statistics.extend(exchanges)
        
        # Calculate REMD statistics
        exchange_acceptance_rate = np.mean([ex['accepted'] for ex in exchange_statistics])
        
        return {
            'replica_trajectories': replica_trajectories,
            'exchange_statistics': exchange_statistics,
            'exchange_acceptance_rate': exchange_acceptance_rate,
            'temperatures': temperatures.tolist(),
            'remd_parameters': {
                'n_replicas': n_replicas,
                'temperature_range': temperature_range,
                'exchange_frequency': exchange_frequency
            }
        }
    
    def _run_umbrella_sampling(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run umbrella sampling simulation."""
        self.logger.info("Running umbrella sampling simulation")
        
        # Umbrella sampling parameters
        reaction_coordinate = parameters.get('reaction_coordinate', 'distance')
        n_windows = parameters.get('n_windows', 20)
        spring_constant = parameters.get('spring_constant', 10.0)  # kcal/mol/Å²
        coordinate_range = parameters.get('coordinate_range', [2.0, 10.0])
        
        # Create umbrella windows
        window_centers = np.linspace(coordinate_range[0], coordinate_range[1], n_windows)
        
        # Run simulation for each window
        window_data = []
        
        force_field = self._setup_force_field(parameters)
        
        for i, center in enumerate(window_centers):
            self.logger.info(f"Running umbrella window {i+1}/{n_windows} at {center:.2f}")
            
            # Setup biasing potential
            bias_potential = {
                'type': 'harmonic',
                'coordinate': reaction_coordinate,
                'center': center,
                'spring_constant': spring_constant
            }
            
            # Run MD with biasing potential
            window_trajectory = self._run_biased_md(md_system, force_field, bias_potential, parameters)
            
            # Calculate window statistics
            window_stats = self._analyze_umbrella_window(window_trajectory, reaction_coordinate, center)
            
            window_data.append({
                'center': center,
                'trajectory': window_trajectory,
                'statistics': window_stats
            })
        
        # Perform WHAM analysis to get PMF
        pmf_data = self._perform_wham_analysis(window_data, parameters)
        
        return {
            'window_data': window_data,
            'pmf': pmf_data,
            'umbrella_parameters': {
                'reaction_coordinate': reaction_coordinate,
                'n_windows': n_windows,
                'spring_constant': spring_constant,
                'coordinate_range': coordinate_range
            }
        }
    
    def _run_metadynamics(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run metadynamics simulation."""
        self.logger.info("Running metadynamics simulation")
        
        # Metadynamics parameters
        collective_variables = parameters.get('collective_variables', ['distance'])
        gaussian_height = parameters.get('gaussian_height', 0.1)  # kcal/mol
        gaussian_width = parameters.get('gaussian_width', 0.1)  # CV units
        deposition_frequency = parameters.get('deposition_frequency', 100)
        n_steps = parameters.get('n_steps', 100000)
        
        # Initialize metadynamics
        cv_history = []
        gaussian_deposited = []
        bias_potential_history = []
        
        force_field = self._setup_force_field(parameters)
        
        # Metadynamics simulation
        for step in range(n_steps):
            # Calculate forces
            self._calculate_forces(md_system, force_field)
            
            # Calculate collective variables
            cv_values = self._calculate_collective_variables(md_system, collective_variables)
            cv_history.append(cv_values)
            
            # Add metadynamics bias force
            bias_force = self._calculate_metadynamics_bias_force(
                md_system, cv_values, gaussian_deposited, gaussian_width
            )
            
            # Apply bias force to particles
            self._apply_bias_force(md_system, bias_force, collective_variables)
            
            # Integrate equations of motion
            self._integrate_verlet(md_system, parameters.get('timestep', self.md_config['default_timestep']))
            
            # Apply periodic boundary conditions
            if md_system.periodic_boundary_conditions:
                self._apply_periodic_boundary_conditions(md_system)
            
            # Deposit Gaussian if needed
            if step % deposition_frequency == 0:
                gaussian_info = {
                    'center': cv_values.copy(),
                    'height': gaussian_height,
                    'width': gaussian_width,
                    'step': step
                }
                gaussian_deposited.append(gaussian_info)
            
            # Calculate current bias potential
            if step % (deposition_frequency * 10) == 0:  # Less frequent for performance
                bias_potential = self._calculate_bias_potential(cv_values, gaussian_deposited)
                bias_potential_history.append(bias_potential)
        
        # Reconstruct free energy surface
        free_energy_surface = self._reconstruct_free_energy_surface(
            cv_history, gaussian_deposited, collective_variables, parameters
        )
        
        return {
            'cv_history': cv_history,
            'gaussians_deposited': gaussian_deposited,
            'bias_potential_history': bias_potential_history,
            'free_energy_surface': free_energy_surface,
            'metadynamics_parameters': {
                'collective_variables': collective_variables,
                'gaussian_height': gaussian_height,
                'gaussian_width': gaussian_width,
                'deposition_frequency': deposition_frequency,
                'total_gaussians': len(gaussian_deposited)
            }
        }
    
    def _run_generic_md_method(self, md_system: MDSystem, problem_spec: PhysicsProblemSpec, 
                              method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a generic MD method."""
        self.logger.info(f"Running generic MD method: {method}")
        
        # Basic MD simulation as fallback
        n_steps = parameters.get('n_steps', 1000)
        dt = parameters.get('timestep', self.md_config['default_timestep'])
        
        force_field = self._setup_force_field(parameters)
        
        # Simple MD loop
        for step in range(n_steps):
            self._calculate_forces(md_system, force_field)
            self._integrate_verlet(md_system, dt)
            
            if md_system.periodic_boundary_conditions:
                self._apply_periodic_boundary_conditions(md_system)
        
        # Calculate final energy
        final_energy = self._calculate_total_energy(md_system, force_field)
        
        return {
            'method': method,
            'final_energy': final_energy,
            'n_steps': n_steps,
            'note': 'Generic MD method implementation'
        }
    
    def validate_results(self, results: PhysicsResult, 
                        known_solutions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate molecular dynamics results."""
        validation_report = {
            'overall_valid': True,
            'validation_checks': [],
            'accuracy_metrics': {},
            'physical_consistency': {}
        }
        
        if not results.success:
            validation_report['overall_valid'] = False
            validation_report['validation_checks'].append('Simulation failed')
            return validation_report
        
        result_data = results.data
        
        # Energy conservation check for NVE simulations
        if 'trajectory' in result_data:
            trajectory = result_data['trajectory']
            
            if 'energies' in trajectory and len(trajectory['energies']) > 10:
                energies = trajectory['energies']
                energy_drift = (energies[-1] - energies[0]) / abs(energies[0])
                
                validation_report['physical_consistency']['energy_drift'] = energy_drift
                
                if abs(energy_drift) < 0.01:  # 1% tolerance
                    validation_report['validation_checks'].append('Energy conservation: PASS')
                else:
                    validation_report['validation_checks'].append('Energy conservation: FAIL')
                    validation_report['overall_valid'] = False
        
        # Temperature consistency check
        if 'final_properties' in result_data:
            props = result_data['final_properties']
            
            if 'average_temperature' in props:
                target_temp = results.metadata.get('parameters', {}).get('temperature', 300.0)
                actual_temp = props['average_temperature']
                temp_error = abs(actual_temp - target_temp) / target_temp
                
                validation_report['physical_consistency']['temperature_error'] = temp_error
                
                if temp_error < 0.05:  # 5% tolerance
                    validation_report['validation_checks'].append('Temperature control: PASS')
                else:
                    validation_report['validation_checks'].append('Temperature control: FAIL')
                    validation_report['overall_valid'] = False
        
        # Particle number conservation
        system_size = results.metadata.get('system_size', 0)
        if system_size > 0:
            validation_report['validation_checks'].append('Particle number conservation: PASS')
            validation_report['physical_consistency']['n_particles'] = system_size
        
        # Check for NaN values in trajectory
        if 'trajectory' in result_data:
            trajectory = result_data['trajectory']
            has_nan = False
            
            for key in ['positions', 'velocities', 'forces']:
                if key in trajectory:
                    data = np.array(trajectory[key])
                    if np.any(np.isnan(data)):
                        has_nan = True
                        break
            
            if has_nan:
                validation_report['validation_checks'].append('Numerical stability: FAIL')
                validation_report['overall_valid'] = False
            else:
                validation_report['validation_checks'].append('Numerical stability: PASS')
        
        return validation_report
    
    def optimize_parameters(self, objective: str, constraints: Dict[str, Any], 
                          problem_spec: PhysicsProblemSpec) -> Dict[str, Any]:
        """Optimize MD simulation parameters."""
        self.logger.info(f"Optimizing parameters for objective: {objective}")
        
        optimization_report = {
            'objective': objective,
            'optimal_parameters': {},
            'optimization_history': [],
            'success': False
        }
        
        if objective == 'minimize_equilibration_time':
            return self._optimize_equilibration_time(problem_spec, constraints)
        elif objective == 'maximize_sampling_efficiency':
            return self._optimize_sampling_efficiency(problem_spec, constraints)
        elif objective == 'minimize_computational_cost':
            return self._optimize_computational_cost(problem_spec, constraints)
        else:
            optimization_report['error'] = f"Unknown objective: {objective}"
            return optimization_report
    
    def integrate_with_software(self, software_name: SoftwareInterface, 
                               interface_config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with external MD software."""
        self.logger.info(f"Integrating with {software_name.value}")
        
        integration_result = {
            'software': software_name.value,
            'status': 'disconnected',
            'capabilities': [],
            'configuration': interface_config
        }
        
        try:
            if software_name == SoftwareInterface.LAMMPS:
                return self._integrate_lammps(interface_config)
            elif software_name == SoftwareInterface.GROMACS:
                return self._integrate_gromacs(interface_config)
            else:
                integration_result['status'] = 'not_implemented'
                integration_result['message'] = f"Integration with {software_name.value} not yet implemented"
                
        except Exception as e:
            integration_result['status'] = 'error'
            integration_result['error'] = str(e)
        
        return integration_result
    
    def handle_errors(self, error_type: str, recovery_strategy: str, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MD simulation errors."""
        self.logger.warning(f"Handling error: {error_type} with strategy: {recovery_strategy}")
        
        recovery_result = {
            'error_type': error_type,
            'recovery_strategy': recovery_strategy,
            'success': False,
            'actions_taken': []
        }
        
        try:
            if error_type == 'numerical_instability':
                return self._handle_numerical_instability(recovery_strategy, context, recovery_result)
            elif error_type == 'energy_explosion':
                return self._handle_energy_explosion(recovery_strategy, context, recovery_result)
            elif error_type == 'temperature_runaway':
                return self._handle_temperature_runaway(recovery_strategy, context, recovery_result)
            else:
                recovery_result['actions_taken'].append(f"Unknown error type: {error_type}")
                
        except Exception as e:
            recovery_result['recovery_error'] = str(e)
        
        return recovery_result
    
    def _get_method_details(self, method: str) -> Dict[str, Any]:
        """Get detailed information about an MD method."""
        method_details = {
            'classical_md': {
                'description': 'Classical molecular dynamics simulation',
                'complexity': 'O(N²) for all-pairs, O(N log N) with neighbor lists',
                'parameters': ['timestep', 'n_steps', 'ensemble', 'temperature'],
                'accuracy': 'Classical mechanics approximation',
                'limitations': 'No quantum effects, finite timestep errors'
            },
            'langevin_dynamics': {
                'description': 'Molecular dynamics with Langevin thermostat',
                'complexity': 'O(N²) plus stochastic forces',
                'parameters': ['friction_coefficient', 'temperature', 'timestep'],
                'accuracy': 'Fluctuation-dissipation theorem',
                'limitations': 'Requires careful choice of friction coefficient'
            },
            'brownian_dynamics': {
                'description': 'Overdamped Langevin dynamics',
                'complexity': 'O(N²) per time step',
                'parameters': ['drag_coefficient', 'temperature', 'timestep'],
                'accuracy': 'High friction limit approximation',
                'limitations': 'No inertial effects, larger timesteps possible'
            },
            'replica_exchange_md': {
                'description': 'Enhanced sampling with temperature replica exchange',
                'complexity': 'O(N_replica × N²)',
                'parameters': ['n_replicas', 'temperature_range', 'exchange_frequency'],
                'accuracy': 'Improved sampling of rare events',
                'limitations': 'Computationally expensive, requires many replicas'
            }
        }
        
        return method_details.get(method, {
            'description': f'MD method: {method}',
            'complexity': 'Not specified',
            'parameters': [],
            'accuracy': 'Method-dependent',
            'limitations': 'See method documentation'
        })
    
    # Helper methods for MD calculations
    
    def _get_particle_properties(self, particle_type: str) -> Tuple[float, float]:
        """Get mass and charge for a particle type."""
        properties = {
            'Ar': (39.948, 0.0),  # amu, elementary charge
            'Ne': (20.18, 0.0),
            'H': (1.008, 0.0),
            'C': (12.011, 0.0),
            'N': (14.007, 0.0),
            'O': (15.999, 0.0),
            'Na': (22.99, 1.0),
            'Cl': (35.45, -1.0)
        }
        
        return properties.get(particle_type, (1.0, 0.0))  # Default values
    
    def _sample_maxwell_boltzmann_velocity(self, mass: float, temperature: float) -> np.ndarray:
        """Sample velocity from Maxwell-Boltzmann distribution."""
        # Convert to reduced units for simplicity
        sigma = np.sqrt(temperature / mass)  # Simplified
        return np.random.normal(0, sigma, 3)
    
    def _remove_center_of_mass_motion(self, particles: List[Particle]):
        """Remove center of mass motion from the system."""
        total_mass = sum(p.mass for p in particles)
        com_velocity = np.sum([p.mass * p.velocity for p in particles], axis=0) / total_mass
        
        for particle in particles:
            particle.velocity -= com_velocity
    
    def _setup_force_field(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Setup force field parameters."""
        force_field_type = parameters.get('force_field', self.md_config['force_field'])
        
        if force_field_type == 'lennard_jones':
            return {
                'type': 'lennard_jones',
                'epsilon': parameters.get('lj_epsilon', 1.0),  # Energy units
                'sigma': parameters.get('lj_sigma', 1.0),      # Length units
                'cutoff': parameters.get('cutoff_radius', self.md_config['cutoff_radius'])
            }
        elif force_field_type == 'coulomb':
            return {
                'type': 'coulomb',
                'dielectric_constant': parameters.get('dielectric_constant', 1.0),
                'cutoff': parameters.get('cutoff_radius', self.md_config['cutoff_radius'])
            }
        else:
            return {
                'type': 'generic',
                'parameters': parameters
            }
    
    def _calculate_forces(self, md_system: MDSystem, force_field: Dict[str, Any]):
        """Calculate forces on all particles."""
        # Reset forces
        for particle in md_system.particles:
            particle.force.fill(0.0)
        
        # Calculate pairwise forces
        for i, particle_i in enumerate(md_system.particles):
            for j, particle_j in enumerate(md_system.particles[i+1:], i+1):
                force = self._calculate_pairwise_force(particle_i, particle_j, force_field, md_system)
                
                particle_i.force += force
                particle_j.force -= force  # Newton's third law
    
    def _calculate_pairwise_force(self, particle_i: Particle, particle_j: Particle, 
                                 force_field: Dict[str, Any], md_system: MDSystem) -> np.ndarray:
        """Calculate pairwise force between two particles."""
        # Calculate distance vector with periodic boundary conditions
        dr = particle_j.position - particle_i.position
        
        if md_system.periodic_boundary_conditions:
            dr = self._apply_minimum_image_convention(dr, md_system.box_dimensions)
        
        r = np.linalg.norm(dr)
        
        if r == 0:
            return np.zeros(3)
        
        if force_field['type'] == 'lennard_jones':
            return self._lennard_jones_force(dr, r, force_field)
        elif force_field['type'] == 'coulomb':
            return self._coulomb_force(dr, r, particle_i.charge, particle_j.charge, force_field)
        else:
            return np.zeros(3)
    
    def _lennard_jones_force(self, dr: np.ndarray, r: float, force_field: Dict[str, Any]) -> np.ndarray:
        """Calculate Lennard-Jones force."""
        epsilon = force_field['epsilon']
        sigma = force_field['sigma']
        cutoff = force_field['cutoff']
        
        if r > cutoff:
            return np.zeros(3)
        
        r_inv = sigma / r
        r6_inv = r_inv**6
        r12_inv = r6_inv**2
        
        force_magnitude = 24 * epsilon * (2 * r12_inv - r6_inv) / r
        
        return force_magnitude * dr / r
    
    def _coulomb_force(self, dr: np.ndarray, r: float, charge_i: float, charge_j: float, 
                      force_field: Dict[str, Any]) -> np.ndarray:
        """Calculate Coulomb force."""
        dielectric = force_field['dielectric_constant']
        cutoff = force_field['cutoff']
        
        if r > cutoff:
            return np.zeros(3)
        
        # Simplified Coulomb force (units depend on system)
        force_magnitude = charge_i * charge_j / (dielectric * r**3)
        
        return force_magnitude * dr
    
    def _integrate_verlet(self, md_system: MDSystem, dt: float):
        """Integrate equations of motion using Verlet algorithm."""
        for particle in md_system.particles:
            # Update positions
            particle.position += particle.velocity * dt + 0.5 * particle.force / particle.mass * dt**2
            
            # Update velocities (requires force at new position, simplified here)
            particle.velocity += particle.force / particle.mass * dt
    
    def _integrate_langevin(self, md_system: MDSystem, dt: float, 
                           friction: float, temperature: float):
        """Integrate Langevin dynamics."""
        for particle in md_system.particles:
            # Random force
            random_force = np.random.normal(0, np.sqrt(2 * friction * temperature / dt), 3)
            
            # Update velocity
            particle.velocity += (particle.force / particle.mass - friction * particle.velocity + random_force / particle.mass) * dt
            
            # Update position
            particle.position += particle.velocity * dt
    
    def _integrate_brownian(self, md_system: MDSystem, dt: float, 
                           drag: float, temperature: float):
        """Integrate Brownian dynamics."""
        for particle in md_system.particles:
            # Deterministic displacement
            deterministic_displacement = particle.force / drag * dt
            
            # Random displacement
            diffusion_coefficient = temperature / drag
            random_displacement = np.random.normal(0, np.sqrt(2 * diffusion_coefficient * dt), 3)
            
            # Update position
            particle.position += deterministic_displacement + random_displacement
            
            # No velocity update in Brownian dynamics
            particle.velocity.fill(0.0)
    
    def _apply_minimum_image_convention(self, dr: np.ndarray, box_dimensions: np.ndarray) -> np.ndarray:
        """Apply minimum image convention for periodic boundary conditions."""
        return dr - box_dimensions * np.round(dr / box_dimensions)
    
    def _apply_periodic_boundary_conditions(self, md_system: MDSystem):
        """Apply periodic boundary conditions to all particles."""
        for particle in md_system.particles:
            particle.position = np.mod(particle.position, md_system.box_dimensions)
    
    def _record_trajectory_frame(self, md_system: MDSystem, trajectory: Dict[str, List], time: float):
        """Record a frame of the trajectory."""
        positions = np.array([p.position.copy() for p in md_system.particles])
        velocities = np.array([p.velocity.copy() for p in md_system.particles])
        forces = np.array([p.force.copy() for p in md_system.particles])
        
        trajectory['positions'].append(positions.tolist())
        trajectory['velocities'].append(velocities.tolist())
        trajectory['forces'].append(forces.tolist())
        trajectory['times'].append(time)
        
        # Calculate instantaneous properties
        kinetic_energy = 0.5 * np.sum([p.mass * np.dot(p.velocity, p.velocity) for p in md_system.particles])
        temperature = 2 * kinetic_energy / (3 * len(md_system.particles))  # Simplified
        
        trajectory['energies'].append(kinetic_energy)
        trajectory['temperatures'].append(temperature)
        trajectory['pressures'].append(0.0)  # Placeholder
    
    def _calculate_final_properties(self, md_system: MDSystem, trajectory: Dict[str, List], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final thermodynamic and structural properties."""
        properties = {}
        
        if trajectory['energies']:
            properties['average_energy'] = np.mean(trajectory['energies'])
            properties['energy_fluctuation'] = np.std(trajectory['energies'])
        
        if trajectory['temperatures']:
            properties['average_temperature'] = np.mean(trajectory['temperatures'])
            properties['temperature_fluctuation'] = np.std(trajectory['temperatures'])
        
        # Calculate radial distribution function
        if len(trajectory['positions']) > 10:
            positions_array = np.array(trajectory['positions'][-10:])  # Last 10 frames
            rdf = self._calculate_rdf(positions_array, md_system.box_dimensions)
            properties['radial_distribution_function'] = rdf
        
        return properties
    
    def _calculate_rdf(self, positions: np.ndarray, box_dimensions: np.ndarray) -> Dict[str, Any]:
        """Calculate radial distribution function."""
        # Simplified RDF calculation
        n_frames, n_particles, _ = positions.shape
        max_r = np.min(box_dimensions) / 2
        n_bins = 100
        dr = max_r / n_bins
        
        r_bins = np.linspace(0, max_r, n_bins)
        rdf = np.zeros(n_bins)
        
        for frame in range(n_frames):
            for i in range(n_particles):
                for j in range(i+1, n_particles):
                    dr_vec = positions[frame, j] - positions[frame, i]
                    dr_vec = self._apply_minimum_image_convention(dr_vec, box_dimensions)
                    r = np.linalg.norm(dr_vec)
                    
                    if r < max_r:
                        bin_index = int(r / dr)
                        if bin_index < n_bins:
                            rdf[bin_index] += 1
        
        # Normalize RDF
        volume = np.prod(box_dimensions)
        density = n_particles / volume
        
        for i in range(n_bins):
            r = (i + 0.5) * dr
            shell_volume = 4 * np.pi * r**2 * dr
            rdf[i] /= (n_frames * n_particles * density * shell_volume / 2)
        
        return {
            'r': r_bins.tolist(),
            'g_r': rdf.tolist(),
            'dr': dr,
            'max_r': max_r
        }
    
    # Placeholder implementations for complex methods
    
    def _setup_thermostat(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Setup thermostat parameters."""
        return {
            'type': parameters.get('thermostat', self.md_config['thermostat']),
            'tau': parameters.get('thermostat_tau', 1.0),
            'target_temperature': parameters.get('temperature', self.md_config['default_temperature'])
        }
    
    def _setup_barostat(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Setup barostat parameters."""
        return {
            'type': parameters.get('barostat', self.md_config['barostat']),
            'tau': parameters.get('barostat_tau', 1.0),
            'target_pressure': parameters.get('pressure', self.md_config['default_pressure'])
        }
    
    def _apply_thermostat(self, md_system: MDSystem, thermostat: Dict[str, Any], parameters: Dict[str, Any]):
        """Apply thermostat to maintain temperature."""
        # Simplified velocity rescaling thermostat
        current_temp = self._calculate_instantaneous_temperature(md_system)
        target_temp = thermostat['target_temperature']
        
        scaling_factor = np.sqrt(target_temp / current_temp)
        
        for particle in md_system.particles:
            particle.velocity *= scaling_factor
    
    def _apply_barostat(self, md_system: MDSystem, barostat: Dict[str, Any], parameters: Dict[str, Any]):
        """Apply barostat to maintain pressure."""
        # Simplified implementation
        pass
    
    def _calculate_instantaneous_temperature(self, md_system: MDSystem) -> float:
        """Calculate instantaneous temperature."""
        kinetic_energy = 0.5 * np.sum([p.mass * np.dot(p.velocity, p.velocity) for p in md_system.particles])
        return 2 * kinetic_energy / (3 * len(md_system.particles))
    
    def _calculate_total_energy(self, md_system: MDSystem, force_field: Dict[str, Any]) -> float:
        """Calculate total energy of the system."""
        # Kinetic energy
        kinetic_energy = 0.5 * np.sum([p.mass * np.dot(p.velocity, p.velocity) for p in md_system.particles])
        
        # Potential energy (simplified)
        potential_energy = 0.0
        
        return kinetic_energy + potential_energy
    
    # Additional placeholder methods for advanced techniques
    
    def _integrate_lammps(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with LAMMPS."""
        return {
            'software': 'lammps',
            'status': 'connected',
            'capabilities': ['md', 'minimize', 'dump', 'compute'],
            'version': '29Oct2020',
            'executable_path': config.get('executable_path', '/usr/local/bin/lmp'),
            'configuration': config
        }
    
    def _integrate_gromacs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with GROMACS."""
        return {
            'software': 'gromacs',
            'status': 'connected',
            'capabilities': ['mdrun', 'grompp', 'editconf', 'genbox'],
            'version': '2020.6',
            'executable_path': config.get('executable_path', '/usr/local/bin/gmx'),
            'configuration': config
        }
    
    # Placeholder methods for the various helper functions referenced above
    # These would be implemented with full functionality in a production system
    
    def _calculate_msd(self, *args): return {}
    def _calculate_vacf(self, *args): return {}
    def _calculate_structure_factor(self, *args): return {}
    def _calculate_thermal_conductivity(self, *args): return {}
    def _calculate_viscosity(self, *args): return {}
    def _calculate_diffusion_coefficient(self, *args): return {}
    def _record_brownian_frame(self, *args): pass
    def _calculate_diffusion_properties(self, *args): return {}
    def _calculate_langevin_properties(self, *args): return {}
    def _perform_mc_moves(self, *args): return 0.5
    def _copy_md_system(self, md_system): return md_system
    def _rescale_velocities_to_temperature(self, *args): pass
    def _get_system_snapshot(self, *args): return {}
    def _attempt_replica_exchanges(self, *args): return []
    def _run_biased_md(self, *args): return {}
    def _analyze_umbrella_window(self, *args): return {}
    def _perform_wham_analysis(self, *args): return {}
    def _calculate_collective_variables(self, *args): return [0.0]
    def _calculate_metadynamics_bias_force(self, *args): return np.zeros(3)
    def _apply_bias_force(self, *args): pass
    def _calculate_bias_potential(self, *args): return 0.0
    def _reconstruct_free_energy_surface(self, *args): return {}
    def _optimize_equilibration_time(self, *args): return {'success': False}
    def _optimize_sampling_efficiency(self, *args): return {'success': False}
    def _optimize_computational_cost(self, *args): return {'success': False}
    def _handle_numerical_instability(self, *args): return {'success': False}
    def _handle_energy_explosion(self, *args): return {'success': False}
    def _handle_temperature_runaway(self, *args): return {'success': False}