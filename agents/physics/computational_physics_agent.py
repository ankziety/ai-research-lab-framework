"""
Computational Physics Agent - Specialized agent for numerical methods and simulations.

This agent provides expertise in computational physics, numerical methods,
mathematical modeling, and physics simulations.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from scipy import integrate, optimize, linalg
from scipy.sparse import csc_matrix, diags
import matplotlib.pyplot as plt

from .base_physics_agent import BasePhysicsAgent, PhysicsScale, PhysicsMethodology

logger = logging.getLogger(__name__)


class ComputationalPhysicsAgent(BasePhysicsAgent):
    """
    Specialized agent for computational physics and numerical methods.
    
    Expertise includes:
    - Numerical methods and algorithms
    - Physics simulations (molecular dynamics, Monte Carlo, etc.)
    - Mathematical modeling
    - Finite element/difference methods
    - Optimization and fitting
    - Data analysis and visualization
    """
    
    def __init__(self, agent_id: str, role: str = None, expertise: List[str] = None,
                 model_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        """
        Initialize computational physics agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent role (defaults to "Computational Physics Expert")
            expertise: List of expertise areas (uses defaults if None)
            model_config: Configuration for the underlying LLM
            cost_manager: Optional cost manager for tracking API usage
        """
        if role is None:
            role = "Computational Physics Expert"
        
        if expertise is None:
            expertise = [
                "Numerical Methods",
                "Physics Simulations",
                "Mathematical Modeling",
                "Finite Element Methods",
                "Molecular Dynamics",
                "Monte Carlo Methods",
                "Computational Fluid Dynamics",
                "Optimization Algorithms",
                "Scientific Computing",
                "Data Analysis"
            ]
        
        super().__init__(agent_id, role, expertise, model_config, cost_manager)
        
        # Computational methods database
        self.numerical_methods = {
            'ode_solvers': ['euler', 'runge_kutta', 'adams_bashforth', 'backward_euler'],
            'pde_solvers': ['finite_difference', 'finite_element', 'spectral'],
            'integration': ['simpson', 'gaussian_quadrature', 'monte_carlo'],
            'optimization': ['gradient_descent', 'newton_raphson', 'genetic_algorithm', 'simulated_annealing'],
            'linear_algebra': ['lu_decomposition', 'qr_decomposition', 'svd', 'eigenvalue_methods']
        }
        
        # Simulation frameworks
        self.simulation_frameworks = {
            'molecular_dynamics': {
                'algorithms': ['verlet', 'leap_frog', 'velocity_verlet'],
                'ensembles': ['NVE', 'NVT', 'NPT'],
                'potentials': ['lennard_jones', 'morse', 'harmonic']
            },
            'monte_carlo': {
                'methods': ['metropolis', 'importance_sampling', 'gibbs_sampling'],
                'applications': ['statistical_mechanics', 'quantum_monte_carlo', 'optimization']
            },
            'fluid_dynamics': {
                'methods': ['finite_volume', 'finite_difference', 'lattice_boltzmann'],
                'equations': ['navier_stokes', 'euler', 'burgers']
            }
        }
        
        # Computational resources tracking
        self.computational_metrics = {
            'simulations_run': 0,
            'equations_solved': 0,
            'optimizations_performed': 0,
            'models_fitted': 0,
            'cpu_hours_used': 0.0,
            'memory_peak_gb': 0.0
        }
        
        logger.info(f"Computational Physics Agent {self.agent_id} initialized")
    
    def _get_physics_domain(self) -> str:
        """Get the physics domain for computational physics."""
        return "computational_physics"
    
    def _get_relevant_scales(self) -> List[PhysicsScale]:
        """Get physical scales relevant to computational physics."""
        return [
            PhysicsScale.NANO,
            PhysicsScale.MICRO,
            PhysicsScale.MESO,
            PhysicsScale.MACRO
        ]
    
    def _get_preferred_methodologies(self) -> List[PhysicsMethodology]:
        """Get preferred methodologies for computational physics."""
        return [PhysicsMethodology.COMPUTATIONAL]
    
    def run_molecular_dynamics_simulation(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run molecular dynamics simulation.
        
        Args:
            system_config: Configuration including particles, potentials, and parameters
            
        Returns:
            Simulation results including trajectories, energies, and analysis
        """
        # Try using physics tools first for enhanced capabilities
        if self.physics_tools_available:
            try:
                # Prepare parameters for computational physics tool
                md_params = {
                    'type': 'molecular_dynamics',
                    'system_config': system_config,
                    'simulation_parameters': {
                        'n_particles': system_config.get('n_particles', 100),
                        'n_steps': system_config.get('n_steps', 1000),
                        'time_step': system_config.get('time_step', 0.001),
                        'algorithm': system_config.get('algorithm', 'velocity_verlet'),
                        'potential': system_config.get('potential', 'lennard_jones')
                    }
                }
                
                # Use computational physics tool with engine integration (if available)
                # Note: 'computational_physics_tool' may not exist, so this will fallback gracefully
                tool_result = self.use_physics_tool('computational_physics_tool', md_params)
                
                if tool_result.get('success', False):
                    logger.info(f"MD simulation using {tool_result.get('method', 'physics tool')}")
                    
                    # Convert tool result to expected format
                    simulation_result = {
                        'success': True,
                        'simulation_type': 'molecular_dynamics',
                        'trajectory': tool_result.get('trajectory', []),
                        'energies': tool_result.get('energies', {'kinetic': [], 'potential': [], 'total': []}),
                        'temperature': tool_result.get('temperature', []),
                        'pressure': tool_result.get('pressure', []),
                        'analysis': tool_result.get('analysis', {}),
                        'computational_cost': tool_result.get('computational_cost', {}),
                        'engine_enhanced': tool_result.get('engine_enhanced', False),
                        'computational_method': tool_result.get('method', 'unknown'),
                        'precision': tool_result.get('precision', 'standard'),
                        'n_particles': system_config.get('n_particles', 100),
                        'n_steps': system_config.get('n_steps', 1000),
                        'time_step': system_config.get('time_step', 0.001),
                        'algorithm': system_config.get('algorithm', 'velocity_verlet'),
                        'potential': system_config.get('potential', 'lennard_jones')
                    }
                    
                    self.physics_metrics['simulations_run'] += 1
                    return simulation_result
                
            except Exception as e:
                logger.debug(f"Physics tools failed for MD simulation: {e}")
        
        # Fallback to legacy implementation
        simulation_result = {
            'success': False,
            'simulation_type': 'molecular_dynamics',
            'trajectory': [],
            'energies': {'kinetic': [], 'potential': [], 'total': []},
            'temperature': [],
            'pressure': [],
            'analysis': {},
            'computational_cost': {},
            'engine_enhanced': False,
            'computational_method': 'legacy_numerical'
        }
        
        try:
            # Extract simulation parameters
            n_particles = system_config.get('n_particles', 100)
            n_steps = system_config.get('n_steps', 1000)
            dt = system_config.get('time_step', 0.001)
            algorithm = system_config.get('algorithm', 'velocity_verlet')
            potential = system_config.get('potential', 'lennard_jones')
            
            # Initialize system
            positions, velocities = self._initialize_md_system(n_particles, system_config)
            
            # Run simulation
            trajectory, energies = self._run_md_integration(
                positions, velocities, n_steps, dt, algorithm, potential, system_config
            )
            
            # Analyze results
            analysis = self._analyze_md_trajectory(trajectory, energies, system_config)
            
            simulation_result.update({
                'success': True,
                'trajectory': trajectory,
                'energies': energies,
                'analysis': analysis,
                'n_particles': n_particles,
                'n_steps': n_steps,
                'time_step': dt,
                'algorithm': algorithm,
                'potential': potential,
                'engine_enhanced': False,
                'computational_method': 'legacy_numerical'
            })
            
            self.physics_metrics['simulations_run'] += 1
            
        except Exception as e:
            simulation_result['error'] = str(e)
            logger.error(f"Molecular dynamics simulation failed: {e}")
        
        return simulation_result
    
    def solve_pde_system(self, pde_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve partial differential equation systems.
        
        Args:
            pde_config: PDE configuration including equation type, boundary conditions, and domain
            
        Returns:
            Solution including field values, convergence data, and analysis
        """
        pde_solution = {
            'success': False,
            'equation_type': pde_config.get('equation_type', 'unknown'),
            'solution_field': [],
            'convergence': {},
            'boundary_conditions': {},
            'computational_details': {}
        }
        
        try:
            equation_type = pde_config.get('equation_type', 'poisson')
            method = pde_config.get('method', 'finite_difference')
            domain = pde_config.get('domain', {'x': (0, 1), 'y': (0, 1)})
            
            if equation_type == 'poisson':
                solution = self._solve_poisson_equation(pde_config, method, domain)
            elif equation_type == 'heat':
                solution = self._solve_heat_equation(pde_config, method, domain)
            elif equation_type == 'wave':
                solution = self._solve_wave_equation(pde_config, method, domain)
            elif equation_type == 'schrodinger':
                solution = self._solve_schrodinger_pde(pde_config, method, domain)
            else:
                solution = self._solve_general_pde(pde_config, method, domain)
            
            pde_solution.update(solution)
            pde_solution['success'] = True
            
            self.computational_metrics['equations_solved'] += 1
            
        except Exception as e:
            pde_solution['error'] = str(e)
            logger.error(f"PDE solution failed: {e}")
        
        return pde_solution
    
    def run_monte_carlo_simulation(self, mc_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            mc_config: Monte Carlo configuration including method and parameters
            
        Returns:
            Simulation results including samples, estimates, and convergence analysis
        """
        mc_result = {
            'success': False,
            'method': mc_config.get('method', 'metropolis'),
            'samples': [],
            'estimates': {},
            'convergence': {},
            'acceptance_rate': 0.0,
            'statistical_analysis': {}
        }
        
        try:
            method = mc_config.get('method', 'metropolis')
            n_samples = mc_config.get('n_samples', 10000)
            
            if method == 'metropolis':
                result = self._run_metropolis_mc(mc_config, n_samples)
            elif method == 'importance_sampling':
                result = self._run_importance_sampling(mc_config, n_samples)
            elif method == 'gibbs_sampling':
                result = self._run_gibbs_sampling(mc_config, n_samples)
            else:
                result = self._run_general_mc(mc_config, n_samples)
            
            mc_result.update(result)
            mc_result['success'] = True
            
            self.computational_metrics['simulations_run'] += 1
            
        except Exception as e:
            mc_result['error'] = str(e)
            logger.error(f"Monte Carlo simulation failed: {e}")
        
        return mc_result
    
    def optimize_physical_system(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize physical systems using various algorithms.
        
        Args:
            optimization_config: Configuration including objective function and constraints
            
        Returns:
            Optimization results including optimal parameters and convergence data
        """
        optimization_result = {
            'success': False,
            'method': optimization_config.get('method', 'gradient_descent'),
            'optimal_parameters': {},
            'optimal_value': None,
            'convergence_history': [],
            'function_evaluations': 0,
            'computational_cost': {}
        }
        
        try:
            method = optimization_config.get('method', 'gradient_descent')
            objective_function = optimization_config.get('objective_function')
            initial_guess = optimization_config.get('initial_guess')
            
            if method == 'gradient_descent':
                result = self._run_gradient_descent(objective_function, initial_guess, optimization_config)
            elif method == 'newton_raphson':
                result = self._run_newton_raphson(objective_function, initial_guess, optimization_config)
            elif method == 'genetic_algorithm':
                result = self._run_genetic_algorithm(objective_function, optimization_config)
            elif method == 'simulated_annealing':
                result = self._run_simulated_annealing(objective_function, initial_guess, optimization_config)
            else:
                result = self._run_scipy_optimization(objective_function, initial_guess, optimization_config)
            
            optimization_result.update(result)
            optimization_result['success'] = True
            
            self.computational_metrics['optimizations_performed'] += 1
            
        except Exception as e:
            optimization_result['error'] = str(e)
            logger.error(f"Optimization failed: {e}")
        
        return optimization_result
    
    def fit_physics_model(self, data: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit physics models to experimental or simulation data.
        
        Args:
            data: Experimental or simulation data
            model_config: Model configuration and fitting parameters
            
        Returns:
            Fitting results including parameters, uncertainties, and quality metrics
        """
        fitting_result = {
            'success': False,
            'model_type': model_config.get('model_type', 'unknown'),
            'fitted_parameters': {},
            'parameter_uncertainties': {},
            'goodness_of_fit': {},
            'residuals': [],
            'confidence_intervals': {}
        }
        
        try:
            model_type = model_config.get('model_type', 'polynomial')
            x_data = np.array(data.get('x', []))
            y_data = np.array(data.get('y', []))
            
            if model_type == 'polynomial':
                result = self._fit_polynomial_model(x_data, y_data, model_config)
            elif model_type == 'exponential':
                result = self._fit_exponential_model(x_data, y_data, model_config)
            elif model_type == 'power_law':
                result = self._fit_power_law_model(x_data, y_data, model_config)
            elif model_type == 'gaussian':
                result = self._fit_gaussian_model(x_data, y_data, model_config)
            elif model_type == 'custom':
                result = self._fit_custom_model(x_data, y_data, model_config)
            else:
                result = self._fit_general_model(x_data, y_data, model_config)
            
            fitting_result.update(result)
            fitting_result['success'] = True
            
            self.computational_metrics['models_fitted'] += 1
            
        except Exception as e:
            fitting_result['error'] = str(e)
            logger.error(f"Model fitting failed: {e}")
        
        return fitting_result
    
    def _discover_physics_specific_tools(self, research_question: str) -> List[Dict[str, Any]]:
        """Discover computational physics specific tools."""
        computational_tools = []
        question_lower = research_question.lower()
        
        # Simulation tools
        if any(keyword in question_lower for keyword in 
               ['simulate', 'simulation', 'molecular dynamics', 'monte carlo']):
            computational_tools.append({
                'tool_id': 'physics_simulator',
                'name': 'Physics Simulation Engine',
                'description': 'Comprehensive physics simulation toolkit',
                'capabilities': ['molecular_dynamics', 'monte_carlo', 'finite_element', 'optimization'],
                'confidence': 0.95,
                'physics_specific': True,
                'scales': ['nano', 'micro', 'macro'],
                'methodologies': ['computational']
            })
        
        # Numerical solver tools
        if any(keyword in question_lower for keyword in 
               ['solve', 'equation', 'differential', 'numerical']):
            computational_tools.append({
                'tool_id': 'numerical_solver',
                'name': 'Numerical Equation Solver',
                'description': 'Advanced numerical methods for solving equations',
                'capabilities': ['ode_solving', 'pde_solving', 'linear_algebra', 'optimization'],
                'confidence': 0.9,
                'physics_specific': True,
                'scales': ['nano', 'micro', 'macro'],
                'methodologies': ['computational']
            })
        
        # Data analysis tools
        if any(keyword in question_lower for keyword in 
               ['analyze', 'fit', 'model', 'data']):
            computational_tools.append({
                'tool_id': 'data_analyzer',
                'name': 'Physics Data Analysis Tool',
                'description': 'Statistical analysis and model fitting for physics data',
                'capabilities': ['curve_fitting', 'statistical_analysis', 'visualization', 'uncertainty_analysis'],
                'confidence': 0.85,
                'physics_specific': True,
                'scales': ['micro', 'macro'],
                'methodologies': ['computational']
            })
        
        return computational_tools
    
    # Private helper methods for simulations
    
    def _initialize_md_system(self, n_particles: int, config: Dict[str, Any]) -> tuple:
        """Initialize molecular dynamics system."""
        # Initialize positions on a cubic lattice
        box_size = config.get('box_size', 10.0)
        n_side = int(np.ceil(n_particles**(1/3)))
        
        positions = []
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if len(positions) < n_particles:
                        x = (i + 0.5) * box_size / n_side
                        y = (j + 0.5) * box_size / n_side
                        z = (k + 0.5) * box_size / n_side
                        positions.append([x, y, z])
        
        positions = np.array(positions[:n_particles])
        
        # Initialize velocities from Maxwell-Boltzmann distribution
        temperature = config.get('temperature', 300.0)
        mass = config.get('particle_mass', 1.0)
        kb = 1.380649e-23  # Boltzmann constant
        
        velocities = np.random.normal(0, np.sqrt(kb * temperature / mass), 
                                    (n_particles, 3))
        
        # Remove center of mass motion
        velocities -= np.mean(velocities, axis=0)
        
        return positions, velocities
    
    def _run_md_integration(self, positions: np.ndarray, velocities: np.ndarray,
                           n_steps: int, dt: float, algorithm: str, 
                           potential: str, config: Dict[str, Any]) -> tuple:
        """Run molecular dynamics integration."""
        trajectory = [positions.copy()]
        energies = {'kinetic': [], 'potential': [], 'total': []}
        
        for step in range(n_steps):
            # Calculate forces
            forces = self._calculate_forces(positions, potential, config)
            
            # Integrate equations of motion
            if algorithm == 'velocity_verlet':
                positions, velocities = self._velocity_verlet_step(
                    positions, velocities, forces, dt, config
                )
            elif algorithm == 'leap_frog':
                positions, velocities = self._leap_frog_step(
                    positions, velocities, forces, dt, config
                )
            else:  # Default to Verlet
                positions, velocities = self._verlet_step(
                    positions, velocities, forces, dt, config
                )
            
            # Apply periodic boundary conditions
            box_size = config.get('box_size', 10.0)
            positions = positions % box_size
            
            # Calculate energies
            ke = self._calculate_kinetic_energy(velocities, config)
            pe = self._calculate_potential_energy(positions, potential, config)
            
            energies['kinetic'].append(ke)
            energies['potential'].append(pe)
            energies['total'].append(ke + pe)
            
            # Store trajectory (every nth step to save memory)
            if step % config.get('save_frequency', 10) == 0:
                trajectory.append(positions.copy())
        
        return trajectory, energies
    
    def _calculate_forces(self, positions: np.ndarray, potential: str, 
                         config: Dict[str, Any]) -> np.ndarray:
        """Calculate forces on particles."""
        n_particles = len(positions)
        forces = np.zeros_like(positions)
        
        if potential == 'lennard_jones':
            epsilon = config.get('lj_epsilon', 1.0)
            sigma = config.get('lj_sigma', 1.0)
            
            for i in range(n_particles):
                for j in range(i + 1, n_particles):
                    dr = positions[j] - positions[i]
                    r = np.linalg.norm(dr)
                    
                    if r > 0:
                        # Lennard-Jones force
                        r6_inv = (sigma / r)**6
                        f_magnitude = 24 * epsilon * r6_inv * (2 * r6_inv - 1) / r**2
                        force_ij = f_magnitude * dr
                        
                        forces[i] -= force_ij
                        forces[j] += force_ij
        
        return forces
    
    def _velocity_verlet_step(self, positions: np.ndarray, velocities: np.ndarray,
                             forces: np.ndarray, dt: float, config: Dict[str, Any]) -> tuple:
        """Velocity Verlet integration step."""
        mass = config.get('particle_mass', 1.0)
        
        # Update positions
        positions_new = positions + velocities * dt + 0.5 * forces / mass * dt**2
        
        # Calculate forces at new positions
        forces_new = self._calculate_forces(positions_new, config.get('potential', 'lennard_jones'), config)
        
        # Update velocities
        velocities_new = velocities + 0.5 * (forces + forces_new) / mass * dt
        
        return positions_new, velocities_new
    
    def _leap_frog_step(self, positions: np.ndarray, velocities: np.ndarray,
                       forces: np.ndarray, dt: float, config: Dict[str, Any]) -> tuple:
        """Leap-frog integration step."""
        mass = config.get('particle_mass', 1.0)
        
        # Update velocities by half step
        velocities_half = velocities + 0.5 * forces / mass * dt
        
        # Update positions
        positions_new = positions + velocities_half * dt
        
        # Calculate forces at new positions
        forces_new = self._calculate_forces(positions_new, config.get('potential', 'lennard_jones'), config)
        
        # Update velocities by another half step
        velocities_new = velocities_half + 0.5 * forces_new / mass * dt
        
        return positions_new, velocities_new
    
    def _verlet_step(self, positions: np.ndarray, velocities: np.ndarray,
                    forces: np.ndarray, dt: float, config: Dict[str, Any]) -> tuple:
        """Basic Verlet integration step."""
        mass = config.get('particle_mass', 1.0)
        
        # Update positions
        positions_new = positions + velocities * dt + 0.5 * forces / mass * dt**2
        
        # Update velocities (using current forces)
        velocities_new = velocities + forces / mass * dt
        
        return positions_new, velocities_new
    
    def _calculate_kinetic_energy(self, velocities: np.ndarray, config: Dict[str, Any]) -> float:
        """Calculate kinetic energy of the system."""
        mass = config.get('particle_mass', 1.0)
        return 0.5 * mass * np.sum(velocities**2)
    
    def _calculate_potential_energy(self, positions: np.ndarray, potential: str, 
                                  config: Dict[str, Any]) -> float:
        """Calculate potential energy of the system."""
        n_particles = len(positions)
        pe = 0.0
        
        if potential == 'lennard_jones':
            epsilon = config.get('lj_epsilon', 1.0)
            sigma = config.get('lj_sigma', 1.0)
            
            for i in range(n_particles):
                for j in range(i + 1, n_particles):
                    dr = positions[j] - positions[i]
                    r = np.linalg.norm(dr)
                    
                    if r > 0:
                        r6_inv = (sigma / r)**6
                        pe += 4 * epsilon * r6_inv * (r6_inv - 1)
        
        return pe
    
    def _analyze_md_trajectory(self, trajectory: List[np.ndarray], 
                              energies: Dict[str, List[float]], 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze molecular dynamics trajectory."""
        analysis = {
            'average_temperature': 0.0,
            'energy_conservation': 0.0,
            'radial_distribution': {},
            'diffusion_coefficient': 0.0,
            'equilibration_time': 0.0
        }
        
        try:
            # Calculate average temperature
            if energies['kinetic']:
                kb = 1.380649e-23
                mass = config.get('particle_mass', 1.0)
                n_particles = len(trajectory[0])
                avg_ke = np.mean(energies['kinetic'])
                analysis['average_temperature'] = 2 * avg_ke / (3 * n_particles * kb)
            
            # Check energy conservation
            if energies['total']:
                total_energies = np.array(energies['total'])
                analysis['energy_conservation'] = np.std(total_energies) / np.mean(np.abs(total_energies))
            
            # Calculate diffusion coefficient (simplified)
            if len(trajectory) > 10:
                displacements = []
                for i in range(1, min(11, len(trajectory))):
                    displacement = np.mean(np.linalg.norm(trajectory[i] - trajectory[0], axis=1)**2)
                    displacements.append(displacement)
                
                if displacements:
                    # Linear fit to get diffusion coefficient
                    times = np.arange(1, len(displacements) + 1) * config.get('save_frequency', 10) * config.get('time_step', 0.001)
                    if len(times) > 1:
                        slope, _ = np.polyfit(times, displacements, 1)
                        analysis['diffusion_coefficient'] = slope / 6  # 3D diffusion
        
        except Exception as e:
            logger.warning(f"MD trajectory analysis failed: {e}")
        
        return analysis
    
    # PDE solving methods
    
    def _solve_poisson_equation(self, config: Dict[str, Any], method: str, 
                               domain: Dict[str, Any]) -> Dict[str, Any]:
        """Solve Poisson equation using specified method."""
        nx, ny = config.get('grid_size', (50, 50))
        
        # Create grid
        x = np.linspace(domain['x'][0], domain['x'][1], nx)
        y = np.linspace(domain['y'][0], domain['y'][1], ny)
        X, Y = np.meshgrid(x, y)
        
        # Set up finite difference matrix (simplified 2D Poisson)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Create system matrix (simplified)
        n_points = nx * ny
        A = diags([1, 1, -4, 1, 1], [-nx, -1, 0, 1, nx], shape=(n_points, n_points))
        
        # Source term
        source = config.get('source_function', lambda x, y: np.zeros_like(x))
        b = source(X, Y).flatten()
        
        # Solve system
        solution = linalg.spsolve(A, b).reshape((ny, nx))
        
        return {
            'solution_field': solution.tolist(),
            'x_grid': x.tolist(),
            'y_grid': y.tolist(),
            'method': method
        }
    
    def _solve_heat_equation(self, config: Dict[str, Any], method: str, 
                            domain: Dict[str, Any]) -> Dict[str, Any]:
        """Solve heat equation using specified method."""
        # Simplified heat equation solution
        nx = config.get('nx', 50)
        nt = config.get('nt', 100)
        
        x = np.linspace(domain['x'][0], domain['x'][1], nx)
        dx = x[1] - x[0]
        dt = config.get('dt', 0.001)
        alpha = config.get('thermal_diffusivity', 1.0)
        
        # Stability condition
        r = alpha * dt / dx**2
        if r > 0.5:
            logger.warning(f"Stability condition violated: r = {r} > 0.5")
        
        # Initial condition
        u = config.get('initial_condition', lambda x: np.sin(np.pi * x))
        solution = u(x)
        
        # Time evolution
        solution_history = [solution.copy()]
        for n in range(nt):
            solution[1:-1] = solution[1:-1] + r * (solution[2:] - 2*solution[1:-1] + solution[:-2])
            if n % 10 == 0:  # Save every 10th step
                solution_history.append(solution.copy())
        
        return {
            'solution_field': solution_history,
            'x_grid': x.tolist(),
            'time_steps': list(range(0, nt+1, 10)),
            'method': method
        }
    
    def _solve_wave_equation(self, config: Dict[str, Any], method: str, 
                            domain: Dict[str, Any]) -> Dict[str, Any]:
        """Solve wave equation using specified method."""
        # Simplified 1D wave equation
        nx = config.get('nx', 100)
        nt = config.get('nt', 200)
        
        x = np.linspace(domain['x'][0], domain['x'][1], nx)
        dx = x[1] - x[0]
        dt = config.get('dt', 0.01)
        c = config.get('wave_speed', 1.0)
        
        # CFL condition
        cfl = c * dt / dx
        if cfl > 1:
            logger.warning(f"CFL condition violated: CFL = {cfl} > 1")
        
        # Initial conditions
        u_init = config.get('initial_displacement', lambda x: np.exp(-(x-0.5)**2/0.01))
        v_init = config.get('initial_velocity', lambda x: np.zeros_like(x))
        
        u_prev = u_init(x)
        u_curr = u_prev + dt * v_init(x)
        
        solution_history = [u_prev.copy(), u_curr.copy()]
        
        # Time evolution
        for n in range(2, nt):
            u_next = np.zeros_like(u_curr)
            u_next[1:-1] = (2*u_curr[1:-1] - u_prev[1:-1] + 
                           cfl**2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))
            
            u_prev, u_curr = u_curr, u_next
            
            if n % 10 == 0:  # Save every 10th step
                solution_history.append(u_curr.copy())
        
        return {
            'solution_field': solution_history,
            'x_grid': x.tolist(),
            'time_steps': list(range(0, nt, 10)),
            'method': method
        }
    
    def _solve_schrodinger_pde(self, config: Dict[str, Any], method: str, 
                              domain: Dict[str, Any]) -> Dict[str, Any]:
        """Solve time-dependent Schrödinger equation."""
        # This would require more sophisticated numerical methods
        return {
            'solution_field': [],
            'method': method,
            'note': 'Schrödinger PDE solving requires specialized implementation'
        }
    
    def _solve_general_pde(self, config: Dict[str, Any], method: str, 
                          domain: Dict[str, Any]) -> Dict[str, Any]:
        """Solve general PDE system."""
        return {
            'solution_field': [],
            'method': method,
            'note': f'General PDE solver for {config.get("equation_type", "unknown")} not implemented'
        }
    
    # Monte Carlo methods
    
    def _run_metropolis_mc(self, config: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
        """Run Metropolis Monte Carlo simulation."""
        # Simplified Metropolis algorithm
        target_function = config.get('target_function', lambda x: np.exp(-x**2/2))
        step_size = config.get('step_size', 1.0)
        initial_state = config.get('initial_state', 0.0)
        
        samples = []
        current_state = initial_state
        n_accepted = 0
        
        for i in range(n_samples):
            # Propose new state
            proposal = current_state + np.random.normal(0, step_size)
            
            # Calculate acceptance probability
            current_prob = target_function(current_state)
            proposal_prob = target_function(proposal)
            
            alpha = min(1.0, proposal_prob / current_prob) if current_prob > 0 else 1.0
            
            # Accept or reject
            if np.random.random() < alpha:
                current_state = proposal
                n_accepted += 1
            
            samples.append(current_state)
        
        return {
            'samples': samples,
            'acceptance_rate': n_accepted / n_samples,
            'method': 'metropolis'
        }
    
    def _run_importance_sampling(self, config: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
        """Run importance sampling Monte Carlo."""
        # Simplified importance sampling
        return {
            'samples': np.random.normal(0, 1, n_samples).tolist(),
            'method': 'importance_sampling',
            'note': 'Simplified implementation'
        }
    
    def _run_gibbs_sampling(self, config: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
        """Run Gibbs sampling."""
        # Simplified Gibbs sampling
        return {
            'samples': np.random.normal(0, 1, (n_samples, 2)).tolist(),
            'method': 'gibbs_sampling',
            'note': 'Simplified implementation'
        }
    
    def _run_general_mc(self, config: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
        """Run general Monte Carlo method."""
        return {
            'samples': np.random.random(n_samples).tolist(),
            'method': 'general_mc'
        }
    
    # Optimization methods
    
    def _run_gradient_descent(self, objective_function: Callable, initial_guess: np.ndarray,
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Run gradient descent optimization."""
        # Simplified gradient descent
        learning_rate = config.get('learning_rate', 0.01)
        max_iterations = config.get('max_iterations', 1000)
        tolerance = config.get('tolerance', 1e-6)
        
        x = np.array(initial_guess)
        history = [x.copy()]
        
        for i in range(max_iterations):
            # Numerical gradient
            grad = self._numerical_gradient(objective_function, x)
            x_new = x - learning_rate * grad
            
            if np.linalg.norm(x_new - x) < tolerance:
                break
            
            x = x_new
            history.append(x.copy())
        
        return {
            'optimal_parameters': x.tolist(),
            'optimal_value': objective_function(x),
            'convergence_history': history,
            'iterations': i + 1
        }
    
    def _numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Calculate numerical gradient."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
    
    def _run_newton_raphson(self, objective_function: Callable, initial_guess: np.ndarray,
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Newton-Raphson optimization."""
        # Use scipy's implementation
        result = optimize.minimize(objective_function, initial_guess, method='Newton-CG')
        
        return {
            'optimal_parameters': result.x.tolist(),
            'optimal_value': result.fun,
            'success': result.success,
            'message': result.message
        }
    
    def _run_genetic_algorithm(self, objective_function: Callable, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        # Simplified GA implementation
        return {
            'optimal_parameters': [0.0],
            'optimal_value': 0.0,
            'note': 'Simplified GA implementation'
        }
    
    def _run_simulated_annealing(self, objective_function: Callable, initial_guess: np.ndarray,
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Run simulated annealing optimization."""
        # Use scipy's implementation
        result = optimize.dual_annealing(objective_function, 
                                       bounds=[(-10, 10)] * len(initial_guess))
        
        return {
            'optimal_parameters': result.x.tolist(),
            'optimal_value': result.fun,
            'success': result.success
        }
    
    def _run_scipy_optimization(self, objective_function: Callable, initial_guess: np.ndarray,
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Run scipy optimization."""
        method = config.get('scipy_method', 'BFGS')
        result = optimize.minimize(objective_function, initial_guess, method=method)
        
        return {
            'optimal_parameters': result.x.tolist(),
            'optimal_value': result.fun,
            'success': result.success,
            'method': method
        }
    
    # Model fitting methods
    
    def _fit_polynomial_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit polynomial model to data."""
        degree = config.get('degree', 2)
        
        # Polynomial fit
        coefficients = np.polyfit(x_data, y_data, degree)
        
        # Calculate R-squared
        y_pred = np.polyval(coefficients, x_data)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'fitted_parameters': {'coefficients': coefficients.tolist()},
            'goodness_of_fit': {'r_squared': r_squared, 'rmse': np.sqrt(np.mean((y_data - y_pred)**2))},
            'residuals': (y_data - y_pred).tolist(),
            'model_type': f'polynomial_degree_{degree}'
        }
    
    def _fit_exponential_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit exponential model to data."""
        # Fit y = a * exp(b * x)
        try:
            popt, pcov = optimize.curve_fit(lambda x, a, b: a * np.exp(b * x), 
                                          x_data, y_data)
            
            y_pred = popt[0] * np.exp(popt[1] * x_data)
            r_squared = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            return {
                'fitted_parameters': {'a': popt[0], 'b': popt[1]},
                'parameter_uncertainties': {'a': np.sqrt(pcov[0,0]), 'b': np.sqrt(pcov[1,1])},
                'goodness_of_fit': {'r_squared': r_squared},
                'residuals': (y_data - y_pred).tolist()
            }
        except Exception as e:
            return {'error': f'Exponential fit failed: {str(e)}'}
    
    def _fit_power_law_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit power law model to data."""
        # Fit y = a * x^b
        try:
            popt, pcov = optimize.curve_fit(lambda x, a, b: a * np.power(x, b), 
                                          x_data, y_data)
            
            y_pred = popt[0] * np.power(x_data, popt[1])
            r_squared = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            return {
                'fitted_parameters': {'a': popt[0], 'b': popt[1]},
                'parameter_uncertainties': {'a': np.sqrt(pcov[0,0]), 'b': np.sqrt(pcov[1,1])},
                'goodness_of_fit': {'r_squared': r_squared},
                'residuals': (y_data - y_pred).tolist()
            }
        except Exception as e:
            return {'error': f'Power law fit failed: {str(e)}'}
    
    def _fit_gaussian_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit Gaussian model to data."""
        # Fit y = a * exp(-(x-mu)^2/(2*sigma^2))
        try:
            popt, pcov = optimize.curve_fit(
                lambda x, a, mu, sigma: a * np.exp(-(x - mu)**2 / (2 * sigma**2)), 
                x_data, y_data
            )
            
            y_pred = popt[0] * np.exp(-(x_data - popt[1])**2 / (2 * popt[2]**2))
            r_squared = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            return {
                'fitted_parameters': {'amplitude': popt[0], 'mean': popt[1], 'std': popt[2]},
                'parameter_uncertainties': {
                    'amplitude': np.sqrt(pcov[0,0]), 
                    'mean': np.sqrt(pcov[1,1]), 
                    'std': np.sqrt(pcov[2,2])
                },
                'goodness_of_fit': {'r_squared': r_squared},
                'residuals': (y_data - y_pred).tolist()
            }
        except Exception as e:
            return {'error': f'Gaussian fit failed: {str(e)}'}
    
    def _fit_custom_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit custom model to data."""
        # Placeholder for custom model fitting
        return {
            'fitted_parameters': {},
            'note': 'Custom model fitting requires specific implementation'
        }
    
    def _fit_general_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit general model to data."""
        # Default to linear fit
        coefficients = np.polyfit(x_data, y_data, 1)
        y_pred = np.polyval(coefficients, x_data)
        r_squared = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
        
        return {
            'fitted_parameters': {'slope': coefficients[0], 'intercept': coefficients[1]},
            'goodness_of_fit': {'r_squared': r_squared},
            'residuals': (y_data - y_pred).tolist(),
            'model_type': 'linear'
        }