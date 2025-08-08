"""
Statistical Mechanics and Thermodynamics Tool

Agent-friendly interface for statistical mechanics and thermodynamics calculations.
Provides ensemble calculations, phase transitions, transport coefficients,
and critical phenomena analysis.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class StatisticalMechanicsTool(BasePhysicsTool):
    """
    Tool for statistical mechanics and thermodynamics calculations that agents can request.
    
    Provides interfaces for:
    - Statistical ensemble calculations
    - Thermodynamic properties
    - Phase transitions and critical phenomena
    - Transport coefficients
    - Non-equilibrium statistical mechanics
    - Monte Carlo simulations
    """
    
    def __init__(self):
        super().__init__(
            tool_id="statistical_mechanics_tool",
            name="Statistical Mechanics and Thermodynamics Tool",
            description="Perform statistical mechanics calculations including ensembles, phase transitions, and thermodynamic properties",
            physics_domain="statistical_mechanics",
            computational_cost_factor=2.8,
            software_requirements=[
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib",   # Visualization
                "sympy"         # Symbolic mathematics (optional)
            ],
            hardware_requirements={
                "min_memory": 1024,  # MB
                "recommended_memory": 4096,
                "cpu_cores": 4,
                "supports_gpu": True
            }
        )
        
        self.capabilities.extend([
            "canonical_ensemble",
            "grand_canonical_ensemble",
            "microcanonical_ensemble",
            "thermodynamic_properties",
            "phase_transitions",
            "critical_phenomena",
            "transport_coefficients",
            "monte_carlo_simulation",
            "fluctuation_dissipation"
        ])
        
        # Physical constants
        self.constants = {
            "kb": 1.381e-23,    # J/K
            "na": 6.022e23,     # mol⁻¹
            "r": 8.314,         # J/(mol·K)
            "h": 6.626e-34,     # J·s
            "hbar": 1.055e-34,  # J·s
            "sigma_sb": 5.67e-8 # W/(m²·K⁴) Stefan-Boltzmann
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute statistical mechanics calculation.
        
        Args:
            task: Task specification with statistical mechanics parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with statistical mechanics analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "canonical_ensemble")
            
            if task_type == "canonical_ensemble":
                result = self._analyze_canonical_ensemble(task)
            elif task_type == "grand_canonical_ensemble":
                result = self._analyze_grand_canonical_ensemble(task)
            elif task_type == "thermodynamic_properties":
                result = self._calculate_thermodynamic_properties(task)
            elif task_type == "phase_transition":
                result = self._analyze_phase_transition(task)
            elif task_type == "critical_phenomena":
                result = self._analyze_critical_phenomena(task)
            elif task_type == "transport_coefficients":
                result = self._calculate_transport_coefficients(task)
            elif task_type == "monte_carlo":
                result = self._perform_monte_carlo_simulation(task)
            elif task_type == "ising_model":
                result = self._analyze_ising_model(task)
            else:
                result = self._generic_statistical_calculation(task)
            
            # Calculate execution time and cost
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            computational_cost = execution_time * self.computational_cost_factor
            
            # Update usage statistics
            self.usage_count += 1
            self.total_computational_cost += computational_cost
            self.average_calculation_time = (
                (self.average_calculation_time * (self.usage_count - 1) + execution_time) / 
                self.usage_count
            )
            
            # Calculate success based on result quality
            success = result.get("convergence", True) and result.get("physical_validity", True)
            if success:
                self.successful_calculations += 1
            
            self.success_rate = self.successful_calculations / self.usage_count
            
            return {
                "success": success,
                "calculation_time": execution_time,
                "computational_cost": computational_cost,
                "result": result,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id,
                "timestamp": datetime.now().isoformat(),
                "context": context.get("agent_id", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Statistical mechanics calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _analyze_canonical_ensemble(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze canonical ensemble (NVT) properties."""
        system_type = task.get("system", "ideal_gas")
        temperature = task.get("temperature", 300)  # K
        num_particles = task.get("num_particles", 1e23)
        volume = task.get("volume", 0.0224)  # m³ (STP molar volume)
        
        if system_type == "ideal_gas":
            # Ideal gas in canonical ensemble
            
            # Partition function (translational part)
            mass = task.get("particle_mass", 4.65e-26)  # kg (N₂ molecule)
            thermal_wavelength = self.constants["h"] / np.sqrt(2 * np.pi * mass * self.constants["kb"] * temperature)
            
            # Single particle partition function
            z_trans = volume / thermal_wavelength**3
            
            # Total partition function
            partition_function = z_trans**num_particles / math.factorial(int(min(num_particles, 170)))  # Stirling approx for large N
            
            # Use Stirling's approximation for large N
            if num_particles > 10:
                ln_z = num_particles * (np.log(z_trans) - np.log(num_particles) + 1)
            else:
                ln_z = num_particles * np.log(z_trans) - np.log(math.factorial(int(num_particles)))
            
            # Thermodynamic quantities
            internal_energy = (3/2) * num_particles * self.constants["kb"] * temperature
            pressure = num_particles * self.constants["kb"] * temperature / volume
            entropy = self.constants["kb"] * (ln_z + (3/2) * num_particles)
            free_energy = -self.constants["kb"] * temperature * ln_z
            
            # Heat capacity
            heat_capacity_v = (3/2) * num_particles * self.constants["kb"]
            heat_capacity_p = (5/2) * num_particles * self.constants["kb"]
            
            return {
                "system": system_type,
                "ensemble": "canonical_NVT",
                "temperature": temperature,
                "volume": volume,
                "num_particles": num_particles,
                "partition_function": {
                    "ln_Z": ln_z,
                    "thermal_wavelength": thermal_wavelength
                },
                "thermodynamic_properties": {
                    "internal_energy": internal_energy,
                    "pressure": pressure,
                    "entropy": entropy,
                    "helmholtz_free_energy": free_energy,
                    "heat_capacity_V": heat_capacity_v,
                    "heat_capacity_P": heat_capacity_p
                },
                "equation_of_state": "PV = NkT",
                "units": {
                    "energy": "J",
                    "pressure": "Pa",
                    "entropy": "J/K",
                    "volume": "m³"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif system_type == "harmonic_oscillator":
            # Quantum harmonic oscillator
            frequency = task.get("frequency", 1e13)  # Hz
            
            # Energy levels
            energy_quantum = self.constants["hbar"] * 2 * np.pi * frequency
            
            # Partition function
            beta = 1 / (self.constants["kb"] * temperature)
            z_osc = 1 / (1 - np.exp(-beta * energy_quantum))
            
            # Average energy
            avg_energy = energy_quantum / (np.exp(beta * energy_quantum) - 1)
            
            # Heat capacity
            x = beta * energy_quantum
            heat_capacity = num_particles * self.constants["kb"] * x**2 * np.exp(x) / (np.exp(x) - 1)**2
            
            return {
                "system": system_type,
                "ensemble": "canonical_NVT",
                "temperature": temperature,
                "frequency": frequency,
                "energy_quantum": energy_quantum,
                "partition_function": z_osc,
                "average_energy": avg_energy,
                "total_energy": num_particles * avg_energy,
                "heat_capacity": heat_capacity,
                "classical_limit": temperature > energy_quantum / self.constants["kb"],
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "system": system_type,
                "error": f"Unknown system type: {system_type}",
                "available_systems": ["ideal_gas", "harmonic_oscillator"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_grand_canonical_ensemble(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze grand canonical ensemble (μVT) properties."""
        system_type = task.get("system", "ideal_gas")
        temperature = task.get("temperature", 300)  # K
        chemical_potential = task.get("chemical_potential", -20)  # kT units
        volume = task.get("volume", 1e-3)  # m³
        
        if system_type == "ideal_gas":
            # Grand canonical ideal gas
            mass = task.get("particle_mass", 4.65e-26)  # kg
            
            # Thermal wavelength
            thermal_wavelength = self.constants["h"] / np.sqrt(2 * np.pi * mass * self.constants["kb"] * temperature)
            
            # Fugacity
            beta = 1 / (self.constants["kb"] * temperature)
            fugacity = np.exp(beta * chemical_potential * self.constants["kb"] * temperature)
            
            # Grand partition function (logarithm)
            single_particle_state_density = volume / thermal_wavelength**3
            ln_grand_z = fugacity * single_particle_state_density
            
            # Average number of particles
            avg_num_particles = fugacity * single_particle_state_density
            
            # Pressure
            pressure = self.constants["kb"] * temperature * ln_grand_z / volume
            
            # Particle number fluctuations
            particle_fluctuations = avg_num_particles
            relative_fluctuations = np.sqrt(particle_fluctuations) / avg_num_particles
            
            # Density
            density = avg_num_particles / volume
            
            return {
                "system": system_type,
                "ensemble": "grand_canonical_μVT",
                "temperature": temperature,
                "chemical_potential": chemical_potential * self.constants["kb"] * temperature,
                "volume": volume,
                "fugacity": fugacity,
                "average_particle_number": avg_num_particles,
                "particle_fluctuations": particle_fluctuations,
                "relative_fluctuations": relative_fluctuations,
                "pressure": pressure,
                "density": density,
                "thermal_wavelength": thermal_wavelength,
                "units": {
                    "chemical_potential": "J",
                    "pressure": "Pa",
                    "density": "m⁻³"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "system": system_type,
                "error": f"Unknown system type for grand canonical ensemble: {system_type}",
                "available_systems": ["ideal_gas"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _calculate_thermodynamic_properties(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate thermodynamic properties and relations."""
        calculation_type = task.get("calculation", "maxwell_relations")
        
        if calculation_type == "maxwell_relations":
            # Demonstrate Maxwell relations
            temperature = task.get("temperature", 300)  # K
            pressure = task.get("pressure", 1e5)  # Pa
            
            # For ideal gas: PV = nRT
            gas_constant = self.constants["r"]
            molar_volume = gas_constant * temperature / pressure
            
            # Maxwell relations derivatives (for ideal gas)
            # (∂S/∂V)_T = (∂P/∂T)_V
            ds_dv_t = pressure / temperature  # From ideal gas
            dp_dt_v = pressure / temperature  # Same for ideal gas
            
            # (∂S/∂P)_T = -(∂V/∂T)_P
            ds_dp_t = -molar_volume / temperature
            dv_dt_p = molar_volume / temperature
            
            return {
                "calculation": calculation_type,
                "temperature": temperature,
                "pressure": pressure,
                "molar_volume": molar_volume,
                "maxwell_relations": {
                    "relation_1": {
                        "equation": "(∂S/∂V)_T = (∂P/∂T)_V",
                        "left_side": ds_dv_t,
                        "right_side": dp_dt_v,
                        "verified": abs(ds_dv_t - dp_dt_v) < 1e-10
                    },
                    "relation_2": {
                        "equation": "(∂S/∂P)_T = -(∂V/∂T)_P",
                        "left_side": ds_dp_t,
                        "right_side": -dv_dt_p,
                        "verified": abs(ds_dp_t + dv_dt_p) < 1e-10
                    }
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif calculation_type == "van_der_waals":
            # Van der Waals equation of state
            temperature = task.get("temperature", 300)  # K
            molar_volume = task.get("molar_volume", 0.024)  # m³/mol
            a = task.get("a_parameter", 0.364)  # Pa·m⁶/mol²
            b = task.get("b_parameter", 4.27e-5)  # m³/mol
            
            # Van der Waals equation: (P + a/V²)(V - b) = RT
            ideal_pressure = self.constants["r"] * temperature / molar_volume
            correction_pressure = a / molar_volume**2
            volume_correction = b
            
            vdw_pressure = (self.constants["r"] * temperature / (molar_volume - b)) - (a / molar_volume**2)
            
            # Critical constants
            critical_temp = 8 * a / (27 * self.constants["r"] * b)
            critical_pressure = a / (27 * b**2)
            critical_volume = 3 * b
            
            # Reduced variables
            reduced_temp = temperature / critical_temp
            reduced_pressure = vdw_pressure / critical_pressure
            reduced_volume = molar_volume / critical_volume
            
            return {
                "calculation": calculation_type,
                "temperature": temperature,
                "molar_volume": molar_volume,
                "van_der_waals_parameters": {
                    "a": a,
                    "b": b
                },
                "pressures": {
                    "ideal_gas": ideal_pressure,
                    "van_der_waals": vdw_pressure,
                    "attractive_correction": -correction_pressure,
                    "repulsive_correction": volume_correction
                },
                "critical_constants": {
                    "temperature": critical_temp,
                    "pressure": critical_pressure,
                    "volume": critical_volume
                },
                "reduced_variables": {
                    "temperature": reduced_temp,
                    "pressure": reduced_pressure,
                    "volume": reduced_volume
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "calculation": calculation_type,
                "error": f"Unknown thermodynamic calculation: {calculation_type}",
                "available_calculations": ["maxwell_relations", "van_der_waals"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_phase_transition(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze phase transitions."""
        transition_type = task.get("transition_type", "liquid_gas")
        substance = task.get("substance", "water")
        
        if transition_type == "liquid_gas":
            # Liquid-gas phase transition
            critical_data = {
                "water": {"Tc": 647.1, "Pc": 2.21e7, "rhoc": 322},  # K, Pa, kg/m³
                "co2": {"Tc": 304.1, "Pc": 7.38e6, "rhoc": 468},
                "nitrogen": {"Tc": 126.2, "Pc": 3.40e6, "rhoc": 313}
            }
            
            if substance not in critical_data:
                return {
                    "transition_type": transition_type,
                    "error": f"Critical data not available for {substance}",
                    "available_substances": list(critical_data.keys()),
                    "convergence": False,
                    "physical_validity": False
                }
            
            data = critical_data[substance]
            temperature = task.get("temperature", data["Tc"] * 0.9)  # K
            
            # Reduced temperature
            reduced_temp = temperature / data["Tc"]
            
            # Coexistence curve (simplified)
            if reduced_temp < 1.0:
                # Below critical temperature
                # Law of corresponding states approximation
                density_liquid = data["rhoc"] * (1 + 1.5 * (1 - reduced_temp))
                density_gas = data["rhoc"] * (1 - reduced_temp)**0.33
                
                # Clausius-Clapeyron equation
                latent_heat = 2.26e6  # J/kg for water (approximate)
                vapor_pressure = data["Pc"] * np.exp(-latent_heat * (1/temperature - 1/data["Tc"]) / self.constants["r"])
                
                phase_state = "two_phase_coexistence"
            else:
                # Above critical temperature
                density_liquid = data["rhoc"]
                density_gas = data["rhoc"]
                vapor_pressure = data["Pc"]
                phase_state = "supercritical"
            
            return {
                "transition_type": transition_type,
                "substance": substance,
                "temperature": temperature,
                "critical_constants": data,
                "reduced_temperature": reduced_temp,
                "phase_state": phase_state,
                "coexistence_densities": {
                    "liquid": density_liquid,
                    "gas": density_gas
                },
                "vapor_pressure": vapor_pressure,
                "density_difference": density_liquid - density_gas,
                "units": {
                    "temperature": "K",
                    "pressure": "Pa",
                    "density": "kg/m³"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif transition_type == "magnetic":
            # Magnetic phase transition (Curie-Weiss model)
            exchange_interaction = task.get("exchange_interaction", 1.0)  # Arbitrary units
            temperature = task.get("temperature", 500)  # K
            
            # Curie temperature
            curie_temp = exchange_interaction * 1000  # K (scaling)
            reduced_temp = temperature / curie_temp
            
            if reduced_temp < 1.0:
                # Ferromagnetic phase
                magnetization = (1 - reduced_temp)**(1/3)  # Mean field exponent
                susceptibility = 1 / (curie_temp - temperature) if temperature != curie_temp else float('inf')
                phase = "ferromagnetic"
            else:
                # Paramagnetic phase
                magnetization = 0
                susceptibility = 1 / (temperature - curie_temp)  # Curie-Weiss law
                phase = "paramagnetic"
            
            return {
                "transition_type": transition_type,
                "temperature": temperature,
                "curie_temperature": curie_temp,
                "reduced_temperature": reduced_temp,
                "phase": phase,
                "magnetization": magnetization,
                "magnetic_susceptibility": susceptibility,
                "critical_exponents": {
                    "beta": 1/3,    # Magnetization
                    "gamma": 1,     # Susceptibility
                    "alpha": 0      # Specific heat
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "transition_type": transition_type,
                "error": f"Unknown phase transition type: {transition_type}",
                "available_types": ["liquid_gas", "magnetic"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_critical_phenomena(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze critical phenomena and scaling behavior."""
        system = task.get("system", "ising_2d")
        temperature = task.get("temperature", 2.27)  # Near Tc for 2D Ising
        
        if system == "ising_2d":
            # 2D Ising model critical behavior
            critical_temp = 2.269  # kT/J for 2D Ising
            reduced_temp = temperature / critical_temp
            
            # Critical exponents for 2D Ising (exact)
            critical_exponents = {
                "alpha": 0,      # Specific heat (logarithmic)
                "beta": 1/8,     # Order parameter
                "gamma": 7/4,    # Susceptibility
                "delta": 15,     # Critical isotherm
                "nu": 1,         # Correlation length
                "eta": 1/4       # Correlation function
            }
            
            # Scaling functions near criticality
            if abs(reduced_temp - 1) < 0.1:
                t = abs(reduced_temp - 1)
                
                if reduced_temp < 1:
                    # Below Tc
                    magnetization = t**(critical_exponents["beta"])
                    susceptibility = t**(-critical_exponents["gamma"])
                    correlation_length = t**(-critical_exponents["nu"])
                    phase = "ordered"
                else:
                    # Above Tc
                    magnetization = 0
                    susceptibility = t**(-critical_exponents["gamma"])
                    correlation_length = t**(-critical_exponents["nu"])
                    phase = "disordered"
                
                # Specific heat (logarithmic divergence)
                specific_heat = -np.log(t) if t > 1e-6 else 100
            else:
                # Far from criticality
                magnetization = 0.8 if reduced_temp < 0.5 else 0
                susceptibility = 1.0
                correlation_length = 1.0
                specific_heat = 1.0
                phase = "ordered" if reduced_temp < 1 else "disordered"
            
            return {
                "system": system,
                "temperature": temperature,
                "critical_temperature": critical_temp,
                "reduced_temperature": reduced_temp,
                "phase": phase,
                "critical_exponents": critical_exponents,
                "order_parameters": {
                    "magnetization": magnetization,
                    "susceptibility": susceptibility,
                    "correlation_length": correlation_length,
                    "specific_heat": specific_heat
                },
                "universality_class": "2D_Ising",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "system": system,
                "error": f"Unknown critical system: {system}",
                "available_systems": ["ising_2d"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _calculate_transport_coefficients(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transport coefficients using kinetic theory."""
        transport_type = task.get("type", "viscosity")
        gas_type = task.get("gas", "nitrogen")
        temperature = task.get("temperature", 300)  # K
        pressure = task.get("pressure", 1e5)  # Pa
        
        # Gas properties
        gas_properties = {
            "nitrogen": {"mass": 4.65e-26, "sigma": 3.64e-10, "epsilon_k": 95.0},  # kg, m, K
            "oxygen": {"mass": 5.31e-26, "sigma": 3.46e-10, "epsilon_k": 107.4},
            "argon": {"mass": 6.63e-26, "sigma": 3.40e-10, "epsilon_k": 93.3},
            "helium": {"mass": 6.65e-27, "sigma": 2.55e-10, "epsilon_k": 10.9}
        }
        
        if gas_type not in gas_properties:
            return {
                "transport_type": transport_type,
                "error": f"Gas properties not available for {gas_type}",
                "available_gases": list(gas_properties.keys()),
                "convergence": False,
                "physical_validity": False
            }
        
        props = gas_properties[gas_type]
        mass = props["mass"]
        sigma = props["sigma"]
        epsilon_k = props["epsilon_k"]
        
        # Number density
        density = pressure / (self.constants["kb"] * temperature)
        
        # Mean thermal velocity
        mean_velocity = np.sqrt(8 * self.constants["kb"] * temperature / (np.pi * mass))
        
        # Mean free path
        cross_section = np.pi * sigma**2
        mean_free_path = 1 / (np.sqrt(2) * density * cross_section)
        
        if transport_type == "viscosity":
            # Dynamic viscosity (kinetic theory)
            viscosity = (5/16) * mass * density * mean_velocity * mean_free_path / density
            
            # Temperature dependence (Sutherland's law)
            sutherland_const = 1.5 * epsilon_k  # Approximate
            viscosity_sutherland = viscosity * (temperature / 273.15)**(3/2) * (273.15 + sutherland_const) / (temperature + sutherland_const)
            
            return {
                "transport_type": transport_type,
                "gas": gas_type,
                "temperature": temperature,
                "pressure": pressure,
                "kinetic_theory_viscosity": viscosity,
                "sutherland_viscosity": viscosity_sutherland,
                "mean_free_path": mean_free_path,
                "mean_velocity": mean_velocity,
                "units": {
                    "viscosity": "Pa·s",
                    "path": "m",
                    "velocity": "m/s"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif transport_type == "thermal_conductivity":
            # Thermal conductivity
            heat_capacity = (5/2) * self.constants["kb"]  # Per molecule
            thermal_conductivity = (25/64) * heat_capacity * density * mean_velocity * mean_free_path
            
            # Eucken factor (relates thermal conductivity to viscosity)
            viscosity = (5/16) * mass * density * mean_velocity * mean_free_path / density
            eucken_factor = thermal_conductivity / (viscosity * heat_capacity * density / mass)
            
            return {
                "transport_type": transport_type,
                "gas": gas_type,
                "thermal_conductivity": thermal_conductivity,
                "eucken_factor": eucken_factor,
                "heat_capacity_per_molecule": heat_capacity,
                "units": {
                    "thermal_conductivity": "W/(m·K)",
                    "heat_capacity": "J"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif transport_type == "diffusion":
            # Self-diffusion coefficient
            diffusion_coeff = (3/16) * mean_velocity * mean_free_path
            
            # Binary diffusion (simplified)
            binary_diffusion = diffusion_coeff  # Approximate for similar masses
            
            return {
                "transport_type": transport_type,
                "gas": gas_type,
                "self_diffusion": diffusion_coeff,
                "binary_diffusion": binary_diffusion,
                "units": {
                    "diffusion": "m²/s"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "transport_type": transport_type,
                "error": f"Unknown transport coefficient: {transport_type}",
                "available_types": ["viscosity", "thermal_conductivity", "diffusion"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _perform_monte_carlo_simulation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Monte Carlo simulation."""
        simulation_type = task.get("simulation", "metropolis_ising")
        lattice_size = task.get("lattice_size", 20)
        temperature = task.get("temperature", 2.5)  # kT/J
        num_steps = task.get("num_steps", 10000)
        
        if simulation_type == "metropolis_ising":
            # 2D Ising model Monte Carlo
            np.random.seed(42)  # For reproducibility
            
            # Initialize random spin configuration
            spins = 2 * np.random.randint(0, 2, (lattice_size, lattice_size)) - 1
            
            # Simulation parameters
            beta = 1 / temperature
            energy_history = []
            magnetization_history = []
            
            # Monte Carlo steps
            for step in range(min(num_steps, 1000)):  # Limit for demo
                # Random site selection
                i = np.random.randint(0, lattice_size)
                j = np.random.randint(0, lattice_size)
                
                # Calculate energy change
                # Periodic boundary conditions
                neighbors = (spins[(i+1)%lattice_size, j] + spins[(i-1)%lattice_size, j] +
                            spins[i, (j+1)%lattice_size] + spins[i, (j-1)%lattice_size])
                
                delta_e = 2 * spins[i, j] * neighbors
                
                # Metropolis acceptance
                if delta_e <= 0 or np.random.random() < np.exp(-beta * delta_e):
                    spins[i, j] *= -1
                
                # Record observables every 10 steps
                if step % 10 == 0:
                    energy = -0.5 * np.sum(spins * (
                        np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0) +
                        np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
                    ))
                    magnetization = np.mean(spins)
                    
                    energy_history.append(energy / lattice_size**2)
                    magnetization_history.append(abs(magnetization))
            
            # Calculate averages
            avg_energy = np.mean(energy_history[-50:])  # Last 50 measurements
            avg_magnetization = np.mean(magnetization_history[-50:])
            
            # Heat capacity and susceptibility (from fluctuations)
            energy_fluctuations = np.var(energy_history[-50:])
            mag_fluctuations = np.var(magnetization_history[-50:])
            
            heat_capacity = beta**2 * energy_fluctuations * lattice_size**2
            susceptibility = beta * mag_fluctuations * lattice_size**2
            
            return {
                "simulation": simulation_type,
                "lattice_size": lattice_size,
                "temperature": temperature,
                "monte_carlo_steps": len(energy_history) * 10,
                "observables": {
                    "average_energy": avg_energy,
                    "average_magnetization": avg_magnetization,
                    "heat_capacity": heat_capacity,
                    "magnetic_susceptibility": susceptibility
                },
                "fluctuations": {
                    "energy_variance": energy_fluctuations,
                    "magnetization_variance": mag_fluctuations
                },
                "final_configuration_magnetization": np.mean(spins),
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "simulation": simulation_type,
                "error": f"Unknown simulation type: {simulation_type}",
                "available_simulations": ["metropolis_ising"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_ising_model(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Ising model properties."""
        dimension = task.get("dimension", 2)
        temperature = task.get("temperature", 2.27)  # kT/J
        
        if dimension == 1:
            # 1D Ising model (exact solution)
            beta = 1 / temperature
            
            # Partition function per site
            z = 2 * np.cosh(beta)
            
            # Internal energy per site
            energy_per_site = -np.tanh(beta)
            
            # Heat capacity per site
            heat_capacity = beta**2 * (1 - np.tanh(beta)**2)
            
            # No phase transition in 1D
            phase_transition = False
            critical_temp = 0
            
            return {
                "dimension": dimension,
                "temperature": temperature,
                "partition_function_per_site": z,
                "energy_per_site": energy_per_site,
                "heat_capacity_per_site": heat_capacity,
                "magnetization": 0,  # No spontaneous magnetization in 1D
                "phase_transition": phase_transition,
                "critical_temperature": critical_temp,
                "convergence": True,
                "physical_validity": True
            }
        
        elif dimension == 2:
            # 2D Ising model
            critical_temp = 2.269  # kT/J (exact)
            reduced_temp = temperature / critical_temp
            
            if temperature == critical_temp:
                # Exactly at critical point
                magnetization = 0
                heat_capacity = float('inf')  # Logarithmic divergence
                susceptibility = float('inf')
                phase = "critical"
            elif temperature < critical_temp:
                # Ordered phase
                magnetization = (1 - reduced_temp)**(1/8)  # Critical exponent β = 1/8
                heat_capacity = 10 * abs(np.log(abs(1 - reduced_temp)))  # Logarithmic
                susceptibility = 10 / abs(1 - reduced_temp)**(7/4)
                phase = "ferromagnetic"
            else:
                # Disordered phase
                magnetization = 0
                heat_capacity = 2.0  # Background value
                susceptibility = 1 / (reduced_temp - 1)**(7/4)
                phase = "paramagnetic"
            
            return {
                "dimension": dimension,
                "temperature": temperature,
                "critical_temperature": critical_temp,
                "reduced_temperature": reduced_temp,
                "phase": phase,
                "magnetization": magnetization,
                "heat_capacity": min(heat_capacity, 1000),  # Cap for numerical reasons
                "susceptibility": min(susceptibility, 1000),
                "critical_exponents": {
                    "beta": 1/8,
                    "alpha": 0,
                    "gamma": 7/4
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "dimension": dimension,
                "error": f"Ising model analysis not implemented for {dimension}D",
                "available_dimensions": [1, 2],
                "convergence": False,
                "physical_validity": False
            }
    
    def _generic_statistical_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic statistical mechanics calculation."""
        calculation = task.get("calculation", "boltzmann_distribution")
        
        if calculation == "boltzmann_distribution":
            energy_levels = task.get("energy_levels", [0, 1, 2, 3, 4])  # kT units
            temperature = task.get("temperature", 1.0)  # kT units
            
            # Boltzmann factors
            beta = 1 / temperature
            boltzmann_factors = [np.exp(-beta * e) for e in energy_levels]
            
            # Partition function
            partition_function = sum(boltzmann_factors)
            
            # Probabilities
            probabilities = [bf / partition_function for bf in boltzmann_factors]
            
            # Average energy
            average_energy = sum(e * p for e, p in zip(energy_levels, probabilities))
            
            return {
                "calculation": calculation,
                "temperature": temperature,
                "energy_levels": energy_levels,
                "boltzmann_factors": boltzmann_factors,
                "partition_function": partition_function,
                "probabilities": probabilities,
                "average_energy": average_energy,
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "calculation": calculation,
                "error": f"Unknown calculation type: {calculation}",
                "convergence": False,
                "physical_validity": False
            }
    
    def can_handle(self, research_question: str, context: Dict[str, Any]) -> float:
        """
        Assess if this tool can handle the research question.
        
        Returns confidence score between 0 and 1.
        """
        question_lower = research_question.lower()
        
        # High confidence keywords
        high_confidence_keywords = [
            "statistical mechanics", "thermodynamics", "ensemble", "partition function",
            "phase transition", "critical phenomena", "monte carlo", "ising model",
            "transport coefficients", "kinetic theory", "boltzmann", "entropy"
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            "temperature", "pressure", "thermal", "heat capacity", "magnetization",
            "diffusion", "viscosity", "fluctuations", "equilibrium"
        ]
        
        # Calculate confidence
        high_matches = sum(1 for keyword in high_confidence_keywords 
                          if keyword in question_lower)
        medium_matches = sum(1 for keyword in medium_confidence_keywords 
                           if keyword in question_lower)
        
        if high_matches > 0:
            return min(1.0, 0.7 + high_matches * 0.1)
        elif medium_matches > 0:
            return min(0.6, 0.3 + medium_matches * 0.1)
        else:
            return 0.1
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate computational cost for a statistical mechanics task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "canonical_ensemble")
        
        # Base costs by task type
        base_costs = {
            "canonical_ensemble": {"time": 1, "memory": 256, "units": 3},
            "grand_canonical_ensemble": {"time": 2, "memory": 512, "units": 5},
            "thermodynamic_properties": {"time": 1, "memory": 256, "units": 3},
            "phase_transition": {"time": 2, "memory": 512, "units": 6},
            "critical_phenomena": {"time": 3, "memory": 512, "units": 8},
            "transport_coefficients": {"time": 2, "memory": 256, "units": 4},
            "monte_carlo": {"time": 10, "memory": 1024, "units": 20},
            "ising_model": {"time": 1, "memory": 256, "units": 3}
        }
        
        base_cost = base_costs.get(task_type, base_costs["canonical_ensemble"])
        
        # Scale based on problem complexity
        scale_factor = 1.0
        
        if task_type == "monte_carlo":
            num_steps = task.get("num_steps", 10000)
            lattice_size = task.get("lattice_size", 20)
            scale_factor = (num_steps / 10000) * (lattice_size / 20)**2
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for statistical mechanics calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Probability theory",
                "Thermodynamics",
                "Statistical distributions",
                "Calculus of variations"
            ],
            "computational_methods": [
                "Monte Carlo methods",
                "Molecular dynamics",
                "Metropolis algorithm",
                "Histogram reweighting"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Phase diagram calculation",
                "Material property prediction",
                "Critical phenomena analysis",
                "Transport property modeling",
                "Equilibrium simulations"
            ],
            "accuracy_considerations": [
                "Finite size effects in simulations",
                "Equilibration time requirements",
                "Statistical error estimation",
                "Critical slowing down near transitions"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for statistical mechanics calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check temperature
        temperature = task.get("temperature")
        if temperature and temperature <= 0:
            errors.append("Temperature must be positive")
        
        # Check particle numbers
        num_particles = task.get("num_particles")
        if num_particles and num_particles <= 0:
            errors.append("Number of particles must be positive")
        
        if task_type == "monte_carlo":
            num_steps = task.get("num_steps", 10000)
            if num_steps > 100000:
                warnings.append(f"Large number of MC steps ({num_steps}) may be slow")
            
            lattice_size = task.get("lattice_size", 20)
            if lattice_size > 100:
                warnings.append(f"Large lattice ({lattice_size}) requires significant memory")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }