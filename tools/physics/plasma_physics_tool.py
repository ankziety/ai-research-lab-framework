"""
Plasma Physics Tool

Agent-friendly interface for plasma physics calculations.
Provides plasma parameter analysis, magnetohydrodynamics,
fusion plasma modeling, and space plasma simulations.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class PlasmaPhysicsTool(BasePhysicsTool):
    """
    Tool for plasma physics calculations that agents can request.
    
    Provides interfaces for:
    - Plasma parameter calculations
    - Magnetohydrodynamics (MHD)
    - Fusion plasma analysis
    - Space plasma physics
    - Plasma instabilities
    - Wave propagation in plasmas
    """
    
    def __init__(self):
        super().__init__(
            tool_id="plasma_physics_tool",
            name="Plasma Physics Tool",
            description="Perform plasma physics calculations including MHD, fusion plasmas, and space plasma analysis",
            physics_domain="plasma_physics",
            computational_cost_factor=3.5,
            software_requirements=[
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib",   # Visualization
                "sympy"         # Symbolic mathematics (optional)
            ],
            hardware_requirements={
                "min_memory": 2048,  # MB
                "recommended_memory": 8192,
                "cpu_cores": 4,
                "supports_gpu": True
            }
        )
        
        self.capabilities.extend([
            "plasma_parameters",
            "magnetohydrodynamics",
            "fusion_plasma",
            "space_plasma",
            "plasma_instabilities",
            "wave_propagation",
            "plasma_confinement",
            "tokamak_analysis"
        ])
        
        # Plasma physics constants
        self.constants = {
            "e": 1.602e-19,     # C
            "me": 9.109e-31,    # kg
            "mp": 1.673e-27,    # kg
            "epsilon0": 8.854e-12,  # F/m
            "kb": 1.381e-23,    # J/K
            "mu0": 4*np.pi*1e-7,  # H/m
            "c": 2.998e8        # m/s
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plasma physics calculation.
        
        Args:
            task: Task specification with plasma physics parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with plasma physics analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "plasma_parameters")
            
            if task_type == "plasma_parameters":
                result = self._calculate_plasma_parameters(task)
            elif task_type == "magnetohydrodynamics":
                result = self._analyze_mhd(task)
            elif task_type == "fusion_plasma":
                result = self._analyze_fusion_plasma(task)
            elif task_type == "space_plasma":
                result = self._analyze_space_plasma(task)
            elif task_type == "plasma_waves":
                result = self._analyze_plasma_waves(task)
            elif task_type == "plasma_instabilities":
                result = self._analyze_plasma_instabilities(task)
            elif task_type == "tokamak":
                result = self._analyze_tokamak(task)
            else:
                result = self._generic_plasma_calculation(task)
            
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
            logger.error(f"Plasma physics calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _calculate_plasma_parameters(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fundamental plasma parameters."""
        density = task.get("electron_density", 1e19)  # m^-3
        temperature = task.get("temperature", 1000)   # eV
        magnetic_field = task.get("magnetic_field", 1.0)  # T
        
        # Convert temperature to Kelvin
        temp_kelvin = temperature * 11604.5  # eV to K
        
        # Debye length
        debye_length = np.sqrt(self.constants["epsilon0"] * temperature * self.constants["e"] / 
                              (density * self.constants["e"]**2))
        
        # Plasma frequency
        plasma_frequency = np.sqrt(density * self.constants["e"]**2 / 
                                 (self.constants["epsilon0"] * self.constants["me"]))
        
        # Cyclotron frequency
        cyclotron_freq_e = self.constants["e"] * magnetic_field / self.constants["me"]
        cyclotron_freq_i = self.constants["e"] * magnetic_field / self.constants["mp"]
        
        # Thermal velocity
        thermal_vel_e = np.sqrt(2 * temperature * self.constants["e"] / self.constants["me"])
        thermal_vel_i = np.sqrt(2 * temperature * self.constants["e"] / self.constants["mp"])
        
        # Larmor radius (gyroradius)
        larmor_radius_e = thermal_vel_e / cyclotron_freq_e
        larmor_radius_i = thermal_vel_i / cyclotron_freq_i
        
        # Number of particles in Debye sphere
        debye_number = (4/3) * np.pi * density * debye_length**3
        
        # Plasma beta
        pressure = density * self.constants["kb"] * temp_kelvin
        magnetic_pressure = magnetic_field**2 / (2 * self.constants["mu0"])
        beta = pressure / magnetic_pressure
        
        # Collision frequency (rough estimate)
        coulomb_log = 15  # Typical value
        collision_freq = 2.9e-12 * density * coulomb_log / (temperature * self.constants["e"])**(3/2)
        
        # Plasma classification
        if debye_number > 100:
            plasma_validity = "valid_plasma"
        else:
            plasma_validity = "weakly_coupled"
        
        if cyclotron_freq_e > plasma_frequency:
            magnetization = "strongly_magnetized"
        elif cyclotron_freq_e > collision_freq:
            magnetization = "weakly_magnetized"
        else:
            magnetization = "unmagnetized"
        
        return {
            "input_parameters": {
                "electron_density": density,
                "temperature": temperature,
                "magnetic_field": magnetic_field
            },
            "characteristic_lengths": {
                "debye_length": debye_length,
                "larmor_radius_electron": larmor_radius_e,
                "larmor_radius_ion": larmor_radius_i
            },
            "characteristic_frequencies": {
                "plasma_frequency": plasma_frequency,
                "cyclotron_frequency_electron": cyclotron_freq_e,
                "cyclotron_frequency_ion": cyclotron_freq_i,
                "collision_frequency": collision_freq
            },
            "thermal_properties": {
                "thermal_velocity_electron": thermal_vel_e,
                "thermal_velocity_ion": thermal_vel_i,
                "temperature_kelvin": temp_kelvin
            },
            "plasma_properties": {
                "debye_number": debye_number,
                "plasma_beta": beta,
                "plasma_validity": plasma_validity,
                "magnetization": magnetization
            },
            "units": {
                "length": "m",
                "frequency": "Hz",
                "velocity": "m/s",
                "density": "m^-3",
                "temperature": "eV"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_mhd(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze magnetohydrodynamic properties."""
        density = task.get("mass_density", 1e-12)  # kg/m^3
        magnetic_field = task.get("magnetic_field", 1e-4)  # T
        velocity = task.get("velocity", 1000)  # m/s
        pressure = task.get("pressure", 1e-9)  # Pa
        
        # Alfvén velocity
        alfven_velocity = magnetic_field / np.sqrt(self.constants["mu0"] * density)
        
        # Sound speed
        gamma = 5/3  # Adiabatic index
        sound_speed = np.sqrt(gamma * pressure / density)
        
        # Magnetosonic speeds
        va2 = alfven_velocity**2
        cs2 = sound_speed**2
        
        # Fast magnetosonic speed
        fast_speed = np.sqrt(0.5 * (va2 + cs2 + np.sqrt((va2 + cs2)**2 - 4*va2*cs2*np.cos(np.pi/4)**2)))
        
        # Slow magnetosonic speed
        slow_speed = np.sqrt(0.5 * (va2 + cs2 - np.sqrt((va2 + cs2)**2 - 4*va2*cs2*np.cos(np.pi/4)**2)))
        
        # Mach numbers
        alfven_mach = velocity / alfven_velocity
        sonic_mach = velocity / sound_speed
        fast_mach = velocity / fast_speed
        
        # Magnetic Reynolds number
        conductivity = task.get("conductivity", 1e7)  # S/m
        length_scale = task.get("length_scale", 1e6)  # m
        magnetic_reynolds = self.constants["mu0"] * conductivity * velocity * length_scale
        
        # Beta plasma
        magnetic_pressure = magnetic_field**2 / (2 * self.constants["mu0"])
        beta = pressure / magnetic_pressure
        
        # MHD regime classification
        if magnetic_reynolds > 1:
            mhd_regime = "ideal_mhd"
        else:
            mhd_regime = "resistive_mhd"
        
        if beta > 1:
            pressure_regime = "gas_pressure_dominated"
        else:
            pressure_regime = "magnetic_pressure_dominated"
        
        return {
            "input_parameters": {
                "mass_density": density,
                "magnetic_field": magnetic_field,
                "velocity": velocity,
                "pressure": pressure,
                "length_scale": length_scale
            },
            "characteristic_speeds": {
                "alfven_velocity": alfven_velocity,
                "sound_speed": sound_speed,
                "fast_magnetosonic": fast_speed,
                "slow_magnetosonic": slow_speed
            },
            "mach_numbers": {
                "alfven_mach": alfven_mach,
                "sonic_mach": sonic_mach,
                "fast_mach": fast_mach
            },
            "dimensionless_parameters": {
                "magnetic_reynolds": magnetic_reynolds,
                "plasma_beta": beta
            },
            "regime_classification": {
                "mhd_regime": mhd_regime,
                "pressure_regime": pressure_regime
            },
            "units": {
                "velocity": "m/s",
                "density": "kg/m^3",
                "pressure": "Pa",
                "magnetic_field": "T"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_fusion_plasma(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fusion plasma properties."""
        density = task.get("ion_density", 1e20)  # m^-3
        temperature = task.get("temperature", 10000)  # eV (10 keV)
        magnetic_field = task.get("magnetic_field", 5.0)  # T
        major_radius = task.get("major_radius", 6.2)  # m (ITER-like)
        minor_radius = task.get("minor_radius", 2.0)  # m
        
        # Fusion reaction rate (D-T)
        # Parametric fit for D-T reaction rate
        temp_kev = temperature / 1000
        if temp_kev > 1:
            sigma_v = 1.1e-24 * temp_kev**(-2/3) * np.exp(-19.94 * temp_kev**(-1/3))  # m^3/s
        else:
            sigma_v = 1e-30  # Very low at low temperatures
        
        fusion_rate = 0.25 * density**2 * sigma_v  # reactions/m^3/s (D-T with 50% each)
        fusion_power_density = fusion_rate * 17.6 * 1.602e-19 * 1e6  # MW/m^3
        
        # Confinement properties
        # Energy confinement time (empirical scaling)
        plasma_current = task.get("plasma_current", 15e6)  # A
        heating_power = task.get("heating_power", 50)  # MW
        
        # ITER H-mode scaling
        mass_number = 2.5  # D-T mixture
        confinement_time = 0.145 * (plasma_current/1e6)**0.93 * (magnetic_field)**0.15 * \
                          (heating_power)**(-0.69) * (density/1e19)**0.41 * \
                          (major_radius)**2.58 * (minor_radius/major_radius)**0.58 * \
                          mass_number**0.19
        
        # Triple product
        triple_product = density * temperature * confinement_time
        
        # Lawson criterion
        lawson_criterion = 3e21  # m^-3 * keV * s for D-T
        lawson_ratio = triple_product / lawson_criterion
        
        # Q factor (fusion gain)
        total_fusion_power = fusion_power_density * np.pi**2 * major_radius * minor_radius**2
        q_factor = total_fusion_power / heating_power if heating_power > 0 else 0
        
        # Plasma pressure and beta
        pressure = density * self.constants["kb"] * temperature * self.constants["e"] / self.constants["kb"]
        magnetic_pressure = magnetic_field**2 / (2 * self.constants["mu0"])
        beta = pressure / magnetic_pressure
        beta_limit = 0.028  # Troyon limit
        
        # Bootstrap current fraction
        bootstrap_fraction = 0.4 * np.sqrt(beta)  # Approximate
        
        return {
            "plasma_parameters": {
                "ion_density": density,
                "temperature": temperature,
                "magnetic_field": magnetic_field,
                "major_radius": major_radius,
                "minor_radius": minor_radius
            },
            "fusion_performance": {
                "fusion_reaction_rate": sigma_v,
                "fusion_power_density": fusion_power_density,
                "total_fusion_power": total_fusion_power,
                "q_factor": q_factor
            },
            "confinement": {
                "energy_confinement_time": confinement_time,
                "triple_product": triple_product,
                "lawson_criterion_ratio": lawson_ratio,
                "ignition_feasible": lawson_ratio > 1
            },
            "stability": {
                "plasma_beta": beta,
                "beta_limit": beta_limit,
                "beta_normalized": beta / beta_limit,
                "bootstrap_current_fraction": bootstrap_fraction
            },
            "units": {
                "density": "m^-3",
                "temperature": "eV",
                "power": "MW",
                "time": "s",
                "triple_product": "m^-3 * eV * s"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_space_plasma(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze space plasma environments."""
        environment = task.get("environment", "solar_wind")
        
        if environment == "solar_wind":
            # Typical solar wind parameters at 1 AU
            density = task.get("density", 5e6)  # m^-3
            temperature = task.get("temperature", 10)  # eV
            velocity = task.get("velocity", 400e3)  # m/s
            magnetic_field = task.get("magnetic_field", 5e-9)  # T (5 nT)
            
            # Solar wind properties
            thermal_speed = np.sqrt(2 * temperature * self.constants["e"] / self.constants["mp"])
            mach_number = velocity / thermal_speed
            
            # Magnetic field properties
            alfven_speed = magnetic_field / np.sqrt(self.constants["mu0"] * density * self.constants["mp"])
            alfven_mach = velocity / alfven_speed
            
            # Plasma beta
            pressure = density * self.constants["kb"] * temperature * 11604.5  # Convert eV to K
            magnetic_pressure = magnetic_field**2 / (2 * self.constants["mu0"])
            beta = pressure / magnetic_pressure
            
            # Classification
            if mach_number > 3:
                flow_type = "supersonic"
            else:
                flow_type = "subsonic"
            
            if alfven_mach > 1:
                magnetic_flow = "super_alfvenic"
            else:
                magnetic_flow = "sub_alfvenic"
            
            return {
                "environment": environment,
                "plasma_parameters": {
                    "density": density,
                    "temperature": temperature,
                    "velocity": velocity,
                    "magnetic_field": magnetic_field
                },
                "characteristic_speeds": {
                    "thermal_speed": thermal_speed,
                    "alfven_speed": alfven_speed
                },
                "flow_properties": {
                    "mach_number": mach_number,
                    "alfven_mach_number": alfven_mach,
                    "flow_classification": flow_type,
                    "magnetic_flow_classification": magnetic_flow
                },
                "plasma_properties": {
                    "plasma_beta": beta,
                    "kinetic_energy_density": 0.5 * density * self.constants["mp"] * velocity**2,
                    "magnetic_energy_density": magnetic_field**2 / (2 * self.constants["mu0"])
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif environment == "magnetosphere":
            # Earth's magnetosphere
            density = task.get("density", 1e6)  # m^-3 (plasmasphere)
            temperature = task.get("temperature", 1)  # eV
            magnetic_field = task.get("magnetic_field", 50e-6)  # T (50 μT)
            
            # Magnetospheric parameters
            plasma_frequency = np.sqrt(density * self.constants["e"]**2 / 
                                     (self.constants["epsilon0"] * self.constants["me"]))
            cyclotron_frequency = self.constants["e"] * magnetic_field / self.constants["me"]
            
            # Drift velocities
            grad_b_drift = 1000  # m/s (typical)
            curvature_drift = 500  # m/s (typical)
            
            return {
                "environment": environment,
                "plasma_parameters": {
                    "density": density,
                    "temperature": temperature,
                    "magnetic_field": magnetic_field
                },
                "wave_properties": {
                    "plasma_frequency": plasma_frequency,
                    "cyclotron_frequency": cyclotron_frequency,
                    "upper_hybrid_frequency": np.sqrt(plasma_frequency**2 + cyclotron_frequency**2)
                },
                "particle_drifts": {
                    "gradient_b_drift": grad_b_drift,
                    "curvature_drift": curvature_drift,
                    "total_drift": grad_b_drift + curvature_drift
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "environment": environment,
                "error": f"Unknown space environment: {environment}",
                "available_environments": ["solar_wind", "magnetosphere"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_plasma_waves(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze wave propagation in plasmas."""
        wave_type = task.get("wave_type", "langmuir")
        density = task.get("electron_density", 1e18)  # m^-3
        magnetic_field = task.get("magnetic_field", 0.1)  # T
        frequency = task.get("frequency", None)  # Hz
        
        # Plasma frequency
        plasma_freq = np.sqrt(density * self.constants["e"]**2 / 
                             (self.constants["epsilon0"] * self.constants["me"]))
        
        # Cyclotron frequency
        cyclotron_freq = self.constants["e"] * magnetic_field / self.constants["me"]
        
        if wave_type == "langmuir":
            # Langmuir waves (electrostatic)
            wave_frequency = plasma_freq
            thermal_velocity = np.sqrt(self.constants["kb"] * 1000 * 11604.5 / self.constants["me"])  # Assume 1 keV
            phase_velocity = thermal_velocity * 1.5  # Approximate
            wavelength = phase_velocity / wave_frequency
            
            return {
                "wave_type": wave_type,
                "wave_frequency": wave_frequency,
                "plasma_frequency": plasma_freq,
                "phase_velocity": phase_velocity,
                "wavelength": wavelength,
                "wave_nature": "electrostatic",
                "propagation": "longitudinal",
                "convergence": True,
                "physical_validity": True
            }
        
        elif wave_type == "alfven":
            # Alfvén waves
            mass_density = density * self.constants["mp"]  # Assuming hydrogen
            alfven_velocity = magnetic_field / np.sqrt(self.constants["mu0"] * mass_density)
            
            if frequency:
                wavelength = alfven_velocity / frequency
            else:
                frequency = alfven_velocity / 1e6  # Assume 1 Mm wavelength
                wavelength = 1e6
            
            return {
                "wave_type": wave_type,
                "wave_frequency": frequency,
                "alfven_velocity": alfven_velocity,
                "wavelength": wavelength,
                "wave_nature": "magnetohydrodynamic",
                "propagation": "transverse",
                "convergence": True,
                "physical_validity": True
            }
        
        elif wave_type == "whistler":
            # Whistler waves
            if frequency and frequency < cyclotron_freq:
                # Whistler mode condition
                k_perp = frequency / (cyclotron_freq * self.constants["c"])  # Approximate
                group_velocity = 2 * frequency / k_perp
                
                return {
                    "wave_type": wave_type,
                    "wave_frequency": frequency,
                    "cyclotron_frequency": cyclotron_freq,
                    "group_velocity": group_velocity,
                    "wave_nature": "electromagnetic",
                    "propagation": "right_hand_polarized",
                    "convergence": True,
                    "physical_validity": True
                }
            else:
                return {
                    "wave_type": wave_type,
                    "error": "Frequency must be less than cyclotron frequency for whistler mode",
                    "cyclotron_frequency": cyclotron_freq,
                    "convergence": False,
                    "physical_validity": False
                }
        
        else:
            return {
                "wave_type": wave_type,
                "error": f"Unknown wave type: {wave_type}",
                "available_types": ["langmuir", "alfven", "whistler"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_plasma_instabilities(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze plasma instabilities."""
        instability_type = task.get("instability_type", "two_stream")
        
        if instability_type == "two_stream":
            # Two-stream instability
            beam_density = task.get("beam_density", 1e17)  # m^-3
            background_density = task.get("background_density", 1e18)  # m^-3
            beam_velocity = task.get("beam_velocity", 1e7)  # m/s
            
            # Plasma frequencies
            beam_plasma_freq = np.sqrt(beam_density * self.constants["e"]**2 / 
                                     (self.constants["epsilon0"] * self.constants["me"]))
            background_plasma_freq = np.sqrt(background_density * self.constants["e"]**2 / 
                                           (self.constants["epsilon0"] * self.constants["me"]))
            
            # Growth rate (simplified)
            density_ratio = beam_density / background_density
            growth_rate = 0.5 * beam_plasma_freq * (density_ratio)**(1/3)
            
            # Threshold velocity
            thermal_velocity = 1e6  # m/s (assume 1 eV temperature)
            threshold_velocity = 3 * thermal_velocity
            
            unstable = beam_velocity > threshold_velocity
            
            return {
                "instability_type": instability_type,
                "parameters": {
                    "beam_density": beam_density,
                    "background_density": background_density,
                    "beam_velocity": beam_velocity,
                    "density_ratio": density_ratio
                },
                "growth_properties": {
                    "growth_rate": growth_rate,
                    "growth_time": 1/growth_rate,
                    "threshold_velocity": threshold_velocity,
                    "unstable": unstable
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif instability_type == "kink_mode":
            # Kink mode instability in tokamaks
            plasma_current = task.get("plasma_current", 1e7)  # A
            toroidal_field = task.get("toroidal_field", 5.0)  # T
            minor_radius = task.get("minor_radius", 1.0)  # m
            
            # Safety factor
            q_factor = 5 * toroidal_field * minor_radius**2 / (2 * plasma_current * 1e-7)
            
            # Kink mode threshold
            q_threshold = 2.0  # For m=2, n=1 mode
            unstable = q_factor < q_threshold
            
            if unstable:
                growth_rate = 1000  # s^-1 (typical)
            else:
                growth_rate = 0
            
            return {
                "instability_type": instability_type,
                "parameters": {
                    "plasma_current": plasma_current,
                    "toroidal_field": toroidal_field,
                    "minor_radius": minor_radius
                },
                "stability_analysis": {
                    "safety_factor": q_factor,
                    "threshold_q": q_threshold,
                    "unstable": unstable,
                    "growth_rate": growth_rate
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "instability_type": instability_type,
                "error": f"Unknown instability type: {instability_type}",
                "available_types": ["two_stream", "kink_mode"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_tokamak(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tokamak plasma equilibrium and performance."""
        major_radius = task.get("major_radius", 6.2)  # m
        minor_radius = task.get("minor_radius", 2.0)  # m
        toroidal_field = task.get("toroidal_field", 5.3)  # T
        plasma_current = task.get("plasma_current", 15e6)  # A
        density = task.get("density", 1e20)  # m^-3
        temperature = task.get("temperature", 10000)  # eV
        
        # Geometric parameters
        aspect_ratio = major_radius / minor_radius
        elongation = task.get("elongation", 1.7)
        triangularity = task.get("triangularity", 0.33)
        
        # Safety factor
        q95 = 5 * toroidal_field * minor_radius**2 / (2 * major_radius * plasma_current * 1e-7)
        
        # Plasma volume and surface area
        volume = 2 * np.pi**2 * major_radius * minor_radius**2 * elongation
        surface_area = 4 * np.pi**2 * major_radius * minor_radius * elongation
        
        # Beta values
        pressure = density * self.constants["kb"] * temperature * 11604.5
        magnetic_pressure = toroidal_field**2 / (2 * self.constants["mu0"])
        beta_toroidal = pressure / magnetic_pressure
        
        # Troyon beta limit
        beta_n_limit = 2.8  # %·m·T/MA
        beta_limit = beta_n_limit * plasma_current/1e6 * toroidal_field / 100
        
        # Confinement scaling (IPB98(y,2))
        heating_power = task.get("heating_power", 50)  # MW
        mass_number = 2.5  # D-T
        
        tau_e = 0.0562 * (plasma_current/1e6)**0.93 * toroidal_field**0.15 * \
                heating_power**(-0.69) * (density/1e19)**0.41 * major_radius**1.97 * \
                minor_radius**0.58 * elongation**0.78 * mass_number**0.19
        
        # Fusion performance
        sigma_v = 1.1e-24 * (temperature/1000)**(-2/3) * np.exp(-19.94 * (temperature/1000)**(-1/3))
        fusion_power_density = 0.25 * density**2 * sigma_v * 17.6 * 1.602e-19 * 1e6  # MW/m^3
        total_fusion_power = fusion_power_density * volume
        
        # Q factor
        q_factor = total_fusion_power / heating_power if heating_power > 0 else 0
        
        return {
            "geometry": {
                "major_radius": major_radius,
                "minor_radius": minor_radius,
                "aspect_ratio": aspect_ratio,
                "elongation": elongation,
                "triangularity": triangularity,
                "volume": volume,
                "surface_area": surface_area
            },
            "magnetic_configuration": {
                "toroidal_field": toroidal_field,
                "plasma_current": plasma_current,
                "safety_factor_95": q95
            },
            "plasma_parameters": {
                "density": density,
                "temperature": temperature,
                "pressure": pressure
            },
            "stability_limits": {
                "beta_toroidal": beta_toroidal,
                "beta_limit": beta_limit,
                "beta_normalized": beta_toroidal / beta_limit
            },
            "confinement": {
                "energy_confinement_time": tau_e,
                "heating_power": heating_power
            },
            "fusion_performance": {
                "fusion_power_density": fusion_power_density,
                "total_fusion_power": total_fusion_power,
                "q_factor": q_factor
            },
            "units": {
                "length": "m",
                "current": "A",
                "field": "T",
                "power": "MW",
                "density": "m^-3",
                "temperature": "eV"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _generic_plasma_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic plasma physics calculation."""
        calculation = task.get("calculation", "debye_length")
        
        if calculation == "debye_length":
            density = task.get("density", 1e18)  # m^-3
            temperature = task.get("temperature", 100)  # eV
            
            debye_length = np.sqrt(self.constants["epsilon0"] * temperature * self.constants["e"] / 
                                  (density * self.constants["e"]**2))
            
            return {
                "calculation": calculation,
                "density": density,
                "temperature": temperature,
                "debye_length": debye_length,
                "units": {
                    "length": "m",
                    "density": "m^-3",
                    "temperature": "eV"
                },
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
            "plasma", "fusion", "tokamak", "magnetohydrodynamics", "mhd",
            "solar wind", "magnetosphere", "plasma waves", "instability",
            "alfven", "langmuir", "cyclotron", "debye", "plasma physics"
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            "magnetic field", "ionized", "electromagnetic", "space physics",
            "conductivity", "magnetic confinement", "beta", "equilibrium"
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
        Estimate computational cost for a plasma physics task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "plasma_parameters")
        
        # Base costs by task type
        base_costs = {
            "plasma_parameters": {"time": 1, "memory": 256, "units": 3},
            "magnetohydrodynamics": {"time": 3, "memory": 512, "units": 8},
            "fusion_plasma": {"time": 5, "memory": 1024, "units": 15},
            "space_plasma": {"time": 2, "memory": 512, "units": 5},
            "plasma_waves": {"time": 2, "memory": 512, "units": 6},
            "plasma_instabilities": {"time": 4, "memory": 1024, "units": 12},
            "tokamak": {"time": 6, "memory": 1024, "units": 18}
        }
        
        base_cost = base_costs.get(task_type, base_costs["plasma_parameters"])
        
        # Scale based on problem complexity
        scale_factor = 1.0
        
        if task_type == "tokamak":
            # More complex geometry increases cost
            aspect_ratio = task.get("major_radius", 6) / task.get("minor_radius", 2)
            if aspect_ratio < 2.5:
                scale_factor = 1.5  # Low aspect ratio is more complex
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for plasma physics calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Electromagnetism",
                "Fluid mechanics",
                "Statistical mechanics",
                "Kinetic theory"
            ],
            "computational_methods": [
                "Magnetohydrodynamics simulations",
                "Particle-in-cell methods",
                "Gyrokinetic calculations",
                "Stability analysis"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Fusion energy research",
                "Space weather modeling",
                "Plasma processing",
                "Astrophysical plasmas",
                "Laboratory plasma experiments"
            ],
            "accuracy_considerations": [
                "Multi-scale physics important",
                "Nonlinear effects significant",
                "Boundary conditions critical",
                "Kinetic vs fluid approximations"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for plasma physics calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check for negative densities
        density = task.get("electron_density") or task.get("density") or task.get("ion_density")
        if density and density <= 0:
            errors.append("Plasma density must be positive")
        
        # Check temperature
        temperature = task.get("temperature")
        if temperature and temperature <= 0:
            errors.append("Temperature must be positive")
        
        # Check magnetic field
        magnetic_field = task.get("magnetic_field")
        if magnetic_field and magnetic_field < 0:
            errors.append("Magnetic field magnitude must be non-negative")
        
        if task_type == "tokamak":
            major_radius = task.get("major_radius")
            minor_radius = task.get("minor_radius")
            if major_radius and minor_radius:
                aspect_ratio = major_radius / minor_radius
                if aspect_ratio < 1.5:
                    warnings.append("Very low aspect ratio may not be realistic for tokamak")
                if aspect_ratio > 10:
                    warnings.append("Very high aspect ratio may have poor confinement")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }