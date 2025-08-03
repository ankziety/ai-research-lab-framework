"""
Astrophysics Tool

Agent-friendly interface for astrophysics simulations.
Provides stellar evolution, cosmological calculations,
orbital mechanics, and astronomical data analysis.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime
import math

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class AstrophysicsTool(BasePhysicsTool):
    """
    Tool for astrophysics calculations that agents can request.
    
    Provides interfaces for:
    - Stellar evolution modeling
    - Cosmological calculations
    - Orbital mechanics
    - Galaxy dynamics
    - Black hole physics
    """
    
    def __init__(self):
        super().__init__(
            tool_id="astrophysics_tool",
            name="Astrophysics Tool",
            description="Perform astrophysics simulations including stellar evolution, cosmology, and orbital mechanics",
            physics_domain="astrophysics",
            computational_cost_factor=2.0,
            software_requirements=[
                "astropy",      # Astronomy calculations (optional)
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib"    # Visualization
            ],
            hardware_requirements={
                "min_memory": 512,   # MB
                "recommended_memory": 2048,
                "cpu_cores": 2,
                "supports_gpu": False
            }
        )
        
        # Add astrophysics specific capabilities
        self.capabilities.extend([
            "stellar_evolution",
            "cosmological_calculations",
            "orbital_mechanics",
            "galaxy_dynamics",
            "black_hole_physics",
            "exoplanet_analysis",
            "supernova_modeling",
            "neutron_star_physics"
        ])
        
        # Available calculation types
        self.calculation_types = [
            "stellar_evolution",
            "orbital_dynamics",
            "cosmological_distance",
            "galaxy_rotation",
            "black_hole_properties",
            "exoplanet_detection",
            "supernova_lightcurve",
            "gravitational_waves"
        ]
        
        # Physical constants (in SI units unless noted)
        self.constants = {
            "G": 6.67430e-11,           # Gravitational constant (m³/kg·s²)
            "c": 299792458,             # Speed of light (m/s)
            "h": 6.62607015e-34,        # Planck constant (J·s)
            "k_B": 1.380649e-23,        # Boltzmann constant (J/K)
            "sigma_SB": 5.670374419e-8, # Stefan-Boltzmann constant (W/m²·K⁴)
            "M_sun": 1.9891e30,         # Solar mass (kg)
            "R_sun": 6.96e8,            # Solar radius (m)
            "L_sun": 3.828e26,          # Solar luminosity (W)
            "pc": 3.0857e16,            # Parsec (m)
            "AU": 1.496e11,             # Astronomical unit (m)
            "year": 3.156e7             # Year (s)
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute astrophysics calculation requested by an agent.
        
        Args:
            task: Task specification with astrophysics parameters
            context: Agent context and execution environment
            
        Returns:
            Astrophysics results formatted for agents
        """
        start_time = datetime.now()
        
        try:
            # Validate input parameters
            validation_result = self.validate_input(task)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "validation_errors": validation_result["errors"],
                    "suggestions": validation_result["suggestions"]
                }
            
            # Extract task parameters
            task_type = task.get("type", "stellar_evolution")
            parameters = task.get("parameters", {})
            
            # Route to appropriate calculation
            if task_type == "stellar_evolution":
                result = self._model_stellar_evolution(parameters, context)
            elif task_type == "orbital_dynamics":
                result = self._calculate_orbital_dynamics(parameters, context)
            elif task_type == "cosmological_distance":
                result = self._calculate_cosmological_distance(parameters, context)
            elif task_type == "galaxy_rotation":
                result = self._model_galaxy_rotation(parameters, context)
            elif task_type == "black_hole_properties":
                result = self._calculate_black_hole_properties(parameters, context)
            elif task_type == "exoplanet_detection":
                result = self._analyze_exoplanet(parameters, context)
            elif task_type == "supernova_lightcurve":
                result = self._model_supernova(parameters, context)
            elif task_type == "gravitational_waves":
                result = self._calculate_gravitational_waves(parameters, context)
            else:
                raise ValueError(f"Unknown astrophysics task type: {task_type}")
            
            # Process and format output for agents
            formatted_result = self.process_output(result)
            
            # Update statistics
            calculation_time = (datetime.now() - start_time).total_seconds()
            estimated_cost = self._calculate_actual_cost(task, calculation_time)
            self.update_calculation_stats(calculation_time, estimated_cost, True)
            
            return {
                "success": True,
                "task_type": task_type,
                "results": formatted_result,
                "calculation_time": calculation_time,
                "computational_cost": estimated_cost,
                "confidence": self._assess_result_confidence(result, task_type),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.update_calculation_stats(calculation_time, 0.0, False)
            return self.handle_errors(e, {"task": task, "context": context})
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate astrophysics input parameters.
        
        Args:
            input_data: Input parameters from agent
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        task_type = input_data.get("type", "stellar_evolution")
        if task_type not in self.calculation_types:
            errors.append(f"Unknown calculation type '{task_type}'")
            suggestions.append(f"Available types: {', '.join(self.calculation_types)}")
        
        parameters = input_data.get("parameters", {})
        if not parameters:
            errors.append("Missing 'parameters' for calculation")
            suggestions.append("Provide relevant physical parameters for the calculation")
        
        # Validate specific parameters based on task type
        if task_type == "stellar_evolution":
            if "mass" not in parameters:
                errors.append("Stellar mass required for stellar evolution")
                suggestions.append("Provide stellar mass in solar masses (M_sun)")
            else:
                mass = parameters["mass"]
                if mass <= 0 or mass > 200:
                    errors.append("Stellar mass must be between 0 and 200 solar masses")
        
        elif task_type == "orbital_dynamics":
            required_params = ["mass1", "mass2", "separation"]
            missing = [p for p in required_params if p not in parameters]
            if missing:
                errors.append(f"Missing orbital parameters: {', '.join(missing)}")
                suggestions.append("Provide masses and separation for orbital calculation")
        
        elif task_type == "cosmological_distance":
            if "redshift" not in parameters:
                errors.append("Redshift required for cosmological distance calculation")
                suggestions.append("Provide redshift value (z)")
            else:
                z = parameters["redshift"]
                if z < 0 or z > 10:
                    warnings.append("Redshift outside typical range (0-10)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format astrophysics results for agents.
        
        Args:
            output_data: Raw calculation results
            
        Returns:
            Agent-friendly formatted results
        """
        formatted = {
            "summary": self._generate_result_summary(output_data),
            "physical_properties": {},
            "derived_quantities": {},
            "observational_predictions": {},
            "analysis": {}
        }
        
        # Physical properties
        if "mass" in output_data:
            formatted["physical_properties"]["mass"] = {
                "value": output_data["mass"],
                "units": "solar_masses",
                "description": "Object mass"
            }
        
        if "radius" in output_data:
            formatted["physical_properties"]["radius"] = {
                "value": output_data["radius"],
                "units": "solar_radii",
                "description": "Object radius"
            }
        
        if "luminosity" in output_data:
            formatted["physical_properties"]["luminosity"] = {
                "value": output_data["luminosity"],
                "units": "solar_luminosities",
                "description": "Object luminosity"
            }
        
        if "temperature" in output_data:
            formatted["physical_properties"]["temperature"] = {
                "value": output_data["temperature"],
                "units": "K",
                "description": "Effective temperature"
            }
        
        # Derived quantities
        if "lifetime" in output_data:
            formatted["derived_quantities"]["main_sequence_lifetime"] = {
                "value": output_data["lifetime"],
                "units": "years",
                "description": "Main sequence lifetime"
            }
        
        if "distance" in output_data:
            formatted["derived_quantities"]["distance"] = {
                "value": output_data["distance"],
                "units": output_data.get("distance_units", "pc"),
                "description": "Distance measurement"
            }
        
        # Observational predictions
        if "apparent_magnitude" in output_data:
            formatted["observational_predictions"]["apparent_magnitude"] = {
                "value": output_data["apparent_magnitude"],
                "description": "Apparent magnitude in V-band"
            }
        
        if "orbital_period" in output_data:
            formatted["observational_predictions"]["orbital_period"] = {
                "value": output_data["orbital_period"],
                "units": "days",
                "description": "Orbital period"
            }
        
        # Analysis and insights
        formatted["analysis"]["classification"] = self._classify_object(output_data)
        formatted["analysis"]["evolutionary_stage"] = output_data.get("evolutionary_stage", "unknown")
        formatted["analysis"]["insights"] = self._generate_insights(output_data)
        formatted["analysis"]["recommendations"] = self._generate_recommendations(output_data)
        
        return formatted
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for astrophysics calculation.
        
        Args:
            task: Task specification
            
        Returns:
            Cost estimates (time, memory, computational units)
        """
        base_cost = 1.0
        
        # Calculation type cost factors
        calc_costs = {
            "stellar_evolution": 2.0,
            "orbital_dynamics": 1.0,
            "cosmological_distance": 0.5,
            "galaxy_rotation": 3.0,
            "black_hole_properties": 1.5,
            "exoplanet_detection": 2.5,
            "supernova_lightcurve": 4.0,
            "gravitational_waves": 5.0
        }
        
        task_type = task.get("type", "stellar_evolution")
        calc_cost = calc_costs.get(task_type, 1.0)
        
        # Parameter complexity factors
        parameters = task.get("parameters", {})
        complexity_factor = 1.0
        
        if "time_evolution" in parameters:
            complexity_factor *= 2.0
        if "high_precision" in parameters and parameters["high_precision"]:
            complexity_factor *= 1.5
        
        total_cost_factor = calc_cost * complexity_factor * self.computational_cost_factor
        
        # Estimate time (in seconds)
        estimated_time = base_cost * total_cost_factor * 2
        
        # Estimate memory (in MB)
        estimated_memory = 100 + total_cost_factor * 20
        
        # Computational units
        computational_units = total_cost_factor * 25
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_memory_mb": estimated_memory,
            "computational_units": computational_units,
            "cost_breakdown": {
                "calculation_factor": calc_cost,
                "complexity_factor": complexity_factor,
                "total_factor": total_cost_factor
            }
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get astrophysics specific requirements."""
        return {
            "physics_domain": "astrophysics",
            "available_calculation_types": self.calculation_types,
            "physical_constants": {k: v for k, v in self.constants.items()},
            "typical_calculation_time": "1 second to several minutes",
            "memory_scaling": "Linear to quadratic with time evolution",
            "software_dependencies": self.software_requirements,
            "hardware_recommendations": self.hardware_requirements
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Get astrophysics domain keywords."""
        return [
            "stellar", "star", "galaxy", "cosmology", "black", "hole",
            "orbit", "planet", "exoplanet", "supernova", "neutron",
            "redshift", "distance", "luminosity", "magnitude", "mass",
            "radius", "evolution", "gravitational", "waves"
        ]
    
    def _model_stellar_evolution(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Model stellar evolution (simplified implementation)."""
        mass = parameters.get("mass", 1.0)  # Solar masses
        
        # Simplified stellar evolution relationships
        # Main sequence lifetime (years)
        lifetime = 10e9 * (mass ** -2.5)
        
        # Luminosity (solar luminosities)
        luminosity = mass ** 3.5
        
        # Radius (solar radii)
        radius = mass ** 0.8
        
        # Temperature (K)
        temperature = 5778 * (mass ** 0.5)  # Sun's temp = 5778 K
        
        # Evolutionary stage determination
        if mass < 0.08:
            stage = "brown_dwarf"
        elif mass < 0.5:
            stage = "red_dwarf"
        elif mass < 2.0:
            stage = "main_sequence"
        elif mass < 8.0:
            stage = "massive_star"
        else:
            stage = "very_massive_star"
        
        return {
            "mass": mass,
            "luminosity": luminosity,
            "radius": radius,
            "temperature": temperature,
            "lifetime": lifetime,
            "evolutionary_stage": stage,
            "surface_gravity": math.log10(mass / (radius ** 2)),  # log g
            "spectral_type": self._determine_spectral_type(temperature)
        }
    
    def _calculate_orbital_dynamics(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate orbital dynamics (simplified implementation)."""
        mass1 = parameters.get("mass1", 1.0) * self.constants["M_sun"]  # kg
        mass2 = parameters.get("mass2", 1.0) * self.constants["M_sun"]  # kg
        separation = parameters.get("separation", 1.0) * self.constants["AU"]  # m
        
        # Total mass
        total_mass = mass1 + mass2
        
        # Orbital period (Kepler's 3rd law)
        period_seconds = 2 * math.pi * math.sqrt(separation ** 3 / (self.constants["G"] * total_mass))
        period_days = period_seconds / (24 * 3600)
        period_years = period_days / 365.25
        
        # Orbital velocity
        velocity = 2 * math.pi * separation / period_seconds / 1000  # km/s
        
        # Reduced mass
        reduced_mass = (mass1 * mass2) / total_mass
        
        # Gravitational binding energy
        binding_energy = -self.constants["G"] * mass1 * mass2 / separation / 1.602e-19 / 1e9  # GeV
        
        return {
            "mass1": mass1 / self.constants["M_sun"],
            "mass2": mass2 / self.constants["M_sun"],
            "separation": separation / self.constants["AU"],
            "orbital_period": period_days,
            "orbital_period_years": period_years,
            "orbital_velocity": velocity,
            "binding_energy": binding_energy,
            "reduced_mass": reduced_mass / self.constants["M_sun"]
        }
    
    def _calculate_cosmological_distance(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cosmological distances (simplified implementation)."""
        z = parameters.get("redshift", 0.1)
        H0 = parameters.get("hubble_constant", 70)  # km/s/Mpc
        
        # Hubble distance (for small z)
        hubble_distance = self.constants["c"] * z / (H0 * 1000) * self.constants["pc"] / 1e6  # Mpc
        
        # Luminosity distance (approximate)
        luminosity_distance = hubble_distance * (1 + z)
        
        # Angular diameter distance
        angular_distance = hubble_distance / ((1 + z) ** 2)
        
        # Light travel time
        light_travel_time = hubble_distance * 1e6 * self.constants["pc"] / self.constants["c"] / self.constants["year"] / 1e9  # Gyr
        
        return {
            "redshift": z,
            "hubble_distance": hubble_distance,
            "luminosity_distance": luminosity_distance,
            "angular_diameter_distance": angular_distance,
            "light_travel_time": light_travel_time,
            "distance_units": "Mpc",
            "lookback_time": light_travel_time
        }
    
    def _model_galaxy_rotation(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Model galaxy rotation curve (simplified implementation)."""
        radius = parameters.get("radius", 10)  # kpc
        mass = parameters.get("total_mass", 1e12) * self.constants["M_sun"]  # kg
        
        # Keplerian velocity
        keplerian_velocity = math.sqrt(self.constants["G"] * mass / (radius * 1000 * self.constants["pc"])) / 1000  # km/s
        
        # Flat rotation curve velocity (dark matter dominated)
        flat_velocity = 220  # km/s (typical for Milky Way)
        
        # Enclosed mass within radius
        enclosed_mass = mass * (radius / 50) ** 1.5  # Rough approximation
        
        return {
            "radius": radius,
            "keplerian_velocity": keplerian_velocity,
            "observed_velocity": flat_velocity,
            "enclosed_mass": enclosed_mass / self.constants["M_sun"],
            "dark_matter_fraction": 0.85,
            "velocity_units": "km/s"
        }
    
    def _calculate_black_hole_properties(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate black hole properties (simplified implementation)."""
        mass = parameters.get("mass", 10) * self.constants["M_sun"]  # kg
        
        # Schwarzschild radius
        rs = 2 * self.constants["G"] * mass / (self.constants["c"] ** 2)  # m
        
        # Hawking temperature
        hawking_temp = self.constants["h"] * (self.constants["c"] ** 3) / (8 * math.pi * self.constants["k_B"] * self.constants["G"] * mass)
        
        # Luminosity (Hawking radiation)
        hawking_luminosity = self.constants["h"] * (self.constants["c"] ** 6) / (15360 * math.pi * (self.constants["G"] ** 2) * (mass ** 2))
        
        # Evaporation time
        evaporation_time = (5120 * math.pi * (self.constants["G"] ** 2) * (mass ** 3)) / (self.constants["h"] * (self.constants["c"] ** 4))
        evaporation_time_years = evaporation_time / self.constants["year"]
        
        return {
            "mass": mass / self.constants["M_sun"],
            "schwarzschild_radius": rs,
            "schwarzschild_radius_km": rs / 1000,
            "hawking_temperature": hawking_temp,
            "hawking_luminosity": hawking_luminosity,
            "evaporation_time": evaporation_time_years,
            "surface_gravity": self.constants["c"] ** 4 / (4 * self.constants["G"] * mass)
        }
    
    def _analyze_exoplanet(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze exoplanet properties (simplified implementation)."""
        stellar_mass = parameters.get("stellar_mass", 1.0) * self.constants["M_sun"]
        planet_radius = parameters.get("planet_radius", 1.0) * 6.371e6  # Earth radii to m
        orbital_period = parameters.get("orbital_period", 365.25)  # days
        
        # Semi-major axis from Kepler's 3rd law
        period_seconds = orbital_period * 24 * 3600
        semi_major_axis = ((self.constants["G"] * stellar_mass * (period_seconds ** 2)) / (4 * (math.pi ** 2))) ** (1/3)
        
        # Equilibrium temperature (assuming Earth-like albedo)
        stellar_luminosity = parameters.get("stellar_luminosity", 1.0) * self.constants["L_sun"]
        equilibrium_temp = ((stellar_luminosity * (1 - 0.3)) / (16 * math.pi * self.constants["sigma_SB"] * (semi_major_axis ** 2))) ** 0.25
        
        # Habitable zone boundaries
        hz_inner = math.sqrt(stellar_luminosity / self.constants["L_sun"]) * 0.95 * self.constants["AU"]
        hz_outer = math.sqrt(stellar_luminosity / self.constants["L_sun"]) * 1.37 * self.constants["AU"]
        
        # Planet classification
        if planet_radius < 1.5 * 6.371e6:
            planet_type = "rocky"
        elif planet_radius < 4.0 * 6.371e6:
            planet_type = "super_earth"
        else:
            planet_type = "gas_giant"
        
        return {
            "stellar_mass": stellar_mass / self.constants["M_sun"],
            "planet_radius": planet_radius / 6.371e6,
            "orbital_period": orbital_period,
            "semi_major_axis": semi_major_axis / self.constants["AU"],
            "equilibrium_temperature": equilibrium_temp,
            "habitable_zone_inner": hz_inner / self.constants["AU"],
            "habitable_zone_outer": hz_outer / self.constants["AU"],
            "in_habitable_zone": hz_inner < semi_major_axis < hz_outer,
            "planet_type": planet_type
        }
    
    def _model_supernova(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Model supernova lightcurve (simplified implementation)."""
        progenitor_mass = parameters.get("progenitor_mass", 20.0)  # Solar masses
        explosion_energy = parameters.get("explosion_energy", 1e44)  # Joules
        
        # Peak luminosity (rough correlation)
        peak_luminosity = 1e9 * self.constants["L_sun"] * (explosion_energy / 1e44) ** 0.5
        
        # Rise time and decay time
        rise_time = 20  # days
        decay_time = 100  # days
        
        # Generate mock lightcurve
        times = np.linspace(0, 300, 100)  # days
        luminosity = np.zeros_like(times)
        
        for i, t in enumerate(times):
            if t < rise_time:
                luminosity[i] = peak_luminosity * (t / rise_time) ** 2
            else:
                luminosity[i] = peak_luminosity * np.exp(-(t - rise_time) / decay_time)
        
        return {
            "progenitor_mass": progenitor_mass,
            "explosion_energy": explosion_energy,
            "peak_luminosity": peak_luminosity / self.constants["L_sun"],
            "rise_time": rise_time,
            "decay_time": decay_time,
            "lightcurve_times": times.tolist(),
            "lightcurve_luminosity": (luminosity / self.constants["L_sun"]).tolist(),
            "supernova_type": "core_collapse" if progenitor_mass > 8 else "thermonuclear"
        }
    
    def _calculate_gravitational_waves(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate gravitational wave properties (simplified implementation)."""
        mass1 = parameters.get("mass1", 30) * self.constants["M_sun"]
        mass2 = parameters.get("mass2", 30) * self.constants["M_sun"]
        distance = parameters.get("distance", 1000) * 1e6 * self.constants["pc"]  # Mpc to m
        
        # Chirp mass
        chirp_mass = ((mass1 * mass2) ** 0.6) / ((mass1 + mass2) ** 0.2)
        
        # Frequency evolution (simplified)
        frequency_initial = parameters.get("frequency", 35)  # Hz
        frequency_final = 250  # Hz at merger
        
        # Strain amplitude (order of magnitude)
        strain = (self.constants["G"] / self.constants["c"] ** 4) * (chirp_mass * (math.pi * frequency_final) ** (2/3)) / distance
        
        # Energy radiated
        energy_radiated = 0.05 * (mass1 + mass2) * self.constants["c"] ** 2  # ~5% of total mass
        
        return {
            "mass1": mass1 / self.constants["M_sun"],
            "mass2": mass2 / self.constants["M_sun"],
            "chirp_mass": chirp_mass / self.constants["M_sun"],
            "distance": distance / (1e6 * self.constants["pc"]),
            "strain_amplitude": strain,
            "frequency_initial": frequency_initial,
            "frequency_final": frequency_final,
            "energy_radiated": energy_radiated,
            "detection_probability": min(1.0, (1e-21 / strain) ** 2)
        }
    
    def _calculate_actual_cost(self, task: Dict[str, Any], actual_time: float) -> float:
        """Calculate actual computational cost."""
        estimates = self.estimate_cost(task)
        estimated_time = estimates["estimated_time_seconds"]
        
        time_ratio = actual_time / max(estimated_time, 0.1)
        actual_cost = estimates["computational_units"] * time_ratio
        
        return actual_cost
    
    def _assess_result_confidence(self, result: Dict[str, Any], task_type: str) -> float:
        """Assess confidence in calculation results."""
        # Task type reliability
        task_confidence = {
            "stellar_evolution": 0.8,
            "orbital_dynamics": 0.95,
            "cosmological_distance": 0.9,
            "galaxy_rotation": 0.7,
            "black_hole_properties": 0.85,
            "exoplanet_detection": 0.8,
            "supernova_lightcurve": 0.6,
            "gravitational_waves": 0.75
        }
        
        return task_confidence.get(task_type, 0.7)
    
    def _determine_spectral_type(self, temperature: float) -> str:
        """Determine spectral type from temperature."""
        if temperature > 30000:
            return "O"
        elif temperature > 10000:
            return "B"
        elif temperature > 7500:
            return "A"
        elif temperature > 6000:
            return "F"
        elif temperature > 5200:
            return "G"
        elif temperature > 3700:
            return "K"
        else:
            return "M"
    
    def _classify_object(self, result: Dict[str, Any]) -> str:
        """Classify astronomical object based on properties."""
        if "evolutionary_stage" in result:
            return result["evolutionary_stage"]
        elif "planet_type" in result:
            return result["planet_type"]
        elif "supernova_type" in result:
            return result["supernova_type"]
        else:
            return "unknown"
    
    def _generate_result_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable summary of results."""
        summary = "Astrophysics calculation completed. "
        
        if "mass" in result:
            summary += f"Object mass: {result['mass']:.2f} solar masses. "
        
        if "distance" in result:
            units = result.get("distance_units", "pc")
            summary += f"Distance: {result['distance']:.2f} {units}. "
        
        if "orbital_period" in result:
            summary += f"Orbital period: {result['orbital_period']:.1f} days. "
        
        if "luminosity" in result:
            summary += f"Luminosity: {result['luminosity']:.2e} solar luminosities. "
        
        return summary
    
    def _generate_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate scientific insights from results."""
        insights = []
        
        if "evolutionary_stage" in result:
            stage = result["evolutionary_stage"]
            if stage == "red_dwarf":
                insights.append("Long-lived star suitable for stable planetary systems")
            elif stage == "massive_star":
                insights.append("Will end as supernova, enriching interstellar medium")
        
        if "in_habitable_zone" in result and result["in_habitable_zone"]:
            insights.append("Planet located in habitable zone - potential for liquid water")
        
        if "strain_amplitude" in result:
            strain = result["strain_amplitude"]
            if strain > 1e-21:
                insights.append("Gravitational wave signal detectable by current instruments")
        
        return insights
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for follow-up observations/calculations."""
        recommendations = []
        
        if "supernova_type" in result:
            recommendations.append("Monitor lightcurve evolution for detailed classification")
        
        if "planet_type" in result and result["planet_type"] == "super_earth":
            recommendations.append("Investigate atmospheric composition and retention")
        
        if "black_hole" in str(result):
            recommendations.append("Search for X-ray emission from accretion disk")
        
        return recommendations