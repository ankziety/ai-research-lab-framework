"""
Optics and Photonics Tool

Agent-friendly interface for optics and photonics calculations.
Provides ray tracing, wave optics, laser physics, nonlinear optics,
and photonic device analysis.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class OpticsPhotonicsTool(BasePhysicsTool):
    """
    Tool for optics and photonics calculations that agents can request.
    
    Provides interfaces for:
    - Geometric ray tracing
    - Wave optics and diffraction
    - Laser physics and cavity design
    - Nonlinear optics
    - Fiber optics and waveguides
    - Photonic crystals and metamaterials
    """
    
    def __init__(self):
        super().__init__(
            tool_id="optics_photonics_tool",
            name="Optics and Photonics Tool",
            description="Perform optics and photonics calculations including ray tracing, wave optics, laser physics, and photonic devices",
            physics_domain="optics_photonics",
            computational_cost_factor=2.5,
            software_requirements=[
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib",   # Visualization
                "sympy"         # Symbolic mathematics (optional)
            ],
            hardware_requirements={
                "min_memory": 1024,  # MB
                "recommended_memory": 4096,
                "cpu_cores": 2,
                "supports_gpu": True
            }
        )
        
        self.capabilities.extend([
            "ray_tracing",
            "wave_optics",
            "laser_physics",
            "nonlinear_optics",
            "fiber_optics",
            "photonic_crystals",
            "interferometry",
            "polarization_analysis"
        ])
        
        # Optical constants
        self.constants = {
            "c": 2.998e8,       # m/s (speed of light)
            "h": 6.626e-34,     # J·s (Planck constant)
            "hbar": 1.055e-34,  # J·s
            "e": 1.602e-19,     # C
            "epsilon0": 8.854e-12,  # F/m
            "mu0": 4*np.pi*1e-7,    # H/m
            "k_b": 1.381e-23    # J/K
        }
        
        # Common material properties
        self.materials = {
            "air": {"n": 1.0003, "dispersion": 0},
            "water": {"n": 1.333, "dispersion": -0.01},
            "glass_bk7": {"n": 1.517, "dispersion": 0.01},
            "fused_silica": {"n": 1.458, "dispersion": 0.008},
            "silicon": {"n": 3.48, "dispersion": 0.1},
            "germanium": {"n": 4.05, "dispersion": 0.2},
            "diamond": {"n": 2.42, "dispersion": 0.06}
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optics/photonics calculation.
        
        Args:
            task: Task specification with optical parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with optical analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "ray_tracing")
            
            if task_type == "ray_tracing":
                result = self._perform_ray_tracing(task)
            elif task_type == "wave_optics":
                result = self._analyze_wave_optics(task)
            elif task_type == "laser_physics":
                result = self._analyze_laser_physics(task)
            elif task_type == "nonlinear_optics":
                result = self._analyze_nonlinear_optics(task)
            elif task_type == "fiber_optics":
                result = self._analyze_fiber_optics(task)
            elif task_type == "interferometry":
                result = self._analyze_interferometry(task)
            elif task_type == "polarization":
                result = self._analyze_polarization(task)
            else:
                result = self._generic_optics_calculation(task)
            
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
            logger.error(f"Optics/photonics calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _perform_ray_tracing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform geometric ray tracing analysis."""
        optical_system = task.get("optical_system", "simple_lens")
        wavelength = task.get("wavelength", 589e-9)  # m (sodium D-line)
        
        if optical_system == "simple_lens":
            focal_length = task.get("focal_length", 0.1)  # m
            object_distance = task.get("object_distance", 0.15)  # m
            
            # Thin lens equation: 1/f = 1/do + 1/di
            if object_distance != focal_length:
                image_distance = 1 / (1/focal_length - 1/object_distance)
                magnification = -image_distance / object_distance
                
                # Classification
                if image_distance > 0:
                    image_type = "real"
                else:
                    image_type = "virtual"
                    
                if abs(magnification) > 1:
                    size_type = "magnified"
                elif abs(magnification) < 1:
                    size_type = "diminished"
                else:
                    size_type = "same_size"
                
                # Angular magnification for virtual images
                angular_magnification = abs(magnification) if image_type == "virtual" else 1
                
                return {
                    "optical_system": optical_system,
                    "focal_length": focal_length,
                    "object_distance": object_distance,
                    "image_distance": image_distance,
                    "magnification": magnification,
                    "image_properties": {
                        "type": image_type,
                        "orientation": "inverted" if magnification < 0 else "upright",
                        "size": size_type
                    },
                    "angular_magnification": angular_magnification,
                    "convergence": True,
                    "physical_validity": True
                }
            else:
                return {
                    "optical_system": optical_system,
                    "error": "Object at focal point - image at infinity",
                    "convergence": False,
                    "physical_validity": True
                }
        
        elif optical_system == "spherical_mirror":
            radius_curvature = task.get("radius_curvature", 0.2)  # m
            object_distance = task.get("object_distance", 0.15)  # m
            
            focal_length = radius_curvature / 2
            
            if object_distance != focal_length:
                image_distance = 1 / (1/focal_length - 1/object_distance)
                magnification = -image_distance / object_distance
                
                return {
                    "optical_system": optical_system,
                    "radius_curvature": radius_curvature,
                    "focal_length": focal_length,
                    "object_distance": object_distance,
                    "image_distance": image_distance,
                    "magnification": magnification,
                    "image_type": "real" if image_distance > 0 else "virtual",
                    "convergence": True,
                    "physical_validity": True
                }
            else:
                return {
                    "optical_system": optical_system,
                    "error": "Object at focal point",
                    "convergence": False,
                    "physical_validity": True
                }
        
        elif optical_system == "prism":
            apex_angle = task.get("apex_angle", np.pi/3)  # radians (60 degrees)
            incident_angle = task.get("incident_angle", np.pi/4)  # radians
            material = task.get("material", "glass_bk7")
            
            n = self.materials.get(material, {"n": 1.5})["n"]
            
            # Snell's law at first surface
            refracted_angle1 = np.arcsin(np.sin(incident_angle) / n)
            
            # Geometry inside prism
            angle_inside = apex_angle - refracted_angle1
            
            # Snell's law at second surface
            emergence_angle = np.arcsin(n * np.sin(angle_inside))
            
            # Total deviation
            deviation = incident_angle + emergence_angle - apex_angle
            
            # Minimum deviation condition
            min_deviation = 2 * np.arcsin(n * np.sin(apex_angle/2)) - apex_angle
            
            return {
                "optical_system": optical_system,
                "apex_angle": np.degrees(apex_angle),
                "incident_angle": np.degrees(incident_angle),
                "refractive_index": n,
                "emergence_angle": np.degrees(emergence_angle),
                "deviation": np.degrees(deviation),
                "minimum_deviation": np.degrees(min_deviation),
                "dispersion": "yes" if self.materials.get(material, {"dispersion": 0})["dispersion"] > 0 else "no",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "optical_system": optical_system,
                "error": f"Unknown optical system: {optical_system}",
                "available_systems": ["simple_lens", "spherical_mirror", "prism"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_wave_optics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze wave optics phenomena."""
        phenomenon = task.get("phenomenon", "single_slit_diffraction")
        wavelength = task.get("wavelength", 632.8e-9)  # m (He-Ne laser)
        
        if phenomenon == "single_slit_diffraction":
            slit_width = task.get("slit_width", 10e-6)  # m
            screen_distance = task.get("screen_distance", 1.0)  # m
            
            # Angular positions of minima
            minima_angles = []
            minima_positions = []
            for m in range(1, 6):  # First 5 minima
                angle = np.arcsin(m * wavelength / slit_width)
                position = screen_distance * np.tan(angle)
                minima_angles.append(np.degrees(angle))
                minima_positions.append(position * 1000)  # mm
            
            # Central maximum width
            central_width = 2 * screen_distance * np.tan(np.arcsin(wavelength / slit_width))
            
            return {
                "phenomenon": phenomenon,
                "wavelength": wavelength * 1e9,  # nm
                "slit_width": slit_width * 1e6,  # μm
                "screen_distance": screen_distance,
                "minima_angles": minima_angles[:3],  # First 3
                "minima_positions": minima_positions[:3],  # mm
                "central_maximum_width": central_width * 1000,  # mm
                "angular_resolution": np.degrees(wavelength / slit_width),
                "units": {
                    "wavelength": "nm",
                    "slit_width": "μm",
                    "positions": "mm",
                    "angles": "degrees"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif phenomenon == "double_slit_interference":
            slit_separation = task.get("slit_separation", 50e-6)  # m
            screen_distance = task.get("screen_distance", 1.0)  # m
            
            # Fringe spacing
            fringe_spacing = wavelength * screen_distance / slit_separation
            
            # Positions of bright fringes
            bright_positions = []
            for m in range(-2, 3):  # -2 to +2
                position = m * fringe_spacing
                bright_positions.append(position * 1000)  # mm
            
            # Visibility (assuming equal intensity slits)
            visibility = 1.0
            
            return {
                "phenomenon": phenomenon,
                "wavelength": wavelength * 1e9,  # nm
                "slit_separation": slit_separation * 1e6,  # μm
                "screen_distance": screen_distance,
                "fringe_spacing": fringe_spacing * 1000,  # mm
                "bright_fringe_positions": bright_positions,
                "visibility": visibility,
                "angular_fringe_spacing": np.degrees(wavelength / slit_separation),
                "convergence": True,
                "physical_validity": True
            }
        
        elif phenomenon == "circular_aperture_diffraction":
            aperture_diameter = task.get("aperture_diameter", 1e-3)  # m
            
            # Airy disk radius (angular)
            airy_radius_angular = 1.22 * wavelength / aperture_diameter
            
            # Linear radius at screen
            screen_distance = task.get("screen_distance", 1.0)  # m
            airy_radius_linear = screen_distance * np.tan(airy_radius_angular)
            
            # Rayleigh resolution criterion
            resolution_angle = airy_radius_angular
            resolution_distance = wavelength / (2 * np.sin(aperture_diameter / (2 * screen_distance)))
            
            return {
                "phenomenon": phenomenon,
                "aperture_diameter": aperture_diameter * 1000,  # mm
                "airy_radius_angular": np.degrees(airy_radius_angular),
                "airy_radius_linear": airy_radius_linear * 1000,  # mm
                "rayleigh_resolution": np.degrees(resolution_angle),
                "diffraction_limit": "yes",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "phenomenon": phenomenon,
                "error": f"Unknown wave optics phenomenon: {phenomenon}",
                "available_phenomena": ["single_slit_diffraction", "double_slit_interference", "circular_aperture_diffraction"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_laser_physics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze laser physics and cavity design."""
        laser_type = task.get("laser_type", "fabry_perot_cavity")
        wavelength = task.get("wavelength", 1064e-9)  # m (Nd:YAG)
        
        if laser_type == "fabry_perot_cavity":
            cavity_length = task.get("cavity_length", 0.3)  # m
            mirror_reflectivity = task.get("mirror_reflectivity", [0.99, 0.95])  # R1, R2
            gain_medium_length = task.get("gain_medium_length", 0.1)  # m
            
            # Mode spacing
            mode_spacing_freq = self.constants["c"] / (2 * cavity_length)
            mode_spacing_wavelength = wavelength**2 / (2 * cavity_length)
            
            # Finesse
            r1, r2 = mirror_reflectivity
            finesse = np.pi * np.sqrt(np.sqrt(r1 * r2)) / (1 - np.sqrt(r1 * r2))
            
            # Q factor
            q_factor = 2 * np.pi * cavity_length / (wavelength * (1 - np.sqrt(r1 * r2)))
            
            # Threshold gain
            loss_per_pass = 1 - np.sqrt(r1 * r2)
            threshold_gain = loss_per_pass / (2 * gain_medium_length)
            
            # Beam waist (fundamental mode)
            beam_waist = np.sqrt(wavelength * cavity_length / (2 * np.pi))
            
            return {
                "laser_type": laser_type,
                "cavity_parameters": {
                    "length": cavity_length,
                    "mirror_reflectivities": mirror_reflectivity,
                    "gain_medium_length": gain_medium_length
                },
                "mode_properties": {
                    "longitudinal_mode_spacing_freq": mode_spacing_freq / 1e9,  # GHz
                    "longitudinal_mode_spacing_wavelength": mode_spacing_wavelength * 1e12,  # pm
                    "finesse": finesse,
                    "q_factor": q_factor
                },
                "threshold_conditions": {
                    "threshold_gain": threshold_gain,
                    "loss_per_pass": loss_per_pass
                },
                "beam_properties": {
                    "fundamental_mode_waist": beam_waist * 1e6,  # μm
                    "divergence_angle": np.degrees(wavelength / (np.pi * beam_waist))
                },
                "units": {
                    "frequency": "GHz",
                    "wavelength": "pm",
                    "length": "μm",
                    "gain": "m^-1"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif laser_type == "rate_equations":
            # Simple two-level system
            pump_rate = task.get("pump_rate", 1e6)  # s^-1
            spontaneous_lifetime = task.get("spontaneous_lifetime", 1e-3)  # s
            stimulated_cross_section = task.get("stimulated_cross_section", 1e-20)  # m^2
            photon_density = task.get("photon_density", 1e20)  # m^-3
            
            # Rate equation parameters
            spontaneous_rate = 1 / spontaneous_lifetime
            stimulated_rate = stimulated_cross_section * photon_density * self.constants["c"]
            
            # Steady state population inversion
            if pump_rate > spontaneous_rate:
                population_inversion = (pump_rate - spontaneous_rate) / (spontaneous_rate + stimulated_rate)
                gain = population_inversion * stimulated_cross_section
                lasing = "yes" if gain > 0 else "no"
            else:
                population_inversion = 0
                gain = 0
                lasing = "no"
            
            return {
                "laser_type": laser_type,
                "rate_parameters": {
                    "pump_rate": pump_rate,
                    "spontaneous_rate": spontaneous_rate,
                    "stimulated_rate": stimulated_rate
                },
                "steady_state": {
                    "population_inversion": population_inversion,
                    "optical_gain": gain,
                    "lasing_condition": lasing
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "laser_type": laser_type,
                "error": f"Unknown laser type: {laser_type}",
                "available_types": ["fabry_perot_cavity", "rate_equations"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_nonlinear_optics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nonlinear optical processes."""
        process = task.get("process", "second_harmonic_generation")
        wavelength = task.get("fundamental_wavelength", 1064e-9)  # m
        intensity = task.get("intensity", 1e12)  # W/m^2
        
        if process == "second_harmonic_generation":
            crystal_length = task.get("crystal_length", 1e-3)  # m
            nonlinear_coefficient = task.get("nonlinear_coefficient", 1e-12)  # m/V
            
            # Second harmonic wavelength
            shg_wavelength = wavelength / 2
            shg_frequency = self.constants["c"] / shg_wavelength
            
            # Phase matching condition
            # Simplified - assumes perfect phase matching
            phase_mismatch = 0  # Δk = 0 for perfect phase matching
            
            # Conversion efficiency (small signal approximation)
            # η ∝ (d_eff)^2 * L^2 * I
            relative_efficiency = (nonlinear_coefficient * crystal_length)**2 * intensity
            
            # Conversion efficiency (normalized)
            max_efficiency = 0.8  # Theoretical maximum
            efficiency = min(max_efficiency, relative_efficiency / 1e15)  # Rough scaling
            
            # Output power ratio
            shg_power_ratio = efficiency
            fundamental_power_ratio = 1 - efficiency
            
            return {
                "process": process,
                "fundamental_wavelength": wavelength * 1e9,  # nm
                "second_harmonic_wavelength": shg_wavelength * 1e9,  # nm
                "crystal_parameters": {
                    "length": crystal_length * 1000,  # mm
                    "nonlinear_coefficient": nonlinear_coefficient * 1e12,  # pm/V
                    "phase_matching": "achieved" if phase_mismatch == 0 else "not_achieved"
                },
                "conversion_properties": {
                    "efficiency": efficiency,
                    "shg_power_fraction": shg_power_ratio,
                    "remaining_fundamental_fraction": fundamental_power_ratio
                },
                "units": {
                    "wavelength": "nm",
                    "length": "mm",
                    "coefficient": "pm/V"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif process == "third_harmonic_generation":
            # Third harmonic generation
            thg_wavelength = wavelength / 3
            
            return {
                "process": process,
                "fundamental_wavelength": wavelength * 1e9,  # nm
                "third_harmonic_wavelength": thg_wavelength * 1e9,  # nm
                "note": "Third-order nonlinear process",
                "convergence": True,
                "physical_validity": True
            }
        
        elif process == "kerr_effect":
            # Optical Kerr effect
            kerr_coefficient = task.get("kerr_coefficient", 1e-20)  # m^2/W
            beam_diameter = task.get("beam_diameter", 1e-3)  # m
            
            # Nonlinear refractive index change
            delta_n = kerr_coefficient * intensity
            
            # Self-focusing critical power
            critical_power = np.pi * (0.61 * wavelength)**2 / (8 * kerr_coefficient)
            
            # Self-focusing length
            if intensity > 0:
                beam_area = np.pi * (beam_diameter/2)**2
                power = intensity * beam_area
                if power < critical_power:
                    self_focus_length = float('inf')
                else:
                    self_focus_length = 0.367 * beam_diameter / np.sqrt(power/critical_power - 1)
            else:
                self_focus_length = float('inf')
            
            return {
                "process": process,
                "kerr_coefficient": kerr_coefficient * 1e20,  # x10^-20 m^2/W
                "intensity": intensity / 1e12,  # TW/m^2
                "nonlinear_index_change": delta_n,
                "critical_power": critical_power / 1e6,  # MW
                "self_focusing_length": self_focus_length * 1000 if self_focus_length != float('inf') else "infinite",
                "self_focusing": "yes" if self_focus_length != float('inf') else "no",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "process": process,
                "error": f"Unknown nonlinear process: {process}",
                "available_processes": ["second_harmonic_generation", "third_harmonic_generation", "kerr_effect"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_fiber_optics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fiber optic properties."""
        fiber_type = task.get("fiber_type", "step_index_multimode")
        wavelength = task.get("wavelength", 1550e-9)  # m (telecom)
        
        if fiber_type == "step_index_multimode":
            core_diameter = task.get("core_diameter", 50e-6)  # m
            core_index = task.get("core_index", 1.465)
            cladding_index = task.get("cladding_index", 1.460)
            fiber_length = task.get("fiber_length", 1000)  # m
            
            # Numerical aperture
            numerical_aperture = np.sqrt(core_index**2 - cladding_index**2)
            
            # Number of modes
            v_number = (np.pi * core_diameter * numerical_aperture) / wavelength
            num_modes = v_number**2 / 2  # Approximate for step-index
            
            # Acceptance angle
            acceptance_angle = np.arcsin(numerical_aperture)
            
            # Modal dispersion
            max_ray_angle = acceptance_angle
            slowest_ray_time = fiber_length * core_index / self.constants["c"] * (1 + 0.5 * (numerical_aperture)**2)
            fastest_ray_time = fiber_length * core_index / self.constants["c"]
            modal_dispersion = slowest_ray_time - fastest_ray_time
            
            # Bandwidth-length product (rough estimate)
            bandwidth_length = 0.5 / modal_dispersion  # Hz·m
            
            return {
                "fiber_type": fiber_type,
                "core_diameter": core_diameter * 1e6,  # μm
                "numerical_aperture": numerical_aperture,
                "v_number": v_number,
                "number_of_modes": int(num_modes),
                "acceptance_angle": np.degrees(acceptance_angle),
                "modal_dispersion": modal_dispersion * 1e9,  # ns
                "bandwidth_length_product": bandwidth_length / 1e6,  # MHz·m
                "fiber_classification": "multimode",
                "units": {
                    "diameter": "μm",
                    "angle": "degrees",
                    "dispersion": "ns",
                    "bandwidth": "MHz·m"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif fiber_type == "single_mode":
            core_diameter = task.get("core_diameter", 8e-6)  # m
            core_index = task.get("core_index", 1.468)
            cladding_index = task.get("cladding_index", 1.463)
            
            # Numerical aperture
            numerical_aperture = np.sqrt(core_index**2 - cladding_index**2)
            
            # V-number
            v_number = (np.pi * core_diameter * numerical_aperture) / wavelength
            
            # Single mode condition: V < 2.405
            single_mode = v_number < 2.405
            
            # Mode field diameter
            if single_mode:
                # Gaussian approximation
                mode_field_diameter = core_diameter * (0.65 + 1.619 * v_number**(-3/2) + 2.879 * v_number**(-6))
            else:
                mode_field_diameter = core_diameter
            
            # Chromatic dispersion (simplified)
            material_dispersion = 17  # ps/(nm·km) at 1550 nm
            waveguide_dispersion = -5  # ps/(nm·km)
            total_dispersion = material_dispersion + waveguide_dispersion
            
            return {
                "fiber_type": fiber_type,
                "core_diameter": core_diameter * 1e6,  # μm
                "v_number": v_number,
                "single_mode_operation": single_mode,
                "mode_field_diameter": mode_field_diameter * 1e6,  # μm
                "numerical_aperture": numerical_aperture,
                "chromatic_dispersion": total_dispersion,
                "dispersion_components": {
                    "material": material_dispersion,
                    "waveguide": waveguide_dispersion
                },
                "units": {
                    "diameter": "μm",
                    "dispersion": "ps/(nm·km)"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "fiber_type": fiber_type,
                "error": f"Unknown fiber type: {fiber_type}",
                "available_types": ["step_index_multimode", "single_mode"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_interferometry(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interferometric measurements."""
        interferometer_type = task.get("type", "michelson")
        wavelength = task.get("wavelength", 632.8e-9)  # m
        
        if interferometer_type == "michelson":
            path_difference = task.get("path_difference", 1e-6)  # m
            
            # Phase difference
            phase_difference = 2 * np.pi * path_difference / wavelength
            
            # Fringe visibility
            intensity_ratio = task.get("intensity_ratio", 1.0)  # I1/I2
            visibility = 2 * np.sqrt(intensity_ratio) / (1 + intensity_ratio)
            
            # Interference pattern
            if phase_difference % (2 * np.pi) < np.pi:
                interference = "constructive"
                intensity_factor = (1 + visibility)**2
            else:
                interference = "destructive"
                intensity_factor = (1 - visibility)**2
            
            # Measurement sensitivity
            fringe_shift_per_meter = 2 / wavelength
            
            return {
                "interferometer_type": interferometer_type,
                "wavelength": wavelength * 1e9,  # nm
                "path_difference": path_difference * 1e6,  # μm
                "phase_difference": np.degrees(phase_difference),
                "visibility": visibility,
                "interference_type": interference,
                "relative_intensity": intensity_factor,
                "measurement_sensitivity": fringe_shift_per_meter / 1e6,  # fringes/μm
                "units": {
                    "wavelength": "nm",
                    "path": "μm",
                    "phase": "degrees"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif interferometer_type == "fabry_perot":
            cavity_length = task.get("cavity_length", 1e-3)  # m
            reflectivity = task.get("reflectivity", 0.95)
            
            # Free spectral range
            free_spectral_range = wavelength**2 / (2 * cavity_length)
            
            # Finesse
            finesse = np.pi * np.sqrt(reflectivity) / (1 - reflectivity)
            
            # Resolution
            resolution = wavelength / (finesse * 2 * cavity_length / wavelength)
            
            return {
                "interferometer_type": interferometer_type,
                "cavity_length": cavity_length * 1000,  # mm
                "finesse": finesse,
                "free_spectral_range": free_spectral_range * 1e12,  # pm
                "resolution": resolution,
                "units": {
                    "length": "mm",
                    "spectral_range": "pm"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "interferometer_type": interferometer_type,
                "error": f"Unknown interferometer type: {interferometer_type}",
                "available_types": ["michelson", "fabry_perot"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_polarization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze polarization properties."""
        polarization_element = task.get("element", "linear_polarizer")
        incident_polarization = task.get("incident_polarization", "unpolarized")
        
        if polarization_element == "linear_polarizer":
            transmission_axis = task.get("transmission_axis", 0)  # degrees
            incident_angle = task.get("incident_angle", 0)  # degrees from transmission axis
            
            if incident_polarization == "unpolarized":
                # Malus's law for unpolarized light
                transmitted_intensity = 0.5  # Half intensity transmitted
                polarization_state = "linear"
            elif incident_polarization == "linear":
                # Malus's law
                transmitted_intensity = np.cos(np.radians(incident_angle))**2
                polarization_state = "linear"
            else:
                transmitted_intensity = 0.5  # Default
                polarization_state = "linear"
            
            return {
                "polarization_element": polarization_element,
                "incident_polarization": incident_polarization,
                "transmission_axis": transmission_axis,
                "incident_angle": incident_angle,
                "transmitted_intensity": transmitted_intensity,
                "output_polarization": polarization_state,
                "extinction_ratio": 1000 if transmitted_intensity > 0.1 else float('inf'),
                "convergence": True,
                "physical_validity": True
            }
        
        elif polarization_element == "quarter_wave_plate":
            fast_axis = task.get("fast_axis", 45)  # degrees
            wavelength = task.get("wavelength", 632.8e-9)  # m
            
            # Phase retardation
            phase_retardation = np.pi / 2  # Quarter wave
            
            if incident_polarization == "linear" and fast_axis == 45:
                output_polarization = "circular"
            elif incident_polarization == "circular":
                output_polarization = "linear"
            else:
                output_polarization = "elliptical"
            
            return {
                "polarization_element": polarization_element,
                "phase_retardation": np.degrees(phase_retardation),
                "fast_axis": fast_axis,
                "input_polarization": incident_polarization,
                "output_polarization": output_polarization,
                "wavelength_dependence": "yes",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "polarization_element": polarization_element,
                "error": f"Unknown polarization element: {polarization_element}",
                "available_elements": ["linear_polarizer", "quarter_wave_plate"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _generic_optics_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic optics calculation."""
        calculation = task.get("calculation", "snells_law")
        
        if calculation == "snells_law":
            n1 = task.get("n1", 1.0)  # Air
            n2 = task.get("n2", 1.5)  # Glass
            incident_angle = task.get("incident_angle", 30)  # degrees
            
            # Snell's law
            refracted_angle = np.degrees(np.arcsin(n1 * np.sin(np.radians(incident_angle)) / n2))
            
            # Critical angle
            if n1 > n2:
                critical_angle = np.degrees(np.arcsin(n2 / n1))
            else:
                critical_angle = 90
            
            # Total internal reflection
            total_internal_reflection = incident_angle > critical_angle if n1 > n2 else False
            
            return {
                "calculation": calculation,
                "refractive_indices": [n1, n2],
                "incident_angle": incident_angle,
                "refracted_angle": refracted_angle,
                "critical_angle": critical_angle,
                "total_internal_reflection": total_internal_reflection,
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
            "optics", "photonics", "laser", "light", "optical", "photon",
            "wavelength", "diffraction", "interference", "polarization",
            "fiber optics", "nonlinear optics", "ray tracing", "lens"
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            "electromagnetic", "wave", "reflection", "refraction", "mirror",
            "prism", "spectrum", "coherent", "beam", "cavity"
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
        Estimate computational cost for an optics/photonics task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "ray_tracing")
        
        # Base costs by task type
        base_costs = {
            "ray_tracing": {"time": 1, "memory": 256, "units": 3},
            "wave_optics": {"time": 3, "memory": 512, "units": 8},
            "laser_physics": {"time": 2, "memory": 512, "units": 6},
            "nonlinear_optics": {"time": 4, "memory": 1024, "units": 12},
            "fiber_optics": {"time": 2, "memory": 512, "units": 5},
            "interferometry": {"time": 2, "memory": 512, "units": 6},
            "polarization": {"time": 1, "memory": 256, "units": 2}
        }
        
        base_cost = base_costs.get(task_type, base_costs["ray_tracing"])
        
        # Scale based on problem complexity
        scale_factor = 1.0
        
        if task_type == "wave_optics":
            # More apertures or complex geometries increase cost
            if task.get("phenomenon") == "circular_aperture_diffraction":
                scale_factor = 1.5
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for optics/photonics calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Wave theory",
                "Electromagnetic theory",
                "Fourier optics",
                "Complex analysis"
            ],
            "computational_methods": [
                "Ray tracing algorithms",
                "Fourier transform methods",
                "Finite-difference time-domain",
                "Beam propagation methods"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Optical system design",
                "Laser development",
                "Fiber optic communications",
                "Precision measurements",
                "Photonic device design"
            ],
            "accuracy_considerations": [
                "Wavelength-dependent effects",
                "Nonlinear effects at high power",
                "Polarization state changes",
                "Coherence effects important"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for optics/photonics calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check wavelength
        wavelength = task.get("wavelength")
        if wavelength and (wavelength <= 0 or wavelength > 1e-3):
            if wavelength <= 0:
                errors.append("Wavelength must be positive")
            else:
                warnings.append("Unusual wavelength value - check units")
        
        # Check refractive indices
        for key in ["n1", "n2", "core_index", "cladding_index"]:
            n = task.get(key)
            if n and n < 1:
                warnings.append(f"Refractive index {key} < 1 is unusual for normal materials")
        
        if task_type == "ray_tracing":
            focal_length = task.get("focal_length")
            object_distance = task.get("object_distance")
            if focal_length and focal_length == 0:
                errors.append("Focal length cannot be zero")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }