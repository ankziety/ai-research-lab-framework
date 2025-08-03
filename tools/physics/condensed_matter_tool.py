"""
Condensed Matter Physics Tool

Agent-friendly interface for condensed matter physics calculations.
Provides electronic structure, phase transitions, superconductivity,
magnetism, and solid state physics calculations.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class CondensedMatterTool(BasePhysicsTool):
    """
    Tool for condensed matter physics calculations that agents can request.
    
    Provides interfaces for:
    - Electronic band structure calculations
    - Phase transition analysis
    - Superconductivity modeling
    - Magnetic property calculations
    - Transport properties
    - Many-body systems
    """
    
    def __init__(self):
        super().__init__(
            tool_id="condensed_matter_tool",
            name="Condensed Matter Physics Tool",
            description="Perform condensed matter calculations including electronic structure, phase transitions, and many-body physics",
            physics_domain="condensed_matter",
            computational_cost_factor=3.5,  # Many-body systems are computationally intensive
            software_requirements=[
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib",   # Visualization
                "sympy",        # Symbolic mathematics (optional)
                "kwant"         # Quantum transport (optional)
            ],
            hardware_requirements={
                "min_memory": 2048,  # MB
                "recommended_memory": 8192,
                "cpu_cores": 4,
                "supports_gpu": True
            }
        )
        
        self.capabilities.extend([
            "band_structure",
            "density_of_states",
            "fermi_surface",
            "phase_transitions",
            "superconductivity",
            "magnetism",
            "transport_properties",
            "many_body_physics",
            "quantum_phase_transitions",
            "topological_phases"
        ])
        
        # Physical constants
        self.constants = {
            "hbar": 1.055e-34,  # J·s
            "kb": 1.381e-23,    # J/K
            "e": 1.602e-19,     # C
            "me": 9.109e-31,    # kg
            "bohr_radius": 5.292e-11,  # m
            "rydberg": 13.606,  # eV
            "bohr_magneton": 5.788e-5  # eV/T
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute condensed matter physics calculation.
        
        Args:
            task: Task specification with condensed matter parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with condensed matter analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "band_structure")
            
            if task_type == "band_structure":
                result = self._calculate_band_structure(task)
            elif task_type == "density_of_states":
                result = self._calculate_density_of_states(task)
            elif task_type == "phase_transition":
                result = self._analyze_phase_transition(task)
            elif task_type == "superconductivity":
                result = self._analyze_superconductivity(task)
            elif task_type == "magnetism":
                result = self._analyze_magnetism(task)
            elif task_type == "transport":
                result = self._calculate_transport_properties(task)
            elif task_type == "many_body":
                result = self._analyze_many_body_system(task)
            else:
                result = self._generic_condensed_matter_calculation(task)
            
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
            logger.error(f"Condensed matter calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _calculate_band_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate electronic band structure."""
        crystal_structure = task.get("crystal_structure", "simple_cubic")
        lattice_constant = task.get("lattice_constant", 3.0)  # Angstroms
        num_bands = task.get("num_bands", 4)
        k_points = task.get("k_points", 50)
        
        # Generate k-path (simplified for cubic lattice)
        k_path = np.linspace(0, np.pi/lattice_constant, k_points)
        
        # Simple tight-binding model for demonstration
        bands = []
        for band_idx in range(num_bands):
            energies = []
            for k in k_path:
                # Simple cosine dispersion
                t = -1.0  # Hopping parameter (eV)
                epsilon_0 = band_idx * 2.0  # Band offset
                energy = epsilon_0 + 2 * t * np.cos(k * lattice_constant)
                energies.append(energy)
            bands.append(energies)
        
        # Calculate band gap
        valence_band = np.array(bands[num_bands//2 - 1])
        conduction_band = np.array(bands[num_bands//2])
        
        band_gap = np.min(conduction_band) - np.max(valence_band)
        
        # Classify material
        if band_gap > 3.0:
            material_type = "insulator"
        elif band_gap > 0.1:
            material_type = "semiconductor"
        elif band_gap > -0.1:
            material_type = "semimetal"
        else:
            material_type = "metal"
        
        return {
            "crystal_structure": crystal_structure,
            "lattice_constant": lattice_constant,
            "num_bands": num_bands,
            "num_k_points": k_points,
            "k_path": k_path.tolist()[:10],  # First 10 points for brevity
            "band_energies": [band[:10] for band in bands],  # First 10 energies per band
            "band_gap": band_gap,
            "material_type": material_type,
            "valence_band_max": float(np.max(valence_band)),
            "conduction_band_min": float(np.min(conduction_band)),
            "units": {
                "energy": "eV",
                "k_space": "1/Angstrom",
                "lattice": "Angstrom"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _calculate_density_of_states(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate electronic density of states."""
        system_type = task.get("system_type", "2D_electron_gas")
        fermi_energy = task.get("fermi_energy", 1.0)  # eV
        temperature = task.get("temperature", 300)  # K
        
        # Energy range
        energy_range = np.linspace(-2, 4, 100)  # eV
        
        if system_type == "2D_electron_gas":
            # 2D density of states (constant)
            dos_values = np.full_like(energy_range, 1.0)  # States per eV per unit area
            
        elif system_type == "3D_free_electron":
            # 3D free electron gas
            dos_values = []
            for E in energy_range:
                if E > 0:
                    dos = (1/2) * (2*self.constants["me"])**(3/2) * np.sqrt(E) / (np.pi**2 * self.constants["hbar"]**3)
                    dos_values.append(dos * 1e-40)  # Normalize
                else:
                    dos_values.append(0)
            dos_values = np.array(dos_values)
            
        elif system_type == "1D_nanowire":
            # 1D density of states
            dos_values = []
            for E in energy_range:
                if E > 0:
                    dos = np.sqrt(2 * self.constants["me"]) / (np.pi * self.constants["hbar"] * np.sqrt(E))
                    dos_values.append(dos * 1e-20)  # Normalize
                else:
                    dos_values.append(0)
            dos_values = np.array(dos_values)
            
        else:
            # Generic parabolic DOS
            dos_values = np.maximum(0, energy_range**1.5)
        
        # Calculate Fermi-Dirac distribution
        kb_T = self.constants["kb"] * temperature / self.constants["e"]  # eV
        fermi_dirac = 1 / (1 + np.exp((energy_range - fermi_energy) / kb_T))
        
        # Occupied density of states
        occupied_dos = dos_values * fermi_dirac
        
        # Total number of electrons
        total_electrons = np.trapz(occupied_dos, energy_range)
        
        return {
            "system_type": system_type,
            "fermi_energy": fermi_energy,
            "temperature": temperature,
            "energy_range": energy_range[:20].tolist(),  # First 20 points
            "density_of_states": dos_values[:20].tolist(),
            "fermi_dirac_distribution": fermi_dirac[:20].tolist(),
            "occupied_dos": occupied_dos[:20].tolist(),
            "total_electrons": total_electrons,
            "units": {
                "energy": "eV",
                "temperature": "K",
                "dos": "states/eV"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_phase_transition(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze phase transitions in condensed matter systems."""
        transition_type = task.get("transition_type", "magnetic")
        temperature = task.get("temperature", 300)  # K
        material = task.get("material", "iron")
        
        if transition_type == "magnetic":
            # Magnetic phase transitions (simplified Ising model)
            critical_temperatures = {
                "iron": 1043,      # K
                "nickel": 631,     # K
                "cobalt": 1388,    # K
                "gadolinium": 293  # K
            }
            
            tc = critical_temperatures.get(material, 500)
            
            # Order parameter (magnetization)
            if temperature < tc:
                # Below critical temperature
                reduced_temp = temperature / tc
                magnetization = (1 - reduced_temp)**(1/8)  # Critical exponent β ≈ 1/8
                phase = "ferromagnetic"
            else:
                magnetization = 0
                phase = "paramagnetic"
            
            # Susceptibility
            if temperature != tc:
                susceptibility = 1 / abs(temperature - tc)**(4/3)  # Critical exponent γ ≈ 4/3
            else:
                susceptibility = float('inf')
            
            return {
                "transition_type": transition_type,
                "material": material,
                "temperature": temperature,
                "critical_temperature": tc,
                "phase": phase,
                "order_parameter": magnetization,
                "magnetic_susceptibility": min(susceptibility, 1e6),  # Cap for numerical reasons
                "reduced_temperature": temperature / tc,
                "critical_exponents": {
                    "beta": 1/8,   # Order parameter
                    "gamma": 4/3,  # Susceptibility
                    "alpha": 0     # Specific heat (3D Ising)
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif transition_type == "superconducting":
            # Superconducting transition
            critical_temperatures = {
                "aluminum": 1.2,      # K
                "lead": 7.2,          # K
                "niobium": 9.3,       # K
                "ybco": 93,           # K (high-Tc)
                "mercury_cuprate": 133 # K (high-Tc)
            }
            
            tc = critical_temperatures.get(material, 10)
            
            if temperature < tc:
                # BCS gap equation (simplified)
                delta = 1.76 * self.constants["kb"] * tc * np.sqrt(1 - temperature/tc)
                resistance = 0
                phase = "superconducting"
            else:
                delta = 0
                resistance = 1  # Normalized
                phase = "normal"
            
            return {
                "transition_type": transition_type,
                "material": material,
                "temperature": temperature,
                "critical_temperature": tc,
                "phase": phase,
                "superconducting_gap": delta / self.constants["e"] * 1000,  # meV
                "resistance": resistance,
                "coherence_length": 100e-9,  # m (typical)
                "penetration_depth": 50e-9,  # m (typical)
                "convergence": True,
                "physical_validity": True
            }
        
        elif transition_type == "metal_insulator":
            # Metal-insulator transition (Mott transition)
            correlation_strength = task.get("correlation_strength", 1.0)
            bandwidth = task.get("bandwidth", 2.0)  # eV
            
            # Simplified Mott criterion
            mott_ratio = correlation_strength / bandwidth
            critical_ratio = 1.5
            
            if mott_ratio > critical_ratio:
                phase = "insulating"
                gap = 0.5 * (correlation_strength - bandwidth)
                conductivity = 1e-10  # Very small
            else:
                phase = "metallic"
                gap = 0
                conductivity = 1.0  # Normalized
            
            return {
                "transition_type": transition_type,
                "correlation_strength": correlation_strength,
                "bandwidth": bandwidth,
                "mott_ratio": mott_ratio,
                "critical_ratio": critical_ratio,
                "phase": phase,
                "energy_gap": gap,
                "electrical_conductivity": conductivity,
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "transition_type": transition_type,
                "error": f"Unknown transition type: {transition_type}",
                "available_types": ["magnetic", "superconducting", "metal_insulator"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_superconductivity(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze superconducting properties."""
        superconductor_type = task.get("type", "conventional")
        temperature = task.get("temperature", 4.2)  # K
        material = task.get("material", "niobium")
        
        # Superconductor properties database
        sc_properties = {
            "aluminum": {"tc": 1.2, "type": "conventional", "gap_ratio": 3.5},
            "lead": {"tc": 7.2, "type": "conventional", "gap_ratio": 3.5},
            "niobium": {"tc": 9.3, "type": "conventional", "gap_ratio": 3.5},
            "ybco": {"tc": 93, "type": "unconventional", "gap_ratio": 5.0},
            "bscco": {"tc": 110, "type": "unconventional", "gap_ratio": 6.0},
            "iron_pnictide": {"tc": 55, "type": "unconventional", "gap_ratio": 4.0}
        }
        
        if material not in sc_properties:
            return {
                "material": material,
                "error": f"Superconductor data not available for {material}",
                "available_materials": list(sc_properties.keys()),
                "convergence": False,
                "physical_validity": False
            }
        
        props = sc_properties[material]
        tc = props["tc"]
        gap_ratio = props["gap_ratio"]
        
        # BCS calculations
        if temperature < tc:
            # Gap equation
            delta_0 = gap_ratio * self.constants["kb"] * tc / self.constants["e"] * 1000  # meV
            delta_t = delta_0 * np.sqrt(1 - temperature/tc) if temperature < tc else 0
            
            # Thermodynamic properties
            specific_heat = np.exp(-delta_t * self.constants["e"] / (self.constants["kb"] * temperature))
            
            # London penetration depth
            lambda_0 = 100e-9  # m (typical)
            lambda_t = lambda_0 / np.sqrt(1 - (temperature/tc)**4)
            
            # Coherence length
            xi_0 = self.constants["hbar"] * 1e8 / (2 * np.pi * self.constants["kb"] * tc)  # nm
            
            phase = "superconducting"
        else:
            delta_t = 0
            specific_heat = 1.0  # Normal state
            lambda_t = float('inf')
            xi_0 = 0
            phase = "normal"
        
        # Critical fields (Type II superconductor)
        hc1 = 0.01  # T (typical)
        hc2 = 10    # T (typical)
        
        return {
            "material": material,
            "superconductor_type": props["type"],
            "temperature": temperature,
            "critical_temperature": tc,
            "phase": phase,
            "superconducting_gap": delta_t,
            "gap_ratio": gap_ratio,
            "specific_heat": specific_heat,
            "penetration_depth": lambda_t,
            "coherence_length": xi_0,
            "lower_critical_field": hc1,
            "upper_critical_field": hc2,
            "units": {
                "temperature": "K",
                "gap": "meV",
                "length": "m",
                "field": "T"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_magnetism(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze magnetic properties."""
        magnetic_system = task.get("system", "ferromagnet")
        temperature = task.get("temperature", 300)  # K
        magnetic_field = task.get("magnetic_field", 0.1)  # T
        
        if magnetic_system == "ferromagnet":
            # Ferromagnetic system (Heisenberg model)
            exchange_coupling = task.get("exchange_coupling", 10)  # meV
            tc = exchange_coupling * 1e-3 * self.constants["e"] / self.constants["kb"]  # Convert to K
            
            if temperature < tc:
                magnetization = (1 - temperature/tc)**(1/3)  # Mean field approximation
                susceptibility = magnetization / magnetic_field
            else:
                magnetization = magnetic_field / temperature  # Curie law
                susceptibility = 1 / temperature
            
            return {
                "magnetic_system": magnetic_system,
                "temperature": temperature,
                "magnetic_field": magnetic_field,
                "curie_temperature": tc,
                "exchange_coupling": exchange_coupling,
                "magnetization": magnetization,
                "magnetic_susceptibility": susceptibility,
                "magnetic_phase": "ferromagnetic" if temperature < tc else "paramagnetic",
                "convergence": True,
                "physical_validity": True
            }
        
        elif magnetic_system == "antiferromagnet":
            # Antiferromagnetic system
            exchange_coupling = task.get("exchange_coupling", -5)  # meV (negative)
            tn = abs(exchange_coupling) * 1e-3 * self.constants["e"] / self.constants["kb"]  # Neel temperature
            
            if temperature < tn:
                staggered_magnetization = (1 - temperature/tn)**(1/3)
                susceptibility = 0.1 / temperature  # Reduced susceptibility
            else:
                staggered_magnetization = 0
                susceptibility = 1 / (temperature + abs(exchange_coupling))  # Curie-Weiss
            
            return {
                "magnetic_system": magnetic_system,
                "temperature": temperature,
                "neel_temperature": tn,
                "exchange_coupling": exchange_coupling,
                "staggered_magnetization": staggered_magnetization,
                "magnetic_susceptibility": susceptibility,
                "magnetic_phase": "antiferromagnetic" if temperature < tn else "paramagnetic",
                "convergence": True,
                "physical_validity": True
            }
        
        elif magnetic_system == "spin_glass":
            # Spin glass system
            freezing_temperature = task.get("freezing_temperature", 20)  # K
            
            if temperature < freezing_temperature:
                order_parameter = (1 - temperature/freezing_temperature)**(1/2)
                susceptibility = 1 / np.sqrt(temperature)
                phase = "spin_glass"
            else:
                order_parameter = 0
                susceptibility = 1 / temperature
                phase = "paramagnetic"
            
            return {
                "magnetic_system": magnetic_system,
                "temperature": temperature,
                "freezing_temperature": freezing_temperature,
                "spin_glass_order": order_parameter,
                "magnetic_susceptibility": susceptibility,
                "magnetic_phase": phase,
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "magnetic_system": magnetic_system,
                "error": f"Unknown magnetic system: {magnetic_system}",
                "available_systems": ["ferromagnet", "antiferromagnet", "spin_glass"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _calculate_transport_properties(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transport properties."""
        transport_type = task.get("type", "electrical")
        temperature = task.get("temperature", 300)  # K
        material = task.get("material", "copper")
        
        if transport_type == "electrical":
            # Electrical conductivity
            resistivities = {  # Ohm·m at room temperature
                "copper": 1.68e-8,
                "aluminum": 2.65e-8,
                "gold": 2.44e-8,
                "silicon": 1e3,  # Intrinsic
                "germanium": 0.5
            }
            
            rho_0 = resistivities.get(material, 1e-6)
            
            # Temperature dependence (simplified)
            if material in ["copper", "aluminum", "gold"]:
                # Metallic behavior
                rho_t = rho_0 * (1 + 0.004 * (temperature - 300))
                conductivity = 1 / rho_t
                carrier_type = "electrons"
            else:
                # Semiconductor behavior
                gap = 1.1 if material == "silicon" else 0.7  # eV
                intrinsic_conc = 1e16 * np.exp(-gap * self.constants["e"] / (2 * self.constants["kb"] * temperature))
                conductivity = intrinsic_conc * self.constants["e"] * 0.1  # Simplified
                carrier_type = "electrons and holes"
            
            # Mobility (simplified)
            mobility = conductivity / (intrinsic_conc * self.constants["e"]) if 'intrinsic_conc' in locals() else 1e-3
            
            return {
                "transport_type": transport_type,
                "material": material,
                "temperature": temperature,
                "electrical_conductivity": conductivity,
                "electrical_resistivity": 1/conductivity,
                "carrier_type": carrier_type,
                "carrier_mobility": mobility,
                "units": {
                    "conductivity": "S/m",
                    "resistivity": "Ohm·m", 
                    "mobility": "m²/V·s"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif transport_type == "thermal":
            # Thermal conductivity
            thermal_conductivities = {  # W/m·K at room temperature
                "copper": 401,
                "aluminum": 237,
                "silicon": 149,
                "diamond": 2000
            }
            
            k_0 = thermal_conductivities.get(material, 100)
            
            # Wiedemann-Franz law for metals
            if material in ["copper", "aluminum"]:
                lorenz_number = 2.44e-8  # W·Ohm/K²
                electrical_cond = 1 / resistivities.get(material, 1e-6)
                thermal_cond = lorenz_number * electrical_cond * temperature
            else:
                # Temperature dependence for non-metals
                thermal_cond = k_0 * (300/temperature)**(1/2)
            
            return {
                "transport_type": transport_type,
                "material": material,
                "temperature": temperature,
                "thermal_conductivity": thermal_cond,
                "lorenz_number": 2.44e-8 if material in ["copper", "aluminum"] else None,
                "units": {
                    "thermal_conductivity": "W/m·K"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "transport_type": transport_type,
                "error": f"Unknown transport type: {transport_type}",
                "available_types": ["electrical", "thermal"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_many_body_system(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze many-body quantum systems."""
        system_type = task.get("system", "hubbard_model")
        lattice_size = task.get("lattice_size", 4)
        interaction_strength = task.get("interaction_strength", 1.0)
        
        if system_type == "hubbard_model":
            # Hubbard model on a square lattice
            hopping = task.get("hopping", 1.0)
            filling = task.get("filling", 0.5)  # Half-filling
            
            # Phase diagram (simplified)
            if interaction_strength / hopping > 3.8:  # Critical ratio for square lattice
                phase = "mott_insulator"
                gap = 0.5 * interaction_strength
                correlation_length = 1.0  # Lattice constants
            else:
                phase = "metallic"
                gap = 0
                correlation_length = lattice_size / 2
            
            # Ground state energy (rough estimate)
            kinetic_energy = -4 * hopping * filling * (1 - filling)
            interaction_energy = 0.5 * interaction_strength * filling**2
            total_energy = kinetic_energy + interaction_energy
            
            return {
                "system_type": system_type,
                "lattice_size": lattice_size,
                "hopping_parameter": hopping,
                "interaction_strength": interaction_strength,
                "filling": filling,
                "phase": phase,
                "energy_gap": gap,
                "ground_state_energy": total_energy,
                "correlation_length": correlation_length,
                "kinetic_energy": kinetic_energy,
                "interaction_energy": interaction_energy,
                "convergence": True,
                "physical_validity": True
            }
        
        elif system_type == "quantum_spin_chain":
            # Quantum spin chain (Heisenberg model)
            spin = task.get("spin", 0.5)
            chain_length = lattice_size
            
            # Ground state properties
            if spin == 0.5:
                # Spin-1/2 chain has gapless excitations
                energy_per_site = -0.443 * interaction_strength  # Exact result
                gap = 0
                correlation_function = "power_law"
            else:
                # Integer spin chains have gaps (Haldane gap)
                energy_per_site = -0.4 * interaction_strength
                gap = 0.41 * interaction_strength  # Haldane gap
                correlation_function = "exponential"
            
            return {
                "system_type": system_type,
                "chain_length": chain_length,
                "spin": spin,
                "coupling_strength": interaction_strength,
                "ground_state_energy_per_site": energy_per_site,
                "energy_gap": gap,
                "correlation_function": correlation_function,
                "total_ground_state_energy": energy_per_site * chain_length,
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "system_type": system_type,
                "error": f"Unknown many-body system: {system_type}",
                "available_systems": ["hubbard_model", "quantum_spin_chain"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _generic_condensed_matter_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic condensed matter calculation."""
        calculation = task.get("calculation", "fermi_energy")
        
        if calculation == "fermi_energy":
            electron_density = task.get("electron_density", 1e28)  # m^-3
            
            # Free electron model
            fermi_wavevector = (3 * np.pi**2 * electron_density)**(1/3)
            fermi_energy = (self.constants["hbar"]**2 * fermi_wavevector**2) / (2 * self.constants["me"])
            fermi_energy_eV = fermi_energy / self.constants["e"]
            
            return {
                "calculation": calculation,
                "electron_density": electron_density,
                "fermi_wavevector": fermi_wavevector,
                "fermi_energy": fermi_energy_eV,
                "units": {
                    "density": "m^-3",
                    "wavevector": "m^-1",
                    "energy": "eV"
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
            "condensed matter", "solid state", "band structure", "electronic structure",
            "superconductivity", "magnetism", "phase transition", "transport", 
            "fermi", "density of states", "many-body", "hubbard", "electronic",
            "ferromagnetic", "antiferromagnetic", "metal", "insulator", "semiconductor"
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            "crystal", "lattice", "phonon", "electron", "conductivity", "resistance",
            "magnetic", "thermal", "quantum", "correlation", "transition"
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
        Estimate computational cost for a condensed matter task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "band_structure")
        
        # Base costs by task type
        base_costs = {
            "band_structure": {"time": 5, "memory": 1024, "units": 15},
            "density_of_states": {"time": 2, "memory": 512, "units": 8},
            "phase_transition": {"time": 3, "memory": 512, "units": 10},
            "superconductivity": {"time": 1, "memory": 256, "units": 5},
            "magnetism": {"time": 2, "memory": 512, "units": 7},
            "transport": {"time": 1, "memory": 256, "units": 3},
            "many_body": {"time": 10, "memory": 2048, "units": 25}
        }
        
        base_cost = base_costs.get(task_type, base_costs["band_structure"])
        
        # Scale based on system size
        scale_factor = 1.0
        
        if task_type == "band_structure":
            k_points = task.get("k_points", 50)
            num_bands = task.get("num_bands", 4)
            scale_factor = (k_points * num_bands) / (50 * 4)
        elif task_type == "many_body":
            lattice_size = task.get("lattice_size", 4)
            scale_factor = (lattice_size / 4)**3  # Exponential scaling
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for condensed matter calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Quantum mechanics",
                "Statistical mechanics",
                "Solid state physics",
                "Many-body theory basics"
            ],
            "computational_methods": [
                "Tight-binding models",
                "Density functional theory",
                "Monte Carlo methods",
                "Green's function techniques"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Electronic device design",
                "Superconductor research",
                "Magnetic material development",
                "Quantum phase transition studies",
                "Transport property prediction"
            ],
            "accuracy_considerations": [
                "Finite size effects important",
                "Many-body corrections significant",
                "Temperature effects on properties",
                "Disorder and impurity effects"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for condensed matter calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if task_type == "band_structure":
            k_points = task.get("k_points", 50)
            if k_points > 1000:
                warnings.append(f"Large number of k-points ({k_points}) may be slow")
            
            num_bands = task.get("num_bands", 4)
            if num_bands > 20:
                warnings.append(f"Many bands ({num_bands}) increases computation time")
        
        elif task_type == "many_body":
            lattice_size = task.get("lattice_size", 4)
            if lattice_size > 8:
                warnings.append(f"Large lattice ({lattice_size}) has exponential cost")
            
            interaction = task.get("interaction_strength", 1.0)
            if interaction < 0:
                warnings.append("Negative interaction strength may indicate attractive interactions")
        
        elif task_type == "phase_transition":
            temperature = task.get("temperature", 300)
            if temperature < 0:
                errors.append("Temperature must be positive")
            if temperature > 10000:
                warnings.append("Very high temperature - may need relativistic corrections")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }