"""
Nuclear Physics Tool

Agent-friendly interface for nuclear physics calculations.
Provides nuclear structure analysis, radioactive decay modeling,
nuclear reactions, and nuclear astrophysics calculations.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class NuclearPhysicsTool(BasePhysicsTool):
    """
    Tool for nuclear physics calculations that agents can request.
    
    Provides interfaces for:
    - Nuclear structure calculations
    - Radioactive decay analysis
    - Nuclear reaction modeling
    - Binding energy calculations
    - Fission and fusion processes
    - Nuclear astrophysics
    """
    
    def __init__(self):
        super().__init__(
            tool_id="nuclear_physics_tool",
            name="Nuclear Physics Tool",
            description="Perform nuclear physics calculations including nuclear structure, decay processes, and nuclear reactions",
            physics_domain="nuclear_physics",
            computational_cost_factor=2.5,
            software_requirements=[
                "scipy",        # Mathematical functions
                "numpy",        # Core calculations
                "matplotlib",   # Visualization
                "sympy"         # Symbolic mathematics (optional)
            ],
            hardware_requirements={
                "min_memory": 1024,  # MB
                "recommended_memory": 4096,
                "cpu_cores": 4,
                "supports_gpu": False
            }
        )
        
        self.capabilities.extend([
            "nuclear_structure",
            "radioactive_decay",
            "nuclear_reactions",
            "binding_energy",
            "mass_defect",
            "nuclear_fission",
            "nuclear_fusion",
            "nuclear_astrophysics",
            "isotope_analysis"
        ])
        
        # Nuclear physics constants
        self.constants = {
            "atomic_mass_unit": 931.494,  # MeV/c²
            "avogadro": 6.022e23,  # mol⁻¹
            "nuclear_radius_const": 1.2,  # fm
            "alpha": 7.297e-3,  # Fine structure constant
            "electron_mass": 0.511,  # MeV/c²
            "proton_mass": 938.272,  # MeV/c²
            "neutron_mass": 939.565,  # MeV/c²
            "hbar_c": 197.327  # MeV·fm
        }
        
        # Nuclear data (simplified database)
        self.nuclear_data = {
            "H1": {"mass": 1.007825, "Z": 1, "N": 0, "binding_energy": 0},
            "H2": {"mass": 2.014102, "Z": 1, "N": 1, "binding_energy": 2.225},
            "H3": {"mass": 3.016049, "Z": 1, "N": 2, "binding_energy": 8.482},
            "He3": {"mass": 3.016029, "Z": 2, "N": 1, "binding_energy": 7.718},
            "He4": {"mass": 4.002603, "Z": 2, "N": 2, "binding_energy": 28.296},
            "Li6": {"mass": 6.015122, "Z": 3, "N": 3, "binding_energy": 31.995},
            "Li7": {"mass": 7.016004, "Z": 3, "N": 4, "binding_energy": 39.245},
            "C12": {"mass": 12.000000, "Z": 6, "N": 6, "binding_energy": 92.162},
            "C14": {"mass": 14.003242, "Z": 6, "N": 8, "binding_energy": 105.285},
            "O16": {"mass": 15.994915, "Z": 8, "N": 8, "binding_energy": 127.619},
            "Fe56": {"mass": 55.934942, "Z": 26, "N": 30, "binding_energy": 492.254},
            "U235": {"mass": 235.043923, "Z": 92, "N": 143, "binding_energy": 1783.870},
            "U238": {"mass": 238.050788, "Z": 92, "N": 146, "binding_energy": 1801.695}
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute nuclear physics calculation.
        
        Args:
            task: Task specification with nuclear physics parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with nuclear physics analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "binding_energy")
            
            if task_type == "binding_energy":
                result = self._calculate_binding_energy(task)
            elif task_type == "radioactive_decay":
                result = self._analyze_radioactive_decay(task)
            elif task_type == "nuclear_reaction":
                result = self._calculate_nuclear_reaction(task)
            elif task_type == "nuclear_structure":
                result = self._analyze_nuclear_structure(task)
            elif task_type == "fission":
                result = self._analyze_fission(task)
            elif task_type == "fusion":
                result = self._analyze_fusion(task)
            elif task_type == "nuclear_astrophysics":
                result = self._calculate_nuclear_astrophysics(task)
            else:
                result = self._generic_nuclear_calculation(task)
            
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
            logger.error(f"Nuclear physics calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _calculate_binding_energy(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate nuclear binding energy."""
        nucleus = task.get("nucleus", "He4")
        
        if nucleus in self.nuclear_data:
            data = self.nuclear_data[nucleus]
            
            # Mass-energy calculation
            Z = data["Z"]
            N = data["N"]
            A = Z + N
            
            # Calculate binding energy from mass defect
            nuclear_mass = data["mass"]
            constituent_mass = Z * self.constants["proton_mass"]/self.constants["atomic_mass_unit"] + \
                             N * self.constants["neutron_mass"]/self.constants["atomic_mass_unit"]
            
            mass_defect = constituent_mass - nuclear_mass
            binding_energy = mass_defect * self.constants["atomic_mass_unit"]
            binding_energy_per_nucleon = binding_energy / A
            
            # Nuclear radius
            radius = self.constants["nuclear_radius_const"] * A**(1/3)
            
            return {
                "nucleus": nucleus,
                "mass_number": A,
                "atomic_number": Z,
                "neutron_number": N,
                "nuclear_mass": nuclear_mass,
                "mass_defect": mass_defect,
                "binding_energy": binding_energy,
                "binding_energy_per_nucleon": binding_energy_per_nucleon,
                "nuclear_radius": radius,
                "units": {
                    "mass": "atomic mass units (u)",
                    "energy": "MeV",
                    "radius": "fm"
                },
                "convergence": True,
                "physical_validity": True
            }
        else:
            return {
                "nucleus": nucleus,
                "error": f"Nuclear data not available for {nucleus}",
                "available_nuclei": list(self.nuclear_data.keys()),
                "convergence": False,
                "physical_validity": False
            }
    
    def _analyze_radioactive_decay(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze radioactive decay processes."""
        parent_nucleus = task.get("parent", "C14")
        decay_type = task.get("decay_type", "beta_minus")
        time = task.get("time", None)  # seconds
        
        # Decay constants and half-lives (simplified data)
        decay_data = {
            "C14": {"half_life": 5730 * 365.25 * 24 * 3600, "decay_type": "beta_minus", "daughter": "N14"},
            "U238": {"half_life": 4.468e9 * 365.25 * 24 * 3600, "decay_type": "alpha", "daughter": "Th234"},
            "U235": {"half_life": 7.04e8 * 365.25 * 24 * 3600, "decay_type": "alpha", "daughter": "Th231"},
            "H3": {"half_life": 12.32 * 365.25 * 24 * 3600, "decay_type": "beta_minus", "daughter": "He3"},
            "Ra226": {"half_life": 1600 * 365.25 * 24 * 3600, "decay_type": "alpha", "daughter": "Rn222"}
        }
        
        if parent_nucleus not in decay_data:
            return {
                "parent": parent_nucleus,
                "error": f"Decay data not available for {parent_nucleus}",
                "available_nuclei": list(decay_data.keys()),
                "convergence": False,
                "physical_validity": False
            }
        
        data = decay_data[parent_nucleus]
        half_life = data["half_life"]
        decay_constant = math.log(2) / half_life
        
        result = {
            "parent_nucleus": parent_nucleus,
            "daughter_nucleus": data["daughter"],
            "decay_type": data["decay_type"],
            "half_life": half_life,
            "half_life_years": half_life / (365.25 * 24 * 3600),
            "decay_constant": decay_constant,
            "mean_lifetime": 1 / decay_constant,
            "units": {
                "time": "seconds",
                "decay_constant": "s⁻¹"
            }
        }
        
        # Calculate decay if time is specified
        if time is not None:
            initial_nuclei = task.get("initial_nuclei", 1e12)
            remaining_nuclei = initial_nuclei * math.exp(-decay_constant * time)
            decayed_nuclei = initial_nuclei - remaining_nuclei
            activity = decay_constant * remaining_nuclei
            
            result.update({
                "time": time,
                "initial_nuclei": initial_nuclei,
                "remaining_nuclei": remaining_nuclei,
                "decayed_nuclei": decayed_nuclei,
                "fraction_remaining": remaining_nuclei / initial_nuclei,
                "activity": activity,
                "activity_units": "decays/second (Bq)"
            })
        
        result.update({
            "convergence": True,
            "physical_validity": True
        })
        
        return result
    
    def _calculate_nuclear_reaction(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate nuclear reaction energetics."""
        reaction = task.get("reaction", "H2 + H3 -> He4 + n")
        
        # Parse reaction string (simplified)
        if "->" in reaction:
            reactants_str, products_str = reaction.split("->")
            reactants = [r.strip() for r in reactants_str.split("+")]
            products = [p.strip() for p in products_str.split("+")]
        else:
            return {
                "reaction": reaction,
                "error": "Invalid reaction format. Use 'A + B -> C + D' format",
                "convergence": False,
                "physical_validity": False
            }
        
        # Map common particle names
        particle_mapping = {
            "n": "neutron",
            "p": "proton", 
            "alpha": "He4",
            "d": "H2",
            "t": "H3"
        }
        
        # Replace common names
        for i, r in enumerate(reactants):
            if r in particle_mapping:
                reactants[i] = particle_mapping[r]
        for i, p in enumerate(products):
            if p in particle_mapping:
                products[i] = particle_mapping[p]
        
        # Calculate Q-value
        reactant_mass = 0
        product_mass = 0
        
        missing_data = []
        
        # Sum reactant masses
        for reactant in reactants:
            if reactant == "neutron":
                reactant_mass += self.constants["neutron_mass"] / self.constants["atomic_mass_unit"]
            elif reactant == "proton":
                reactant_mass += self.constants["proton_mass"] / self.constants["atomic_mass_unit"]
            elif reactant in self.nuclear_data:
                reactant_mass += self.nuclear_data[reactant]["mass"]
            else:
                missing_data.append(reactant)
        
        # Sum product masses
        for product in products:
            if product == "neutron":
                product_mass += self.constants["neutron_mass"] / self.constants["atomic_mass_unit"]
            elif product == "proton":
                product_mass += self.constants["proton_mass"] / self.constants["atomic_mass_unit"]
            elif product in self.nuclear_data:
                product_mass += self.nuclear_data[product]["mass"]
            else:
                missing_data.append(product)
        
        if missing_data:
            return {
                "reaction": reaction,
                "error": f"Missing nuclear data for: {missing_data}",
                "available_data": list(self.nuclear_data.keys()) + ["neutron", "proton"],
                "convergence": False,
                "physical_validity": False
            }
        
        # Q-value calculation
        mass_difference = reactant_mass - product_mass
        q_value = mass_difference * self.constants["atomic_mass_unit"]
        
        # Reaction classification
        if q_value > 0:
            reaction_type = "exothermic"
        elif q_value < 0:
            reaction_type = "endothermic"
        else:
            reaction_type = "threshold"
        
        # Calculate threshold energy for endothermic reactions
        threshold_energy = 0
        if q_value < 0:
            # Simplified threshold calculation
            target_mass = reactant_mass / len(reactants)  # Approximation
            threshold_energy = -q_value * (1 + 1/target_mass)
        
        return {
            "reaction": reaction,
            "reactants": reactants,
            "products": products,
            "reactant_mass": reactant_mass,
            "product_mass": product_mass,
            "mass_difference": mass_difference,
            "q_value": q_value,
            "reaction_type": reaction_type,
            "threshold_energy": threshold_energy,
            "units": {
                "mass": "atomic mass units (u)",
                "energy": "MeV"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_nuclear_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nuclear structure properties."""
        nucleus = task.get("nucleus", "Fe56")
        
        if nucleus not in self.nuclear_data:
            return {
                "nucleus": nucleus,
                "error": f"Nuclear data not available for {nucleus}",
                "available_nuclei": list(self.nuclear_data.keys()),
                "convergence": False,
                "physical_validity": False
            }
        
        data = self.nuclear_data[nucleus]
        Z = data["Z"]
        N = data["N"]
        A = Z + N
        
        # Nuclear properties
        radius = self.constants["nuclear_radius_const"] * A**(1/3)
        density = A * self.constants["atomic_mass_unit"] / ((4/3) * math.pi * radius**3)
        
        # Shell model predictions (simplified)
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        proton_shell = "closed" if Z in magic_numbers else "open"
        neutron_shell = "closed" if N in magic_numbers else "open"
        
        # Nuclear stability
        n_to_p_ratio = N / Z if Z > 0 else float('inf')
        
        # Empirical stability line (N = Z for light nuclei, N > Z for heavy)
        if A < 40:
            stable_ratio = 1.0
        else:
            stable_ratio = 1.0 + 0.4 * (A - 40) / 200
        
        stability = "stable" if abs(n_to_p_ratio - stable_ratio) < 0.1 else "unstable"
        
        # Pairing energy (simplified)
        if N % 2 == 0 and Z % 2 == 0:
            pairing = "even-even (most stable)"
        elif N % 2 == 1 and Z % 2 == 1:
            pairing = "odd-odd (least stable)"
        else:
            pairing = "even-odd (intermediate)"
        
        return {
            "nucleus": nucleus,
            "mass_number": A,
            "atomic_number": Z,
            "neutron_number": N,
            "nuclear_radius": radius,
            "nuclear_density": density,
            "n_to_p_ratio": n_to_p_ratio,
            "proton_shell": proton_shell,
            "neutron_shell": neutron_shell,
            "stability": stability,
            "pairing": pairing,
            "magic_numbers": {
                "proton_magic": Z in magic_numbers,
                "neutron_magic": N in magic_numbers,
                "doubly_magic": Z in magic_numbers and N in magic_numbers
            },
            "units": {
                "radius": "fm",
                "density": "kg/m³"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_fission(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nuclear fission process."""
        fissile_nucleus = task.get("nucleus", "U235")
        
        # Fission data (simplified)
        fission_data = {
            "U235": {
                "fission_fragments": ["Ba140", "Kr93"],
                "neutrons_released": 2.4,
                "energy_released": 200,  # MeV
                "fission_probability": 0.85
            },
            "U238": {
                "fission_fragments": ["Ba144", "Kr90"],
                "neutrons_released": 2.2,
                "energy_released": 190,  # MeV
                "fission_probability": 0.1  # Requires fast neutrons
            },
            "Pu239": {
                "fission_fragments": ["Ba142", "Kr91"],
                "neutrons_released": 2.9,
                "energy_released": 210,  # MeV
                "fission_probability": 0.9
            }
        }
        
        if fissile_nucleus not in fission_data:
            return {
                "nucleus": fissile_nucleus,
                "error": f"Fission data not available for {fissile_nucleus}",
                "available_nuclei": list(fission_data.keys()),
                "convergence": False,
                "physical_validity": False
            }
        
        data = fission_data[fissile_nucleus]
        
        # Critical mass calculation (simplified)
        if fissile_nucleus == "U235":
            critical_mass = 52  # kg (sphere)
        elif fissile_nucleus == "Pu239":
            critical_mass = 10  # kg (sphere)
        else:
            critical_mass = 100  # kg (estimate)
        
        # Chain reaction parameters
        neutrons_per_fission = data["neutrons_released"]
        reproduction_factor = neutrons_per_fission * data["fission_probability"]
        
        return {
            "fissile_nucleus": fissile_nucleus,
            "typical_fragments": data["fission_fragments"],
            "neutrons_per_fission": neutrons_per_fission,
            "energy_per_fission": data["energy_released"],
            "fission_probability": data["fission_probability"],
            "reproduction_factor": reproduction_factor,
            "critical_mass": critical_mass,
            "chain_reaction": "sustained" if reproduction_factor >= 1.0 else "subcritical",
            "energy_density": data["energy_released"] / 235,  # MeV per nucleon
            "units": {
                "energy": "MeV",
                "mass": "kg"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_fusion(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nuclear fusion process."""
        reaction = task.get("reaction", "D + T -> He4 + n")
        temperature = task.get("temperature", 1e8)  # K
        
        # Common fusion reactions
        fusion_reactions = {
            "D + T -> He4 + n": {
                "reactants": ["H2", "H3"],
                "products": ["He4", "neutron"],
                "q_value": 17.59,  # MeV
                "cross_section_peak": 5e-24,  # cm²
                "peak_temperature": 1e8  # K
            },
            "D + D -> T + p": {
                "reactants": ["H2", "H2"],
                "products": ["H3", "proton"],
                "q_value": 4.03,  # MeV
                "cross_section_peak": 1e-26,  # cm²
                "peak_temperature": 1e9  # K
            },
            "D + D -> He3 + n": {
                "reactants": ["H2", "H2"],
                "products": ["He3", "neutron"],
                "q_value": 3.27,  # MeV
                "cross_section_peak": 1e-26,  # cm²
                "peak_temperature": 1e9  # K
            },
            "D + He3 -> He4 + p": {
                "reactants": ["H2", "He3"],
                "products": ["He4", "proton"],
                "q_value": 18.35,  # MeV
                "cross_section_peak": 8e-25,  # cm²
                "peak_temperature": 5e8  # K
            }
        }
        
        if reaction not in fusion_reactions:
            return {
                "reaction": reaction,
                "error": f"Fusion data not available for reaction: {reaction}",
                "available_reactions": list(fusion_reactions.keys()),
                "convergence": False,
                "physical_validity": False
            }
        
        data = fusion_reactions[reaction]
        
        # Calculate reaction rate (simplified)
        kb = 8.617e-5  # eV/K
        kt = kb * temperature / 1e6  # Convert to MeV
        
        # Coulomb barrier
        Z1, Z2 = 1, 1  # For D-T reaction
        coulomb_barrier = 1.44 * Z1 * Z2 / 2  # MeV (approximate)
        
        # Gamow peak energy
        gamow_energy = 0.98 * (Z1 * Z2)**2 * (1/kt)**(1/3)  # keV
        
        # Tunneling probability (very simplified)
        tunneling_prob = math.exp(-2 * math.pi * Z1 * Z2 * math.sqrt(0.511 / (2 * gamow_energy * 1e-3)))
        
        # Reaction rate coefficient (order of magnitude)
        rate_coefficient = data["cross_section_peak"] * math.sqrt(8 * kt * 1e6 / (math.pi * 1)) * tunneling_prob
        
        # Lawson criterion
        density_time_product = 1e20  # m⁻³·s for D-T
        lawson_criterion = "satisfied" if temperature > 5e7 else "not satisfied"
        
        return {
            "reaction": reaction,
            "reactants": data["reactants"],
            "products": data["products"],
            "q_value": data["q_value"],
            "temperature": temperature,
            "coulomb_barrier": coulomb_barrier,
            "gamow_energy": gamow_energy,
            "tunneling_probability": tunneling_prob,
            "rate_coefficient": rate_coefficient,
            "peak_cross_section": data["cross_section_peak"],
            "optimal_temperature": data["peak_temperature"],
            "lawson_criterion": lawson_criterion,
            "required_density_time": density_time_product,
            "units": {
                "energy": "MeV",
                "temperature": "K",
                "cross_section": "cm²",
                "rate": "cm³/s"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _calculate_nuclear_astrophysics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate nuclear processes in astrophysical environments."""
        process = task.get("process", "pp_chain")
        stellar_temperature = task.get("temperature", 1.5e7)  # K
        stellar_density = task.get("density", 100)  # g/cm³
        
        if process == "pp_chain":
            # Proton-proton chain in stars
            steps = [
                "p + p -> d + e+ + νe",
                "d + p -> He3 + γ",
                "He3 + He3 -> He4 + 2p"
            ]
            
            total_energy = 26.73  # MeV per He4 produced
            neutrino_energy = 0.26  # MeV average
            
            # Rate depends on temperature (very simplified)
            kb = 8.617e-5  # eV/K
            kt = kb * stellar_temperature
            
            # Gamow peak for p-p reaction
            rate_factor = math.exp(-stellar_temperature / 1e7)
            
            return {
                "process": process,
                "reaction_steps": steps,
                "stellar_temperature": stellar_temperature,
                "stellar_density": stellar_density,
                "energy_per_helium": total_energy,
                "neutrino_energy": neutrino_energy,
                "relative_rate": rate_factor,
                "energy_production_rate": rate_factor * stellar_density / 100,
                "units": {
                    "temperature": "K",
                    "density": "g/cm³",
                    "energy": "MeV"
                },
                "convergence": True,
                "physical_validity": True
            }
        
        elif process == "cno_cycle":
            # CNO cycle in massive stars
            catalyst_nuclei = ["C12", "N13", "N14", "O15"]
            total_energy = 26.73  # MeV (same as pp-chain)
            
            # CNO cycle dominates at higher temperatures
            cno_efficiency = 1 if stellar_temperature > 2e7 else 0.1
            
            return {
                "process": process,
                "catalyst_nuclei": catalyst_nuclei,
                "stellar_temperature": stellar_temperature,
                "energy_per_helium": total_energy,
                "efficiency": cno_efficiency,
                "temperature_threshold": 2e7,
                "dominant_process": stellar_temperature > 2e7,
                "convergence": True,
                "physical_validity": True
            }
        
        elif process == "nucleosynthesis":
            # Big Bang nucleosynthesis
            nuclei_produced = ["H1", "H2", "He3", "He4", "Li7"]
            abundances = {
                "H1": 0.75,   # Mass fraction
                "H2": 2.5e-5,
                "He3": 1e-5,
                "He4": 0.25,
                "Li7": 4e-10
            }
            
            return {
                "process": process,
                "nuclei_produced": nuclei_produced,
                "predicted_abundances": abundances,
                "temperature_range": "1e9 - 1e8 K",
                "time_scale": "100 - 1000 seconds after Big Bang",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "process": process,
                "error": f"Unknown astrophysical process: {process}",
                "available_processes": ["pp_chain", "cno_cycle", "nucleosynthesis"],
                "convergence": False,
                "physical_validity": False
            }
    
    def _generic_nuclear_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic nuclear physics calculation."""
        calculation = task.get("calculation", "nuclear_radius")
        
        if calculation == "nuclear_radius":
            mass_number = task.get("mass_number", 12)
            radius = self.constants["nuclear_radius_const"] * mass_number**(1/3)
            
            return {
                "calculation": calculation,
                "mass_number": mass_number,
                "nuclear_radius": radius,
                "units": "fm",
                "convergence": True,
                "physical_validity": True
            }
        
        elif calculation == "nuclear_density":
            # Nuclear density is approximately constant
            nuclear_density = 2.3e17  # kg/m³
            
            return {
                "calculation": calculation,
                "nuclear_density": nuclear_density,
                "units": "kg/m³",
                "note": "Nuclear density is approximately constant for all nuclei",
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
            "nuclear", "radioactive", "decay", "fission", "fusion", "binding energy",
            "nucleus", "isotope", "half-life", "neutron", "proton", "alpha decay",
            "beta decay", "gamma ray", "nuclear reaction", "nucleosynthesis"
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            "atomic", "radiation", "mass defect", "nuclear physics", "reactor",
            "uranium", "plutonium", "carbon dating", "stellar", "nuclear force"
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
        Estimate computational cost for a nuclear physics task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "binding_energy")
        
        # Base costs by task type
        base_costs = {
            "binding_energy": {"time": 0.1, "memory": 128, "units": 1},
            "radioactive_decay": {"time": 0.5, "memory": 256, "units": 2},
            "nuclear_reaction": {"time": 1, "memory": 512, "units": 5},
            "nuclear_structure": {"time": 2, "memory": 512, "units": 8},
            "fission": {"time": 3, "memory": 1024, "units": 10},
            "fusion": {"time": 5, "memory": 1024, "units": 15},
            "nuclear_astrophysics": {"time": 3, "memory": 512, "units": 12}
        }
        
        base_cost = base_costs.get(task_type, base_costs["binding_energy"])
        
        # Scale based on problem complexity
        scale_factor = 1.0
        
        if task_type == "nuclear_astrophysics":
            process = task.get("process", "pp_chain")
            if process == "nucleosynthesis":
                scale_factor = 2.0
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for nuclear physics calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Quantum mechanics basics",
                "Special relativity",
                "Statistical mechanics",
                "Exponential decay mathematics"
            ],
            "computational_methods": [
                "Numerical integration",
                "Monte Carlo methods", 
                "Statistical analysis",
                "Differential equation solving"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Nuclear reactor design",
                "Radiocarbon dating",
                "Medical isotope production",
                "Nuclear astrophysics",
                "Nuclear waste analysis"
            ],
            "accuracy_considerations": [
                "Nuclear data uncertainties",
                "Approximations in nuclear models",
                "Statistical fluctuations in decay",
                "Temperature and pressure effects"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for nuclear physics calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if task_type == "radioactive_decay":
            time = task.get("time")
            if time and time < 0:
                errors.append("Time must be positive")
            
            parent = task.get("parent")
            if parent and parent not in ["C14", "U238", "U235", "H3", "Ra226"]:
                warnings.append(f"Limited decay data available for {parent}")
        
        elif task_type == "nuclear_reaction":
            reaction = task.get("reaction")
            if not reaction or "->" not in reaction:
                errors.append("Nuclear reaction must be specified in 'A + B -> C + D' format")
        
        elif task_type == "fusion":
            temperature = task.get("temperature")
            if temperature and temperature < 1e6:
                warnings.append("Temperature may be too low for significant fusion")
            if temperature and temperature > 1e10:
                warnings.append("Extremely high temperature - relativistic effects important")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }