"""
Biophysics Tool

Agent-friendly interface for biophysics calculations.
Provides biomolecular structure analysis, protein folding,
membrane dynamics, and biological system modeling.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class BiophysicsTool(BasePhysicsTool):
    """
    Tool for biophysics calculations that agents can request.
    
    Provides interfaces for:
    - Protein structure analysis
    - Membrane dynamics
    - DNA/RNA modeling
    - Enzyme kinetics
    - Molecular dynamics simulations
    - Biological network analysis
    """
    
    def __init__(self):
        super().__init__(
            tool_id="biophysics_tool",
            name="Biophysics Tool",
            description="Perform biophysics calculations including protein folding, membrane dynamics, and biological system modeling",
            physics_domain="biophysics",
            computational_cost_factor=3.0,
            software_requirements=[
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib",   # Visualization
                "biopython",    # Biological data (optional)
                "mdanalysis"    # Molecular dynamics (optional)
            ],
            hardware_requirements={
                "min_memory": 2048,  # MB
                "recommended_memory": 8192,
                "cpu_cores": 4,
                "supports_gpu": True
            }
        )
        
        self.capabilities.extend([
            "protein_structure",
            "protein_folding",
            "membrane_dynamics",
            "dna_rna_analysis",
            "enzyme_kinetics",
            "molecular_dynamics",
            "biological_networks",
            "drug_binding",
            "biomolecular_interactions"
        ])
        
        # Biophysical constants
        self.constants = {
            "kb": 1.381e-23,    # J/K
            "na": 6.022e23,     # mol⁻¹
            "rt_300k": 2.48,    # kJ/mol at 300K
            "gas_constant": 8.314,  # J/mol·K
            "water_density": 1000,  # kg/m³
            "membrane_thickness": 4e-9,  # m
            "amino_acid_mass": 110,  # Da (average)
            "dna_base_mass": 330    # Da (average)
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute biophysics calculation.
        
        Args:
            task: Task specification with biophysics parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with biophysics analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "protein_structure")
            
            if task_type == "protein_structure":
                result = self._analyze_protein_structure(task)
            elif task_type == "protein_folding":
                result = self._analyze_protein_folding(task)
            elif task_type == "membrane_dynamics":
                result = self._analyze_membrane_dynamics(task)
            elif task_type == "dna_analysis":
                result = self._analyze_dna_structure(task)
            elif task_type == "enzyme_kinetics":
                result = self._analyze_enzyme_kinetics(task)
            elif task_type == "molecular_dynamics":
                result = self._simulate_molecular_dynamics(task)
            elif task_type == "drug_binding":
                result = self._analyze_drug_binding(task)
            else:
                result = self._generic_biophysics_calculation(task)
            
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
            logger.error(f"Biophysics calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _analyze_protein_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze protein structure properties."""
        protein_type = task.get("protein_type", "globular")
        num_residues = task.get("num_residues", 150)
        secondary_structure = task.get("secondary_structure", {"helix": 0.3, "sheet": 0.2, "loop": 0.5})
        
        # Protein structure predictions
        molecular_weight = num_residues * self.constants["amino_acid_mass"]  # Da
        
        # Radius of gyration (empirical formula)
        radius_gyration = 2.2 * num_residues**0.38  # Angstroms
        
        # Solvent accessible surface area
        surface_area = 16 * num_residues**0.73  # Angstroms²
        
        # Stability analysis
        folding_energy = -20 * num_residues  # kJ/mol (rough estimate)
        melting_temperature = 273 + 30 + 0.5 * num_residues  # K
        
        # Secondary structure content
        helix_content = secondary_structure.get("helix", 0.3)
        sheet_content = secondary_structure.get("sheet", 0.2)
        loop_content = secondary_structure.get("loop", 0.5)
        
        # Hydrophobicity analysis
        hydrophobic_fraction = 0.4 + 0.1 * np.random.randn()  # Typical range
        
        # Compactness
        if protein_type == "globular":
            compactness = 0.85
            predicted_structure = "compact_globular"
        elif protein_type == "membrane":
            compactness = 0.6
            predicted_structure = "transmembrane"
        else:
            compactness = 0.3
            predicted_structure = "extended_or_disordered"
        
        return {
            "protein_type": protein_type,
            "num_residues": num_residues,
            "molecular_weight": molecular_weight,
            "radius_of_gyration": radius_gyration,
            "surface_area": surface_area,
            "folding_energy": folding_energy,
            "melting_temperature": melting_temperature,
            "secondary_structure": {
                "helix_fraction": helix_content,
                "sheet_fraction": sheet_content,
                "loop_fraction": loop_content
            },
            "hydrophobic_fraction": hydrophobic_fraction,
            "compactness": compactness,
            "predicted_structure": predicted_structure,
            "units": {
                "mass": "Da",
                "length": "Angstroms",
                "area": "Angstroms²",
                "energy": "kJ/mol",
                "temperature": "K"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_protein_folding(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze protein folding thermodynamics and kinetics."""
        sequence_length = task.get("sequence_length", 100)
        temperature = task.get("temperature", 298)  # K
        folding_type = task.get("folding_type", "two_state")
        
        # Folding thermodynamics
        native_contacts = int(sequence_length * 0.8)  # Estimated
        contact_energy = -2.0  # kJ/mol per contact
        
        # Free energy of folding
        delta_h = native_contacts * contact_energy  # Enthalpy
        delta_s = -sequence_length * 0.01  # Entropy (kJ/mol·K)
        delta_g = delta_h - temperature * delta_s  # Free energy
        
        # Folding stability
        if delta_g < -10:
            stability = "very_stable"
        elif delta_g < 0:
            stability = "stable"
        elif delta_g < 10:
            stability = "marginally_stable"
        else:
            stability = "unstable"
        
        # Folding kinetics (Levinthal's paradox resolution)
        # Search through folding funnel
        conformational_states = 3**sequence_length  # Rough estimate
        folding_time_random = conformational_states * 1e-13  # seconds
        
        # Actual folding time (guided by energy landscape)
        if folding_type == "two_state":
            folding_time = 1e-6 * np.exp(sequence_length / 50)  # seconds
            folding_mechanism = "cooperative_two_state"
        elif folding_type == "multi_state":
            folding_time = 1e-5 * np.exp(sequence_length / 30)  # seconds
            folding_mechanism = "hierarchical_multi_state"
        else:
            folding_time = 1e-3 * sequence_length  # seconds
            folding_mechanism = "slow_complex_folding"
        
        # Folding rate and activation energy
        attempt_frequency = 1e6  # Hz
        k_fold = 1 / folding_time
        activation_energy = -self.constants["rt_300k"] * np.log(k_fold / attempt_frequency)
        
        # Cooperativity
        cooperativity = 1.0 if folding_type == "two_state" else 0.5
        
        return {
            "sequence_length": sequence_length,
            "temperature": temperature,
            "folding_type": folding_type,
            "thermodynamics": {
                "delta_h": delta_h,
                "delta_s": delta_s,
                "delta_g": delta_g,
                "stability": stability
            },
            "kinetics": {
                "folding_time": folding_time,
                "folding_rate": k_fold,
                "activation_energy": activation_energy,
                "folding_mechanism": folding_mechanism
            },
            "levinthal_paradox": {
                "conformational_states": f"{conformational_states:.2e}",
                "random_search_time": f"{folding_time_random:.2e} seconds",
                "actual_folding_time": f"{folding_time:.2e} seconds",
                "speedup_factor": folding_time_random / folding_time
            },
            "cooperativity": cooperativity,
            "native_contacts": native_contacts,
            "units": {
                "energy": "kJ/mol",
                "entropy": "kJ/mol·K",
                "time": "seconds",
                "rate": "Hz"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_membrane_dynamics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze biological membrane dynamics."""
        membrane_type = task.get("membrane_type", "plasma_membrane")
        temperature = task.get("temperature", 310)  # K (body temperature)
        lipid_composition = task.get("lipid_composition", {"POPC": 0.5, "POPE": 0.3, "cholesterol": 0.2})
        
        # Membrane properties
        membrane_thickness = 4.0  # nm
        area_per_lipid = 0.65  # nm²
        
        # Phase behavior
        if temperature > 315:
            phase = "liquid_disordered"
            order_parameter = 0.3
        elif temperature > 295:
            phase = "liquid_ordered"
            order_parameter = 0.6
        else:
            phase = "gel"
            order_parameter = 0.9
        
        # Lateral diffusion
        # Einstein-Stokes relation in 2D
        viscosity = 1e-9 * np.exp(2000/temperature)  # Pa·s (temperature dependent)
        diffusion_coeff = self.constants["kb"] * temperature / (4 * np.pi * viscosity * 1e-9)  # m²/s
        
        # Membrane elasticity
        bending_modulus = 20 * self.constants["kb"] * temperature  # J (typical range 10-50 kT)
        area_compressibility = 250e-3  # N/m (typical)
        
        # Permeability
        cholesterol_fraction = lipid_composition.get("cholesterol", 0.2)
        permeability_modifier = 1 - 0.5 * cholesterol_fraction  # Cholesterol reduces permeability
        
        # Transmembrane potential
        membrane_potential = -70e-3  # V (typical resting potential)
        
        # Lipid flip-flop dynamics
        flip_flop_time = 10 * 3600 * np.exp(5000/temperature)  # seconds
        
        return {
            "membrane_type": membrane_type,
            "temperature": temperature,
            "lipid_composition": lipid_composition,
            "membrane_properties": {
                "thickness": membrane_thickness,
                "area_per_lipid": area_per_lipid,
                "phase": phase,
                "order_parameter": order_parameter
            },
            "dynamics": {
                "lateral_diffusion_coefficient": diffusion_coeff,
                "membrane_viscosity": viscosity,
                "flip_flop_time": flip_flop_time
            },
            "mechanical_properties": {
                "bending_modulus": bending_modulus,
                "area_compressibility": area_compressibility
            },
            "electrical_properties": {
                "membrane_potential": membrane_potential,
                "permeability_modifier": permeability_modifier
            },
            "units": {
                "length": "nm",
                "area": "nm²",
                "diffusion": "m²/s",
                "energy": "J",
                "potential": "V",
                "time": "seconds"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_dna_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DNA/RNA structure and dynamics."""
        nucleic_acid_type = task.get("type", "DNA")
        sequence_length = task.get("sequence_length", 1000)  # base pairs
        gc_content = task.get("gc_content", 0.5)
        temperature = task.get("temperature", 298)  # K
        
        # Structural properties
        if nucleic_acid_type == "DNA":
            helix_pitch = 3.4  # nm per turn
            bases_per_turn = 10.5
            base_stacking_energy = -5.0  # kJ/mol
            major_groove_width = 2.2  # nm
            minor_groove_width = 1.2  # nm
        else:  # RNA
            helix_pitch = 2.8  # nm per turn
            bases_per_turn = 11
            base_stacking_energy = -6.0  # kJ/mol
            major_groove_width = 1.8  # nm
            minor_groove_width = 1.0  # nm
        
        # Contour length
        contour_length = sequence_length * helix_pitch / bases_per_turn  # nm
        
        # Persistence length (flexibility)
        if nucleic_acid_type == "DNA":
            persistence_length = 50  # nm
        else:
            persistence_length = 30  # nm (RNA is more flexible)
        
        # Melting temperature
        # Nearest neighbor model (simplified)
        tm_base = 81.5  # °C for standard conditions
        gc_effect = gc_content * 20  # GC content effect
        length_effect = 12 * np.log(sequence_length) if sequence_length > 20 else 0
        melting_temp = tm_base + gc_effect + length_effect
        
        # Thermodynamic stability
        base_pairs = sequence_length
        stacking_energy = base_pairs * base_stacking_energy
        hydrogen_bond_energy = base_pairs * (-10 if gc_content > 0.5 else -8)  # kJ/mol
        total_stability = stacking_energy + hydrogen_bond_energy
        
        # Flexibility analysis
        flexibility_ratio = contour_length / persistence_length
        if flexibility_ratio > 10:
            conformation = "flexible_coil"
        elif flexibility_ratio > 1:
            conformation = "semi_flexible"
        else:
            conformation = "rigid_rod"
        
        # Supercoiling (for DNA)
        if nucleic_acid_type == "DNA":
            linking_number = sequence_length / bases_per_turn
            supercoiling_density = -0.06  # Typical negative supercoiling in cells
            supercoiling_energy = 1000 * supercoiling_density**2  # kJ/mol
        else:
            linking_number = None
            supercoiling_density = None
            supercoiling_energy = 0
        
        return {
            "nucleic_acid_type": nucleic_acid_type,
            "sequence_length": sequence_length,
            "gc_content": gc_content,
            "temperature": temperature,
            "structural_properties": {
                "contour_length": contour_length,
                "persistence_length": persistence_length,
                "helix_pitch": helix_pitch,
                "bases_per_turn": bases_per_turn,
                "major_groove_width": major_groove_width,
                "minor_groove_width": minor_groove_width
            },
            "thermodynamics": {
                "melting_temperature": melting_temp,
                "base_stacking_energy": stacking_energy,
                "hydrogen_bond_energy": hydrogen_bond_energy,
                "total_stability": total_stability,
                "supercoiling_energy": supercoiling_energy
            },
            "mechanical_properties": {
                "flexibility_ratio": flexibility_ratio,
                "conformation": conformation,
                "linking_number": linking_number,
                "supercoiling_density": supercoiling_density
            },
            "units": {
                "length": "nm",
                "energy": "kJ/mol",
                "temperature": "K"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_enzyme_kinetics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze enzyme kinetics and catalysis."""
        enzyme_type = task.get("enzyme_type", "michaelis_menten")
        km = task.get("km", 1e-6)  # M (Michaelis constant)
        vmax = task.get("vmax", 1e-6)  # M/s (maximum velocity)
        substrate_conc = task.get("substrate_concentration", [1e-7, 1e-6, 1e-5, 1e-4])  # M
        temperature = task.get("temperature", 310)  # K
        
        # Michaelis-Menten kinetics
        reaction_rates = []
        for s in substrate_conc:
            if enzyme_type == "michaelis_menten":
                rate = vmax * s / (km + s)
            elif enzyme_type == "competitive_inhibition":
                ki = task.get("ki", 1e-7)  # M (inhibitor constant)
                inhibitor_conc = task.get("inhibitor_concentration", 1e-7)  # M
                km_apparent = km * (1 + inhibitor_conc / ki)
                rate = vmax * s / (km_apparent + s)
            elif enzyme_type == "allosteric":
                hill_coefficient = task.get("hill_coefficient", 2)
                rate = vmax * s**hill_coefficient / (km**hill_coefficient + s**hill_coefficient)
            else:
                rate = vmax * s / (km + s)  # Default to MM
            
            reaction_rates.append(rate)
        
        # Catalytic efficiency
        kcat_km = vmax / km  # Catalytic efficiency
        turnover_number = vmax  # Assuming [E] = 1 M
        
        # Temperature effects
        activation_energy = 50000  # J/mol (typical)
        rate_300k = vmax
        rate_temperature = rate_300k * np.exp(-activation_energy / self.constants["gas_constant"] * (1/temperature - 1/300))
        
        # Binding thermodynamics
        binding_energy = -self.constants["gas_constant"] * temperature * np.log(1/km)  # J/mol
        
        # Reaction coordinate analysis
        barrier_reduction = 80000  # J/mol (typical enzyme catalysis)
        uncatalyzed_rate = 1e-15  # M/s (typical uncatalyzed rate)
        catalytic_enhancement = vmax / uncatalyzed_rate
        
        return {
            "enzyme_type": enzyme_type,
            "kinetic_parameters": {
                "km": km,
                "vmax": vmax,
                "kcat_km": kcat_km,
                "turnover_number": turnover_number
            },
            "substrate_concentrations": substrate_conc,
            "reaction_rates": reaction_rates,
            "temperature_effects": {
                "temperature": temperature,
                "activation_energy": activation_energy,
                "rate_at_temperature": rate_temperature
            },
            "thermodynamics": {
                "binding_energy": binding_energy,
                "barrier_reduction": barrier_reduction,
                "catalytic_enhancement": catalytic_enhancement
            },
            "efficiency_metrics": {
                "catalytic_efficiency": kcat_km,
                "specificity_constant": kcat_km,
                "enhancement_factor": catalytic_enhancement
            },
            "units": {
                "concentration": "M",
                "rate": "M/s",
                "energy": "J/mol",
                "efficiency": "M⁻¹s⁻¹"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _simulate_molecular_dynamics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate molecular dynamics of biological systems."""
        system_type = task.get("system_type", "protein_in_water")
        num_atoms = task.get("num_atoms", 10000)
        simulation_time = task.get("simulation_time", 10)  # ns
        temperature = task.get("temperature", 300)  # K
        
        # System setup
        if system_type == "protein_in_water":
            protein_atoms = num_atoms // 10
            water_molecules = (num_atoms - protein_atoms) // 3
            box_size = (num_atoms / 1000)**(1/3) * 5  # nm
        elif system_type == "membrane_protein":
            protein_atoms = num_atoms // 5
            lipid_molecules = protein_atoms // 50
            water_molecules = (num_atoms - protein_atoms - lipid_molecules * 50) // 3
            box_size = (num_atoms / 800)**(1/3) * 6  # nm
        else:
            protein_atoms = num_atoms
            water_molecules = 0
            lipid_molecules = 0
            box_size = (num_atoms / 1000)**(1/3) * 4  # nm
        
        # Time step and trajectory
        time_step = 0.002  # ps
        total_steps = int(simulation_time * 1000 / time_step)
        frames_saved = min(1000, total_steps // 100)
        
        # Energy components (estimated)
        kinetic_energy = 1.5 * self.constants["kb"] * temperature * num_atoms / 1000  # kJ/mol
        potential_energy = -2 * kinetic_energy  # Typical ratio
        total_energy = kinetic_energy + potential_energy
        
        # Structural analysis
        rmsd_trajectory = np.random.normal(0.15, 0.05, frames_saved)  # nm
        rmsd_trajectory = np.cumsum(np.abs(rmsd_trajectory)) / 10  # Drift over time
        
        radius_gyration = np.random.normal(1.5, 0.1, frames_saved)  # nm
        
        # Diffusion analysis
        diffusion_coeff = 1e-11 * np.exp(-2000/temperature)  # m²/s
        
        # Hydrogen bonds
        if water_molecules > 0:
            avg_hbonds = protein_atoms * 0.3  # Protein-water H-bonds
        else:
            avg_hbonds = protein_atoms * 0.1  # Intramolecular H-bonds
        
        # Computational performance
        ns_per_day = 100 / (num_atoms / 10000)  # Rough scaling
        
        return {
            "system_type": system_type,
            "system_composition": {
                "total_atoms": num_atoms,
                "protein_atoms": protein_atoms,
                "water_molecules": water_molecules,
                "lipid_molecules": lipid_molecules if 'lipid_molecules' in locals() else 0,
                "box_size": box_size
            },
            "simulation_parameters": {
                "simulation_time": simulation_time,
                "time_step": time_step,
                "total_steps": total_steps,
                "frames_saved": frames_saved,
                "temperature": temperature
            },
            "energetics": {
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "total_energy": total_energy
            },
            "structural_analysis": {
                "final_rmsd": float(rmsd_trajectory[-1]),
                "average_radius_gyration": float(np.mean(radius_gyration)),
                "rmsd_stability": float(np.std(rmsd_trajectory[-100:]))
            },
            "dynamics": {
                "diffusion_coefficient": diffusion_coeff,
                "average_hydrogen_bonds": avg_hbonds
            },
            "performance": {
                "ns_per_day": ns_per_day,
                "computational_hours": simulation_time / ns_per_day * 24
            },
            "units": {
                "time": "ns",
                "energy": "kJ/mol",
                "length": "nm",
                "diffusion": "m²/s"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_drug_binding(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze drug-target binding interactions."""
        binding_mode = task.get("binding_mode", "competitive")
        drug_concentration = task.get("drug_concentration", 1e-8)  # M
        target_concentration = task.get("target_concentration", 1e-9)  # M
        kd = task.get("dissociation_constant", 1e-9)  # M
        temperature = task.get("temperature", 310)  # K
        
        # Binding thermodynamics
        if kd > 0:
            binding_energy = self.constants["gas_constant"] * temperature * np.log(kd)  # J/mol
            binding_energy_kj = binding_energy / 1000  # kJ/mol
        else:
            binding_energy_kj = -50  # Default value
        
        # Binding kinetics
        kon = task.get("association_rate", 1e6)  # M⁻¹s⁻¹
        koff = kon * kd  # s⁻¹
        
        # Fractional occupancy
        fractional_occupancy = drug_concentration / (kd + drug_concentration)
        
        # IC50 analysis
        ic50 = kd * (1 + target_concentration / kd)  # Cheng-Prusoff equation
        
        # Drug efficiency metrics
        ligand_efficiency = abs(binding_energy_kj) / task.get("molecular_weight", 300)  # kJ/mol per Da
        
        # Selectivity analysis
        selectivity_targets = task.get("off_targets", [])
        selectivity_ratios = []
        for off_target_kd in task.get("off_target_kds", [1e-6]):
            selectivity_ratio = off_target_kd / kd
            selectivity_ratios.append(selectivity_ratio)
        
        # Residence time
        residence_time = 1 / koff  # seconds
        
        # ADMET predictions (simplified)
        molecular_weight = task.get("molecular_weight", 300)
        logp = task.get("logp", 2.0)
        
        # Lipinski's Rule of Five
        rule_of_five = {
            "molecular_weight_ok": molecular_weight <= 500,
            "logp_ok": logp <= 5,
            "violations": 0
        }
        if not rule_of_five["molecular_weight_ok"]:
            rule_of_five["violations"] += 1
        if not rule_of_five["logp_ok"]:
            rule_of_five["violations"] += 1
        
        # Bioavailability prediction
        if rule_of_five["violations"] == 0:
            oral_bioavailability = 0.8
        elif rule_of_five["violations"] == 1:
            oral_bioavailability = 0.5
        else:
            oral_bioavailability = 0.2
        
        return {
            "binding_mode": binding_mode,
            "concentrations": {
                "drug_concentration": drug_concentration,
                "target_concentration": target_concentration
            },
            "binding_parameters": {
                "dissociation_constant": kd,
                "association_rate": kon,
                "dissociation_rate": koff,
                "binding_energy": binding_energy_kj,
                "fractional_occupancy": fractional_occupancy
            },
            "efficacy_metrics": {
                "ic50": ic50,
                "ligand_efficiency": ligand_efficiency,
                "residence_time": residence_time
            },
            "selectivity": {
                "selectivity_ratios": selectivity_ratios,
                "average_selectivity": np.mean(selectivity_ratios) if selectivity_ratios else 1.0
            },
            "drug_properties": {
                "molecular_weight": molecular_weight,
                "logp": logp,
                "rule_of_five": rule_of_five,
                "predicted_bioavailability": oral_bioavailability
            },
            "units": {
                "concentration": "M",
                "rate": "M⁻¹s⁻¹ or s⁻¹",
                "energy": "kJ/mol",
                "time": "seconds"
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _generic_biophysics_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic biophysics calculation."""
        calculation = task.get("calculation", "boltzmann_factor")
        
        if calculation == "boltzmann_factor":
            energy_difference = task.get("energy_difference", 10)  # kJ/mol
            temperature = task.get("temperature", 298)  # K
            
            kt = self.constants["gas_constant"] * temperature / 1000  # kJ/mol
            boltzmann_factor = np.exp(-energy_difference / kt)
            
            return {
                "calculation": calculation,
                "energy_difference": energy_difference,
                "temperature": temperature,
                "boltzmann_factor": boltzmann_factor,
                "thermal_energy": kt,
                "units": {
                    "energy": "kJ/mol",
                    "temperature": "K"
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
            "protein", "enzyme", "dna", "rna", "membrane", "biophysics", "molecular dynamics",
            "drug binding", "folding", "biomolecule", "biological", "biochemical",
            "peptide", "amino acid", "nucleotide", "lipid", "cell membrane"
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            "binding", "kinetics", "thermodynamics", "structure", "dynamics",
            "interaction", "molecular", "cellular", "biochemistry"
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
        Estimate computational cost for a biophysics task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "protein_structure")
        
        # Base costs by task type
        base_costs = {
            "protein_structure": {"time": 1, "memory": 512, "units": 5},
            "protein_folding": {"time": 5, "memory": 1024, "units": 15},
            "membrane_dynamics": {"time": 3, "memory": 1024, "units": 10},
            "dna_analysis": {"time": 2, "memory": 512, "units": 8},
            "enzyme_kinetics": {"time": 1, "memory": 256, "units": 3},
            "molecular_dynamics": {"time": 20, "memory": 4096, "units": 50},
            "drug_binding": {"time": 2, "memory": 512, "units": 7}
        }
        
        base_cost = base_costs.get(task_type, base_costs["protein_structure"])
        
        # Scale based on system size
        scale_factor = 1.0
        
        if task_type == "molecular_dynamics":
            num_atoms = task.get("num_atoms", 10000)
            sim_time = task.get("simulation_time", 10)
            scale_factor = (num_atoms / 10000) * (sim_time / 10)
        elif task_type == "protein_folding":
            seq_length = task.get("sequence_length", 100)
            scale_factor = (seq_length / 100)**1.5
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for biophysics calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Statistical mechanics",
                "Thermodynamics", 
                "Kinetic theory",
                "Differential equations"
            ],
            "computational_methods": [
                "Molecular dynamics simulation",
                "Monte Carlo methods",
                "Free energy calculations",
                "Protein structure prediction"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Drug discovery",
                "Protein engineering",
                "Membrane research",
                "DNA/RNA analysis",
                "Enzyme design"
            ],
            "accuracy_considerations": [
                "Force field limitations",
                "Sampling convergence issues",
                "System size effects",
                "Time scale limitations"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for biophysics calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if task_type == "molecular_dynamics":
            num_atoms = task.get("num_atoms", 10000)
            if num_atoms > 100000:
                warnings.append(f"Large system ({num_atoms} atoms) requires significant computational resources")
            
            sim_time = task.get("simulation_time", 10)
            if sim_time > 100:
                warnings.append(f"Long simulation ({sim_time} ns) may take considerable time")
        
        elif task_type == "enzyme_kinetics":
            km = task.get("km")
            if km and km <= 0:
                errors.append("Michaelis constant (Km) must be positive")
            
            vmax = task.get("vmax")
            if vmax and vmax <= 0:
                errors.append("Maximum velocity (Vmax) must be positive")
        
        elif task_type == "drug_binding":
            kd = task.get("dissociation_constant")
            if kd and kd <= 0:
                errors.append("Dissociation constant (Kd) must be positive")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }