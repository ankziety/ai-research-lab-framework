"""
Quantum Chemistry Tool

Agent-friendly interface for quantum chemistry calculations.
Provides molecular property calculations, electronic structure analysis,
and quantum mechanical simulations.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class QuantumChemistryTool(BasePhysicsTool):
    """
    Tool for quantum chemistry calculations that agents can request.
    
    Provides interfaces for:
    - Molecular property calculations
    - Electronic structure analysis  
    - Quantum mechanical simulations
    - Orbital analysis
    - Energy calculations
    """
    
    def __init__(self):
        super().__init__(
            tool_id="quantum_chemistry_tool",
            name="Quantum Chemistry Tool",
            description="Perform quantum chemistry calculations including molecular properties, electronic structure, and energy analysis",
            physics_domain="quantum_chemistry",
            computational_cost_factor=3.0,  # QC is computationally expensive
            software_requirements=[
                "psi4",      # Quantum chemistry package (optional)
                "rdkit",     # Molecular handling (optional)
                "numpy",     # Core calculations
                "scipy"      # Mathematical functions
            ],
            hardware_requirements={
                "min_memory": 2048,  # MB
                "recommended_memory": 8192,
                "cpu_cores": 4,
                "supports_gpu": True
            }
        )
        
        # Add quantum chemistry specific capabilities
        self.capabilities.extend([
            "molecular_property_calculation",
            "electronic_structure_analysis",
            "energy_calculation",
            "orbital_analysis",
            "geometry_optimization",
            "vibrational_analysis",
            "reaction_pathway_analysis"
        ])
        
        # Quantum chemistry calculation methods
        self.available_methods = [
            "hartree_fock",
            "dft",
            "mp2",
            "ccsd",
            "ccsd_t"
        ]
        
        # Basis sets
        self.available_basis_sets = [
            "sto-3g",
            "3-21g",
            "6-31g",
            "6-31g*",
            "6-311g**",
            "cc-pvdz",
            "cc-pvtz"
        ]
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute quantum chemistry calculation requested by an agent.
        
        Args:
            task: Task specification with quantum chemistry parameters
            context: Agent context and execution environment
            
        Returns:
            Quantum chemistry results formatted for agents
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
            task_type = task.get("type", "energy_calculation")
            molecule = task.get("molecule", {})
            method = task.get("method", "hartree_fock")
            basis_set = task.get("basis_set", "6-31g")
            
            # Route to appropriate calculation
            if task_type == "energy_calculation":
                result = self._calculate_energy(molecule, method, basis_set, context)
            elif task_type == "geometry_optimization":
                result = self._optimize_geometry(molecule, method, basis_set, context)
            elif task_type == "molecular_properties":
                result = self._calculate_molecular_properties(molecule, method, basis_set, context)
            elif task_type == "orbital_analysis":
                result = self._analyze_orbitals(molecule, method, basis_set, context)
            elif task_type == "vibrational_analysis":
                result = self._analyze_vibrations(molecule, method, basis_set, context)
            else:
                raise ValueError(f"Unknown quantum chemistry task type: {task_type}")
            
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
                "method_used": method,
                "basis_set_used": basis_set,
                "confidence": self._assess_result_confidence(result, method, basis_set),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.update_calculation_stats(calculation_time, 0.0, False)
            return self.handle_errors(e, {"task": task, "context": context})
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quantum chemistry input parameters.
        
        Args:
            input_data: Input parameters from agent
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        if "molecule" not in input_data:
            errors.append("Missing 'molecule' specification")
            suggestions.append("Provide molecule geometry or structure")
        else:
            molecule = input_data["molecule"]
            
            # Validate molecule format
            if "atoms" not in molecule and "smiles" not in molecule and "xyz" not in molecule:
                errors.append("Molecule must specify 'atoms', 'smiles', or 'xyz' format")
                suggestions.append("Use atoms list: [{'element': 'H', 'position': [0,0,0]}] or SMILES string")
            
            # Check for reasonable molecule size
            if "atoms" in molecule:
                num_atoms = len(molecule["atoms"])
                if num_atoms > 100:
                    warnings.append(f"Large molecule ({num_atoms} atoms) - calculation may be expensive")
                elif num_atoms > 500:
                    errors.append("Molecule too large for this tool (>500 atoms)")
                    suggestions.append("Consider fragmenting molecule or using simplified model")
        
        # Validate method
        method = input_data.get("method", "hartree_fock")
        if method not in self.available_methods:
            errors.append(f"Unknown method '{method}'")
            suggestions.append(f"Available methods: {', '.join(self.available_methods)}")
        
        # Validate basis set
        basis_set = input_data.get("basis_set", "6-31g")
        if basis_set not in self.available_basis_sets:
            errors.append(f"Unknown basis set '{basis_set}'")
            suggestions.append(f"Available basis sets: {', '.join(self.available_basis_sets)}")
        
        # Check for expensive combinations
        if method in ["ccsd", "ccsd_t"] and basis_set in ["cc-pvtz", "6-311g**"]:
            warnings.append("Expensive method/basis combination - consider smaller basis set for initial calculations")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format quantum chemistry results for agents.
        
        Args:
            output_data: Raw calculation results
            
        Returns:
            Agent-friendly formatted results
        """
        # Format numerical results for agent consumption
        formatted = {
            "summary": self._generate_result_summary(output_data),
            "energies": {},
            "properties": {},
            "geometry": {},
            "analysis": {}
        }
        
        # Energy information
        if "energy" in output_data:
            formatted["energies"]["total_energy"] = {
                "value": output_data["energy"],
                "units": "hartree",
                "value_eV": output_data["energy"] * 27.211,  # Convert to eV
                "description": "Total electronic energy"
            }
        
        # Molecular properties
        if "dipole_moment" in output_data:
            formatted["properties"]["dipole_moment"] = {
                "magnitude": output_data["dipole_moment"],
                "units": "debye",
                "description": "Electric dipole moment magnitude"
            }
        
        if "homo_lumo_gap" in output_data:
            formatted["properties"]["homo_lumo_gap"] = {
                "value": output_data["homo_lumo_gap"],
                "units": "eV",
                "description": "HOMO-LUMO energy gap"
            }
        
        # Geometry information
        if "optimized_geometry" in output_data:
            formatted["geometry"]["optimized"] = output_data["optimized_geometry"]
            formatted["geometry"]["optimization_converged"] = output_data.get("optimization_converged", False)
        
        # Analysis and insights
        formatted["analysis"]["calculation_insights"] = self._generate_insights(output_data)
        formatted["analysis"]["recommendations"] = self._generate_recommendations(output_data)
        
        return formatted
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for quantum chemistry calculation.
        
        Args:
            task: Task specification
            
        Returns:
            Cost estimates (time, memory, computational units)
        """
        base_cost = 1.0
        
        # Get molecule size
        molecule = task.get("molecule", {})
        num_atoms = len(molecule.get("atoms", []))
        
        # Scale with number of atoms (roughly N^4 for many methods)
        atom_cost_factor = (num_atoms / 10) ** 3.5
        
        # Method cost factors
        method_costs = {
            "hartree_fock": 1.0,
            "dft": 1.5,
            "mp2": 4.0,
            "ccsd": 15.0,
            "ccsd_t": 50.0
        }
        
        method = task.get("method", "hartree_fock")
        method_cost = method_costs.get(method, 1.0)
        
        # Basis set cost factors
        basis_costs = {
            "sto-3g": 0.5,
            "3-21g": 1.0,
            "6-31g": 1.5,
            "6-31g*": 2.0,
            "6-311g**": 3.0,
            "cc-pvdz": 2.5,
            "cc-pvtz": 5.0
        }
        
        basis_set = task.get("basis_set", "6-31g")
        basis_cost = basis_costs.get(basis_set, 1.0)
        
        total_cost_factor = atom_cost_factor * method_cost * basis_cost * self.computational_cost_factor
        
        # Estimate time (in seconds)
        estimated_time = base_cost * total_cost_factor * 10
        
        # Estimate memory (in MB)
        estimated_memory = 500 + (num_atoms ** 2) * method_cost * 10
        
        # Computational units (arbitrary units for cost comparison)
        computational_units = total_cost_factor * 100
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_memory_mb": estimated_memory,
            "computational_units": computational_units,
            "cost_breakdown": {
                "atom_factor": atom_cost_factor,
                "method_factor": method_cost,
                "basis_factor": basis_cost,
                "total_factor": total_cost_factor
            }
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get quantum chemistry specific requirements."""
        return {
            "physics_domain": "quantum_chemistry",
            "available_methods": self.available_methods,
            "available_basis_sets": self.available_basis_sets,
            "supported_molecule_formats": ["atoms", "smiles", "xyz"],
            "max_atoms": 500,
            "typical_calculation_time": "1 minute to several hours",
            "memory_scaling": "O(N^2) to O(N^4) with number of atoms",
            "software_dependencies": self.software_requirements,
            "hardware_recommendations": self.hardware_requirements
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Get quantum chemistry domain keywords."""
        return [
            "quantum", "chemistry", "molecular", "electronic", "orbital",
            "energy", "optimization", "dft", "hartree", "fock", "ccsd",
            "molecule", "atom", "bond", "dipole", "homo", "lumo",
            "basis", "set", "calculation", "structure"
        ]
    
    def _calculate_energy(self, molecule: Dict[str, Any], method: str, basis_set: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate molecular energy (mock implementation)."""
        # Mock calculation - in real implementation would use quantum chemistry software
        num_atoms = len(molecule.get("atoms", []))
        
        # Simple mock energy calculation
        base_energy = -num_atoms * 1.5  # Rough approximation
        method_correction = {"hartree_fock": 0, "dft": -0.1, "mp2": -0.05, "ccsd": -0.02}
        energy = base_energy + method_correction.get(method, 0)
        
        return {
            "energy": energy,
            "method": method,
            "basis_set": basis_set,
            "num_atoms": num_atoms,
            "convergence_achieved": True
        }
    
    def _optimize_geometry(self, molecule: Dict[str, Any], method: str, basis_set: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize molecular geometry (mock implementation)."""
        # Mock geometry optimization
        result = self._calculate_energy(molecule, method, basis_set, context)
        
        # Add optimization-specific results
        result.update({
            "optimized_geometry": molecule.get("atoms", []),  # Mock: return input geometry
            "optimization_converged": True,
            "optimization_steps": 15,
            "final_gradient_norm": 1e-6
        })
        
        return result
    
    def _calculate_molecular_properties(self, molecule: Dict[str, Any], method: str, basis_set: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate molecular properties (mock implementation)."""
        result = self._calculate_energy(molecule, method, basis_set, context)
        
        # Add molecular properties
        result.update({
            "dipole_moment": 2.5,  # Mock value in debye
            "homo_lumo_gap": 8.2,  # Mock value in eV
            "molecular_volume": 125.4,  # Mock value in Å³
            "polarizability": 15.3  # Mock value in Å³
        })
        
        return result
    
    def _analyze_orbitals(self, molecule: Dict[str, Any], method: str, basis_set: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze molecular orbitals (mock implementation)."""
        result = self._calculate_energy(molecule, method, basis_set, context)
        
        # Add orbital analysis
        result.update({
            "homo_energy": -8.5,  # eV
            "lumo_energy": -0.3,  # eV
            "homo_lumo_gap": 8.2,  # eV
            "num_occupied_orbitals": 12,
            "orbital_symmetries": ["A1", "A1", "B2", "A1", "B1"]
        })
        
        return result
    
    def _analyze_vibrations(self, molecule: Dict[str, Any], method: str, basis_set: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze molecular vibrations (mock implementation)."""
        result = self._calculate_energy(molecule, method, basis_set, context)
        
        # Add vibrational analysis
        num_atoms = len(molecule.get("atoms", []))
        num_vibs = 3 * num_atoms - 6  # 3N-6 vibrational modes
        
        # Mock vibrational frequencies
        frequencies = np.random.uniform(200, 3500, num_vibs).tolist()
        
        result.update({
            "vibrational_frequencies": frequencies,
            "zero_point_energy": sum(frequencies) * 0.5 * 4.184 / 1000,  # kJ/mol
            "thermal_corrections": {
                "enthalpy_298K": 15.2,
                "entropy_298K": 185.4,
                "free_energy_298K": -40.1
            }
        })
        
        return result
    
    def _calculate_actual_cost(self, task: Dict[str, Any], actual_time: float) -> float:
        """Calculate actual computational cost based on task and timing."""
        estimates = self.estimate_cost(task)
        estimated_time = estimates["estimated_time_seconds"]
        
        # Cost is proportional to actual computational units used
        time_ratio = actual_time / max(estimated_time, 0.1)  # Avoid division by zero
        actual_cost = estimates["computational_units"] * time_ratio
        
        return actual_cost
    
    def _assess_result_confidence(self, result: Dict[str, Any], method: str, basis_set: str) -> float:
        """Assess confidence in calculation results."""
        base_confidence = 0.8
        
        # Method reliability
        method_confidence = {
            "hartree_fock": 0.7,
            "dft": 0.85,
            "mp2": 0.9,
            "ccsd": 0.95,
            "ccsd_t": 0.98
        }
        
        # Basis set quality
        basis_confidence = {
            "sto-3g": 0.6,
            "3-21g": 0.7,
            "6-31g": 0.8,
            "6-31g*": 0.85,
            "6-311g**": 0.9,
            "cc-pvdz": 0.85,
            "cc-pvtz": 0.95
        }
        
        # Check convergence
        convergence_bonus = 0.1 if result.get("convergence_achieved", False) else -0.2
        
        confidence = (base_confidence * 
                     method_confidence.get(method, 0.7) * 
                     basis_confidence.get(basis_set, 0.7) + 
                     convergence_bonus)
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_result_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable summary of results."""
        energy = result.get("energy", 0)
        method = result.get("method", "unknown")
        basis_set = result.get("basis_set", "unknown")
        
        summary = f"Quantum chemistry calculation completed using {method}/{basis_set}. "
        summary += f"Total energy: {energy:.6f} hartree ({energy * 27.211:.3f} eV). "
        
        if "dipole_moment" in result:
            summary += f"Dipole moment: {result['dipole_moment']:.2f} debye. "
        
        if "homo_lumo_gap" in result:
            summary += f"HOMO-LUMO gap: {result['homo_lumo_gap']:.2f} eV. "
        
        return summary
    
    def _generate_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate scientific insights from results."""
        insights = []
        
        if "homo_lumo_gap" in result:
            gap = result["homo_lumo_gap"]
            if gap < 2.0:
                insights.append("Small HOMO-LUMO gap suggests high reactivity")
            elif gap > 5.0:
                insights.append("Large HOMO-LUMO gap indicates chemical stability")
        
        if "dipole_moment" in result:
            dipole = result["dipole_moment"]
            if dipole > 3.0:
                insights.append("High dipole moment indicates strong polarity")
            elif dipole < 0.5:
                insights.append("Low dipole moment suggests non-polar character")
        
        return insights
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for follow-up calculations."""
        recommendations = []
        
        method = result.get("method", "")
        basis_set = result.get("basis_set", "")
        
        if method == "hartree_fock":
            recommendations.append("Consider DFT calculation for better electron correlation")
        
        if basis_set in ["sto-3g", "3-21g"]:
            recommendations.append("Use larger basis set for more accurate results")
        
        if not result.get("optimization_converged", True):
            recommendations.append("Perform geometry optimization before property calculations")
        
        return recommendations