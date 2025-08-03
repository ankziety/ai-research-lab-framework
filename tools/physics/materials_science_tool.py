"""
Materials Science Tool

Agent-friendly interface for materials property calculations.
Provides crystal structure analysis, mechanical properties,
electronic properties, and materials design capabilities.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class MaterialsScienceTool(BasePhysicsTool):
    """
    Tool for materials science calculations that agents can request.
    
    Provides interfaces for:
    - Crystal structure analysis
    - Mechanical property calculations
    - Electronic band structure
    - Phase diagram analysis
    - Materials property prediction
    """
    
    def __init__(self):
        super().__init__(
            tool_id="materials_science_tool",
            name="Materials Science Tool",
            description="Perform materials science calculations including crystal structure analysis, mechanical properties, and electronic structure",
            physics_domain="materials_science",
            computational_cost_factor=2.5,
            software_requirements=[
                "pymatgen",     # Materials analysis (optional)
                "ase",          # Atomic simulation (optional)
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib"    # Visualization
            ],
            hardware_requirements={
                "min_memory": 1024,  # MB
                "recommended_memory": 4096,
                "cpu_cores": 2,
                "supports_gpu": True
            }
        )
        
        # Add materials science specific capabilities
        self.capabilities.extend([
            "crystal_structure_analysis",
            "mechanical_property_calculation",
            "electronic_band_structure",
            "phase_diagram_analysis",
            "materials_property_prediction",
            "defect_analysis",
            "surface_analysis",
            "materials_design"
        ])
        
        # Available calculation types
        self.calculation_types = [
            "structure_analysis",
            "mechanical_properties",
            "electronic_properties",
            "thermal_properties",
            "phase_stability",
            "defect_formation",
            "surface_properties"
        ]
        
        # Crystal systems
        self.crystal_systems = [
            "cubic", "tetragonal", "orthorhombic", 
            "hexagonal", "trigonal", "monoclinic", "triclinic"
        ]
        
        # Property databases
        self.materials_database = {
            "common_materials": {
                "Si": {"structure": "diamond", "band_gap": 1.12, "bulk_modulus": 100},
                "Fe": {"structure": "bcc", "bulk_modulus": 170, "magnetic": True},
                "Al": {"structure": "fcc", "bulk_modulus": 76, "conductivity": "metallic"},
                "GaAs": {"structure": "zinc_blende", "band_gap": 1.42, "semiconductor": True}
            }
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute materials science calculation requested by an agent.
        
        Args:
            task: Task specification with materials parameters
            context: Agent context and execution environment
            
        Returns:
            Materials science results formatted for agents
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
            task_type = task.get("type", "structure_analysis")
            material = task.get("material", {})
            properties = task.get("properties", ["all"])
            
            # Route to appropriate calculation
            if task_type == "structure_analysis":
                result = self._analyze_structure(material, context)
            elif task_type == "mechanical_properties":
                result = self._calculate_mechanical_properties(material, context)
            elif task_type == "electronic_properties":
                result = self._calculate_electronic_properties(material, context)
            elif task_type == "thermal_properties":
                result = self._calculate_thermal_properties(material, context)
            elif task_type == "phase_stability":
                result = self._analyze_phase_stability(material, context)
            elif task_type == "defect_formation":
                result = self._analyze_defects(material, task.get("defect_type"), context)
            elif task_type == "surface_properties":
                result = self._analyze_surface(material, task.get("surface_index"), context)
            else:
                raise ValueError(f"Unknown materials science task type: {task_type}")
            
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
        Validate materials science input parameters.
        
        Args:
            input_data: Input parameters from agent
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        if "material" not in input_data:
            errors.append("Missing 'material' specification")
            suggestions.append("Provide material composition or crystal structure")
        else:
            material = input_data["material"]
            
            # Validate material format
            if "composition" not in material and "structure" not in material and "formula" not in material:
                errors.append("Material must specify 'composition', 'structure', or 'formula'")
                suggestions.append("Use composition: {'Si': 1} or formula: 'SiO2' or structure data")
            
            # Check for reasonable system size
            if "unit_cell" in material:
                atoms_in_cell = material["unit_cell"].get("num_atoms", 0)
                if atoms_in_cell > 200:
                    warnings.append(f"Large unit cell ({atoms_in_cell} atoms) - calculation may be expensive")
                elif atoms_in_cell > 1000:
                    errors.append("Unit cell too large for this tool (>1000 atoms)")
                    suggestions.append("Consider using supercell approximation or smaller model")
        
        # Validate calculation type
        calc_type = input_data.get("type", "structure_analysis")
        if calc_type not in self.calculation_types:
            errors.append(f"Unknown calculation type '{calc_type}'")
            suggestions.append(f"Available types: {', '.join(self.calculation_types)}")
        
        # Check for expensive calculations
        if calc_type in ["electronic_properties", "defect_formation"] and "large_system" in input_data:
            warnings.append("Electronic structure calculations can be expensive for large systems")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format materials science results for agents.
        
        Args:
            output_data: Raw calculation results
            
        Returns:
            Agent-friendly formatted results
        """
        formatted = {
            "summary": self._generate_result_summary(output_data),
            "structure": {},
            "mechanical": {},
            "electronic": {},
            "thermal": {},
            "analysis": {}
        }
        
        # Structure information
        if "crystal_system" in output_data:
            formatted["structure"]["crystal_system"] = output_data["crystal_system"]
        if "space_group" in output_data:
            formatted["structure"]["space_group"] = output_data["space_group"]
        if "lattice_parameters" in output_data:
            formatted["structure"]["lattice_parameters"] = output_data["lattice_parameters"]
        if "density" in output_data:
            formatted["structure"]["density"] = {
                "value": output_data["density"],
                "units": "g/cm³",
                "description": "Bulk density"
            }
        
        # Mechanical properties
        if "bulk_modulus" in output_data:
            formatted["mechanical"]["bulk_modulus"] = {
                "value": output_data["bulk_modulus"],
                "units": "GPa",
                "description": "Bulk modulus (resistance to compression)"
            }
        if "shear_modulus" in output_data:
            formatted["mechanical"]["shear_modulus"] = {
                "value": output_data["shear_modulus"],
                "units": "GPa",
                "description": "Shear modulus (resistance to shear deformation)"
            }
        if "youngs_modulus" in output_data:
            formatted["mechanical"]["youngs_modulus"] = {
                "value": output_data["youngs_modulus"],
                "units": "GPa",
                "description": "Young's modulus (stiffness)"
            }
        
        # Electronic properties
        if "band_gap" in output_data:
            formatted["electronic"]["band_gap"] = {
                "value": output_data["band_gap"],
                "units": "eV",
                "type": output_data.get("band_gap_type", "unknown"),
                "description": "Electronic band gap"
            }
        if "conductivity_type" in output_data:
            formatted["electronic"]["conductivity_type"] = output_data["conductivity_type"]
        
        # Thermal properties
        if "melting_point" in output_data:
            formatted["thermal"]["melting_point"] = {
                "value": output_data["melting_point"],
                "units": "K",
                "description": "Melting temperature"
            }
        if "thermal_conductivity" in output_data:
            formatted["thermal"]["thermal_conductivity"] = {
                "value": output_data["thermal_conductivity"],
                "units": "W/m·K",
                "description": "Thermal conductivity"
            }
        
        # Analysis and insights
        formatted["analysis"]["material_classification"] = self._classify_material(output_data)
        formatted["analysis"]["applications"] = self._suggest_applications(output_data)
        formatted["analysis"]["insights"] = self._generate_insights(output_data)
        
        return formatted
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for materials calculation.
        
        Args:
            task: Task specification
            
        Returns:
            Cost estimates (time, memory, computational units)
        """
        base_cost = 1.0
        
        # Get system size
        material = task.get("material", {})
        num_atoms = material.get("unit_cell", {}).get("num_atoms", 10)
        
        # Scale with system size
        size_cost_factor = (num_atoms / 10) ** 2.5
        
        # Calculation type cost factors
        calc_costs = {
            "structure_analysis": 0.5,
            "mechanical_properties": 2.0,
            "electronic_properties": 5.0,
            "thermal_properties": 3.0,
            "phase_stability": 4.0,
            "defect_formation": 8.0,
            "surface_properties": 6.0
        }
        
        calc_type = task.get("type", "structure_analysis")
        calc_cost = calc_costs.get(calc_type, 1.0)
        
        total_cost_factor = size_cost_factor * calc_cost * self.computational_cost_factor
        
        # Estimate time (in seconds)
        estimated_time = base_cost * total_cost_factor * 5
        
        # Estimate memory (in MB)
        estimated_memory = 200 + (num_atoms ** 1.5) * calc_cost * 5
        
        # Computational units
        computational_units = total_cost_factor * 50
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_memory_mb": estimated_memory,
            "computational_units": computational_units,
            "cost_breakdown": {
                "size_factor": size_cost_factor,
                "calculation_factor": calc_cost,
                "total_factor": total_cost_factor
            }
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get materials science specific requirements."""
        return {
            "physics_domain": "materials_science",
            "available_calculation_types": self.calculation_types,
            "supported_crystal_systems": self.crystal_systems,
            "supported_material_formats": ["composition", "structure", "formula"],
            "max_atoms_per_cell": 1000,
            "typical_calculation_time": "30 seconds to several hours",
            "memory_scaling": "O(N^2.5) with number of atoms",
            "software_dependencies": self.software_requirements,
            "hardware_recommendations": self.hardware_requirements
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Get materials science domain keywords."""
        return [
            "materials", "crystal", "structure", "mechanical", "electronic",
            "thermal", "phase", "defect", "surface", "bulk", "modulus",
            "conductivity", "semiconductor", "metal", "insulator", "lattice",
            "band", "gap", "property", "composition", "alloy"
        ]
    
    def _analyze_structure(self, material: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze crystal structure (mock implementation)."""
        # Mock structure analysis
        composition = material.get("composition", {})
        formula = material.get("formula", "Unknown")
        
        # Determine likely crystal system based on composition
        if "Si" in composition:
            crystal_system = "cubic"
            space_group = "Fd-3m"  # Diamond structure
            lattice_param = 5.43
        elif "Fe" in composition:
            crystal_system = "cubic"
            space_group = "Im-3m"  # BCC
            lattice_param = 2.87
        else:
            crystal_system = "cubic"
            space_group = "Fm-3m"  # FCC default
            lattice_param = 4.0
        
        # Calculate density (mock)
        atomic_masses = {"Si": 28.09, "Fe": 55.85, "Al": 26.98, "O": 16.0}
        total_mass = sum(atomic_masses.get(elem, 50) * count for elem, count in composition.items())
        volume = lattice_param ** 3
        density = total_mass * 1.66054e-24 / (volume * 1e-24)  # g/cm³
        
        return {
            "crystal_system": crystal_system,
            "space_group": space_group,
            "lattice_parameters": {"a": lattice_param, "b": lattice_param, "c": lattice_param},
            "density": density,
            "formula": formula,
            "composition": composition
        }
    
    def _calculate_mechanical_properties(self, material: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mechanical properties (mock implementation)."""
        result = self._analyze_structure(material, context)
        
        # Mock mechanical properties based on known materials
        composition = material.get("composition", {})
        
        if "Si" in composition:
            bulk_modulus = 100  # GPa
            shear_modulus = 80
        elif "Fe" in composition:
            bulk_modulus = 170
            shear_modulus = 82
        elif "Al" in composition:
            bulk_modulus = 76
            shear_modulus = 26
        else:
            bulk_modulus = 120  # Default
            shear_modulus = 60
        
        # Calculate derived properties
        youngs_modulus = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
        poissons_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (6 * bulk_modulus + 2 * shear_modulus)
        
        result.update({
            "bulk_modulus": bulk_modulus,
            "shear_modulus": shear_modulus,
            "youngs_modulus": youngs_modulus,
            "poissons_ratio": poissons_ratio,
            "hardness_estimate": shear_modulus / 10  # Rough estimate
        })
        
        return result
    
    def _calculate_electronic_properties(self, material: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate electronic properties (mock implementation)."""
        result = self._analyze_structure(material, context)
        
        # Mock electronic properties
        composition = material.get("composition", {})
        
        if "Si" in composition:
            band_gap = 1.12
            band_gap_type = "indirect"
            conductivity_type = "semiconductor"
        elif "Fe" in composition:
            band_gap = 0.0
            band_gap_type = "metallic"
            conductivity_type = "metallic"
        elif "GaAs" in str(composition):
            band_gap = 1.42
            band_gap_type = "direct"
            conductivity_type = "semiconductor"
        else:
            band_gap = 2.0  # Default insulator
            band_gap_type = "direct"
            conductivity_type = "insulator"
        
        result.update({
            "band_gap": band_gap,
            "band_gap_type": band_gap_type,
            "conductivity_type": conductivity_type,
            "effective_mass_electron": 0.26,  # Mock value
            "effective_mass_hole": 0.39       # Mock value
        })
        
        return result
    
    def _calculate_thermal_properties(self, material: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate thermal properties (mock implementation)."""
        result = self._analyze_structure(material, context)
        
        # Mock thermal properties
        composition = material.get("composition", {})
        
        if "Si" in composition:
            melting_point = 1687  # K
            thermal_conductivity = 150  # W/m·K
        elif "Fe" in composition:
            melting_point = 1811
            thermal_conductivity = 80
        elif "Al" in composition:
            melting_point = 933
            thermal_conductivity = 237
        else:
            melting_point = 1500  # Default
            thermal_conductivity = 100
        
        result.update({
            "melting_point": melting_point,
            "thermal_conductivity": thermal_conductivity,
            "thermal_expansion": 5e-6,  # /K
            "specific_heat": 700        # J/kg·K
        })
        
        return result
    
    def _analyze_phase_stability(self, material: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze phase stability (mock implementation)."""
        result = self._analyze_structure(material, context)
        
        # Mock phase stability analysis
        result.update({
            "formation_energy": -0.5,  # eV/atom
            "stability": "stable",
            "competing_phases": ["SiO2", "Si2O"],
            "phase_transition_temperature": 1200  # K
        })
        
        return result
    
    def _analyze_defects(self, material: Dict[str, Any], defect_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze defect formation (mock implementation)."""
        result = self._analyze_structure(material, context)
        
        # Mock defect analysis
        defect_formation_energies = {
            "vacancy": 2.5,      # eV
            "interstitial": 3.2,
            "substitutional": 1.8,
            "antisite": 2.1
        }
        
        result.update({
            "defect_type": defect_type,
            "formation_energy": defect_formation_energies.get(defect_type, 2.0),
            "migration_barrier": 1.2,  # eV
            "defect_concentration": 1e15  # cm⁻³
        })
        
        return result
    
    def _analyze_surface(self, material: Dict[str, Any], surface_index: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze surface properties (mock implementation)."""
        result = self._analyze_structure(material, context)
        
        # Mock surface analysis
        result.update({
            "surface_index": surface_index or "(100)",
            "surface_energy": 1.2,  # J/m²
            "work_function": 4.5,   # eV
            "surface_reconstruction": "2x1"
        })
        
        return result
    
    def _calculate_actual_cost(self, task: Dict[str, Any], actual_time: float) -> float:
        """Calculate actual computational cost."""
        estimates = self.estimate_cost(task)
        estimated_time = estimates["estimated_time_seconds"]
        
        time_ratio = actual_time / max(estimated_time, 0.1)
        actual_cost = estimates["computational_units"] * time_ratio
        
        return actual_cost
    
    def _assess_result_confidence(self, result: Dict[str, Any], task_type: str) -> float:
        """Assess confidence in calculation results."""
        base_confidence = 0.8
        
        # Task type reliability
        task_confidence = {
            "structure_analysis": 0.9,
            "mechanical_properties": 0.8,
            "electronic_properties": 0.75,
            "thermal_properties": 0.7,
            "phase_stability": 0.65,
            "defect_formation": 0.6,
            "surface_properties": 0.65
        }
        
        return task_confidence.get(task_type, 0.7)
    
    def _generate_result_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable summary of results."""
        formula = result.get("formula", "Unknown")
        crystal_system = result.get("crystal_system", "unknown")
        
        summary = f"Materials analysis completed for {formula}. "
        summary += f"Crystal system: {crystal_system}. "
        
        if "bulk_modulus" in result:
            summary += f"Bulk modulus: {result['bulk_modulus']:.1f} GPa. "
        
        if "band_gap" in result:
            summary += f"Band gap: {result['band_gap']:.2f} eV ({result.get('conductivity_type', 'unknown')}). "
        
        return summary
    
    def _classify_material(self, result: Dict[str, Any]) -> str:
        """Classify material type based on properties."""
        if "conductivity_type" in result:
            conductivity = result["conductivity_type"]
            if conductivity == "metallic":
                return "metal"
            elif conductivity == "semiconductor":
                return "semiconductor"
            else:
                return "insulator"
        
        return "unknown"
    
    def _suggest_applications(self, result: Dict[str, Any]) -> List[str]:
        """Suggest potential applications based on properties."""
        applications = []
        
        conductivity = result.get("conductivity_type", "")
        band_gap = result.get("band_gap", 0)
        bulk_modulus = result.get("bulk_modulus", 0)
        
        if conductivity == "metallic":
            applications.extend(["electrical conductors", "structural materials"])
        elif conductivity == "semiconductor":
            if 1.0 < band_gap < 2.0:
                applications.extend(["solar cells", "photodetectors"])
            elif band_gap > 3.0:
                applications.extend(["UV detectors", "power electronics"])
        
        if bulk_modulus > 200:
            applications.append("high-strength applications")
        
        return applications
    
    def _generate_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate scientific insights from results."""
        insights = []
        
        band_gap = result.get("band_gap", 0)
        if 1.1 < band_gap < 1.8:
            insights.append("Band gap suitable for photovoltaic applications")
        
        bulk_modulus = result.get("bulk_modulus", 0)
        if bulk_modulus > 150:
            insights.append("High bulk modulus indicates resistance to deformation")
        
        density = result.get("density", 0)
        if density < 3.0:
            insights.append("Low density material suitable for lightweight applications")
        
        return insights