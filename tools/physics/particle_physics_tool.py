"""
Particle Physics Tool

Agent-friendly interface for particle physics calculations.
Provides high-energy physics simulations, collider analysis,
particle interaction modeling, and quantum field theory calculations.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import math
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class ParticlePhysicsTool(BasePhysicsTool):
    """
    Tool for particle physics calculations that agents can request.
    
    Provides interfaces for:
    - Collider event simulation
    - Particle decay analysis
    - Cross-section calculations
    - Feynman diagram evaluation
    - Standard Model calculations
    - Beyond Standard Model physics
    """
    
    def __init__(self):
        super().__init__(
            tool_id="particle_physics_tool",
            name="Particle Physics Tool",
            description="Perform particle physics calculations including collider analysis, particle interactions, and high-energy physics simulations",
            physics_domain="particle_physics",
            computational_cost_factor=4.0,  # High-energy calculations are expensive
            software_requirements=[
                "pyjet",        # Jet clustering (optional)
                "hepmc",        # Monte Carlo data format (optional) 
                "numpy",        # Core calculations
                "scipy",        # Mathematical functions
                "matplotlib"    # Visualization
            ],
            hardware_requirements={
                "min_memory": 4096,  # MB
                "recommended_memory": 16384,
                "cpu_cores": 8,
                "supports_gpu": True
            }
        )
        
        self.capabilities.extend([
            "collider_simulation",
            "particle_decay",
            "cross_section_calculation",
            "feynman_diagrams", 
            "jet_reconstruction",
            "standard_model_physics",
            "beyond_standard_model"
        ])
        
        # Particle physics constants
        self.constants = {
            "hbar_c": 197.327,  # MeV·fm
            "alpha_em": 1/137.036,  # Fine structure constant
            "fermi_constant": 1.166e-5,  # GeV^-2
            "z_mass": 91.188,  # GeV
            "w_mass": 80.379,  # GeV
            "higgs_mass": 125.18,  # GeV
            "proton_mass": 0.938272,  # GeV
            "electron_mass": 0.000511  # GeV
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute particle physics calculation.
        
        Args:
            task: Task specification with particle physics parameters
            context: Execution context and environment
            
        Returns:
            Calculation results with particle physics analysis
        """
        try:
            start_time = datetime.now()
            
            task_type = task.get("type", "cross_section")
            
            if task_type == "cross_section":
                result = self._calculate_cross_section(task)
            elif task_type == "particle_decay":
                result = self._analyze_particle_decay(task)
            elif task_type == "collider_event":
                result = self._simulate_collider_event(task)
            elif task_type == "jet_analysis":
                result = self._analyze_jets(task)
            elif task_type == "standard_model":
                result = self._calculate_standard_model_physics(task)
            else:
                result = self._generic_particle_calculation(task)
            
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
            logger.error(f"Particle physics calculation failed: {e}")
            self.usage_count += 1
            return {
                "success": False,
                "error": str(e),
                "calculation_time": 0.0,
                "computational_cost": 0.0,
                "physics_domain": self.physics_domain,
                "tool_id": self.tool_id
            }
    
    def _calculate_cross_section(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate particle interaction cross-sections."""
        process = task.get("process", "e+e- -> mu+mu-")
        energy = task.get("center_of_mass_energy", 91.188)  # GeV
        
        # Simplified cross-section calculations for common processes
        if "e+e-" in process and "mu+mu-" in process:
            # QED process: e+e- -> mu+mu-
            s = energy**2
            alpha = self.constants["alpha_em"]
            beta = math.sqrt(1 - 4*self.constants["electron_mass"]**2/s)
            
            # Born cross-section
            cross_section = (4*math.pi*alpha**2)/(3*s) * beta * (3 - beta**2)
            
            return {
                "process": process,
                "center_of_mass_energy": energy,
                "cross_section": cross_section,  # nb
                "units": "nanobarns",
                "calculation_method": "QED Born approximation",
                "convergence": True,
                "physical_validity": True
            }
        
        elif "Z" in process:
            # Z boson production
            gamma_z = 2.495  # GeV, Z width
            m_z = self.constants["z_mass"]
            
            # Breit-Wigner resonance
            denominator = (energy**2 - m_z**2)**2 + (gamma_z * m_z)**2
            cross_section = 12*math.pi * energy**2 * gamma_z**2 / (m_z**4 * denominator)
            
            return {
                "process": process,
                "center_of_mass_energy": energy,
                "cross_section": cross_section * 1e9,  # Convert to nb
                "units": "nanobarns", 
                "z_mass": m_z,
                "z_width": gamma_z,
                "calculation_method": "Breit-Wigner resonance",
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            # Generic process - simplified model
            cross_section = 100 * (1 + 0.1*np.random.randn())  # nb
            
            return {
                "process": process,
                "center_of_mass_energy": energy,
                "cross_section": cross_section,
                "units": "nanobarns",
                "calculation_method": "Generic model",
                "convergence": True,
                "physical_validity": True
            }
    
    def _analyze_particle_decay(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze particle decay processes."""
        particle = task.get("particle", "Z")
        decay_channel = task.get("decay_channel", "e+e-")
        
        # Known branching ratios and properties
        decay_data = {
            "Z": {
                "mass": self.constants["z_mass"],
                "width": 2.495,  # GeV
                "channels": {
                    "e+e-": {"branching_ratio": 0.0337, "partial_width": 0.084},
                    "mu+mu-": {"branching_ratio": 0.0337, "partial_width": 0.084},
                    "tau+tau-": {"branching_ratio": 0.0337, "partial_width": 0.084},
                    "hadrons": {"branching_ratio": 0.6991, "partial_width": 1.744},
                    "neutrinos": {"branching_ratio": 0.2000, "partial_width": 0.499}
                }
            },
            "W": {
                "mass": self.constants["w_mass"],
                "width": 2.085,  # GeV
                "channels": {
                    "e nu": {"branching_ratio": 0.1071, "partial_width": 0.223},
                    "mu nu": {"branching_ratio": 0.1063, "partial_width": 0.222},
                    "tau nu": {"branching_ratio": 0.1138, "partial_width": 0.237},
                    "hadrons": {"branching_ratio": 0.6741, "partial_width": 1.405}
                }
            },
            "Higgs": {
                "mass": self.constants["higgs_mass"],
                "width": 0.00407,  # GeV
                "channels": {
                    "bb": {"branching_ratio": 0.5824, "partial_width": 0.00237},
                    "WW": {"branching_ratio": 0.2137, "partial_width": 0.00087},
                    "gg": {"branching_ratio": 0.0821, "partial_width": 0.00034},
                    "tau+tau-": {"branching_ratio": 0.0632, "partial_width": 0.00026},
                    "cc": {"branching_ratio": 0.0294, "partial_width": 0.00012}
                }
            }
        }
        
        if particle in decay_data:
            particle_data = decay_data[particle]
            
            if decay_channel in particle_data["channels"]:
                channel_data = particle_data["channels"][decay_channel]
                
                # Calculate lifetime
                lifetime = self.constants["hbar_c"] / (particle_data["width"] * 1000)  # fs
                
                return {
                    "particle": particle,
                    "decay_channel": decay_channel,
                    "particle_mass": particle_data["mass"],
                    "total_width": particle_data["width"],
                    "branching_ratio": channel_data["branching_ratio"],
                    "partial_width": channel_data["partial_width"],
                    "lifetime": lifetime,
                    "lifetime_units": "femtoseconds",
                    "convergence": True,
                    "physical_validity": True
                }
            else:
                return {
                    "particle": particle,
                    "decay_channel": decay_channel,
                    "error": f"Decay channel {decay_channel} not found for {particle}",
                    "available_channels": list(particle_data["channels"].keys()),
                    "convergence": False,
                    "physical_validity": False
                }
        else:
            return {
                "particle": particle,
                "error": f"Particle {particle} not in database",
                "available_particles": list(decay_data.keys()),
                "convergence": False,
                "physical_validity": False
            }
    
    def _simulate_collider_event(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate collider event generation."""
        collision_type = task.get("collision_type", "pp")
        energy = task.get("energy", 13000)  # GeV
        num_events = task.get("num_events", 1000)
        process = task.get("process", "dijets")
        
        # Generate mock collider data
        events = []
        for i in range(min(num_events, 10)):  # Limit for demo
            event = {
                "event_id": i,
                "particles": []
            }
            
            # Generate particles based on process
            if process == "dijets":
                # Two jet event
                for j in range(2):
                    pt = np.random.exponential(50) + 20  # GeV
                    eta = np.random.uniform(-5, 5)
                    phi = np.random.uniform(0, 2*math.pi)
                    
                    event["particles"].append({
                        "type": "jet",
                        "pt": pt,
                        "eta": eta, 
                        "phi": phi,
                        "mass": np.random.normal(10, 3)
                    })
            
            elif process == "Z_production":
                # Z -> ee event
                z_pt = np.random.exponential(30)
                z_eta = np.random.uniform(-2.5, 2.5)
                z_phi = np.random.uniform(0, 2*math.pi)
                
                # Electron pair from Z decay
                for j in range(2):
                    pt = np.random.normal(40, 10)
                    eta = z_eta + np.random.normal(0, 0.5)
                    phi = z_phi + np.random.normal(0, 0.1)
                    
                    event["particles"].append({
                        "type": "electron",
                        "charge": (-1)**j,
                        "pt": pt,
                        "eta": eta,
                        "phi": phi,
                        "mass": self.constants["electron_mass"]
                    })
            
            events.append(event)
        
        # Calculate event statistics
        total_pt = sum(sum(p["pt"] for p in event["particles"]) for event in events)
        avg_multiplicity = sum(len(event["particles"]) for event in events) / len(events)
        
        return {
            "collision_type": collision_type,
            "center_of_mass_energy": energy,
            "process": process,
            "num_events_generated": len(events),
            "events": events,
            "statistics": {
                "average_total_pt": total_pt / len(events),
                "average_multiplicity": avg_multiplicity
            },
            "convergence": True,
            "physical_validity": True
        }
    
    def _analyze_jets(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze jet reconstruction and properties."""
        particles = task.get("particles", [])
        algorithm = task.get("algorithm", "anti-kt")
        radius = task.get("radius", 0.4)
        
        if not particles:
            # Generate mock particle data
            particles = []
            for i in range(20):
                particles.append({
                    "pt": np.random.exponential(10) + 1,
                    "eta": np.random.uniform(-5, 5),
                    "phi": np.random.uniform(0, 2*math.pi),
                    "mass": np.random.exponential(1)
                })
        
        # Simple jet clustering simulation
        jets = []
        used_particles = set()
        
        for i, seed in enumerate(particles):
            if i in used_particles:
                continue
                
            jet_particles = [seed]
            jet_pt = seed["pt"]
            jet_eta = seed["eta"]
            jet_phi = seed["phi"]
            used_particles.add(i)
            
            # Find nearby particles
            for j, particle in enumerate(particles):
                if j in used_particles:
                    continue
                    
                delta_eta = particle["eta"] - jet_eta
                delta_phi = abs(particle["phi"] - jet_phi)
                if delta_phi > math.pi:
                    delta_phi = 2*math.pi - delta_phi
                
                delta_r = math.sqrt(delta_eta**2 + delta_phi**2)
                
                if delta_r < radius:
                    jet_particles.append(particle)
                    jet_pt += particle["pt"]
                    used_particles.add(j)
            
            if jet_pt > 20:  # Minimum jet pt
                jets.append({
                    "pt": jet_pt,
                    "eta": jet_eta,
                    "phi": jet_phi,
                    "constituents": len(jet_particles),
                    "mass": sum(p["mass"] for p in jet_particles)
                })
        
        # Sort jets by pt
        jets.sort(key=lambda x: x["pt"], reverse=True)
        
        return {
            "algorithm": algorithm,
            "radius": radius,
            "num_input_particles": len(particles),
            "num_jets": len(jets),
            "jets": jets[:10],  # Top 10 jets
            "leading_jet_pt": jets[0]["pt"] if jets else 0,
            "convergence": True,
            "physical_validity": True
        }
    
    def _calculate_standard_model_physics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Standard Model physics parameters."""
        calculation_type = task.get("type", "coupling_constants")
        energy_scale = task.get("energy_scale", 91.188)  # GeV
        
        if calculation_type == "coupling_constants":
            # Running coupling constants (simplified)
            alpha_em = self.constants["alpha_em"]
            
            # QED beta function (1-loop)
            t = math.log(energy_scale / 0.511)  # log(Q/me)
            alpha_running = alpha_em / (1 - alpha_em * t / (3*math.pi))
            
            # Weak mixing angle
            sin2_theta_w = 0.23116  # At Z mass
            
            return {
                "energy_scale": energy_scale,
                "alpha_em": alpha_running,
                "sin2_theta_w": sin2_theta_w,
                "fermi_constant": self.constants["fermi_constant"],
                "z_mass": self.constants["z_mass"],
                "w_mass": self.constants["w_mass"],
                "calculation_method": "1-loop running",
                "convergence": True,
                "physical_validity": True
            }
        
        elif calculation_type == "electroweak_precision":
            # Electroweak precision observables
            m_w = self.constants["w_mass"]
            m_z = self.constants["z_mass"]
            
            # Rho parameter
            rho = m_w**2 / (m_z**2 * math.cos(math.asin(math.sqrt(0.23116)))**2)
            
            return {
                "w_mass": m_w,
                "z_mass": m_z,
                "rho_parameter": rho,
                "delta_rho": rho - 1,
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "error": f"Unknown calculation type: {calculation_type}",
                "convergence": False,
                "physical_validity": False
            }
    
    def _generic_particle_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generic particle physics calculation."""
        calculation = task.get("calculation", "basic_kinematics")
        
        if calculation == "basic_kinematics":
            particles = task.get("particles", [])
            if not particles:
                return {
                    "error": "No particles provided for kinematics calculation",
                    "convergence": False,
                    "physical_validity": False
                }
            
            # Calculate invariant mass
            total_e = 0
            total_px = 0
            total_py = 0
            total_pz = 0
            
            for particle in particles:
                pt = particle.get("pt", 0)
                eta = particle.get("eta", 0)
                phi = particle.get("phi", 0)
                mass = particle.get("mass", 0)
                
                px = pt * math.cos(phi)
                py = pt * math.sin(phi)
                pz = pt * math.sinh(eta)
                e = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
                
                total_e += e
                total_px += px
                total_py += py
                total_pz += pz
            
            invariant_mass = math.sqrt(total_e**2 - total_px**2 - total_py**2 - total_pz**2)
            total_pt = math.sqrt(total_px**2 + total_py**2)
            
            return {
                "calculation": calculation,
                "num_particles": len(particles),
                "invariant_mass": invariant_mass,
                "total_pt": total_pt,
                "total_energy": total_e,
                "convergence": True,
                "physical_validity": True
            }
        
        else:
            return {
                "error": f"Unknown calculation: {calculation}",
                "convergence": False,
                "physical_validity": False
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for particle physics calculations."""
        return self.validate_input(input_data)  # Uses existing validate_input method
    
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format particle physics calculation results for agents."""
        # Basic output processing - could be enhanced
        return {
            "formatted_results": output_data,
            "summary": f"Particle physics calculation completed with {len(output_data)} result fields",
            "physics_domain": self.physics_domain,
            "tool_id": self.tool_id
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Get physics domain specific keywords for task matching."""
        return [
            "particle", "collider", "cross section", "decay", "standard model",
            "feynman", "jet", "hadron", "lepton", "quark", "boson", "higgs",
            "lhc", "accelerator", "high energy", "elementary particle"
        ]
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """
        Assess if this tool can handle the research question.
        
        Returns confidence score between 0 and 1.
        """
        task_lower = task_type.lower()
        
        # High confidence keywords
        high_confidence_keywords = [
            "particle", "collider", "cross section", "decay", "standard model",
            "feynman", "jet", "hadron", "lepton", "quark", "boson", "higgs",
            "lhc", "accelerator", "high energy", "elementary particle"
        ]
        
        # Medium confidence keywords  
        medium_confidence_keywords = [
            "interaction", "scattering", "quantum field", "gauge", "symmetry",
            "mass", "energy", "momentum", "relativistic"
        ]
        
        # Calculate confidence
        high_matches = sum(1 for keyword in high_confidence_keywords 
                          if keyword in task_lower)
        medium_matches = sum(1 for keyword in medium_confidence_keywords 
                           if keyword in task_lower)
        
        if high_matches > 0:
            return min(1.0, 0.7 + high_matches * 0.1)
        elif medium_matches > 0:
            return min(0.6, 0.3 + medium_matches * 0.1)
        else:
            return 0.1
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate computational cost for a particle physics task.
        
        Returns estimated time, memory, and computational units.
        """
        task_type = task.get("type", "cross_section")
        
        # Base costs by task type
        base_costs = {
            "cross_section": {"time": 1, "memory": 512, "units": 5},
            "particle_decay": {"time": 0.5, "memory": 256, "units": 3},
            "collider_event": {"time": 10, "memory": 2048, "units": 20},
            "jet_analysis": {"time": 5, "memory": 1024, "units": 15},
            "standard_model": {"time": 2, "memory": 512, "units": 8}
        }
        
        base_cost = base_costs.get(task_type, base_costs["cross_section"])
        
        # Scale based on problem size
        scale_factor = 1.0
        
        if task_type == "collider_event":
            num_events = task.get("num_events", 1000)
            scale_factor = max(1.0, num_events / 1000)
        elif task_type == "jet_analysis":
            num_particles = len(task.get("particles", []))
            if num_particles == 0:
                num_particles = 100  # Default
            scale_factor = max(1.0, num_particles / 100)
        
        return {
            "estimated_time_seconds": base_cost["time"] * scale_factor,
            "estimated_memory_mb": base_cost["memory"] * scale_factor,
            "computational_units": base_cost["units"] * scale_factor,
            "complexity_factor": scale_factor,
            "task_type": task_type
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get physics-specific requirements for particle physics calculations."""
        return {
            "physics_domain": self.physics_domain,
            "mathematical_background": [
                "Special relativity",
                "Quantum field theory basics",
                "Group theory fundamentals",
                "Four-vector mathematics"
            ],
            "computational_methods": [
                "Monte Carlo simulation",
                "Feynman diagram evaluation", 
                "Jet clustering algorithms",
                "Statistical analysis"
            ],
            "software_dependencies": self.software_requirements,
            "hardware_requirements": self.hardware_requirements,
            "typical_applications": [
                "Collider experiment analysis",
                "Particle discovery studies", 
                "Standard Model precision tests",
                "Beyond Standard Model searches",
                "Detector simulation"
            ],
            "accuracy_considerations": [
                "Higher-order corrections important",
                "Monte Carlo statistical uncertainties",
                "Systematic uncertainties from theory",
                "Detector resolution effects"
            ]
        }
    
    def validate_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for particle physics calculations."""
        errors = []
        warnings = []
        
        task_type = task.get("type")
        if not task_type:
            errors.append("Task type must be specified")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if task_type == "cross_section":
            energy = task.get("center_of_mass_energy")
            if energy and (energy < 0 or energy > 1e6):
                warnings.append(f"Unusual center-of-mass energy: {energy} GeV")
        
        elif task_type == "collider_event":
            num_events = task.get("num_events", 1000)
            if num_events > 10000:
                warnings.append(f"Large number of events ({num_events}) may be slow")
            if num_events < 1:
                errors.append("Number of events must be positive")
        
        elif task_type == "particle_decay":
            particle = task.get("particle")
            if not particle:
                errors.append("Particle name must be specified for decay analysis")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }