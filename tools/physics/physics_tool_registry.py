"""
Physics Tool Registry

Separate registry for managing physics-specific tools that agents can discover and use.
Provides enhanced discovery, recommendation, and access management specifically
tailored for physics research tools.
"""

from typing import Dict, List, Any, Optional, Type
import logging
import numpy as np
from datetime import datetime

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class PhysicsToolRegistry:
    """
    Specialized registry for managing physics research tools.
    
    Provides physics-domain-specific discovery, recommendation,
    and access management for tools that agents can use to
    conduct physics research and calculations.
    """
    
    def __init__(self, auto_register_default: bool = True):
        """Initialize the physics tool registry."""
        self.physics_tools: Dict[str, BasePhysicsTool] = {}
        self.domain_categories: Dict[str, List[str]] = {}
        self.capability_index: Dict[str, List[str]] = {}
        self.agent_usage_history: Dict[str, Dict[str, Any]] = {}
        self.cost_history: Dict[str, List[float]] = {}
        
        # Physics domain hierarchy
        self.domain_hierarchy = {
            "quantum_mechanics": ["atomic_physics", "molecular_physics", "condensed_matter"],
            "classical_mechanics": ["fluid_dynamics", "solid_mechanics", "thermodynamics"],
            "electromagnetism": ["optics", "plasma_physics", "electronics"],
            "astrophysics": ["stellar_physics", "cosmology", "galactic_dynamics"],
            "materials_science": ["crystallography", "electronic_materials", "magnetic_materials"],
            "experimental_physics": ["data_analysis", "instrumentation", "metrology"]
        }
        
        if auto_register_default:
            self._register_default_physics_tools()
    
    def _register_default_physics_tools(self):
        """Register default physics tools."""
        try:
            # Import and register quantum chemistry tool
            from .quantum_chemistry_tool import QuantumChemistryTool
            qc_tool = QuantumChemistryTool()
            self.register_physics_tool(qc_tool, domains=["quantum_chemistry", "molecular_physics"])
            
            # Import and register materials science tool
            from .materials_science_tool import MaterialsScienceTool
            ms_tool = MaterialsScienceTool()
            self.register_physics_tool(ms_tool, domains=["materials_science", "condensed_matter"])
            
            # Import and register astrophysics tool
            from .astrophysics_tool import AstrophysicsTool
            astro_tool = AstrophysicsTool()
            self.register_physics_tool(astro_tool, domains=["astrophysics", "cosmology"])
            
            # Import and register experimental tool
            from .experimental_tool import ExperimentalTool
            exp_tool = ExperimentalTool()
            self.register_physics_tool(exp_tool, domains=["experimental_physics", "data_analysis"])
            
            # Import and register visualization tool
            from .visualization_tool import VisualizationTool
            viz_tool = VisualizationTool()
            self.register_physics_tool(viz_tool, domains=["data_visualization", "experimental_physics"])
            
            logger.info(f"Registered {len(self.physics_tools)} physics tools in PhysicsToolRegistry")
            
        except Exception as e:
            logger.error(f"Failed to register default physics tools: {e}")
    
    def register_physics_tool(self, tool: BasePhysicsTool, domains: List[str] = None):
        """
        Register a physics tool in the registry.
        
        Args:
            tool: Physics tool instance to register
            domains: Physics domains this tool belongs to
        """
        self.physics_tools[tool.tool_id] = tool
        
        # Add to domain categories
        domains = domains or [tool.physics_domain]
        for domain in domains:
            if domain not in self.domain_categories:
                self.domain_categories[domain] = []
            self.domain_categories[domain].append(tool.tool_id)
        
        # Index capabilities
        for capability in tool.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(tool.tool_id)
        
        # Initialize cost tracking
        self.cost_history[tool.tool_id] = []
        
        logger.info(f"Registered physics tool: {tool.name} ({tool.tool_id})")
    
    def discover_physics_tools(self, 
                             agent_id: str, 
                             research_question: str,
                             physics_domain: Optional[str] = None,
                             required_capabilities: Optional[List[str]] = None,
                             cost_constraint: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Discover physics tools suitable for a research question.
        
        Args:
            agent_id: ID of the requesting agent
            research_question: Physics research question or task description
            physics_domain: Specific physics domain (optional)
            required_capabilities: Required tool capabilities (optional)
            cost_constraint: Maximum acceptable computational cost (optional)
            
        Returns:
            List of recommended physics tools with enhanced metadata
        """
        recommendations = []
        
        # Parse research question for physics keywords
        physics_keywords = self._extract_physics_keywords(research_question)
        
        for tool_id, tool in self.physics_tools.items():
            # Calculate base confidence based on tool's capability assessment
            base_confidence = tool.can_handle(research_question, {})
            
            # Physics domain matching bonus
            domain_bonus = self._calculate_domain_match_bonus(
                tool.physics_domain, physics_domain, physics_keywords
            )
            
            # Capability matching bonus
            capability_bonus = self._calculate_capability_bonus(
                tool.capabilities, required_capabilities
            )
            
            # Agent usage history bonus
            usage_bonus = self._get_agent_usage_bonus(agent_id, tool_id)
            
            # Cost penalty if constraint specified
            cost_penalty = self._calculate_cost_penalty(tool_id, cost_constraint)
            
            # Combined confidence score
            total_confidence = min(1.0, base_confidence + domain_bonus + 
                                 capability_bonus + usage_bonus - cost_penalty)
            
            if total_confidence > 0.1:  # Minimum threshold
                # Get physics-specific requirements
                physics_requirements = tool.get_physics_requirements()
                
                recommendation = {
                    'tool_id': tool_id,
                    'confidence': total_confidence,
                    'name': tool.name,
                    'description': tool.description,
                    'physics_domain': tool.physics_domain,
                    'capabilities': tool.capabilities,
                    'success_rate': tool.success_rate,
                    'usage_count': tool.usage_count,
                    'average_cost': self._get_average_cost(tool_id),
                    'physics_requirements': physics_requirements,
                    'computational_cost_factor': tool.computational_cost_factor,
                    'match_details': {
                        'base_confidence': base_confidence,
                        'domain_bonus': domain_bonus,
                        'capability_bonus': capability_bonus,
                        'usage_bonus': usage_bonus,
                        'cost_penalty': cost_penalty
                    }
                }
                
                recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add physics-specific insights
        for rec in recommendations:
            rec['physics_insights'] = self._generate_physics_insights(
                rec, research_question, physics_keywords
            )
        
        return recommendations
    
    def request_physics_tool(self, 
                           agent_id: str, 
                           tool_id: str,
                           task_specification: Dict[str, Any],
                           context: Dict[str, Any]) -> Optional[BasePhysicsTool]:
        """
        Request access to a specific physics tool with enhanced validation.
        
        Args:
            agent_id: ID of the requesting agent
            tool_id: ID of the requested physics tool
            task_specification: Detailed task specification
            context: Enhanced execution context
            
        Returns:
            Physics tool instance if accessible, None if not available
        """
        if tool_id not in self.physics_tools:
            logger.warning(f"Physics tool {tool_id} not found in registry")
            return None
        
        tool = self.physics_tools[tool_id]
        
        # Enhanced validation for physics tools
        validation_result = self._validate_physics_tool_request(
            tool, task_specification, context
        )
        
        if not validation_result["valid"]:
            logger.warning(f"Physics tool {tool_id} validation failed for agent {agent_id}: "
                         f"{validation_result['errors']}")
            return None
        
        # Track enhanced usage
        self._track_physics_tool_usage(agent_id, tool_id, task_specification)
        
        logger.info(f"Agent {agent_id} granted access to physics tool {tool_id}")
        return tool
    
    def get_physics_tools_by_domain(self, domain: str, 
                                  include_subdomain: bool = True) -> List[BasePhysicsTool]:
        """
        Get all physics tools in a specific domain.
        
        Args:
            domain: Physics domain
            include_subdomain: Include tools from subdomains
            
        Returns:
            List of physics tools in the domain
        """
        tools = []
        
        # Direct domain match
        if domain in self.domain_categories:
            tools.extend([self.physics_tools[tool_id] 
                         for tool_id in self.domain_categories[domain]])
        
        # Include subdomain tools if requested
        if include_subdomain and domain in self.domain_hierarchy:
            for subdomain in self.domain_hierarchy[domain]:
                if subdomain in self.domain_categories:
                    tools.extend([self.physics_tools[tool_id] 
                                 for tool_id in self.domain_categories[subdomain]])
        
        # Sort by success rate and usage
        tools.sort(key=lambda t: (t.success_rate, t.usage_count), reverse=True)
        return tools
    
    def recommend_physics_workflow(self, 
                                 research_goal: str,
                                 available_data: Dict[str, Any],
                                 constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recommend a complete physics research workflow using multiple tools.
        
        Args:
            research_goal: High-level research objective
            available_data: Data and resources available
            constraints: Computational and other constraints
            
        Returns:
            Recommended workflow with tool sequence and parameters
        """
        constraints = constraints or {}
        workflow_steps = []
        
        # Analyze research goal for workflow planning
        goal_analysis = self._analyze_research_goal(research_goal)
        
        # Phase 1: Data preparation and validation
        if "experimental" in goal_analysis["data_type"]:
            exp_tools = self.get_physics_tools_by_domain("experimental_physics")
            if exp_tools:
                workflow_steps.append({
                    "phase": "data_preparation",
                    "tool": exp_tools[0].tool_id,
                    "purpose": "Validate and preprocess experimental data",
                    "estimated_time": "5-30 minutes"
                })
        
        # Phase 2: Main calculation/simulation
        domain_tools = self.get_physics_tools_by_domain(goal_analysis["primary_domain"])
        if domain_tools:
            main_tool = domain_tools[0]  # Best tool for domain
            workflow_steps.append({
                "phase": "main_calculation",
                "tool": main_tool.tool_id,
                "purpose": f"Perform {goal_analysis['primary_domain']} calculations",
                "estimated_time": "10 minutes to 2 hours"
            })
        
        # Phase 3: Visualization and analysis
        viz_tools = self.get_physics_tools_by_domain("data_visualization")
        if viz_tools:
            workflow_steps.append({
                "phase": "visualization",
                "tool": viz_tools[0].tool_id,
                "purpose": "Create scientific visualizations of results",
                "estimated_time": "2-10 minutes"
            })
        
        # Phase 4: Additional analysis if needed
        if goal_analysis["requires_statistical_analysis"]:
            exp_tools = self.get_physics_tools_by_domain("experimental_physics")
            if exp_tools:
                workflow_steps.append({
                    "phase": "statistical_analysis",
                    "tool": exp_tools[0].tool_id,
                    "purpose": "Perform statistical analysis and uncertainty quantification",
                    "estimated_time": "5-20 minutes"
                })
        
        return {
            "workflow_id": f"physics_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "research_goal": research_goal,
            "workflow_steps": workflow_steps,
            "total_estimated_time": self._estimate_total_workflow_time(workflow_steps),
            "required_capabilities": goal_analysis["required_capabilities"],
            "physics_domains": goal_analysis["domains_involved"]
        }
    
    def get_physics_tool_performance_stats(self, tool_id: str) -> Dict[str, Any]:
        """Get detailed performance statistics for a physics tool."""
        if tool_id not in self.physics_tools:
            return {}
        
        tool = self.physics_tools[tool_id]
        
        return {
            "tool_id": tool_id,
            "physics_domain": tool.physics_domain,
            "total_calculations": tool.usage_count,
            "success_rate": tool.success_rate,
            "average_calculation_time": tool.average_calculation_time,
            "total_computational_cost": tool.total_computational_cost,
            "cost_history": self.cost_history.get(tool_id, [])[-20:],  # Last 20 costs
            "physics_statistics": tool.get_calculation_stats(),
            "agent_adoption": len([agent for agent, tools in self.agent_usage_history.items() 
                                 if tool_id in tools]),
            "reliability_score": self._calculate_reliability_score(tool_id)
        }
    
    def suggest_physics_tools_for_learning(self, 
                                         agent_id: str,
                                         learning_goals: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest physics tools for an agent's learning and development.
        
        Args:
            agent_id: ID of the learning agent
            learning_goals: List of physics topics/skills to learn
            
        Returns:
            List of suggested tools with learning pathways
        """
        suggestions = []
        
        # Get agent's current tool usage
        agent_usage = self.agent_usage_history.get(agent_id, {})
        used_tools = set(agent_usage.keys())
        
        # Map learning goals to physics domains
        goal_domains = []
        for goal in learning_goals:
            domains = self._map_learning_goal_to_domains(goal)
            goal_domains.extend(domains)
        
        # Find tools in target domains not yet used by agent
        for domain in goal_domains:
            domain_tools = self.get_physics_tools_by_domain(domain)
            
            for tool in domain_tools:
                if tool.tool_id not in used_tools:
                    difficulty = self._assess_tool_difficulty(tool)
                    learning_value = self._assess_learning_value(tool, learning_goals)
                    
                    suggestion = {
                        "tool": tool.tool_id,
                        "name": tool.name,
                        "physics_domain": tool.physics_domain,
                        "difficulty": difficulty,
                        "learning_value": learning_value,
                        "prerequisites": self._get_tool_prerequisites(tool),
                        "learning_pathway": self._generate_learning_pathway(tool, agent_usage),
                        "estimated_learning_time": self._estimate_learning_time(tool, difficulty)
                    }
                    
                    suggestions.append(suggestion)
        
        # Sort by learning value and appropriate difficulty
        suggestions.sort(key=lambda x: (x["learning_value"], -x["difficulty"]), reverse=True)
        
        return suggestions[:10]  # Top 10 suggestions
    
    def _extract_physics_keywords(self, text: str) -> List[str]:
        """Extract physics-related keywords from text."""
        physics_keywords = []
        
        # Physics domain keywords
        domain_keywords = {
            "quantum": ["quantum", "wave", "particle", "orbital", "electron", "photon"],
            "classical": ["force", "energy", "momentum", "mass", "velocity", "acceleration"],
            "thermal": ["temperature", "heat", "entropy", "thermal", "thermodynamics"],
            "electromagnetic": ["electric", "magnetic", "field", "charge", "current"],
            "astrophysics": ["star", "galaxy", "cosmology", "planet", "orbit", "gravitational"],
            "materials": ["crystal", "material", "solid", "lattice", "structure"],
            "experimental": ["measurement", "data", "experiment", "analysis", "uncertainty"]
        }
        
        text_lower = text.lower()
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    physics_keywords.append(keyword)
        
        return physics_keywords
    
    def _calculate_domain_match_bonus(self, 
                                    tool_domain: str, 
                                    requested_domain: Optional[str],
                                    keywords: List[str]) -> float:
        """Calculate bonus for domain matching."""
        bonus = 0.0
        
        # Direct domain match
        if requested_domain and tool_domain == requested_domain:
            bonus += 0.3
        
        # Keyword matching - this is a simplified implementation
        # In practice, would get keywords from the specific tool
        domain_keywords = {
            "quantum_chemistry": ["quantum", "molecular", "orbital", "electronic"],
            "materials_science": ["materials", "crystal", "structure", "mechanical"],
            "astrophysics": ["stellar", "galaxy", "cosmology", "gravitational"],
            "experimental_physics": ["experimental", "data", "analysis", "measurement"],
            "data_visualization": ["plot", "chart", "visualization", "figure"]
        }
        
        tool_keywords = domain_keywords.get(tool_domain, [])
        keyword_matches = len(set(keywords) & set(tool_keywords))
        bonus += min(0.2, keyword_matches * 0.05)
        
        return bonus
    
    def _calculate_capability_bonus(self, 
                                  tool_capabilities: List[str],
                                  required_capabilities: Optional[List[str]]) -> float:
        """Calculate bonus for capability matching."""
        if not required_capabilities:
            return 0.0
        
        matches = len(set(tool_capabilities) & set(required_capabilities))
        return min(0.3, matches * 0.1)
    
    def _get_agent_usage_bonus(self, agent_id: str, tool_id: str) -> float:
        """Calculate usage bonus for agent-tool pair."""
        if agent_id not in self.agent_usage_history:
            return 0.0
        
        agent_tools = self.agent_usage_history[agent_id]
        if tool_id not in agent_tools:
            return 0.0
        
        usage_count = agent_tools[tool_id].get("count", 0)
        success_rate = agent_tools[tool_id].get("success_rate", 0.5)
        
        # Bonus for successful previous usage
        return min(0.15, usage_count * 0.02 * success_rate)
    
    def _calculate_cost_penalty(self, tool_id: str, cost_constraint: Optional[float]) -> float:
        """Calculate penalty for tools exceeding cost constraints."""
        if not cost_constraint:
            return 0.0
        
        tool = self.physics_tools[tool_id]
        avg_cost = self._get_average_cost(tool_id)
        
        if avg_cost > cost_constraint:
            # Exponential penalty for cost overrun
            overrun_factor = avg_cost / cost_constraint
            return min(0.5, (overrun_factor - 1) * 0.3)
        
        return 0.0
    
    def _get_average_cost(self, tool_id: str) -> float:
        """Get average computational cost for a tool."""
        costs = self.cost_history.get(tool_id, [])
        if not costs:
            return self.physics_tools[tool_id].computational_cost_factor * 10
        return sum(costs) / len(costs)
    
    def _validate_physics_tool_request(self, 
                                     tool: BasePhysicsTool,
                                     task_spec: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation for physics tool requests."""
        errors = []
        warnings = []
        
        # Basic validation using tool's method
        if not tool.validate_requirements(context):
            errors.append("Tool requirements not met in execution context")
        
        # Physics-specific validation
        if hasattr(tool, 'validate_input'):
            try:
                validation = tool.validate_input(task_spec)
                if not validation.get("valid", True):
                    errors.extend(validation.get("errors", []))
                    warnings.extend(validation.get("warnings", []))
            except Exception as e:
                errors.append(f"Input validation failed: {str(e)}")
        
        # Cost validation
        if "cost_limit" in context:
            try:
                cost_estimate = tool.estimate_cost(task_spec)
                if cost_estimate["computational_units"] > context["cost_limit"]:
                    errors.append("Estimated cost exceeds agent's limit")
            except Exception:
                warnings.append("Could not estimate computational cost")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _track_physics_tool_usage(self, 
                                agent_id: str, 
                                tool_id: str,
                                task_spec: Dict[str, Any]):
        """Track enhanced usage statistics for physics tools."""
        if agent_id not in self.agent_usage_history:
            self.agent_usage_history[agent_id] = {}
        
        if tool_id not in self.agent_usage_history[agent_id]:
            self.agent_usage_history[agent_id][tool_id] = {
                "count": 0,
                "success_count": 0,
                "total_cost": 0.0,
                "task_types": [],
                "first_used": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat()
            }
        
        usage = self.agent_usage_history[agent_id][tool_id]
        usage["count"] += 1
        usage["last_used"] = datetime.now().isoformat()
        
        # Track task type
        task_type = task_spec.get("type", "unknown")
        if task_type not in usage["task_types"]:
            usage["task_types"].append(task_type)
    
    def _generate_physics_insights(self, 
                                 recommendation: Dict[str, Any],
                                 research_question: str,
                                 keywords: List[str]) -> List[str]:
        """Generate physics-specific insights for tool recommendations."""
        insights = []
        
        tool_id = recommendation["tool_id"]
        physics_domain = recommendation["physics_domain"]
        
        # Domain-specific insights
        if "quantum" in physics_domain:
            insights.append("Quantum calculations may require careful consideration of basis sets and convergence")
        elif "materials" in physics_domain:
            insights.append("Materials simulations benefit from understanding crystal symmetry")
        elif "astrophysics" in physics_domain:
            insights.append("Astrophysical calculations often involve large scales and approximations")
        
        # Cost insights
        if recommendation["computational_cost_factor"] > 2.0:
            insights.append("High computational cost - consider starting with smaller test calculations")
        
        # Usage insights
        if recommendation["usage_count"] > 100:
            insights.append("Well-tested tool with extensive usage history")
        elif recommendation["usage_count"] < 10:
            insights.append("Newer tool - consider checking documentation and examples")
        
        return insights
    
    def _analyze_research_goal(self, goal: str) -> Dict[str, Any]:
        """Analyze research goal to determine workflow requirements."""
        goal_lower = goal.lower()
        
        # Determine primary domain
        domain_indicators = {
            "quantum_chemistry": ["molecule", "quantum", "orbital", "electronic"],
            "materials_science": ["material", "crystal", "solid", "mechanical"],
            "astrophysics": ["star", "galaxy", "cosmic", "gravitational"],
            "experimental_physics": ["measurement", "data", "experiment", "analysis"]
        }
        
        primary_domain = "experimental_physics"  # default
        for domain, indicators in domain_indicators.items():
            if any(indicator in goal_lower for indicator in indicators):
                primary_domain = domain
                break
        
        # Determine data type
        data_type = "theoretical"
        if any(word in goal_lower for word in ["data", "measurement", "experiment"]):
            data_type = "experimental"
        elif any(word in goal_lower for word in ["simulation", "calculate", "model"]):
            data_type = "computational"
        
        # Determine if statistical analysis is needed
        requires_stats = any(word in goal_lower for word in 
                           ["uncertainty", "error", "statistics", "correlation", "fit"])
        
        return {
            "primary_domain": primary_domain,
            "data_type": data_type,
            "requires_statistical_analysis": requires_stats,
            "domains_involved": [primary_domain],
            "required_capabilities": ["physics_calculation", "result_formatting"]
        }
    
    def _estimate_total_workflow_time(self, workflow_steps: List[Dict[str, Any]]) -> str:
        """Estimate total time for workflow completion."""
        # Simple estimation based on step count and complexity
        total_minutes = len(workflow_steps) * 15  # Base time per step
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours} hours {minutes} minutes"
    
    def _calculate_reliability_score(self, tool_id: str) -> float:
        """Calculate reliability score for a physics tool."""
        tool = self.physics_tools[tool_id]
        
        # Base score from success rate
        base_score = tool.success_rate
        
        # Adjust for usage count (more usage = more reliable data)
        usage_factor = min(1.0, tool.usage_count / 50)  # Normalize at 50 uses
        
        # Adjust for recent performance
        recent_costs = self.cost_history.get(tool_id, [])[-10:]
        cost_consistency = 1.0
        if len(recent_costs) > 1:
            cost_variation = np.std(recent_costs) / (np.mean(recent_costs) + 1e-6)
            cost_consistency = max(0.5, 1.0 - cost_variation)
        
        return base_score * usage_factor * cost_consistency
    
    def _map_learning_goal_to_domains(self, goal: str) -> List[str]:
        """Map learning goal to physics domains."""
        goal_lower = goal.lower()
        domains = []
        
        domain_mapping = {
            "quantum": ["quantum_chemistry"],
            "materials": ["materials_science"],
            "astronomy": ["astrophysics"],
            "experimental": ["experimental_physics"],
            "data": ["experimental_physics", "data_visualization"]
        }
        
        for keyword, mapped_domains in domain_mapping.items():
            if keyword in goal_lower:
                domains.extend(mapped_domains)
        
        return domains if domains else ["experimental_physics"]
    
    def _assess_tool_difficulty(self, tool: BasePhysicsTool) -> float:
        """Assess difficulty level of using a physics tool."""
        # Base difficulty from computational complexity
        base_difficulty = tool.computational_cost_factor / 5.0
        
        # Adjust for number of required parameters
        requirements = tool.get_physics_requirements()
        param_complexity = len(requirements.get("software_dependencies", [])) * 0.1
        
        return min(1.0, base_difficulty + param_complexity)
    
    def _assess_learning_value(self, tool: BasePhysicsTool, learning_goals: List[str]) -> float:
        """Assess learning value of a tool for given goals."""
        # Base value from tool capabilities
        base_value = len(tool.capabilities) * 0.1
        
        # Bonus for matching learning goals
        goal_text = " ".join(learning_goals).lower()
        domain_match = 1.0 if tool.physics_domain in goal_text else 0.5
        
        return min(1.0, base_value * domain_match)
    
    def _get_tool_prerequisites(self, tool: BasePhysicsTool) -> List[str]:
        """Get prerequisites for using a physics tool."""
        prerequisites = []
        
        # Software prerequisites
        requirements = tool.get_physics_requirements()
        if requirements.get("software_dependencies"):
            prerequisites.append("Install required software packages")
        
        # Domain knowledge prerequisites
        domain = tool.physics_domain
        if "quantum" in domain:
            prerequisites.append("Basic quantum mechanics knowledge")
        elif "materials" in domain:
            prerequisites.append("Understanding of crystal structures")
        elif "astrophysics" in domain:
            prerequisites.append("Familiarity with astronomical units and scales")
        
        return prerequisites
    
    def _generate_learning_pathway(self, 
                                 tool: BasePhysicsTool,
                                 agent_usage: Dict[str, Any]) -> List[str]:
        """Generate learning pathway for a physics tool."""
        pathway = []
        
        # Start with documentation
        pathway.append("Review tool documentation and capabilities")
        
        # Suggest starting with simple examples
        pathway.append("Try basic examples with known results")
        
        # Progress to more complex tasks
        pathway.append("Gradually increase problem complexity")
        
        # Integration with other tools
        if len(agent_usage) > 0:
            pathway.append("Combine with previously learned tools for complete workflows")
        
        return pathway
    
    def _estimate_learning_time(self, tool: BasePhysicsTool, difficulty: float) -> str:
        """Estimate time needed to learn a physics tool."""
        base_hours = 2  # Base learning time
        difficulty_multiplier = 1 + difficulty * 2
        
        total_hours = base_hours * difficulty_multiplier
        
        if total_hours < 1:
            return "30-60 minutes"
        elif total_hours < 4:
            return f"{int(total_hours)} hours"
        else:
            return f"{int(total_hours)} hours over multiple sessions"
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the physics tool registry."""
        return {
            "total_physics_tools": len(self.physics_tools),
            "physics_domains": list(self.domain_categories.keys()),
            "tools_per_domain": {
                domain: len(tools) for domain, tools in self.domain_categories.items()
            },
            "total_capabilities": len(self.capability_index),
            "most_used_tools": sorted(
                [(tool.tool_id, tool.usage_count) for tool in self.physics_tools.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "highest_success_rate": sorted(
                [(tool.tool_id, tool.success_rate) for tool in self.physics_tools.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "active_agents": len(self.agent_usage_history),
            "total_calculations": sum(tool.usage_count for tool in self.physics_tools.values()),
            "domain_hierarchy": self.domain_hierarchy
        }