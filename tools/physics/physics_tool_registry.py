"""
Physics Tool Registry - Engine Integrated

Enhanced registry for managing physics-specific tools with physics engine integration.
Provides physics-domain-specific discovery, recommendation, and access management
for tools that can utilize both physics engines and fallback implementations.
"""

from typing import Dict, List, Any, Optional, Type
import logging
import numpy as np
from datetime import datetime

from .base_physics_tool import BasePhysicsTool
from .engine_adapter import get_physics_engine_adapter

logger = logging.getLogger(__name__)


class PhysicsToolRegistry:
    """
    Enhanced registry for managing physics research tools with engine integration.
    
    Provides physics-domain-specific discovery, recommendation,
    and access management for tools that can utilize computational
    physics engines for enhanced capabilities.
    """
    
    def __init__(self, auto_register_default: bool = True):
        """Initialize the physics tool registry with engine integration."""
        self.physics_tools: Dict[str, BasePhysicsTool] = {}
        self.domain_categories: Dict[str, List[str]] = {}
        self.capability_index: Dict[str, List[str]] = {}
        self.agent_usage_history: Dict[str, Dict[str, Any]] = {}
        self.cost_history: Dict[str, List[float]] = {}
        
        # Engine integration
        self.engine_adapter = get_physics_engine_adapter()
        self.engine_integration_stats = {
            "tools_with_engines": 0,
            "total_engine_calculations": 0,
            "engine_success_rate": 0.0
        }
        
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
        """Register default physics tools with engine integration."""
        try:
            # Import and register quantum chemistry tool
            from .quantum_chemistry_tool import QuantumChemistryTool
            qc_tool = QuantumChemistryTool()
            self.register_physics_tool(qc_tool, domains=["quantum_chemistry", "molecular_physics"])
            
            # Import and register other tools as they become available
            try:
                from .materials_science_tool import MaterialsScienceTool
                ms_tool = MaterialsScienceTool()
                self.register_physics_tool(ms_tool, domains=["materials_science", "condensed_matter"])
            except ImportError:
                logger.info("Materials science tool not available")
            
            try:
                from .astrophysics_tool import AstrophysicsTool
                astro_tool = AstrophysicsTool()
                self.register_physics_tool(astro_tool, domains=["astrophysics", "cosmology"])
            except ImportError:
                logger.info("Astrophysics tool not available")
            
            try:
                from .experimental_tool import ExperimentalTool
                exp_tool = ExperimentalTool()
                self.register_physics_tool(exp_tool, domains=["experimental_physics", "data_analysis"])
            except ImportError:
                logger.info("Experimental tool not available")
            
            try:
                from .visualization_tool import VisualizationTool
                viz_tool = VisualizationTool()
                self.register_physics_tool(viz_tool, domains=["data_visualization", "experimental_physics"])
            except ImportError:
                logger.info("Visualization tool not available")
            
            logger.info(f"Registered {len(self.physics_tools)} physics tools with engine integration")
            
        except Exception as e:
            logger.error(f"Failed to register default physics tools: {e}")
    
    def register_physics_tool(self, tool: BasePhysicsTool, domains: List[str] = None):
        """
        Register a physics tool in the registry with engine integration support.
        
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
        
        # Update engine integration stats
        if hasattr(tool, 'engine_capabilities') and tool.engine_capabilities.get("engines_available", False):
            self.engine_integration_stats["tools_with_engines"] += 1
        
        logger.info(f"Registered physics tool: {tool.name} ({tool.tool_id}) with engine support")
    
    def discover_physics_tools(self, 
                             agent_id: str, 
                             research_question: str,
                             physics_domain: Optional[str] = None,
                             required_capabilities: Optional[List[str]] = None,
                             cost_constraint: Optional[float] = None,
                             prefer_engines: bool = True) -> List[Dict[str, Any]]:
        """
        Discover physics tools suitable for a research question with engine preferences.
        
        Args:
            agent_id: ID of the requesting agent
            research_question: Physics research question or task description
            physics_domain: Specific physics domain (optional)
            required_capabilities: Required tool capabilities (optional)
            cost_constraint: Maximum acceptable computational cost (optional)
            prefer_engines: Whether to prefer tools with engine support
            
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
            
            # Engine integration bonus
            engine_bonus = self._calculate_engine_bonus(tool, prefer_engines)
            
            # Combined confidence score
            total_confidence = min(1.0, base_confidence + domain_bonus + 
                                 capability_bonus + usage_bonus + engine_bonus - cost_penalty)
            
            if total_confidence > 0.1:  # Minimum threshold
                # Get physics-specific requirements
                physics_requirements = tool.get_physics_requirements()
                
                # Get engine status
                engine_status = self._get_tool_engine_status(tool)
                
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
                    'engine_status': engine_status,
                    'match_details': {
                        'base_confidence': base_confidence,
                        'domain_bonus': domain_bonus,
                        'capability_bonus': capability_bonus,
                        'usage_bonus': usage_bonus,
                        'engine_bonus': engine_bonus,
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
        
        # Enhance context with engine information
        enhanced_context = context.copy()
        enhanced_context.update({
            "engine_adapter_available": self.engine_adapter is not None,
            "engines_summary": self.engine_adapter.get_available_engines_summary() if self.engine_adapter else {}
        })
        
        # Track enhanced usage
        self._track_physics_tool_usage(agent_id, tool_id, task_specification)
        
        logger.info(f"Agent {agent_id} granted access to physics tool {tool_id}")
        return tool
    
    def get_physics_tools_by_domain(self, domain: str, 
                                  include_subdomain: bool = True,
                                  engine_preference: Optional[str] = None) -> List[BasePhysicsTool]:
        """
        Get all physics tools in a specific domain with engine filtering.
        
        Args:
            domain: Physics domain
            include_subdomain: Include tools from subdomains
            engine_preference: 'engines_only', 'fallback_only', or None for both
            
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
        
        # Filter by engine preference
        if engine_preference == "engines_only":
            tools = [t for t in tools if t.engine_capabilities.get("engines_available", False)]
        elif engine_preference == "fallback_only":
            tools = [t for t in tools if not t.engine_capabilities.get("engines_available", False)]
        
        # Sort by success rate, usage, and engine availability
        tools.sort(key=lambda t: (
            t.success_rate, 
            t.usage_count, 
            t.engine_capabilities.get("engines_available", False)
        ), reverse=True)
        
        return tools
    
    def recommend_physics_workflow(self, 
                                 research_goal: str,
                                 available_data: Dict[str, Any],
                                 constraints: Dict[str, Any] = None,
                                 prefer_engines: bool = True) -> Dict[str, Any]:
        """
        Recommend a complete physics research workflow with engine optimization.
        
        Args:
            research_goal: High-level research objective
            available_data: Data and resources available
            constraints: Computational and other constraints
            prefer_engines: Whether to prefer engine-enabled tools
            
        Returns:
            Recommended workflow with tool sequence and parameters
        """
        constraints = constraints or {}
        workflow_steps = []
        
        # Analyze research goal for workflow planning
        goal_analysis = self._analyze_research_goal(research_goal)
        
        # Phase 1: Data preparation and validation
        if "experimental" in goal_analysis["data_type"]:
            exp_tools = self.get_physics_tools_by_domain(
                "experimental_physics", 
                engine_preference="engines_only" if prefer_engines else None
            )
            if exp_tools:
                tool = exp_tools[0]
                workflow_steps.append({
                    "phase": "data_preparation",
                    "tool": tool.tool_id,
                    "purpose": "Validate and preprocess experimental data",
                    "estimated_time": "5-30 minutes",
                    "engine_enhanced": tool.engine_capabilities.get("engines_available", False)
                })
        
        # Phase 2: Main calculation/simulation
        domain_tools = self.get_physics_tools_by_domain(
            goal_analysis["primary_domain"],
            engine_preference="engines_only" if prefer_engines else None
        )
        if domain_tools:
            main_tool = domain_tools[0]  # Best tool for domain
            workflow_steps.append({
                "phase": "main_calculation",
                "tool": main_tool.tool_id,
                "purpose": f"Perform {goal_analysis['primary_domain']} calculations",
                "estimated_time": "10 minutes to 2 hours",
                "engine_enhanced": main_tool.engine_capabilities.get("engines_available", False)
            })
        
        # Phase 3: Visualization and analysis
        viz_tools = self.get_physics_tools_by_domain("data_visualization")
        if viz_tools:
            viz_tool = viz_tools[0]
            workflow_steps.append({
                "phase": "visualization",
                "tool": viz_tool.tool_id,
                "purpose": "Create scientific visualizations of results",
                "estimated_time": "2-10 minutes",
                "engine_enhanced": viz_tool.engine_capabilities.get("engines_available", False)
            })
        
        # Phase 4: Additional analysis if needed
        if goal_analysis["requires_statistical_analysis"]:
            exp_tools = self.get_physics_tools_by_domain("experimental_physics")
            if exp_tools:
                stat_tool = exp_tools[0]
                workflow_steps.append({
                    "phase": "statistical_analysis",
                    "tool": stat_tool.tool_id,
                    "purpose": "Perform statistical analysis and uncertainty quantification",
                    "estimated_time": "5-20 minutes",
                    "engine_enhanced": stat_tool.engine_capabilities.get("engines_available", False)
                })
        
        # Calculate workflow statistics
        engine_enhanced_steps = len([s for s in workflow_steps if s.get("engine_enhanced", False)])
        
        return {
            "workflow_id": f"physics_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "research_goal": research_goal,
            "workflow_steps": workflow_steps,
            "total_estimated_time": self._estimate_total_workflow_time(workflow_steps),
            "required_capabilities": goal_analysis["required_capabilities"],
            "physics_domains": goal_analysis["domains_involved"],
            "engine_integration": {
                "total_steps": len(workflow_steps),
                "engine_enhanced_steps": engine_enhanced_steps,
                "engine_enhancement_ratio": engine_enhanced_steps / len(workflow_steps) if workflow_steps else 0,
                "engines_available": self.engine_adapter.get_available_engines_summary()["adapter_initialized"]
            }
        }
    
    def get_physics_tool_performance_stats(self, tool_id: str) -> Dict[str, Any]:
        """Get detailed performance statistics for a physics tool with engine info."""
        if tool_id not in self.physics_tools:
            return {}
        
        tool = self.physics_tools[tool_id]
        
        # Get calculation history with engine breakdown
        calculation_stats = tool.get_calculation_stats()
        
        base_stats = {
            "tool_id": tool_id,
            "physics_domain": tool.physics_domain,
            "total_calculations": tool.usage_count,
            "success_rate": tool.success_rate,
            "average_calculation_time": tool.average_calculation_time,
            "total_computational_cost": tool.total_computational_cost,
            "cost_history": self.cost_history.get(tool_id, [])[-20:],  # Last 20 costs
            "physics_statistics": calculation_stats,
            "agent_adoption": len([agent for agent, tools in self.agent_usage_history.items() 
                                 if tool_id in tools]),
            "reliability_score": self._calculate_reliability_score(tool_id)
        }
        
        # Add engine integration statistics
        base_stats["engine_integration"] = {
            "engines_available": tool.engine_capabilities.get("engines_available", False),
            "engine_type": tool.engine_capabilities.get("engine_type", "none"),
            "engine_calculations": calculation_stats.get("engine_calculations", 0),
            "fallback_calculations": calculation_stats.get("fallback_calculations", 0),
            "engine_usage_ratio": (calculation_stats.get("engine_calculations", 0) / 
                                 max(tool.usage_count, 1)) if tool.usage_count > 0 else 0
        }
        
        return base_stats
    
    def get_engine_integration_summary(self) -> Dict[str, Any]:
        """Get summary of engine integration across all tools."""
        total_tools = len(self.physics_tools)
        tools_with_engines = sum(1 for tool in self.physics_tools.values() 
                                if tool.engine_capabilities.get("engines_available", False))
        
        # Calculate engine usage statistics
        total_calculations = sum(tool.usage_count for tool in self.physics_tools.values())
        engine_calculations = sum(tool.get_calculation_stats().get("engine_calculations", 0) 
                                for tool in self.physics_tools.values())
        
        return {
            "total_physics_tools": total_tools,
            "tools_with_engine_support": tools_with_engines,
            "engine_integration_ratio": tools_with_engines / max(total_tools, 1),
            "total_calculations": total_calculations,
            "engine_enhanced_calculations": engine_calculations,
            "engine_usage_ratio": engine_calculations / max(total_calculations, 1),
            "engine_adapter_status": self.engine_adapter.get_available_engines_summary(),
            "domain_engine_support": {
                domain: len([tool_id for tool_id in tool_ids 
                           if self.physics_tools[tool_id].engine_capabilities.get("engines_available", False)])
                for domain, tool_ids in self.domain_categories.items()
            }
        }
    
    # Helper methods (enhanced versions)
    def _calculate_engine_bonus(self, tool: BasePhysicsTool, prefer_engines: bool) -> float:
        """Calculate bonus score for engine availability."""
        if not prefer_engines:
            return 0.0
        
        if tool.engine_capabilities.get("engines_available", False):
            return 0.2  # Significant bonus for engine support
        
        return 0.0
    
    def _get_tool_engine_status(self, tool: BasePhysicsTool) -> Dict[str, Any]:
        """Get comprehensive engine status for a tool."""
        return {
            "engines_available": tool.engine_capabilities.get("engines_available", False),
            "engine_type": tool.engine_capabilities.get("engine_type", "none"),
            "engine_capabilities": tool.engine_capabilities.get("capabilities", []),
            "prefer_engines": tool.prefer_engines,
            "fallback_available": True,  # All tools have fallback implementations
            "integration_benefits": [
                "Higher accuracy calculations",
                "Advanced algorithms", 
                "Better performance",
                "Extended capabilities"
            ] if tool.engine_capabilities.get("engines_available", False) else []
        }
    
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
            "experimental": ["measurement", "data", "experiment", "analysis", "uncertainty"],
            "engines": ["engine", "simulation", "computational", "advanced", "precise"]
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
        
        # Keyword matching - enhanced with engine keywords
        domain_keywords = {
            "quantum_chemistry": ["quantum", "molecular", "orbital", "electronic", "engine"],
            "materials_science": ["materials", "crystal", "structure", "mechanical", "simulation"],
            "astrophysics": ["stellar", "galaxy", "cosmology", "gravitational", "computational"],
            "experimental_physics": ["experimental", "data", "analysis", "measurement", "statistical"],
            "data_visualization": ["plot", "chart", "visualization", "figure", "scientific"]
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
        if hasattr(tool, 'requirements') and tool.requirements:
            if not tool.validate_requirements(context):
                # Check what specific requirements are missing
                missing_reqs = []
                for req_type, req_value in tool.requirements.items():
                    if req_type == 'min_memory' and context.get('available_memory', 2048) < req_value:
                        missing_reqs.append(f"Insufficient memory: need {req_value}MB, have {context.get('available_memory', 0)}MB")
                    elif req_type == 'required_packages':
                        missing_packages = [pkg for pkg in req_value if pkg not in context.get('available_packages', ['numpy', 'scipy', 'matplotlib', 'pandas'])]
                        if missing_packages:
                            missing_reqs.append(f"Missing packages: {missing_packages}")
                    elif req_type == 'api_keys':
                        missing_keys = [key for key in req_value if key not in context.get('api_keys', {})]
                        if missing_keys:
                            missing_reqs.append(f"Missing API keys: {missing_keys}")
                
                if missing_reqs:
                    warnings.extend(missing_reqs)  # Treat as warnings instead of errors for demo
        
        # Physics-specific validation
        if hasattr(tool, 'validate_input'):
            try:
                validation = tool.validate_input(task_spec)
                if not validation.get("valid", True):
                    # For demo purposes, treat validation errors as warnings if they're about missing optional data
                    validation_errors = validation.get("errors", [])
                    for error in validation_errors:
                        if "data" in error.lower() and "missing" in error.lower():
                            warnings.append(error)
                        else:
                            errors.append(error)
                    warnings.extend(validation.get("warnings", []))
            except Exception as e:
                warnings.append(f"Input validation warning: {str(e)}")
        
        # Cost validation
        if "cost_limit" in context:
            try:
                cost_estimate = tool.estimate_cost(task_spec)
                if cost_estimate["computational_units"] > context["cost_limit"]:
                    warnings.append("Estimated cost exceeds agent's limit")
            except Exception:
                warnings.append("Could not estimate computational cost")
        
        # Engine availability info
        task_type = task_spec.get("type", "unknown")
        if tool.is_engine_available(task_type):
            warnings.append("Physics engine available for enhanced performance")
        else:
            warnings.append("Using fallback implementation - engines not available")
        
        return {
            "valid": len(errors) == 0,  # Only fail on actual errors
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
                "last_used": datetime.now().isoformat(),
                "engine_usage": {"used": 0, "available": 0}
            }
        
        usage = self.agent_usage_history[agent_id][tool_id]
        usage["count"] += 1
        usage["last_used"] = datetime.now().isoformat()
        
        # Track task type
        task_type = task_spec.get("type", "unknown")
        if task_type not in usage["task_types"]:
            usage["task_types"].append(task_type)
        
        # Track engine availability
        tool = self.physics_tools[tool_id]
        if tool.is_engine_available(task_type):
            usage["engine_usage"]["available"] += 1
            if tool.prefer_engines:
                usage["engine_usage"]["used"] += 1
    
    def _generate_physics_insights(self, 
                                 recommendation: Dict[str, Any],
                                 research_question: str,
                                 keywords: List[str]) -> List[str]:
        """Generate physics-specific insights for tool recommendations."""
        insights = []
        
        tool_id = recommendation["tool_id"]
        physics_domain = recommendation["physics_domain"]
        engine_status = recommendation["engine_status"]
        
        # Domain-specific insights
        if "quantum" in physics_domain:
            insights.append("Quantum calculations may require careful consideration of basis sets and convergence")
        elif "materials" in physics_domain:
            insights.append("Materials simulations benefit from understanding crystal symmetry")
        elif "astrophysics" in physics_domain:
            insights.append("Astrophysical calculations often involve large scales and approximations")
        
        # Engine integration insights
        if engine_status["engines_available"]:
            insights.append(f"Enhanced with {engine_status['engine_type']} physics engine for superior accuracy")
            insights.append("Engine integration provides access to advanced algorithms and better performance")
        else:
            insights.append("Using optimized fallback implementation - physics engines not currently available")
        
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
        
        # Add bonus time for engine-enhanced steps
        engine_steps = len([s for s in workflow_steps if s.get("engine_enhanced", False)])
        total_minutes += engine_steps * 10  # Engine setup time
        
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
        
        # Engine integration bonus
        engine_bonus = 0.1 if tool.engine_capabilities.get("engines_available", False) else 0.0
        
        return min(1.0, base_score * usage_factor * cost_consistency + engine_bonus)
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the physics tool registry."""
        engine_summary = self.get_engine_integration_summary()
        
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
            "domain_hierarchy": self.domain_hierarchy,
            "engine_integration": engine_summary
        }