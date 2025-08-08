"""
Physics Tools Package - Engine Integrated

Physics-specific tools that agents can request and use for research.
Provides computational interfaces for quantum chemistry, materials science,
astrophysics, experimental data analysis, and physics visualization.

Enhanced with physics engine integration for superior computational capabilities.
"""

__version__ = "1.0.0"

# Import base classes
from .base_physics_tool import BasePhysicsTool

# Import engine integration
from .engine_adapter import PhysicsEngineAdapter, get_physics_engine_adapter

# Import specific physics tools
from .quantum_chemistry_tool import QuantumChemistryTool

# Import registry and factory (when ready)
from .physics_tool_registry import PhysicsToolRegistry

# Try to import other tools - they may not all be available yet
try:
    from .materials_science_tool import MaterialsScienceTool
except ImportError:
    MaterialsScienceTool = None

try:
    from .astrophysics_tool import AstrophysicsTool
except ImportError:
    AstrophysicsTool = None

try:
    from .experimental_tool import ExperimentalTool
except ImportError:
    ExperimentalTool = None

try:
    from .visualization_tool import VisualizationTool
except ImportError:
    VisualizationTool = None

try:
    from .physics_tool_factory import PhysicsToolFactory
except ImportError:
    PhysicsToolFactory = None


# Available exports
__all__ = [
    # Base classes and interfaces
    'BasePhysicsTool',
    'PhysicsEngineAdapter',
    'get_physics_engine_adapter',
    
    # Physics tools
    'QuantumChemistryTool',
    
    # Registry and management
    'PhysicsToolRegistry',
]

# Add optional tools to exports if available
if MaterialsScienceTool:
    __all__.append('MaterialsScienceTool')
if AstrophysicsTool:
    __all__.append('AstrophysicsTool') 
if ExperimentalTool:
    __all__.append('ExperimentalTool')
if VisualizationTool:
    __all__.append('VisualizationTool')
if PhysicsToolFactory:
    __all__.append('PhysicsToolFactory')


def create_physics_research_team(research_question: str, 
                                prefer_engines: bool = True) -> PhysicsToolRegistry:
    """
    Create a physics research team for a specific research question.
    
    Automatically discovers and sets up appropriate physics tools
    based on the research question, with engine integration support.
    
    Args:
        research_question: Description of the physics research to be conducted
        prefer_engines: Whether to prefer physics engines when available
        
    Returns:
        Configured PhysicsToolRegistry with relevant tools
    """
    registry = PhysicsToolRegistry(auto_register_default=True)
    
    # Get tool recommendations
    recommendations = registry.discover_physics_tools(
        agent_id="research_team_builder",
        research_question=research_question,
        prefer_engines=prefer_engines
    )
    
    return registry


def get_engine_integration_status() -> dict:
    """
    Get the current status of physics engine integration.
    
    Returns:
        Dictionary with engine availability and integration status
    """
    adapter = get_physics_engine_adapter()
    return adapter.get_available_engines_summary()


def create_quantum_chemistry_tool(prefer_engines: bool = True) -> QuantumChemistryTool:
    """
    Create a quantum chemistry tool with engine integration.
    
    Args:
        prefer_engines: Whether to prefer physics engines when available
        
    Returns:
        Configured QuantumChemistryTool instance
    """
    tool = QuantumChemistryTool()
    tool.prefer_engines = prefer_engines
    return tool


# Convenience function for engine information
def show_physics_engines_info():
    """Display information about available physics engines."""
    adapter = get_physics_engine_adapter()
    summary = adapter.get_available_engines_summary()
    
    print("Physics Engine Integration Status")
    print("=" * 40)
    print(f"Adapter Initialized: {summary['adapter_initialized']}")
    print(f"Total Engines: {summary['total_engines']}")
    print(f"Available Engines: {', '.join(summary['available_engines'])}")
    print(f"Supported Domains: {', '.join(summary['supported_domains'])}")
    
    if summary['adapter_initialized']:
        print("\nEngine Mappings:")
        for domain, mapping in summary['engine_mappings'].items():
            print(f"  {domain}: {mapping['engine_type']}")
    else:
        print("\nPhysics engines not available - using fallback implementations")


# Integration validation function
def validate_integration() -> dict:
    """
    Validate the integration between physics tools and engines.
    
    Returns:
        Validation results with status and any issues found
    """
    results = {
        "integration_status": "checking",
        "tools_available": [],
        "engines_available": False,
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Check tool availability
        if QuantumChemistryTool:
            results["tools_available"].append("QuantumChemistryTool")
        if MaterialsScienceTool:
            results["tools_available"].append("MaterialsScienceTool")
        if AstrophysicsTool:
            results["tools_available"].append("AstrophysicsTool")
        if ExperimentalTool:
            results["tools_available"].append("ExperimentalTool")
        if VisualizationTool:
            results["tools_available"].append("VisualizationTool")
        
        # Check engine integration
        adapter = get_physics_engine_adapter()
        engine_summary = adapter.get_available_engines_summary()
        results["engines_available"] = engine_summary["adapter_initialized"]
        
        # Test tool creation
        qc_tool = QuantumChemistryTool()
        registry = PhysicsToolRegistry()
        
        if not results["engines_available"]:
            results["issues"].append("Physics engines not available")
            results["recommendations"].append("Install physics engine package from PR #18")
        
        if len(results["tools_available"]) < 2:
            results["issues"].append("Limited tool availability")
            results["recommendations"].append("Additional physics tools may not be imported")
        
        results["integration_status"] = "success" if not results["issues"] else "partial"
        
    except Exception as e:
        results["integration_status"] = "failed"
        results["issues"].append(f"Integration test failed: {str(e)}")
        results["recommendations"].append("Check package installation and imports")
    
    return results


# Module level convenience instances
default_registry = None

def get_default_registry() -> PhysicsToolRegistry:
    """Get the default physics tool registry instance."""
    global default_registry
    if default_registry is None:
        default_registry = PhysicsToolRegistry(auto_register_default=True)
    return default_registry