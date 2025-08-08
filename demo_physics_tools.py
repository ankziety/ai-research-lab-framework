#!/usr/bin/env python3
"""
Physics Tools Demonstration

This script demonstrates how AI agents can discover, request, and use
physics tools for research calculations and analysis.
"""

import sys
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')

from tools.physics.physics_tool_registry import PhysicsToolRegistry
from tools.physics.physics_tool_factory import PhysicsToolFactory


def demonstrate_agent_workflow():
    """Demonstrate a complete agent workflow using physics tools."""
    
    print("=" * 60)
    print("AI RESEARCH LAB FRAMEWORK - PHYSICS TOOLS DEMONSTRATION")
    print("=" * 60)
    print()
    
    # 1. Initialize the physics tools ecosystem
    print("1. Initializing Physics Tools Ecosystem...")
    registry = PhysicsToolRegistry(auto_register_default=True)
    factory = PhysicsToolFactory(registry=registry)
    
    print(f"   ✓ Registered {len(registry.physics_tools)} physics tools")
    print(f"   ✓ Available domains: {list(registry.domain_categories.keys())}")
    print()
    
    # 2. Agent discovers tools for research question
    print("2. Agent Tool Discovery...")
    research_question = "I need to calculate the electronic structure of a water molecule and analyze the results"
    
    print(f"   Research Question: '{research_question}'")
    
    recommendations = registry.discover_physics_tools(
        agent_id="demo_agent",
        research_question=research_question,
        physics_domain="quantum_chemistry"
    )
    
    print(f"   ✓ Found {len(recommendations)} tool recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"      {i}. {rec['name']} (confidence: {rec['confidence']:.3f})")
        print(f"         Domain: {rec['physics_domain']}")
        print(f"         Average cost: {rec['average_cost']:.1f} units")
    print()
    
    # 3. Agent requests and uses quantum chemistry tool
    print("3. Quantum Chemistry Calculation...")
    
    qc_tool = registry.request_physics_tool(
        agent_id="demo_agent",
        tool_id="quantum_chemistry_tool",
        task_specification={
            "type": "molecular_properties",
            "molecule": {
                "atoms": [
                    {"element": "O", "position": [0.0, 0.0, 0.0]},
                    {"element": "H", "position": [0.757, 0.587, 0.0]},
                    {"element": "H", "position": [-0.757, 0.587, 0.0]}
                ]
            },
            "method": "dft",
            "basis_set": "6-31g*"
        },
        context={"agent_id": "demo_agent", "available_memory": 4096}
    )
    
    if qc_tool:
        print("   ✓ Agent granted access to Quantum Chemistry Tool")
        
        # Execute calculation
        task = {
            "type": "molecular_properties",
            "molecule": {
                "atoms": [
                    {"element": "O", "position": [0.0, 0.0, 0.0]},
                    {"element": "H", "position": [0.757, 0.587, 0.0]},
                    {"element": "H", "position": [-0.757, 0.587, 0.0]}
                ]
            },
            "method": "dft",
            "basis_set": "6-31g*"
        }
        
        context = {"agent_id": "demo_agent"}
        
        print("   ✓ Executing molecular properties calculation...")
        result = qc_tool.execute(task, context)
        
        if result["success"]:
            print(f"   ✓ Calculation completed in {result['calculation_time']:.3f} seconds")
            print(f"   ✓ Computational cost: {result['computational_cost']:.1f} units")
            print(f"   ✓ Result confidence: {result['confidence']:.3f}")
            
            # Display key results
            results = result["results"]
            if "energies" in results:
                energy = results["energies"].get("total_energy", {})
                if energy:
                    print(f"   ✓ Total energy: {energy['value']:.6f} {energy['units']}")
            
            if "properties" in results:
                props = results["properties"]
                if "dipole_moment" in props:
                    dipole = props["dipole_moment"]
                    print(f"   ✓ Dipole moment: {dipole['magnitude']:.3f} {dipole['units']}")
        
        print()
    
    # 4. Agent uses experimental tool for analysis
    print("4. Statistical Analysis...")
    
    exp_tool = registry.request_physics_tool(
        agent_id="demo_agent",
        tool_id="experimental_tool",
        task_specification={"type": "descriptive_statistics"},
        context={"agent_id": "demo_agent"}
    )
    
    if exp_tool:
        print("   ✓ Agent granted access to Experimental Tool")
        
        # Generate some mock experimental data
        np.random.seed(42)
        experimental_data = {
            "energies": np.random.normal(-76.4, 0.1, 50).tolist(),
            "dipole_moments": np.random.normal(1.85, 0.05, 50).tolist()
        }
        
        analysis_task = {
            "type": "descriptive_statistics",
            "data": experimental_data
        }
        
        print("   ✓ Analyzing experimental data statistics...")
        analysis_result = exp_tool.execute(analysis_task, context)
        
        if analysis_result["success"]:
            print(f"   ✓ Analysis completed in {analysis_result['calculation_time']:.3f} seconds")
            
            stats = analysis_result["results"]["statistics"]
            if "mean" in stats:
                print(f"   ✓ Mean energy: {stats['mean']['value']:.6f}")
            if "standard_deviation" in stats:
                print(f"   ✓ Energy std dev: {stats['standard_deviation']['value']:.6f}")
        
        print()
    
    # 5. Agent creates visualization
    print("5. Data Visualization...")
    
    viz_tool = registry.request_physics_tool(
        agent_id="demo_agent",
        tool_id="visualization_tool",
        task_specification={"type": "line_plot"},
        context={"agent_id": "demo_agent"}
    )
    
    if viz_tool:
        print("   ✓ Agent granted access to Visualization Tool")
        
        # Create a simple visualization
        x = list(range(10))
        y = [i**2 for i in x]
        
        viz_task = {
            "type": "line_plot",
            "data": {"x": x, "y": y},
            "style": {
                "title": "Quadratic Function",
                "xlabel": "X",
                "ylabel": "X²",
                "marker": "o"
            },
            "physics_domain": "default"
        }
        
        print("   ✓ Creating scientific visualization...")
        viz_result = viz_tool.execute(viz_task, context)
        
        if viz_result["success"]:
            print(f"   ✓ Visualization created in {viz_result['calculation_time']:.3f} seconds")
            print(f"   ✓ Plot type: {viz_result['plot_type']}")
            
            if "image_data" in viz_result["results"]:
                img_size = viz_result["results"]["image_data"].get("size_bytes", 0)
                print(f"   ✓ Generated image: {img_size} bytes")
        
        print()
    
    # 6. Workflow recommendation
    print("6. Workflow Recommendation...")
    
    workflow = registry.recommend_physics_workflow(
        research_goal="comprehensive molecular analysis with visualization",
        available_data={"experimental_measurements": [1, 2, 3]},
        constraints={"max_cost": 200}
    )
    
    print(f"   ✓ Generated {len(workflow['workflow_steps'])}-step workflow:")
    for i, step in enumerate(workflow['workflow_steps'], 1):
        print(f"      Step {i}: {step['phase'].title()} - {step['purpose']}")
    
    print(f"   ✓ Estimated total time: {workflow['total_estimated_time']}")
    print()
    
    # 7. System statistics
    print("7. System Performance Statistics...")
    
    registry_stats = registry.get_registry_statistics()
    print(f"   ✓ Total physics tools: {registry_stats['total_physics_tools']}")
    print(f"   ✓ Physics domains: {len(registry_stats['physics_domains'])}")
    print(f"   ✓ Total calculations performed: {registry_stats['total_calculations']}")
    print(f"   ✓ Active agents: {registry_stats['active_agents']}")
    
    if registry_stats['most_used_tools']:
        most_used = registry_stats['most_used_tools'][0]
        print(f"   ✓ Most used tool: {most_used[0]} ({most_used[1]} uses)")
    
    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("• Agent tool discovery based on research questions")
    print("• Physics domain-specific tool recommendations")
    print("• Cost estimation for computational resource planning")
    print("• Multi-step workflow generation for complex research")
    print("• Integration of quantum chemistry, experimental analysis, and visualization")
    print("• Comprehensive error handling and validation")
    print("• Performance tracking and usage statistics")
    print()
    print("The physics tools package provides a complete framework for")
    print("AI agents to conduct sophisticated physics research with")
    print("computational tools designed for scientific rigor and accuracy.")


if __name__ == "__main__":
    demonstrate_agent_workflow()