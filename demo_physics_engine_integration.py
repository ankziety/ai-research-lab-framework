#!/usr/bin/env python3
"""
Physics Tools with Engine Integration Demo

This script demonstrates how AI agents can use physics tools that integrate
with physics engines for enhanced computational capabilities.

Shows the integration between PR #23 (physics tools) and PR #18 (physics engines).
"""

import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')

def demo_engine_integration():
    """Demonstrate physics tools with engine integration."""
    
    print("=" * 70)
    print("PHYSICS TOOLS WITH ENGINE INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    try:
        # Import physics tools
        from tools.physics import (
            get_physics_engine_adapter,
            create_quantum_chemistry_tool,
            PhysicsToolRegistry,
            validate_integration,
            show_physics_engines_info
        )
        
        print("✓ Physics tools package imported successfully")
        
        # 1. Show engine integration status
        print("\n1. Physics Engine Integration Status")
        print("-" * 40)
        show_physics_engines_info()
        
        # 2. Validate integration
        print("\n2. Integration Validation")
        print("-" * 40)
        validation = validate_integration()
        print(f"Integration Status: {validation['integration_status']}")
        print(f"Tools Available: {len(validation['tools_available'])}")
        print(f"Engines Available: {validation['engines_available']}")
        
        if validation['issues']:
            print("Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['recommendations']:
            print("Recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        # 3. Create quantum chemistry tool with engine integration
        print("\n3. Quantum Chemistry Tool with Engine Integration")
        print("-" * 40)
        qc_tool = create_quantum_chemistry_tool(prefer_engines=True)
        print(f"✓ Created quantum chemistry tool: {qc_tool.name}")
        print(f"  Physics Domain: {qc_tool.physics_domain}")
        print(f"  Prefers Engines: {qc_tool.prefer_engines}")
        print(f"  Engine Capabilities: {qc_tool.engine_capabilities}")
        
        # 4. Test engine adapter
        print("\n4. Engine Adapter Testing")
        print("-" * 40)
        adapter = get_physics_engine_adapter()
        adapter_summary = adapter.get_available_engines_summary()
        print(f"  Adapter Initialized: {adapter_summary['adapter_initialized']}")
        print(f"  Total Engines: {adapter_summary['total_engines']}")
        print(f"  Supported Domains: {adapter_summary['supported_domains']}")
        
        # Test engine availability check
        engine_available = adapter.is_engine_available("quantum_chemistry", "energy_calculation")
        print(f"  Engine available for QC energy calculation: {engine_available}")
        
        # 5. Create physics tool registry
        print("\n5. Physics Tool Registry with Engine Integration")
        print("-" * 40)
        registry = PhysicsToolRegistry(auto_register_default=True)
        
        registry_stats = registry.get_registry_statistics()
        print(f"✓ Registry created with {registry_stats['total_physics_tools']} tools")
        print(f"  Physics domains: {', '.join(registry_stats['physics_domains'])}")
        
        # Get engine integration summary
        engine_integration = registry.get_engine_integration_summary()
        print(f"  Tools with engine support: {engine_integration['tools_with_engine_support']}")
        print(f"  Engine integration ratio: {engine_integration['engine_integration_ratio']:.2%}")
        
        # 6. Tool discovery with engine preferences
        print("\n6. Tool Discovery with Engine Preferences")
        print("-" * 40)
        research_question = "Calculate the electronic structure and binding energy of a water molecule"
        
        recommendations = registry.discover_physics_tools(
            agent_id="demo_agent",
            research_question=research_question,
            prefer_engines=True
        )
        
        print(f"✓ Found {len(recommendations)} tool recommendations for:")
        print(f"  '{research_question}'")
        
        for i, rec in enumerate(recommendations[:2], 1):
            print(f"\n  {i}. {rec['name']} (confidence: {rec['confidence']:.3f})")
            print(f"     Domain: {rec['physics_domain']}")
            print(f"     Engines available: {rec['engine_status']['engines_available']}")
            if rec['engine_status']['engines_available']:
                print(f"     Engine type: {rec['engine_status']['engine_type']}")
            print(f"     Average cost: {rec['average_cost']:.1f} units")
        
        # 7. Demonstrate quantum chemistry calculation
        print("\n7. Quantum Chemistry Calculation Demonstration")
        print("-" * 40)
        
        # Request quantum chemistry tool
        qc_tool_from_registry = registry.request_physics_tool(
            agent_id="demo_agent",
            tool_id="quantum_chemistry_tool",
            task_specification={
                "type": "energy_calculation",
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
            context={
                "agent_id": "demo_agent",
                "available_memory": 4096,
                "available_cpu_cores": 4
            }
        )
        
        if qc_tool_from_registry:
            print("✓ Agent granted access to quantum chemistry tool")
            
            # Perform calculation
            task = {
                "type": "energy_calculation",
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
            
            context = {
                "agent_id": "demo_agent",
                "available_memory": 4096,
                "available_cpu_cores": 4
            }
            
            print("  Executing water molecule energy calculation...")
            result = qc_tool_from_registry.execute(task, context)
            
            if result["success"]:
                print(f"  ✓ Calculation completed in {result['calculation_time']:.3f} seconds")
                print(f"  ✓ Execution method: {result['execution_method']}")
                print(f"  ✓ Method used: {result['method_used']}")
                print(f"  ✓ Computational cost: {result['computational_cost']:.1f} units")
                print(f"  ✓ Result confidence: {result['confidence']:.3f}")
                
                # Show key results
                results = result["results"]
                if "energies" in results:
                    energy = results["energies"].get("total_energy", {})
                    if energy:
                        print(f"  ✓ Total energy: {energy['value']:.6f} {energy['units']}")
                
                if "execution_info" in results:
                    exec_info = results["execution_info"]
                    print(f"  ✓ Engine used: {exec_info['engine_used']}")
                    print(f"  ✓ Convergence: {exec_info['convergence_achieved']}")
            else:
                print(f"  ✗ Calculation failed: {result.get('error', 'Unknown error')}")
        
        # 8. Workflow recommendation with engine optimization
        print("\n8. Physics Research Workflow with Engine Optimization")
        print("-" * 40)
        
        workflow_goal = "Comprehensive molecular analysis with electronic structure calculation and visualization"
        workflow = registry.recommend_physics_workflow(
            research_goal=workflow_goal,
            available_data={"molecular_structure": True},
            constraints={"max_cost": 500},
            prefer_engines=True
        )
        
        print(f"✓ Generated workflow for: '{workflow_goal}'")
        print(f"  Workflow ID: {workflow['workflow_id']}")
        print(f"  Total steps: {workflow['engine_integration']['total_steps']}")
        print(f"  Engine-enhanced steps: {workflow['engine_integration']['engine_enhanced_steps']}")
        print(f"  Engine enhancement ratio: {workflow['engine_integration']['engine_enhancement_ratio']:.2%}")
        print(f"  Estimated time: {workflow['total_estimated_time']}")
        
        print("\n  Workflow Steps:")
        for i, step in enumerate(workflow['workflow_steps'], 1):
            engine_indicator = "🚀" if step.get("engine_enhanced", False) else "⚙️"
            print(f"    {i}. {engine_indicator} {step['phase'].title()}: {step['purpose']}")
            print(f"       Tool: {step['tool']}")
            print(f"       Time: {step['estimated_time']}")
        
        # 9. Performance statistics
        print("\n9. Tool Performance Statistics with Engine Integration")
        print("-" * 40)
        
        qc_stats = registry.get_physics_tool_performance_stats("quantum_chemistry_tool")
        if qc_stats:
            print(f"  Tool: {qc_stats['physics_domain']}")
            print(f"  Total calculations: {qc_stats['total_calculations']}")
            print(f"  Success rate: {qc_stats['success_rate']:.2%}")
            print(f"  Average time: {qc_stats['average_calculation_time']:.3f} seconds")
            
            engine_info = qc_stats.get('engine_integration', {})
            print(f"  Engine calculations: {engine_info.get('engine_calculations', 0)}")
            print(f"  Fallback calculations: {engine_info.get('fallback_calculations', 0)}")
            print(f"  Engine usage ratio: {engine_info.get('engine_usage_ratio', 0):.2%}")
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)
        print()
        print("Key Features Demonstrated:")
        print("• Physics tools with physics engine integration")
        print("• Automatic fallback when engines are not available")
        print("• Enhanced tool discovery with engine preferences")
        print("• Engine-aware workflow generation")
        print("• Performance tracking for both engine and fallback calculations")
        print("• Seamless integration between agent tools and computational engines")
        print()
        print("Integration Benefits:")
        print("• Higher accuracy calculations when engines are available")
        print("• Graceful degradation to fallback implementations")
        print("• Enhanced performance and advanced algorithms")
        print("• Unified interface for agents regardless of backend")
        print("• Cost estimation and resource management")
        print()
        
        if not validation['engines_available']:
            print("Note: Physics engines from PR #18 are not currently available.")
            print("All calculations are using fallback implementations.")
            print("Install the physics engine package to see enhanced capabilities.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the physics tools package is properly installed.")
        return False
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = demo_engine_integration()
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo encountered errors.")
        sys.exit(1)