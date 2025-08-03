#!/usr/bin/env python3
"""
Physics Agents + Tools + Engines Integration Demo

This script demonstrates the complete integration between physics agents,
physics tools, and physics engines, showing how they work together to provide
enhanced computational capabilities for AI-powered physics research.
"""

import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')

def demo_agent_tools_engines_integration():
    """Demonstrate the complete physics agents, tools, and engines integration."""
    
    print("=" * 80)
    print("PHYSICS AGENTS + TOOLS + ENGINES INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        # Import physics agents
        from agents.physics import (
            QuantumPhysicsAgent, ComputationalPhysicsAgent, MaterialsPhysicsAgent,
            PhysicsAgentRegistry
        )
        
        print("✓ Physics agents package imported successfully")
        
        # Import physics tools
        from tools.physics import (
            PhysicsToolRegistry,
            validate_integration,
            show_physics_engines_info
        )
        
        print("✓ Physics tools package imported successfully")
        
        # Show engine integration status
        print("\n1. Physics Engine Integration Status")
        print("-" * 50)
        show_physics_engines_info()
        
        # Validate tools and engines integration
        print("\n2. Tools and Engines Integration Validation")
        print("-" * 50)
        validation = validate_integration()
        print(f"Integration Status: {validation['integration_status']}")
        print(f"Tools Available: {len(validation['tools_available'])}")
        print(f"Engines Available: {validation['engines_available']}")
        
        # Create physics agents with tools integration
        print("\n3. Creating Physics Agents with Tools Integration")
        print("-" * 50)
        quantum_agent = QuantumPhysicsAgent("quantum_agent_demo", "Quantum Physics Expert")
        comp_agent = ComputationalPhysicsAgent("comp_agent_demo", "Computational Physics Expert")
        materials_agent = MaterialsPhysicsAgent("materials_agent_demo", "Materials Physics Expert")
        
        agents = [quantum_agent, comp_agent, materials_agent]
        
        for agent in agents:
            status = agent.get_physics_integration_status()
            print(f"Agent: {agent.agent_id}")
            print(f"  - Physics domain: {status['physics_domain']}")
            print(f"  - Tools available: {status['physics_tools_available']}")
            print(f"  - Enhanced tools: {status['total_enhanced_tools']}")
            print(f"  - Integration health: {status['integration_health']}")
        
        # Demonstrate quantum physics with engine enhancement
        print("\n4. Quantum Physics Calculations with Engine Enhancement")
        print("-" * 50)
        
        print("Solving Schrödinger equation for hydrogen atom...")
        hydrogen_params = {"atomic_number": 1, "max_n": 3}
        result = quantum_agent.solve_schrodinger_equation("hydrogen_atom", hydrogen_params)
        
        if result['success']:
            print(f"✓ Calculation successful!")
            print(f"  - Engine enhanced: {result.get('engine_enhanced', False)}")
            print(f"  - Method: {result.get('computational_method', 'unknown')}")
            print(f"  - Energy levels found: {len(result.get('energy_levels', []))}")
            if result.get('energy_levels'):
                for i, level in enumerate(result['energy_levels'][:3]):
                    if isinstance(level, dict) and 'energy_eV' in level:
                        print(f"    n={level.get('n', i+1)}: E = {level['energy_eV']:.3f} eV")
        else:
            print(f"⚠ Calculation failed: {result.get('error', 'Unknown error')}")
            if result.get('fallback_used'):
                print(f"  Suggestion: {result.get('suggestion', '')}")
        
        # Demonstrate computational physics with engine enhancement
        print("\n5. Molecular Dynamics Simulation with Engine Enhancement")
        print("-" * 50)
        
        print("Running molecular dynamics simulation...")
        md_config = {
            'n_particles': 50,
            'n_steps': 500,
            'time_step': 0.001,
            'potential': 'lennard_jones',
            'temperature': 300
        }
        
        result = comp_agent.run_molecular_dynamics_simulation(md_config)
        
        if result['success']:
            print(f"✓ Simulation successful!")
            print(f"  - Engine enhanced: {result.get('engine_enhanced', False)}")
            print(f"  - Method: {result.get('computational_method', 'unknown')}")
            print(f"  - Particles: {result.get('n_particles', 0)}")
            print(f"  - Steps: {result.get('n_steps', 0)}")
            
            energies = result.get('energies', {})
            if energies.get('total'):
                final_energy = energies['total'][-1] if energies['total'] else 0
                print(f"  - Final total energy: {final_energy:.2e} J")
        else:
            print(f"⚠ Simulation failed: {result.get('error', 'Unknown error')}")
        
        # Demonstrate agent tool discovery
        print("\n6. Agent Tool Discovery and Optimization")
        print("-" * 50)
        
        research_question = "Design quantum sensors for dark matter detection using superconducting qubits"
        print(f"Research question: {research_question}")
        
        tools = quantum_agent.discover_available_tools(research_question)
        physics_tools = [tool for tool in tools if tool.get('physics_specific', False)]
        
        print(f"✓ Discovered {len(tools)} total tools, {len(physics_tools)} physics-specific")
        
        if physics_tools:
            print("Top physics tools:")
            for tool in physics_tools[:3]:
                enhanced = "✓" if tool.get('engine_enhanced', False) else "○"
                print(f"  {enhanced} {tool['name']}: confidence {tool['confidence']:.3f}")
        
        # Show agent performance metrics
        print("\n7. Agent Performance Metrics")
        print("-" * 50)
        
        for agent in agents:
            status = agent.get_physics_integration_status()
            metrics = status['physics_metrics']
            print(f"{agent.agent_id}:")
            print(f"  - Tools used: {metrics.get('tools_used', 0)}")
            print(f"  - Engine enhanced calculations: {metrics.get('engine_enhanced_calculations', 0)}")
            print(f"  - Equations solved: {metrics.get('equations_solved', 0)}")
            print(f"  - Simulations run: {metrics.get('simulations_run', 0)}")
        
        # Demonstrate physics agent registry with tool integration
        print("\n8. Physics Agent Registry with Tool Integration")
        print("-" * 50)
        
        registry = PhysicsAgentRegistry()
        
        # Get recommendations that consider tool availability
        recommendations = registry.recommend_physics_agents_for_research(
            "Quantum computational chemistry with molecular dynamics", 
            max_agents=2
        )
        
        print(f"Registry recommendations for combined quantum/computational research:")
        for rec in recommendations:
            print(f"  - {rec['agent_type']}: relevance {rec['relevance_score']:.3f}")
        
        # Create research team with tool integration
        team = registry.create_physics_agent_team(
            "Multi-scale quantum materials modeling with experimental validation",
            team_size=3
        )
        
        print(f"\nCreated research team of {len(team)} agents:")
        for agent in team:
            status = agent.get_physics_integration_status()
            print(f"  - {agent.physics_domain}: {status['total_enhanced_tools']} enhanced tools")
        
        print("\n" + "=" * 80)
        print("✅ COMPLETE PHYSICS INTEGRATION DEMONSTRATION SUCCESSFUL!")
        print("=" * 80)
        print("\nThe physics agents now seamlessly integrate with physics tools and engines,")
        print("providing enhanced computational capabilities while maintaining reliable fallback.")
        print("This creates a powerful AI research platform with professional-grade physics.")
        
    except Exception as e:
        print(f"\n❌ Error during integration demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def show_integration_summary():
    """Show a summary of the complete integration."""
    
    print("\n" + "=" * 80)
    print("PHYSICS INTEGRATION ARCHITECTURE SUMMARY")
    print("=" * 80)
    print()
    
    architecture = """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        AI RESEARCH LAB FRAMEWORK                        │
    │                     Complete Physics Integration                        │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Physics Agents │    │  Physics Tools  │    │ Physics Engines │
    │                 │    │                 │    │                 │
    │ • Quantum       │◄──►│ • Engine Adapter│◄──►│ • High-perf     │
    │ • Computational │    │ • QC Tool       │    │   engines       │
    │ • Materials     │    │ • Base Tool     │    │ • Fallback      │
    │ • Experimental  │    │ • Registry      │    │   implementations│
    │ • Astrophysics  │    │ • Smart Discovery│    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
    
    🔗 INTEGRATION FEATURES:
    • Agents automatically discover and use enhanced physics tools
    • Tools leverage engines when available, graceful fallback otherwise
    • Unified interface regardless of computational backend
    • Enhanced accuracy and performance with engine integration
    • Professional-grade physics calculations for AI research
    """
    
    print(architecture)


def main():
    """Run the complete integration demonstration."""
    print("PHYSICS AGENTS + TOOLS + ENGINES INTEGRATION")
    print(f"Demonstration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = demo_agent_tools_engines_integration()
    
    if success:
        show_integration_summary()
    
    print(f"\nDemonstration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()