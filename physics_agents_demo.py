"""
Physics Agents Usage Examples

This file demonstrates how to use the physics-specific agents for various
research scenarios, showcasing the capabilities of each specialized agent.
"""

from agents.physics import (
    QuantumPhysicsAgent, ComputationalPhysicsAgent, ExperimentalPhysicsAgent,
    MaterialsPhysicsAgent, AstrophysicsAgent, PhysicsAgentRegistry,
    create_physics_agent, get_physics_domain_for_query
)
from agents.physics.physics_agent_registry import PhysicsAgentType


def demonstrate_quantum_physics():
    """Demonstrate quantum physics agent capabilities."""
    print("=" * 60)
    print("QUANTUM PHYSICS AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = QuantumPhysicsAgent("quantum_demo", "Quantum Physics Expert")
    
    # Solve Schrödinger equation for hydrogen atom
    print("\n1. Solving Schrödinger equation for hydrogen atom:")
    result = agent.solve_schrodinger_equation("hydrogen_atom", {"atomic_number": 1, "max_n": 3})
    if result['success']:
        print(f"   ✓ Successfully calculated {len(result['energy_levels'])} energy levels")
        for level in result['energy_levels'][:3]:
            print(f"   n={level['n']}: E = {level['energy_eV']:.2f} eV")
    
    # Design quantum circuit
    print("\n2. Designing Grover's search algorithm circuit:")
    circuit = agent.design_quantum_circuit("grover", {"n_items": 16})
    if circuit['success']:
        print(f"   ✓ Circuit designed with {circuit['qubit_count']} qubits")
        print(f"   Expected success probability: {circuit['expected_success_probability']:.3f}")
    
    print(f"\n3. Agent expertise: {', '.join(agent.expertise[:5])}...")


def demonstrate_computational_physics():
    """Demonstrate computational physics agent capabilities."""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL PHYSICS AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = ComputationalPhysicsAgent("comp_demo", "Computational Physics Expert")
    
    # Run molecular dynamics simulation
    print("\n1. Running molecular dynamics simulation:")
    md_config = {
        'n_particles': 50,
        'n_steps': 500,
        'time_step': 0.001,
        'potential': 'lennard_jones',
        'temperature': 300
    }
    result = agent.run_molecular_dynamics_simulation(md_config)
    if result['success']:
        print(f"   ✓ Simulated {result['n_particles']} particles for {result['n_steps']} steps")
        print(f"   Final total energy: {result['energies']['total'][-1]:.2e} J")
    
    # Solve PDE
    print("\n2. Solving heat equation:")
    pde_config = {
        'equation_type': 'heat',
        'nx': 50,
        'nt': 100,
        'thermal_diffusivity': 1.0
    }
    result = agent.solve_pde_system(pde_config)
    if result['success']:
        print(f"   ✓ Solved heat equation on {pde_config['nx']} grid points")
        print(f"   Time evolution calculated for {pde_config['nt']} steps")


def demonstrate_experimental_physics():
    """Demonstrate experimental physics agent capabilities."""
    print("\n" + "=" * 60)
    print("EXPERIMENTAL PHYSICS AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = ExperimentalPhysicsAgent("exp_demo", "Experimental Physics Expert")
    
    # Design experiment
    print("\n1. Designing superconductivity experiment:")
    hypothesis = "The ceramic material exhibits superconductivity below 77K"
    constraints = {"budget": 25000, "time_limit": 60}
    result = agent.design_physics_experiment(hypothesis, constraints)
    if result['success']:
        print(f"   ✓ Experimental design completed")
        print(f"   Protocols designed: {len(result['protocols'])}")
        print(f"   Estimated total cost: ${result['resource_estimate']['cost_estimate']['total_cost']:.0f}")
    
    # Analyze experimental data
    print("\n2. Analyzing experimental data:")
    data = {
        'temperature': [300, 250, 200, 150, 100, 77, 50],
        'resistance': [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]
    }
    analysis_config = {'test_type': 't_test', 'significance_level': 0.05}
    result = agent.analyze_experimental_data(data, analysis_config)
    if result['success']:
        print(f"   ✓ Data analysis completed")
        print(f"   Statistical tests performed: {len(result['hypothesis_testing'])}")


def demonstrate_materials_physics():
    """Demonstrate materials physics agent capabilities."""
    print("\n" + "=" * 60)
    print("MATERIALS PHYSICS AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = MaterialsPhysicsAgent("mat_demo", "Materials Physics Expert")
    
    # Analyze crystal structure
    print("\n1. Analyzing silicon crystal structure:")
    structure_data = {
        'lattice_parameters': {'a': 5.43, 'b': 5.43, 'c': 5.43},
        'atomic_positions': [
            {'element': 'Si', 'x': 0.0, 'y': 0.0, 'z': 0.0},
            {'element': 'Si', 'x': 0.25, 'y': 0.25, 'z': 0.25}
        ]
    }
    result = agent.analyze_crystal_structure(structure_data)
    if result['success']:
        print(f"   ✓ Crystal system identified: {result['crystal_system']}")
        print(f"   Unit cell volume: {result['unit_cell_volume']:.2f} Ų")
        print(f"   Density: {result['density']:.2e} kg/m³")
    
    # Calculate electronic properties
    print("\n2. Calculating electronic properties:")
    material_config = {'type': 'semiconductor', 'elements': ['Si']}
    result = agent.calculate_electronic_properties(material_config)
    if result['success']:
        print(f"   ✓ Band gap calculated: {result['band_gap']:.2f} eV")
        print(f"   Conductivity type: {result['conductivity_type']}")


def demonstrate_astrophysics():
    """Demonstrate astrophysics agent capabilities."""
    print("\n" + "=" * 60)
    print("ASTROPHYSICS AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = AstrophysicsAgent("astro_demo", "Astrophysics Expert")
    
    # Analyze stellar evolution
    print("\n1. Analyzing stellar evolution (Sun-like star):")
    stellar_config = {'mass': 1.0, 'metallicity': 0.02, 'age': 4.6e9}
    result = agent.analyze_stellar_evolution(stellar_config)
    if result['success']:
        print(f"   ✓ Current phase: {result['current_state']['phase']}")
        print(f"   Main sequence lifetime: {result['lifetime_estimates']['main_sequence']:.2e} years")
        print(f"   Final fate: {result['final_fate']}")
    
    # Model galactic dynamics
    print("\n2. Modeling Milky Way galaxy dynamics:")
    galaxy_config = {
        'type': 'spiral',
        'total_mass': 1e12,
        'scale_radius': 3.0
    }
    result = agent.model_galactic_dynamics(galaxy_config)
    if result['success']:
        print(f"   ✓ Galaxy type: {result['galaxy_type']}")
        print(f"   Maximum rotation velocity: {result['rotation_curve']['maximum_velocity']:.0f} km/s")
        print(f"   Dark matter fraction: {result['dark_matter_profile']['dark_matter_fraction']:.2f}")
    
    # Analyze cosmological model
    print("\n3. Analyzing ΛCDM cosmological model:")
    cosmology_config = {
        'hubble_constant': 70.0,
        'omega_matter': 0.31,
        'omega_lambda': 0.69
    }
    result = agent.analyze_cosmological_model(cosmology_config)
    if result['success']:
        print(f"   ✓ Age of universe: {result['age_of_universe']:.2e} years")
        print(f"   Critical density: {result['critical_density']:.2e} kg/m³")


def demonstrate_physics_registry():
    """Demonstrate physics agent registry capabilities."""
    print("\n" + "=" * 60)
    print("PHYSICS AGENT REGISTRY DEMONSTRATION")
    print("=" * 60)
    
    registry = PhysicsAgentRegistry()
    
    # Create agents through registry
    print("\n1. Creating physics agents through registry:")
    quantum_agent = registry.create_physics_agent(PhysicsAgentType.QUANTUM_PHYSICS, "reg_quantum")
    astro_agent = registry.create_physics_agent(PhysicsAgentType.ASTROPHYSICS, "reg_astro")
    print(f"   ✓ Created quantum agent: {quantum_agent.agent_id}")
    print(f"   ✓ Created astrophysics agent: {astro_agent.agent_id}")
    
    # Get recommendations for research
    print("\n2. Getting agent recommendations for research:")
    research_question = "Design a quantum sensor for detecting dark matter"
    recommendations = registry.recommend_physics_agents_for_research(research_question, max_agents=3)
    print(f"   ✓ Found {len(recommendations)} suitable agents:")
    for rec in recommendations:
        print(f"   - {rec['agent_type']}: relevance {rec['relevance_score']:.2f}")
    
    # Create research team
    print("\n3. Creating specialized research team:")
    team = registry.create_physics_agent_team(
        "Investigate quantum effects in cosmological dark energy",
        team_size=2
    )
    print(f"   ✓ Created team of {len(team)} agents:")
    for agent in team:
        print(f"   - {agent.physics_domain}: {agent.agent_id}")
    
    # Get registry statistics
    print("\n4. Registry statistics:")
    stats = registry.get_physics_agent_statistics()
    print(f"   ✓ Total physics agents: {stats['total_physics_agents']}")
    print(f"   ✓ Available agents: {stats['available_physics_agents']}")
    print(f"   ✓ Utilization rate: {stats['utilization_rate']:.2f}")


def demonstrate_factory_functions():
    """Demonstrate factory functions and utilities."""
    print("\n" + "=" * 60)
    print("FACTORY FUNCTIONS AND UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Automatic domain detection
    print("\n1. Automatic physics domain detection:")
    queries = [
        "How do quantum entangled states behave under decoherence?",
        "What is the stellar nucleosynthesis process in massive stars?",
        "How can we simulate molecular dynamics of protein folding?",
        "What are the electronic properties of graphene?",
        "Design an experiment to measure the Higgs boson mass"
    ]
    
    for query in queries:
        domain = get_physics_domain_for_query(query)
        print(f"   '{query[:50]}...' → {domain}")
    
    # Create agents using factory function
    print("\n2. Creating agents using factory function:")
    agent = create_physics_agent("quantum_physics", "factory_quantum")
    print(f"   ✓ Created {agent.physics_domain} agent: {agent.agent_id}")
    
    # Demonstrate tool discovery
    print("\n3. Physics-specific tool discovery:")
    tools = agent.discover_available_tools("quantum circuit optimization for NISQ devices")
    physics_tools = [tool for tool in tools if tool.get('physics_specific', False)]
    print(f"   ✓ Found {len(physics_tools)} physics-specific tools:")
    for tool in physics_tools[:3]:
        print(f"   - {tool['name']}: confidence {tool['confidence']:.2f}")


def main():
    """Run all demonstrations."""
    print("PHYSICS AGENTS COMPREHENSIVE DEMONSTRATION")
    print("AI Research Lab Framework - Physics Extensions")
    print("=" * 80)
    
    try:
        demonstrate_quantum_physics()
        demonstrate_computational_physics()
        demonstrate_experimental_physics()
        demonstrate_materials_physics()
        demonstrate_astrophysics()
        demonstrate_physics_registry()
        demonstrate_factory_functions()
        
        print("\n" + "=" * 80)
        print("✅ ALL PHYSICS AGENTS DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nThe physics agents are ready for advanced research applications")
        print("covering the full spectrum from quantum mechanics to cosmology.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()