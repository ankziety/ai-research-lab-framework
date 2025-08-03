# Physics-Specific Agents for AI Research Lab Framework

This directory contains specialized physics agents that extend the AI Research Lab Framework with domain-specific expertise across all major physics disciplines.

## Overview

The physics agents are inspired by the Virtual Lab methodology from "The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies" by Swanson et al., adapted for physics research. These agents provide sophisticated physics expertise from quantum mechanics to cosmology.

## Agent Hierarchy

```
BasePhysicsAgent (Abstract)
├── QuantumPhysicsAgent
├── ComputationalPhysicsAgent  
├── ExperimentalPhysicsAgent
├── MaterialsPhysicsAgent
└── AstrophysicsAgent
```

## Specialized Agents

### 1. QuantumPhysicsAgent
**Domain**: Quantum mechanics and quantum computing
**Expertise**: 
- Schrödinger equation solving (hydrogen atom, harmonic oscillator, particle in box)
- Quantum circuit design (Grover, Shor, VQE, QAOA algorithms)
- Quantum entanglement analysis with Schmidt decomposition
- Quantum fidelity calculations and similarity measures
- Quantum state manipulation and measurement

**Key Methods**:
- `solve_schrodinger_equation(system_type, parameters)`
- `analyze_quantum_entanglement(quantum_state, system_description)`  
- `design_quantum_circuit(algorithm, parameters)`
- `calculate_quantum_fidelity(state1, state2, metric_type)`

### 2. ComputationalPhysicsAgent
**Domain**: Numerical methods and physics simulations
**Expertise**:
- Molecular dynamics simulations (Verlet, Leap-frog, Velocity-Verlet)
- PDE solving (heat, wave, Poisson equations)
- Monte Carlo methods (Metropolis, importance sampling, Gibbs)
- Optimization algorithms and model fitting
- Advanced numerical analysis

**Key Methods**:
- `run_molecular_dynamics_simulation(system_config)`
- `solve_pde_system(pde_config)`
- `run_monte_carlo_simulation(mc_config)`
- `optimize_physical_system(optimization_config)`
- `fit_physics_model(data, model_config)`

### 3. ExperimentalPhysicsAgent  
**Domain**: Experimental design and data analysis
**Expertise**:
- Experimental design methodologies (factorial, response surface, Taguchi)
- Statistical analysis and hypothesis testing
- Uncertainty analysis and error propagation
- Data validation and quality control
- Measurement techniques and instrumentation

**Key Methods**:
- `design_physics_experiment(hypothesis, constraints)`
- `analyze_experimental_data(data, analysis_config)`
- `perform_uncertainty_analysis(measurements, error_sources)`
- `validate_experimental_results(results, validation_config)`

### 4. MaterialsPhysicsAgent
**Domain**: Materials science and condensed matter physics  
**Expertise**:
- Crystal structure analysis (7 crystal systems, space groups)
- Electronic properties calculation (band gaps, DOS)
- Mechanical properties analysis (elastic constants, moduli)
- Phase diagram prediction and stability analysis
- Materials characterization experiment design

**Key Methods**:
- `analyze_crystal_structure(structure_data)`
- `calculate_electronic_properties(material_config)`
- `analyze_mechanical_properties(material_data)`
- `predict_phase_diagram(system_config)`
- `design_characterization_experiment(material_info, research_goals)`

### 5. AstrophysicsAgent
**Domain**: Astrophysics and cosmology
**Expertise**:
- Stellar evolution modeling and nucleosynthesis
- Galactic dynamics and dark matter analysis  
- Cosmological models (ΛCDM, distance measures)
- Observational astronomy planning
- High-energy astrophysics phenomena

**Key Methods**:
- `analyze_stellar_evolution(stellar_config)`
- `model_galactic_dynamics(galaxy_config)`
- `analyze_cosmological_model(cosmology_config)`
- `design_observational_strategy(research_target, constraints)`

## Physics Agent Registry

The `PhysicsAgentRegistry` provides centralized management of physics agents:

```python
from agents.physics import PhysicsAgentRegistry, PhysicsAgentType

registry = PhysicsAgentRegistry()

# Create agents
quantum_agent = registry.create_physics_agent(PhysicsAgentType.QUANTUM_PHYSICS, "q1")

# Get recommendations  
recommendations = registry.recommend_physics_agents_for_research(
    "quantum effects in superconducting materials", max_agents=3
)

# Create research teams
team = registry.create_physics_agent_team(
    "design quantum sensors for dark matter detection", team_size=4
)
```

## Factory Functions

Convenient factory functions for agent creation:

```python
from agents.physics import create_physics_agent, get_physics_domain_for_query

# Automatic domain detection
domain = get_physics_domain_for_query("How do quantum computers work?")
# Returns: "quantum_physics"

# Create agent
agent = create_physics_agent(domain, "agent_id")
```

## Cross-Scale Physics Coverage

The agents cover physics phenomena across all relevant scales:

- **Quantum Scale** (10⁻¹⁵ to 10⁻¹⁰ m): Quantum mechanics, particle physics
- **Atomic Scale** (10⁻¹⁰ to 10⁻⁹ m): Atomic structure, chemical bonding  
- **Nano Scale** (10⁻⁹ to 10⁻⁶ m): Nanostructures, quantum dots
- **Micro Scale** (10⁻⁶ to 10⁻³ m): Microdevices, MEMS
- **Macro Scale** (10⁻³ to 10³ m): Laboratory experiments, engineering
- **Planetary Scale** (10⁶ to 10⁷ m): Planetary physics, geophysics
- **Stellar Scale** (10⁸ to 10¹² m): Stars, stellar systems
- **Galactic Scale** (10¹⁶ to 10²¹ m): Galaxies, galaxy clusters
- **Cosmic Scale** (10²² to 10²⁶ m): Universe, cosmology

## Advanced Features

### Multi-Methodology Integration
- **Theoretical**: Mathematical modeling, analytical solutions
- **Computational**: Numerical simulations, algorithms  
- **Experimental**: Lab experiments, measurements
- **Observational**: Astronomical observations, data analysis

### Physics-Specific Tool Discovery
Each agent can discover and optimize tools specific to their domain:

```python
tools = agent.discover_available_tools("quantum circuit optimization")
optimized = agent.optimize_tool_usage("research question", tools)
```

### Comprehensive Error Handling
All methods include robust error handling with detailed logging and fallback strategies.

### Serialization Support
Full state serialization for persistence and distributed computing:

```python
agent_state = agent.to_dict()
# Save, transmit, or restore agent state
```

## Usage Examples

### Quick Start
```python
from agents.physics import QuantumPhysicsAgent

# Create quantum physics agent
agent = QuantumPhysicsAgent("quantum_expert", "Quantum Physics Specialist")

# Solve hydrogen atom
result = agent.solve_schrodinger_equation("hydrogen_atom", {"atomic_number": 1})
print(f"Ground state energy: {result['energy_levels'][0]['energy_eV']} eV")

# Design quantum circuit
circuit = agent.design_quantum_circuit("grover", {"n_items": 16})
print(f"Circuit uses {circuit['qubit_count']} qubits")
```

### Research Team Creation
```python
from agents.physics import PhysicsAgentRegistry

registry = PhysicsAgentRegistry()
team = registry.create_physics_agent_team(
    "investigate quantum effects in dark matter detection", 
    team_size=3
)

# Team automatically includes relevant physics domains
for agent in team:
    print(f"{agent.physics_domain}: {agent.role}")
```

## Integration with Existing Framework

The physics agents seamlessly integrate with the existing AI Research Lab Framework:

- **Inheritance**: All agents inherit from `BaseAgent`
- **Compatibility**: Full compatibility with existing agent marketplace
- **Separation**: Independent registry prevents conflicts
- **Extension**: Adds physics capabilities without modifying existing code

## File Structure

```
agents/physics/
├── __init__.py                     # Factory functions and exports
├── base_physics_agent.py           # Abstract base class (27KB)
├── physics_agent_registry.py       # Agent registry system (21KB)  
├── quantum_physics_agent.py        # Quantum physics specialist (30KB)
├── computational_physics_agent.py  # Computational physics (42KB)
├── experimental_physics_agent.py   # Experimental physics (56KB)
├── materials_physics_agent.py      # Materials science (52KB)
└── astrophysics_agent.py          # Astrophysics & cosmology (50KB)
```

## Performance and Scalability

- **Efficient Algorithms**: Optimized numerical methods and algorithms
- **Caching**: Intelligent caching of expensive calculations  
- **Memory Management**: Careful memory usage for large simulations
- **Parallel Processing**: Support for parallel and distributed computing
- **Cost Tracking**: Integration with framework cost management

## Future Extensions

The modular design enables easy extension with additional physics domains:

- **Nuclear Physics**: Nuclear reactions, decay processes
- **Plasma Physics**: Plasma dynamics, fusion physics  
- **Biophysics**: Biological systems, protein folding
- **Geophysics**: Earth sciences, seismology
- **Medical Physics**: Medical imaging, radiation therapy

## Quality Assurance

- **Type Hints**: Complete type annotations for all methods
- **Documentation**: Comprehensive docstrings and examples
- **Error Handling**: Robust error handling and logging
- **Testing**: Comprehensive test suite with validation
- **Performance**: Optimized for speed and memory efficiency

This physics agent system represents a significant advancement in AI-powered scientific research, providing domain experts with sophisticated tools for investigation across all major physics disciplines.