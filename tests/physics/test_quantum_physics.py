"""
Quantum Physics Tests

Comprehensive unit tests for quantum physics algorithms and simulations.
Tests quantum mechanical calculations, state evolution, and quantum algorithms.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, patch

# Quantum physics test markers
pytestmark = pytest.mark.quantum


class MockQuantumPhysicsAgent:
    """Mock quantum physics agent for testing."""
    
    def __init__(self):
        self.name = "QuantumPhysicsAgent"
        self.initialized = True
        self.quantum_systems = {}
    
    def solve_schrodinger_equation(self, hamiltonian: np.ndarray, 
                                 initial_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Mock Schrodinger equation solver."""
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'ground_state_energy': eigenvalues[0],
            'ground_state_wavefunction': eigenvectors[:, 0]
        }
    
    def calculate_ground_state(self, hamiltonian: np.ndarray) -> Dict[str, Any]:
        """Calculate ground state of quantum system."""
        result = self.solve_schrodinger_equation(hamiltonian)
        return {
            'energy': result['ground_state_energy'],
            'wavefunction': result['ground_state_wavefunction'],
            'probability_density': np.abs(result['ground_state_wavefunction'])**2
        }
    
    def evolve_quantum_state(self, initial_state: np.ndarray, 
                           hamiltonian: np.ndarray, time: float) -> np.ndarray:
        """Evolve quantum state in time using time evolution operator."""
        # U(t) = exp(-i * H * t / ħ), assuming ħ = 1
        time_evolution_operator = self._matrix_exponential(-1j * hamiltonian * time)
        return time_evolution_operator @ initial_state
    
    def calculate_expectation_value(self, state: np.ndarray, 
                                  operator: np.ndarray) -> complex:
        """Calculate expectation value of an operator."""
        return np.conj(state) @ operator @ state
    
    def calculate_entanglement_entropy(self, state: np.ndarray, 
                                     subsystem_size: int) -> float:
        """Calculate entanglement entropy of a bipartite system."""
        # For testing purposes, return a mock entropy value
        total_size = len(state)
        if subsystem_size >= total_size:
            return 0.0
        
        # Mock calculation - in reality would require partial trace
        return np.log(min(subsystem_size, total_size - subsystem_size))
    
    def apply_quantum_gate(self, state: np.ndarray, gate: np.ndarray, 
                         qubit_indices: List[int]) -> np.ndarray:
        """Apply quantum gate to specified qubits."""
        # Simple mock implementation for single qubit gates
        if gate.shape == (2, 2) and len(qubit_indices) == 1:
            return gate @ state
        return state  # Mock return for multi-qubit gates
    
    def measure_quantum_state(self, state: np.ndarray, 
                            measurement_basis: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Simulate quantum measurement."""
        probabilities = np.abs(state)**2
        
        # Mock measurement outcome
        outcome_index = np.random.choice(len(probabilities), p=probabilities)
        
        return {
            'outcome': outcome_index,
            'probability': probabilities[outcome_index],
            'post_measurement_state': self._collapse_state(state, outcome_index),
            'measurement_probabilities': probabilities
        }
    
    def simulate_quantum_algorithm(self, algorithm_name: str, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate various quantum algorithms."""
        if algorithm_name == 'grover':
            return self._simulate_grover_algorithm(parameters)
        elif algorithm_name == 'shor':
            return self._simulate_shor_algorithm(parameters)
        elif algorithm_name == 'vqe':
            return self._simulate_vqe_algorithm(parameters)
        else:
            raise ValueError(f"Unknown quantum algorithm: {algorithm_name}")
    
    def _matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix exponential using eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.conj().T
    
    def _collapse_state(self, state: np.ndarray, outcome: int) -> np.ndarray:
        """Collapse quantum state after measurement."""
        collapsed_state = np.zeros_like(state)
        collapsed_state[outcome] = 1.0
        return collapsed_state
    
    def _simulate_grover_algorithm(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Grover's algorithm simulation."""
        n_qubits = parameters.get('n_qubits', 3)
        target_item = parameters.get('target_item', 0)
        
        # Mock result - in reality would simulate full Grover circuit
        return {
            'success_probability': 1.0 - 1.0/2**n_qubits,
            'optimal_iterations': int(np.pi/4 * np.sqrt(2**n_qubits)),
            'measured_outcome': target_item,
            'final_state': np.zeros(2**n_qubits)  # Mock final state
        }
    
    def _simulate_shor_algorithm(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Shor's algorithm simulation."""
        number_to_factor = parameters.get('number', 15)
        
        # Mock factorization result
        if number_to_factor == 15:
            return {
                'factors': [3, 5],
                'success_probability': 0.8,
                'quantum_period': 4,
                'classical_post_processing': 'successful'
            }
        else:
            return {
                'factors': [1, number_to_factor],
                'success_probability': 0.0,
                'quantum_period': None,
                'classical_post_processing': 'failed'
            }
    
    def _simulate_vqe_algorithm(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Variational Quantum Eigensolver simulation."""
        hamiltonian = parameters.get('hamiltonian', np.array([[1, 0], [0, -1]]))
        
        # Mock VQE optimization
        ground_state_energy = np.min(np.linalg.eigvals(hamiltonian))
        
        return {
            'ground_state_energy': ground_state_energy,
            'optimization_iterations': 100,
            'final_parameters': np.random.random(4),  # Mock ansatz parameters
            'convergence_achieved': True
        }


class TestQuantumPhysicsAgent:
    """Test class for quantum physics functionality."""
    
    @pytest.fixture
    def quantum_agent(self):
        """Create a quantum physics agent instance for testing."""
        return MockQuantumPhysicsAgent()
    
    def test_agent_initialization(self, quantum_agent):
        """Test quantum physics agent initialization."""
        assert quantum_agent.name == "QuantumPhysicsAgent"
        assert quantum_agent.initialized is True
        assert hasattr(quantum_agent, 'quantum_systems')
    
    def test_solve_schrodinger_equation(self, quantum_agent, physics_test_config):
        """Test Schrödinger equation solving for simple systems."""
        # Test with Pauli-Z Hamiltonian
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]])
        
        result = quantum_agent.solve_schrodinger_equation(hamiltonian)
        
        # Check that eigenvalues are correct
        expected_eigenvalues = np.array([-1.0, 1.0])
        np.testing.assert_array_almost_equal(
            np.sort(result['eigenvalues']), 
            expected_eigenvalues,
            decimal=10
        )
        
        # Check that eigenvectors are orthonormal
        eigenvectors = result['eigenvectors']
        identity = eigenvectors.conj().T @ eigenvectors
        np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=10)
        
        # Check ground state energy
        assert result['ground_state_energy'] == min(result['eigenvalues'])
    
    def test_calculate_ground_state(self, quantum_agent, physics_tolerance_config):
        """Test ground state calculation."""
        # Harmonic oscillator Hamiltonian (simplified 2x2 version)
        hamiltonian = np.array([[0.5, 0.1], [0.1, 1.5]])
        
        result = quantum_agent.calculate_ground_state(hamiltonian)
        
        # Check that we get an energy value
        assert 'energy' in result
        assert 'wavefunction' in result
        assert 'probability_density' in result
        
        # Check wavefunction normalization
        wavefunction = result['wavefunction']
        norm = np.sum(np.abs(wavefunction)**2)
        tolerance = physics_tolerance_config['quantum_mechanics']['wavefunction_tolerance']
        assert abs(norm - 1.0) < tolerance
        
        # Check probability density
        prob_density = result['probability_density']
        assert np.sum(prob_density) == pytest.approx(1.0, rel=1e-10)
    
    def test_time_evolution(self, quantum_agent):
        """Test quantum state time evolution."""
        # Initial state |0⟩
        initial_state = np.array([1.0, 0.0])
        
        # Simple Hamiltonian
        hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]])  # Pauli-X
        
        # Evolve for time π/2 (should give |1⟩ state)
        time = np.pi / 2
        evolved_state = quantum_agent.evolve_quantum_state(initial_state, hamiltonian, time)
        
        # Check that state is properly normalized
        norm = np.sum(np.abs(evolved_state)**2)
        assert abs(norm - 1.0) < 1e-10
        
        # For Pauli-X evolution, |0⟩ → -i|1⟩ at t=π/2
        expected_state = np.array([0.0, -1j])
        np.testing.assert_array_almost_equal(evolved_state, expected_state, decimal=10)
    
    def test_expectation_value_calculation(self, quantum_agent):
        """Test expectation value calculations."""
        # State |+⟩ = (|0⟩ + |1⟩)/√2
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        
        # Pauli-Z operator
        pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        
        expectation_value = quantum_agent.calculate_expectation_value(state, pauli_z)
        
        # For |+⟩ state, ⟨Z⟩ = 0
        assert abs(expectation_value) < 1e-10
        
        # Test with |0⟩ state
        state_0 = np.array([1.0, 0.0])
        expectation_0 = quantum_agent.calculate_expectation_value(state_0, pauli_z)
        assert expectation_0 == pytest.approx(1.0, rel=1e-10)
    
    def test_entanglement_entropy(self, quantum_agent):
        """Test entanglement entropy calculation."""
        # Test with product state (should have zero entanglement)
        product_state = np.array([1.0, 0.0, 0.0, 0.0])  # |00⟩
        entropy = quantum_agent.calculate_entanglement_entropy(product_state, subsystem_size=1)
        
        # Product states should have low entanglement
        assert entropy >= 0.0
        
        # Test with maximally entangled state
        bell_state = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)])  # |Φ+⟩
        entropy_entangled = quantum_agent.calculate_entanglement_entropy(bell_state, subsystem_size=1)
        
        # Entangled states should have higher entropy
        assert entropy_entangled > entropy
    
    def test_quantum_gate_application(self, quantum_agent):
        """Test application of quantum gates."""
        # Initial state |0⟩
        initial_state = np.array([1.0, 0.0])
        
        # Pauli-X gate (bit flip)
        pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        final_state = quantum_agent.apply_quantum_gate(initial_state, pauli_x, qubit_indices=[0])
        
        # Should result in |1⟩ state
        expected_state = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(final_state, expected_state, decimal=10)
        
        # Test Hadamard gate
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
        hadamard_result = quantum_agent.apply_quantum_gate(initial_state, hadamard, qubit_indices=[0])
        
        # Should result in |+⟩ state
        expected_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(hadamard_result, expected_plus, decimal=10)
    
    def test_quantum_measurement(self, quantum_agent):
        """Test quantum measurement simulation."""
        # Test with |0⟩ state (deterministic measurement)
        state_0 = np.array([1.0, 0.0])
        
        result = quantum_agent.measure_quantum_state(state_0)
        
        assert 'outcome' in result
        assert 'probability' in result
        assert 'post_measurement_state' in result
        assert 'measurement_probabilities' in result
        
        # For |0⟩ state, should always measure 0
        assert result['outcome'] == 0
        assert result['probability'] == pytest.approx(1.0, rel=1e-10)
        
        # Test with superposition state
        superposition_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        superposition_result = quantum_agent.measure_quantum_state(superposition_state)
        
        # Should have equal probabilities
        probabilities = superposition_result['measurement_probabilities']
        assert abs(probabilities[0] - 0.5) < 1e-10
        assert abs(probabilities[1] - 0.5) < 1e-10
    
    def test_grover_algorithm_simulation(self, quantum_agent):
        """Test Grover's algorithm simulation."""
        parameters = {
            'n_qubits': 3,
            'target_item': 5
        }
        
        result = quantum_agent.simulate_quantum_algorithm('grover', parameters)
        
        assert 'success_probability' in result
        assert 'optimal_iterations' in result
        assert 'measured_outcome' in result
        assert 'final_state' in result
        
        # Check that success probability is reasonable
        assert result['success_probability'] > 0.5
        
        # Check optimal iterations formula
        n_qubits = parameters['n_qubits']
        expected_iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
        assert result['optimal_iterations'] == expected_iterations
    
    def test_shor_algorithm_simulation(self, quantum_agent):
        """Test Shor's algorithm simulation."""
        # Test factoring 15
        parameters = {'number': 15}
        
        result = quantum_agent.simulate_quantum_algorithm('shor', parameters)
        
        assert 'factors' in result
        assert 'success_probability' in result
        assert 'quantum_period' in result
        assert 'classical_post_processing' in result
        
        # Check that 15 is correctly factored
        factors = result['factors']
        assert set(factors) == {3, 5}
        assert np.prod(factors) == 15
    
    def test_vqe_algorithm_simulation(self, quantum_agent):
        """Test Variational Quantum Eigensolver simulation."""
        # Test with Pauli-Z Hamiltonian
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]])
        parameters = {'hamiltonian': hamiltonian}
        
        result = quantum_agent.simulate_quantum_algorithm('vqe', parameters)
        
        assert 'ground_state_energy' in result
        assert 'optimization_iterations' in result
        assert 'final_parameters' in result
        assert 'convergence_achieved' in result
        
        # Check that ground state energy is correct
        expected_ground_energy = -1.0
        assert result['ground_state_energy'] == pytest.approx(expected_ground_energy, rel=1e-10)
        
        # Check convergence
        assert result['convergence_achieved'] is True
    
    @pytest.mark.parametrize("hamiltonian_type", [
        "harmonic_oscillator",
        "hydrogen_atom",
        "infinite_square_well"
    ])
    def test_parametrized_quantum_systems(self, quantum_agent, physics_test_config, hamiltonian_type):
        """Test various quantum systems from configuration."""
        config = physics_test_config['quantum_physics']
        
        if hamiltonian_type in config['expected_energies']:
            # Create a mock Hamiltonian based on the system type
            if hamiltonian_type == "harmonic_oscillator":
                # Simple 2x2 approximation
                hamiltonian = np.array([[0.5, 0.0], [0.0, 1.5]])
            elif hamiltonian_type == "hydrogen_atom":
                # Simplified 2-level atom
                hamiltonian = np.array([[-13.6, 0.1], [0.1, -3.4]])
            else:  # infinite_square_well
                hamiltonian = np.array([[9.87, 0.0], [0.0, 39.48]])
            
            result = quantum_agent.solve_schrodinger_equation(hamiltonian)
            
            # Check that we get eigenvalues
            assert len(result['eigenvalues']) > 0
            assert all(np.isreal(result['eigenvalues']))
    
    @pytest.mark.slow
    def test_large_quantum_system(self, quantum_agent):
        """Test quantum calculations on larger systems."""
        # Create a larger random Hermitian matrix
        n = 10
        random_matrix = np.random.random((n, n)) + 1j * np.random.random((n, n))
        hamiltonian = random_matrix + random_matrix.conj().T
        
        result = quantum_agent.solve_schrodinger_equation(hamiltonian)
        
        # Check eigenvalue properties
        eigenvalues = result['eigenvalues']
        assert len(eigenvalues) == n
        assert all(np.isreal(eigenvalues))
        
        # Check eigenvector orthonormality
        eigenvectors = result['eigenvectors']
        identity = eigenvectors.conj().T @ eigenvectors
        np.testing.assert_array_almost_equal(identity, np.eye(n), decimal=8)
    
    def test_quantum_error_handling(self, quantum_agent):
        """Test error handling in quantum calculations."""
        # Test with invalid Hamiltonian
        with pytest.raises(np.linalg.LinAlgError):
            invalid_hamiltonian = np.array([[np.inf, 0], [0, np.nan]])
            quantum_agent.solve_schrodinger_equation(invalid_hamiltonian)
        
        # Test with incompatible dimensions
        hamiltonian = np.array([[1, 0], [0, -1]])
        wrong_state = np.array([1, 0, 0])  # Wrong dimension
        
        with pytest.raises((ValueError, IndexError)):
            quantum_agent.calculate_expectation_value(wrong_state, hamiltonian)
    
    def test_quantum_algorithm_error_handling(self, quantum_agent):
        """Test error handling in quantum algorithm simulations."""
        # Test with unknown algorithm
        with pytest.raises(ValueError, match="Unknown quantum algorithm"):
            quantum_agent.simulate_quantum_algorithm('unknown_algorithm', {})
        
        # Test with invalid parameters
        with pytest.raises((KeyError, ValueError)):
            quantum_agent.simulate_quantum_algorithm('grover', {'invalid_param': 123})


class TestQuantumPhysicsIntegration:
    """Integration tests for quantum physics workflows."""
    
    @pytest.fixture
    def quantum_workflow(self):
        """Create a complete quantum physics workflow."""
        agent = MockQuantumPhysicsAgent()
        
        workflow = {
            'agent': agent,
            'systems': [],
            'results': {}
        }
        
        return workflow
    
    def test_complete_quantum_calculation_workflow(self, quantum_workflow, physics_test_config):
        """Test a complete quantum calculation workflow."""
        agent = quantum_workflow['agent']
        
        # Step 1: Define quantum system
        hamiltonian = np.array([[1.0, 0.1], [0.1, -1.0]])
        
        # Step 2: Solve eigenvalue problem
        eigenvalue_result = agent.solve_schrodinger_equation(hamiltonian)
        
        # Step 3: Calculate ground state properties
        ground_state_result = agent.calculate_ground_state(hamiltonian)
        
        # Step 4: Time evolution
        initial_state = ground_state_result['wavefunction']
        evolved_state = agent.evolve_quantum_state(initial_state, hamiltonian, time=1.0)
        
        # Step 5: Calculate observables
        pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        expectation_initial = agent.calculate_expectation_value(initial_state, pauli_z)
        expectation_evolved = agent.calculate_expectation_value(evolved_state, pauli_z)
        
        # Verify workflow consistency
        assert eigenvalue_result['ground_state_energy'] == ground_state_result['energy']
        assert np.allclose(eigenvalue_result['ground_state_wavefunction'], initial_state)
        
        # Store results
        quantum_workflow['results'] = {
            'eigenvalues': eigenvalue_result['eigenvalues'],
            'ground_state_energy': ground_state_result['energy'],
            'time_evolution': evolved_state,
            'expectation_values': [expectation_initial, expectation_evolved]
        }
        
        # Verify all steps completed successfully
        assert len(quantum_workflow['results']) == 4
    
    @pytest.mark.integration
    def test_quantum_measurement_statistics(self, quantum_workflow):
        """Test quantum measurement statistics over multiple runs."""
        agent = quantum_workflow['agent']
        
        # Prepare superposition state
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        
        # Perform multiple measurements
        n_measurements = 1000
        outcomes = []
        
        for _ in range(n_measurements):
            result = agent.measure_quantum_state(state)
            outcomes.append(result['outcome'])
        
        # Check statistics
        outcome_0_count = outcomes.count(0)
        outcome_1_count = outcomes.count(1)
        
        # Should be approximately 50-50 for superposition state
        expected_ratio = 0.5
        tolerance = 0.1  # 10% tolerance for statistical fluctuations
        
        ratio_0 = outcome_0_count / n_measurements
        ratio_1 = outcome_1_count / n_measurements
        
        assert abs(ratio_0 - expected_ratio) < tolerance
        assert abs(ratio_1 - expected_ratio) < tolerance
    
    @pytest.mark.asyncio
    async def test_async_quantum_simulation(self, async_physics_simulator):
        """Test asynchronous quantum simulation."""
        # Start simulation
        simulation_task = asyncio.create_task(
            async_physics_simulator.run_simulation(duration=0.1, dt=0.01)
        )
        
        # Verify simulation is running
        assert async_physics_simulator.running is True
        
        # Wait for completion
        result = await simulation_task
        
        # Verify completion
        assert result['status'] == 'completed'
        assert result['final_time'] == 0.1
        assert async_physics_simulator.running is False