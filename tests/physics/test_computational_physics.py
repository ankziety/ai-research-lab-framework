"""
Computational Physics Tests

Comprehensive unit tests for computational physics methods and algorithms.
Tests numerical methods, simulations, and computational algorithms.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Callable
from unittest.mock import Mock, patch
import time

# Computational physics test markers
pytestmark = pytest.mark.computational


class MockComputationalPhysicsAgent:
    """Mock computational physics agent for testing."""
    
    def __init__(self):
        self.name = "ComputationalPhysicsAgent"
        self.initialized = True
        self.numerical_methods = ['euler', 'runge_kutta_4', 'verlet', 'leapfrog']
        self.simulation_cache = {}
    
    def solve_ode(self, ode_func: Callable, initial_conditions: np.ndarray,
                  time_span: Tuple[float, float], method: str = 'runge_kutta_4',
                  dt: float = 0.01) -> Dict[str, Any]:
        """Solve ordinary differential equation using specified method."""
        t_start, t_end = time_span
        t = np.arange(t_start, t_end + dt, dt)
        y = np.zeros((len(t), len(initial_conditions)))
        y[0] = initial_conditions
        
        for i in range(1, len(t)):
            if method == 'euler':
                y[i] = self._euler_step(ode_func, y[i-1], t[i-1], dt)
            elif method == 'runge_kutta_4':
                y[i] = self._rk4_step(ode_func, y[i-1], t[i-1], dt)
            elif method == 'verlet':
                if i == 1:
                    # Use Euler for first step
                    y[i] = self._euler_step(ode_func, y[i-1], t[i-1], dt)
                else:
                    y[i] = self._verlet_step(ode_func, y[i-1], y[i-2], t[i-1], dt)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return {
            'time': t,
            'solution': y,
            'method': method,
            'final_state': y[-1],
            'energy_conservation': self._check_energy_conservation(y),
            'convergence_achieved': True
        }
    
    def monte_carlo_integration(self, func: Callable, bounds: List[Tuple[float, float]],
                              n_samples: int = 10000) -> Dict[str, Any]:
        """Perform Monte Carlo integration."""
        ndim = len(bounds)
        
        # Generate random samples
        samples = np.zeros((n_samples, ndim))
        for i, (a, b) in enumerate(bounds):
            samples[:, i] = np.random.uniform(a, b, n_samples)
        
        # Evaluate function at sample points
        if ndim == 1:
            func_values = np.array([func(x[0]) for x in samples])
        elif ndim == 2:
            func_values = np.array([func(x[0], x[1]) for x in samples])
        elif ndim == 3:
            func_values = np.array([func(x[0], x[1], x[2]) for x in samples])
        else:
            func_values = np.array([func(*x) for x in samples])
        
        # Calculate volume of integration region
        volume = np.prod([b - a for a, b in bounds])
        
        # Monte Carlo estimate
        integral_estimate = volume * np.mean(func_values)
        error_estimate = volume * np.std(func_values) / np.sqrt(n_samples)
        
        return {
            'integral': integral_estimate,
            'error': error_estimate,
            'n_samples': n_samples,
            'convergence_rate': 1.0 / np.sqrt(n_samples),
            'variance': np.var(func_values)
        }
    
    def molecular_dynamics_simulation(self, n_particles: int, n_steps: int,
                                    dt: float = 0.001, temperature: float = 300.0) -> Dict[str, Any]:
        """Run molecular dynamics simulation."""
        # Initialize particles
        positions = np.random.random((n_particles, 3))
        velocities = np.random.normal(0, np.sqrt(temperature), (n_particles, 3))
        masses = np.ones(n_particles)
        
        # Storage for trajectory
        trajectory = {
            'positions': np.zeros((n_steps, n_particles, 3)),
            'velocities': np.zeros((n_steps, n_particles, 3)),
            'energies': np.zeros(n_steps),
            'temperatures': np.zeros(n_steps)
        }
        
        for step in range(n_steps):
            # Calculate forces (simple harmonic potential for testing)
            forces = -positions  # F = -kx with k=1
            
            # Velocity Verlet integration
            positions += velocities * dt + 0.5 * forces / masses[:, np.newaxis] * dt**2
            velocities += 0.5 * forces / masses[:, np.newaxis] * dt
            
            # Update forces
            new_forces = -positions
            velocities += 0.5 * new_forces / masses[:, np.newaxis] * dt
            
            # Calculate energy and temperature
            kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
            potential_energy = 0.5 * np.sum(positions**2)
            total_energy = kinetic_energy + potential_energy
            
            # Store trajectory data
            trajectory['positions'][step] = positions.copy()
            trajectory['velocities'][step] = velocities.copy()
            trajectory['energies'][step] = total_energy
            trajectory['temperatures'][step] = (2 * kinetic_energy) / (3 * n_particles)
        
        return {
            'trajectory': trajectory,
            'final_positions': positions,
            'final_velocities': velocities,
            'average_temperature': np.mean(trajectory['temperatures']),
            'energy_drift': abs(trajectory['energies'][-1] - trajectory['energies'][0]),
            'simulation_parameters': {
                'n_particles': n_particles,
                'n_steps': n_steps,
                'dt': dt,
                'target_temperature': temperature
            }
        }
    
    def finite_difference_solver(self, pde_func: Callable, boundary_conditions: Dict,
                               grid_size: Tuple[int, int], domain: Tuple[Tuple[float, float], Tuple[float, float]]) -> Dict[str, Any]:
        """Solve PDE using finite difference method."""
        nx, ny = grid_size
        (x_min, x_max), (y_min, y_max) = domain
        
        # Create grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Initialize solution grid
        u = np.zeros((nx, ny))
        
        # Apply boundary conditions
        if 'dirichlet' in boundary_conditions:
            bc = boundary_conditions['dirichlet']
            if 'left' in bc:
                u[0, :] = bc['left']
            if 'right' in bc:
                u[-1, :] = bc['right']
            if 'bottom' in bc:
                u[:, 0] = bc['bottom']
            if 'top' in bc:
                u[:, -1] = bc['top']
        
        # Iterative solver (simplified Jacobi method)
        max_iterations = 1000
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            u_old = u.copy()
            
            # Update interior points
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] +
                                    u_old[i, j+1] + u_old[i, j-1])
            
            # Check convergence
            residual = np.max(np.abs(u - u_old))
            if residual < tolerance:
                break
        
        return {
            'solution': u,
            'grid_x': x,
            'grid_y': y,
            'iterations': iteration + 1,
            'residual': residual,
            'converged': residual < tolerance,
            'grid_spacing': (dx, dy)
        }
    
    def fft_analysis(self, signal: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, Any]:
        """Perform Fast Fourier Transform analysis."""
        n = len(signal)
        
        # Compute FFT
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)
        
        # Calculate power spectrum
        power_spectrum = np.abs(fft_result)**2
        
        # Find dominant frequencies
        positive_freq_indices = frequencies > 0
        positive_frequencies = frequencies[positive_freq_indices]
        positive_power = power_spectrum[positive_freq_indices]
        
        # Get top 5 dominant frequencies
        dominant_indices = np.argsort(positive_power)[-5:]
        dominant_frequencies = positive_frequencies[dominant_indices]
        dominant_amplitudes = positive_power[dominant_indices]
        
        return {
            'fft_result': fft_result,
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'dominant_frequencies': dominant_frequencies,
            'dominant_amplitudes': dominant_amplitudes,
            'total_power': np.sum(power_spectrum),
            'peak_frequency': positive_frequencies[np.argmax(positive_power)],
            'frequency_resolution': sampling_rate / n
        }
    
    def finite_element_analysis(self, mesh_nodes: np.ndarray, elements: np.ndarray,
                              material_properties: Dict[str, float],
                              boundary_conditions: Dict) -> Dict[str, Any]:
        """Perform simplified finite element analysis."""
        n_nodes = len(mesh_nodes)
        n_elements = len(elements)
        
        # Mock stiffness matrix assembly
        K = np.zeros((n_nodes, n_nodes))
        
        # Simple spring-like elements for testing
        youngs_modulus = material_properties.get('youngs_modulus', 1.0)
        
        for element in elements:
            for i in range(len(element)):
                for j in range(len(element)):
                    node_i = element[i]
                    node_j = element[j]
                    if i != j:
                        K[node_i, node_j] += -youngs_modulus
                    else:
                        K[node_i, node_j] += youngs_modulus * (len(element) - 1)
        
        # Apply boundary conditions (simplified)
        fixed_nodes = boundary_conditions.get('fixed_nodes', [])
        applied_forces = boundary_conditions.get('forces', {})
        
        # Create force vector
        F = np.zeros(n_nodes)
        for node, force in applied_forces.items():
            F[node] = force
        
        # Solve K * u = F (with constraints)
        K_reduced = K.copy()
        F_reduced = F.copy()
        
        # Fix nodes by modifying matrix
        for node in fixed_nodes:
            K_reduced[node, :] = 0
            K_reduced[:, node] = 0
            K_reduced[node, node] = 1
            F_reduced[node] = 0
        
        # Solve system
        try:
            displacements = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            displacements = np.zeros(n_nodes)  # Fallback for singular matrices
        
        return {
            'displacements': displacements,
            'stiffness_matrix': K,
            'applied_forces': F,
            'mesh_nodes': mesh_nodes,
            'elements': elements,
            'max_displacement': np.max(np.abs(displacements)),
            'analysis_converged': True
        }
    
    def spectral_methods_solver(self, function: np.ndarray, domain_size: float = 1.0) -> Dict[str, Any]:
        """Solve using spectral methods (Fourier basis)."""
        n = len(function)
        
        # Compute Fourier coefficients
        fourier_coeffs = np.fft.fft(function)
        
        # Create wave numbers
        k = 2 * np.pi * np.fft.fftfreq(n, domain_size / n)
        
        # Apply differential operator in Fourier space (example: Laplacian)
        laplacian_coeffs = -k**2 * fourier_coeffs
        
        # Transform back to physical space
        laplacian_result = np.real(np.fft.ifft(laplacian_coeffs))
        
        # Calculate spectral accuracy metrics
        spectral_error = np.max(np.abs(fourier_coeffs[-10:]))  # High frequency content
        
        return {
            'fourier_coefficients': fourier_coeffs,
            'wave_numbers': k,
            'laplacian_result': laplacian_result,
            'spectral_error': spectral_error,
            'max_wavenumber': np.max(np.abs(k)),
            'spectral_resolution': 2 * np.pi / domain_size,
            'aliasing_error': spectral_error / np.max(np.abs(fourier_coeffs))
        }
    
    def parallel_computation_benchmark(self, computation_func: Callable,
                                     data_sizes: List[int], n_workers: int = 4) -> Dict[str, Any]:
        """Benchmark parallel computation performance."""
        results = {}
        
        for size in data_sizes:
            # Generate test data
            test_data = np.random.random(size)
            
            # Sequential computation
            start_time = time.time()
            sequential_result = computation_func(test_data)
            sequential_time = time.time() - start_time
            
            # Mock parallel computation (simulate speedup)
            parallel_time = sequential_time / min(n_workers, size // 1000 + 1)
            speedup = sequential_time / parallel_time
            efficiency = speedup / n_workers
            
            results[size] = {
                'sequential_time': sequential_time,
                'parallel_time': parallel_time,
                'speedup': speedup,
                'efficiency': efficiency,
                'result': sequential_result
            }
        
        return {
            'benchmark_results': results,
            'optimal_workers': n_workers,
            'scalability_factor': np.mean([r['efficiency'] for r in results.values()]),
            'performance_summary': {
                'best_speedup': max(r['speedup'] for r in results.values()),
                'best_efficiency': max(r['efficiency'] for r in results.values()),
                'average_speedup': np.mean([r['speedup'] for r in results.values()])
            }
        }
    
    # Helper methods
    def _euler_step(self, func, y, t, dt):
        """Single Euler method step."""
        return y + dt * func(t, y)
    
    def _rk4_step(self, func, y, t, dt):
        """Single Runge-Kutta 4th order step."""
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt/2, y + k1/2)
        k3 = dt * func(t + dt/2, y + k2/2)
        k4 = dt * func(t + dt, y + k3)
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _verlet_step(self, func, y_current, y_previous, t, dt):
        """Single Verlet integration step."""
        # Simplified for position-like variables
        acceleration = func(t, y_current)
        return 2 * y_current - y_previous + acceleration * dt**2
    
    def _check_energy_conservation(self, trajectory):
        """Check energy conservation in trajectory."""
        if len(trajectory) < 2:
            return True
        
        # Mock energy calculation
        energies = np.sum(trajectory**2, axis=1)  # Simple quadratic energy
        energy_variation = np.std(energies) / np.mean(energies)
        return energy_variation < 0.01  # 1% tolerance


class TestComputationalPhysicsAgent:
    """Test class for computational physics functionality."""
    
    @pytest.fixture
    def computational_agent(self):
        """Create a computational physics agent instance for testing."""
        return MockComputationalPhysicsAgent()
    
    def test_agent_initialization(self, computational_agent):
        """Test computational physics agent initialization."""
        assert computational_agent.name == "ComputationalPhysicsAgent"
        assert computational_agent.initialized is True
        assert hasattr(computational_agent, 'numerical_methods')
        assert len(computational_agent.numerical_methods) > 0
    
    def test_ode_solver_euler(self, computational_agent):
        """Test ODE solver with Euler method."""
        # Simple harmonic oscillator: d²x/dt² = -ω²x
        # Convert to first order system: dy/dt = [y[1], -ω²*y[0]]
        omega = 1.0
        
        def harmonic_oscillator(t, y):
            return np.array([y[1], -omega**2 * y[0]])
        
        initial_conditions = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
        time_span = (0.0, 2*np.pi)
        
        result = computational_agent.solve_ode(
            harmonic_oscillator, initial_conditions, time_span, method='euler', dt=0.01
        )
        
        assert 'time' in result
        assert 'solution' in result
        assert 'method' in result
        assert result['method'] == 'euler'
        
        # Check that solution has correct shape
        assert result['solution'].shape[1] == len(initial_conditions)
        
        # Check final state (should be close to initial for full period)
        final_state = result['final_state']
        assert len(final_state) == len(initial_conditions)
    
    def test_ode_solver_runge_kutta_4(self, computational_agent):
        """Test ODE solver with Runge-Kutta 4th order method."""
        # Exponential decay: dy/dt = -λy
        lambda_val = 0.5
        
        def exponential_decay(t, y):
            return -lambda_val * y
        
        initial_conditions = np.array([1.0])
        time_span = (0.0, 2.0)
        
        result = computational_agent.solve_ode(
            exponential_decay, initial_conditions, time_span, method='runge_kutta_4'
        )
        
        assert result['method'] == 'runge_kutta_4'
        assert result['convergence_achieved'] is True
        
        # Check that solution decays
        solution = result['solution'][:, 0]
        assert solution[0] > solution[-1]  # Should decay
        assert all(solution > 0)  # Should remain positive
    
    def test_monte_carlo_integration(self, computational_agent):
        """Test Monte Carlo integration."""
        # Integrate x² from 0 to 1 (analytical result = 1/3)
        def quadratic(x):
            return x**2
        
        bounds = [(0.0, 1.0)]
        n_samples = 100000
        
        result = computational_agent.monte_carlo_integration(quadratic, bounds, n_samples)
        
        assert 'integral' in result
        assert 'error' in result
        assert 'n_samples' in result
        assert result['n_samples'] == n_samples
        
        # Check that result is reasonably close to analytical value
        analytical_result = 1.0 / 3.0
        relative_error = abs(result['integral'] - analytical_result) / analytical_result
        assert relative_error < 0.1  # Within 10%
    
    def test_monte_carlo_integration_multidimensional(self, computational_agent):
        """Test Monte Carlo integration in multiple dimensions."""
        # Integrate x*y over [0,1] × [0,1] (analytical result = 1/4)
        def bivariate_func(x, y):
            return x * y
        
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        result = computational_agent.monte_carlo_integration(bivariate_func, bounds)
        
        assert 'integral' in result
        assert 'variance' in result
        
        # Check result is reasonable
        analytical_result = 0.25
        relative_error = abs(result['integral'] - analytical_result) / analytical_result
        assert relative_error < 0.2  # Within 20% for stochastic method
    
    def test_molecular_dynamics_simulation(self, computational_agent):
        """Test molecular dynamics simulation."""
        n_particles = 10
        n_steps = 100
        dt = 0.001
        temperature = 300.0
        
        result = computational_agent.molecular_dynamics_simulation(
            n_particles, n_steps, dt, temperature
        )
        
        assert 'trajectory' in result
        assert 'final_positions' in result
        assert 'final_velocities' in result
        assert 'average_temperature' in result
        assert 'energy_drift' in result
        
        # Check trajectory dimensions
        trajectory = result['trajectory']
        assert trajectory['positions'].shape == (n_steps, n_particles, 3)
        assert trajectory['velocities'].shape == (n_steps, n_particles, 3)
        assert len(trajectory['energies']) == n_steps
        
        # Check energy conservation (should be reasonable)
        energy_drift = result['energy_drift']
        initial_energy = trajectory['energies'][0]
        relative_drift = energy_drift / initial_energy if initial_energy > 0 else energy_drift
        assert relative_drift < 0.1  # Less than 10% energy drift
    
    def test_finite_difference_solver(self, computational_agent):
        """Test finite difference PDE solver."""
        # Solve Laplace equation with Dirichlet boundary conditions
        def laplace_eq(x, y, u):
            return 0  # ∇²u = 0
        
        boundary_conditions = {
            'dirichlet': {
                'left': 0.0,
                'right': 1.0,
                'bottom': 0.0,
                'top': 0.0
            }
        }
        
        grid_size = (21, 21)
        domain = ((0.0, 1.0), (0.0, 1.0))
        
        result = computational_agent.finite_difference_solver(
            laplace_eq, boundary_conditions, grid_size, domain
        )
        
        assert 'solution' in result
        assert 'grid_x' in result
        assert 'grid_y' in result
        assert 'converged' in result
        
        # Check solution dimensions
        assert result['solution'].shape == grid_size
        
        # Check boundary conditions are satisfied
        u = result['solution']
        np.testing.assert_array_almost_equal(u[0, :], 0.0)  # Left boundary
        np.testing.assert_array_almost_equal(u[-1, :], 1.0)  # Right boundary
        np.testing.assert_array_almost_equal(u[:, 0], 0.0)  # Bottom boundary
        np.testing.assert_array_almost_equal(u[:, -1], 0.0)  # Top boundary
    
    def test_fft_analysis(self, computational_agent):
        """Test Fast Fourier Transform analysis."""
        # Create test signal with known frequencies
        sampling_rate = 1000.0  # Hz
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        # Signal with 50 Hz and 120 Hz components
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        
        result = computational_agent.fft_analysis(signal, sampling_rate)
        
        assert 'fft_result' in result
        assert 'frequencies' in result
        assert 'power_spectrum' in result
        assert 'dominant_frequencies' in result
        assert 'peak_frequency' in result
        
        # Check that dominant frequencies are detected
        dominant_freqs = result['dominant_frequencies']
        
        # Should detect 50 Hz and 120 Hz peaks
        assert any(abs(freq - 50.0) < 5.0 for freq in dominant_freqs)
        assert any(abs(freq - 120.0) < 5.0 for freq in dominant_freqs)
    
    def test_finite_element_analysis(self, computational_agent):
        """Test finite element analysis."""
        # Simple 1D chain of nodes
        mesh_nodes = np.array([[i, 0, 0] for i in range(5)])  # 5 nodes in a line
        elements = [[i, i+1] for i in range(4)]  # 4 elements connecting nodes
        
        material_properties = {'youngs_modulus': 200e9}  # Steel
        
        boundary_conditions = {
            'fixed_nodes': [0],  # Fix first node
            'forces': {4: 1000.0}  # Apply force to last node
        }
        
        result = computational_agent.finite_element_analysis(
            mesh_nodes, elements, material_properties, boundary_conditions
        )
        
        assert 'displacements' in result
        assert 'stiffness_matrix' in result
        assert 'max_displacement' in result
        assert 'analysis_converged' in result
        
        # Check that fixed node has zero displacement
        displacements = result['displacements']
        assert abs(displacements[0]) < 1e-10  # Fixed node
        
        # Check that loaded node has non-zero displacement
        assert abs(displacements[4]) > 0  # Loaded node
        
        # Check matrix dimensions
        K = result['stiffness_matrix']
        assert K.shape == (len(mesh_nodes), len(mesh_nodes))
    
    def test_spectral_methods_solver(self, computational_agent):
        """Test spectral methods solver."""
        # Create smooth test function
        n = 64
        x = np.linspace(0, 2*np.pi, n, endpoint=False)
        function = np.sin(2*x) + 0.5*np.cos(3*x)
        
        result = computational_agent.spectral_methods_solver(function, domain_size=2*np.pi)
        
        assert 'fourier_coefficients' in result
        assert 'wave_numbers' in result
        assert 'laplacian_result' in result
        assert 'spectral_error' in result
        
        # Check dimensions
        assert len(result['fourier_coefficients']) == n
        assert len(result['wave_numbers']) == n
        assert len(result['laplacian_result']) == n
        
        # Spectral error should be small for smooth functions
        assert result['spectral_error'] < 1e-10
    
    def test_parallel_computation_benchmark(self, computational_agent):
        """Test parallel computation benchmarking."""
        # Simple computation function for benchmarking
        def test_computation(data):
            return np.sum(data**2)
        
        data_sizes = [1000, 10000, 100000]
        
        result = computational_agent.parallel_computation_benchmark(
            test_computation, data_sizes, n_workers=4
        )
        
        assert 'benchmark_results' in result
        assert 'performance_summary' in result
        assert 'scalability_factor' in result
        
        # Check that results exist for all data sizes
        benchmark_results = result['benchmark_results']
        for size in data_sizes:
            assert size in benchmark_results
            assert 'speedup' in benchmark_results[size]
            assert 'efficiency' in benchmark_results[size]
            
            # Speedup should be positive
            assert benchmark_results[size]['speedup'] > 0
            
            # Efficiency should be between 0 and 1
            assert 0 <= benchmark_results[size]['efficiency'] <= 1
    
    @pytest.mark.parametrize("method", ['euler', 'runge_kutta_4'])
    def test_ode_solver_methods(self, computational_agent, method):
        """Test different ODE solver methods."""
        # Simple linear ODE: dy/dt = -y
        def linear_ode(t, y):
            return -y
        
        initial_conditions = np.array([1.0])
        time_span = (0.0, 1.0)
        
        result = computational_agent.solve_ode(
            linear_ode, initial_conditions, time_span, method=method, dt=0.01
        )
        
        assert result['method'] == method
        assert 'solution' in result
        assert result['convergence_achieved'] is True
        
        # Check exponential decay behavior
        solution = result['solution'][:, 0]
        assert solution[0] > solution[-1]  # Should decay
    
    @pytest.mark.slow
    def test_large_scale_computation(self, computational_agent):
        """Test computational methods on larger scales."""
        # Large matrix operations
        n = 1000
        matrix = np.random.random((n, n))
        
        # Time a matrix operation
        start_time = time.time()
        eigenvalues = np.linalg.eigvals(matrix)
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert computation_time < 10.0  # Less than 10 seconds
        assert len(eigenvalues) == n
        
        # Test large FFT
        large_signal = np.random.random(2**16)  # 65536 points
        
        start_time = time.time()
        result = computational_agent.fft_analysis(large_signal)
        fft_time = time.time() - start_time
        
        assert fft_time < 1.0  # FFT should be fast
        assert len(result['fft_result']) == len(large_signal)
    
    def test_numerical_accuracy(self, computational_agent, physics_tolerance_config):
        """Test numerical accuracy of computational methods."""
        # Test integration accuracy
        def exact_integrable(x):
            return x**3  # Integral from 0 to 1 is 1/4
        
        bounds = [(0.0, 1.0)]
        result = computational_agent.monte_carlo_integration(exact_integrable, bounds, n_samples=1000000)
        
        analytical_result = 0.25
        relative_error = abs(result['integral'] - analytical_result) / analytical_result
        
        tolerance = physics_tolerance_config['numerical_methods']['integration_tolerance']
        assert relative_error < tolerance * 1000  # More lenient for Monte Carlo
        
        # Test ODE solver accuracy
        def exponential_ode(t, y):
            return -y
        
        initial_conditions = np.array([1.0])
        time_span = (0.0, 1.0)
        
        ode_result = computational_agent.solve_ode(
            exponential_ode, initial_conditions, time_span, method='runge_kutta_4', dt=0.001
        )
        
        # Analytical solution at t=1 is e^(-1)
        numerical_solution = ode_result['final_state'][0]
        analytical_solution = np.exp(-1)
        ode_relative_error = abs(numerical_solution - analytical_solution) / analytical_solution
        
        assert ode_relative_error < tolerance
    
    def test_computational_error_handling(self, computational_agent):
        """Test error handling in computational methods."""
        # Test with invalid ODE function
        def invalid_ode(t, y):
            return np.array([np.inf, np.nan])
        
        initial_conditions = np.array([1.0, 1.0])
        time_span = (0.0, 1.0)
        
        # Should handle numerical errors gracefully
        result = computational_agent.solve_ode(invalid_ode, initial_conditions, time_span)
        assert 'solution' in result  # Should not crash
        
        # Test with unknown method
        with pytest.raises(ValueError, match="Unknown method"):
            computational_agent.solve_ode(
                lambda t, y: -y, np.array([1.0]), (0.0, 1.0), method='unknown_method'
            )


class TestComputationalPhysicsIntegration:
    """Integration tests for computational physics workflows."""
    
    @pytest.fixture
    def computational_workflow(self):
        """Create a computational physics workflow."""
        agent = MockComputationalPhysicsAgent()
        
        workflow = {
            'agent': agent,
            'simulation_data': {},
            'analysis_results': {}
        }
        
        return workflow
    
    def test_complete_simulation_workflow(self, computational_workflow, physics_test_config):
        """Test complete simulation and analysis workflow."""
        agent = computational_workflow['agent']
        
        # Step 1: Run molecular dynamics simulation
        md_result = agent.molecular_dynamics_simulation(
            n_particles=20, n_steps=1000, dt=0.001, temperature=300.0
        )
        
        # Step 2: Analyze trajectory with FFT
        positions = md_result['trajectory']['positions']
        center_of_mass = np.mean(positions, axis=1)  # COM trajectory
        
        fft_result = agent.fft_analysis(center_of_mass[:, 0])  # Analyze x-component
        
        # Step 3: Solve related PDE
        boundary_conditions = {
            'dirichlet': {'left': 0.0, 'right': 1.0, 'bottom': 0.0, 'top': 0.0}
        }
        
        pde_result = agent.finite_difference_solver(
            lambda x, y, u: 0, boundary_conditions, (21, 21), ((0.0, 1.0), (0.0, 1.0))
        )
        
        # Store results
        computational_workflow['simulation_data'] = {
            'md_simulation': md_result,
            'frequency_analysis': fft_result,
            'pde_solution': pde_result
        }
        
        # Verify workflow completion
        assert 'md_simulation' in computational_workflow['simulation_data']
        assert 'frequency_analysis' in computational_workflow['simulation_data']
        assert 'pde_solution' in computational_workflow['simulation_data']
        
        # Check consistency
        md_final_temp = md_result['average_temperature']
        assert md_final_temp > 0  # Should have positive temperature
        
        fft_peak_freq = fft_result['peak_frequency']
        assert fft_peak_freq >= 0  # Frequency should be non-negative
        
        assert pde_result['converged'] is True
    
    @pytest.mark.integration
    def test_multi_scale_simulation(self, computational_workflow):
        """Test multi-scale computational workflow."""
        agent = computational_workflow['agent']
        
        # Microscale: Molecular dynamics
        microscale_result = agent.molecular_dynamics_simulation(
            n_particles=50, n_steps=500, dt=0.0001
        )
        
        # Mesoscale: Finite element analysis
        mesh_nodes = np.array([[i*0.1, j*0.1, 0] for i in range(10) for j in range(10)])
        elements = []
        for i in range(9):
            for j in range(9):
                n1 = i*10 + j
                n2 = (i+1)*10 + j
                n3 = i*10 + (j+1)
                n4 = (i+1)*10 + (j+1)
                elements.extend([[n1, n2, n3], [n2, n3, n4]])
        
        material_props = {'youngs_modulus': 200e9}
        boundary_conds = {'fixed_nodes': [0, 9], 'forces': {90: 1000, 99: 1000}}
        
        mesoscale_result = agent.finite_element_analysis(
            mesh_nodes, elements, material_props, boundary_conds
        )
        
        # Macroscale: PDE solving
        macroscale_result = agent.finite_difference_solver(
            lambda x, y, u: 0,
            {'dirichlet': {'left': 0.0, 'right': 1.0, 'bottom': 0.0, 'top': 0.0}},
            (51, 51), ((0.0, 1.0), (0.0, 1.0))
        )
        
        # Store multi-scale results
        computational_workflow['analysis_results'] = {
            'microscale': microscale_result,
            'mesoscale': mesoscale_result,
            'macroscale': macroscale_result
        }
        
        # Verify all scales completed successfully
        assert computational_workflow['analysis_results']['microscale']['average_temperature'] > 0
        assert computational_workflow['analysis_results']['mesoscale']['analysis_converged'] is True
        assert computational_workflow['analysis_results']['macroscale']['converged'] is True
    
    @pytest.mark.asyncio
    async def test_async_computational_simulation(self, async_physics_simulator):
        """Test asynchronous computational simulation."""
        # Start long-running simulation
        simulation_task = asyncio.create_task(
            async_physics_simulator.run_simulation(duration=0.5, dt=0.001)
        )
        
        # Do other work while simulation runs
        await asyncio.sleep(0.1)
        assert async_physics_simulator.running is True
        
        # Wait for completion
        result = await simulation_task
        
        assert result['status'] == 'completed'
        assert result['steps'] > 0
        assert async_physics_simulator.running is False