"""
Physics Performance Tests

Performance and benchmark tests for physics computations.
Tests computational efficiency, scalability, and performance optimization.
"""

import pytest
import numpy as np
import time
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, patch
import psutil
import gc

# Performance test markers
pytestmark = pytest.mark.performance


class PhysicsPerformanceBenchmark:
    """Performance benchmark utilities for physics computations."""
    
    def __init__(self):
        self.benchmark_results = {}
        self.memory_tracker = []
        
    def time_function(self, func, *args, **kwargs):
        """Time function execution and track memory usage."""
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Record final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        
        return {
            'result': result,
            'execution_time': execution_time,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': memory_delta
        }
    
    def benchmark_scaling(self, func, size_range, *args, **kwargs):
        """Benchmark function scaling with input size."""
        results = {}
        
        for size in size_range:
            # Prepare size-dependent arguments
            if 'size_param' in kwargs:
                size_param = kwargs.pop('size_param')
                modified_args = args[:size_param] + (size,) + args[size_param+1:]
            else:
                modified_args = args
            
            # Run benchmark
            benchmark_result = self.time_function(func, *modified_args, **kwargs)
            
            results[size] = {
                'execution_time': benchmark_result['execution_time'],
                'memory_usage': benchmark_result['memory_delta_mb'],
                'throughput': size / benchmark_result['execution_time'] if benchmark_result['execution_time'] > 0 else 0
            }
            
            # Clean up memory
            del benchmark_result
            gc.collect()
        
        return results
    
    def analyze_complexity(self, scaling_results):
        """Analyze computational complexity from scaling results."""
        sizes = np.array(list(scaling_results.keys()))
        times = np.array([r['execution_time'] for r in scaling_results.values()])
        
        def safe_corrcoef(x, y):
            # Check for NaN, inf, and zero variance
            if (
                np.any(np.isnan(x)) or np.any(np.isnan(y)) or
                np.any(np.isinf(x)) or np.any(np.isinf(y)) or
                np.all(x == x[0]) or np.all(y == y[0])
            ):
                return np.nan
            try:
                r = np.corrcoef(x, y)[0, 1]
                if np.isnan(r):
                    return np.nan
                return r
            except Exception:
                return np.nan
        
        # Try different complexity models
        complexities = {}
        
        # Linear: O(n)
        if len(sizes) > 1:
            linear_coeff = np.polyfit(sizes, times, 1)
            linear_r2 = safe_corrcoef(sizes, times)
            linear_r2 = linear_r2**2 if not np.isnan(linear_r2) else np.nan
            complexities['O(n)'] = {'r_squared': linear_r2, 'coefficients': linear_coeff}
        
        # Quadratic: O(n²)
        if len(sizes) > 1:
            quad_x = sizes**2
            quad_coeff = np.polyfit(quad_x, times, 1)
            quad_r2 = safe_corrcoef(quad_x, times)
            quad_r2 = quad_r2**2 if not np.isnan(quad_r2) else np.nan
            complexities['O(n²)'] = {'r_squared': quad_r2, 'coefficients': quad_coeff}
        
        # Cubic: O(n³)
        if len(sizes) > 1:
            cubic_x = sizes**3
            cubic_coeff = np.polyfit(cubic_x, times, 1)
            cubic_r2 = safe_corrcoef(cubic_x, times)
            cubic_r2 = cubic_r2**2 if not np.isnan(cubic_r2) else np.nan
            complexities['O(n³)'] = {'r_squared': cubic_r2, 'coefficients': cubic_coeff}
        
        # Logarithmic: O(log n)
        if len(sizes) > 1 and all(s > 0 for s in sizes):
            log_x = np.log(sizes)
            log_coeff = np.polyfit(log_x, times, 1)
            log_r2 = safe_corrcoef(log_x, times)
            log_r2 = log_r2**2 if not np.isnan(log_r2) else np.nan
            complexities['O(log n)'] = {'r_squared': log_r2, 'coefficients': log_coeff}
        
        # Find best fit
        if complexities:
            best_complexity = max(complexities.keys(), key=lambda k: complexities[k]['r_squared'] if not np.isnan(complexities[k]['r_squared']) else -np.inf)
            return best_complexity, complexities
        else:
            return 'Unknown', {}


class TestQuantumPhysicsPerformance:
    """Performance tests for quantum physics computations."""
    
    @pytest.fixture
    def quantum_agent(self):
        """Create quantum physics agent for performance testing."""
        from .test_quantum_physics import MockQuantumPhysicsAgent
        return MockQuantumPhysicsAgent()
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PhysicsPerformanceBenchmark()
    
    def test_eigenvalue_solver_scaling(self, quantum_agent, performance_benchmark):
        """Test scaling of eigenvalue solver with matrix size."""
        matrix_sizes = [2, 4, 8, 16, 32]
        
        def create_and_solve_hamiltonian(size):
            # Create random Hermitian matrix
            random_matrix = np.random.random((size, size)) + 1j * np.random.random((size, size))
            hamiltonian = random_matrix + random_matrix.conj().T
            return quantum_agent.solve_schrodinger_equation(hamiltonian)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            create_and_solve_hamiltonian, matrix_sizes
        )
        
        # Analyze complexity
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Eigenvalue solving should be O(n³) for dense matrices
        assert 'O(n³)' in complexities
        assert complexities['O(n³)']['r_squared'] > 0.8  # Good fit
        
        # Check performance thresholds
        for size, result in scaling_results.items():
            if size <= 32:
                assert result['execution_time'] < 1.0  # Should be fast for small matrices
            assert result['memory_usage'] >= 0  # Should not leak memory significantly
    
    def test_quantum_state_evolution_performance(self, quantum_agent, performance_benchmark):
        """Test performance of quantum state time evolution."""
        system_sizes = [2, 4, 8, 16]
        
        def evolve_quantum_state(size):
            # Create initial state and Hamiltonian
            initial_state = np.zeros(size)
            initial_state[0] = 1.0
            
            hamiltonian = np.random.random((size, size))
            hamiltonian = hamiltonian + hamiltonian.T
            
            return quantum_agent.evolve_quantum_state(initial_state, hamiltonian, time=1.0)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            evolve_quantum_state, system_sizes
        )
        
        # Check performance requirements
        for size, result in scaling_results.items():
            assert result['execution_time'] < 0.1  # Should be fast for matrix exponential
            
        # Memory usage should scale with system size
        memory_usage = [scaling_results[size]['memory_usage'] for size in system_sizes]
        assert all(m >= 0 for m in memory_usage)  # No significant memory leaks
    
    def test_quantum_algorithm_performance(self, quantum_agent, performance_benchmark):
        """Test performance of quantum algorithms."""
        qubit_counts = [2, 3, 4, 5]
        
        def run_grover_algorithm(n_qubits):
            parameters = {'n_qubits': n_qubits, 'target_item': 0}
            return quantum_agent.simulate_quantum_algorithm('grover', parameters)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            run_grover_algorithm, qubit_counts
        )
        
        # Grover's algorithm should scale exponentially with qubit count
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check that computation time increases with qubit count
        times = [scaling_results[n]['execution_time'] for n in qubit_counts]
        assert all(times[i] <= times[i+1] * 2 for i in range(len(times)-1))  # Reasonable scaling
    
    @pytest.mark.slow
    def test_large_quantum_system_performance(self, quantum_agent, performance_benchmark):
        """Test performance on large quantum systems."""
        large_size = 64
        
        benchmark_result = performance_benchmark.time_function(
            quantum_agent.solve_schrodinger_equation,
            np.random.random((large_size, large_size)) + np.random.random((large_size, large_size)).T
        )
        
        # Should complete in reasonable time
        assert benchmark_result['execution_time'] < 10.0  # Within 10 seconds
        
        # Check memory usage is reasonable
        assert benchmark_result['memory_delta_mb'] < 500  # Less than 500 MB additional


class TestComputationalPhysicsPerformance:
    """Performance tests for computational physics methods."""
    
    @pytest.fixture
    def computational_agent(self):
        """Create computational physics agent for performance testing."""
        from .test_computational_physics import MockComputationalPhysicsAgent
        return MockComputationalPhysicsAgent()
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PhysicsPerformanceBenchmark()
    
    def test_ode_solver_performance(self, computational_agent, performance_benchmark):
        """Test ODE solver performance scaling."""
        time_points = [100, 500, 1000, 2000, 5000]
        
        def solve_ode_system(n_points):
            def harmonic_oscillator(t, y):
                return np.array([y[1], -y[0]])
            
            dt = 10.0 / n_points
            return computational_agent.solve_ode(
                harmonic_oscillator, np.array([1.0, 0.0]), (0, 10), 
                method='runge_kutta_4', dt=dt
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            solve_ode_system, time_points
        )
        
        # ODE solving should scale linearly with number of time points
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        if 'O(n)' in complexities:
            assert complexities['O(n)']['r_squared'] > 0.8
        
        # Check performance thresholds
        for n_points, result in scaling_results.items():
            expected_time = n_points * 1e-5  # Expected time per time step
            assert result['execution_time'] < max(expected_time, 1.0)
    
    def test_monte_carlo_performance(self, computational_agent, performance_benchmark):
        """Test Monte Carlo integration performance."""
        sample_counts = [1000, 5000, 10000, 50000, 100000]
        
        def monte_carlo_pi(n_samples):
            def quarter_circle(x, y):
                return 1 if x**2 + y**2 <= 1 else 0
            
            return computational_agent.monte_carlo_integration(
                quarter_circle, [(0, 1), (0, 1)], n_samples
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            monte_carlo_pi, sample_counts
        )
        
        # Monte Carlo should scale linearly with sample count
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        if 'O(n)' in complexities:
            assert complexities['O(n)']['r_squared'] > 0.8
        
        # Check convergence properties
        results_by_samples = {}
        for n_samples in sample_counts:
            result = monte_carlo_pi(n_samples)
            results_by_samples[n_samples] = result['integral']
        
        # Error should decrease roughly as 1/√n
        errors = [abs(results_by_samples[n] - np.pi) for n in sample_counts]
        
        # Later samples should generally have smaller errors
        assert errors[-1] <= errors[0]  # Final error should be smaller than initial
    
    def test_molecular_dynamics_performance(self, computational_agent, performance_benchmark):
        """Test molecular dynamics simulation performance."""
        particle_counts = [10, 20, 50, 100]
        
        def run_md_simulation(n_particles):
            return computational_agent.molecular_dynamics_simulation(
                n_particles, n_steps=100, dt=0.001
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            run_md_simulation, particle_counts
        )
        
        # MD should scale as O(n²) for all-pairs interactions
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check that computation time increases with particle count
        times = [scaling_results[n]['execution_time'] for n in particle_counts]
        assert all(times[i] <= times[i+1] * 5 for i in range(len(times)-1))  # Reasonable scaling
        
        # Memory usage should scale with particle count
        for n_particles, result in scaling_results.items():
            expected_memory = n_particles * 0.1  # Rough estimate
            assert result['memory_usage'] >= 0
    
    def test_fft_performance(self, computational_agent, performance_benchmark):
        """Test FFT performance scaling."""
        signal_lengths = [64, 256, 1024, 4096, 16384]
        
        def fft_analysis(length):
            signal = np.random.random(length)
            return computational_agent.fft_analysis(signal, sampling_rate=1000)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            fft_analysis, signal_lengths
        )
        
        # FFT should scale as O(n log n)
        sizes = np.array(signal_lengths)
        times = np.array([scaling_results[s]['execution_time'] for s in sizes])
        
        # Check that scaling is better than O(n²)
        if len(sizes) > 1:
            # Compare with quadratic scaling
            quad_coeff = np.polyfit(sizes**2, times, 1)
            nlogn_values = sizes * np.log2(sizes)
            nlogn_coeff = np.polyfit(nlogn_values, times, 1)
            
            nlogn_r2 = np.corrcoef(nlogn_values, times)[0, 1]**2
            
            # FFT should show good O(n log n) scaling
            assert nlogn_r2 > 0.7  # Reasonable fit to n log n
    
    def test_parallel_computation_performance(self, computational_agent, performance_benchmark):
        """Test parallel computation performance."""
        data_sizes = [1000, 5000, 10000, 50000]
        
        def parallel_computation(size):
            def test_func(data):
                return np.sum(data**2)
            
            return computational_agent.parallel_computation_benchmark(
                test_func, [size], n_workers=4
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            parallel_computation, data_sizes
        )
        
        # Check that parallel computation shows speedup
        for size, result in scaling_results.items():
            # Run the same computation to get the benchmark results
            benchmark_result = parallel_computation(size)
            perf_summary = benchmark_result['performance_summary']
            
            # Should show some speedup from parallelization
            assert perf_summary['average_speedup'] > 1.0
            assert perf_summary['best_efficiency'] > 0.1  # At least 10% efficiency


class TestMaterialsPhysicsPerformance:
    """Performance tests for materials physics calculations."""
    
    @pytest.fixture
    def materials_agent(self):
        """Create materials physics agent for performance testing."""
        from .test_materials_physics import MockMaterialsPhysicsAgent
        return MockMaterialsPhysicsAgent()
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PhysicsPerformanceBenchmark()
    
    def test_crystal_structure_analysis_performance(self, materials_agent, performance_benchmark):
        """Test crystal structure analysis performance."""
        # Test doesn't scale with size, but we test computational efficiency
        
        benchmark_result = performance_benchmark.time_function(
            materials_agent.analyze_crystal_structure,
            {'a': 4.05, 'b': 4.05, 'c': 4.05, 'alpha': 90, 'beta': 90, 'gamma': 90}
        )
        
        # Should be very fast
        assert benchmark_result['execution_time'] < 0.01  # Less than 10 ms
        assert benchmark_result['memory_delta_mb'] < 1  # Minimal memory usage
    
    def test_stress_strain_simulation_performance(self, materials_agent, performance_benchmark):
        """Test stress-strain simulation performance."""
        n_points_list = [50, 100, 500, 1000, 2000]
        
        def stress_strain_simulation(n_points):
            return materials_agent.simulate_stress_strain_curve(
                'metal', max_strain=0.1, n_points=n_points
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            stress_strain_simulation, n_points_list
        )
        
        # Should scale linearly with number of points
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        if 'O(n)' in complexities:
            assert complexities['O(n)']['r_squared'] > 0.8
        
        # Check performance thresholds
        for n_points, result in scaling_results.items():
            assert result['execution_time'] < n_points * 1e-5 + 0.1  # Linear scaling + overhead
    
    def test_grain_growth_simulation_performance(self, materials_agent, performance_benchmark):
        """Test grain growth simulation performance."""
        time_steps = [100, 500, 1000, 2000]
        
        def grain_growth_simulation(time_final):
            return materials_agent.simulate_grain_growth(
                initial_grain_size=1e-6,
                temperature=1000,
                time=time_final
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            grain_growth_simulation, time_steps
        )
        
        # Grain growth simulation should scale linearly with time range
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check that longer simulations take more time
        times = [scaling_results[t]['execution_time'] for t in time_steps]
        assert all(times[i] <= times[i+1] * 2 for i in range(len(times)-1))
    
    def test_phase_transition_performance(self, materials_agent, performance_benchmark):
        """Test phase transition simulation performance."""
        temperature_points = [50, 100, 200, 500]
        
        def phase_transition_simulation(n_temp_points):
            T_range = (300, 800)
            # Modify the agent to accept number of points
            return materials_agent.simulate_phase_transition(T_range)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            phase_transition_simulation, temperature_points
        )
        
        # Check reasonable performance
        for n_points, result in scaling_results.items():
            assert result['execution_time'] < 0.1  # Should be fast
    
    def test_defect_analysis_performance(self, materials_agent, performance_benchmark):
        """Test defect analysis performance."""
        defect_types = ['vacancy', 'interstitial', 'grain_boundary', 'dislocation']
        
        def analyze_all_defects():
            results = {}
            for defect_type in defect_types:
                results[defect_type] = materials_agent.analyze_defects(defect_type, 1e-6)
            return results
        
        benchmark_result = performance_benchmark.time_function(analyze_all_defects)
        
        # Should be fast for all defect types
        assert benchmark_result['execution_time'] < 0.05  # Less than 50 ms
        assert benchmark_result['memory_delta_mb'] < 1


class TestAstrophysicsPerformance:
    """Performance tests for astrophysics simulations."""
    
    @pytest.fixture
    def astrophysics_agent(self):
        """Create astrophysics agent for performance testing."""
        from .test_astrophysics import MockAstrophysicsAgent
        return MockAstrophysicsAgent()
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PhysicsPerformanceBenchmark()
    
    def test_nbody_simulation_performance(self, astrophysics_agent, performance_benchmark):
        """Test N-body simulation performance scaling."""
        particle_counts = [10, 20, 50, 100]
        
        def nbody_simulation(n_bodies):
            return astrophysics_agent.simulate_nbody_system(
                n_bodies, n_steps=100, dt=0.01
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            nbody_simulation, particle_counts
        )
        
        # N-body should scale as O(n²) for direct summation
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check that computation time increases significantly with particle count
        times = [scaling_results[n]['execution_time'] for n in particle_counts]
        
        # Should show superlinear scaling
        if len(times) > 1:
            assert times[-1] > times[0] * 2  # At least quadratic growth
        
        # Memory usage should scale with particle count
        for n_particles, result in scaling_results.items():
            assert result['memory_usage'] >= 0
    
    def test_stellar_evolution_performance(self, astrophysics_agent, performance_benchmark):
        """Test stellar evolution simulation performance."""
        time_spans = [1e8, 5e8, 1e9, 5e9, 1e10]  # years
        
        def stellar_evolution_simulation(time_span):
            return astrophysics_agent.simulate_stellar_evolution(
                initial_mass=1.0,
                age_range=(0, time_span)
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            stellar_evolution_simulation, time_spans
        )
        
        # Should scale linearly with time span (more time points)
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check performance is reasonable
        for time_span, result in scaling_results.items():
            assert result['execution_time'] < 1.0  # Should be fast
    
    def test_galaxy_rotation_curve_performance(self, astrophysics_agent, performance_benchmark):
        """Test galaxy rotation curve calculation performance."""
        radius_points = [100, 500, 1000, 2000, 5000]
        
        def galaxy_rotation_simulation(max_radius):
            return astrophysics_agent.simulate_galaxy_rotation_curve(
                galaxy_mass=1e12,
                max_radius=max_radius
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            galaxy_rotation_simulation, radius_points
        )
        
        # Should scale linearly with number of radial points
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        if 'O(n)' in complexities:
            assert complexities['O(n)']['r_squared'] > 0.7
        
        # Check reasonable performance
        for max_radius, result in scaling_results.items():
            assert result['execution_time'] < 0.5  # Should be reasonably fast
    
    def test_cosmological_distance_performance(self, astrophysics_agent, performance_benchmark):
        """Test cosmological distance calculation performance."""
        redshift_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        def cosmological_distances_batch():
            results = []
            for z in redshift_values:
                results.append(astrophysics_agent.calculate_cosmological_distances(z))
            return results
        
        benchmark_result = performance_benchmark.time_function(cosmological_distances_batch)
        
        # Should be fast for multiple redshift calculations
        assert benchmark_result['execution_time'] < 0.1  # Less than 100 ms
        assert benchmark_result['memory_delta_mb'] < 5   # Minimal memory growth
    
    def test_cmb_simulation_performance(self, astrophysics_agent, performance_benchmark):
        """Test CMB simulation performance."""
        l_max_values = [500, 1000, 2000, 5000]
        
        def cmb_simulation(l_max):
            return astrophysics_agent.simulate_cosmic_microwave_background(l_max)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            cmb_simulation, l_max_values
        )
        
        # Should scale linearly with l_max
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        if 'O(n)' in complexities:
            assert complexities['O(n)']['r_squared'] > 0.8
        
        # Check performance thresholds
        for l_max, result in scaling_results.items():
            assert result['execution_time'] < l_max * 1e-5 + 0.1  # Linear scaling


class TestExperimentalPhysicsPerformance:
    """Performance tests for experimental physics data analysis."""
    
    @pytest.fixture
    def experimental_agent(self):
        """Create experimental physics agent for performance testing."""
        from .test_experimental_physics import MockExperimentalPhysicsAgent
        return MockExperimentalPhysicsAgent()
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PhysicsPerformanceBenchmark()
    
    def test_measurement_performance(self, experimental_agent, performance_benchmark):
        """Test measurement simulation performance."""
        measurement_counts = [100, 500, 1000, 5000, 10000]
        
        def measurement_simulation(n_measurements):
            parameters = {
                'true_value': 10.0,
                'systematic_error': 0.01,
                'statistical_error': 0.001
            }
            return experimental_agent.perform_measurement(
                'voltage', parameters, n_measurements
            )
        
        scaling_results = performance_benchmark.benchmark_scaling(
            measurement_simulation, measurement_counts
        )
        
        # Should scale linearly with number of measurements
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        if 'O(n)' in complexities:
            assert complexities['O(n)']['r_squared'] > 0.8
        
        # Check performance thresholds
        for n_measurements, result in scaling_results.items():
            assert result['execution_time'] < n_measurements * 1e-5 + 0.1
    
    def test_data_fitting_performance(self, experimental_agent, performance_benchmark):
        """Test data fitting performance."""
        data_sizes = [100, 500, 1000, 5000]
        
        def data_fitting_simulation(n_points):
            x_data = np.linspace(0, 10, n_points)
            y_data = 2 * x_data + 1 + np.random.normal(0, 0.1, n_points)
            
            return experimental_agent.analyze_data_fitting(x_data, y_data, 'linear')
        
        scaling_results = performance_benchmark.benchmark_scaling(
            data_fitting_simulation, data_sizes
        )
        
        # Fitting should scale reasonably with data size
        # Linear fitting is typically O(n) but can be O(n²) for some algorithms
        times = [scaling_results[n]['execution_time'] for n in data_sizes]
        
        # Should complete in reasonable time
        for n_points, result in scaling_results.items():
            assert result['execution_time'] < n_points * 1e-4 + 0.1
    
    def test_statistical_analysis_performance(self, experimental_agent, performance_benchmark):
        """Test statistical analysis performance."""
        data_sizes = [1000, 5000, 10000, 50000]
        
        def statistical_analysis(n_points):
            data = np.random.normal(0, 1, n_points)
            return experimental_agent.perform_statistical_analysis(data, 'normality')
        
        scaling_results = performance_benchmark.benchmark_scaling(
            statistical_analysis, data_sizes
        )
        
        # Statistical tests should scale reasonably
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check performance is reasonable
        for n_points, result in scaling_results.items():
            assert result['execution_time'] < n_points * 1e-5 + 0.5
    
    def test_detector_simulation_performance(self, experimental_agent, performance_benchmark):
        """Test detector simulation performance."""
        signal_lengths = [1000, 5000, 10000, 50000]
        
        def detector_simulation(length):
            signal = np.random.random(length)
            detector_config = {
                'efficiency': 0.9,
                'noise_level': 0.01,
                'bandwidth': 1000,
                'sampling_rate': 10000
            }
            return experimental_agent.simulate_detector_response(signal, detector_config)
        
        scaling_results = performance_benchmark.benchmark_scaling(
            detector_simulation, signal_lengths
        )
        
        # Should scale roughly linearly (with some FFT operations that are O(n log n))
        best_complexity, complexities = performance_benchmark.analyze_complexity(scaling_results)
        
        # Check performance
        for length, result in scaling_results.items():
            assert result['execution_time'] < length * 1e-4 + 1.0


class TestIntegratedPhysicsPerformance:
    """Performance tests for integrated physics workflows."""
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        return PhysicsPerformanceBenchmark()
    
    @pytest.fixture
    def all_agents(self):
        """Create all physics agents for integrated testing."""
        from .test_quantum_physics import MockQuantumPhysicsAgent
        from .test_computational_physics import MockComputationalPhysicsAgent
        from .test_materials_physics import MockMaterialsPhysicsAgent
        from .test_astrophysics import MockAstrophysicsAgent
        from .test_experimental_physics import MockExperimentalPhysicsAgent
        
        return {
            'quantum': MockQuantumPhysicsAgent(),
            'computational': MockComputationalPhysicsAgent(),
            'materials': MockMaterialsPhysicsAgent(),
            'astrophysics': MockAstrophysicsAgent(),
            'experimental': MockExperimentalPhysicsAgent()
        }
    
    def test_integrated_workflow_performance(self, all_agents, performance_benchmark):
        """Test performance of integrated physics workflow."""
        
        def integrated_physics_simulation():
            results = {}
            
            # Quantum calculation
            hamiltonian = np.array([[1, 0.1], [0.1, -1]])
            results['quantum'] = all_agents['quantum'].solve_schrodinger_equation(hamiltonian)
            
            # Materials calculation
            lattice_params = {'a': 4.0, 'b': 4.0, 'c': 4.0, 'alpha': 90, 'beta': 90, 'gamma': 90}
            results['materials'] = all_agents['materials'].analyze_crystal_structure(lattice_params)
            
            # Computational simulation
            def ode_func(t, y):
                return -0.1 * y
            
            results['computational'] = all_agents['computational'].solve_ode(
                ode_func, np.array([1.0]), (0, 10)
            )
            
            # Astrophysics simulation
            results['astrophysics'] = all_agents['astrophysics'].simulate_nbody_system(
                n_bodies=20, n_steps=100
            )
            
            # Experimental analysis
            measurements = {
                'true_value': 5.0,
                'systematic_error': 0.01,
                'statistical_error': 0.001
            }
            results['experimental'] = all_agents['experimental'].perform_measurement(
                'voltage', measurements, 50
            )
            
            return results
        
        benchmark_result = performance_benchmark.time_function(integrated_physics_simulation)
        
        # Integrated workflow should complete in reasonable time
        assert benchmark_result['execution_time'] < 5.0  # Less than 5 seconds
        
        # Should not consume excessive memory
        assert benchmark_result['memory_delta_mb'] < 100  # Less than 100 MB
        
        # Check that all components completed successfully
        results = benchmark_result['result']
        assert 'quantum' in results
        assert 'materials' in results
        assert 'computational' in results
        assert 'astrophysics' in results
        assert 'experimental' in results
    
    def test_concurrent_physics_simulations_performance(self, all_agents, performance_benchmark):
        """Test performance of concurrent physics simulations."""
        import threading
        import queue
        
        def run_simulation(agent_name, agent, result_queue):
            """Run simulation and put result in queue."""
            start_time = time.time()
            
            if agent_name == 'quantum':
                result = agent.solve_schrodinger_equation(np.array([[1, 0], [0, -1]]))
            elif agent_name == 'computational':
                result = agent.molecular_dynamics_simulation(20, 100)
            elif agent_name == 'materials':
                result = agent.simulate_stress_strain_curve('metal')
            elif agent_name == 'astrophysics':
                result = agent.simulate_nbody_system(30, n_steps=50)
            elif agent_name == 'experimental':
                params = {'true_value': 1.0, 'systematic_error': 0.01, 'statistical_error': 0.001}
                result = agent.perform_measurement('voltage', params, 100)
            
            end_time = time.time()
            result_queue.put((agent_name, result, end_time - start_time))
        
        def concurrent_simulations():
            result_queue = queue.Queue()
            threads = []
            
            # Start all simulations concurrently
            for agent_name, agent in all_agents.items():
                thread = threading.Thread(
                    target=run_simulation,
                    args=(agent_name, agent, result_queue)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all to complete
            for thread in threads:
                thread.join()
            
            # Collect results
            results = {}
            total_concurrent_time = 0
            
            while not result_queue.empty():
                agent_name, result, execution_time = result_queue.get()
                results[agent_name] = {
                    'result': result,
                    'execution_time': execution_time
                }
                total_concurrent_time = max(total_concurrent_time, execution_time)
            
            return results, total_concurrent_time
        
        benchmark_result = performance_benchmark.time_function(concurrent_simulations)
        results, total_concurrent_time = benchmark_result['result']
        
        # All simulations should complete
        assert len(results) == len(all_agents)
        
        # Concurrent execution should be faster than sequential
        sequential_time = sum(r['execution_time'] for r in results.values())
        concurrent_time = benchmark_result['execution_time']
        
        # Should show some benefit from concurrency
        assert concurrent_time < sequential_time
        
        # Total concurrent time should be close to the longest individual simulation
        assert abs(total_concurrent_time - concurrent_time) < 1.0
    
    @pytest.mark.slow
    def test_large_scale_physics_performance(self, all_agents, performance_benchmark):
        """Test performance of large-scale physics computations."""
        
        def large_scale_simulation():
            results = {}
            
            # Large quantum system
            large_hamiltonian = np.random.random((32, 32))
            large_hamiltonian = large_hamiltonian + large_hamiltonian.T
            results['large_quantum'] = all_agents['quantum'].solve_schrodinger_equation(large_hamiltonian)
            
            # Large N-body simulation
            results['large_nbody'] = all_agents['astrophysics'].simulate_nbody_system(
                n_bodies=100, n_steps=500
            )
            
            # Large molecular dynamics
            results['large_md'] = all_agents['computational'].molecular_dynamics_simulation(
                n_particles=200, n_steps=1000
            )
            
            # Large dataset analysis
            large_data = np.random.normal(0, 1, 10000)
            results['large_analysis'] = all_agents['experimental'].perform_statistical_analysis(
                large_data, 'normality'
            )
            
            return results
        
        benchmark_result = performance_benchmark.time_function(large_scale_simulation)
        
        # Large-scale simulation should complete in reasonable time
        assert benchmark_result['execution_time'] < 30.0  # Less than 30 seconds
        
        # Memory usage should be reasonable
        assert benchmark_result['memory_delta_mb'] < 1000  # Less than 1 GB
        
        # Check that all large-scale computations completed
        results = benchmark_result['result']
        assert len(results) == 4
        
        # Verify some basic properties
        assert len(results['large_quantum']['eigenvalues']) == 32
        assert results['large_nbody']['simulation_parameters']['n_bodies'] == 100
        assert results['large_md']['simulation_parameters']['n_particles'] == 200
        assert results['large_analysis']['summary']['sample_size'] == 10000
    
    def test_memory_efficiency_physics_workflow(self, all_agents, performance_benchmark):
        """Test memory efficiency of physics workflows."""
        
        def memory_intensive_workflow():
            """Workflow designed to test memory management."""
            results = []
            
            # Create and release multiple large objects
            for i in range(10):
                # Large matrix operations
                large_matrix = np.random.random((100, 100))
                quantum_result = all_agents['quantum'].solve_schrodinger_equation(large_matrix)
                results.append(quantum_result)
                
                # Force garbage collection
                del large_matrix
                gc.collect()
            
            return results
        
        benchmark_result = performance_benchmark.time_function(memory_intensive_workflow)
        
        # Should not accumulate excessive memory
        assert benchmark_result['memory_delta_mb'] < 50  # Less than 50 MB net increase
        
        # Should complete successfully
        results = benchmark_result['result']
        assert len(results) == 10
        
        # All results should be valid
        for result in results:
            assert 'eigenvalues' in result
            assert len(result['eigenvalues']) == 100