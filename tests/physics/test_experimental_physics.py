"""
Experimental Physics Tests

Comprehensive unit tests for experimental physics methods and data analysis.
Tests measurement techniques, data analysis, calibration, and statistical methods.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Callable
from unittest.mock import Mock, patch
import scipy.stats as stats
from scipy.optimize import curve_fit

# Experimental physics test markers
pytestmark = pytest.mark.experimental


class MockExperimentalPhysicsAgent:
    """Mock experimental physics agent for testing."""
    
    def __init__(self):
        self.name = "ExperimentalPhysicsAgent"
        self.initialized = True
        self.calibration_database = {}
        self.measurement_history = []
    
    def perform_measurement(self, measurement_type: str, parameters: Dict[str, Any],
                          n_measurements: int = 100) -> Dict[str, Any]:
        """Simulate experimental measurements with realistic noise and uncertainties."""
        np.random.seed(42)  # For reproducible tests
        
        if measurement_type == 'length':
            # Length measurement with systematic and statistical errors
            true_value = parameters.get('true_value', 1.0)  # meters
            systematic_error = parameters.get('systematic_error', 0.001)
            statistical_error = parameters.get('statistical_error', 0.0001)
            
            # Simulate measurements
            measurements = np.random.normal(
                true_value + systematic_error,
                statistical_error,
                n_measurements
            )
            
        elif measurement_type == 'voltage':
            # Voltage measurement with drift and noise
            true_value = parameters.get('true_value', 5.0)  # volts
            drift_rate = parameters.get('drift_rate', 0.001)  # V/measurement
            noise_level = parameters.get('noise_level', 0.01)  # V RMS
            
            measurements = np.zeros(n_measurements)
            for i in range(n_measurements):
                drift = drift_rate * i
                noise = np.random.normal(0, noise_level)
                measurements[i] = true_value + drift + noise
                
        elif measurement_type == 'frequency':
            # Frequency measurement with 1/f noise
            true_value = parameters.get('true_value', 1000.0)  # Hz
            relative_stability = parameters.get('relative_stability', 1e-6)
            
            # Generate 1/f noise
            frequencies = np.fft.fftfreq(n_measurements)[1:n_measurements//2]
            psd = 1.0 / frequencies
            phases = np.random.uniform(0, 2*np.pi, len(frequencies))
            
            # Create noise time series
            noise_spectrum = np.sqrt(psd) * np.exp(1j * phases)
            noise_full = np.concatenate([np.array([0]), noise_spectrum, 
                                       np.conj(noise_spectrum[::-1])])
            if n_measurements % 2 == 0:
                noise_full = noise_full[:-1]
            
            noise_time = np.real(np.fft.ifft(noise_full))
            noise_time *= relative_stability * true_value / np.std(noise_time)
            
            measurements = true_value + noise_time
            
        elif measurement_type == 'temperature':
            # Temperature measurement with calibration uncertainty
            true_value = parameters.get('true_value', 300.0)  # K
            calibration_uncertainty = parameters.get('calibration_uncertainty', 0.1)
            readout_noise = parameters.get('readout_noise', 0.01)
            
            calibration_offset = np.random.normal(0, calibration_uncertainty)
            measurements = np.random.normal(
                true_value + calibration_offset,
                readout_noise,
                n_measurements
            )
            
        else:
            # Generic measurement
            true_value = parameters.get('true_value', 1.0)
            uncertainty = parameters.get('uncertainty', 0.01)
            
            measurements = np.random.normal(true_value, uncertainty, n_measurements)
        
        # Calculate statistics
        mean_value = np.mean(measurements)
        std_dev = np.std(measurements, ddof=1)
        std_error = std_dev / np.sqrt(n_measurements)
        
        # Chi-squared test for normality
        chi2_stat, p_value = self._chi_squared_test(measurements)
        
        # Store measurement in history
        measurement_record = {
            'type': measurement_type,
            'parameters': parameters,
            'n_measurements': n_measurements,
            'mean': mean_value,
            'std_dev': std_dev,
            'std_error': std_error
        }
        self.measurement_history.append(measurement_record)
        
        return {
            'measurements': measurements,
            'mean': mean_value,
            'standard_deviation': std_dev,
            'standard_error': std_error,
            'measurement_type': measurement_type,
            'parameters': parameters,
            'n_measurements': n_measurements,
            'chi_squared_stat': chi2_stat,
            'p_value': p_value,
            'confidence_interval_95': (
                mean_value - 1.96 * std_error,
                mean_value + 1.96 * std_error
            )
        }
    
    def calibrate_instrument(self, instrument_type: str, calibration_standards: List[Dict],
                           measurement_data: np.ndarray) -> Dict[str, Any]:
        """Perform instrument calibration using reference standards."""
        if len(calibration_standards) != len(measurement_data):
            raise ValueError("Number of standards must match number of measurements")
        
        # Extract true values and measured values
        true_values = np.array([std['true_value'] for std in calibration_standards])
        measured_values = measurement_data
        uncertainties = np.array([std.get('uncertainty', 0.01) for std in calibration_standards])
        
        # Fit calibration curve (linear by default)
        try:
            # Weighted linear fit
            weights = 1.0 / uncertainties**2
            fit_params, fit_covariance = np.polyfit(measured_values, true_values, 1, 
                                                   w=weights, cov=True)
            slope, intercept = fit_params
            
            # Calculate R-squared
            predicted_values = slope * measured_values + intercept
            ss_res = np.sum((true_values - predicted_values)**2)
            ss_tot = np.sum((true_values - np.mean(true_values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate calibration function
            def calibration_function(raw_measurement):
                return slope * raw_measurement + intercept
            
            # Estimate calibration uncertainty
            calibration_uncertainty = np.sqrt(np.diagonal(fit_covariance))
            
            # Store calibration in database
            self.calibration_database[instrument_type] = {
                'slope': slope,
                'intercept': intercept,
                'uncertainty': calibration_uncertainty,
                'r_squared': r_squared,
                'calibration_date': 'mock_date',
                'calibration_function': calibration_function
            }
            
        except Exception as e:
            # Fallback to simple linear calibration
            slope = 1.0
            intercept = 0.0
            r_squared = 0.0
            calibration_uncertainty = [0.01, 0.01]
            
            def calibration_function(raw_measurement):
                return raw_measurement
        
        return {
            'instrument_type': instrument_type,
            'calibration_parameters': {
                'slope': slope,
                'intercept': intercept
            },
            'calibration_uncertainty': calibration_uncertainty,
            'r_squared': r_squared,
            'residuals': true_values - (slope * measured_values + intercept),
            'calibration_function': calibration_function,
            'calibration_valid': r_squared > 0.95
        }
    
    def analyze_data_fitting(self, x_data: np.ndarray, y_data: np.ndarray,
                           fit_function: str = 'linear', 
                           weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform data fitting with various functional forms."""
        
        # Define fitting functions
        def linear_func(x, a, b):
            return a * x + b
        
        def quadratic_func(x, a, b, c):
            return a * x**2 + b * x + c
        
        def exponential_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        def gaussian_func(x, a, b, c):
            return a * np.exp(-0.5 * ((x - b) / c)**2)
        
        def power_law_func(x, a, b):
            return a * x**b
        
        # Select function
        function_map = {
            'linear': (linear_func, 2),
            'quadratic': (quadratic_func, 3),
            'exponential': (exponential_func, 3),
            'gaussian': (gaussian_func, 3),
            'power_law': (power_law_func, 2)
        }
        
        if fit_function not in function_map:
            raise ValueError(f"Unknown fit function: {fit_function}")
        
        func, n_params = function_map[fit_function]
        
        try:
            # Initial parameter guess
            if fit_function == 'linear':
                p0 = [1.0, 0.0]
            elif fit_function == 'quadratic':
                p0 = [0.0, 1.0, 0.0]
            elif fit_function == 'exponential':
                p0 = [1.0, 0.1, 0.0]
            elif fit_function == 'gaussian':
                p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
            elif fit_function == 'power_law':
                p0 = [1.0, 1.0]
            
            # Perform fit
            if weights is not None:
                sigma = 1.0 / weights
            else:
                sigma = None
            
            popt, pcov = curve_fit(func, x_data, y_data, p0=p0, sigma=sigma)
            
            # Calculate fit quality metrics
            y_fit = func(x_data, *popt)
            residuals = y_data - y_fit
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Reduced chi-squared
            dof = len(y_data) - n_params
            if weights is not None:
                chi_squared = np.sum((residuals * weights)**2)
            else:
                chi_squared = ss_res
            
            reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf
            
            # Parameter uncertainties
            param_uncertainties = np.sqrt(np.diagonal(pcov))
            
            # AIC and BIC
            aic = 2 * n_params + len(y_data) * np.log(ss_res / len(y_data))
            bic = np.log(len(y_data)) * n_params + len(y_data) * np.log(ss_res / len(y_data))
            
        except Exception as e:
            # Fallback to failed fit
            popt = np.zeros(n_params)
            param_uncertainties = np.ones(n_params)
            y_fit = np.zeros_like(y_data)
            residuals = y_data
            r_squared = 0.0
            reduced_chi_squared = np.inf
            aic = np.inf
            bic = np.inf
        
        return {
            'fit_function': fit_function,
            'parameters': popt,
            'parameter_uncertainties': param_uncertainties,
            'covariance_matrix': pcov if 'pcov' in locals() else np.eye(n_params),
            'fitted_values': y_fit,
            'residuals': residuals,
            'r_squared': r_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'aic': aic,
            'bic': bic,
            'degrees_of_freedom': dof if 'dof' in locals() else 0,
            'fit_successful': r_squared > 0.5
        }
    
    def perform_statistical_analysis(self, data: np.ndarray, 
                                   test_type: str = 'normality') -> Dict[str, Any]:
        """Perform various statistical tests on experimental data."""
        
        if test_type == 'normality':
            # Shapiro-Wilk test for normality
            statistic, p_value = stats.shapiro(data)
            
            # Anderson-Darling test
            ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(data)
            
            # Kolmogorov-Smirnov test against normal distribution
            ks_statistic, ks_p_value = stats.kstest(data, 'norm', 
                                                   args=(np.mean(data), np.std(data)))
            
            return {
                'test_type': test_type,
                'shapiro_wilk': {
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                },
                'anderson_darling': {
                    'statistic': ad_statistic,
                    'critical_values': ad_critical_values,
                    'significance_levels': ad_significance_levels
                },
                'kolmogorov_smirnov': {
                    'statistic': ks_statistic,
                    'p_value': ks_p_value,
                    'is_normal': ks_p_value > 0.05
                },
                'summary': {
                    'likely_normal': p_value > 0.05 and ks_p_value > 0.05,
                    'sample_size': len(data),
                    'mean': np.mean(data),
                    'std_dev': np.std(data)
                }
            }
        
        elif test_type == 'outliers':
            # Modified Z-score for outlier detection
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            
            # IQR method
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_modified_z = np.abs(modified_z_scores) > 3.5
            outliers_iqr = (data < lower_bound) | (data > upper_bound)
            
            return {
                'test_type': test_type,
                'modified_z_score': {
                    'scores': modified_z_scores,
                    'outliers': outliers_modified_z,
                    'n_outliers': np.sum(outliers_modified_z)
                },
                'iqr_method': {
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers': outliers_iqr,
                    'n_outliers': np.sum(outliers_iqr)
                },
                'summary': {
                    'total_outliers': np.sum(outliers_modified_z | outliers_iqr),
                    'outlier_fraction': np.sum(outliers_modified_z | outliers_iqr) / len(data)
                }
            }
        
        elif test_type == 'correlation':
            # Requires paired data - reshape if 1D
            if data.ndim == 1:
                # Split data in half for correlation test
                n_half = len(data) // 2
                x = data[:n_half]
                y = data[n_half:2*n_half]
            else:
                x, y = data[:, 0], data[:, 1]
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y)
            
            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            # Kendall tau
            kendall_tau, kendall_p = stats.kendalltau(x, y)
            
            return {
                'test_type': test_type,
                'pearson': {
                    'correlation': pearson_r,
                    'p_value': pearson_p,
                    'significant': pearson_p < 0.05
                },
                'spearman': {
                    'correlation': spearman_r,
                    'p_value': spearman_p,
                    'significant': spearman_p < 0.05
                },
                'kendall': {
                    'tau': kendall_tau,
                    'p_value': kendall_p,
                    'significant': kendall_p < 0.05
                },
                'summary': {
                    'strongest_correlation': max([abs(pearson_r), abs(spearman_r), abs(kendall_tau)]),
                    'correlation_type': 'pearson' if abs(pearson_r) == max([abs(pearson_r), abs(spearman_r), abs(kendall_tau)]) else 'spearman'
                }
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def calculate_measurement_uncertainty(self, measurements: List[Dict[str, Any]],
                                        correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate combined measurement uncertainty using GUM methodology."""
        
        n_measurements = len(measurements)
        
        # Extract values and uncertainties
        values = np.array([m['value'] for m in measurements])
        uncertainties = np.array([m['uncertainty'] for m in measurements])
        
        # Type A uncertainty (statistical)
        mean_value = np.mean(values)
        type_a_uncertainty = np.std(values, ddof=1) / np.sqrt(n_measurements)
        
        # Type B uncertainty (systematic)
        type_b_uncertainty = np.sqrt(np.mean(uncertainties**2))
        
        # Combined uncertainty
        if correlation_matrix is not None:
            # Account for correlations
            combined_variance = type_a_uncertainty**2
            for i in range(n_measurements):
                for j in range(n_measurements):
                    if i != j:
                        combined_variance += correlation_matrix[i, j] * uncertainties[i] * uncertainties[j]
            combined_uncertainty = np.sqrt(combined_variance + type_b_uncertainty**2)
        else:
            # Uncorrelated case
            combined_uncertainty = np.sqrt(type_a_uncertainty**2 + type_b_uncertainty**2)
        
        # Expanded uncertainty (k=2 for 95% confidence)
        k_factor = 2.0
        expanded_uncertainty = k_factor * combined_uncertainty
        
        # Degrees of freedom (Welch-Satterthwaite formula)
        if type_a_uncertainty > 0:
            nu_eff = combined_uncertainty**4 / (type_a_uncertainty**4 / (n_measurements - 1))
            if type_b_uncertainty > 0:
                nu_eff = combined_uncertainty**4 / (
                    type_a_uncertainty**4 / (n_measurements - 1) + type_b_uncertainty**4 / np.inf
                )
        else:
            nu_eff = np.inf
        
        return {
            'mean_value': mean_value,
            'type_a_uncertainty': type_a_uncertainty,
            'type_b_uncertainty': type_b_uncertainty,
            'combined_uncertainty': combined_uncertainty,
            'expanded_uncertainty': expanded_uncertainty,
            'coverage_factor': k_factor,
            'effective_degrees_of_freedom': nu_eff,
            'relative_uncertainty': combined_uncertainty / abs(mean_value) if mean_value != 0 else np.inf,
            'uncertainty_budget': {
                'statistical_contribution': (type_a_uncertainty / combined_uncertainty)**2,
                'systematic_contribution': (type_b_uncertainty / combined_uncertainty)**2
            }
        }
    
    def simulate_detector_response(self, signal: np.ndarray, detector_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate realistic detector response including noise and bandwidth limitations."""
        
        # Detector parameters
        efficiency = detector_config.get('efficiency', 0.9)
        noise_level = detector_config.get('noise_level', 0.01)
        bandwidth = detector_config.get('bandwidth', 1000.0)  # Hz
        sampling_rate = detector_config.get('sampling_rate', 10000.0)  # Hz
        deadtime = detector_config.get('deadtime', 1e-6)  # seconds
        
        # Apply efficiency
        detected_signal = signal * efficiency
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(signal))
        noisy_signal = detected_signal + noise
        
        # Apply bandwidth limitation (low-pass filter)
        from scipy import signal as scipy_signal
        nyquist = sampling_rate / 2
        if bandwidth < nyquist:
            sos = scipy_signal.butter(4, bandwidth / nyquist, btype='low', output='sos')
            filtered_signal = scipy_signal.sosfilt(sos, noisy_signal)
        else:
            filtered_signal = noisy_signal
        
        # Apply deadtime effects (simplified)
        if deadtime > 0:
            dt = 1.0 / sampling_rate
            deadtime_samples = int(deadtime / dt)
            
            # Find peaks and apply deadtime
            peaks = np.where(np.diff(np.sign(np.diff(filtered_signal))) < 0)[0] + 1
            
            deadtime_signal = filtered_signal.copy()
            for peak in peaks:
                start_dead = max(0, peak)
                end_dead = min(len(deadtime_signal), peak + deadtime_samples)
                # Reduce signal during deadtime
                deadtime_signal[start_dead:end_dead] *= 0.1
        else:
            deadtime_signal = filtered_signal
        
        # Calculate signal-to-noise ratio
        signal_power = np.var(detected_signal)
        noise_power = np.var(noise)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        # Resolution calculation
        resolution = noise_level / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else np.inf
        
        return {
            'original_signal': signal,
            'detected_signal': detected_signal,
            'noisy_signal': noisy_signal,
            'filtered_signal': filtered_signal,
            'final_signal': deadtime_signal,
            'detector_config': detector_config,
            'performance_metrics': {
                'snr_db': snr,
                'resolution': resolution,
                'efficiency': efficiency,
                'noise_level': noise_level
            },
            'signal_statistics': {
                'mean': np.mean(deadtime_signal),
                'rms': np.sqrt(np.mean(deadtime_signal**2)),
                'peak_to_peak': np.max(deadtime_signal) - np.min(deadtime_signal)
            }
        }
    
    def _chi_squared_test(self, data: np.ndarray, bins: int = 10) -> Tuple[float, float]:
        """Perform chi-squared goodness of fit test."""
        observed, bin_edges = np.histogram(data, bins=bins)
        
        # Expected frequencies for normal distribution
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean, std = np.mean(data), np.std(data)
        
        expected = len(data) * (
            stats.norm.cdf(bin_edges[1:], mean, std) - 
            stats.norm.cdf(bin_edges[:-1], mean, std)
        )
        
        # Avoid division by zero
        expected = np.maximum(expected, 1)
        
        chi2_stat = np.sum((observed - expected)**2 / expected)
        dof = bins - 3  # bins - parameters - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof) if dof > 0 else 1.0
        
        return chi2_stat, p_value


class TestExperimentalPhysicsAgent:
    """Test class for experimental physics functionality."""
    
    @pytest.fixture
    def experimental_agent(self):
        """Create an experimental physics agent instance for testing."""
        return MockExperimentalPhysicsAgent()
    
    def test_agent_initialization(self, experimental_agent):
        """Test experimental physics agent initialization."""
        assert experimental_agent.name == "ExperimentalPhysicsAgent"
        assert experimental_agent.initialized is True
        assert hasattr(experimental_agent, 'calibration_database')
        assert hasattr(experimental_agent, 'measurement_history')
        assert len(experimental_agent.measurement_history) == 0
    
    @pytest.mark.parametrize("measurement_type", ['length', 'voltage', 'frequency', 'temperature'])
    def test_measurement_simulation(self, experimental_agent, measurement_type):
        """Test measurement simulation for different measurement types."""
        parameters = {
            'true_value': 10.0,
            'systematic_error': 0.01,
            'statistical_error': 0.001
        }
        
        n_measurements = 50
        
        result = experimental_agent.perform_measurement(measurement_type, parameters, n_measurements)
        
        assert 'measurements' in result
        assert 'mean' in result
        assert 'standard_deviation' in result
        assert 'standard_error' in result
        assert 'confidence_interval_95' in result
        
        measurements = result['measurements']
        assert len(measurements) == n_measurements
        
        # Check statistical properties
        mean_value = result['mean']
        std_error = result['standard_error']
        
        # Mean should be close to true value (within systematic error)
        true_value = parameters['true_value']
        assert abs(mean_value - true_value) < 0.1  # Reasonable tolerance
        
        # Standard error should decrease with sample size
        expected_std_error = result['standard_deviation'] / np.sqrt(n_measurements)
        assert abs(std_error - expected_std_error) < 1e-10
        
        # Confidence interval should contain the mean
        ci_lower, ci_upper = result['confidence_interval_95']
        assert ci_lower <= mean_value <= ci_upper
        
        # Check that measurement is recorded in history
        assert len(experimental_agent.measurement_history) == 1
        assert experimental_agent.measurement_history[0]['type'] == measurement_type
    
    def test_instrument_calibration(self, experimental_agent, physics_test_config):
        """Test instrument calibration process."""
        # Create calibration standards
        calibration_standards = [
            {'true_value': 1.0, 'uncertainty': 0.001},
            {'true_value': 2.0, 'uncertainty': 0.001},
            {'true_value': 3.0, 'uncertainty': 0.001},
            {'true_value': 4.0, 'uncertainty': 0.001},
            {'true_value': 5.0, 'uncertainty': 0.001}
        ]
        
        # Simulate measurements with known offset and scale
        true_values = np.array([std['true_value'] for std in calibration_standards])
        # Add systematic error: measured = 0.95 * true + 0.1
        measured_values = 0.95 * true_values + 0.1 + np.random.normal(0, 0.01, len(true_values))
        
        instrument_type = 'voltmeter'
        
        result = experimental_agent.calibrate_instrument(instrument_type, calibration_standards, measured_values)
        
        assert 'instrument_type' in result
        assert 'calibration_parameters' in result
        assert 'r_squared' in result
        assert 'calibration_function' in result
        assert 'calibration_valid' in result
        
        # Check calibration parameters
        slope = result['calibration_parameters']['slope']
        intercept = result['calibration_parameters']['intercept']
        
        # Should recover the inverse of the systematic error
        assert abs(slope - 1/0.95) < 0.1  # Within 10%
        assert abs(intercept - (-0.1/0.95)) < 0.1
        
        # Check R-squared is high for good linear fit
        assert result['r_squared'] > 0.9
        
        # Check calibration is stored in database
        assert instrument_type in experimental_agent.calibration_database
        
        # Test calibration function
        cal_func = result['calibration_function']
        test_raw = 2.0
        corrected_value = cal_func(test_raw)
        # Should correct the systematic error
        expected_corrected = (test_raw - 0.1) / 0.95
        assert abs(corrected_value - expected_corrected) < 0.1
    
    @pytest.mark.parametrize("fit_function", ['linear', 'quadratic', 'exponential', 'gaussian'])
    def test_data_fitting_analysis(self, experimental_agent, fit_function):
        """Test data fitting with various functional forms."""
        # Generate test data
        x_data = np.linspace(0, 10, 50)
        
        if fit_function == 'linear':
            true_params = [2.0, 1.0]  # slope, intercept
            y_true = true_params[0] * x_data + true_params[1]
        elif fit_function == 'quadratic':
            true_params = [0.1, 2.0, 1.0]  # a, b, c
            y_true = true_params[0] * x_data**2 + true_params[1] * x_data + true_params[2]
        elif fit_function == 'exponential':
            true_params = [2.0, 0.1, 0.5]  # a, b, c
            y_true = true_params[0] * np.exp(true_params[1] * x_data) + true_params[2]
        elif fit_function == 'gaussian':
            true_params = [2.0, 5.0, 1.5]  # amplitude, center, width
            y_true = true_params[0] * np.exp(-0.5 * ((x_data - true_params[1]) / true_params[2])**2)
        
        # Add noise
        noise_level = 0.1
        y_data = y_true + np.random.normal(0, noise_level, len(x_data))
        
        result = experimental_agent.analyze_data_fitting(x_data, y_data, fit_function)
        
        assert 'fit_function' in result
        assert 'parameters' in result
        assert 'parameter_uncertainties' in result
        assert 'r_squared' in result
        assert 'fitted_values' in result
        assert 'residuals' in result
        assert 'fit_successful' in result
        
        # Check fit quality
        assert result['fit_successful'] is True
        assert result['r_squared'] > 0.8  # Should be good fit for synthetic data
        
        # Check parameter recovery (within reasonable tolerance)
        fitted_params = result['parameters']
        for i, (fitted, true) in enumerate(zip(fitted_params, true_params)):
            relative_error = abs(fitted - true) / abs(true) if true != 0 else abs(fitted)
            assert relative_error < 0.2  # Within 20%
        
        # Check uncertainties are reasonable
        param_uncertainties = result['parameter_uncertainties']
        assert all(unc > 0 for unc in param_uncertainties)
        assert all(unc < abs(param) for unc, param in zip(param_uncertainties, fitted_params))
    
    @pytest.mark.parametrize("test_type", ['normality', 'outliers', 'correlation'])
    def test_statistical_analysis(self, experimental_agent, test_type):
        """Test statistical analysis methods."""
        if test_type == 'normality':
            # Generate normal data
            data = np.random.normal(0, 1, 100)
            
            result = experimental_agent.perform_statistical_analysis(data, test_type)
            
            assert 'shapiro_wilk' in result
            assert 'anderson_darling' in result
            assert 'kolmogorov_smirnov' in result
            assert 'summary' in result
            
            # Should detect normality for normal data
            assert result['summary']['likely_normal'] is True
            assert result['shapiro_wilk']['is_normal'] is True
            
        elif test_type == 'outliers':
            # Generate data with outliers
            clean_data = np.random.normal(0, 1, 95)
            outliers = np.array([5, -5, 6, -6, 7])  # Clear outliers
            data = np.concatenate([clean_data, outliers])
            
            result = experimental_agent.perform_statistical_analysis(data, test_type)
            
            assert 'modified_z_score' in result
            assert 'iqr_method' in result
            assert 'summary' in result
            
            # Should detect some outliers
            assert result['summary']['total_outliers'] > 0
            assert result['summary']['outlier_fraction'] > 0.01
            
        elif test_type == 'correlation':
            # Generate correlated data
            x = np.random.normal(0, 1, 50)
            y = 2 * x + np.random.normal(0, 0.5, 50)  # Strong positive correlation
            data = np.column_stack([x, y])
            
            result = experimental_agent.perform_statistical_analysis(data, test_type)
            
            assert 'pearson' in result
            assert 'spearman' in result
            assert 'kendall' in result
            assert 'summary' in result
            
            # Should detect strong positive correlation
            assert result['pearson']['correlation'] > 0.7
            assert result['pearson']['significant'] is True
            assert result['summary']['strongest_correlation'] > 0.7
    
    def test_measurement_uncertainty_calculation(self, experimental_agent):
        """Test measurement uncertainty calculation using GUM methodology."""
        # Create multiple measurements with different uncertainties
        measurements = [
            {'value': 10.0, 'uncertainty': 0.1},
            {'value': 10.1, 'uncertainty': 0.1},
            {'value': 9.9, 'uncertainty': 0.1},
            {'value': 10.05, 'uncertainty': 0.05},
            {'value': 9.95, 'uncertainty': 0.05}
        ]
        
        result = experimental_agent.calculate_measurement_uncertainty(measurements)
        
        assert 'mean_value' in result
        assert 'type_a_uncertainty' in result
        assert 'type_b_uncertainty' in result
        assert 'combined_uncertainty' in result
        assert 'expanded_uncertainty' in result
        assert 'uncertainty_budget' in result
        
        # Check mean value
        values = [m['value'] for m in measurements]
        expected_mean = np.mean(values)
        assert abs(result['mean_value'] - expected_mean) < 1e-10
        
        # Check uncertainty components
        assert result['type_a_uncertainty'] > 0  # Statistical uncertainty
        assert result['type_b_uncertainty'] > 0  # Systematic uncertainty
        assert result['combined_uncertainty'] > 0
        
        # Combined should be larger than individual components
        assert result['combined_uncertainty'] >= result['type_a_uncertainty']
        assert result['combined_uncertainty'] >= result['type_b_uncertainty']
        
        # Expanded uncertainty should be larger
        assert result['expanded_uncertainty'] > result['combined_uncertainty']
        
        # Check uncertainty budget sums to 1
        budget = result['uncertainty_budget']
        total_contribution = budget['statistical_contribution'] + budget['systematic_contribution']
        assert abs(total_contribution - 1.0) < 1e-10
    
    def test_detector_response_simulation(self, experimental_agent):
        """Test detector response simulation."""
        # Create test signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)  # 10 Hz + 30 Hz
        
        detector_config = {
            'efficiency': 0.8,
            'noise_level': 0.05,
            'bandwidth': 50.0,  # Hz
            'sampling_rate': 1000.0,  # Hz
            'deadtime': 1e-3  # 1 ms
        }
        
        result = experimental_agent.simulate_detector_response(signal, detector_config)
        
        assert 'original_signal' in result
        assert 'detected_signal' in result
        assert 'final_signal' in result
        assert 'performance_metrics' in result
        assert 'signal_statistics' in result
        
        # Check signal processing chain
        original = result['original_signal']
        detected = result['detected_signal']
        final = result['final_signal']
        
        # Detected signal should be reduced by efficiency
        efficiency = detector_config['efficiency']
        expected_amplitude_ratio = efficiency
        actual_ratio = np.std(detected) / np.std(original)
        assert abs(actual_ratio - expected_amplitude_ratio) < 0.1
        
        # Check SNR calculation
        snr = result['performance_metrics']['snr_db']
        assert snr > 0  # Should have positive SNR for this test case
        
        # Check bandwidth limitation (high frequency component should be attenuated)
        # Compare frequency content
        original_fft = np.abs(np.fft.fft(original))
        final_fft = np.abs(np.fft.fft(final))
        
        # High frequency content should be reduced
        freqs = np.fft.fftfreq(len(t), 1.0/detector_config['sampling_rate'])
        high_freq_mask = np.abs(freqs) > detector_config['bandwidth']
        
        if np.any(high_freq_mask):
            high_freq_reduction = np.mean(final_fft[high_freq_mask]) / np.mean(original_fft[high_freq_mask])
            assert high_freq_reduction < 0.5  # High frequencies should be significantly reduced
    
    def test_experimental_workflow_integration(self, experimental_agent, physics_test_config):
        """Test integrated experimental workflow."""
        # Step 1: Calibrate instrument
        calibration_standards = [
            {'true_value': 1.0, 'uncertainty': 0.001},
            {'true_value': 5.0, 'uncertainty': 0.001},
            {'true_value': 10.0, 'uncertainty': 0.001}
        ]
        
        measured_standards = np.array([1.05, 5.02, 9.98])  # With small systematic error
        
        calibration_result = experimental_agent.calibrate_instrument(
            'test_instrument', calibration_standards, measured_standards
        )
        
        # Step 2: Perform measurements
        measurement_params = {
            'true_value': 7.5,
            'systematic_error': 0.02,
            'statistical_error': 0.01
        }
        
        measurement_result = experimental_agent.perform_measurement(
            'voltage', measurement_params, n_measurements=20
        )
        
        # Step 3: Apply calibration correction
        cal_func = calibration_result['calibration_function']
        corrected_measurements = [cal_func(m) for m in measurement_result['measurements']]
        
        # Step 4: Statistical analysis
        statistical_result = experimental_agent.perform_statistical_analysis(
            np.array(corrected_measurements), 'normality'
        )
        
        # Step 5: Uncertainty analysis
        measurement_list = [
            {'value': m, 'uncertainty': 0.01} for m in corrected_measurements
        ]
        
        uncertainty_result = experimental_agent.calculate_measurement_uncertainty(measurement_list)
        
        # Verify workflow integration
        assert calibration_result['calibration_valid'] is True
        assert len(corrected_measurements) == measurement_result['n_measurements']
        assert statistical_result['summary']['likely_normal'] is True
        assert uncertainty_result['combined_uncertainty'] > 0
        
        # Check that calibration improves accuracy
        original_mean = measurement_result['mean']
        corrected_mean = np.mean(corrected_measurements)
        true_value = measurement_params['true_value']
        
        original_error = abs(original_mean - true_value)
        corrected_error = abs(corrected_mean - true_value)
        
        # Calibration should reduce systematic error
        assert corrected_error <= original_error
    
    @pytest.mark.slow
    def test_large_dataset_analysis(self, experimental_agent):
        """Test analysis of large experimental datasets."""
        # Generate large dataset
        n_points = 10000
        x_data = np.linspace(0, 100, n_points)
        
        # Complex signal with multiple components and noise
        signal = (2.0 * np.sin(0.1 * x_data) + 
                 0.5 * np.sin(0.5 * x_data) + 
                 0.1 * x_data +  # Linear trend
                 np.random.normal(0, 0.1, n_points))  # Noise
        
        # Test fitting
        fit_result = experimental_agent.analyze_data_fitting(x_data, signal, 'linear')
        
        # Test statistical analysis
        stats_result = experimental_agent.perform_statistical_analysis(signal, 'outliers')
        
        # Test detector simulation
        detector_config = {
            'efficiency': 0.95,
            'noise_level': 0.02,
            'bandwidth': 1000.0,
            'sampling_rate': 10000.0
        }
        
        detector_result = experimental_agent.simulate_detector_response(signal, detector_config)
        
        # Verify all analyses complete successfully
        assert fit_result['fit_successful'] is True
        assert stats_result['summary']['total_outliers'] >= 0
        assert len(detector_result['final_signal']) == n_points
        
        # Check performance on large dataset
        assert detector_result['performance_metrics']['snr_db'] > 0
    
    def test_experimental_error_handling(self, experimental_agent):
        """Test error handling in experimental physics calculations."""
        # Test with empty measurement list
        with pytest.raises(ValueError):
            experimental_agent.calculate_measurement_uncertainty([])
        
        # Test with mismatched calibration data
        standards = [{'true_value': 1.0, 'uncertainty': 0.01}]
        measurements = np.array([1.0, 2.0])  # Mismatched length
        
        with pytest.raises(ValueError):
            experimental_agent.calibrate_instrument('test', standards, measurements)
        
        # Test with invalid fit function
        x_data = np.array([1, 2, 3])
        y_data = np.array([1, 4, 9])
        
        with pytest.raises(ValueError):
            experimental_agent.analyze_data_fitting(x_data, y_data, 'invalid_function')
        
        # Test with invalid statistical test
        data = np.random.normal(0, 1, 100)
        
        with pytest.raises(ValueError):
            experimental_agent.perform_statistical_analysis(data, 'invalid_test')


class TestExperimentalPhysicsIntegration:
    """Integration tests for experimental physics workflows."""
    
    @pytest.fixture
    def experimental_workflow(self):
        """Create an experimental physics workflow."""
        agent = MockExperimentalPhysicsAgent()
        
        workflow = {
            'agent': agent,
            'experiments': {},
            'calibrations': {},
            'analysis_results': {}
        }
        
        return workflow
    
    def test_complete_experimental_campaign(self, experimental_workflow, physics_test_config):
        """Test complete experimental campaign workflow."""
        agent = experimental_workflow['agent']
        
        # Phase 1: Instrument calibration
        instruments = ['voltmeter', 'thermometer', 'pressure_gauge']
        
        for instrument in instruments:
            standards = [
                {'true_value': 1.0, 'uncertainty': 0.001},
                {'true_value': 5.0, 'uncertainty': 0.001},
                {'true_value': 10.0, 'uncertainty': 0.001}
            ]
            
            # Simulate measurements with realistic errors
            measured_values = np.array([1.02, 4.98, 10.01]) + np.random.normal(0, 0.01, 3)
            
            calibration = agent.calibrate_instrument(instrument, standards, measured_values)
            experimental_workflow['calibrations'][instrument] = calibration
        
        # Phase 2: Data collection
        experiment_conditions = [
            {'temperature': 300, 'pressure': 1.0, 'voltage': 5.0},
            {'temperature': 350, 'pressure': 1.2, 'voltage': 7.5},
            {'temperature': 400, 'pressure': 1.5, 'voltage': 10.0}
        ]
        
        for i, conditions in enumerate(experiment_conditions):
            experiment_data = {}
            
            # Collect measurements for each parameter
            for param, true_value in conditions.items():
                measurement_params = {
                    'true_value': true_value,
                    'systematic_error': 0.01,
                    'statistical_error': 0.005
                }
                
                measurement = agent.perform_measurement(
                    'temperature' if param == 'temperature' else 'voltage',
                    measurement_params,
                    n_measurements=10
                )
                
                # Apply calibration correction
                if param == 'temperature':
                    cal_func = experimental_workflow['calibrations']['thermometer']['calibration_function']
                elif param == 'voltage':
                    cal_func = experimental_workflow['calibrations']['voltmeter']['calibration_function']
                else:
                    cal_func = experimental_workflow['calibrations']['pressure_gauge']['calibration_function']
                
                corrected_values = [cal_func(m) for m in measurement['measurements']]
                measurement['corrected_measurements'] = corrected_values
                measurement['corrected_mean'] = np.mean(corrected_values)
                
                experiment_data[param] = measurement
            
            experimental_workflow['experiments'][f'condition_{i}'] = experiment_data
        
        # Phase 3: Data analysis
        # Extract temperature vs voltage relationship
        temperatures = []
        voltages = []
        voltage_uncertainties = []
        
        for exp_data in experimental_workflow['experiments'].values():
            temperatures.append(exp_data['temperature']['corrected_mean'])
            voltages.append(exp_data['voltage']['corrected_mean'])
            voltage_uncertainties.append(exp_data['voltage']['standard_error'])
        
        temperatures = np.array(temperatures)
        voltages = np.array(voltages)
        weights = 1.0 / np.array(voltage_uncertainties)**2
        
        # Fit temperature-voltage relationship
        fit_result = agent.analyze_data_fitting(temperatures, voltages, 'linear', weights=weights)
        
        # Statistical analysis of residuals
        residuals = fit_result['residuals']
        stats_result = agent.perform_statistical_analysis(residuals, 'normality')
        
        # Uncertainty propagation
        measurement_list = [
            {'value': v, 'uncertainty': u} 
            for v, u in zip(voltages, voltage_uncertainties)
        ]
        uncertainty_result = agent.calculate_measurement_uncertainty(measurement_list)
        
        # Store analysis results
        experimental_workflow['analysis_results'] = {
            'temperature_voltage_fit': fit_result,
            'residual_analysis': stats_result,
            'combined_uncertainty': uncertainty_result
        }
        
        # Verify complete campaign
        assert len(experimental_workflow['calibrations']) == 3
        assert len(experimental_workflow['experiments']) == 3
        assert fit_result['fit_successful'] is True
        assert fit_result['r_squared'] > 0.8  # Good correlation expected
        assert stats_result['summary']['likely_normal'] is True  # Residuals should be normal
        
        # Check calibration effectiveness
        for cal_result in experimental_workflow['calibrations'].values():
            assert cal_result['calibration_valid'] is True
            assert cal_result['r_squared'] > 0.95
    
    def test_detector_characterization_study(self, experimental_workflow):
        """Test comprehensive detector characterization study."""
        agent = experimental_workflow['agent']
        
        # Study detector performance across different conditions
        test_conditions = [
            {'efficiency': 0.9, 'noise_level': 0.01, 'bandwidth': 1000},
            {'efficiency': 0.8, 'noise_level': 0.02, 'bandwidth': 500},
            {'efficiency': 0.95, 'noise_level': 0.005, 'bandwidth': 2000}
        ]
        
        # Test signal
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave
        
        characterization_results = {}
        
        for i, config in enumerate(test_conditions):
            detector_config = {
                **config,
                'sampling_rate': 5000.0,
                'deadtime': 1e-4
            }
            
            # Simulate detector response
            response = agent.simulate_detector_response(test_signal, detector_config)
            
            # Analyze signal quality
            snr = response['performance_metrics']['snr_db']
            resolution = response['performance_metrics']['resolution']
            
            # Frequency analysis
            final_signal = response['final_signal']
            fft_result = np.abs(np.fft.fft(final_signal))
            freqs = np.fft.fftfreq(len(final_signal), 1.0/detector_config['sampling_rate'])
            
            # Find peak frequency
            peak_idx = np.argmax(fft_result[:len(fft_result)//2])
            detected_frequency = abs(freqs[peak_idx])
            
            characterization_results[f'config_{i}'] = {
                'detector_config': detector_config,
                'response': response,
                'performance': {
                    'snr_db': snr,
                    'resolution': resolution,
                    'detected_frequency': detected_frequency,
                    'frequency_accuracy': abs(detected_frequency - 100) / 100
                }
            }
        
        experimental_workflow['analysis_results']['detector_study'] = characterization_results
        
        # Verify detector characterization
        for result in characterization_results.values():
            perf = result['performance']
            
            # All detectors should detect the signal
            assert perf['snr_db'] > 0
            assert perf['frequency_accuracy'] < 0.1  # Within 10% of true frequency
            
            # Higher efficiency should give better SNR
            efficiency = result['detector_config']['efficiency']
            assert perf['snr_db'] > 10 * np.log10(efficiency) - 10  # Rough correlation
    
    @pytest.mark.integration
    def test_measurement_reproducibility_study(self, experimental_workflow):
        """Test measurement reproducibility and systematic error analysis."""
        agent = experimental_workflow['agent']
        
        # Simulate repeated measurements over time to study reproducibility
        true_value = 10.0
        n_sessions = 5
        measurements_per_session = 20
        
        session_results = {}
        
        for session in range(n_sessions):
            # Each session might have slightly different systematic errors
            session_systematic = np.random.normal(0, 0.01)  # Session-to-session variation
            
            measurement_params = {
                'true_value': true_value,
                'systematic_error': session_systematic,
                'statistical_error': 0.005
            }
            
            session_measurement = agent.perform_measurement(
                'voltage', measurement_params, measurements_per_session
            )
            
            session_results[f'session_{session}'] = session_measurement
        
        # Analyze reproducibility
        session_means = [result['mean'] for result in session_results.values()]
        session_std_errors = [result['standard_error'] for result in session_results.values()]
        
        # Between-session variability
        between_session_std = np.std(session_means, ddof=1)
        within_session_std = np.mean(session_std_errors)
        
        # Statistical analysis of session means
        stats_result = agent.perform_statistical_analysis(np.array(session_means), 'normality')
        
        # Overall uncertainty analysis
        all_measurements = []
        for result in session_results.values():
            measurements = result['measurements']
            measurement_list = [{'value': m, 'uncertainty': 0.005} for m in measurements]
            all_measurements.extend(measurement_list)
        
        overall_uncertainty = agent.calculate_measurement_uncertainty(all_measurements)
        
        experimental_workflow['analysis_results']['reproducibility'] = {
            'session_results': session_results,
            'between_session_std': between_session_std,
            'within_session_std': within_session_std,
            'session_statistics': stats_result,
            'overall_uncertainty': overall_uncertainty
        }
        
        # Verify reproducibility analysis
        assert len(session_results) == n_sessions
        assert between_session_std > 0  # Should have some session-to-session variation
        assert within_session_std > 0   # Should have statistical uncertainty
        
        # Check that overall uncertainty accounts for both sources
        total_uncertainty = overall_uncertainty['combined_uncertainty']
        assert total_uncertainty > within_session_std
        assert total_uncertainty > between_session_std
        
        # Session means should be normally distributed
        assert stats_result['summary']['likely_normal'] is True
    
    @pytest.mark.asyncio
    async def test_async_experimental_data_collection(self, async_physics_simulator):
        """Test asynchronous experimental data collection."""
        # Start long data collection simulation
        collection_task = asyncio.create_task(
            async_physics_simulator.run_simulation(duration=1.5, dt=0.001)
        )
        
        # Verify data collection is running
        assert async_physics_simulator.running is True
        
        # Wait for completion
        result = await collection_task
        
        assert result['status'] == 'completed'
        assert result['steps'] > 0
        assert async_physics_simulator.running is False