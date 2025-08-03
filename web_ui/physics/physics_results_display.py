"""
Physics Results Display Component

This module provides comprehensive display and analysis of physics research results
including statistical analysis, uncertainty quantification, and data visualization.
"""

import time
import logging
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class ResultType(Enum):
    """Types of physics results."""
    EXPERIMENTAL_DATA = "experimental_data"
    SIMULATION_OUTPUT = "simulation_output"
    THEORETICAL_CALCULATION = "theoretical_calculation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    FITTING_RESULTS = "fitting_results"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"
    COMPARISON_STUDY = "comparison_study"
    META_ANALYSIS = "meta_analysis"

class DataFormat(Enum):
    """Data formats for results."""
    NUMERICAL = "numerical"
    TIME_SERIES = "time_series"
    DISTRIBUTION = "distribution"
    CORRELATION_MATRIX = "correlation_matrix"
    SPECTRUM = "spectrum"
    IMAGE = "image"
    GRAPH = "graph"
    TEXT = "text"

@dataclass
class UncertaintyInfo:
    """Uncertainty information for measurements."""
    value: float
    uncertainty: float
    uncertainty_type: str = "standard"  # standard, systematic, statistical, combined
    confidence_level: float = 0.68  # 1-sigma by default
    distribution: str = "gaussian"  # gaussian, uniform, poisson, etc.
    sources: List[str] = field(default_factory=list)

@dataclass
class StatisticalSummary:
    """Statistical summary of data."""
    count: int
    mean: float
    median: float
    std: float
    variance: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    skewness: float
    kurtosis: float
    confidence_interval_95: Tuple[float, float]

@dataclass
class FittingResult:
    """Result from curve fitting analysis."""
    model_name: str
    parameters: Dict[str, UncertaintyInfo]
    goodness_of_fit: Dict[str, float]
    residuals: List[float]
    fitted_values: List[float]
    covariance_matrix: List[List[float]]
    parameter_correlations: Dict[str, Dict[str, float]]

@dataclass
class PhysicsResult:
    """Complete physics result with metadata."""
    result_id: str
    name: str
    description: str
    result_type: ResultType
    data_format: DataFormat
    data: Any
    metadata: Dict[str, Any]
    uncertainty_info: Optional[UncertaintyInfo] = None
    statistical_summary: Optional[StatisticalSummary] = None
    fitting_result: Optional[FittingResult] = None
    created_at: float = field(default_factory=time.time)
    source_experiment: Optional[str] = None
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

class PhysicsResultsDisplay:
    """
    Physics Results Display and Analysis System
    
    Provides comprehensive display, analysis, and visualization of physics research results
    with statistical analysis, uncertainty quantification, and comparison capabilities.
    """
    
    def __init__(self):
        self.results: Dict[str, PhysicsResult] = {}
        self.result_collections: Dict[str, List[str]] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
    def add_result(self, result: PhysicsResult) -> bool:
        """Add a new physics result."""
        try:
            # Generate ID if not provided
            if not result.result_id:
                result.result_id = f"result_{int(time.time() * 1000)}"
            
            # Perform automatic analysis
            self._perform_automatic_analysis(result)
            
            # Store result
            self.results[result.result_id] = result
            
            logger.info(f"Added physics result: {result.name} ({result.result_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding result: {e}")
            return False
    
    def get_result(self, result_id: str) -> Optional[PhysicsResult]:
        """Get a specific physics result."""
        return self.results.get(result_id)
    
    def list_results(self, result_type: Optional[ResultType] = None,
                    tags: Optional[List[str]] = None,
                    date_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """List results with optional filtering."""
        try:
            results_list = []
            
            for result_id, result in self.results.items():
                # Apply filters
                if result_type and result.result_type != result_type:
                    continue
                
                if tags and not any(tag in result.tags for tag in tags):
                    continue
                
                if date_range:
                    start_time, end_time = date_range
                    if not (start_time <= result.created_at <= end_time):
                        continue
                
                # Create summary
                result_summary = {
                    'result_id': result_id,
                    'name': result.name,
                    'description': result.description,
                    'type': result.result_type.value,
                    'format': result.data_format.value,
                    'created_at': result.created_at,
                    'source_experiment': result.source_experiment,
                    'tags': result.tags,
                    'has_uncertainty': result.uncertainty_info is not None,
                    'has_statistics': result.statistical_summary is not None,
                    'has_fitting': result.fitting_result is not None
                }
                
                # Add data size information
                if isinstance(result.data, (list, np.ndarray)):
                    result_summary['data_size'] = len(result.data)
                elif isinstance(result.data, dict):
                    result_summary['data_keys'] = list(result.data.keys())
                
                results_list.append(result_summary)
            
            # Sort by creation time (newest first)
            results_list.sort(key=lambda x: x['created_at'], reverse=True)
            
            return results_list
            
        except Exception as e:
            logger.error(f"Error listing results: {e}")
            return []
    
    def analyze_result_statistics(self, result_id: str) -> Dict[str, Any]:
        """Perform detailed statistical analysis on a result."""
        try:
            if result_id not in self.results:
                return {'error': 'Result not found'}
            
            result = self.results[result_id]
            
            # Check if already analyzed
            cache_key = f"stats_{result_id}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            analysis = {}
            
            # Extract numerical data
            numerical_data = self._extract_numerical_data(result.data)
            
            if numerical_data is not None and len(numerical_data) > 0:
                # Calculate statistical summary
                stats = self._calculate_statistical_summary(numerical_data)
                result.statistical_summary = stats
                
                analysis['statistical_summary'] = {
                    'count': stats.count,
                    'mean': stats.mean,
                    'median': stats.median,
                    'std': stats.std,
                    'variance': stats.variance,
                    'min': stats.min_value,
                    'max': stats.max_value,
                    'percentile_25': stats.percentile_25,
                    'percentile_75': stats.percentile_75,
                    'skewness': stats.skewness,
                    'kurtosis': stats.kurtosis,
                    'confidence_interval_95': stats.confidence_interval_95
                }
                
                # Perform normality tests
                analysis['normality_tests'] = self._perform_normality_tests(numerical_data)
                
                # Detect outliers
                analysis['outlier_detection'] = self._detect_outliers(numerical_data)
                
                # Calculate correlations if multivariate
                if len(numerical_data.shape) > 1 and numerical_data.shape[1] > 1:
                    analysis['correlations'] = self._calculate_correlations(numerical_data)
            
            # Analyze uncertainties if present
            if result.uncertainty_info:
                analysis['uncertainty_analysis'] = self._analyze_uncertainties(result.uncertainty_info)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing result statistics: {e}")
            return {'error': str(e)}
    
    def perform_curve_fitting(self, result_id: str, model_function: str,
                            initial_parameters: Dict[str, float],
                            x_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """Perform curve fitting on result data."""
        try:
            if result_id not in self.results:
                return {'error': 'Result not found'}
            
            result = self.results[result_id]
            
            # Extract data for fitting
            y_data = self._extract_numerical_data(result.data)
            if y_data is None:
                return {'error': 'No numerical data found for fitting'}
            
            # Generate x_data if not provided
            if x_data is None:
                x_data = np.arange(len(y_data))
            else:
                x_data = np.array(x_data)
            
            y_data = np.array(y_data)
            
            # Perform fitting (simplified implementation)
            fitting_result = self._perform_curve_fitting(
                x_data, y_data, model_function, initial_parameters
            )
            
            # Store fitting result
            result.fitting_result = fitting_result
            
            # Return fitting summary
            fitting_summary = {
                'model': model_function,
                'parameters': {
                    name: {
                        'value': param.value,
                        'uncertainty': param.uncertainty,
                        'relative_uncertainty': param.uncertainty / abs(param.value) if param.value != 0 else float('inf')
                    }
                    for name, param in fitting_result.parameters.items()
                },
                'goodness_of_fit': fitting_result.goodness_of_fit,
                'parameter_correlations': fitting_result.parameter_correlations
            }
            
            return fitting_summary
            
        except Exception as e:
            logger.error(f"Error performing curve fitting: {e}")
            return {'error': str(e)}
    
    def compare_results(self, result_ids: List[str], comparison_type: str = "statistical") -> Dict[str, Any]:
        """Compare multiple physics results."""
        try:
            if len(result_ids) < 2:
                return {'error': 'At least two results required for comparison'}
            
            # Get results
            results = []
            for result_id in result_ids:
                if result_id in self.results:
                    results.append(self.results[result_id])
                else:
                    return {'error': f'Result {result_id} not found'}
            
            comparison = {
                'comparison_type': comparison_type,
                'result_count': len(results),
                'results_info': [
                    {
                        'id': r.result_id,
                        'name': r.name,
                        'type': r.result_type.value,
                        'created_at': r.created_at
                    }
                    for r in results
                ]
            }
            
            if comparison_type == "statistical":
                comparison.update(self._compare_statistical_properties(results))
            elif comparison_type == "uncertainty":
                comparison.update(self._compare_uncertainties(results))
            elif comparison_type == "temporal":
                comparison.update(self._compare_temporal_evolution(results))
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return {'error': str(e)}
    
    def quantify_uncertainties(self, result_id: str, 
                             uncertainty_sources: Dict[str, float],
                             correlation_matrix: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """Perform comprehensive uncertainty quantification."""
        try:
            if result_id not in self.results:
                return {'error': 'Result not found'}
            
            result = self.results[result_id]
            
            # Calculate combined uncertainty
            combined_uncertainty = self._calculate_combined_uncertainty(
                uncertainty_sources, correlation_matrix
            )
            
            # Perform uncertainty propagation
            propagation_result = self._propagate_uncertainties(
                result.data, uncertainty_sources
            )
            
            # Create uncertainty info
            uncertainty_info = UncertaintyInfo(
                value=float(np.mean(self._extract_numerical_data(result.data)) if isinstance(result.data, (list, np.ndarray)) else 0),
                uncertainty=combined_uncertainty,
                uncertainty_type="combined",
                confidence_level=0.68,
                distribution="gaussian",
                sources=list(uncertainty_sources.keys())
            )
            
            result.uncertainty_info = uncertainty_info
            
            uncertainty_analysis = {
                'combined_uncertainty': combined_uncertainty,
                'individual_contributions': uncertainty_sources,
                'uncertainty_budget': self._create_uncertainty_budget(uncertainty_sources),
                'propagation_result': propagation_result,
                'confidence_intervals': self._calculate_confidence_intervals(
                    uncertainty_info.value, combined_uncertainty
                )
            }
            
            return uncertainty_analysis
            
        except Exception as e:
            logger.error(f"Error quantifying uncertainties: {e}")
            return {'error': str(e)}
    
    def generate_result_report(self, result_id: str, include_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive report for a physics result."""
        try:
            if result_id not in self.results:
                return {'error': 'Result not found'}
            
            result = self.results[result_id]
            
            # Basic information
            report = {
                'result_id': result_id,
                'name': result.name,
                'description': result.description,
                'type': result.result_type.value,
                'format': result.data_format.value,
                'created_at': result.created_at,
                'created_at_formatted': datetime.fromtimestamp(result.created_at).isoformat(),
                'source_experiment': result.source_experiment,
                'tags': result.tags,
                'references': result.references,
                'metadata': result.metadata
            }
            
            # Statistical analysis
            if result.statistical_summary or result_id in [k.split('_')[1] for k in self.analysis_cache.keys() if k.startswith('stats_')]:
                stats_analysis = self.analyze_result_statistics(result_id)
                if 'error' not in stats_analysis:
                    report['statistical_analysis'] = stats_analysis
            
            # Uncertainty information
            if result.uncertainty_info:
                report['uncertainty_info'] = {
                    'value': result.uncertainty_info.value,
                    'uncertainty': result.uncertainty_info.uncertainty,
                    'relative_uncertainty_percent': (result.uncertainty_info.uncertainty / 
                                                   abs(result.uncertainty_info.value) * 100) if result.uncertainty_info.value != 0 else 0,
                    'uncertainty_type': result.uncertainty_info.uncertainty_type,
                    'confidence_level': result.uncertainty_info.confidence_level,
                    'distribution': result.uncertainty_info.distribution,
                    'sources': result.uncertainty_info.sources
                }
            
            # Fitting results
            if result.fitting_result:
                report['fitting_analysis'] = {
                    'model': result.fitting_result.model_name,
                    'parameters': {
                        name: {
                            'value': param.value,
                            'uncertainty': param.uncertainty,
                            'relative_uncertainty_percent': (param.uncertainty / abs(param.value) * 100) if param.value != 0 else 0
                        }
                        for name, param in result.fitting_result.parameters.items()
                    },
                    'goodness_of_fit': result.fitting_result.goodness_of_fit,
                    'parameter_correlations': result.fitting_result.parameter_correlations
                }
            
            # Data summary
            data_info = self._summarize_data(result.data)
            report['data_summary'] = data_info
            
            # Recommendations
            report['recommendations'] = self._generate_recommendations(result)
            
            # Validation checklist
            report['validation'] = self._validate_result_quality(result)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating result report: {e}")
            return {'error': str(e)}
    
    def export_results(self, result_ids: List[str], 
                      format_type: str = "json",
                      include_raw_data: bool = True) -> Dict[str, Any]:
        """Export multiple results in specified format."""
        try:
            export_data = {
                'export_timestamp': time.time(),
                'export_format': format_type,
                'result_count': len(result_ids),
                'results': [],
                'metadata': {
                    'version': '1.0.0',
                    'generated_by': 'physics_results_display'
                }
            }
            
            for result_id in result_ids:
                if result_id not in self.results:
                    continue
                
                result = self.results[result_id]
                
                result_data = {
                    'result_id': result_id,
                    'name': result.name,
                    'description': result.description,
                    'type': result.result_type.value,
                    'format': result.data_format.value,
                    'created_at': result.created_at,
                    'source_experiment': result.source_experiment,
                    'tags': result.tags,
                    'references': result.references,
                    'metadata': result.metadata
                }
                
                if include_raw_data:
                    result_data['data'] = self._serialize_data(result.data)
                
                if result.uncertainty_info:
                    result_data['uncertainty_info'] = {
                        'value': result.uncertainty_info.value,
                        'uncertainty': result.uncertainty_info.uncertainty,
                        'uncertainty_type': result.uncertainty_info.uncertainty_type,
                        'confidence_level': result.uncertainty_info.confidence_level,
                        'distribution': result.uncertainty_info.distribution,
                        'sources': result.uncertainty_info.sources
                    }
                
                if result.statistical_summary:
                    result_data['statistical_summary'] = {
                        'count': result.statistical_summary.count,
                        'mean': result.statistical_summary.mean,
                        'median': result.statistical_summary.median,
                        'std': result.statistical_summary.std,
                        'variance': result.statistical_summary.variance,
                        'min': result.statistical_summary.min_value,
                        'max': result.statistical_summary.max_value,
                        'percentile_25': result.statistical_summary.percentile_25,
                        'percentile_75': result.statistical_summary.percentile_75,
                        'skewness': result.statistical_summary.skewness,
                        'kurtosis': result.statistical_summary.kurtosis,
                        'confidence_interval_95': result.statistical_summary.confidence_interval_95
                    }
                
                export_data['results'].append(result_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return {'error': str(e)}
    
    # Helper methods
    
    def _perform_automatic_analysis(self, result: PhysicsResult):
        """Perform automatic analysis on new results."""
        try:
            # Extract numerical data if possible
            numerical_data = self._extract_numerical_data(result.data)
            
            if numerical_data is not None and len(numerical_data) > 0:
                # Calculate statistical summary
                result.statistical_summary = self._calculate_statistical_summary(numerical_data)
                
                # Detect if uncertainty information can be inferred
                if result.uncertainty_info is None:
                    inferred_uncertainty = self._infer_uncertainty(numerical_data)
                    if inferred_uncertainty:
                        result.uncertainty_info = inferred_uncertainty
        except Exception as e:
            logger.warning(f"Error in automatic analysis: {e}")
    
    def _extract_numerical_data(self, data: Any) -> Optional[np.ndarray]:
        """Extract numerical data from various data formats."""
        try:
            if isinstance(data, (list, tuple)):
                # Try to convert to numerical array
                try:
                    return np.array(data, dtype=float)
                except:
                    # Try to extract numbers from nested structures
                    flat_data = []
                    for item in data:
                        if isinstance(item, (int, float)):
                            flat_data.append(item)
                        elif isinstance(item, (list, tuple)) and len(item) > 0:
                            if isinstance(item[0], (int, float)):
                                flat_data.extend(item)
                    return np.array(flat_data, dtype=float) if flat_data else None
            elif isinstance(data, np.ndarray):
                return data.astype(float)
            elif isinstance(data, dict):
                # Try to extract numerical values from dictionary
                numerical_values = []
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        numerical_values.append(value)
                    elif isinstance(value, (list, tuple)):
                        try:
                            numerical_values.extend([float(x) for x in value if isinstance(x, (int, float))])
                        except:
                            pass
                return np.array(numerical_values) if numerical_values else None
            elif isinstance(data, (int, float)):
                return np.array([data])
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting numerical data: {e}")
            return None
    
    def _calculate_statistical_summary(self, data: np.ndarray) -> StatisticalSummary:
        """Calculate comprehensive statistical summary."""
        data_flat = data.flatten()
        
        # Remove any invalid values
        data_clean = data_flat[np.isfinite(data_flat)]
        
        if len(data_clean) == 0:
            # Handle empty data
            return StatisticalSummary(
                count=0, mean=0, median=0, std=0, variance=0,
                min_value=0, max_value=0, percentile_25=0, percentile_75=0,
                skewness=0, kurtosis=0, confidence_interval_95=(0, 0)
            )
        
        mean_val = float(np.mean(data_clean))
        std_val = float(np.std(data_clean, ddof=1)) if len(data_clean) > 1 else 0
        
        # Calculate confidence interval
        if len(data_clean) > 1:
            sem = std_val / np.sqrt(len(data_clean))
            ci_margin = 1.96 * sem  # 95% confidence interval
            ci_95 = (mean_val - ci_margin, mean_val + ci_margin)
        else:
            ci_95 = (mean_val, mean_val)
        
        return StatisticalSummary(
            count=len(data_clean),
            mean=mean_val,
            median=float(np.median(data_clean)),
            std=std_val,
            variance=float(np.var(data_clean, ddof=1)) if len(data_clean) > 1 else 0,
            min_value=float(np.min(data_clean)),
            max_value=float(np.max(data_clean)),
            percentile_25=float(np.percentile(data_clean, 25)),
            percentile_75=float(np.percentile(data_clean, 75)),
            skewness=self._calculate_skewness(data_clean),
            kurtosis=self._calculate_kurtosis(data_clean),
            confidence_interval_95=ci_95
        )
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        if std_val == 0:
            return 0.0
        
        skew = np.mean(((data - mean_val) / std_val) ** 3)
        return float(skew)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        if std_val == 0:
            return 0.0
        
        kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
        return float(kurt)
    
    def _perform_normality_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform tests for normality."""
        # Simplified normality tests
        skewness = self._calculate_skewness(data)
        kurtosis = self._calculate_kurtosis(data)
        
        # Rough normality assessment
        is_approximately_normal = (abs(skewness) < 1.0 and abs(kurtosis) < 1.0)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'approximately_normal': is_approximately_normal,
            'normality_score': 1.0 / (1.0 + abs(skewness) + abs(kurtosis))  # Simple score
        }
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers in data."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(data) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': outliers.tolist()
        }
    
    def _calculate_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate correlations for multivariate data."""
        try:
            correlation_matrix = np.corrcoef(data.T)
            
            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'strong_correlations': self._find_strong_correlations(correlation_matrix),
                'average_correlation': float(np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])))
            }
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            return {}
    
    def _find_strong_correlations(self, corr_matrix: np.ndarray, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strongly correlated variable pairs."""
        strong_correlations = []
        
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                correlation = corr_matrix[i, j]
                if abs(correlation) >= threshold:
                    strong_correlations.append({
                        'variable_1': i,
                        'variable_2': j,
                        'correlation': float(correlation),
                        'strength': 'strong' if abs(correlation) >= 0.8 else 'moderate'
                    })
        
        return strong_correlations
    
    def _analyze_uncertainties(self, uncertainty_info: UncertaintyInfo) -> Dict[str, Any]:
        """Analyze uncertainty information."""
        relative_uncertainty = uncertainty_info.uncertainty / abs(uncertainty_info.value) if uncertainty_info.value != 0 else float('inf')
        
        # Classify uncertainty level
        if relative_uncertainty < 0.01:
            uncertainty_level = "very_low"
        elif relative_uncertainty < 0.05:
            uncertainty_level = "low"
        elif relative_uncertainty < 0.1:
            uncertainty_level = "moderate"
        elif relative_uncertainty < 0.2:
            uncertainty_level = "high"
        else:
            uncertainty_level = "very_high"
        
        return {
            'relative_uncertainty': relative_uncertainty,
            'relative_uncertainty_percent': relative_uncertainty * 100,
            'uncertainty_level': uncertainty_level,
            'confidence_level': uncertainty_info.confidence_level,
            'distribution_type': uncertainty_info.distribution,
            'source_count': len(uncertainty_info.sources)
        }
    
    def _perform_curve_fitting(self, x_data: np.ndarray, y_data: np.ndarray,
                             model_function: str, initial_params: Dict[str, float]) -> FittingResult:
        """Perform curve fitting (simplified implementation)."""
        # This is a simplified implementation
        # In practice, would use scipy.optimize.curve_fit or similar
        
        # Simulate fitting results
        fitted_params = {}
        for name, initial_value in initial_params.items():
            # Add some variation to simulate fitting
            fitted_value = initial_value * (1 + np.random.normal(0, 0.1))
            uncertainty = abs(fitted_value * 0.05)  # 5% uncertainty
            
            fitted_params[name] = UncertaintyInfo(
                value=fitted_value,
                uncertainty=uncertainty,
                uncertainty_type="fitting",
                confidence_level=0.68
            )
        
        # Calculate residuals and fitted values
        # Simplified: assume linear model y = a*x + b
        if 'a' in fitted_params and 'b' in fitted_params:
            fitted_values = fitted_params['a'].value * x_data + fitted_params['b'].value
        else:
            fitted_values = np.mean(y_data) * np.ones_like(x_data)
        
        residuals = y_data - fitted_values
        
        # Calculate goodness of fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        chi_squared = ss_res / len(residuals) if len(residuals) > 0 else 0
        
        # Generate covariance matrix (simplified)
        n_params = len(fitted_params)
        covariance_matrix = np.eye(n_params) * 0.01  # Simplified diagonal covariance
        
        return FittingResult(
            model_name=model_function,
            parameters=fitted_params,
            goodness_of_fit={'r_squared': r_squared, 'chi_squared': chi_squared},
            residuals=residuals.tolist(),
            fitted_values=fitted_values.tolist(),
            covariance_matrix=covariance_matrix.tolist(),
            parameter_correlations={}  # Simplified
        )
    
    def _compare_statistical_properties(self, results: List[PhysicsResult]) -> Dict[str, Any]:
        """Compare statistical properties of multiple results."""
        comparison = {'statistical_comparison': {}}
        
        # Extract numerical data from all results
        all_data = []
        for result in results:
            data = self._extract_numerical_data(result.data)
            if data is not None:
                all_data.append(data.flatten())
        
        if len(all_data) >= 2:
            # Calculate means and check for significant differences
            means = [np.mean(data) for data in all_data]
            stds = [np.std(data, ddof=1) if len(data) > 1 else 0 for data in all_data]
            
            comparison['statistical_comparison'].update({
                'means': means,
                'standard_deviations': stds,
                'mean_difference': max(means) - min(means),
                'coefficient_of_variation': [std/abs(mean) if mean != 0 else float('inf') 
                                           for mean, std in zip(means, stds)]
            })
        
        return comparison
    
    def _compare_uncertainties(self, results: List[PhysicsResult]) -> Dict[str, Any]:
        """Compare uncertainties across results."""
        comparison = {'uncertainty_comparison': {}}
        
        uncertainties = []
        for result in results:
            if result.uncertainty_info:
                uncertainties.append({
                    'result_id': result.result_id,
                    'absolute_uncertainty': result.uncertainty_info.uncertainty,
                    'relative_uncertainty': result.uncertainty_info.uncertainty / abs(result.uncertainty_info.value) if result.uncertainty_info.value != 0 else float('inf'),
                    'uncertainty_type': result.uncertainty_info.uncertainty_type
                })
        
        if uncertainties:
            comparison['uncertainty_comparison']['uncertainties'] = uncertainties
            comparison['uncertainty_comparison']['best_precision'] = min(uncertainties, key=lambda x: x['relative_uncertainty'])
            comparison['uncertainty_comparison']['worst_precision'] = max(uncertainties, key=lambda x: x['relative_uncertainty'])
        
        return comparison
    
    def _compare_temporal_evolution(self, results: List[PhysicsResult]) -> Dict[str, Any]:
        """Compare temporal evolution of results."""
        # Sort results by creation time
        sorted_results = sorted(results, key=lambda x: x.created_at)
        
        temporal_data = []
        for result in sorted_results:
            data = self._extract_numerical_data(result.data)
            if data is not None:
                temporal_data.append({
                    'timestamp': result.created_at,
                    'mean_value': float(np.mean(data)),
                    'std_value': float(np.std(data, ddof=1)) if len(data) > 1 else 0
                })
        
        return {'temporal_comparison': temporal_data}
    
    def _calculate_combined_uncertainty(self, uncertainty_sources: Dict[str, float],
                                      correlation_matrix: Optional[List[List[float]]]) -> float:
        """Calculate combined uncertainty from multiple sources."""
        if correlation_matrix is None:
            # Assume uncorrelated sources
            combined_variance = sum(u**2 for u in uncertainty_sources.values())
        else:
            # Account for correlations
            uncertainties = list(uncertainty_sources.values())
            combined_variance = 0
            
            for i, u_i in enumerate(uncertainties):
                for j, u_j in enumerate(uncertainties):
                    correlation = correlation_matrix[i][j] if i < len(correlation_matrix) and j < len(correlation_matrix[i]) else (1 if i == j else 0)
                    combined_variance += u_i * u_j * correlation
        
        return math.sqrt(max(0, combined_variance))
    
    def _propagate_uncertainties(self, data: Any, uncertainty_sources: Dict[str, float]) -> Dict[str, Any]:
        """Propagate uncertainties through calculations."""
        # Simplified uncertainty propagation
        total_relative_uncertainty = math.sqrt(sum((u / abs(list(uncertainty_sources.values())[0]))**2 
                                                 for u in uncertainty_sources.values()))
        
        return {
            'propagation_method': 'linear_approximation',
            'total_relative_uncertainty': total_relative_uncertainty,
            'dominant_source': max(uncertainty_sources.items(), key=lambda x: x[1])[0]
        }
    
    def _create_uncertainty_budget(self, uncertainty_sources: Dict[str, float]) -> Dict[str, Any]:
        """Create uncertainty budget showing contribution of each source."""
        total_variance = sum(u**2 for u in uncertainty_sources.values())
        total_uncertainty = math.sqrt(total_variance)
        
        budget = {}
        for source, uncertainty in uncertainty_sources.items():
            contribution_percent = (uncertainty**2 / total_variance * 100) if total_variance > 0 else 0
            budget[source] = {
                'uncertainty': uncertainty,
                'contribution_percent': contribution_percent,
                'is_dominant': contribution_percent > 50
            }
        
        return budget
    
    def _calculate_confidence_intervals(self, value: float, uncertainty: float) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for different confidence levels."""
        intervals = {}
        
        # Different confidence levels and their z-scores
        confidence_levels = {
            '68.3%': 1.0,    # 1 sigma
            '95.4%': 2.0,    # 2 sigma
            '99.7%': 3.0,    # 3 sigma
            '95.0%': 1.96,   # Standard 95%
            '99.0%': 2.576   # Standard 99%
        }
        
        for level, z_score in confidence_levels.items():
            margin = z_score * uncertainty
            intervals[level] = (value - margin, value + margin)
        
        return intervals
    
    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """Create summary of data structure and content."""
        summary = {'data_type': type(data).__name__}
        
        if isinstance(data, (list, tuple)):
            summary.update({
                'length': len(data),
                'element_types': list(set(type(item).__name__ for item in data[:100]))  # Sample first 100
            })
        elif isinstance(data, np.ndarray):
            summary.update({
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size': data.size
            })
        elif isinstance(data, dict):
            summary.update({
                'keys': list(data.keys()),
                'key_count': len(data)
            })
        elif isinstance(data, (int, float)):
            summary['value'] = data
        
        return summary
    
    def _generate_recommendations(self, result: PhysicsResult) -> List[str]:
        """Generate recommendations for improving result quality."""
        recommendations = []
        
        # Check data quality
        numerical_data = self._extract_numerical_data(result.data)
        if numerical_data is not None and len(numerical_data) > 0:
            if len(numerical_data) < 10:
                recommendations.append("Consider collecting more data points for better statistical significance")
            
            # Check for outliers
            outliers = self._detect_outliers(numerical_data)
            if outliers['outlier_percentage'] > 10:
                recommendations.append("High percentage of outliers detected - review data collection procedure")
        
        # Check uncertainty information
        if result.uncertainty_info is None:
            recommendations.append("Consider adding uncertainty information for better result interpretation")
        elif result.uncertainty_info:
            rel_uncertainty = result.uncertainty_info.uncertainty / abs(result.uncertainty_info.value) if result.uncertainty_info.value != 0 else 0
            if rel_uncertainty > 0.1:
                recommendations.append("High relative uncertainty - consider improving measurement precision")
        
        # Check metadata completeness
        if not result.references:
            recommendations.append("Add references to support the theoretical background")
        
        if not result.tags:
            recommendations.append("Add descriptive tags for better result organization")
        
        return recommendations
    
    def _validate_result_quality(self, result: PhysicsResult) -> Dict[str, bool]:
        """Validate result quality and completeness."""
        validation = {
            'has_description': bool(result.description),
            'has_metadata': bool(result.metadata),
            'has_uncertainty_info': result.uncertainty_info is not None,
            'has_statistical_summary': result.statistical_summary is not None,
            'has_references': bool(result.references),
            'has_tags': bool(result.tags),
            'data_is_numerical': self._extract_numerical_data(result.data) is not None
        }
        
        # Overall quality score
        validation['quality_score'] = sum(validation.values()) / len(validation)
        validation['quality_level'] = 'high' if validation['quality_score'] > 0.8 else ('medium' if validation['quality_score'] > 0.5 else 'low')
        
        return validation
    
    def _infer_uncertainty(self, data: np.ndarray) -> Optional[UncertaintyInfo]:
        """Infer uncertainty information from data."""
        if len(data) > 1:
            std_uncertainty = float(np.std(data, ddof=1) / np.sqrt(len(data)))
            mean_value = float(np.mean(data))
            
            return UncertaintyInfo(
                value=mean_value,
                uncertainty=std_uncertainty,
                uncertainty_type="statistical",
                confidence_level=0.68,
                distribution="gaussian",
                sources=["statistical_variation"]
            )
        return None
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for export."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple, dict, str, int, float, bool)) or data is None:
            return data
        else:
            return str(data)

# Global results display instance
physics_results_display = PhysicsResultsDisplay()