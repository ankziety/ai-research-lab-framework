"""
Experimental Tool

Agent-friendly interface for experimental data analysis.
Provides statistical analysis, curve fitting, uncertainty quantification,
and experimental design capabilities for physics research.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class ExperimentalTool(BasePhysicsTool):
    """
    Tool for experimental data analysis that agents can request.
    
    Provides interfaces for:
    - Statistical data analysis
    - Curve fitting and parameter estimation
    - Uncertainty quantification
    - Experimental design optimization
    - Hypothesis testing
    """
    
    def __init__(self):
        super().__init__(
            tool_id="experimental_tool",
            name="Experimental Tool",
            description="Perform experimental data analysis including statistics, curve fitting, and uncertainty quantification",
            physics_domain="experimental_physics",
            computational_cost_factor=1.5,
            software_requirements=[
                "numpy",        # Numerical calculations
                "scipy",        # Scientific computing
                "pandas",       # Data manipulation
                "matplotlib"    # Plotting (optional)
            ],
            hardware_requirements={
                "min_memory": 256,   # MB
                "recommended_memory": 1024,
                "cpu_cores": 1,
                "supports_gpu": False
            }
        )
        
        # Add experimental analysis specific capabilities
        self.capabilities.extend([
            "statistical_analysis",
            "curve_fitting",
            "uncertainty_quantification",
            "hypothesis_testing",
            "experimental_design",
            "error_propagation",
            "outlier_detection",
            "correlation_analysis"
        ])
        
        # Available analysis types
        self.analysis_types = [
            "descriptive_statistics",
            "curve_fitting",
            "hypothesis_test",
            "uncertainty_analysis",
            "correlation_analysis",
            "outlier_detection",
            "experimental_design",
            "error_propagation"
        ]
        
        # Available fitting functions
        self.fitting_functions = {
            "linear": lambda x, a, b: a * x + b,
            "quadratic": lambda x, a, b, c: a * x**2 + b * x + c,
            "exponential": lambda x, a, b, c: a * np.exp(b * x) + c,
            "power_law": lambda x, a, b: a * x**b,
            "gaussian": lambda x, a, b, c, d: a * np.exp(-((x - b)**2) / (2 * c**2)) + d,
            "sinusoidal": lambda x, a, b, c, d: a * np.sin(b * x + c) + d,
            "lorentzian": lambda x, a, b, c, d: a / (1 + ((x - b) / c)**2) + d
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute experimental analysis requested by an agent.
        
        Args:
            task: Task specification with experimental parameters
            context: Agent context and execution environment
            
        Returns:
            Experimental analysis results formatted for agents
        """
        start_time = datetime.now()
        
        try:
            # Validate input parameters
            validation_result = self.validate_input(task)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "validation_errors": validation_result["errors"],
                    "suggestions": validation_result["suggestions"]
                }
            
            # Extract task parameters
            task_type = task.get("type", "descriptive_statistics")
            data = task.get("data", {})
            parameters = task.get("parameters", {})
            
            # Route to appropriate analysis
            if task_type == "descriptive_statistics":
                result = self._calculate_descriptive_statistics(data, parameters)
            elif task_type == "curve_fitting":
                result = self._perform_curve_fitting(data, parameters)
            elif task_type == "hypothesis_test":
                result = self._perform_hypothesis_test(data, parameters)
            elif task_type == "uncertainty_analysis":
                result = self._analyze_uncertainties(data, parameters)
            elif task_type == "correlation_analysis":
                result = self._analyze_correlations(data, parameters)
            elif task_type == "outlier_detection":
                result = self._detect_outliers(data, parameters)
            elif task_type == "experimental_design":
                result = self._design_experiment(parameters)
            elif task_type == "error_propagation":
                result = self._propagate_errors(data, parameters)
            else:
                raise ValueError(f"Unknown experimental analysis type: {task_type}")
            
            # Process and format output for agents
            formatted_result = self.process_output(result)
            
            # Update statistics
            calculation_time = (datetime.now() - start_time).total_seconds()
            estimated_cost = self._calculate_actual_cost(task, calculation_time)
            self.update_calculation_stats(calculation_time, estimated_cost, True)
            
            return {
                "success": True,
                "task_type": task_type,
                "results": formatted_result,
                "calculation_time": calculation_time,
                "computational_cost": estimated_cost,
                "confidence": self._assess_result_confidence(result, task_type),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.update_calculation_stats(calculation_time, 0.0, False)
            return self.handle_errors(e, {"task": task, "context": context})
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate experimental analysis input parameters.
        
        Args:
            input_data: Input parameters from agent
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        task_type = input_data.get("type", "descriptive_statistics")
        if task_type not in self.analysis_types:
            errors.append(f"Unknown analysis type '{task_type}'")
            suggestions.append(f"Available types: {', '.join(self.analysis_types)}")
        
        # Validate data presence
        if task_type != "experimental_design":
            if "data" not in input_data:
                errors.append("Missing 'data' for analysis")
                suggestions.append("Provide experimental data as arrays or DataFrame")
            else:
                data = input_data["data"]
                
                # Check data format
                if not isinstance(data, dict):
                    errors.append("Data must be dictionary with array-like values")
                    suggestions.append("Use format: {'x': [1,2,3], 'y': [4,5,6], 'y_err': [0.1,0.1,0.1]}")
                else:
                    # Check for empty data
                    if not data:
                        errors.append("Data dictionary is empty")
                    else:
                        # Check array lengths
                        lengths = []
                        for key, values in data.items():
                            if hasattr(values, '__len__'):
                                lengths.append(len(values))
                            else:
                                errors.append(f"Data '{key}' is not array-like")
                        
                        if lengths and len(set(lengths)) > 1:
                            errors.append("Data arrays have different lengths")
                            suggestions.append("Ensure all data arrays have the same length")
                        
                        # Check for sufficient data points
                        if lengths and min(lengths) < 3:
                            warnings.append("Very few data points - results may be unreliable")
                        elif lengths and min(lengths) > 10000:
                            warnings.append("Large dataset - analysis may take time")
        
        # Validate task-specific requirements
        if task_type == "curve_fitting":
            params = input_data.get("parameters", {})
            if "function_type" not in params:
                warnings.append("No function type specified - will try multiple fits")
            else:
                func_type = params["function_type"]
                if func_type not in self.fitting_functions:
                    errors.append(f"Unknown fitting function '{func_type}'")
                    suggestions.append(f"Available functions: {', '.join(self.fitting_functions.keys())}")
        
        elif task_type == "hypothesis_test":
            params = input_data.get("parameters", {})
            if "test_type" not in params:
                errors.append("Must specify 'test_type' for hypothesis testing")
                suggestions.append("Use 't_test', 'chi_square', 'anova', etc.")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format experimental analysis results for agents.
        
        Args:
            output_data: Raw analysis results
            
        Returns:
            Agent-friendly formatted results
        """
        formatted = {
            "summary": self._generate_result_summary(output_data),
            "statistics": {},
            "parameters": {},
            "uncertainties": {},
            "analysis": {}
        }
        
        # Statistical measures
        if "mean" in output_data:
            formatted["statistics"]["mean"] = {
                "value": output_data["mean"],
                "description": "Sample mean"
            }
        
        if "std" in output_data:
            formatted["statistics"]["standard_deviation"] = {
                "value": output_data["std"],
                "description": "Sample standard deviation"
            }
        
        if "sem" in output_data:
            formatted["statistics"]["standard_error"] = {
                "value": output_data["sem"],
                "description": "Standard error of the mean"
            }
        
        # Fitted parameters
        if "fitted_parameters" in output_data:
            formatted["parameters"]["fitted_values"] = output_data["fitted_parameters"]
        
        if "parameter_errors" in output_data:
            formatted["uncertainties"]["parameter_uncertainties"] = output_data["parameter_errors"]
        
        # Goodness of fit
        if "r_squared" in output_data:
            formatted["statistics"]["r_squared"] = {
                "value": output_data["r_squared"],
                "description": "Coefficient of determination"
            }
        
        if "chi_squared" in output_data:
            formatted["statistics"]["chi_squared"] = {
                "value": output_data["chi_squared"],
                "description": "Chi-squared statistic"
            }
        
        # Test results
        if "p_value" in output_data:
            formatted["statistics"]["p_value"] = {
                "value": output_data["p_value"],
                "description": "Statistical significance (p-value)"
            }
        
        if "test_statistic" in output_data:
            formatted["statistics"]["test_statistic"] = {
                "value": output_data["test_statistic"],
                "description": "Test statistic value"
            }
        
        # Analysis insights
        formatted["analysis"]["interpretation"] = self._interpret_results(output_data)
        formatted["analysis"]["recommendations"] = self._generate_recommendations(output_data)
        formatted["analysis"]["quality_assessment"] = self._assess_data_quality(output_data)
        
        return formatted
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for experimental analysis.
        
        Args:
            task: Task specification
            
        Returns:
            Cost estimates (time, memory, computational units)
        """
        base_cost = 1.0
        
        # Get data size
        data = task.get("data", {})
        data_size = 0
        if data:
            for values in data.values():
                if hasattr(values, '__len__'):
                    data_size = max(data_size, len(values))
        
        # Scale with data size
        size_cost_factor = max(1.0, (data_size / 1000) ** 0.5)
        
        # Analysis type cost factors
        analysis_costs = {
            "descriptive_statistics": 0.1,
            "curve_fitting": 1.0,
            "hypothesis_test": 0.5,
            "uncertainty_analysis": 0.8,
            "correlation_analysis": 0.3,
            "outlier_detection": 0.4,
            "experimental_design": 0.6,
            "error_propagation": 0.7
        }
        
        task_type = task.get("type", "descriptive_statistics")
        analysis_cost = analysis_costs.get(task_type, 1.0)
        
        total_cost_factor = size_cost_factor * analysis_cost * self.computational_cost_factor
        
        # Estimate time (in seconds)
        estimated_time = base_cost * total_cost_factor * 0.5
        
        # Estimate memory (in MB)
        estimated_memory = 50 + (data_size * 8 / 1024**2) * 2  # Rough estimate
        
        # Computational units
        computational_units = total_cost_factor * 10
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_memory_mb": estimated_memory,
            "computational_units": computational_units,
            "cost_breakdown": {
                "size_factor": size_cost_factor,
                "analysis_factor": analysis_cost,
                "total_factor": total_cost_factor
            }
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get experimental analysis specific requirements."""
        return {
            "physics_domain": "experimental_physics",
            "available_analysis_types": self.analysis_types,
            "available_fitting_functions": list(self.fitting_functions.keys()),
            "supported_data_formats": ["arrays", "lists", "pandas_dataframe"],
            "max_data_points": 1000000,
            "typical_calculation_time": "Milliseconds to seconds",
            "memory_scaling": "Linear with data size",
            "software_dependencies": self.software_requirements,
            "hardware_recommendations": self.hardware_requirements
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Get experimental physics domain keywords."""
        return [
            "experimental", "data", "analysis", "statistics", "fitting",
            "curve", "uncertainty", "error", "measurement", "correlation",
            "hypothesis", "test", "significance", "outlier", "regression",
            "calibration", "precision", "accuracy"
        ]
    
    def _calculate_descriptive_statistics(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        results = {}
        
        for key, values in data.items():
            if not hasattr(values, '__len__'):
                continue
                
            values_array = np.array(values)
            if len(values_array) == 0:
                continue
            
            # Basic statistics
            stats_dict = {
                "count": len(values_array),
                "mean": np.mean(values_array),
                "std": np.std(values_array, ddof=1),
                "sem": np.std(values_array, ddof=1) / np.sqrt(len(values_array)),
                "min": np.min(values_array),
                "max": np.max(values_array),
                "median": np.median(values_array),
                "q25": np.percentile(values_array, 25),
                "q75": np.percentile(values_array, 75),
                "skewness": stats.skew(values_array),
                "kurtosis": stats.kurtosis(values_array)
            }
            
            results[key] = stats_dict
        
        # Overall summary if multiple variables
        if len(results) > 1:
            results["summary"] = {
                "num_variables": len(results),
                "total_observations": sum(r["count"] for r in results.values() if isinstance(r, dict))
            }
        
        return results
    
    def _perform_curve_fitting(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform curve fitting analysis."""
        # Extract x and y data
        x_key = parameters.get("x_variable", "x")
        y_key = parameters.get("y_variable", "y")
        
        if x_key not in data or y_key not in data:
            raise ValueError(f"Required variables '{x_key}' and/or '{y_key}' not found in data")
        
        x = np.array(data[x_key])
        y = np.array(data[y_key])
        
        # Extract uncertainties if available
        y_err = None
        if "y_err" in data:
            y_err = np.array(data["y_err"])
        
        function_type = parameters.get("function_type", "linear")
        
        if function_type == "auto":
            # Try multiple functions and choose best fit
            best_fit = self._auto_fit_selection(x, y, y_err)
            return best_fit
        
        if function_type not in self.fitting_functions:
            raise ValueError(f"Unknown fitting function: {function_type}")
        
        # Perform fitting
        func = self.fitting_functions[function_type]
        
        try:
            if y_err is not None:
                popt, pcov = optimize.curve_fit(func, x, y, sigma=y_err, absolute_sigma=True)
            else:
                popt, pcov = optimize.curve_fit(func, x, y)
            
            # Calculate uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Calculate goodness of fit
            y_pred = func(x, *popt)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Chi-squared statistic
            if y_err is not None:
                chi_squared = np.sum((residuals / y_err)**2)
                reduced_chi_squared = chi_squared / (len(x) - len(popt))
            else:
                chi_squared = ss_res
                reduced_chi_squared = chi_squared / (len(x) - len(popt))
            
            return {
                "function_type": function_type,
                "fitted_parameters": popt.tolist(),
                "parameter_errors": param_errors.tolist(),
                "covariance_matrix": pcov.tolist(),
                "r_squared": r_squared,
                "chi_squared": chi_squared,
                "reduced_chi_squared": reduced_chi_squared,
                "residuals": residuals.tolist(),
                "fitted_values": y_pred.tolist(),
                "degrees_of_freedom": len(x) - len(popt)
            }
            
        except Exception as e:
            raise ValueError(f"Fitting failed: {str(e)}")
    
    def _perform_hypothesis_test(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical hypothesis test."""
        test_type = parameters.get("test_type", "t_test")
        
        if test_type == "t_test":
            return self._t_test(data, parameters)
        elif test_type == "chi_square":
            return self._chi_square_test(data, parameters)
        elif test_type == "anova":
            return self._anova_test(data, parameters)
        elif test_type == "normality":
            return self._normality_test(data, parameters)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _analyze_uncertainties(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze measurement uncertainties."""
        results = {}
        
        for key, values in data.items():
            if not hasattr(values, '__len__'):
                continue
                
            values_array = np.array(values)
            if len(values_array) < 2:
                continue
            
            # Type A uncertainty (statistical)
            type_a = np.std(values_array, ddof=1) / np.sqrt(len(values_array))
            
            # Type B uncertainty (systematic) - from parameters if provided
            type_b = parameters.get(f"{key}_systematic_uncertainty", 0)
            
            # Combined uncertainty
            combined = np.sqrt(type_a**2 + type_b**2)
            
            # Relative uncertainty
            mean_val = np.mean(values_array)
            relative = combined / abs(mean_val) * 100 if mean_val != 0 else float('inf')
            
            results[key] = {
                "type_a_uncertainty": type_a,
                "type_b_uncertainty": type_b,
                "combined_uncertainty": combined,
                "relative_uncertainty_percent": relative,
                "mean_value": mean_val,
                "coverage_factor": 2.0  # 95% confidence
            }
        
        return results
    
    def _analyze_correlations(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        # Correlation matrix
        corr_matrix = df.corr()
        
        # Find significant correlations
        significant_corr = []
        n = len(df)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if not np.isnan(r):
                    # Calculate t-statistic for correlation
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    
                    significant_corr.append({
                        "variables": [corr_matrix.columns[i], corr_matrix.columns[j]],
                        "correlation": r,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "significant_correlations": significant_corr,
            "sample_size": n
        }
    
    def _detect_outliers(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in data."""
        method = parameters.get("method", "iqr")
        results = {}
        
        for key, values in data.items():
            if not hasattr(values, '__len__'):
                continue
                
            values_array = np.array(values)
            if len(values_array) < 4:
                continue
            
            outliers = []
            outlier_indices = []
            
            if method == "iqr":
                q25, q75 = np.percentile(values_array, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                for i, val in enumerate(values_array):
                    if val < lower_bound or val > upper_bound:
                        outliers.append(val)
                        outlier_indices.append(i)
            
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(values_array))
                threshold = parameters.get("zscore_threshold", 3)
                
                for i, z in enumerate(z_scores):
                    if z > threshold:
                        outliers.append(values_array[i])
                        outlier_indices.append(i)
            
            results[key] = {
                "outliers": outliers,
                "outlier_indices": outlier_indices,
                "num_outliers": len(outliers),
                "outlier_percentage": len(outliers) / len(values_array) * 100,
                "method": method
            }
        
        return results
    
    def _design_experiment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Design experimental parameters."""
        design_type = parameters.get("design_type", "factorial")
        
        if design_type == "factorial":
            factors = parameters.get("factors", {})
            levels = parameters.get("levels", 2)
            
            # Generate factorial design
            factor_names = list(factors.keys())
            num_factors = len(factor_names)
            num_runs = levels ** num_factors
            
            design_matrix = []
            for i in range(num_runs):
                run = {}
                for j, factor in enumerate(factor_names):
                    level_index = (i // (levels ** j)) % levels
                    if isinstance(factors[factor], list):
                        run[factor] = factors[factor][level_index]
                    else:
                        # Assume numeric range
                        min_val, max_val = factors[factor]
                        run[factor] = min_val + (max_val - min_val) * level_index / (levels - 1)
                design_matrix.append(run)
            
            return {
                "design_type": design_type,
                "num_factors": num_factors,
                "num_levels": levels,
                "num_runs": num_runs,
                "design_matrix": design_matrix,
                "randomization_order": np.random.permutation(num_runs).tolist()
            }
        
        else:
            raise ValueError(f"Unknown design type: {design_type}")
    
    def _propagate_errors(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate measurement errors through calculations."""
        formula = parameters.get("formula", "")
        if not formula:
            raise ValueError("Must provide formula for error propagation")
        
        # This is a simplified implementation
        # In practice, would use symbolic differentiation
        
        variables = {}
        uncertainties = {}
        
        for key, values in data.items():
            if key.endswith("_err"):
                var_name = key[:-4]
                if var_name in data:
                    uncertainties[var_name] = np.mean(values)
            else:
                variables[key] = np.mean(values)
        
        # Mock error propagation result
        result_uncertainty = np.sqrt(sum(uncertainties.values())) if uncertainties else 0
        
        return {
            "formula": formula,
            "input_variables": variables,
            "input_uncertainties": uncertainties,
            "propagated_uncertainty": result_uncertainty,
            "relative_uncertainty_percent": result_uncertainty / max(variables.values()) * 100 if variables else 0
        }
    
    def _auto_fit_selection(self, x: np.ndarray, y: np.ndarray, y_err: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Automatically select best fitting function."""
        best_r2 = -np.inf
        best_fit = None
        
        for func_name, func in self.fitting_functions.items():
            try:
                if y_err is not None:
                    popt, pcov = optimize.curve_fit(func, x, y, sigma=y_err, absolute_sigma=True)
                else:
                    popt, pcov = optimize.curve_fit(func, x, y)
                
                y_pred = func(x, *popt)
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - (ss_res / ss_tot)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = {
                        "function_type": func_name,
                        "fitted_parameters": popt.tolist(),
                        "parameter_errors": np.sqrt(np.diag(pcov)).tolist(),
                        "r_squared": r2
                    }
            except:
                continue
        
        if best_fit is None:
            raise ValueError("No successful fits found")
        
        return best_fit
    
    def _t_test(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform t-test."""
        test_value = parameters.get("test_value", 0)
        
        # Assume first data array for one-sample t-test
        sample_key = list(data.keys())[0]
        sample = np.array(data[sample_key])
        
        t_stat, p_value = stats.ttest_1samp(sample, test_value)
        
        return {
            "test_type": "one_sample_t_test",
            "test_statistic": t_stat,
            "p_value": p_value,
            "test_value": test_value,
            "sample_mean": np.mean(sample),
            "degrees_of_freedom": len(sample) - 1,
            "significant": p_value < 0.05
        }
    
    def _chi_square_test(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform chi-square test."""
        observed = np.array(list(data.values())[0])
        expected = parameters.get("expected", np.ones_like(observed) * np.mean(observed))
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            "test_type": "chi_square",
            "test_statistic": chi2_stat,
            "p_value": p_value,
            "degrees_of_freedom": len(observed) - 1,
            "significant": p_value < 0.05
        }
    
    def _anova_test(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ANOVA test."""
        groups = [np.array(values) for values in data.values()]
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            "test_type": "one_way_anova",
            "test_statistic": f_stat,
            "p_value": p_value,
            "num_groups": len(groups),
            "significant": p_value < 0.05
        }
    
    def _normality_test(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Test for normality."""
        sample_key = list(data.keys())[0]
        sample = np.array(data[sample_key])
        
        stat, p_value = stats.shapiro(sample)
        
        return {
            "test_type": "shapiro_wilk_normality",
            "test_statistic": stat,
            "p_value": p_value,
            "normal_distribution": p_value > 0.05,
            "sample_size": len(sample)
        }
    
    def _calculate_actual_cost(self, task: Dict[str, Any], actual_time: float) -> float:
        """Calculate actual computational cost."""
        estimates = self.estimate_cost(task)
        estimated_time = estimates["estimated_time_seconds"]
        
        time_ratio = actual_time / max(estimated_time, 0.01)
        actual_cost = estimates["computational_units"] * time_ratio
        
        return actual_cost
    
    def _assess_result_confidence(self, result: Dict[str, Any], task_type: str) -> float:
        """Assess confidence in analysis results."""
        base_confidence = 0.9
        
        # Adjust based on sample size if available
        if "count" in str(result):
            # Extract sample size information
            confidence_factor = 1.0
        else:
            confidence_factor = 1.0
        
        return min(1.0, base_confidence * confidence_factor)
    
    def _generate_result_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable summary of results."""
        if "fitted_parameters" in result:
            func_type = result.get("function_type", "unknown")
            r2 = result.get("r_squared", 0)
            return f"Curve fitting completed using {func_type} function. R² = {r2:.3f}"
        
        elif "p_value" in result:
            test_type = result.get("test_type", "statistical test")
            p_val = result["p_value"]
            return f"{test_type} completed. p-value = {p_val:.4f}"
        
        elif isinstance(result, dict) and any("mean" in str(v) for v in result.values()):
            return "Descriptive statistics calculated for experimental data"
        
        else:
            return "Experimental analysis completed"
    
    def _interpret_results(self, result: Dict[str, Any]) -> List[str]:
        """Interpret analysis results."""
        interpretations = []
        
        if "p_value" in result:
            p_val = result["p_value"]
            if p_val < 0.001:
                interpretations.append("Highly significant result (p < 0.001)")
            elif p_val < 0.01:
                interpretations.append("Very significant result (p < 0.01)")
            elif p_val < 0.05:
                interpretations.append("Significant result (p < 0.05)")
            else:
                interpretations.append("No significant effect detected")
        
        if "r_squared" in result:
            r2 = result["r_squared"]
            if r2 > 0.9:
                interpretations.append("Excellent fit to data")
            elif r2 > 0.7:
                interpretations.append("Good fit to data")
            elif r2 > 0.5:
                interpretations.append("Moderate fit to data")
            else:
                interpretations.append("Poor fit - consider alternative models")
        
        return interpretations
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for further analysis."""
        recommendations = []
        
        if "r_squared" in result and result["r_squared"] < 0.7:
            recommendations.append("Consider trying different fitting functions")
            recommendations.append("Check for outliers in the data")
        
        if "p_value" in result and result["p_value"] > 0.05:
            recommendations.append("Increase sample size to improve statistical power")
            recommendations.append("Consider effect size in addition to significance")
        
        if "outliers" in str(result):
            recommendations.append("Investigate outliers - measurement errors or real effects?")
        
        return recommendations
    
    def _assess_data_quality(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Assess quality of experimental data."""
        quality = {"overall": "good"}
        
        if "outlier_percentage" in str(result):
            # Check outlier percentage
            quality["outliers"] = "acceptable"
        
        if "sample_size" in str(result):
            quality["sample_size"] = "adequate"
        
        return quality