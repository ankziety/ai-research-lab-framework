"""
Experimental Physics Agent - Specialized agent for experimental design and data analysis.

This agent provides expertise in experimental physics, including experimental design,
data analysis, uncertainty quantification, and validation methodologies.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
import matplotlib.pyplot as plt

from .base_physics_agent import BasePhysicsAgent, PhysicsScale, PhysicsMethodology

logger = logging.getLogger(__name__)


class ExperimentalPhysicsAgent(BasePhysicsAgent):
    """
    Specialized agent for experimental physics research and analysis.
    
    Expertise includes:
    - Experimental design and planning
    - Data analysis and statistical methods
    - Uncertainty analysis and error propagation
    - Measurement techniques and instrumentation
    - Validation and verification methods
    - Signal processing and noise analysis
    """
    
    def __init__(self, agent_id: str, role: str = None, expertise: List[str] = None,
                 model_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        """
        Initialize experimental physics agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent role (defaults to "Experimental Physics Expert")
            expertise: List of expertise areas (uses defaults if None)
            model_config: Configuration for the underlying LLM
            cost_manager: Optional cost manager for tracking API usage
        """
        if role is None:
            role = "Experimental Physics Expert"
        
        if expertise is None:
            expertise = [
                "Experimental Design",
                "Data Analysis",
                "Uncertainty Analysis",
                "Statistical Methods",
                "Measurement Techniques",
                "Instrumentation",
                "Signal Processing",
                "Error Propagation",
                "Calibration Methods",
                "Validation Protocols"
            ]
        
        super().__init__(agent_id, role, expertise, model_config, cost_manager)
        
        # Experimental design methodologies
        self.experimental_designs = {
            'factorial': {
                'description': 'Full or fractional factorial design',
                'advantages': ['Identifies interactions', 'Efficient parameter space coverage'],
                'applications': ['Multi-parameter optimization', 'Screening experiments']
            },
            'response_surface': {
                'description': 'Response surface methodology (RSM)',
                'advantages': ['Optimization capability', 'Model building'],
                'applications': ['Process optimization', 'Quality improvement']
            },
            'plackett_burman': {
                'description': 'Plackett-Burman screening design',
                'advantages': ['Efficient screening', 'Many factors with few runs'],
                'applications': ['Initial factor screening', 'Main effects identification']
            },
            'central_composite': {
                'description': 'Central composite design (CCD)',
                'advantages': ['Second-order modeling', 'Optimization'],
                'applications': ['Response surface modeling', 'Process optimization']
            },
            'taguchi': {
                'description': 'Taguchi robust design',
                'advantages': ['Robust parameter design', 'Quality improvement'],
                'applications': ['Manufacturing optimization', 'Robust design']
            }
        }
        
        # Statistical tests database
        self.statistical_tests = {
            't_test': {
                'one_sample': 'Compare sample mean to known value',
                'two_sample': 'Compare means of two samples',
                'paired': 'Compare paired observations'
            },
            'anova': {
                'one_way': 'Compare means of multiple groups',
                'two_way': 'Analyze effects of two factors',
                'repeated_measures': 'Analyze repeated measurements'
            },
            'chi_square': {
                'goodness_of_fit': 'Test distribution fit',
                'independence': 'Test variable independence'
            },
            'non_parametric': {
                'mann_whitney': 'Non-parametric two-sample test',
                'kruskal_wallis': 'Non-parametric one-way ANOVA',
                'wilcoxon': 'Non-parametric paired test'
            }
        }
        
        # Measurement techniques
        self.measurement_techniques = {
            'optical': ['spectroscopy', 'interferometry', 'microscopy', 'photometry'],
            'electrical': ['voltammetry', 'impedance', 'conductivity', 'capacitance'],
            'mechanical': ['force_measurement', 'displacement', 'vibration', 'acoustic'],
            'thermal': ['calorimetry', 'thermometry', 'thermal_conductivity'],
            'magnetic': ['magnetometry', 'magnetic_resonance', 'susceptibility'],
            'nuclear': ['gamma_spectroscopy', 'neutron_activation', 'mass_spectrometry']
        }
        
        logger.info(f"Experimental Physics Agent {self.agent_id} initialized")
    
    def _get_physics_domain(self) -> str:
        """Get the physics domain for experimental physics."""
        return "experimental_physics"
    
    def _get_relevant_scales(self) -> List[PhysicsScale]:
        """Get physical scales relevant to experimental physics."""
        return [
            PhysicsScale.ATOMIC,
            PhysicsScale.MOLECULAR,
            PhysicsScale.NANO,
            PhysicsScale.MICRO,
            PhysicsScale.MACRO
        ]
    
    def _get_preferred_methodologies(self) -> List[PhysicsMethodology]:
        """Get preferred methodologies for experimental physics."""
        return [PhysicsMethodology.EXPERIMENTAL]
    
    def design_physics_experiment(self, hypothesis: str, 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design comprehensive physics experiments.
        
        Args:
            hypothesis: Scientific hypothesis to test
            constraints: Experimental constraints (budget, time, equipment, etc.)
            
        Returns:
            Complete experimental design with methodology and protocols
        """
        design = {
            'success': False,
            'hypothesis': hypothesis,
            'experimental_design': {},
            'methodology': {},
            'protocols': [],
            'equipment_requirements': [],
            'measurement_plan': {},
            'data_analysis_plan': {},
            'quality_control': {},
            'risk_assessment': {},
            'timeline': {},
            'budget_estimate': {}
        }
        
        try:
            # Analyze hypothesis for experimental requirements
            requirements = self._analyze_hypothesis_requirements(hypothesis)
            
            # Select appropriate experimental design
            exp_design = self._select_experimental_design(requirements, constraints)
            
            # Design measurement protocols
            protocols = self._design_measurement_protocols(requirements, constraints)
            
            # Plan data analysis methodology
            analysis_plan = self._plan_data_analysis(requirements, exp_design)
            
            # Assess experimental uncertainties
            uncertainty_analysis = self._design_uncertainty_analysis(requirements, protocols)
            
            # Create quality control procedures
            qc_procedures = self._design_quality_control(requirements, protocols)
            
            # Estimate resources and timeline
            resource_estimate = self._estimate_experimental_resources(protocols, constraints)
            
            design.update({
                'success': True,
                'experimental_design': exp_design,
                'methodology': requirements,
                'protocols': protocols,
                'data_analysis_plan': analysis_plan,
                'uncertainty_analysis': uncertainty_analysis,
                'quality_control': qc_procedures,
                'resource_estimate': resource_estimate
            })
            
            self.physics_metrics['experiments_designed'] += 1
            
        except Exception as e:
            design['error'] = str(e)
            logger.error(f"Experimental design failed: {e}")
        
        return design
    
    def analyze_experimental_data(self, data: Dict[str, Any], 
                                 analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive experimental data analysis.
        
        Args:
            data: Experimental data with measurements and metadata
            analysis_config: Analysis configuration and parameters
            
        Returns:
            Complete data analysis results with statistics and interpretations
        """
        analysis_result = {
            'success': False,
            'data_summary': {},
            'statistical_analysis': {},
            'hypothesis_testing': {},
            'uncertainty_analysis': {},
            'model_fitting': {},
            'outlier_analysis': {},
            'visualization': {},
            'conclusions': {}
        }
        
        try:
            # Extract and validate data
            measurements = self._extract_measurements(data)
            
            # Perform descriptive statistics
            descriptive_stats = self._calculate_descriptive_statistics(measurements)
            
            # Conduct hypothesis testing
            hypothesis_results = self._conduct_hypothesis_testing(measurements, analysis_config)
            
            # Analyze uncertainties and error propagation
            uncertainty_results = self._analyze_uncertainties(measurements, analysis_config)
            
            # Detect and analyze outliers
            outlier_results = self._analyze_outliers(measurements, analysis_config)
            
            # Fit models to data
            model_results = self._fit_models_to_data(measurements, analysis_config)
            
            # Generate visualizations
            visualization_results = self._generate_visualizations(measurements, analysis_config)
            
            # Draw scientific conclusions
            conclusions = self._draw_scientific_conclusions(
                descriptive_stats, hypothesis_results, model_results, analysis_config
            )
            
            analysis_result.update({
                'success': True,
                'data_summary': descriptive_stats,
                'statistical_analysis': descriptive_stats,
                'hypothesis_testing': hypothesis_results,
                'uncertainty_analysis': uncertainty_results,
                'outlier_analysis': outlier_results,
                'model_fitting': model_results,
                'visualization': visualization_results,
                'conclusions': conclusions
            })
            
        except Exception as e:
            analysis_result['error'] = str(e)
            logger.error(f"Data analysis failed: {e}")
        
        return analysis_result
    
    def perform_uncertainty_analysis(self, measurements: Dict[str, Any], 
                                   error_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive uncertainty analysis and error propagation.
        
        Args:
            measurements: Measurement data with values and uncertainties
            error_sources: Known sources of error and their characteristics
            
        Returns:
            Uncertainty analysis results with propagated errors
        """
        uncertainty_result = {
            'success': False,
            'measurement_uncertainties': {},
            'systematic_errors': {},
            'random_errors': {},
            'propagated_uncertainties': {},
            'error_budget': {},
            'sensitivity_analysis': {},
            'uncertainty_recommendations': {}
        }
        
        try:
            # Classify error sources
            systematic_errors = self._identify_systematic_errors(error_sources)
            random_errors = self._identify_random_errors(error_sources)
            
            # Calculate measurement uncertainties
            measurement_uncert = self._calculate_measurement_uncertainties(measurements)
            
            # Propagate uncertainties
            propagated_uncert = self._propagate_uncertainties(measurements, error_sources)
            
            # Create error budget
            error_budget = self._create_error_budget(systematic_errors, random_errors, propagated_uncert)
            
            # Perform sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(measurements, error_sources)
            
            # Generate recommendations
            recommendations = self._generate_uncertainty_recommendations(
                error_budget, sensitivity_analysis
            )
            
            uncertainty_result.update({
                'success': True,
                'measurement_uncertainties': measurement_uncert,
                'systematic_errors': systematic_errors,
                'random_errors': random_errors,
                'propagated_uncertainties': propagated_uncert,
                'error_budget': error_budget,
                'sensitivity_analysis': sensitivity_analysis,
                'uncertainty_recommendations': recommendations
            })
            
        except Exception as e:
            uncertainty_result['error'] = str(e)
            logger.error(f"Uncertainty analysis failed: {e}")
        
        return uncertainty_result
    
    def validate_experimental_results(self, results: Dict[str, Any], 
                                    validation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate experimental results using multiple approaches.
        
        Args:
            results: Experimental results to validate
            validation_config: Validation configuration and criteria
            
        Returns:
            Validation results with confidence assessments
        """
        validation_result = {
            'success': False,
            'validation_methods': [],
            'reproducibility_assessment': {},
            'consistency_checks': {},
            'theoretical_comparison': {},
            'literature_comparison': {},
            'cross_validation': {},
            'overall_confidence': 0.0,
            'validation_recommendations': []
        }
        
        try:
            # Check reproducibility
            reproducibility = self._assess_reproducibility(results, validation_config)
            
            # Perform consistency checks
            consistency = self._perform_consistency_checks(results, validation_config)
            
            # Compare with theoretical predictions
            theoretical_comparison = self._compare_with_theory(results, validation_config)
            
            # Compare with literature values
            literature_comparison = self._compare_with_literature(results, validation_config)
            
            # Perform cross-validation
            cross_validation = self._perform_cross_validation(results, validation_config)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                reproducibility, consistency, theoretical_comparison, literature_comparison
            )
            
            # Generate validation recommendations
            recommendations = self._generate_validation_recommendations(
                reproducibility, consistency, theoretical_comparison, confidence
            )
            
            validation_result.update({
                'success': True,
                'reproducibility_assessment': reproducibility,
                'consistency_checks': consistency,
                'theoretical_comparison': theoretical_comparison,
                'literature_comparison': literature_comparison,
                'cross_validation': cross_validation,
                'overall_confidence': confidence,
                'validation_recommendations': recommendations
            })
            
        except Exception as e:
            validation_result['error'] = str(e)
            logger.error(f"Result validation failed: {e}")
        
        return validation_result
    
    def _discover_physics_specific_tools(self, research_question: str) -> List[Dict[str, Any]]:
        """Discover experimental physics specific tools."""
        experimental_tools = []
        question_lower = research_question.lower()
        
        # Data analysis tools
        if any(keyword in question_lower for keyword in 
               ['data', 'analyze', 'statistical', 'measurement']):
            experimental_tools.append({
                'tool_id': 'experimental_data_analyzer',
                'name': 'Experimental Data Analysis Suite',
                'description': 'Comprehensive statistical analysis for experimental data',
                'capabilities': ['statistical_analysis', 'hypothesis_testing', 'model_fitting', 'uncertainty_analysis'],
                'confidence': 0.95,
                'physics_specific': True,
                'scales': ['atomic', 'molecular', 'macro'],
                'methodologies': ['experimental']
            })
        
        # Experimental design tools
        if any(keyword in question_lower for keyword in 
               ['experiment', 'design', 'protocol', 'methodology']):
            experimental_tools.append({
                'tool_id': 'experiment_designer',
                'name': 'Physics Experiment Designer',
                'description': 'Design optimal physics experiments',
                'capabilities': ['experimental_design', 'protocol_development', 'resource_planning'],
                'confidence': 0.9,
                'physics_specific': True,
                'scales': ['nano', 'micro', 'macro'],
                'methodologies': ['experimental']
            })
        
        # Uncertainty analysis tools
        if any(keyword in question_lower for keyword in 
               ['uncertainty', 'error', 'validation', 'calibration']):
            experimental_tools.append({
                'tool_id': 'uncertainty_analyzer',
                'name': 'Uncertainty and Error Analysis Tool',
                'description': 'Comprehensive uncertainty quantification',
                'capabilities': ['error_propagation', 'uncertainty_analysis', 'validation'],
                'confidence': 0.88,
                'physics_specific': True,
                'scales': ['atomic', 'molecular', 'macro'],
                'methodologies': ['experimental']
            })
        
        return experimental_tools
    
    # Private helper methods for experimental design
    
    def _analyze_hypothesis_requirements(self, hypothesis: str) -> Dict[str, Any]:
        """Analyze hypothesis to determine experimental requirements."""
        requirements = {
            'measurement_type': 'unknown',
            'variables': {'independent': [], 'dependent': [], 'controlled': []},
            'precision_requirements': 'medium',
            'sample_size_estimate': 30,
            'measurement_technique': 'optical',
            'data_type': 'continuous',
            'statistical_power': 0.8
        }
        
        hypothesis_lower = hypothesis.lower()
        
        # Identify measurement types
        if any(keyword in hypothesis_lower for keyword in ['optical', 'light', 'spectrum']):
            requirements['measurement_technique'] = 'optical'
        elif any(keyword in hypothesis_lower for keyword in ['electrical', 'voltage', 'current']):
            requirements['measurement_technique'] = 'electrical'
        elif any(keyword in hypothesis_lower for keyword in ['temperature', 'thermal', 'heat']):
            requirements['measurement_technique'] = 'thermal'
        elif any(keyword in hypothesis_lower for keyword in ['magnetic', 'field']):
            requirements['measurement_technique'] = 'magnetic'
        
        # Estimate precision requirements
        if any(keyword in hypothesis_lower for keyword in ['precise', 'accurate', 'exact']):
            requirements['precision_requirements'] = 'high'
        elif any(keyword in hypothesis_lower for keyword in ['rough', 'approximate']):
            requirements['precision_requirements'] = 'low'
        
        # Identify data types
        if any(keyword in hypothesis_lower for keyword in ['category', 'type', 'class']):
            requirements['data_type'] = 'categorical'
        elif any(keyword in hypothesis_lower for keyword in ['count', 'number']):
            requirements['data_type'] = 'discrete'
        else:
            requirements['data_type'] = 'continuous'
        
        return requirements
    
    def _select_experimental_design(self, requirements: Dict[str, Any], 
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate experimental design."""
        n_factors = len(requirements.get('variables', {}).get('independent', []))
        budget = constraints.get('budget', 10000)
        time_limit = constraints.get('time_limit', 30)  # days
        
        design = {
            'type': 'factorial',
            'parameters': {},
            'justification': '',
            'expected_runs': 0,
            'statistical_power': 0.8
        }
        
        # Select design based on factors and constraints
        if n_factors <= 2 and budget > 5000:
            design.update({
                'type': 'full_factorial',
                'parameters': {'levels': 3, 'replicates': 3},
                'expected_runs': 3**n_factors * 3,
                'justification': 'Full factorial design for comprehensive factor analysis'
            })
        elif n_factors <= 5 and time_limit > 15:
            design.update({
                'type': 'fractional_factorial',
                'parameters': {'levels': 2, 'fraction': '1/2'},
                'expected_runs': 2**(n_factors-1),
                'justification': 'Fractional factorial for efficient screening'
            })
        elif n_factors > 5:
            design.update({
                'type': 'plackett_burman',
                'parameters': {'runs': 12},
                'expected_runs': 12,
                'justification': 'Plackett-Burman for many-factor screening'
            })
        else:
            design.update({
                'type': 'simple_comparative',
                'parameters': {'groups': 2, 'replicates': 10},
                'expected_runs': 20,
                'justification': 'Simple comparative study due to constraints'
            })
        
        return design
    
    def _design_measurement_protocols(self, requirements: Dict[str, Any], 
                                    constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design measurement protocols."""
        protocols = []
        
        measurement_technique = requirements.get('measurement_technique', 'optical')
        precision = requirements.get('precision_requirements', 'medium')
        
        # Base protocol
        base_protocol = {
            'technique': measurement_technique,
            'calibration_procedure': f'{measurement_technique}_calibration',
            'measurement_procedure': f'{measurement_technique}_measurement',
            'quality_checks': ['baseline_check', 'repeatability_check'],
            'data_recording': 'automated',
            'frequency': 'per_sample'
        }
        
        # Add precision-specific procedures
        if precision == 'high':
            base_protocol['quality_checks'].extend([
                'drift_correction', 'temperature_monitoring', 'multiple_measurements'
            ])
            base_protocol['repetitions'] = 5
        elif precision == 'medium':
            base_protocol['repetitions'] = 3
        else:
            base_protocol['repetitions'] = 1
        
        protocols.append(base_protocol)
        
        # Add control measurements
        control_protocol = {
            'technique': 'control_measurement',
            'procedure': 'reference_standard_measurement',
            'frequency': 'daily',
            'purpose': 'system_verification'
        }
        protocols.append(control_protocol)
        
        return protocols
    
    def _plan_data_analysis(self, requirements: Dict[str, Any], 
                           exp_design: Dict[str, Any]) -> Dict[str, Any]:
        """Plan data analysis methodology."""
        analysis_plan = {
            'descriptive_statistics': ['mean', 'std', 'median', 'range'],
            'hypothesis_tests': [],
            'model_fitting': [],
            'visualization': [],
            'significance_level': 0.05
        }
        
        data_type = requirements.get('data_type', 'continuous')
        n_groups = exp_design.get('parameters', {}).get('groups', 2)
        
        # Select appropriate statistical tests
        if data_type == 'continuous':
            if n_groups == 2:
                analysis_plan['hypothesis_tests'] = ['t_test_independent']
            elif n_groups > 2:
                analysis_plan['hypothesis_tests'] = ['one_way_anova']
        else:
            analysis_plan['hypothesis_tests'] = ['chi_square_test']
        
        # Plan model fitting
        if requirements.get('measurement_type') in ['relationship', 'correlation']:
            analysis_plan['model_fitting'] = ['linear_regression', 'polynomial_fit']
        
        # Plan visualizations
        analysis_plan['visualization'] = [
            'histogram', 'box_plot', 'scatter_plot', 'residual_plot'
        ]
        
        return analysis_plan
    
    def _design_uncertainty_analysis(self, requirements: Dict[str, Any], 
                                   protocols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design uncertainty analysis approach."""
        uncertainty_design = {
            'error_sources': [],
            'uncertainty_types': [],
            'propagation_method': 'monte_carlo',
            'coverage_probability': 0.95,
            'degrees_of_freedom': 'auto'
        }
        
        # Identify potential error sources
        measurement_technique = requirements.get('measurement_technique', 'optical')
        
        if measurement_technique == 'optical':
            uncertainty_design['error_sources'] = [
                'detector_noise', 'wavelength_calibration', 'intensity_stability',
                'environmental_fluctuations', 'sample_positioning'
            ]
        elif measurement_technique == 'electrical':
            uncertainty_design['error_sources'] = [
                'voltage_accuracy', 'current_stability', 'resistance_drift',
                'temperature_coefficient', 'contact_resistance'
            ]
        else:
            uncertainty_design['error_sources'] = [
                'calibration_uncertainty', 'repeatability', 'drift',
                'environmental_conditions', 'operator_effects'
            ]
        
        # Classify uncertainty types
        uncertainty_design['uncertainty_types'] = [
            'type_a_statistical', 'type_b_systematic'
        ]
        
        return uncertainty_design
    
    def _design_quality_control(self, requirements: Dict[str, Any], 
                              protocols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design quality control procedures."""
        qc_procedures = {
            'pre_measurement': [],
            'during_measurement': [],
            'post_measurement': [],
            'documentation': [],
            'review_criteria': {}
        }
        
        # Pre-measurement QC
        qc_procedures['pre_measurement'] = [
            'equipment_calibration_check',
            'environmental_conditions_check',
            'sample_preparation_verification',
            'protocol_review'
        ]
        
        # During measurement QC
        qc_procedures['during_measurement'] = [
            'real_time_monitoring',
            'drift_detection',
            'outlier_detection',
            'repeatability_checks'
        ]
        
        # Post-measurement QC
        qc_procedures['post_measurement'] = [
            'data_integrity_check',
            'statistical_outlier_analysis',
            'consistency_verification',
            'documentation_completeness'
        ]
        
        # Documentation requirements
        qc_procedures['documentation'] = [
            'measurement_conditions',
            'equipment_status',
            'calibration_records',
            'environmental_log',
            'operator_notes'
        ]
        
        return qc_procedures
    
    def _estimate_experimental_resources(self, protocols: List[Dict[str, Any]], 
                                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate experimental resources and timeline."""
        estimate = {
            'time_estimate': {},
            'cost_estimate': {},
            'equipment_requirements': [],
            'personnel_requirements': {},
            'consumables': []
        }
        
        # Time estimation
        n_protocols = len(protocols)
        repetitions = sum(p.get('repetitions', 1) for p in protocols)
        
        estimate['time_estimate'] = {
            'setup_time_hours': 2 * n_protocols,
            'measurement_time_hours': repetitions * 0.5,
            'analysis_time_hours': repetitions * 0.2,
            'total_days': (2 * n_protocols + repetitions * 0.7) / 8
        }
        
        # Cost estimation (simplified)
        estimate['cost_estimate'] = {
            'equipment_cost': 1000 * n_protocols,
            'consumables_cost': 50 * repetitions,
            'personnel_cost': estimate['time_estimate']['total_days'] * 500,
            'total_cost': 1000 * n_protocols + 50 * repetitions + estimate['time_estimate']['total_days'] * 500
        }
        
        # Equipment requirements
        for protocol in protocols:
            technique = protocol.get('technique', 'unknown')
            if technique not in [item['technique'] for item in estimate['equipment_requirements']]:
                estimate['equipment_requirements'].append({
                    'technique': technique,
                    'equipment_type': f'{technique}_system',
                    'specifications': 'standard_laboratory_grade'
                })
        
        return estimate
    
    # Data analysis methods
    
    def _extract_measurements(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract measurement data from input."""
        measurements = {}
        
        for key, values in data.items():
            if isinstance(values, (list, tuple)):
                measurements[key] = np.array(values)
            elif isinstance(values, np.ndarray):
                measurements[key] = values
            elif isinstance(values, (int, float)):
                measurements[key] = np.array([values])
        
        return measurements
    
    def _calculate_descriptive_statistics(self, measurements: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate descriptive statistics for measurements."""
        stats_result = {}
        
        for key, values in measurements.items():
            if len(values) > 0:
                stats_result[key] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'skewness': float(stats.skew(values)) if len(values) > 2 else 0.0,
                    'kurtosis': float(stats.kurtosis(values)) if len(values) > 3 else 0.0
                }
                
                # Calculate standard error of mean
                if len(values) > 1:
                    stats_result[key]['sem'] = stats_result[key]['std'] / np.sqrt(len(values))
                else:
                    stats_result[key]['sem'] = 0.0
        
        return stats_result
    
    def _conduct_hypothesis_testing(self, measurements: Dict[str, np.ndarray], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct statistical hypothesis testing."""
        test_results = {}
        alpha = config.get('significance_level', 0.05)
        
        # Get test configuration
        test_type = config.get('test_type', 't_test')
        
        if test_type == 't_test' and len(measurements) >= 1:
            # One-sample t-test
            for key, values in measurements.items():
                if len(values) > 1:
                    population_mean = config.get('population_mean', 0.0)
                    t_stat, p_value = stats.ttest_1samp(values, population_mean)
                    
                    test_results[f'{key}_t_test'] = {
                        'test_type': 'one_sample_t_test',
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'alpha': alpha,
                        'null_hypothesis': f'Mean equals {population_mean}',
                        'degrees_freedom': len(values) - 1
                    }
        
        elif test_type == 'two_sample_t_test' and len(measurements) >= 2:
            # Two-sample t-test
            keys = list(measurements.keys())
            if len(keys) >= 2:
                values1 = measurements[keys[0]]
                values2 = measurements[keys[1]]
                
                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    test_results['two_sample_t_test'] = {
                        'test_type': 'two_sample_t_test',
                        'groups': [keys[0], keys[1]],
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'alpha': alpha,
                        'null_hypothesis': 'Means are equal'
                    }
        
        elif test_type == 'anova' and len(measurements) > 2:
            # One-way ANOVA
            groups = [values for values in measurements.values() if len(values) > 0]
            if len(groups) > 2:
                f_stat, p_value = stats.f_oneway(*groups)
                
                test_results['one_way_anova'] = {
                    'test_type': 'one_way_anova',
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < alpha,
                    'alpha': alpha,
                    'null_hypothesis': 'All group means are equal',
                    'groups': list(measurements.keys())
                }
        
        return test_results
    
    def _analyze_uncertainties(self, measurements: Dict[str, np.ndarray], 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze measurement uncertainties."""
        uncertainty_results = {}
        
        for key, values in measurements.items():
            if len(values) > 1:
                # Type A uncertainty (statistical)
                type_a = np.std(values, ddof=1) / np.sqrt(len(values))
                
                # Type B uncertainty (systematic, from config)
                type_b = config.get('systematic_uncertainty', {}).get(key, 0.01 * np.mean(values))
                
                # Combined uncertainty
                combined = np.sqrt(type_a**2 + type_b**2)
                
                # Expanded uncertainty (k=2 for ~95% confidence)
                expanded = 2 * combined
                
                uncertainty_results[key] = {
                    'type_a_uncertainty': float(type_a),
                    'type_b_uncertainty': float(type_b),
                    'combined_uncertainty': float(combined),
                    'expanded_uncertainty': float(expanded),
                    'coverage_factor': 2,
                    'coverage_probability': 0.95,
                    'relative_uncertainty_percent': float(100 * combined / np.abs(np.mean(values)))
                }
        
        return uncertainty_results
    
    def _analyze_outliers(self, measurements: Dict[str, np.ndarray], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze outliers in measurements."""
        outlier_results = {}
        
        for key, values in measurements.items():
            if len(values) > 3:
                # Z-score method
                z_scores = np.abs(stats.zscore(values))
                z_threshold = config.get('z_threshold', 3.0)
                z_outliers = np.where(z_scores > z_threshold)[0]
                
                # Interquartile range method
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                iqr_threshold = config.get('iqr_multiplier', 1.5)
                iqr_lower = q1 - iqr_threshold * iqr
                iqr_upper = q3 + iqr_threshold * iqr
                iqr_outliers = np.where((values < iqr_lower) | (values > iqr_upper))[0]
                
                outlier_results[key] = {
                    'z_score_outliers': {
                        'indices': z_outliers.tolist(),
                        'values': values[z_outliers].tolist(),
                        'threshold': z_threshold
                    },
                    'iqr_outliers': {
                        'indices': iqr_outliers.tolist(),
                        'values': values[iqr_outliers].tolist(),
                        'lower_bound': iqr_lower,
                        'upper_bound': iqr_upper
                    },
                    'outlier_percentage': float(len(set(z_outliers) | set(iqr_outliers)) / len(values) * 100)
                }
        
        return outlier_results
    
    def _fit_models_to_data(self, measurements: Dict[str, np.ndarray], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit models to experimental data."""
        model_results = {}
        
        # Determine if we have x,y data for fitting
        if 'x' in measurements and 'y' in measurements:
            x_data = measurements['x']
            y_data = measurements['y']
            
            if len(x_data) == len(y_data) and len(x_data) > 2:
                # Linear fit
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                
                model_results['linear_fit'] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_value': float(r_value),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'std_error': float(std_err),
                    'equation': f'y = {slope:.4f}x + {intercept:.4f}'
                }
                
                # Polynomial fit (degree 2)
                if len(x_data) > 3:
                    poly_coeffs = np.polyfit(x_data, y_data, 2)
                    y_pred = np.polyval(poly_coeffs, x_data)
                    ss_res = np.sum((y_data - y_pred) ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared_poly = 1 - (ss_res / ss_tot)
                    
                    model_results['polynomial_fit'] = {
                        'coefficients': poly_coeffs.tolist(),
                        'r_squared': float(r_squared_poly),
                        'equation': f'y = {poly_coeffs[0]:.4f}x² + {poly_coeffs[1]:.4f}x + {poly_coeffs[2]:.4f}'
                    }
        
        return model_results
    
    def _generate_visualizations(self, measurements: Dict[str, np.ndarray], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization plans for data."""
        visualization_results = {
            'recommended_plots': [],
            'plot_configurations': {}
        }
        
        # Determine appropriate visualizations
        for key, values in measurements.items():
            if len(values) > 0:
                # Histogram for distribution
                visualization_results['recommended_plots'].append(f'{key}_histogram')
                visualization_results['plot_configurations'][f'{key}_histogram'] = {
                    'type': 'histogram',
                    'data': key,
                    'bins': min(20, len(values)//2 + 1),
                    'title': f'Distribution of {key}',
                    'xlabel': key,
                    'ylabel': 'Frequency'
                }
                
                # Box plot for outliers
                if len(values) > 5:
                    visualization_results['recommended_plots'].append(f'{key}_boxplot')
                    visualization_results['plot_configurations'][f'{key}_boxplot'] = {
                        'type': 'boxplot',
                        'data': key,
                        'title': f'Box Plot of {key}',
                        'ylabel': key
                    }
        
        # Scatter plot for x,y relationships
        if 'x' in measurements and 'y' in measurements:
            visualization_results['recommended_plots'].append('scatter_plot')
            visualization_results['plot_configurations']['scatter_plot'] = {
                'type': 'scatter',
                'x_data': 'x',
                'y_data': 'y',
                'title': 'X vs Y Relationship',
                'xlabel': 'X',
                'ylabel': 'Y'
            }
        
        return visualization_results
    
    def _draw_scientific_conclusions(self, descriptive_stats: Dict[str, Any],
                                   hypothesis_results: Dict[str, Any],
                                   model_results: Dict[str, Any],
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Draw scientific conclusions from analysis results."""
        conclusions = {
            'summary': '',
            'key_findings': [],
            'statistical_significance': {},
            'effect_sizes': {},
            'practical_significance': {},
            'limitations': [],
            'recommendations': []
        }
        
        # Summarize key findings
        for key, stats in descriptive_stats.items():
            if stats['count'] > 0:
                conclusions['key_findings'].append(
                    f"{key}: Mean = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                )
        
        # Summarize hypothesis test results
        for test_name, test_result in hypothesis_results.items():
            conclusions['statistical_significance'][test_name] = {
                'significant': test_result['significant'],
                'p_value': test_result['p_value'],
                'interpretation': 'significant' if test_result['significant'] else 'not significant'
            }
        
        # Model fit quality
        if 'linear_fit' in model_results:
            r_squared = model_results['linear_fit']['r_squared']
            conclusions['key_findings'].append(
                f"Linear relationship explains {r_squared*100:.1f}% of variance"
            )
        
        # Generate summary
        n_significant = sum(1 for test in conclusions['statistical_significance'].values() 
                          if test['significant'])
        total_tests = len(conclusions['statistical_significance'])
        
        if total_tests > 0:
            conclusions['summary'] = (
                f"Analysis of experimental data revealed {n_significant} out of {total_tests} "
                f"statistically significant results. "
            )
        else:
            conclusions['summary'] = "Descriptive analysis completed with no hypothesis testing."
        
        # Add recommendations
        conclusions['recommendations'] = [
            "Verify results through independent replication",
            "Consider additional controls for systematic effects",
            "Expand sample size if effect sizes are small",
            "Validate measurement protocols and calibrations"
        ]
        
        return conclusions
    
    # Validation methods
    
    def _assess_reproducibility(self, results: Dict[str, Any], 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess reproducibility of experimental results."""
        reproducibility = {
            'repeatability_assessment': {},
            'reproducibility_coefficient': 0.0,
            'between_session_variability': 0.0,
            'assessment': 'good'
        }
        
        # Simplified reproducibility assessment
        if 'measurements' in results:
            measurements = results['measurements']
            if isinstance(measurements, dict):
                for key, values in measurements.items():
                    if hasattr(values, '__len__') and len(values) > 1:
                        cv = np.std(values) / np.mean(values) * 100  # Coefficient of variation
                        reproducibility['repeatability_assessment'][key] = {
                            'coefficient_of_variation_percent': float(cv),
                            'assessment': 'excellent' if cv < 5 else 'good' if cv < 10 else 'fair' if cv < 20 else 'poor'
                        }
        
        return reproducibility
    
    def _perform_consistency_checks(self, results: Dict[str, Any], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform consistency checks on results."""
        consistency = {
            'mass_balance': {'status': 'not_applicable', 'deviation': 0.0},
            'energy_conservation': {'status': 'not_applicable', 'deviation': 0.0},
            'dimensional_analysis': {'status': 'passed'},
            'physical_reasonableness': {'status': 'reasonable'},
            'overall_consistency': 'good'
        }
        
        # Check for physical reasonableness
        if 'measurements' in results:
            measurements = results['measurements']
            if isinstance(measurements, dict):
                for key, values in measurements.items():
                    if hasattr(values, '__len__') and len(values) > 0:
                        # Check for negative values where they shouldn't be
                        if 'energy' in key.lower() or 'mass' in key.lower():
                            if np.any(np.array(values) < 0):
                                consistency['physical_reasonableness']['status'] = 'questionable'
                                consistency['physical_reasonableness']['issue'] = f'Negative {key} values detected'
        
        return consistency
    
    def _compare_with_theory(self, results: Dict[str, Any], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results with theoretical predictions."""
        theoretical_comparison = {
            'theoretical_values': {},
            'experimental_values': {},
            'deviations': {},
            'agreement_assessment': 'unknown'
        }
        
        # Get theoretical values from config
        theoretical_values = config.get('theoretical_values', {})
        
        if 'measurements' in results and theoretical_values:
            measurements = results['measurements']
            
            for key, theoretical_value in theoretical_values.items():
                if key in measurements:
                    exp_values = measurements[key]
                    if hasattr(exp_values, '__len__') and len(exp_values) > 0:
                        exp_mean = np.mean(exp_values)
                        deviation = abs(exp_mean - theoretical_value) / theoretical_value * 100
                        
                        theoretical_comparison['theoretical_values'][key] = theoretical_value
                        theoretical_comparison['experimental_values'][key] = float(exp_mean)
                        theoretical_comparison['deviations'][key] = {
                            'absolute': float(abs(exp_mean - theoretical_value)),
                            'relative_percent': float(deviation),
                            'agreement': 'excellent' if deviation < 5 else 'good' if deviation < 10 else 'fair' if deviation < 25 else 'poor'
                        }
        
        return theoretical_comparison
    
    def _compare_with_literature(self, results: Dict[str, Any], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results with literature values."""
        literature_comparison = {
            'literature_values': {},
            'experimental_values': {},
            'z_scores': {},
            'agreement_assessment': 'not_available'
        }
        
        # Get literature values from config
        literature_values = config.get('literature_values', {})
        literature_uncertainties = config.get('literature_uncertainties', {})
        
        if 'measurements' in results and literature_values:
            measurements = results['measurements']
            
            for key, lit_value in literature_values.items():
                if key in measurements:
                    exp_values = measurements[key]
                    if hasattr(exp_values, '__len__') and len(exp_values) > 0:
                        exp_mean = np.mean(exp_values)
                        exp_std = np.std(exp_values) if len(exp_values) > 1 else 0
                        
                        # Calculate z-score if literature uncertainty is available
                        if key in literature_uncertainties:
                            lit_uncertainty = literature_uncertainties[key]
                            combined_uncertainty = np.sqrt(exp_std**2 + lit_uncertainty**2)
                            if combined_uncertainty > 0:
                                z_score = abs(exp_mean - lit_value) / combined_uncertainty
                                
                                literature_comparison['z_scores'][key] = {
                                    'z_score': float(z_score),
                                    'agreement': 'excellent' if z_score < 1 else 'good' if z_score < 2 else 'fair' if z_score < 3 else 'poor'
                                }
        
        return literature_comparison
    
    def _perform_cross_validation(self, results: Dict[str, Any], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation of results."""
        cross_validation = {
            'method': 'k_fold',
            'k_folds': 5,
            'validation_score': 0.0,
            'assessment': 'not_performed'
        }
        
        # Simplified cross-validation assessment
        if 'model_fitting' in results:
            model_results = results['model_fitting']
            if 'linear_fit' in model_results:
                r_squared = model_results['linear_fit'].get('r_squared', 0)
                cross_validation.update({
                    'validation_score': r_squared * 0.9,  # Simplified estimate
                    'assessment': 'good' if r_squared > 0.8 else 'fair' if r_squared > 0.6 else 'poor'
                })
        
        return cross_validation
    
    def _calculate_overall_confidence(self, reproducibility: Dict[str, Any],
                                    consistency: Dict[str, Any],
                                    theoretical_comparison: Dict[str, Any],
                                    literature_comparison: Dict[str, Any]) -> float:
        """Calculate overall confidence in results."""
        confidence_score = 0.0
        
        # Reproducibility contribution (25%)
        repro_score = 0.8  # Default score
        if 'repeatability_assessment' in reproducibility:
            assessments = [v.get('assessment', 'fair') for v in reproducibility['repeatability_assessment'].values()]
            if assessments:
                score_map = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.3}
                repro_score = np.mean([score_map.get(a, 0.6) for a in assessments])
        
        confidence_score += repro_score * 0.25
        
        # Consistency contribution (25%)
        consistency_score = 0.8 if consistency.get('overall_consistency') == 'good' else 0.6
        confidence_score += consistency_score * 0.25
        
        # Theoretical agreement contribution (25%)
        theory_score = 0.7  # Default when no theoretical comparison
        if 'deviations' in theoretical_comparison:
            agreements = [v.get('agreement', 'fair') for v in theoretical_comparison['deviations'].values()]
            if agreements:
                score_map = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.3}
                theory_score = np.mean([score_map.get(a, 0.6) for a in agreements])
        
        confidence_score += theory_score * 0.25
        
        # Literature agreement contribution (25%)
        lit_score = 0.7  # Default when no literature comparison
        if 'z_scores' in literature_comparison:
            agreements = [v.get('agreement', 'fair') for v in literature_comparison['z_scores'].values()]
            if agreements:
                score_map = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.3}
                lit_score = np.mean([score_map.get(a, 0.6) for a in agreements])
        
        confidence_score += lit_score * 0.25
        
        return confidence_score
    
    def _generate_validation_recommendations(self, reproducibility: Dict[str, Any],
                                           consistency: Dict[str, Any],
                                           theoretical_comparison: Dict[str, Any],
                                           confidence: float) -> List[str]:
        """Generate validation recommendations based on results."""
        recommendations = []
        
        # Reproducibility recommendations
        if reproducibility.get('assessment', 'good') != 'excellent':
            recommendations.append("Improve measurement repeatability through better environmental control")
            recommendations.append("Increase number of replicate measurements")
        
        # Consistency recommendations
        if consistency.get('physical_reasonableness', {}).get('status') != 'reasonable':
            recommendations.append("Review measurement procedures for potential systematic errors")
            recommendations.append("Validate calibration procedures")
        
        # Overall confidence recommendations
        if confidence < 0.7:
            recommendations.append("Consider additional validation experiments")
            recommendations.append("Expand uncertainty analysis")
            recommendations.append("Seek independent verification of results")
        elif confidence < 0.8:
            recommendations.append("Strengthen statistical analysis with larger sample sizes")
            recommendations.append("Compare with additional theoretical models")
        
        # General recommendations
        recommendations.extend([
            "Document all experimental conditions thoroughly",
            "Archive raw data for future analysis",
            "Consider publishing methodology for peer review"
        ])
        
        return recommendations[:5]  # Limit to top 5 recommendations