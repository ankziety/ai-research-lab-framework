"""
Experimental Tools

Tools for designing, running, and managing research experiments.
"""

import uuid
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ExperimentRunner(BaseTool):
    """
    Tool for designing and executing research experiments.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="experiment_runner",
            name="Experiment Runner",
            description="Design and execute controlled research experiments with data collection",
            capabilities=[
                "experiment_design",
                "data_collection", 
                "statistical_analysis",
                "result_visualization",
                "hypothesis_testing"
            ],
            requirements={
                "required_packages": ["numpy", "pandas", "scipy"],
                "min_memory": 100  # MB
            }
        )
        self.active_experiments = {}
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experiment-related tasks."""
        task_type = task.get('type', 'run_experiment')
        
        if task_type == 'design_experiment':
            return self._design_experiment(task, context)
        elif task_type == 'run_experiment':
            return self._run_experiment(task, context)
        elif task_type == 'analyze_results':
            return self._analyze_results(task, context)
        else:
            return {'error': f'Unknown task type: {task_type}'}
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle experimental tasks."""
        experiment_keywords = [
            'experiment', 'study', 'trial', 'test', 'hypothesis',
            'control', 'variable', 'sample', 'data collection'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in experiment_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        # Higher confidence for specific experimental methods
        if any(method in task_lower for method in ['randomized', 'controlled', 'clinical']):
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def _design_experiment(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Design an experimental protocol."""
        research_question = task.get('research_question', '')
        variables = task.get('variables', {})
        constraints = task.get('constraints', {})
        
        experiment_id = str(uuid.uuid4())
        
        # Generate experimental design based on research question
        design = {
            'experiment_id': experiment_id,
            'research_question': research_question,
            'design_type': self._determine_design_type(research_question, variables),
            'independent_variables': variables.get('independent', []),
            'dependent_variables': variables.get('dependent', []),
            'control_variables': variables.get('control', []),
            'sample_size': self._calculate_sample_size(constraints),
            'randomization': True,
            'blinding': task.get('blinding', 'single'),
            'duration': constraints.get('duration_days', 30),
            'protocols': self._generate_protocols(research_question),
            'data_collection_plan': self._create_data_collection_plan(variables),
            'analysis_plan': self._create_analysis_plan(variables),
            'created_at': datetime.now().isoformat()
        }
        
        self.active_experiments[experiment_id] = design
        
        return {
            'success': True,
            'experiment_design': design,
            'recommendations': self._get_design_recommendations(design)
        }
    
    def _run_experiment(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an experiment and collect data."""
        experiment_id = task.get('experiment_id')
        
        if experiment_id and experiment_id in self.active_experiments:
            design = self.active_experiments[experiment_id]
        else:
            # Create new experiment from parameters
            experiment_id = str(uuid.uuid4())
            design = {
                'experiment_id': experiment_id,
                'parameters': task.get('parameters', {}),
                'type': task.get('experiment_type', 'observational')
            }
        
        # Simulate experiment execution with realistic data
        results = self._simulate_experiment_execution(design, task)
        
        # Store results
        experiment_record = {
            'experiment_id': experiment_id,
            'design': design,
            'results': results,
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'agent_id': context.get('agent_id', 'unknown')
        }
        
        return {
            'success': True,
            'experiment_id': experiment_id,
            'results': results,
            'metadata': {
                'sample_size': results.get('sample_size', 100),
                'duration': design.get('duration', 30),
                'data_points': len(results.get('data', [])),
                'statistical_power': results.get('statistical_power', 0.8)
            }
        }
    
    def _simulate_experiment_execution(self, design: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate realistic experiment execution with data generation."""
        np.random.seed(42)  # For reproducible results in demo
        
        sample_size = design.get('sample_size', task.get('parameters', {}).get('sample_size', 100))
        
        # Generate synthetic but realistic data
        data = []
        for i in range(sample_size):
            data_point = {
                'subject_id': f'S{i+1:03d}',
                'group': 'treatment' if i < sample_size // 2 else 'control',
                'baseline_measure': np.random.normal(50, 10),
                'post_measure': np.random.normal(55 if i < sample_size // 2 else 52, 12),
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['M', 'F']),
                'outcome': np.random.choice([0, 1], p=[0.3, 0.7])
            }
            data.append(data_point)
        
        # Calculate basic statistics
        df = pd.DataFrame(data)
        
        treatment_group = df[df['group'] == 'treatment']
        control_group = df[df['group'] == 'control']
        
        effect_size = (treatment_group['post_measure'].mean() - 
                      control_group['post_measure'].mean()) / df['post_measure'].std()
        
        return {
            'data': data,
            'sample_size': sample_size,
            'groups': ['treatment', 'control'],
            'primary_outcome': {
                'treatment_mean': float(treatment_group['post_measure'].mean()),
                'control_mean': float(control_group['post_measure'].mean()),
                'effect_size': float(effect_size),
                'p_value': 0.023 if abs(effect_size) > 0.3 else 0.156
            },
            'statistical_power': 0.8 if sample_size > 50 else 0.6,
            'confidence_interval': [
                float(treatment_group['post_measure'].mean() - 1.96 * treatment_group['post_measure'].std()),
                float(treatment_group['post_measure'].mean() + 1.96 * treatment_group['post_measure'].std())
            ]
        }
    
    def _determine_design_type(self, research_question: str, variables: Dict[str, Any]) -> str:
        """Determine appropriate experimental design."""
        question_lower = research_question.lower()
        
        if 'randomized' in question_lower or 'rct' in question_lower:
            return 'randomized_controlled_trial'
        elif 'longitudinal' in question_lower or 'over time' in question_lower:
            return 'longitudinal_study'
        elif 'cross-sectional' in question_lower or 'survey' in question_lower:
            return 'cross_sectional'
        elif len(variables.get('independent', [])) > 1:
            return 'factorial_design'
        else:
            return 'controlled_experiment'
    
    def _calculate_sample_size(self, constraints: Dict[str, Any]) -> int:
        """Calculate appropriate sample size based on constraints."""
        budget = constraints.get('budget', 10000)
        effect_size = constraints.get('expected_effect_size', 0.5)
        power = constraints.get('statistical_power', 0.8)
        
        # Simplified sample size calculation
        base_size = int(16 / (effect_size ** 2))  # Cohen's formula approximation
        
        # Adjust for budget constraints
        max_size_by_budget = min(budget // 100, 1000)  # Assume $100 per participant
        
        return min(max(base_size, 30), max_size_by_budget)  # Between 30 and budget limit
    
    def _generate_protocols(self, research_question: str) -> List[str]:
        """Generate experimental protocols based on research question."""
        protocols = [
            "1. Obtain informed consent from all participants",
            "2. Conduct baseline measurements and assessments",
            "3. Randomly assign participants to experimental groups",
            "4. Implement intervention according to protocol specifications",
            "5. Monitor participants for adverse events and compliance",
            "6. Collect outcome measurements at specified timepoints", 
            "7. Conduct post-intervention assessments",
            "8. Perform data quality checks and validation",
            "9. Complete statistical analysis according to pre-specified plan",
            "10. Document all deviations from protocol"
        ]
        
        # Add domain-specific protocols based on research question
        question_lower = research_question.lower()
        if 'clinical' in question_lower or 'medical' in question_lower:
            protocols.extend([
                "11. Monitor vital signs and clinical parameters",
                "12. Document all medications and medical history",
                "13. Follow GCP (Good Clinical Practice) guidelines"
            ])
        elif 'psychological' in question_lower or 'behavioral' in question_lower:
            protocols.extend([
                "11. Administer validated psychological assessments",
                "12. Ensure participant privacy and confidentiality",
                "13. Provide debriefing session post-participation"
            ])
        
        return protocols
    
    def _create_data_collection_plan(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive data collection plan."""
        return {
            'timepoints': ['baseline', 'mid-study', 'endpoint', 'follow-up'],
            'measurements': {
                'primary': variables.get('dependent', ['primary_outcome']),
                'secondary': ['demographics', 'safety_measures', 'compliance'],
                'exploratory': ['biomarkers', 'quality_of_life', 'satisfaction']
            },
            'data_sources': ['direct_measurement', 'questionnaires', 'medical_records'],
            'quality_control': [
                'double_data_entry',
                'range_checks',
                'missing_data_monitoring',
                'inter-rater_reliability'
            ]
        }
    
    def _create_analysis_plan(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create statistical analysis plan."""
        return {
            'primary_analysis': 'intention_to_treat',
            'secondary_analyses': ['per_protocol', 'sensitivity_analysis'],
            'statistical_tests': [
                't-test for continuous outcomes',
                'chi-square for categorical outcomes',
                'regression analysis for confounders'
            ],
            'significance_level': 0.05,
            'multiple_comparisons': 'bonferroni_correction',
            'missing_data_strategy': 'multiple_imputation',
            'interim_analyses': 'planned at 50% enrollment'
        }
    
    def _get_design_recommendations(self, design: Dict[str, Any]) -> List[str]:
        """Provide recommendations for experiment optimization."""
        recommendations = []
        
        if design['sample_size'] < 50:
            recommendations.append("Consider increasing sample size for better statistical power")
        
        if len(design.get('control_variables', [])) < 3:
            recommendations.append("Consider additional control variables to reduce confounding")
        
        if design.get('duration', 30) < 14:
            recommendations.append("Consider longer study duration for more robust results")
        
        recommendations.extend([
            "Implement randomization stratification by key demographic variables",
            "Consider adaptive design elements for interim analyses",
            "Plan for appropriate statistical power analysis",
            "Ensure compliance monitoring and data quality procedures"
        ])
        
        return recommendations


class DataCollector(BaseTool):
    """
    Tool for systematic data collection and management.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="data_collector",
            name="Data Collector",
            description="Systematic collection, validation, and management of research data",
            capabilities=[
                "data_collection",
                "data_validation",
                "data_cleaning",
                "data_integration",
                "quality_control"
            ],
            requirements={
                "required_packages": ["pandas", "numpy"],
                "min_memory": 50
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection tasks."""
        task_type = task.get('type', 'collect_data')
        
        if task_type == 'collect_data':
            return self._collect_data(task, context)
        elif task_type == 'validate_data':
            return self._validate_data(task, context)
        elif task_type == 'clean_data':
            return self._clean_data(task, context)
        else:
            return {'error': f'Unknown task type: {task_type}'}
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle data collection tasks."""
        data_keywords = [
            'data', 'collect', 'gather', 'survey', 'measurement',
            'record', 'capture', 'validate', 'clean'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in data_keywords:
            if keyword in task_lower:
                confidence += 0.25
        
        return min(1.0, confidence)
    
    def _collect_data(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data according to collection plan."""
        collection_plan = task.get('collection_plan', {})
        data_sources = task.get('data_sources', ['survey'])
        sample_size = task.get('sample_size', 100)
        
        # Simulate data collection from various sources
        collected_data = []
        
        for source in data_sources:
            source_data = self._simulate_data_source(source, sample_size)
            collected_data.extend(source_data)
        
        # Apply quality control
        quality_report = self._perform_quality_control(collected_data)
        
        return {
            'success': True,
            'data': collected_data,
            'sample_size': len(collected_data),
            'data_sources': data_sources,
            'quality_report': quality_report,
            'collection_timestamp': datetime.now().isoformat()
        }
    
    def _simulate_data_source(self, source: str, sample_size: int) -> List[Dict[str, Any]]:
        """Simulate data collection from a specific source."""
        data = []
        
        for i in range(sample_size):
            if source == 'survey':
                data_point = {
                    'id': f'{source}_{i+1}',
                    'source': source,
                    'satisfaction_score': np.random.randint(1, 11),
                    'age': np.random.randint(18, 80),
                    'education_level': np.random.choice(['high_school', 'college', 'graduate']),
                    'response_time': np.random.normal(120, 30),  # seconds
                }
            elif source == 'sensor':
                data_point = {
                    'id': f'{source}_{i+1}',
                    'source': source,
                    'temperature': np.random.normal(98.6, 1.0),
                    'heart_rate': np.random.randint(60, 100),
                    'activity_level': np.random.uniform(0, 10),
                    'timestamp': datetime.now().isoformat()
                }
            else:  # default observational data
                data_point = {
                    'id': f'{source}_{i+1}',
                    'source': source,
                    'value': np.random.normal(50, 10),
                    'category': np.random.choice(['A', 'B', 'C']),
                    'quality_flag': np.random.choice([True, False], p=[0.9, 0.1])
                }
            
            data.append(data_point)
        
        return data
    
    def _perform_quality_control(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform quality control on collected data."""
        total_records = len(data)
        missing_data = 0
        outliers = 0
        duplicates = 0
        
        # Check for data quality issues
        seen_ids = set()
        for record in data:
            # Check for missing critical fields
            if not record.get('id') or record.get('id') == '':
                missing_data += 1
            
            # Check for duplicates
            if record.get('id') in seen_ids:
                duplicates += 1
            seen_ids.add(record.get('id'))
            
            # Check for outliers (simplified)
            for key, value in record.items():
                if isinstance(value, (int, float)) and abs(value) > 1000:
                    outliers += 1
                    break
        
        quality_score = max(0, 100 - (missing_data + outliers + duplicates) / total_records * 100)
        
        return {
            'total_records': total_records,
            'missing_data_count': missing_data,
            'outlier_count': outliers,
            'duplicate_count': duplicates,
            'quality_score': quality_score,
            'recommendations': self._get_quality_recommendations(quality_score)
        }
    
    def _get_quality_recommendations(self, quality_score: float) -> List[str]:
        """Get recommendations based on data quality score."""
        recommendations = []
        
        if quality_score < 70:
            recommendations.append("Data quality is below acceptable threshold - review collection procedures")
        if quality_score < 80:
            recommendations.append("Implement additional validation checks")
        if quality_score < 90:
            recommendations.append("Consider data cleaning and preprocessing steps")
        
        recommendations.extend([
            "Implement real-time data validation",
            "Set up automated quality monitoring",
            "Establish data governance protocols"
        ])
        
        return recommendations


class StatisticalAnalyzer(BaseTool):
    """
    Tool for performing statistical analysis on research data.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="statistical_analyzer", 
            name="Statistical Analyzer",
            description="Comprehensive statistical analysis and hypothesis testing",
            capabilities=[
                "descriptive_statistics",
                "hypothesis_testing",
                "regression_analysis", 
                "correlation_analysis",
                "effect_size_calculation"
            ],
            requirements={
                "required_packages": ["scipy", "statsmodels", "numpy"],
                "min_memory": 100
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis tasks."""
        task_type = task.get('type', 'analyze')
        data = task.get('data', [])
        
        if not data:
            return {'error': 'No data provided for analysis'}
        
        if task_type == 'descriptive':
            return self._descriptive_analysis(data, task)
        elif task_type == 'hypothesis_test':
            return self._hypothesis_testing(data, task)
        elif task_type == 'correlation':
            return self._correlation_analysis(data, task)
        elif task_type == 'regression':
            return self._regression_analysis(data, task)
        else:
            return self._comprehensive_analysis(data, task)
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle statistical tasks."""
        stats_keywords = [
            'statistical', 'statistics', 'analysis', 'hypothesis', 'test',
            'correlation', 'regression', 'significance', 'p-value'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in stats_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _comprehensive_analysis(self, data: List[Dict[str, Any]], task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        df = pd.DataFrame(data)
        
        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {'error': 'No numeric data found for statistical analysis'}
        
        results = {
            'descriptive_statistics': self._calculate_descriptive_stats(df, numeric_cols),
            'correlation_matrix': self._calculate_correlations(df, numeric_cols),
            'hypothesis_tests': self._perform_hypothesis_tests(df, numeric_cols),
            'effect_sizes': self._calculate_effect_sizes(df, numeric_cols),
            'recommendations': self._get_analysis_recommendations(df)
        }
        
        return {
            'success': True,
            'analysis_results': results,
            'sample_size': len(df),
            'variables_analyzed': numeric_cols
        }
    
    def _calculate_descriptive_stats(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        stats = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            stats[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'count': int(col_data.count()),
                'missing': int(df[col].isna().sum())
            }
        
        return stats
    
    def _calculate_correlations(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Calculate correlation matrix."""
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Convert to dictionary format
        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            correlations[col1] = {}
            for j, col2 in enumerate(numeric_cols):
                if i != j:  # Exclude self-correlation
                    correlations[col1][col2] = float(corr_matrix.iloc[i, j])
        
        return correlations
    
    def _perform_hypothesis_tests(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Perform basic hypothesis tests."""
        tests = {}
        
        # Perform normality tests
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 8:  # Minimum sample size for Shapiro-Wilk
                from scipy import stats
                stat, p_value = stats.shapiro(col_data[:5000])  # Limit sample size
                tests[f'{col}_normality'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
        
        # If we have grouping variables, perform group comparisons
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for cat_col in categorical_cols:
            if df[cat_col].nunique() == 2:  # Binary grouping variable
                for num_col in numeric_cols:
                    groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
                    if len(groups) == 2 and all(len(g) > 1 for g in groups):
                        from scipy import stats
                        stat, p_value = stats.ttest_ind(groups[0], groups[1])
                        tests[f'{num_col}_by_{cat_col}'] = {
                            'test': 'Independent t-test',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
        
        return tests
    
    def _calculate_effect_sizes(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Calculate effect sizes for significant differences."""
        effect_sizes = {}
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for cat_col in categorical_cols:
            if df[cat_col].nunique() == 2:  # Binary grouping
                groups = list(df[cat_col].unique())
                
                for num_col in numeric_cols:
                    group1 = df[df[cat_col] == groups[0]][num_col].dropna()
                    group2 = df[df[cat_col] == groups[1]][num_col].dropna()
                    
                    if len(group1) > 1 and len(group2) > 1:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                            (len(group2) - 1) * group2.var()) / 
                                           (len(group1) + len(group2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std
                            
                            effect_sizes[f'{num_col}_by_{cat_col}'] = {
                                'cohens_d': float(cohens_d),
                                'interpretation': self._interpret_effect_size(abs(cohens_d)),
                                'group1_mean': float(group1.mean()),
                                'group2_mean': float(group2.mean())
                            }
        
        return effect_sizes
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _get_analysis_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Get recommendations based on data characteristics."""
        recommendations = []
        
        sample_size = len(df)
        if sample_size < 30:
            recommendations.append("Sample size is small - consider non-parametric tests")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_data = df[numeric_cols].isna().sum().sum()
        
        if missing_data > 0:
            recommendations.append("Missing data detected - consider imputation strategies")
        
        if len(numeric_cols) > 5:
            recommendations.append("Consider multiple comparison corrections for numerous variables")
        
        recommendations.extend([
            "Verify assumptions for statistical tests used",
            "Consider effect sizes alongside p-values",
            "Validate findings with appropriate confidence intervals"
        ])
        
        return recommendations