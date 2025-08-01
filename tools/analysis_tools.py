"""
Analysis Tools

Tools for data analysis, pattern detection, and hypothesis validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import io
import base64

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class DataVisualizer(BaseTool):
    """
    Tool for creating data visualizations and plots.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="data_visualizer",
            name="Data Visualizer", 
            description="Create comprehensive data visualizations and plots for research analysis",
            capabilities=[
                "statistical_plots",
                "distribution_analysis",
                "relationship_visualization",
                "trend_analysis",
                "publication_quality_figures"
            ],
            requirements={
                "required_packages": ["matplotlib", "seaborn", "pandas"],
                "min_memory": 50
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization tasks."""
        task_type = task.get('type', 'auto_visualize')
        data = task.get('data', [])
        
        if not data:
            return {'error': 'No data provided for visualization'}
        
        df = pd.DataFrame(data)
        
        if task_type == 'distribution':
            return self._create_distribution_plots(df, task)
        elif task_type == 'correlation':
            return self._create_correlation_plots(df, task)
        elif task_type == 'comparison':
            return self._create_comparison_plots(df, task)
        elif task_type == 'time_series':
            return self._create_time_series_plots(df, task)
        else:
            return self._auto_visualize(df, task)
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle visualization tasks."""
        viz_keywords = [
            'visualize', 'plot', 'chart', 'graph', 'figure',
            'distribution', 'correlation', 'trend', 'comparison'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in viz_keywords:
            if keyword in task_lower:
                confidence += 0.25
        
        return min(1.0, confidence)
    
    def _auto_visualize(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically create appropriate visualizations based on data characteristics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        plots_created = []
        plot_files = []
        
        # Create distribution plots for numeric variables
        if numeric_cols:
            dist_result = self._create_distribution_plots(df, {'columns': numeric_cols})
            if dist_result.get('success'):
                plots_created.extend(dist_result.get('plots_created', []))
                plot_files.extend(dist_result.get('plot_files', []))
        
        # Create correlation heatmap if multiple numeric variables
        if len(numeric_cols) > 1:
            corr_result = self._create_correlation_plots(df, {'columns': numeric_cols})
            if corr_result.get('success'):
                plots_created.extend(corr_result.get('plots_created', []))
                plot_files.extend(corr_result.get('plot_files', []))
        
        # Create comparison plots if categorical variables exist
        if categorical_cols and numeric_cols:
            comp_result = self._create_comparison_plots(df, {
                'categorical': categorical_cols[0],
                'numeric': numeric_cols[0]
            })
            if comp_result.get('success'):
                plots_created.extend(comp_result.get('plots_created', []))
                plot_files.extend(comp_result.get('plot_files', []))
        
        return {
            'success': True,
            'plots_created': plots_created,
            'plot_files': plot_files,
            'recommendations': self._get_visualization_recommendations(df)
        }
    
    def _create_distribution_plots(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create distribution plots for numeric variables."""
        columns = task.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if not columns:
            return {'error': 'No numeric columns found for distribution plots'}
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(len(columns), 2, figsize=(12, 4 * len(columns)))
        
        if len(columns) == 1:
            axes = axes.reshape(1, -1)
        
        plots_created = []
        
        for i, col in enumerate(columns):
            # Histogram
            axes[i, 0].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i, 0].set_title(f'Distribution of {col}')
            axes[i, 0].set_xlabel(col)
            axes[i, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[i, 1].boxplot(df[col].dropna())
            axes[i, 1].set_title(f'Box Plot of {col}')
            axes[i, 1].set_ylabel(col)
            
            plots_created.append(f'distribution_{col}')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'distribution_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = f'visualizations/{plot_filename}'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'success': True,
            'plots_created': plots_created,
            'plot_files': [plot_path],
            'summary': f'Created distribution plots for {len(columns)} variables'
        }
    
    def _create_correlation_plots(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create correlation heatmap and scatter plots."""
        columns = task.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if len(columns) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        plt.style.use('seaborn-v0_8')
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        # Save correlation heatmap
        corr_filename = f'correlation_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        corr_path = f'visualizations/{corr_filename}'
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots_created = ['correlation_heatmap']
        plot_files = [corr_path]
        
        # Create scatter plots for highly correlated pairs
        high_corr_pairs = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.5:  # Strong correlation threshold
                    high_corr_pairs.append((columns[i], columns[j], corr_val))
        
        if high_corr_pairs:
            n_pairs = min(len(high_corr_pairs), 6)  # Limit to 6 plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, (col1, col2, corr_val) in enumerate(high_corr_pairs[:n_pairs]):
                axes[idx].scatter(df[col1], df[col2], alpha=0.6)
                axes[idx].set_xlabel(col1)
                axes[idx].set_ylabel(col2)
                axes[idx].set_title(f'{col1} vs {col2} (r={corr_val:.3f})')
                
                # Add trend line
                z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
                p = np.poly1d(z)
                axes[idx].plot(df[col1], p(df[col1]), "r--", alpha=0.8)
                
                plots_created.append(f'scatter_{col1}_vs_{col2}')
            
            # Hide unused subplots
            for idx in range(n_pairs, 6):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            scatter_filename = f'scatter_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            scatter_path = f'visualizations/{scatter_filename}'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_files.append(scatter_path)
        
        return {
            'success': True,
            'plots_created': plots_created,
            'plot_files': plot_files,
            'high_correlations': high_corr_pairs
        }
    
    def _create_comparison_plots(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison plots between groups."""
        categorical_col = task.get('categorical')
        numeric_col = task.get('numeric')
        
        if not categorical_col or not numeric_col:
            return {'error': 'Need both categorical and numeric columns for comparison'}
        
        if categorical_col not in df.columns or numeric_col not in df.columns:
            return {'error': 'Specified columns not found in data'}
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Box plot
        df.boxplot(column=numeric_col, by=categorical_col, ax=axes[0])
        axes[0].set_title(f'{numeric_col} by {categorical_col}')
        axes[0].set_ylabel(numeric_col)
        
        # Violin plot
        sns.violinplot(data=df, x=categorical_col, y=numeric_col, ax=axes[1])
        axes[1].set_title(f'{numeric_col} Distribution by {categorical_col}')
        
        # Bar plot with error bars
        grouped_stats = df.groupby(categorical_col)[numeric_col].agg(['mean', 'std'])
        grouped_stats['mean'].plot(kind='bar', yerr=grouped_stats['std'], ax=axes[2])
        axes[2].set_title(f'Mean {numeric_col} by {categorical_col}')
        axes[2].set_ylabel(f'Mean {numeric_col}')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        comp_filename = f'comparison_{categorical_col}_{numeric_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        comp_path = f'visualizations/{comp_filename}'
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'success': True,
            'plots_created': [f'comparison_{categorical_col}_vs_{numeric_col}'],
            'plot_files': [comp_path],
            'group_statistics': grouped_stats.to_dict()
        }
    
    def _create_time_series_plots(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create time series plots."""
        time_col = task.get('time_column', 'timestamp')
        value_cols = task.get('value_columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if time_col not in df.columns:
            return {'error': f'Time column {time_col} not found in data'}
        
        # Convert time column to datetime if needed
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            return {'error': f'Could not parse {time_col} as datetime'}
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(len(value_cols), 1, figsize=(12, 4 * len(value_cols)))
        
        if len(value_cols) == 1:
            axes = [axes]
        
        plots_created = []
        
        for i, col in enumerate(value_cols):
            axes[i].plot(df[time_col], df[col], marker='o', alpha=0.7)
            axes[i].set_title(f'{col} over Time')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(col)
            axes[i].tick_params(axis='x', rotation=45)
            
            plots_created.append(f'timeseries_{col}')
        
        plt.tight_layout()
        
        # Save plot
        ts_filename = f'timeseries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        ts_path = f'visualizations/{ts_filename}'
        plt.savefig(ts_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'success': True,
            'plots_created': plots_created,
            'plot_files': [ts_path]
        }
    
    def _get_visualization_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Get recommendations for improving visualizations."""
        recommendations = []
        
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        sample_size = len(df)
        
        if sample_size < 30:
            recommendations.append("Small sample size - consider scatter plots over histograms")
        
        if numeric_cols > 5:
            recommendations.append("Many numeric variables - consider dimensionality reduction visualization")
        
        if categorical_cols > 3:
            recommendations.append("Multiple categorical variables - consider faceted plots")
        
        recommendations.extend([
            "Use consistent color schemes across related plots",
            "Include error bars or confidence intervals where appropriate", 
            "Consider log scaling for highly skewed data",
            "Add annotations for key findings or outliers"
        ])
        
        return recommendations


class PatternDetector(BaseTool):
    """
    Tool for detecting patterns and anomalies in data.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="pattern_detector",
            name="Pattern Detector",
            description="Detect patterns, trends, and anomalies in research data",
            capabilities=[
                "anomaly_detection",
                "trend_analysis",
                "clustering",
                "pattern_recognition",
                "outlier_identification"
            ],
            requirements={
                "required_packages": ["scikit-learn", "scipy", "numpy"],
                "min_memory": 100
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern detection tasks."""
        task_type = task.get('type', 'detect_patterns')
        data = task.get('data', [])
        
        if not data:
            return {'error': 'No data provided for pattern detection'}
        
        df = pd.DataFrame(data)
        
        if task_type == 'anomalies':
            return self._detect_anomalies(df, task)
        elif task_type == 'trends':
            return self._analyze_trends(df, task)
        elif task_type == 'clusters':
            return self._detect_clusters(df, task)
        else:
            return self._comprehensive_pattern_analysis(df, task)
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle pattern detection tasks."""
        pattern_keywords = [
            'pattern', 'anomaly', 'outlier', 'trend', 'cluster',
            'detect', 'identify', 'recognize', 'unusual'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in pattern_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _comprehensive_pattern_analysis(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis."""
        results = {}
        
        # Detect anomalies
        anomaly_result = self._detect_anomalies(df, task)
        if anomaly_result.get('success'):
            results['anomalies'] = anomaly_result['anomalies']
        
        # Analyze trends
        trend_result = self._analyze_trends(df, task)
        if trend_result.get('success'):
            results['trends'] = trend_result['trends']
        
        # Detect clusters
        cluster_result = self._detect_clusters(df, task)
        if cluster_result.get('success'):
            results['clusters'] = cluster_result['clusters']
        
        return {
            'success': True,
            'pattern_analysis': results,
            'summary': self._summarize_patterns(results)
        }
    
    def _detect_anomalies(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies and outliers in the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {'error': 'No numeric columns found for anomaly detection'}
        
        anomalies = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            # Statistical outliers (IQR method)
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Z-score outliers
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            z_outliers = col_data[z_scores > 3]
            
            anomalies[col] = {
                'iqr_outliers': {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(col_data) * 100,
                    'values': outliers.tolist()[:10],  # Limit to first 10
                    'bounds': [float(lower_bound), float(upper_bound)]
                },
                'z_score_outliers': {
                    'count': len(z_outliers),
                    'percentage': len(z_outliers) / len(col_data) * 100,
                    'values': z_outliers.tolist()[:10]
                }
            }
        
        return {
            'success': True,
            'anomalies': anomalies,
            'recommendations': self._get_anomaly_recommendations(anomalies)
        }
    
    def _analyze_trends(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {'error': 'No numeric columns found for trend analysis'}
        
        trends = {}
        
        # If there's a time column, analyze time-based trends
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if time_cols:
            time_col = time_cols[0]
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df_sorted = df.sort_values(time_col)
                
                for col in numeric_cols:
                    # Linear trend
                    x = np.arange(len(df_sorted))
                    y = df_sorted[col].dropna()
                    
                    if len(y) > 1:
                        slope, intercept = np.polyfit(x[:len(y)], y, 1)
                        
                        trends[col] = {
                            'trend_type': 'temporal',
                            'slope': float(slope),
                            'direction': 'increasing' if slope > 0 else 'decreasing',
                            'strength': abs(slope),
                            'r_squared': float(np.corrcoef(x[:len(y)], y)[0, 1] ** 2)
                        }
            except:
                # Fall back to index-based trends
                pass
        
        # If no time column, analyze index-based trends
        if not trends:
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 1:
                    x = np.arange(len(col_data))
                    slope, intercept = np.polyfit(x, col_data, 1)
                    
                    trends[col] = {
                        'trend_type': 'sequential',
                        'slope': float(slope),
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'strength': abs(slope),
                        'r_squared': float(np.corrcoef(x, col_data)[0, 1] ** 2)
                    }
        
        return {
            'success': True,
            'trends': trends,
            'summary': self._summarize_trends(trends)
        }
    
    def _detect_clusters(self, df: pd.DataFrame, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect clusters in the data using K-means."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for clustering'}
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {'error': 'scikit-learn not available for clustering'}
        
        # Prepare data
        data_for_clustering = df[numeric_cols].dropna()
        
        if len(data_for_clustering) < 4:
            return {'error': 'Insufficient data points for clustering'}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)
        
        # Determine optimal number of clusters (2-5)
        max_clusters = min(5, len(data_for_clustering) // 2)
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method to find optimal k
        if len(inertias) > 1:
            # Simple elbow detection
            diffs = np.diff(inertias)
            optimal_k = np.argmax(diffs) + 2  # +2 because we start from k=2
        else:
            optimal_k = 2
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        clusters = {}
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            cluster_data = data_for_clustering[cluster_mask]
            
            clusters[f'cluster_{i}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(data_for_clustering) * 100),
                'center': kmeans.cluster_centers_[i].tolist(),
                'characteristics': {
                    col: {
                        'mean': float(cluster_data[col].mean()),
                        'std': float(cluster_data[col].std())
                    } for col in numeric_cols
                }
            }
        
        return {
            'success': True,
            'clusters': clusters,
            'optimal_k': optimal_k,
            'inertias': inertias,
            'features_used': numeric_cols
        }
    
    def _summarize_patterns(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Summarize detected patterns."""
        summary = {}
        
        if 'anomalies' in results:
            total_outliers = sum(
                col_data['iqr_outliers']['count'] 
                for col_data in results['anomalies'].values()
            )
            summary['anomalies'] = f"Detected {total_outliers} outliers across variables"
        
        if 'trends' in results:
            increasing_trends = sum(
                1 for trend in results['trends'].values() 
                if trend['direction'] == 'increasing'
            )
            summary['trends'] = f"{increasing_trends} variables show increasing trends"
        
        if 'clusters' in results:
            n_clusters = len(results['clusters'])
            summary['clusters'] = f"Identified {n_clusters} distinct data clusters"
        
        return summary
    
    def _summarize_trends(self, trends: Dict[str, Any]) -> str:
        """Summarize trend analysis results."""
        if not trends:
            return "No trends detected"
        
        strong_trends = [
            col for col, trend in trends.items() 
            if trend.get('r_squared', 0) > 0.5
        ]
        
        increasing = [
            col for col, trend in trends.items()
            if trend.get('direction') == 'increasing'
        ]
        
        return f"Strong trends in {len(strong_trends)} variables, {len(increasing)} increasing"
    
    def _get_anomaly_recommendations(self, anomalies: Dict[str, Any]) -> List[str]:
        """Get recommendations based on anomaly detection."""
        recommendations = []
        
        high_outlier_cols = [
            col for col, data in anomalies.items()
            if data['iqr_outliers']['percentage'] > 10
        ]
        
        if high_outlier_cols:
            recommendations.append(f"High outlier rates in {', '.join(high_outlier_cols)} - investigate data quality")
        
        recommendations.extend([
            "Review outliers for data entry errors",
            "Consider robust statistical methods for analysis",
            "Investigate biological/technical reasons for extreme values"
        ])
        
        return recommendations


class HypothesisValidator(BaseTool):
    """
    Tool for validating research hypotheses against data.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="hypothesis_validator",
            name="Hypothesis Validator", 
            description="Validate research hypotheses through statistical testing and evidence assessment",
            capabilities=[
                "hypothesis_testing",
                "evidence_assessment",
                "statistical_validation",
                "effect_size_analysis",
                "power_analysis"
            ],
            requirements={
                "required_packages": ["scipy", "statsmodels"],
                "min_memory": 50
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hypothesis validation tasks."""
        hypothesis = task.get('hypothesis', '')
        data = task.get('data', [])
        significance_level = task.get('significance_level', 0.05)
        
        if not hypothesis:
            return {'error': 'No hypothesis provided for validation'}
        
        if not data:
            return {'error': 'No data provided for hypothesis testing'}
        
        df = pd.DataFrame(data)
        
        # Parse hypothesis and determine appropriate test
        test_strategy = self._determine_test_strategy(hypothesis, df)
        
        # Execute statistical tests
        test_results = self._execute_statistical_tests(test_strategy, df, significance_level)
        
        # Assess evidence strength
        evidence_assessment = self._assess_evidence_strength(test_results, hypothesis)
        
        return {
            'success': True,
            'hypothesis': hypothesis,
            'test_strategy': test_strategy,
            'test_results': test_results,
            'evidence_assessment': evidence_assessment,
            'conclusion': self._formulate_conclusion(test_results, evidence_assessment, significance_level)
        }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle hypothesis validation tasks."""
        hypothesis_keywords = [
            'hypothesis', 'test', 'validate', 'evidence', 'significant',
            'prove', 'confirm', 'support', 'reject'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in hypothesis_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _determine_test_strategy(self, hypothesis: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine appropriate statistical test strategy based on hypothesis."""
        hypothesis_lower = hypothesis.lower()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        strategy = {
            'primary_test': 'descriptive',
            'variables_involved': [],
            'test_assumptions': [],
            'alternative_tests': []
        }
        
        # Detect comparison hypotheses
        if any(word in hypothesis_lower for word in ['higher', 'lower', 'greater', 'less', 'different']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                strategy['primary_test'] = 'independent_t_test'
                strategy['variables_involved'] = [categorical_cols[0], numeric_cols[0]]
                strategy['test_assumptions'] = ['normality', 'equal_variances']
                strategy['alternative_tests'] = ['mann_whitney_u', 'welch_t_test']
        
        # Detect correlation hypotheses
        elif any(word in hypothesis_lower for word in ['correlate', 'associate', 'relationship', 'related']):
            if len(numeric_cols) >= 2:
                strategy['primary_test'] = 'correlation'
                strategy['variables_involved'] = numeric_cols[:2]
                strategy['test_assumptions'] = ['linearity', 'normality']
                strategy['alternative_tests'] = ['spearman_correlation']
        
        # Detect proportion/frequency hypotheses
        elif any(word in hypothesis_lower for word in ['proportion', 'frequency', 'rate', 'percentage']):
            if len(categorical_cols) > 0:
                strategy['primary_test'] = 'chi_square'
                strategy['variables_involved'] = categorical_cols[:2] if len(categorical_cols) >= 2 else [categorical_cols[0]]
                strategy['test_assumptions'] = ['expected_frequency_>=5']
                strategy['alternative_tests'] = ['fisher_exact']
        
        return strategy
    
    def _execute_statistical_tests(self, strategy: Dict[str, Any], df: pd.DataFrame, 
                                 significance_level: float) -> Dict[str, Any]:
        """Execute the determined statistical tests."""
        results = {}
        test_type = strategy['primary_test']
        variables = strategy['variables_involved']
        
        try:
            if test_type == 'independent_t_test' and len(variables) >= 2:
                results = self._perform_t_test(df, variables[1], variables[0])
            
            elif test_type == 'correlation' and len(variables) >= 2:
                results = self._perform_correlation_test(df, variables[0], variables[1])
            
            elif test_type == 'chi_square' and len(variables) >= 1:
                results = self._perform_chi_square_test(df, variables)
            
            else:
                results = self._perform_descriptive_analysis(df, variables)
            
        except Exception as e:
            results = {
                'error': str(e),
                'test_type': test_type,
                'fallback_analysis': self._perform_descriptive_analysis(df, variables)
            }
        
        return results
    
    def _perform_t_test(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Perform independent t-test."""
        from scipy import stats
        
        groups = df[group_col].unique()
        
        if len(groups) != 2:
            return {'error': f'Expected 2 groups, found {len(groups)}'}
        
        group1_data = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2_data = df[df[group_col] == groups[1]][numeric_col].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                             (len(group2_data) - 1) * group2_data.var()) / 
                            (len(group1_data) + len(group2_data) - 2))
        
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
        
        return {
            'test_type': 'independent_t_test',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size': float(cohens_d),
            'group1_mean': float(group1_data.mean()),
            'group2_mean': float(group2_data.mean()),
            'group1_n': len(group1_data),
            'group2_n': len(group2_data),
            'groups': groups.tolist()
        }
    
    def _perform_correlation_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Perform correlation test."""
        from scipy import stats
        
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        # Use only overlapping data points
        valid_idx = df[[col1, col2]].dropna().index
        data1 = df.loc[valid_idx, col1]
        data2 = df.loc[valid_idx, col2]
        
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(data1, data2)
        
        # Spearman correlation (non-parametric alternative)
        r_spearman, p_spearman = stats.spearmanr(data1, data2)
        
        return {
            'test_type': 'correlation',
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_r': float(r_spearman),
            'spearman_p': float(p_spearman),
            'sample_size': len(data1),
            'variables': [col1, col2]
        }
    
    def _perform_chi_square_test(self, df: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Perform chi-square test."""
        from scipy import stats
        
        if len(variables) == 1:
            # Goodness of fit test
            observed_freq = df[variables[0]].value_counts()
            expected_freq = [len(df) / len(observed_freq)] * len(observed_freq)
            
            chi2, p_value = stats.chisquare(observed_freq, expected_freq)
            
            return {
                'test_type': 'chi_square_goodness_of_fit',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': len(observed_freq) - 1,
                'observed_frequencies': observed_freq.to_dict()
            }
        
        elif len(variables) >= 2:
            # Test of independence
            contingency_table = pd.crosstab(df[variables[0]], df[variables[1]])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            return {
                'test_type': 'chi_square_independence',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'contingency_table': contingency_table.to_dict(),
                'variables': variables[:2]
            }
    
    def _perform_descriptive_analysis(self, df: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Perform descriptive analysis when specific tests aren't applicable."""
        analysis = {
            'test_type': 'descriptive_analysis',
            'variables_analyzed': variables or df.columns.tolist()
        }
        
        for var in (variables or df.columns.tolist()):
            if var in df.columns:
                if df[var].dtype in ['int64', 'float64']:
                    analysis[f'{var}_stats'] = {
                        'mean': float(df[var].mean()),
                        'median': float(df[var].median()),
                        'std': float(df[var].std()),
                        'min': float(df[var].min()),
                        'max': float(df[var].max())
                    }
                else:
                    analysis[f'{var}_freq'] = df[var].value_counts().to_dict()
        
        return analysis
    
    def _assess_evidence_strength(self, test_results: Dict[str, Any], hypothesis: str) -> Dict[str, Any]:
        """Assess the strength of evidence for the hypothesis."""
        assessment = {
            'evidence_level': 'insufficient',
            'confidence': 'low',
            'supporting_factors': [],
            'limiting_factors': []
        }
        
        p_value = test_results.get('p_value', 1.0)
        effect_size = test_results.get('effect_size', 0.0)
        sample_size = test_results.get('sample_size', test_results.get('group1_n', 0) + test_results.get('group2_n', 0))
        
        # Assess p-value
        if p_value < 0.001:
            assessment['supporting_factors'].append('Very strong statistical significance (p < 0.001)')
            assessment['evidence_level'] = 'strong'
        elif p_value < 0.01:
            assessment['supporting_factors'].append('Strong statistical significance (p < 0.01)')
            assessment['evidence_level'] = 'moderate'
        elif p_value < 0.05:
            assessment['supporting_factors'].append('Statistical significance (p < 0.05)')
            assessment['evidence_level'] = 'weak_support'
        else:
            assessment['limiting_factors'].append(f'Non-significant result (p = {p_value:.3f})')
        
        # Assess effect size
        if abs(effect_size) > 0.8:
            assessment['supporting_factors'].append('Large effect size')
        elif abs(effect_size) > 0.5:
            assessment['supporting_factors'].append('Medium effect size')
        elif abs(effect_size) > 0.2:
            assessment['supporting_factors'].append('Small effect size')
        else:
            assessment['limiting_factors'].append('Very small or negligible effect size')
        
        # Assess sample size
        if sample_size > 100:
            assessment['supporting_factors'].append('Adequate sample size')
        elif sample_size > 30:
            assessment['supporting_factors'].append('Moderate sample size')
        else:
            assessment['limiting_factors'].append('Small sample size may limit power')
        
        # Overall confidence
        if len(assessment['supporting_factors']) > len(assessment['limiting_factors']):
            assessment['confidence'] = 'high' if assessment['evidence_level'] in ['strong', 'moderate'] else 'medium'
        else:
            assessment['confidence'] = 'low'
        
        return assessment
    
    def _formulate_conclusion(self, test_results: Dict[str, Any], evidence_assessment: Dict[str, Any], 
                            significance_level: float) -> str:
        """Formulate a conclusion based on test results and evidence assessment."""
        p_value = test_results.get('p_value', 1.0)
        evidence_level = evidence_assessment.get('evidence_level', 'insufficient')
        confidence = evidence_assessment.get('confidence', 'low')
        
        if p_value < significance_level and evidence_level in ['strong', 'moderate']:
            conclusion = f"The hypothesis is SUPPORTED by the data with {confidence} confidence. "
            conclusion += f"Statistical significance (p = {p_value:.3f}) and {evidence_level} evidence suggest the effect is likely real."
        
        elif p_value < significance_level and evidence_level == 'weak_support':
            conclusion = f"The hypothesis has WEAK SUPPORT from the data. "
            conclusion += f"While statistically significant (p = {p_value:.3f}), the evidence strength is limited."
        
        else:
            conclusion = f"The hypothesis is NOT SUPPORTED by the current data. "
            conclusion += f"Insufficient evidence (p = {p_value:.3f}) to reject the null hypothesis."
        
        # Add caveats
        limiting_factors = evidence_assessment.get('limiting_factors', [])
        if limiting_factors:
            conclusion += f" Important limitations: {'; '.join(limiting_factors[:2])}."
        
        return conclusion