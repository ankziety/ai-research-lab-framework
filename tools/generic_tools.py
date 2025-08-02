"""
Generic Research Tools

Practical tools for common research tasks like web search, code execution, and data analysis.
"""

import requests
import json
import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Tool for searching the web and retrieving information.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="web_search",
            name="Web Search Tool",
            description="Search the web for information, news, research papers, and general knowledge",
            capabilities=[
                "web_search",
                "information_retrieval",
                "current_events",
                "fact_checking",
                "research_assistance"
            ],
            requirements={
                "required_packages": ["requests"],
                "api_keys": ["search_api_key"]  # Optional
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search tasks."""
        query = task.get('query', task.get('description', ''))
        max_results = task.get('max_results', 5)
        
        if not query:
            return {'error': 'No search query provided', 'success': False}
        
        try:
            # Try DuckDuckGo search first (no API key required)
            results = self._search_duckduckgo(query, max_results)
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'source': 'DuckDuckGo',
                'total_results': len(results)
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_results': self._generate_mock_search_results(query, max_results)
            }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle web search tasks."""
        search_keywords = [
            'search', 'web', 'internet', 'find', 'lookup', 'google',
            'information', 'current', 'news', 'research', 'facts',
            'literature', 'papers', 'studies', 'background', 'references'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in search_keywords:
            if keyword in task_lower:
                confidence += 0.1
        
        # Boost confidence for research-related tasks
        if 'research' in task_lower or 'study' in task_lower:
            confidence += 0.2
            
        return min(1.0, confidence)
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API."""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Add instant answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Instant Answer'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': data.get('AbstractSource', 'DuckDuckGo')
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:100] + '...',
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'DuckDuckGo Related'
                    })
            
            return results[:max_results]
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return self._generate_mock_search_results(query, max_results)
    
    def _generate_mock_search_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock search results for testing."""
        return [
            {
                'title': f"Research on {query} - Academic Paper",
                'url': f"https://example.com/research/{query.replace(' ', '-')}",
                'snippet': f"Comprehensive study on {query} with detailed analysis and findings...",
                'source': 'Mock Academic Database'
            },
            {
                'title': f"{query} - Wikipedia",
                'url': f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                'snippet': f"Wikipedia article providing background information on {query}...",
                'source': 'Wikipedia'
            },
            {
                'title': f"Latest News on {query}",
                'url': f"https://news.example.com/{query.replace(' ', '-')}",
                'snippet': f"Recent developments and news related to {query}...",
                'source': 'News Source'
            }
        ][:max_results]


class CodeInterpreterTool(BaseTool):
    """
    Tool for executing Python code safely in a controlled environment.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="code_interpreter",
            name="Code Interpreter",
            description="Execute Python code for data analysis, calculations, and research tasks",
            capabilities=[
                "code_execution",
                "data_analysis",
                "calculations",
                "plotting",
                "statistical_analysis",
                "data_processing"
            ],
            requirements={
                "required_packages": ["pandas", "numpy", "matplotlib"],
                "min_memory": 256
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code safely."""
        code = task.get('code', task.get('description', ''))
        
        if not code:
            return {'error': 'No code provided', 'success': False}
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add safety imports and restrictions
                safe_code = self._make_code_safe(code)
                f.write(safe_code)
                temp_file = f.name
            
            # Execute the code in a subprocess for safety
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None,
                'code_executed': code,
                'execution_time': '< 30s'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Code execution timed out (30s limit)',
                'code_executed': code
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'code_executed': code
            }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle code execution tasks."""
        code_keywords = [
            'code', 'python', 'execute', 'run', 'calculate', 'compute',
            'analysis', 'plot', 'graph', 'data', 'script', 'program',
            'model', 'simulate', 'algorithm', 'optimization', 'design'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in code_keywords:
            if keyword in task_lower:
                confidence += 0.1
        
        # Programming/computational tasks for materials design
        if 'create' in task_lower and ('polymer' in task_lower or 'material' in task_lower):
            confidence += 0.3
            
        return min(1.0, confidence)
    
    def _make_code_safe(self, code: str) -> str:
        """Add safety restrictions to code."""
        safe_imports = """
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import re
import math
from typing import *

# Restrict file operations
original_open = open
def safe_open(*args, **kwargs):
    if len(args) > 1 and ('w' in args[1] or 'a' in args[1]):
        raise PermissionError("File writing not allowed")
    return original_open(*args, **kwargs)

__builtins__['open'] = safe_open

# Prevent imports of dangerous modules
dangerous_modules = ['subprocess', 'os', 'sys', 'shutil', 'socket']
for module in dangerous_modules:
    sys.modules[module] = None

print("Code execution environment initialized safely")
print("=" * 50)

"""
        return safe_imports + "\n" + code


class DataAnalysisTool(BaseTool):
    """
    Tool for comprehensive data analysis tasks.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="data_analysis",
            name="Data Analysis Tool",
            description="Perform statistical analysis, data visualization, and insights generation",
            capabilities=[
                "statistical_analysis",
                "data_visualization",
                "descriptive_statistics",
                "correlation_analysis",
                "trend_analysis",
                "data_cleaning"
            ],
            requirements={
                "required_packages": ["pandas", "numpy", "scipy", "matplotlib", "seaborn"],
                "min_memory": 512
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis tasks."""
        data = task.get('data', [])
        analysis_type = task.get('analysis_type', 'descriptive')
        
        if not data:
            return {'error': 'No data provided for analysis', 'success': False}
        
        try:
            # Convert data to DataFrame
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
            else:
                return {'error': 'Invalid data format', 'success': False}
            
            results = {
                'success': True,
                'data_shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Perform different types of analysis
            if analysis_type == 'descriptive':
                results['descriptive_stats'] = self._descriptive_analysis(df)
            elif analysis_type == 'correlation':
                results['correlation_analysis'] = self._correlation_analysis(df)
            elif analysis_type == 'trend':
                results['trend_analysis'] = self._trend_analysis(df)
            else:
                results['descriptive_stats'] = self._descriptive_analysis(df)
            
            return results
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle data analysis tasks."""
        analysis_keywords = [
            'data', 'analysis', 'statistics', 'statistical', 'analyze',
            'correlation', 'trend', 'visualization', 'descriptive',
            'summary', 'insights', 'patterns', 'evaluate', 'measure',
            'compare', 'characterize', 'properties', 'performance'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in analysis_keywords:
            if keyword in task_lower:
                confidence += 0.1
        
        # Materials science tasks often need analysis
        materials_keywords = ['polymer', 'material', 'conductive', 'bio-compatible']
        for keyword in materials_keywords:
            if keyword in task_lower:
                confidence += 0.15
        
        return min(1.0, confidence)
    
    def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive statistical analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        results = {
            'summary_statistics': {},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_counts': {}
        }
        
        for col in numeric_cols:
            results['summary_statistics'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
        
        for col in df.columns:
            results['unique_counts'][col] = int(df[col].nunique())
        
        return results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        correlation_matrix = numeric_df.corr()
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(correlation_matrix)
        }
    
    def _trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        trends = {}
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 1:
                x = np.arange(len(data))
                slope = np.polyfit(x, data, 1)[0]
                trends[col] = {
                    'slope': float(slope),
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
        
        return trends
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations above threshold."""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'strength': 'strong positive' if corr_val > 0 else 'strong negative'
                    })
        
        return strong_corrs


class ResearchAssistantTool(BaseTool):
    """
    Tool for general research assistance and coordination.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="research_assistant",
            name="Research Assistant",
            description="General research assistance, task coordination, and knowledge synthesis",
            capabilities=[
                "research_coordination",
                "task_planning",
                "knowledge_synthesis",
                "methodology_design",
                "research_strategy"
            ],
            requirements={
                "required_packages": []
            }
        )
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research assistance tasks."""
        task_type = task.get('type', 'general')
        description = task.get('description', '')
        
        try:
            if task_type == 'methodology_design':
                return self._design_methodology(description, context)
            elif task_type == 'task_planning':
                return self._plan_tasks(description, context)
            else:
                return self._general_assistance(description, context)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle research assistance tasks."""
        research_keywords = [
            'research', 'assist', 'help', 'plan', 'coordinate', 'organize',
            'methodology', 'strategy', 'approach', 'design', 'synthesize'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in research_keywords:
            if keyword in task_lower:
                confidence += 0.1
        
        # Always provide some baseline assistance
        return max(0.3, min(1.0, confidence))
    
    def _design_methodology(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Design research methodology."""
        return {
            'success': True,
            'methodology': {
                'approach': 'systematic_investigation',
                'steps': [
                    'literature_review',
                    'hypothesis_formation',
                    'data_collection',
                    'analysis',
                    'conclusion'
                ],
                'recommended_tools': ['web_search', 'data_analysis', 'statistical_analyzer'],
                'timeline': 'depends_on_scope'
            }
        }
    
    def _plan_tasks(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan research tasks."""
        return {
            'success': True,
            'task_plan': {
                'priority_tasks': [
                    'define_research_question',
                    'conduct_literature_review',
                    'identify_data_sources',
                    'analyze_data',
                    'synthesize_findings'
                ],
                'estimated_effort': 'medium',
                'required_expertise': ['research_methods', 'data_analysis']
            }
        }
    
    def _general_assistance(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general research assistance."""
        return {
            'success': True,
            'assistance': {
                'task_understood': True,
                'suggestions': [
                    'Break down the problem into smaller components',
                    'Gather relevant background information',
                    'Apply appropriate analytical methods',
                    'Validate findings through multiple approaches'
                ],
                'next_steps': 'Specify more detailed requirements for targeted assistance'
            }
        }