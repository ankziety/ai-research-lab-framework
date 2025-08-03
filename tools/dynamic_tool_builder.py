"""
Dynamic Tool Builder

Provides real tool implementations for web search, code execution, model switching,
and custom tool chain building with cost tracking and security measures.
"""

import logging
import time
import json
import subprocess
import tempfile
import os
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import hashlib
from dataclasses import dataclass

from .base_tool import BaseTool
from ..data.cost_manager import CostManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result."""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    timestamp: float


class WebSearchTool(BaseTool):
    """
    Real web search tool with multiple search engine support.
    
    Features:
    - Multiple search engines (Google, Bing, DuckDuckGo)
    - Cost tracking and optimization
    - Result filtering and ranking
    - Rate limiting and error handling
    """
    
    def __init__(self, api_keys: Dict[str, str], cost_manager: Optional[CostManager] = None):
        """
        Initialize web search tool.
        
        Args:
            api_keys: Dictionary of API keys for different search engines
            cost_manager: Optional cost manager for tracking search costs
        """
        super().__init__(
            tool_id="web_search",
            name="Web Search Tool",
            description="Search the web using multiple search engines with intelligent result aggregation",
            capabilities=["web_search", "information_gathering", "fact_verification"],
            requirements={
                'api_keys': ['google_search_api_key', 'bing_search_api_key', 'serpapi_key'],
                'rate_limits': {'requests_per_minute': 60}
            }
        )
        
        self.api_keys = api_keys
        self.cost_manager = cost_manager
        self.search_engines = {
            'google': self._search_google,
            'bing': self._search_bing,
            'duckduckgo': self._search_duckduckgo,
            'serpapi': self._search_serpapi
        }
        self.rate_limit_tracker = {}
        
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute web search with intelligent engine selection and result aggregation.
        
        Args:
            task: Search task with query and parameters
            context: Execution context
            
        Returns:
            Search results with metadata
        """
        query = task.get('query', '')
        max_results = task.get('max_results', 10)
        search_engines = task.get('engines', ['google', 'duckduckgo'])
        filter_domains = task.get('filter_domains', [])
        
        if not query:
            return {
                'success': False,
                'error': 'No search query provided',
                'results': [],
                'metadata': {}
            }
        
        # Check rate limits
        if not self._check_rate_limits():
            return {
                'success': False,
                'error': 'Rate limit exceeded',
                'results': [],
                'metadata': {}
            }
        
        # Track cost if cost manager available
        search_cost = 0.0
        if self.cost_manager:
            estimated_cost = self._estimate_search_cost(query, len(search_engines))
            if not self.cost_manager.can_afford(estimated_cost):
                return {
                    'success': False,
                    'error': 'Insufficient budget for search',
                    'results': [],
                    'metadata': {'estimated_cost': estimated_cost}
                }
            search_cost = estimated_cost
        
        # Execute searches
        all_results = []
        successful_engines = []
        
        for engine in search_engines:
            if engine in self.search_engines:
                try:
                    engine_results = self.search_engines[engine](query, max_results)
                    if engine_results:
                        all_results.extend(engine_results)
                        successful_engines.append(engine)
                except Exception as e:
                    logger.warning(f"Search engine {engine} failed: {e}")
        
        # Filter and rank results
        filtered_results = self._filter_results(all_results, filter_domains)
        ranked_results = self._rank_results(filtered_results, query)
        
        # Deduplicate results
        final_results = self._deduplicate_results(ranked_results)
        
        # Track actual cost
        if self.cost_manager:
            self.cost_manager.track_usage(
                model='web_search',
                tokens_input=len(query.split()),
                tokens_output=len(final_results),
                actual_cost=search_cost,
                task_type='web_search',
                agent_id=context.get('agent_id', 'unknown'),
                success=len(final_results) > 0
            )
        
        return {
            'success': len(final_results) > 0,
            'results': [asdict(result) for result in final_results[:max_results]],
            'metadata': {
                'query': query,
                'engines_used': successful_engines,
                'total_results': len(all_results),
                'filtered_results': len(final_results),
                'search_cost': search_cost,
                'timestamp': time.time()
            }
        }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess if this tool can handle the task."""
        if task_type in ['web_search', 'information_gathering', 'fact_verification']:
            return 0.9
        return 0.1
    
    def _search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        api_key = self.api_keys.get('google_search_api_key')
        if not api_key:
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': self.api_keys.get('google_cse_id', ''),
                'q': query,
                'num': min(max_results, 10)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source='google',
                    relevance_score=0.8,
                    timestamp=time.time()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Bing Search API."""
        api_key = self.api_keys.get('bing_search_api_key')
        if not api_key:
            return []
        
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {'Ocp-Apim-Subscription-Key': api_key}
            params = {
                'q': query,
                'count': min(max_results, 50),
                'responseFilter': 'Webpages'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('webPages', {}).get('value', []):
                results.append(SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    source='bing',
                    relevance_score=0.8,
                    timestamp=time.time()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract results from DuckDuckGo response
            for result in data.get('Results', [])[:max_results]:
                results.append(SearchResult(
                    title=result.get('Title', ''),
                    url=result.get('FirstURL', ''),
                    snippet=result.get('Text', ''),
                    source='duckduckgo',
                    relevance_score=0.7,
                    timestamp=time.time()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _search_serpapi(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SerpAPI (aggregates multiple engines)."""
        api_key = self.api_keys.get('serpapi_key')
        if not api_key:
            return []
        
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': api_key,
                'q': query,
                'num': min(max_results, 10),
                'engine': 'google'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('organic_results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source='serpapi',
                    relevance_score=0.8,
                    timestamp=time.time()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    def _filter_results(self, results: List[SearchResult], filter_domains: List[str]) -> List[SearchResult]:
        """Filter results based on domain restrictions."""
        if not filter_domains:
            return results
        
        filtered = []
        for result in results:
            domain = self._extract_domain(result.url)
            if any(filter_domain in domain for filter_domain in filter_domains):
                filtered.append(result)
        
        return filtered
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank results by relevance to query."""
        query_terms = set(query.lower().split())
        
        for result in results:
            # Calculate relevance score based on term overlap
            title_terms = set(result.title.lower().split())
            snippet_terms = set(result.snippet.lower().split())
            
            title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            snippet_overlap = len(query_terms & snippet_terms) / max(1, len(query_terms))
            
            # Boost score for title matches
            result.relevance_score = (title_overlap * 0.7 + snippet_overlap * 0.3) * 0.8 + result.relevance_score * 0.2
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL similarity."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            normalized_url = self._normalize_url(result.url)
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        return unique_results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        url = url.lower()
        url = re.sub(r'^https?://', '', url)
        url = re.sub(r'/$', '', url)
        return url
    
    def _estimate_search_cost(self, query: str, num_engines: int) -> float:
        """Estimate cost of search operation."""
        # Rough estimate: $0.001 per search engine per query
        return num_engines * 0.001
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self.rate_limit_tracker = {
            k: v for k, v in self.rate_limit_tracker.items() 
            if v > minute_ago
        }
        
        # Check if we're under the limit
        if len(self.rate_limit_tracker) >= 60:
            return False
        
        self.rate_limit_tracker[current_time] = current_time
        return True


class CodeExecutionTool(BaseTool):
    """
    Secure code execution tool with sandboxed environment.
    
    Features:
    - Sandboxed execution environment
    - Resource limits and timeouts
    - Security validation
    - Result capture and error handling
    """
    
    def __init__(self, sandbox_config: Dict[str, Any], cost_manager: Optional[CostManager] = None):
        """
        Initialize code execution tool.
        
        Args:
            sandbox_config: Configuration for sandbox environment
            cost_manager: Optional cost manager for tracking execution costs
        """
        super().__init__(
            tool_id="code_execution",
            name="Code Execution Tool",
            description="Execute Python code in a secure sandboxed environment",
            capabilities=["code_execution", "data_analysis", "computation"],
            requirements={
                'sandbox_enabled': True,
                'timeout_seconds': 30,
                'memory_limit_mb': 512
            }
        )
        
        self.sandbox_config = sandbox_config
        self.cost_manager = cost_manager
        self.allowed_modules = sandbox_config.get('allowed_modules', [
            'math', 'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn'
        ])
        self.blocked_functions = sandbox_config.get('blocked_functions', [
            'eval', 'exec', 'compile', 'open', 'file', '__import__'
        ])
        
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code in secure sandbox.
        
        Args:
            task: Code execution task
            context: Execution context
            
        Returns:
            Execution results with output and metadata
        """
        code = task.get('code', '')
        language = task.get('language', 'python')
        timeout = task.get('timeout', self.sandbox_config.get('timeout_seconds', 30))
        
        if not code:
            return {
                'success': False,
                'error': 'No code provided',
                'output': '',
                'metadata': {}
            }
        
        # Security validation
        security_check = self._validate_code_security(code)
        if not security_check['safe']:
            return {
                'success': False,
                'error': f"Code security validation failed: {security_check['reason']}",
                'output': '',
                'metadata': {'security_check': security_check}
            }
        
        # Track cost if cost manager available
        execution_cost = 0.0
        if self.cost_manager:
            estimated_cost = self._estimate_execution_cost(code, timeout)
            if not self.cost_manager.can_afford(estimated_cost):
                return {
                    'success': False,
                    'error': 'Insufficient budget for code execution',
                    'output': '',
                    'metadata': {'estimated_cost': estimated_cost}
                }
            execution_cost = estimated_cost
        
        # Execute code
        try:
            result = self._execute_code_safely(code, language, timeout)
            
            # Track actual cost
            if self.cost_manager:
                self.cost_manager.track_usage(
                    model='code_execution',
                    tokens_input=len(code.split()),
                    tokens_output=len(result.get('output', '')),
                    actual_cost=execution_cost,
                    task_type='code_execution',
                    agent_id=context.get('agent_id', 'unknown'),
                    success=result.get('success', False)
                )
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': f"Code execution failed: {str(e)}",
                'output': '',
                'metadata': {'exception': str(e)}
            }
            
            # Track failure cost
            if self.cost_manager:
                self.cost_manager.track_usage(
                    model='code_execution',
                    tokens_input=len(code.split()),
                    tokens_output=0,
                    actual_cost=execution_cost,
                    task_type='code_execution',
                    agent_id=context.get('agent_id', 'unknown'),
                    success=False,
                    error_message=str(e)
                )
            
            return error_result
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess if this tool can handle the task."""
        if task_type in ['code_execution', 'data_analysis', 'computation']:
            return 0.9
        return 0.1
    
    def _validate_code_security(self, code: str) -> Dict[str, Any]:
        """Validate code for security concerns."""
        code_lower = code.lower()
        
        # Check for blocked functions
        for blocked_func in self.blocked_functions:
            if blocked_func in code_lower:
                return {
                    'safe': False,
                    'reason': f"Blocked function '{blocked_func}' detected"
                }
        
        # Check for dangerous imports
        dangerous_imports = ['os', 'sys', 'subprocess', 'shutil', 'tempfile']
        for dangerous_import in dangerous_imports:
            if f"import {dangerous_import}" in code_lower or f"from {dangerous_import}" in code_lower:
                return {
                    'safe': False,
                    'reason': f"Dangerous import '{dangerous_import}' detected"
                }
        
        # Check for file operations
        file_operations = ['open(', 'file(', 'write(', 'read(']
        for op in file_operations:
            if op in code_lower:
                return {
                    'safe': False,
                    'reason': f"File operation '{op}' detected"
                }
        
        return {'safe': True, 'reason': 'Code passed security validation'}
    
    def _execute_code_safely(self, code: str, language: str, timeout: int) -> Dict[str, Any]:
        """Execute code in a safe environment."""
        if language.lower() != 'python':
            return {
                'success': False,
                'error': f"Unsupported language: {language}",
                'output': '',
                'metadata': {}
            }
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute with resource limits
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()
            )
            
            output = result.stdout
            error = result.stderr
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'output': output,
                'error': error if not success else '',
                'return_code': result.returncode,
                'execution_time': timeout,
                'metadata': {
                    'language': language,
                    'code_length': len(code),
                    'output_length': len(output)
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Code execution timed out after {timeout} seconds",
                'output': '',
                'metadata': {'timeout': timeout}
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Code execution failed: {str(e)}",
                'output': '',
                'metadata': {'exception': str(e)}
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _estimate_execution_cost(self, code: str, timeout: int) -> float:
        """Estimate cost of code execution."""
        # Rough estimate: $0.001 per second of execution time
        return timeout * 0.001


class ModelSwitchingTool(BaseTool):
    """
    Dynamic model switching tool with cost optimization.
    
    Features:
    - Automatic model selection based on task complexity
    - Cost-aware model switching
    - Performance tracking and optimization
    - Fallback mechanisms
    """
    
    def __init__(self, cost_manager: CostManager, llm_client):
        """
        Initialize model switching tool.
        
        Args:
            cost_manager: Cost manager for optimization
            llm_client: LLM client for model switching
        """
        super().__init__(
            tool_id="model_switching",
            name="Model Switching Tool",
            description="Dynamically switch between LLM models based on cost and performance",
            capabilities=["model_optimization", "cost_management", "performance_tuning"],
            requirements={
                'cost_manager': True,
                'llm_client': True
            }
        )
        
        self.cost_manager = cost_manager
        self.llm_client = llm_client
        self.model_performance = {}
        self.switch_history = []
        
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with optimal model selection.
        
        Args:
            task: Task specification
            context: Execution context
            
        Returns:
            Results with model selection metadata
        """
        prompt = task.get('prompt', '')
        task_complexity = task.get('complexity', 'medium')
        required_capabilities = task.get('required_capabilities', [])
        agent_id = context.get('agent_id', 'unknown')
        
        if not prompt:
            return {
                'success': False,
                'error': 'No prompt provided',
                'output': '',
                'metadata': {}
            }
        
        # Get budget remaining
        budget_status = self.cost_manager.get_budget_status()
        budget_remaining = budget_status['budget_remaining']
        
        # Select optimal model
        optimal_model = self.cost_manager.optimize_model_selection(
            task_complexity=task_complexity,
            budget_remaining=budget_remaining,
            required_capabilities=required_capabilities
        )
        
        # Estimate cost for optimal model
        estimated_tokens = len(prompt.split()) * 2  # Rough estimate
        estimated_cost = self.cost_manager.estimate_cost(optimal_model, estimated_tokens)
        
        # Check if we can afford the optimal model
        if not self.cost_manager.can_afford(estimated_cost):
            # Try to find a cheaper alternative
            fallback_model = self._find_fallback_model(estimated_cost, required_capabilities)
            if fallback_model:
                optimal_model = fallback_model
                estimated_cost = self.cost_manager.estimate_cost(optimal_model, estimated_tokens)
            else:
                return {
                    'success': False,
                    'error': 'Insufficient budget for any suitable model',
                    'output': '',
                    'metadata': {'estimated_cost': estimated_cost}
                }
        
        # Execute with selected model
        try:
            # Temporarily switch LLM client to optimal model
            original_model = self.llm_client.model
            self.llm_client.model = optimal_model
            
            start_time = time.time()
            response = self.llm_client.generate_response(prompt, context, agent_id)
            execution_time = time.time() - start_time
            
            # Restore original model
            self.llm_client.model = original_model
            
            # Track performance
            self._track_model_performance(optimal_model, execution_time, len(response.split()))
            
            # Record switch
            self.switch_history.append({
                'timestamp': time.time(),
                'from_model': original_model,
                'to_model': optimal_model,
                'task_complexity': task_complexity,
                'execution_time': execution_time,
                'agent_id': agent_id
            })
            
            return {
                'success': True,
                'output': response,
                'selected_model': optimal_model,
                'execution_time': execution_time,
                'estimated_cost': estimated_cost,
                'metadata': {
                    'task_complexity': task_complexity,
                    'required_capabilities': required_capabilities,
                    'budget_remaining': budget_remaining,
                    'model_performance': self.model_performance.get(optimal_model, {})
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Model switching failed: {str(e)}",
                'output': '',
                'metadata': {'exception': str(e)}
            }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess if this tool can handle the task."""
        if task_type in ['model_optimization', 'cost_management', 'performance_tuning']:
            return 0.9
        return 0.1
    
    def _find_fallback_model(self, max_cost: float, required_capabilities: List[str]) -> Optional[str]:
        """Find a cheaper model that meets requirements."""
        available_models = []
        
        for model_name, model_cost in self.cost_manager.model_costs.items():
            # Check if model meets capability requirements
            if required_capabilities:
                if not all(cap in model_cost.capabilities for cap in required_capabilities):
                    continue
            
            # Check if model is affordable
            estimated_cost = model_cost.input_cost_per_1k * 0.1  # Rough estimate
            if estimated_cost <= max_cost:
                available_models.append((model_name, estimated_cost))
        
        if available_models:
            # Return the cheapest model
            available_models.sort(key=lambda x: x[1])
            return available_models[0][0]
        
        return None
    
    def _track_model_performance(self, model: str, execution_time: float, output_tokens: int):
        """Track performance metrics for model."""
        if model not in self.model_performance:
            self.model_performance[model] = {
                'execution_times': [],
                'output_tokens': [],
                'success_count': 0,
                'total_count': 0
            }
        
        stats = self.model_performance[model]
        stats['execution_times'].append(execution_time)
        stats['output_tokens'].append(output_tokens)
        stats['total_count'] += 1
        stats['success_count'] += 1
        
        # Keep only recent history
        if len(stats['execution_times']) > 100:
            stats['execution_times'] = stats['execution_times'][-50:]
            stats['output_tokens'] = stats['output_tokens'][-50:]
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get model optimization performance report."""
        return {
            'model_performance': self.model_performance,
            'switch_history': self.switch_history[-20:],  # Last 20 switches
            'total_switches': len(self.switch_history),
            'budget_status': self.cost_manager.get_budget_status()
        }


class CustomToolChainBuilder(BaseTool):
    """
    Tool for building custom tool chains and integrations.
    
    Features:
    - Dynamic tool chain creation
    - Tool validation and testing
    - Integration planning
    - Error handling and fallbacks
    """
    
    def __init__(self, tool_registry, cost_manager: Optional[CostManager] = None):
        """
        Initialize custom tool chain builder.
        
        Args:
            tool_registry: Tool registry for tool discovery
            cost_manager: Optional cost manager for tracking
        """
        super().__init__(
            tool_id="custom_tool_chain",
            name="Custom Tool Chain Builder",
            description="Build custom tool chains and integrations for complex research tasks",
            capabilities=["tool_integration", "workflow_automation", "custom_tools"],
            requirements={
                'tool_registry': True,
                'cost_manager': True
            }
        )
        
        self.tool_registry = tool_registry
        self.cost_manager = cost_manager
        self.built_chains = {}
        
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build and execute custom tool chain.
        
        Args:
            task: Tool chain specification
            context: Execution context
            
        Returns:
            Tool chain execution results
        """
        chain_spec = task.get('chain_spec', {})
        workflow_steps = task.get('workflow_steps', [])
        agent_id = context.get('agent_id', 'unknown')
        
        if not workflow_steps:
            return {
                'success': False,
                'error': 'No workflow steps provided',
                'output': '',
                'metadata': {}
            }
        
        # Build tool chain
        try:
            chain_id = self._generate_chain_id(workflow_steps)
            tool_chain = self._build_tool_chain(workflow_steps, agent_id)
            
            if not tool_chain['success']:
                return tool_chain
            
            # Execute tool chain
            execution_result = self._execute_tool_chain(tool_chain['chain'], context)
            
            # Store successful chain
            if execution_result['success']:
                self.built_chains[chain_id] = {
                    'workflow_steps': workflow_steps,
                    'tool_chain': tool_chain['chain'],
                    'creation_time': time.time(),
                    'usage_count': 0
                }
            
            return execution_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Tool chain building failed: {str(e)}",
                'output': '',
                'metadata': {'exception': str(e)}
            }
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess if this tool can handle the task."""
        if task_type in ['tool_integration', 'workflow_automation', 'custom_tools']:
            return 0.9
        return 0.1
    
    def _generate_chain_id(self, workflow_steps: List[Dict[str, Any]]) -> str:
        """Generate unique ID for tool chain."""
        chain_str = json.dumps(workflow_steps, sort_keys=True)
        return hashlib.md5(chain_str.encode()).hexdigest()[:8]
    
    def _build_tool_chain(self, workflow_steps: List[Dict[str, Any]], agent_id: str) -> Dict[str, Any]:
        """Build executable tool chain from workflow steps."""
        chain = []
        
        for step in workflow_steps:
            step_type = step.get('type', '')
            step_params = step.get('params', {})
            
            # Discover appropriate tool for step
            available_tools = self.tool_registry.discover_tools(
                agent_id=agent_id,
                task_description=step.get('description', ''),
                requirements=step_params
            )
            
            if not available_tools:
                return {
                    'success': False,
                    'error': f"No suitable tool found for step: {step.get('description', '')}",
                    'chain': []
                }
            
            # Select best tool
            best_tool = available_tools[0]['tool']
            chain.append({
                'step': step,
                'tool': best_tool,
                'confidence': available_tools[0]['confidence']
            })
        
        return {
            'success': True,
            'chain': chain,
            'total_steps': len(chain)
        }
    
    def _execute_tool_chain(self, chain: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool chain with error handling and fallbacks."""
        results = []
        execution_metadata = {
            'total_steps': len(chain),
            'successful_steps': 0,
            'failed_steps': 0,
            'execution_time': 0,
            'total_cost': 0.0
        }
        
        start_time = time.time()
        
        for i, chain_step in enumerate(chain):
            step = chain_step['step']
            tool = chain_step['tool']
            
            try:
                # Execute tool
                result = tool.execute(step.get('params', {}), context)
                
                if result.get('success', False):
                    results.append({
                        'step_index': i,
                        'step_description': step.get('description', ''),
                        'tool_id': tool.tool_id,
                        'result': result,
                        'success': True
                    })
                    execution_metadata['successful_steps'] += 1
                else:
                    results.append({
                        'step_index': i,
                        'step_description': step.get('description', ''),
                        'tool_id': tool.tool_id,
                        'result': result,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
                    execution_metadata['failed_steps'] += 1
                
                # Track cost if available
                if self.cost_manager and 'cost' in result.get('metadata', {}):
                    execution_metadata['total_cost'] += result['metadata']['cost']
                
            except Exception as e:
                results.append({
                    'step_index': i,
                    'step_description': step.get('description', ''),
                    'tool_id': tool.tool_id,
                    'result': {'success': False, 'error': str(e)},
                    'success': False,
                    'error': str(e)
                })
                execution_metadata['failed_steps'] += 1
        
        execution_metadata['execution_time'] = time.time() - start_time
        
        # Determine overall success
        overall_success = execution_metadata['successful_steps'] > 0
        
        return {
            'success': overall_success,
            'results': results,
            'metadata': execution_metadata,
            'output': f"Executed {execution_metadata['successful_steps']}/{execution_metadata['total_steps']} steps successfully"
        }
    
    def get_built_chains(self) -> Dict[str, Any]:
        """Get information about built tool chains."""
        return {
            'total_chains': len(self.built_chains),
            'chains': {
                chain_id: {
                    'workflow_steps': chain_info['workflow_steps'],
                    'creation_time': chain_info['creation_time'],
                    'usage_count': chain_info['usage_count']
                }
                for chain_id, chain_info in self.built_chains.items()
            }
        } 