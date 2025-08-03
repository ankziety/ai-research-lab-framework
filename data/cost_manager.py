"""
Cost Management System for AI Research Framework

Provides real-time cost tracking, budget enforcement, and cost optimization
for all API calls and tool usage in the Virtual Lab framework.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Record of a single cost transaction."""
    timestamp: float
    model: str
    provider: str
    tokens_input: int
    tokens_output: int
    cost: float
    task_type: str
    agent_id: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelCost:
    """Cost configuration for a specific model."""
    model_name: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_tokens: int
    capabilities: List[str]
    reliability_score: float


class CostManager:
    """
    Comprehensive cost management system for AI research framework.
    
    Features:
    - Real-time cost tracking for all API calls
    - Budget limit enforcement with automatic model switching
    - Cost estimation before API calls
    - Detailed spending analytics and reporting
    - Integration with all LLM providers
    """
    
    def __init__(self, budget_limit: float, config: Dict[str, Any]):
        """
        Initialize cost manager.
        
        Args:
            budget_limit: Total budget limit in USD
            config: Configuration dictionary with model costs and settings
        """
        self.budget_limit = budget_limit
        self.current_spending = 0.0
        self.cost_history: List[CostRecord] = []
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load model costs from config
        self.model_costs = self._load_model_costs(config)
        
        # Budget alerts
        self.budget_alerts = {
            0.5: False,  # 50% warning
            0.8: False,  # 80% warning
            0.95: False  # 95% warning
        }
        
        # Cost optimization settings
        self.optimization_enabled = config.get('cost_optimization', True)
        self.auto_switch_threshold = config.get('auto_switch_threshold', 0.8)
        
        # Load existing cost data if available
        self._load_cost_data()
        
        logger.info(f"Cost manager initialized with budget: ${budget_limit:.2f}")
    
    def _load_model_costs(self, config: Dict[str, Any]) -> Dict[str, ModelCost]:
        """Load model cost configurations."""
        default_costs = {
            'gpt-4o': ModelCost(
                model_name='gpt-4o',
                provider='openai',
                input_cost_per_1k=0.005,
                output_cost_per_1k=0.015,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code', 'vision'],
                reliability_score=0.95
            ),
            'gpt-4o-mini': ModelCost(
                model_name='gpt-4o-mini',
                provider='openai',
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                max_tokens=128000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'gpt-3.5-turbo': ModelCost(
                model_name='gpt-3.5-turbo',
                provider='openai',
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0015,
                max_tokens=16385,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.80
            ),
            'claude-3-sonnet': ModelCost(
                model_name='claude-3-sonnet',
                provider='anthropic',
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.90
            ),
            'claude-3-haiku': ModelCost(
                model_name='claude-3-haiku',
                provider='anthropic',
                input_cost_per_1k=0.00025,
                output_cost_per_1k=0.00125,
                max_tokens=200000,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.75
            ),
            'gemini-pro': ModelCost(
                model_name='gemini-pro',
                provider='google',
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0015,
                max_tokens=32768,
                capabilities=['reasoning', 'analysis', 'code'],
                reliability_score=0.85
            ),
            'llama2': ModelCost(
                model_name='llama2',
                provider='ollama',
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                max_tokens=4096,
                capabilities=['reasoning', 'analysis'],
                reliability_score=0.70
            )
        }
        
        # Override with config values
        if 'model_costs' in config:
            for model_name, cost_config in config['model_costs'].items():
                if model_name in default_costs:
                    default_costs[model_name] = ModelCost(
                        model_name=model_name,
                        provider=cost_config.get('provider', default_costs[model_name].provider),
                        input_cost_per_1k=cost_config.get('input_cost_per_1k', default_costs[model_name].input_cost_per_1k),
                        output_cost_per_1k=cost_config.get('output_cost_per_1k', default_costs[model_name].output_cost_per_1k),
                        max_tokens=cost_config.get('max_tokens', default_costs[model_name].max_tokens),
                        capabilities=cost_config.get('capabilities', default_costs[model_name].capabilities),
                        reliability_score=cost_config.get('reliability_score', default_costs[model_name].reliability_score)
                    )
        
        return default_costs
    
    def estimate_cost(self, model: str, tokens_input: int, tokens_output: int = 0) -> float:
        """
        Estimate cost for a specific model and token usage.
        
        Args:
            model: Model name
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens (estimated)
            
        Returns:
            Estimated cost in USD
        """
        if model not in self.model_costs:
            logger.warning(f"Unknown model: {model}, using default cost")
            return tokens_input * 0.001 + tokens_output * 0.002
        
        model_cost = self.model_costs[model]
        
        input_cost = (tokens_input / 1000) * model_cost.input_cost_per_1k
        output_cost = (tokens_output / 1000) * model_cost.output_cost_per_1k
        
        return input_cost + output_cost
    
    def can_afford(self, estimated_cost: float) -> bool:
        """
        Check if the estimated cost fits within budget.
        
        Args:
            estimated_cost: Estimated cost in USD
            
        Returns:
            True if cost is affordable
        """
        return (self.current_spending + estimated_cost) <= self.budget_limit
    
    def track_usage(self, model: str, tokens_input: int, tokens_output: int, 
                   actual_cost: float, task_type: str, agent_id: str, 
                   success: bool = True, error_message: Optional[str] = None):
        """
        Track actual usage and cost.
        
        Args:
            model: Model used
            tokens_input: Input tokens consumed
            tokens_output: Output tokens generated
            actual_cost: Actual cost incurred
            task_type: Type of task performed
            agent_id: ID of the agent that used the model
            success: Whether the operation was successful
            error_message: Error message if failed
        """
        cost_record = CostRecord(
            timestamp=time.time(),
            model=model,
            provider=self.model_costs.get(model, ModelCost(model, 'unknown', 0, 0, 0, [], 0)).provider,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=actual_cost,
            task_type=task_type,
            agent_id=agent_id,
            success=success,
            error_message=error_message
        )
        
        self.cost_history.append(cost_record)
        self.current_spending += actual_cost
        
        # Update usage statistics
        if model not in self.usage_stats:
            self.usage_stats[model] = {
                'total_tokens': 0,
                'total_cost': 0.0,
                'usage_count': 0,
                'success_count': 0,
                'last_used': None
            }
        
        stats = self.usage_stats[model]
        stats['total_tokens'] += tokens_input + tokens_output
        stats['total_cost'] += actual_cost
        stats['usage_count'] += 1
        stats['last_used'] = time.time()
        
        if success:
            stats['success_count'] += 1
        
        # Check budget alerts
        self._check_budget_alerts()
        
        # Save cost data periodically
        if len(self.cost_history) % 10 == 0:
            self._save_cost_data()
        
        logger.info(f"Cost tracked: ${actual_cost:.4f} for {model} ({tokens_input + tokens_output} tokens)")
    
    def optimize_model_selection(self, task_complexity: str, budget_remaining: float, 
                               required_capabilities: List[str] = None) -> str:
        """
        Select optimal model based on cost, capability, and budget.
        
        Args:
            task_complexity: 'simple', 'medium', or 'complex'
            budget_remaining: Remaining budget
            required_capabilities: List of required capabilities
            
        Returns:
            Optimal model name
        """
        if not self.optimization_enabled:
            return 'gpt-4o'  # Default to most capable model
        
        # Filter models by required capabilities
        available_models = []
        for model_name, model_cost in self.model_costs.items():
            if required_capabilities:
                if not all(cap in model_cost.capabilities for cap in required_capabilities):
                    continue
            available_models.append((model_name, model_cost))
        
        if not available_models:
            logger.warning("No models available with required capabilities")
            return 'gpt-4o'
        
        # Score models based on cost efficiency and capability
        model_scores = []
        for model_name, model_cost in available_models:
            # Cost efficiency score (lower is better)
            avg_cost_per_1k = (model_cost.input_cost_per_1k + model_cost.output_cost_per_1k) / 2
            cost_score = 1.0 / (avg_cost_per_1k + 0.0001)  # Avoid division by zero
            
            # Capability score based on task complexity
            capability_score = 1.0
            if task_complexity == 'complex':
                capability_score = model_cost.reliability_score
            elif task_complexity == 'simple':
                capability_score = 0.5 + (model_cost.reliability_score * 0.5)
            
            # Budget consideration
            budget_score = 1.0
            if budget_remaining < 1.0:  # Low budget
                budget_score = cost_score / max(cost_score for _, _ in available_models)
            
            # Combined score
            total_score = cost_score * capability_score * budget_score
            model_scores.append((model_name, total_score))
        
        # Sort by score and return best model
        model_scores.sort(key=lambda x: x[1], reverse=True)
        optimal_model = model_scores[0][0]
        
        logger.info(f"Selected optimal model: {optimal_model} for {task_complexity} task")
        return optimal_model
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and spending analytics."""
        total_usage = len(self.cost_history)
        successful_usage = sum(1 for record in self.cost_history if record.success)
        
        # Calculate spending by provider
        provider_spending = {}
        for record in self.cost_history:
            provider = record.provider
            if provider not in provider_spending:
                provider_spending[provider] = 0.0
            provider_spending[provider] += record.cost
        
        # Calculate spending by model
        model_spending = {}
        for record in self.cost_history:
            model = record.model
            if model not in model_spending:
                model_spending[model] = 0.0
            model_spending[model] += record.cost
        
        return {
            'budget_limit': self.budget_limit,
            'current_spending': self.current_spending,
            'budget_remaining': self.budget_limit - self.current_spending,
            'budget_utilization': self.current_spending / self.budget_limit,
            'total_usage_count': total_usage,
            'successful_usage_count': successful_usage,
            'success_rate': successful_usage / max(1, total_usage),
            'provider_spending': provider_spending,
            'model_spending': model_spending,
            'usage_stats': self.usage_stats,
            'budget_alerts': self.budget_alerts
        }
    
    def get_cost_analytics(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Get detailed cost analytics for the specified time period.
        
        Args:
            time_period_hours: Hours to look back for analytics
            
        Returns:
            Cost analytics dictionary
        """
        cutoff_time = time.time() - (time_period_hours * 3600)
        recent_records = [r for r in self.cost_history if r.timestamp >= cutoff_time]
        
        if not recent_records:
            return {'error': 'No cost data available for specified period'}
        
        # Calculate analytics
        total_cost = sum(r.cost for r in recent_records)
        total_tokens = sum(r.tokens_input + r.tokens_output for r in recent_records)
        avg_cost_per_token = total_cost / max(1, total_tokens)
        
        # Cost by task type
        task_costs = {}
        for record in recent_records:
            task_type = record.task_type
            if task_type not in task_costs:
                task_costs[task_type] = 0.0
            task_costs[task_type] += record.cost
        
        # Cost by agent
        agent_costs = {}
        for record in recent_records:
            agent_id = record.agent_id
            if agent_id not in agent_costs:
                agent_costs[agent_id] = 0.0
            agent_costs[agent_id] += record.cost
        
        return {
            'period_hours': time_period_hours,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'avg_cost_per_token': avg_cost_per_token,
            'record_count': len(recent_records),
            'task_costs': task_costs,
            'agent_costs': agent_costs,
            'model_usage': {r.model: sum(1 for rr in recent_records if rr.model == r.model) for r in recent_records}
        }
    
    def _check_budget_alerts(self):
        """Check and trigger budget alerts."""
        utilization = self.current_spending / self.budget_limit
        
        for threshold, alerted in self.budget_alerts.items():
            if utilization >= threshold and not alerted:
                logger.warning(f"Budget alert: {utilization:.1%} of budget used (${self.current_spending:.2f}/{self.budget_limit:.2f})")
                self.budget_alerts[threshold] = True
            elif utilization < threshold:
                self.budget_alerts[threshold] = False
    
    def _save_cost_data(self):
        """Save cost data to persistent storage."""
        try:
            data_dir = Path("data/costs")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            cost_data = {
                'budget_limit': self.budget_limit,
                'current_spending': self.current_spending,
                'cost_history': [asdict(record) for record in self.cost_history],
                'usage_stats': self.usage_stats,
                'budget_alerts': self.budget_alerts
            }
            
            with open(data_dir / "cost_data.json", 'w') as f:
                json.dump(cost_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cost data: {e}")
    
    def _load_cost_data(self):
        """Load cost data from persistent storage."""
        try:
            data_file = Path("data/costs/cost_data.json")
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                self.current_spending = data.get('current_spending', 0.0)
                self.usage_stats = data.get('usage_stats', {})
                self.budget_alerts = data.get('budget_alerts', {0.5: False, 0.8: False, 0.95: False})
                
                # Load cost history
                cost_history_data = data.get('cost_history', [])
                self.cost_history = []
                for record_data in cost_history_data:
                    self.cost_history.append(CostRecord(**record_data))
                
                logger.info(f"Loaded cost data: ${self.current_spending:.2f} spent")
                
        except Exception as e:
            logger.warning(f"Failed to load cost data: {e}")
    
    def reset_budget(self, new_budget: float):
        """Reset budget and clear history."""
        self.budget_limit = new_budget
        self.current_spending = 0.0
        self.cost_history.clear()
        self.usage_stats.clear()
        self.budget_alerts = {0.5: False, 0.8: False, 0.95: False}
        
        logger.info(f"Budget reset to ${new_budget:.2f}")
    
    def export_cost_report(self, filepath: str):
        """
        Export detailed cost report to file.
        
        Args:
            filepath: Path to export report
        """
        try:
            report = {
                'budget_status': self.get_budget_status(),
                'cost_analytics_24h': self.get_cost_analytics(24),
                'cost_analytics_7d': self.get_cost_analytics(168),
                'model_costs': {name: asdict(cost) for name, cost in self.model_costs.items()},
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Cost report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export cost report: {e}") 