"""
Complete Agent Tool System Example

Demonstrates the full implementation of agent tool discovery, request, and usage
systems with cost management in the Virtual Lab framework.
"""

import logging
import time
from typing import Dict, List, Any, Optional

# Import the complete system components
from cost_manager import CostManager
from tools.dynamic_tool_builder import WebSearchTool, CodeExecutionTool, ModelSwitchingTool, CustomToolChainBuilder
from tools.tool_registry import ToolRegistry
from agents.base_agent import BaseAgent
from agents.llm_client import get_llm_client
from virtual_lab import VirtualLabMeetingSystem
from agents import PrincipalInvestigatorAgent, ScientificCriticAgent, AgentMarketplace
from multi_agent_framework import MultiAgentResearchFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_complete_system() -> Dict[str, Any]:
    """
    Setup the complete agent tool system with cost management.
    
    Returns:
        Dictionary containing all system components
    """
    logger.info("Setting up complete agent tool system...")
    
    # 1. Initialize cost manager
    budget_limit = 50.0  # $50 budget for demonstration
    cost_config = {
        'budget_limit': budget_limit,
        'cost_optimization': True,
        'enable_dynamic_tools': True,
        'default_model': 'gpt-3.5-turbo',
        'premium_model': 'gpt-4',
        'web_search_apis': {
            'google_search_api_key': 'demo_key',
            'bing_search_api_key': 'demo_key',
            'serpapi_key': 'demo_key'
        },
        'code_execution': {
            'sandbox_enabled': True,
            'timeout_seconds': 30,
            'memory_limit_mb': 512
        }
    }
    
    cost_manager = CostManager(budget_limit, cost_config)
    logger.info(f"Cost manager initialized with budget: ${budget_limit:.2f}")
    
    # 2. Initialize tool registry
    tool_registry = ToolRegistry()
    
    # 3. Create dynamic tools
    web_search_tool = WebSearchTool(
        api_keys=cost_config['web_search_apis'],
        cost_manager=cost_manager
    )
    
    code_execution_tool = CodeExecutionTool(
        sandbox_config=cost_config['code_execution'],
        cost_manager=cost_manager
    )
    
    # 4. Register tools in registry
    tool_registry.register_tool(web_search_tool, categories=['search', 'information_gathering'])
    tool_registry.register_tool(code_execution_tool, categories=['computation', 'data_analysis'])
    
    # 5. Initialize LLM client
    llm_client = get_llm_client({
        'default_llm_provider': 'openai',
        'default_model': 'gpt-3.5-turbo',
        'openai_api_key': 'demo_key'
    })
    
    # 6. Create model switching tool
    model_switching_tool = ModelSwitchingTool(cost_manager, llm_client)
    tool_registry.register_tool(model_switching_tool, categories=['optimization', 'cost_management'])
    
    # 7. Create custom tool chain builder
    custom_chain_builder = CustomToolChainBuilder(tool_registry, cost_manager)
    tool_registry.register_tool(custom_chain_builder, categories=['integration', 'workflow'])
    
    # 8. Initialize agents
    pi_agent = PrincipalInvestigatorAgent(
        agent_id="pi_main",
        role="Principal Investigator",
        expertise=["research_coordination", "project_management"],
        model_config={'default_model': 'gpt-4'}
    )
    
    scientific_critic = ScientificCriticAgent(
        agent_id="scientific_critic",
        role="Scientific Critic",
        expertise=["quality_assessment", "methodology_review"],
        model_config={'default_model': 'gpt-3.5-turbo'}
    )
    
    agent_marketplace = AgentMarketplace()
    
    # 9. Initialize Virtual Lab system
    virtual_lab_config = {
        'budget_limit': budget_limit,
        'cost_optimization': True,
        'enable_dynamic_tools': True,
        'default_model': 'gpt-3.5-turbo',
        'premium_model': 'gpt-4'
    }
    
    virtual_lab = VirtualLabMeetingSystem(
        pi_agent=pi_agent,
        scientific_critic=scientific_critic,
        agent_marketplace=agent_marketplace,
        config=virtual_lab_config
    )
    
    # 10. Initialize multi-agent framework
    framework_config = {
        'budget_limit': budget_limit,
        'cost_optimization': True,
        'enable_dynamic_tools': True,
        'output_dir': 'output',
        'manuscript_dir': 'manuscripts',
        'visualization_dir': 'visualizations',
        'store_all_interactions': True,
        'max_literature_results': 10,
        'experiment_db_path': 'experiments.db'
    }
    
    framework = MultiAgentResearchFramework(config=framework_config)
    
    system_components = {
        'cost_manager': cost_manager,
        'tool_registry': tool_registry,
        'web_search_tool': web_search_tool,
        'code_execution_tool': code_execution_tool,
        'model_switching_tool': model_switching_tool,
        'custom_chain_builder': custom_chain_builder,
        'llm_client': llm_client,
        'pi_agent': pi_agent,
        'scientific_critic': scientific_critic,
        'agent_marketplace': agent_marketplace,
        'virtual_lab': virtual_lab,
        'framework': framework
    }
    
    logger.info("Complete agent tool system setup successful")
    return system_components


def demonstrate_tool_discovery(system: Dict[str, Any], research_question: str) -> Dict[str, Any]:
    """
    Demonstrate real tool discovery by agents.
    
    Args:
        system: System components
        research_question: Research question to investigate
        
    Returns:
        Tool discovery results
    """
    logger.info("Demonstrating tool discovery...")
    
    pi_agent = system['pi_agent']
    tool_registry = system['tool_registry']
    
    # Agent discovers available tools
    discovered_tools = pi_agent.discover_available_tools(research_question)
    
    logger.info(f"PI Agent discovered {len(discovered_tools)} tools")
    for tool in discovered_tools:
        logger.info(f"  - {tool['name']}: {tool['description']} (confidence: {tool['confidence']:.2f})")
    
    # Optimize tool selection
    optimized_tools = pi_agent.optimize_tool_usage(research_question, discovered_tools)
    
    logger.info(f"Optimized {len(optimized_tools)} tools for research")
    for tool in optimized_tools:
        logger.info(f"  - {tool['name']}: {tool['recommended_usage']}")
    
    return {
        'discovered_tools': discovered_tools,
        'optimized_tools': optimized_tools
    }


def demonstrate_tool_execution(system: Dict[str, Any], research_question: str) -> Dict[str, Any]:
    """
    Demonstrate real tool execution by agents.
    
    Args:
        system: System components
        research_question: Research question to investigate
        
    Returns:
        Tool execution results
    """
    logger.info("Demonstrating tool execution...")
    
    pi_agent = system['pi_agent']
    web_search_tool = system['web_search_tool']
    code_execution_tool = system['code_execution_tool']
    
    # Execute web search
    search_task = {
        'query': research_question,
        'max_results': 5,
        'engines': ['duckduckgo']  # Use free engine for demo
    }
    
    search_result = web_search_tool.execute(search_task, {
        'agent_id': pi_agent.agent_id,
        'agent_role': pi_agent.role,
        'agent_expertise': pi_agent.expertise
    })
    
    logger.info(f"Web search executed: {search_result.get('success', False)}")
    if search_result.get('success', False):
        logger.info(f"Found {len(search_result.get('results', []))} results")
    
    # Execute code analysis
    code_task = {
        'code': '''
import numpy as np
import matplotlib.pyplot as plt

# Simple data analysis
data = np.random.normal(0, 1, 1000)
mean_val = np.mean(data)
std_val = np.std(data)

print(f"Data Analysis Results:")
print(f"Mean: {mean_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")
print(f"Sample Size: {len(data)}")
        ''',
        'language': 'python',
        'timeout': 10
    }
    
    code_result = code_execution_tool.execute(code_task, {
        'agent_id': pi_agent.agent_id,
        'agent_role': pi_agent.role,
        'agent_expertise': pi_agent.expertise
    })
    
    logger.info(f"Code execution: {code_result.get('success', False)}")
    if code_result.get('success', False):
        logger.info(f"Code output: {code_result.get('output', '')[:200]}...")
    
    # Execute with multiple tools
    tools = [web_search_tool, code_execution_tool]
    multi_tool_result = pi_agent.execute_with_tools(research_question, tools)
    
    logger.info(f"Multi-tool execution: {multi_tool_result.get('success', False)}")
    logger.info(f"Tools used: {multi_tool_result.get('metadata', {}).get('tools_used', 0)}")
    
    return {
        'search_result': search_result,
        'code_result': code_result,
        'multi_tool_result': multi_tool_result
    }


def demonstrate_cost_management(system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Demonstrate cost management and budget tracking.
    
    Args:
        system: System components
        
    Returns:
        Cost management results
    """
    logger.info("Demonstrating cost management...")
    
    cost_manager = system['cost_manager']
    
    # Get budget status
    budget_status = cost_manager.get_budget_status()
    
    logger.info(f"Budget Status:")
    logger.info(f"  - Budget Limit: ${budget_status['budget_limit']:.2f}")
    logger.info(f"  - Current Spending: ${budget_status['current_spending']:.4f}")
    logger.info(f"  - Budget Remaining: ${budget_status['budget_remaining']:.2f}")
    logger.info(f"  - Utilization: {budget_status['budget_utilization']:.1%}")
    
    # Get cost analytics
    analytics_24h = cost_manager.get_cost_analytics(24)
    
    logger.info(f"24h Cost Analytics:")
    logger.info(f"  - Total Cost: ${analytics_24h.get('total_cost', 0):.4f}")
    logger.info(f"  - Total Tokens: {analytics_24h.get('total_tokens', 0)}")
    logger.info(f"  - Avg Cost per Token: ${analytics_24h.get('avg_cost_per_token', 0):.6f}")
    
    # Test cost estimation
    estimated_cost = cost_manager.estimate_cost('gpt-4', 1000, 500)
    can_afford = cost_manager.can_afford(estimated_cost)
    
    logger.info(f"Cost Estimation Test:")
    logger.info(f"  - Estimated Cost: ${estimated_cost:.4f}")
    logger.info(f"  - Can Afford: {can_afford}")
    
    # Test model optimization
    optimal_model = cost_manager.optimize_model_selection(
        task_complexity='medium',
        budget_remaining=budget_status['budget_remaining'],
        required_capabilities=['reasoning', 'analysis']
    )
    
    logger.info(f"Model Optimization:")
    logger.info(f"  - Optimal Model: {optimal_model}")
    
    return {
        'budget_status': budget_status,
        'analytics_24h': analytics_24h,
        'estimated_cost': estimated_cost,
        'can_afford': can_afford,
        'optimal_model': optimal_model
    }


def demonstrate_virtual_lab_research(system: Dict[str, Any], research_question: str) -> Dict[str, Any]:
    """
    Demonstrate complete Virtual Lab research session with tool integration.
    
    Args:
        system: System components
        research_question: Research question to investigate
        
    Returns:
        Virtual Lab research results
    """
    logger.info("Demonstrating Virtual Lab research session...")
    
    virtual_lab = system['virtual_lab']
    
    # Conduct research session with constraints
    constraints = {
        'budget_limit': 25.0,  # $25 budget for this session
        'time_limit_hours': 2,
        'max_agents': 3,
        'enable_tool_discovery': True,
        'enable_cost_optimization': True
    }
    
    research_result = virtual_lab.conduct_research_session(
        research_question=research_question,
        constraints=constraints,
        context={'session_type': 'demonstration'}
    )
    
    logger.info(f"Virtual Lab Research Session:")
    logger.info(f"  - Session ID: {research_result.get('session_id', 'unknown')}")
    logger.info(f"  - Status: {research_result.get('status', 'unknown')}")
    logger.info(f"  - Duration: {research_result.get('duration', 0):.2f} seconds")
    
    # Get phase results
    phases = research_result.get('phases', {})
    for phase_name, phase_result in phases.items():
        success = phase_result.get('success', False)
        logger.info(f"  - Phase {phase_name}: {'✓' if success else '✗'}")
    
    return research_result


def demonstrate_custom_tool_building(system: Dict[str, Any], research_question: str) -> Dict[str, Any]:
    """
    Demonstrate custom tool building capabilities.
    
    Args:
        system: System components
        research_question: Research question to investigate
        
    Returns:
        Custom tool building results
    """
    logger.info("Demonstrating custom tool building...")
    
    pi_agent = system['pi_agent']
    custom_chain_builder = system['custom_chain_builder']
    
    # Build custom tool chain
    workflow_steps = [
        {
            'type': 'information_gathering',
            'description': 'Gather information about the research topic',
            'params': {'query': research_question}
        },
        {
            'type': 'analysis',
            'description': 'Analyze gathered information',
            'params': {'analysis_type': 'comprehensive'}
        },
        {
            'type': 'synthesis',
            'description': 'Synthesize findings',
            'params': {'synthesis_type': 'summary'}
        }
    ]
    
    tool_chain_spec = {
        'chain_spec': {
            'research_question': research_question,
            'complexity': 'medium'
        },
        'workflow_steps': workflow_steps
    }
    
    chain_result = custom_chain_builder.execute(tool_chain_spec, {
        'agent_id': pi_agent.agent_id,
        'agent_role': pi_agent.role,
        'agent_expertise': pi_agent.expertise
    })
    
    logger.info(f"Custom Tool Chain Building:")
    logger.info(f"  - Success: {chain_result.get('success', False)}")
    logger.info(f"  - Workflow Steps: {len(workflow_steps)}")
    logger.info(f"  - Execution Time: {chain_result.get('metadata', {}).get('execution_time', 0):.2f}s")
    
    # Get built chains info
    built_chains = custom_chain_builder.get_built_chains()
    logger.info(f"  - Total Built Chains: {built_chains.get('total_chains', 0)}")
    
    return {
        'chain_result': chain_result,
        'built_chains': built_chains
    }


def main():
    """Main demonstration function."""
    logger.info("Starting Complete Agent Tool System Demonstration")
    
    # Setup complete system
    system = setup_complete_system()
    
    # Research question for demonstration
    research_question = "What are the latest developments in machine learning for natural language processing?"
    
    try:
        # 1. Demonstrate tool discovery
        discovery_results = demonstrate_tool_discovery(system, research_question)
        
        # 2. Demonstrate tool execution
        execution_results = demonstrate_tool_execution(system, research_question)
        
        # 3. Demonstrate cost management
        cost_results = demonstrate_cost_management(system)
        
        # 4. Demonstrate custom tool building
        custom_tool_results = demonstrate_custom_tool_building(system, research_question)
        
        # 5. Demonstrate Virtual Lab research session
        virtual_lab_results = demonstrate_virtual_lab_research(system, research_question)
        
        # Compile final results
        final_results = {
            'discovery_results': discovery_results,
            'execution_results': execution_results,
            'cost_results': cost_results,
            'custom_tool_results': custom_tool_results,
            'virtual_lab_results': virtual_lab_results,
            'system_status': 'success'
        }
        
        logger.info("Complete Agent Tool System Demonstration finished successfully")
        
        # Export cost report
        cost_manager = system['cost_manager']
        cost_manager.export_cost_report('cost_report_demo.json')
        logger.info("Cost report exported to cost_report_demo.json")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return {'system_status': 'failed', 'error': str(e)}


if __name__ == "__main__":
    results = main()
    print(f"\nDemonstration completed with status: {results.get('system_status', 'unknown')}") 