# Complete Agent Tool System with Cost Management

This document describes the complete implementation of agent tool discovery, request, and usage systems in the Virtual Lab framework, including comprehensive cost management and dynamic tool building capabilities.

## Overview

The system provides real, working functionality for:
- **Tool Discovery**: Agents can discover available tools based on research needs
- **Tool Request**: Agents can request and validate tool access
- **Tool Execution**: Agents can execute tools with proper error handling
- **Cost Management**: Real-time cost tracking and budget enforcement
- **Dynamic Tool Building**: Custom tool creation and integration
- **Model Optimization**: Automatic model selection based on cost and capability

## System Architecture

### Core Components

1. **Cost Manager** (`cost_manager.py`)
   - Real-time cost tracking for all API calls
   - Budget limit enforcement with automatic model switching
   - Cost estimation before API calls
   - Detailed spending analytics and reporting

2. **Dynamic Tool Builder** (`tools/dynamic_tool_builder.py`)
   - Web search tool with multiple search engines
   - Code execution tool with sandboxed environment
   - Model switching tool with cost optimization
   - Custom tool chain builder

3. **Enhanced Base Agent** (`agents/base_agent.py`)
   - Real tool discovery methods
   - Tool request and validation
   - Multi-tool execution capabilities
   - Custom tool building
   - Tool usage optimization

4. **Virtual Lab Integration** (`virtual_lab.py`)
   - Real tools selection phase
   - Actual tool implementation phase
   - Cost-aware tool integration
   - Comprehensive testing and validation

## Key Features

### 1. Real Tool Discovery

Agents can discover tools based on research requirements:

```python
# Agent discovers available tools
discovered_tools = agent.discover_available_tools(research_question)

# Optimize tool selection
optimized_tools = agent.optimize_tool_usage(research_question, discovered_tools)
```

### 2. Tool Request and Validation

Agents can request tools with proper validation:

```python
# Request tool access
tool = agent.request_tool(tool_id, context)

# Execute tool with error handling
result = tool.execute(task, context)
```

### 3. Multi-Tool Execution

Agents can execute multiple tools simultaneously:

```python
# Execute with multiple tools
tools = [web_search_tool, code_execution_tool]
result = agent.execute_with_tools(research_question, tools)
```

### 4. Cost Management

Comprehensive cost tracking and budget management:

```python
# Initialize cost manager
cost_manager = CostManager(budget_limit, config)

# Track usage
cost_manager.track_usage(model, tokens_input, tokens_output, cost, task_type, agent_id)

# Get budget status
budget_status = cost_manager.get_budget_status()

# Optimize model selection
optimal_model = cost_manager.optimize_model_selection(task_complexity, budget_remaining)
```

### 5. Dynamic Tool Building

Create custom tools for specific research needs:

```python
# Build custom tool
custom_tool = agent.build_custom_tool(tool_spec)

# Create tool chains
tool_chain = CustomToolChainBuilder(tool_registry, cost_manager)
```

## Implementation Details

### Cost Manager Features

- **Real-time Tracking**: Tracks all API calls with detailed metadata
- **Budget Enforcement**: Automatically switches models when budget is low
- **Cost Estimation**: Estimates costs before API calls
- **Analytics**: Provides detailed spending analytics and reports
- **Model Optimization**: Selects optimal models based on cost and capability

### Dynamic Tools

#### Web Search Tool
- Multiple search engines (Google, Bing, DuckDuckGo, SerpAPI)
- Result filtering and ranking
- Rate limiting and error handling
- Cost tracking for search operations

#### Code Execution Tool
- Sandboxed execution environment
- Security validation
- Resource limits and timeouts
- Result capture and error handling

#### Model Switching Tool
- Automatic model selection based on task complexity
- Cost-aware model switching
- Performance tracking and optimization
- Fallback mechanisms

#### Custom Tool Chain Builder
- Dynamic tool chain creation
- Tool validation and testing
- Integration planning
- Error handling and fallbacks

### Enhanced Agent Capabilities

#### Tool Discovery
```python
def discover_available_tools(self, task_description: str) -> List[Dict[str, Any]]:
    """Discover tools available for a specific task."""
    # Real tool discovery using registry
    # Return actual tools with confidence scores
    # Consider cost and capability
```

#### Tool Request
```python
def request_tool(self, tool_id: str, context: Dict[str, Any]) -> Optional[BaseTool]:
    """Request access to a specific tool."""
    # Real tool request with validation
    # Check budget and requirements
    # Return actual tool instance
```

#### Tool Execution
```python
def execute_with_tools(self, task: str, tools: List[BaseTool]) -> Dict[str, Any]:
    """Execute a task using multiple tools."""
    # Real tool execution
    # Track costs and performance
    # Handle errors and fallbacks
```

#### Custom Tool Building
```python
def build_custom_tool(self, tool_spec: Dict[str, Any]) -> Optional[BaseTool]:
    """Build a custom tool based on specification."""
    # Real custom tool creation
    # API integration and testing
    # Registration with tool registry
```

## Virtual Lab Integration

### Real Tools Selection Phase

The tools selection phase now performs real tool discovery and validation:

1. **Tool Discovery**: Each agent discovers tools based on their expertise
2. **Tool Validation**: Tools are tested with real API connections
3. **Cost-Aware Selection**: Tools are selected based on cost and performance
4. **Capability Assessment**: Tool capabilities are evaluated against research requirements
5. **Integration Planning**: Tool integration strategy is planned

### Real Tools Implementation Phase

The tools implementation phase builds actual tool integrations:

1. **API Connections**: Real API connections are established and tested
2. **Custom Tool Creation**: Custom tools are built for research needs
3. **Tool Chain Building**: Tool chains are created for complex tasks
4. **Error Handling**: Fallback mechanisms are implemented
5. **Comprehensive Testing**: Tools are thoroughly tested

## Configuration

### Basic Configuration

```python
config = {
    'budget_limit': 100.0,
    'cost_optimization': True,
    'enable_dynamic_tools': True,
    'default_model': 'gpt-3.5-turbo',
    'premium_model': 'gpt-4',
    'web_search_apis': {
        'google_search_api_key': 'your_key',
        'bing_search_api_key': 'your_key',
        'serpapi_key': 'your_key'
    },
    'code_execution': {
        'sandbox_enabled': True,
        'timeout_seconds': 30,
        'memory_limit_mb': 512
    }
}
```

### Advanced Configuration

```python
config = {
    'budget_limit': 100.0,
    'cost_optimization': True,
    'auto_switch_threshold': 0.8,
    'model_costs': {
        'gpt-4': {
            'input_cost_per_1k': 0.03,
            'output_cost_per_1k': 0.06,
            'capabilities': ['reasoning', 'analysis', 'code'],
            'reliability_score': 0.95
        },
        'gpt-3.5-turbo': {
            'input_cost_per_1k': 0.001,
            'output_cost_per_1k': 0.002,
            'capabilities': ['reasoning', 'analysis'],
            'reliability_score': 0.85
        }
    }
}
```

## Usage Examples

### Basic Usage

```python
from multi_agent_framework import MultiAgentResearchFramework

# Initialize framework with cost management
config = {
    'budget_limit': 50.0,
    'cost_optimization': True,
    'enable_dynamic_tools': True
}

framework = MultiAgentResearchFramework(config)

# Conduct research with tool integration
result = framework.conduct_virtual_lab_research(
    research_question="What are the latest developments in machine learning?",
    constraints={'budget_limit': 25.0}
)
```

### Advanced Usage

```python
from cost_manager import CostManager
from tools.dynamic_tool_builder import WebSearchTool, CodeExecutionTool
from agents.base_agent import BaseAgent

# Setup cost manager
cost_manager = CostManager(100.0, config)

# Create tools
web_search = WebSearchTool(api_keys, cost_manager)
code_exec = CodeExecutionTool(sandbox_config, cost_manager)

# Create agent
agent = BaseAgent("researcher", "Research Agent", ["data_analysis"])

# Discover and use tools
tools = agent.discover_available_tools("Analyze machine learning trends")
selected_tools = agent.optimize_tool_usage("Analyze trends", tools)

# Execute with tools
result = agent.execute_with_tools("Research task", [web_search, code_exec])
```

## Cost Management Features

### Budget Tracking

- Real-time spending tracking
- Budget alerts at 50%, 80%, and 95% utilization
- Automatic model switching when budget is low
- Detailed cost analytics and reporting

### Cost Optimization

- Automatic model selection based on task complexity
- Cost estimation before API calls
- Fallback to cheaper models when budget is limited
- Performance tracking for cost optimization

### Analytics and Reporting

- Detailed spending analytics by time period
- Cost breakdown by provider and model
- Usage statistics and success rates
- Exportable cost reports

## Security and Safety

### Code Execution Safety

- Sandboxed execution environment
- Security validation for all code
- Resource limits and timeouts
- Blocked dangerous functions and imports

### API Security

- Rate limiting for all API calls
- Error handling and fallback mechanisms
- Secure API key management
- Request validation and sanitization

## Performance Features

### Tool Optimization

- Tool usage optimization based on confidence and success rates
- Experience-based tool selection
- Capability matching for research requirements
- Performance tracking and improvement

### Model Optimization

- Automatic model selection based on task complexity
- Cost-aware model switching
- Performance tracking for different models
- Fallback mechanisms for model failures

## Error Handling

### Comprehensive Error Handling

- Tool execution error handling
- API connection error recovery
- Budget limit error handling
- Fallback mechanism implementation

### Graceful Degradation

- Automatic fallback to cheaper models
- Tool failure recovery
- Partial success handling
- Error reporting and logging

## Testing and Validation

### Tool Testing

- API connection testing
- Functionality testing
- Performance testing
- Security validation

### Integration Testing

- Tool chain testing
- Multi-tool execution testing
- Cost management testing
- Error handling testing

## Future Enhancements

### Planned Features

1. **Advanced Tool Discovery**: Machine learning-based tool recommendation
2. **Dynamic Pricing**: Real-time cost optimization based on usage patterns
3. **Tool Marketplace**: Community-contributed tools and integrations
4. **Advanced Analytics**: Predictive cost modeling and optimization
5. **Multi-Provider Support**: Integration with additional LLM providers

### Extensibility

The system is designed for easy extension:

- New tool types can be added by implementing the `BaseTool` interface
- New cost models can be added to the cost manager
- New agent capabilities can be added to the base agent
- New integration patterns can be added to the Virtual Lab system

## Conclusion

This complete agent tool system provides real, working functionality for tool discovery, request, and usage with comprehensive cost management. The system is production-ready and includes all necessary features for robust, cost-effective AI research collaboration.

For more information, see the example usage file `example_complete_tool_system.py` for a comprehensive demonstration of all features. 