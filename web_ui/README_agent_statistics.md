# Agent Statistics Implementation

## Overview

The agent statistics system has been implemented to provide real-time insights into the multi-agent research framework's performance. Previously, agent statistics were hardcoded to zero values, but now they reflect actual agent activity and performance.

## New Endpoints

### 1. `/api/agent-statistics` (GET)
Returns comprehensive agent statistics including:
- `total_agents`: Total number of agents in the marketplace
- `avg_quality_score`: Average quality score across all agents
- `critical_issues`: Number of critical issues identified by the scientific critic
- `hired_agents`: Number of currently hired agents
- `available_agents`: Number of available agents
- `agent_details`: Detailed information for each agent

### 2. `/api/agent/<agent_id>/performance` (GET)
Returns detailed performance information for a specific agent:
- Agent role and expertise
- Performance metrics (quality score, success rate, total tasks)
- Current task and activity status
- Conversation history
- Hiring status

### 3. `/api/critic/history` (GET)
Returns the scientific critic's critique history:
- Total number of critiques performed
- Critical issues count
- Average quality score from critiques
- Full critique history with timestamps

## Updated Endpoints

### `/api/metrics` (GET)
Now includes real agent statistics in the `agent_stats` field:
- `total_agents`: Real count from agent marketplace
- `avg_quality_score`: Calculated from agent performance metrics
- `critical_issues`: Count from scientific critic's history

## Real-time Updates

Agent statistics are automatically updated via WebSocket events:
- `agent_statistics`: Emitted every 5 seconds with current agent stats
- `system_metrics`: Includes active agent count

## Implementation Details

### Agent Statistics Calculation

The `get_agent_statistics()` function:
1. Retrieves marketplace statistics from `research_framework.agent_marketplace`
2. Calculates average quality score from all agents' performance metrics
3. Counts critical issues from the scientific critic's critique history
4. Handles errors gracefully with fallback to zero values

### Performance Metrics

Each agent tracks:
- `average_quality_score`: Rolling average of task quality scores
- `success_rate`: Percentage of successful tasks
- `total_tasks`: Number of tasks completed
- `last_active`: Timestamp of last activity

### Critical Issues Tracking

The scientific critic maintains:
- `critique_history`: List of all critiques performed
- `critical_issues`: Array of critical issues for each critique
- Quality scores and recommendations for each critique

## Usage Examples

```bash
# Get overall agent statistics
curl http://localhost:5000/api/agent-statistics

# Get specific agent performance
curl http://localhost:5000/api/agent/research_methodology_1/performance

# Get critic history
curl http://localhost:5000/api/critic/history

# Get metrics including agent stats
curl http://localhost:5000/api/metrics
```

## WebSocket Events

Connect to the WebSocket to receive real-time updates:

```javascript
const socket = io();
socket.on('agent_statistics', (stats) => {
    console.log('Agent stats updated:', stats);
});
```

## Error Handling

All endpoints include comprehensive error handling:
- Returns appropriate HTTP status codes
- Logs errors for debugging
- Provides fallback values when framework is unavailable
- Graceful degradation when components are missing

## Future Enhancements

Potential improvements:
- Agent performance trends over time
- Comparative agent rankings
- Detailed task execution analytics
- Agent collaboration metrics
- Performance benchmarking against historical data 