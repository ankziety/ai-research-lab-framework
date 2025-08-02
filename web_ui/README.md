# AI Research Lab Web UI

A Flask-based web interface for the AI Research Lab framework, providing real-time monitoring and control of multi-agent research sessions.

## Setup

### Development Mode

For development, the application will automatically generate a SECRET_KEY:

```bash
cd web_ui
python app.py
```

### Production Mode

For production deployment, you must set the SECRET_KEY environment variable:

1. Generate a secure secret key:
   ```bash
   python generate_secret_key.py
   ```

2. Set the environment variable:
   ```bash
   export SECRET_KEY='your-generated-secret-key'
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Environment Variables

- `SECRET_KEY`: Required for production. Generate using `generate_secret_key.py`
- `FLASK_ENV`: Set to 'development' for development mode
- `FLASK_DEBUG`: Set to '1' for debug mode

## Troubleshooting

### SECRET_KEY Error
If you see "SECRET_KEY environment variable must be set in production":
1. Run `python generate_secret_key.py` to generate a key
2. Set the environment variable: `export SECRET_KEY='your-key'`
3. Restart the application

### Database Connection Errors
The application automatically handles database connection cleanup. If you see connection errors, restart the application.

### Active Agents Always Showing 2
This is expected behavior - the system always has 2 core agents (Principal Investigator and Scientific Critic) present. The count will increase when additional agents are hired for research tasks.

## Features

- Real-time system metrics monitoring
- Multi-agent research session management
- WebSocket-based live updates
- Session persistence and history
- Agent activity tracking
- Research progress visualization

## API Endpoints

- `GET /api/config` - Get system configuration
- `POST /api/config` - Update system configuration
- `POST /api/research/start` - Start a research session
- `POST /api/research/stop` - Stop current research
- `GET /api/metrics` - Get system metrics
- `GET /api/sessions` - Get session history

## WebSocket Events

- `system_metrics` - Real-time system performance data
- `agent_activity` - Agent status and activity updates
- `research_progress` - Research session progress updates