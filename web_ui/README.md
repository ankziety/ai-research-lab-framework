# AI Research Lab Web UI

Modern web interfaces for the AI Research Lab framework, providing real-time monitoring and control of multi-agent research sessions.

## Interface Options

### 🚀 **Gradio Interface (Recommended)**

The Gradio interface provides a modern, easy-to-setup web experience with better real-time updates and simplified configuration.

**Simple Gradio Interface (Recommended for most users):**
- Streamlined chat interface with research capabilities
- Agent statistics and research status
- Minimal setup required
- Perfect for quick research sessions

**Full-featured Gradio Interface (Advanced users):**
- Comprehensive research dashboard
- Agent management panel
- Advanced settings and configuration
- Real-time monitoring and analytics
- Session management and history

### ⚠️ **Flask Interface (Deprecated)**

> **Warning:** The Flask interface is deprecated and will be removed in a future version. It requires complex setup, environment variables, and provides a less modern user experience compared to the Gradio interface.

The Flask interface is maintained for legacy users but is not recommended for new deployments.

## Setup

### Gradio Interface Setup (Recommended)

**Simple Gradio Interface:**
```bash
cd web_ui
python simple_gradio_app.py
```

**Full-featured Gradio Interface:**
```bash
cd web_ui
python gradio_app.py
```

Both Gradio interfaces will automatically:
- Start a local web server
- Open your browser to the interface
- Handle configuration and setup automatically
- Provide real-time updates without additional configuration

### Flask Interface Setup (Deprecated)

> **Note:** The Flask interface is deprecated. Use the Gradio interface instead for a better experience.

**Development Mode:**

For development, the application will automatically generate a SECRET_KEY:

```bash
cd web_ui
python app.py
```

**Production Mode:**

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

### Gradio Interface
No environment variables required - configuration is handled through the web interface.

### Flask Interface (Deprecated)
- `SECRET_KEY`: Required for production. Generate using `generate_secret_key.py`
- `FLASK_ENV`: Set to 'development' for development mode
- `FLASK_DEBUG`: Set to '1' for debug mode

## Troubleshooting

### Gradio Interface

**Interface not starting:**
- Ensure you have Gradio installed: `pip install gradio`
- Check that all dependencies are installed: `pip install -r requirements.txt`

**Configuration issues:**
- Use the settings panel in the full-featured interface to configure API keys
- The simple interface uses default configuration

### Flask Interface (Deprecated)

**SECRET_KEY Error:**
If you see "SECRET_KEY environment variable must be set in production":
1. Run `python generate_secret_key.py` to generate a key
2. Set the environment variable: `export SECRET_KEY='your-key'`
3. Restart the application

**Database Connection Errors:**
The application automatically handles database connection cleanup. If you see connection errors, restart the application.

**Active Agents Always Showing 2:**
This is expected behavior - the system always has 2 core agents (Principal Investigator and Scientific Critic) present. The count will increase when additional agents are hired for research tasks.

## Features

### Gradio Interface
- **Simple Interface:**
  - Streamlined chat interface with research capabilities
  - Real-time agent statistics
  - Research mode toggle
  - Minimal configuration required

- **Full-featured Interface:**
  - Comprehensive research dashboard
  - Real-time system metrics monitoring
  - Multi-agent research session management
  - Agent management panel
  - Advanced settings and configuration
  - Session persistence and history
  - Agent activity tracking
  - Research progress visualization
  - Live updates without WebSocket complexity

### Flask Interface (Deprecated)
- Real-time system metrics monitoring
- Multi-agent research session management
- WebSocket-based live updates
- Session persistence and history
- Agent activity tracking
- Research progress visualization

## API Endpoints

### Gradio Interface
The Gradio interface provides a modern web interface without requiring direct API access. All functionality is available through the web interface.

### Flask Interface (Deprecated)
- `GET /api/config` - Get system configuration
- `POST /api/config` - Update system configuration
- `POST /api/research/start` - Start a research session
- `POST /api/research/stop` - Stop current research
- `GET /api/metrics` - Get system metrics
- `GET /api/sessions` - Get session history

## WebSocket Events

### Flask Interface (Deprecated)
- `system_metrics` - Real-time system performance data
- `agent_activity` - Agent status and activity updates
- `research_progress` - Research session progress updates

### Gradio Interface
Gradio provides built-in real-time updates without requiring WebSocket configuration.

## Migration Guide: Flask to Gradio

If you're currently using the Flask interface, here's how to migrate to the recommended Gradio interface:

### Why Migrate?
- **Simpler Setup**: No environment variables or complex configuration required
- **Better UX**: Modern interface with improved real-time updates
- **Easier Maintenance**: Less complex codebase and dependencies
- **Future-Proof**: Flask interface will be removed in future versions

### Migration Steps

1. **Stop the Flask interface** if it's running
2. **Choose your Gradio interface:**
   - **Simple**: `python simple_gradio_app.py` (recommended for most users)
   - **Full-featured**: `python gradio_app.py` (for advanced users)
3. **Configure API keys** through the web interface (if using full-featured)
4. **Test your research workflows** - all functionality is preserved

### Feature Comparison

| Feature | Flask (Deprecated) | Gradio Simple | Gradio Full |
|---------|-------------------|---------------|-------------|
| Setup Complexity | High (env vars, SECRET_KEY) | Low (one command) | Low (one command) |
| Real-time Updates | WebSocket (complex) | Built-in | Built-in |
| Configuration | Manual env vars | Default config | Web interface |
| Agent Management | API endpoints | Basic stats | Full panel |
| Research Dashboard | Limited | Basic | Comprehensive |
| Session History | API access | Basic | Full access |