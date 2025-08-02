# AI Research Lab Web Interface

A modern, minimalistic web interface for the AI Research Lab Framework, inspired by Apple and ChatGPT design principles.

## Features

### 🎨 Modern Minimalistic Design
- Clean, elegant interface with smooth animations
- Apple/ChatGPT-inspired design language
- Responsive layout for desktop and mobile
- Dark mode support (CSS variables ready)

### 🔬 Research Configuration
- Intuitive research question input
- Domain and priority selection
- Budget and timeline constraints
- Custom constraint management
- Real-time validation

### 🤖 Agent Visualization
- Animated meeting room with agent avatars
- Real-time agent status indicators (thinking, speaking, idle)
- Speech bubbles for agent communications
- Agent performance metrics
- Team efficiency tracking

### 📊 Progress Tracking
- 7-phase research process visualization
- Interactive progress timeline
- Phase-specific details and descriptions
- Real-time progress updates
- Meeting status indicators

### 💬 Activity Log & Human Intervention
- Real-time activity stream
- Human intervention capabilities
- Message export functionality
- Searchable and filterable logs
- System and agent notifications

### ⚙️ Configuration Management
- Secure API key management
- System settings configuration
- Framework parameter tuning
- Auto-save functionality
- Export/import settings

### 📈 Performance Metrics
- Real-time system monitoring
- CPU and memory usage
- Research session statistics
- Agent performance tracking
- Historical data visualization

## Installation

1. **Install Dependencies**
   ```bash
   cd web_ui
   pip install -r requirements.txt
   ```

2. **Set Environment Variables** (optional)
   ```bash
   export SECRET_KEY="your-secret-key-here"
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Interface**
   Open your browser to `http://localhost:5000`

## Architecture

### Backend (Flask + SocketIO)
- **Flask**: Web framework for API endpoints
- **SocketIO**: Real-time WebSocket communication
- **SQLite**: Session and metrics storage
- **Threading**: Background monitoring and research execution

### Frontend (Vanilla JavaScript)
- **Modern ES6+**: Class-based architecture
- **WebSocket**: Real-time updates
- **CSS Grid/Flexbox**: Responsive layouts
- **CSS Animations**: Smooth transitions and effects

### Integration
- **AI Research Framework**: Direct integration with the multi-agent system
- **Virtual Lab**: Meeting-based research coordination
- **Real-time Updates**: Live progress and agent activity monitoring

## API Endpoints

### Configuration
- `GET /api/config` - Get system configuration
- `POST /api/config` - Update system configuration

### Research Control
- `POST /api/research/start` - Start a research session
- `POST /api/research/stop` - Stop current research session
- `GET /api/research/status` - Get current session status

### Session Management
- `GET /api/sessions` - List research sessions
- `GET /api/sessions/<id>` - Get session details

### Metrics & Monitoring
- `GET /api/metrics` - Get system metrics
- `POST /api/intervention` - Send human intervention

## WebSocket Events

### Client → Server
- `join_session` - Join a research session room
- `leave_session` - Leave a research session room

### Server → Client
- `system_status` - System initialization status
- `system_metrics` - Real-time performance metrics
- `research_status` - Research session updates
- `phase_update` - Research phase progress
- `agent_activity` - Agent status and activities
- `activity_log` - New activity log entries

## Configuration

### System Settings
- **Output Directory**: Where results are saved
- **Max Concurrent Agents**: Agent pool limit
- **Auto-save Results**: Automatic result preservation
- **Enable Notifications**: Browser notifications

### API Keys
- **OpenAI API Key**: For GPT models (secure storage)
- **Anthropic API Key**: For Claude models (secure storage)

### Framework Parameters
- **Experiment Database**: SQLite database path
- **Manuscript Directory**: Generated manuscripts location
- **Visualization Directory**: Chart and graph outputs
- **Literature Results**: Maximum papers to retrieve

## Security Features

### API Key Protection
- Password-masked input fields
- Secure server-side storage
- No client-side exposure
- Configuration validation

### Session Management
- Flask session security
- CSRF protection ready
- Secure WebSocket connections
- Rate limiting ready

## Development

### Adding New Features

1. **Backend API Endpoint**
   ```python
   @app.route('/api/new-feature', methods=['POST'])
   def new_feature():
       # Implementation
       return jsonify({'success': True})
   ```

2. **Frontend Integration**
   ```javascript
   async newFeature() {
       const response = await fetch('/api/new-feature', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify(data)
       });
       return response.json();
   }
   ```

3. **Real-time Updates**
   ```python
   # Backend
   socketio.emit('new_event', data, namespace='/')
   
   # Frontend
   this.socket.on('new_event', (data) => {
       this.handleNewEvent(data);
   });
   ```

### Styling Guidelines

Follow the established design system:

```css
/* Use CSS variables for consistency */
:root {
    --color-primary: #007AFF;
    --space-md: 1rem;
    --radius-md: 0.5rem;
    --transition-base: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Follow naming conventions */
.component-name {
    /* Styles */
}

.component-name__element {
    /* BEM methodology */
}

.component-name--modifier {
    /* State variations */
}
```

## Browser Support

- **Chrome**: 88+
- **Firefox**: 85+
- **Safari**: 14+
- **Edge**: 88+

Features used:
- CSS Grid and Flexbox
- CSS Custom Properties
- ES6+ JavaScript
- WebSocket API
- Fetch API

## Performance

### Optimization Features
- Efficient WebSocket communication
- Lazy loading of heavy components
- CSS animations over JavaScript
- Optimized SVG icons
- Minimal bundle size

### Monitoring
- Real-time system metrics
- Memory usage tracking
- Agent performance monitoring
- Session statistics
- Error tracking

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check if Flask server is running
   - Verify port 5000 is available
   - Check firewall settings

2. **API Keys Not Working**
   - Verify keys are correctly entered
   - Check key permissions and quotas
   - Ensure secure storage is enabled

3. **Research Not Starting**
   - Verify research question is provided
   - Check agent availability
   - Review system logs

4. **WebSocket Disconnections**
   - Check network stability
   - Verify browser WebSocket support
   - Review server logs

### Debug Mode

Enable debug mode for development:

```python
# In app.py
socketio.run(app, debug=True, host='0.0.0.0', port=5000)
```

### Logging

Check application logs:

```bash
# System logs
tail -f research_sessions.db.log

# Browser console
# Open Developer Tools → Console
```

## Contributing

1. Follow the established code style
2. Add tests for new features
3. Update documentation
4. Ensure responsive design
5. Test across browsers

## License

Same as the parent AI Research Lab Framework project.