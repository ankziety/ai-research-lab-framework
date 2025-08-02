#!/usr/bin/env python3
"""
AI Research Lab Web Interface Backend

Flask backend that provides API endpoints for the AI Research Lab web interface.
Integrates with the multi-agent research framework and provides real-time updates.
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, render_template, session, g
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import psutil

# Add parent directory to path to import the research framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_research_lab import create_framework
from virtual_lab import ResearchPhase
from multi_agent_framework import MultiAgentResearchFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, 
           template_folder='.',
           static_folder='.',
           static_url_path='')

# Securely set SECRET_KEY
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    # Determine if running in development mode
    flask_env = os.environ.get('FLASK_ENV', '').lower()
    if flask_env in ('development', 'debug'):
        # Generate a random secret key for development
        secret_key = secrets.token_urlsafe(32)
        logging.warning("SECRET_KEY not set, using a random key for development. Do not use this in production!")
    else:
        raise RuntimeError("SECRET_KEY environment variable must be set in production.")
app.config['SECRET_KEY'] = secret_key
app.config['SESSION_TYPE'] = 'filesystem'

# SocketIO setup for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
research_framework: Optional[MultiAgentResearchFramework] = None
current_session: Optional[Dict[str, Any]] = None
system_config: Dict[str, Any] = {}
active_connections: Dict[str, Any] = {}

# Database setup for persistent storage
def init_db():
    """Initialize SQLite database for session storage."""
    conn = sqlite3.connect('research_sessions.db')
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            research_question TEXT,
            config TEXT,
            results TEXT,
            logs TEXT
        )
    ''')
    
    # Create system metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cpu_usage REAL,
            memory_usage REAL,
            active_agents INTEGER,
            session_id TEXT
        )
    ''')
    
    # Create agent performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT,
            session_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            task_completed INTEGER,
            quality_score REAL,
            status TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection."""
    if 'db' not in g:
        g.db = sqlite3.connect('research_sessions.db')
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.teardown_appcontext
def close_db(error):
    _close_db_connection()

# Configuration management
def load_system_config():
    """Load system configuration."""
    global system_config
    config_file = 'config.json'
    
    default_config = {
        'api_keys': {
            'openai': '',
            'anthropic': ''
        },
        'system': {
            'output_dir': 'output',
            'max_concurrent_agents': 8,
            'auto_save_results': True,
            'enable_notifications': True
        },
        'framework': {
            'experiment_db_path': 'experiments/experiments.db',
            'manuscript_dir': 'manuscripts',
            'visualization_dir': 'visualizations',
            'max_literature_results': 10
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                system_config = {**default_config, **loaded_config}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            system_config = default_config
    else:
        system_config = default_config
        save_system_config()

def save_system_config():
    """Save system configuration."""
    try:
        with open('config.json', 'w') as f:
            json.dump(system_config, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def initialize_framework():
    """Initialize the research framework with current config."""
    global research_framework
    try:
        # Prepare framework config
        framework_config = system_config.get('framework', {})
        if system_config.get('api_keys', {}).get('openai'):
            framework_config['openai_api_key'] = system_config['api_keys']['openai']
        if system_config.get('api_keys', {}).get('anthropic'):
            framework_config['anthropic_api_key'] = system_config['api_keys']['anthropic']
        
        research_framework = create_framework(framework_config)
        logger.info("Research framework initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing framework: {e}")
        return False

# System monitoring
def get_system_metrics():
    """Get current system performance metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
            'timestamp': time.time()
        }
        
        if research_framework and hasattr(research_framework, 'get_active_agents'):
            metrics['active_agents'] = len(research_framework.get_active_agents())
        else:
            metrics['active_agents'] = 0
            
        return metrics
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'memory_available': 0,
            'active_agents': 0,
            'timestamp': time.time()
        }

def store_metrics(session_id: Optional[str] = None):
    """Store system metrics in database."""
    try:
        metrics = get_system_metrics()
        db = get_db()
        db.execute('''
            INSERT INTO metrics (cpu_usage, memory_usage, active_agents, session_id)
            VALUES (?, ?, ?, ?)
        ''', (metrics['cpu_usage'], metrics['memory_usage'], 
              metrics['active_agents'], session_id))
        db.commit()
    except Exception as e:
        logger.error(f"Error storing metrics: {e}")

# Background monitoring thread
def monitoring_thread():
    """Background thread for system monitoring."""
    while True:
        try:
            metrics = get_system_metrics()
            # Emit to all connected clients
            socketio.emit('system_metrics', metrics, namespace='/')
            
            # Store in database
            session_id = current_session.get('session_id') if current_session else None
            store_metrics(session_id)
            
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")
            time.sleep(10)

# Routes
@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration (excluding sensitive data)."""
    safe_config = {
        'system': system_config.get('system', {}),
        'framework': {k: v for k, v in system_config.get('framework', {}).items() 
                     if 'key' not in k.lower()},
        'api_keys_configured': {
            'openai': bool(system_config.get('api_keys', {}).get('openai')),
            'anthropic': bool(system_config.get('api_keys', {}).get('anthropic'))
        }
    }
    return jsonify(safe_config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update system configuration."""
    try:
        data = request.get_json()
        
        # Update configuration
        if 'system' in data:
            system_config.setdefault('system', {}).update(data['system'])
        
        if 'framework' in data:
            system_config.setdefault('framework', {}).update(data['framework'])
            
        if 'api_keys' in data:
            system_config.setdefault('api_keys', {}).update(data['api_keys'])
        
        save_system_config()
        
        # Reinitialize framework if needed
        if 'api_keys' in data or 'framework' in data:
            initialize_framework()
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/research/start', methods=['POST'])
def start_research():
    """Start a new research session."""
    global current_session
    
    try:
        if not research_framework:
            initialize_framework()
            if not research_framework:
                return jsonify({'error': 'Research framework not initialized'}), 500
        
        data = request.get_json()
        research_question = data.get('research_question', '')
        
        if not research_question:
            return jsonify({'error': 'Research question is required'}), 400
        
        # Prepare constraints and context
        constraints = {}
        if data.get('budget'):
            constraints['budget'] = data['budget']
        if data.get('timeline'):
            constraints['timeline_weeks'] = data['timeline']
        if data.get('max_agents'):
            constraints['team_size_max'] = data['max_agents']
        
        context = {}
        if data.get('domain'):
            context['domain'] = data['domain']
        if data.get('priority'):
            context['priority'] = data['priority']
        
        # Start research in background thread
        def research_worker():
            try:
                global current_session
                
                # Emit status update
                socketio.emit('research_status', {
                    'status': 'starting',
                    'message': 'Initializing research session...'
                }, namespace='/')
                
                # Conduct research
                results = research_framework.conduct_virtual_lab_research(
                    research_question=research_question,
                    constraints=constraints,
                    context=context
                )
                
                current_session = {
                    'session_id': results.get('session_id'),
                    'status': results.get('status', 'completed'),
                    'research_question': research_question,
                    'results': results,
                    'start_time': time.time()
                }
                
                # Store in database
                db = get_db()
                db.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (id, research_question, config, results, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (current_session['session_id'], research_question,
                      json.dumps({'constraints': constraints, 'context': context}),
                      json.dumps(results), current_session['status']))
                db.commit()
                
                # Emit completion
                socketio.emit('research_complete', current_session, namespace='/')
                
            except Exception as e:
                logger.error(f"Research error: {e}")
                socketio.emit('research_error', {'error': str(e)}, namespace='/')
        
        thread = threading.Thread(target=research_worker)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Research session started',
            'session_id': f'session_{int(time.time())}'
        })
        
    except Exception as e:
        logger.error(f"Error starting research: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/research/stop', methods=['POST'])
def stop_research():
    """Stop the current research session."""
    global current_session
    
    try:
        if current_session:
            current_session['status'] = 'stopped'
            socketio.emit('research_stopped', current_session, namespace='/')
            
        return jsonify({'success': True, 'message': 'Research session stopped'})
    except Exception as e:
        logger.error(f"Error stopping research: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/research/status', methods=['GET'])
def get_research_status():
    """Get current research session status."""
    if current_session:
        return jsonify(current_session)
    else:
        return jsonify({'status': 'idle', 'message': 'No active research session'})

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get list of research sessions."""
    try:
        db = get_db()
        sessions = db.execute('''
            SELECT id, created_at, research_question, status
            FROM sessions
            ORDER BY created_at DESC
            LIMIT 50
        ''').fetchall()
        
        return jsonify([dict(session) for session in sessions])
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get detailed session information."""
    try:
        db = get_db()
        session = db.execute('''
            SELECT * FROM sessions WHERE id = ?
        ''', (session_id,)).fetchone()
        
        if session:
            session_dict = dict(session)
            # Parse JSON fields
            if session_dict.get('config'):
                session_dict['config'] = json.loads(session_dict['config'])
            if session_dict.get('results'):
                session_dict['results'] = json.loads(session_dict['results'])
            if session_dict.get('logs'):
                session_dict['logs'] = json.loads(session_dict['logs'])
            
            return jsonify(session_dict)
        else:
            return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics."""
    try:
        timeframe = request.args.get('timeframe', 'session')
        
        db = get_db()
        
        if timeframe == 'session' and current_session:
            # Get metrics for current session
            session_id = current_session.get('session_id')
            metrics = db.execute('''
                SELECT * FROM metrics 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (session_id,)).fetchall()
        else:
            # Get recent metrics
            hours = {'24h': 24, '7d': 168, '30d': 720}.get(timeframe, 1)
            metrics = db.execute('''
                SELECT * FROM metrics 
                WHERE timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
                LIMIT 1000
            '''.format(hours)).fetchall()
        
        # Get summary statistics
        current_metrics = get_system_metrics()
        
        # Get session statistics
        session_stats = db.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_sessions
            FROM sessions
        ''').fetchone()
        
        return jsonify({
            'current': current_metrics,
            'history': [dict(m) for m in metrics],
            'session_stats': dict(session_stats) if session_stats else {},
            'agent_stats': {
                'total_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intervention', methods=['POST'])
def send_intervention():
    """Send human intervention message."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Add to activity log
        activity = {
            'type': 'human_intervention',
            'author': 'Human',
            'message': message,
            'timestamp': time.time()
        }
        
        # Emit to all clients
        socketio.emit('activity_log', activity, namespace='/')
        
        # If research is active, potentially influence it
        if current_session and research_framework:
            logger.info(f"Human intervention: {message}")
        
        return jsonify({'success': True, 'message': 'Intervention sent'})
        
    except Exception as e:
        logger.error(f"Error sending intervention: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    client_id = request.sid
    active_connections[client_id] = {
        'connected_at': time.time(),
        'last_seen': time.time()
    }
    
    logger.info(f"Client connected: {client_id}")
    
    # Send current status
    emit('system_status', {
        'framework_initialized': research_framework is not None,
        'current_session': current_session,
        'system_metrics': get_system_metrics()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    client_id = request.sid
    if client_id in active_connections:
        del active_connections[client_id]
    
    logger.info(f"Client disconnected: {client_id}")

@socketio.on('join_session')
def handle_join_session(data):
    """Handle client joining a session room."""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_session', {'session_id': session_id})

@socketio.on('leave_session')
def handle_leave_session(data):
    """Handle client leaving a session room."""
    session_id = data.get('session_id')
    if session_id:
        leave_room(session_id)
        emit('left_session', {'session_id': session_id})

# Research progress simulation for demo
def simulate_research_progress():
    """Simulate research progress for demonstration."""
    phases = [
        {'name': 'Team Selection', 'duration': 5},
        {'name': 'Project Specification', 'duration': 8},
        {'name': 'Tools Selection', 'duration': 6},
        {'name': 'Implementation', 'duration': 15},
        {'name': 'Workflow Design', 'duration': 7},
        {'name': 'Execution', 'duration': 20},
        {'name': 'Synthesis', 'duration': 10}
    ]
    
    for i, phase in enumerate(phases):
        time.sleep(phase['duration'])
        
        socketio.emit('phase_update', {
            'phase': i + 1,
            'name': phase['name'],
            'status': 'completed',
            'progress': ((i + 1) / len(phases)) * 100
        }, namespace='/')
        
        # Simulate agent activities
        socketio.emit('agent_activity', {
            'agent_id': f'agent_{i % 3 + 1}',
            'activity': f'Working on {phase["name"]}',
            'status': 'active'
        }, namespace='/')

if __name__ == '__main__':
    # Initialize
    init_db()
    load_system_config()
    initialize_framework()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitoring_thread)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run the app
    logger.info("Starting AI Research Lab Web Interface...")
    # Configure debug and allow_unsafe_werkzeug based on environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    allow_unsafe_werkzeug = os.environ.get('ALLOW_UNSAFE_WERKZEUG', '0') == '1'
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=debug_mode,
        allow_unsafe_werkzeug=allow_unsafe_werkzeug
    )