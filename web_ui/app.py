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
import secrets
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, render_template, session, g
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
import sqlite3
import psutil

# Add parent directory to path to import the research framework
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from ai_research_lab import create_framework
from virtual_lab import ResearchPhase, MeetingRecord, MeetingAgenda
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
    # Check if we're in development mode
    flask_env = os.environ.get('FLASK_ENV', '').lower()
    flask_debug = os.environ.get('FLASK_DEBUG', '').lower()
    is_development = (
        flask_env in ('development', 'debug') or 
        flask_debug in ('1', 'true', 'yes') or
        os.environ.get('FLASK_APP') is not None or
        __name__ == '__main__'
    )
    
    if is_development:
        # Generate a random secret key for development
        secret_key = secrets.token_urlsafe(32)
        logger.warning("SECRET_KEY not set, using a random key for development. Set SECRET_KEY environment variable for production.")
    else:
        # For production, require SECRET_KEY to be set
        logger.error("SECRET_KEY environment variable must be set in production.")
        logger.error("Set it with: export SECRET_KEY='your-secret-key-here'")
        logger.error("Or generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'")
        raise RuntimeError("SECRET_KEY environment variable must be set in production.")

app.config['SECRET_KEY'] = secret_key
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Exempt API routes from CSRF protection
def exempt_api_routes():
    """Exempt API routes from CSRF protection."""
    api_routes = [
        'start_research', 'stop_research', 'get_research_status',
        'get_config', 'update_config', 'test_api_key',
        'get_sessions', 'get_session', 'get_metrics',
        'send_intervention', 'get_agent_activity', 'get_chat_logs',
        'add_chat_log', 'add_agent_activity', 'get_meetings',
        'add_meeting', 'get_agents'
    ]
    
    for route_name in api_routes:
        if route_name in app.view_functions:
            csrf.exempt(app.view_functions[route_name])

# SocketIO setup for real-time communication
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    logger=True,
    engineio_logger=True
)

# Global state
research_framework: Optional[MultiAgentResearchFramework] = None
current_session: Optional[Dict[str, Any]] = None
system_config: Dict[str, Any] = {}
active_connections: Dict[str, Any] = {}

# Thread-local storage for database connections
_thread_local = threading.local()

def get_db_connection():
    """Get a database connection."""
    try:
        conn = sqlite3.connect('research_sessions.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error creating database connection: {e}")
        return None

def close_db_connections():
    """Close all database connections."""
    # No longer needed with new approach
    pass

# Database setup for persistent storage
def init_db():
    """Initialize SQLite database for session storage."""
    conn = get_db_connection()
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
    
    # Create chat logs table for thoughts, choices, and communications
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            log_type TEXT CHECK(log_type IN ('thought', 'choice', 'communication', 'tool_call', 'system')),
            author TEXT,
            message TEXT,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    ''')
    
    # Create agent activity table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            agent_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            activity_type TEXT CHECK(activity_type IN ('thinking', 'speaking', 'tool_use', 'meeting', 'idle')),
            message TEXT,
            status TEXT,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    ''')
    
    # Create meetings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            meeting_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            participants TEXT,
            topic TEXT,
            duration INTEGER,
            outcome TEXT,
            transcript TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection with proper error handling."""
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        return conn
    except Exception as e:
        logger.error(f"Error getting database: {e}")
        return None

def close_db_connection():
    """Close a database connection."""
    # Connections are now managed individually
    pass

@app.teardown_appcontext
def close_db(error):
    # Connections are managed by the pool
    pass

@app.teardown_appcontext
def cleanup_db(error):
    """Cleanup database connections on app shutdown."""
    close_db_connections()

# Configuration management
def load_system_config():
    """Load system configuration."""
    global system_config
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    
    default_config = {
        'api_keys': {
            'openai': '',
            'anthropic': '',
            'gemini': '',
            'huggingface': '',
            'ollama_endpoint': 'http://localhost:11434'
        },
        'search_api_keys': {
            'google_search': '',
            'google_search_engine_id': '',
            'serpapi': '',
            'semantic_scholar': '',
            'openalex_email': '',
            'core': ''
        },
        'system': {
            'output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output'),
            'max_concurrent_agents': 8,
            'auto_save_results': True,
            'enable_notifications': True
        },
        'framework': {
            'experiment_db_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments', 'experiments.db'),
            'manuscript_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'manuscripts'),
            'visualization_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations'),
            'max_literature_results': 10,
            'default_llm_provider': 'openai',
            'default_model': 'gpt-4',
            'enable_free_search': True,
            'enable_mock_responses': True
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Deep merge to preserve nested structure
                system_config = _deep_merge(default_config, loaded_config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            system_config = default_config
    else:
        system_config = default_config
        save_system_config()

def _deep_merge(base_dict, update_dict):
    """Deep merge two dictionaries."""
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def save_system_config():
    """Save system configuration."""
    try:
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_file, 'w') as f:
            json.dump(system_config, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def initialize_framework():
    """Initialize the research framework with current config."""
    global research_framework
    try:
        # Prepare framework config
        framework_config = system_config.get('framework', {})
        
        # Add API keys to framework config
        api_keys = system_config.get('api_keys', {})
        for key_name, key_value in api_keys.items():
            if key_value:  # Only add non-empty keys
                # Map web UI API key names to framework expected names
                if key_name == 'openai':
                    framework_config['openai_api_key'] = key_value
                elif key_name == 'anthropic':
                    framework_config['anthropic_api_key'] = key_value
                elif key_name == 'gemini':
                    framework_config['gemini_api_key'] = key_value
                elif key_name == 'huggingface':
                    framework_config['huggingface_api_key'] = key_value
                elif key_name == 'ollama_endpoint':
                    framework_config['ollama_endpoint'] = key_value
                else:
                    framework_config[key_name] = key_value
        
        # Add search API keys if configured
        search_api_keys = system_config.get('search_api_keys', {})
        for key_name, key_value in search_api_keys.items():
            if key_value:
                framework_config[f'{key_name}_api_key'] = key_value
        
        # Also add the API keys directly to the config for backward compatibility
        framework_config.update(api_keys)
        
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
        
        # Get active agents count from framework
        active_agents_count = 0
        if research_framework:
            if hasattr(research_framework, 'get_active_agents'):
                active_agents = research_framework.get_active_agents()
                # Only count agents that are actively working, not just present
                active_agents_count = len([agent for agent in active_agents 
                                        if hasattr(agent, 'is_active') and agent.is_active()])
            elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                active_agents = research_framework.virtual_lab.get_active_agents()
                # Only count agents that are actively working, not just present
                active_agents_count = len([agent for agent in active_agents 
                                        if hasattr(agent, 'is_active') and agent.is_active()])
            else:
                active_agents_count = 0
        else:
            active_agents_count = 0
            
        metrics['active_agents'] = active_agents_count
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

def get_agent_stats(db, current_session):
    """Get comprehensive agent statistics."""
    try:
        if db is None:
            return {
                'total_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0,
                'active_agents': 0
            }
        
        # Get session-specific stats if there's an active session
        session_id = current_session.get('session_id') if current_session else None
        
        # Count total agents from agent_activity table
        total_agents_query = '''
            SELECT COUNT(DISTINCT agent_id) as total_agents 
            FROM agent_activity 
        '''
        if session_id:
            total_agents_query += ' WHERE session_id = ?'
            total_agents_result = db.execute(total_agents_query, (session_id,)).fetchone()
        else:
            total_agents_result = db.execute(total_agents_query).fetchone()
        
        total_agents = total_agents_result['total_agents'] if total_agents_result else 0
        
        # Calculate average quality score based on agent performance
        quality_query = '''
            SELECT AVG(quality_score) as avg_quality 
            FROM agent_performance 
        '''
        if session_id:
            quality_query += ' WHERE session_id = ?'
            quality_result = db.execute(quality_query, (session_id,)).fetchone()
        else:
            quality_result = db.execute(quality_query).fetchone()
        
        avg_quality = quality_result['avg_quality'] if quality_result and quality_result['avg_quality'] else 0.0
        
        # Count critical issues (failed tasks or low quality scores)
        issues_query = '''
            SELECT COUNT(*) as critical_issues 
            FROM agent_performance 
            WHERE quality_score < 0.5 OR task_completed = 0
        '''
        if session_id:
            issues_query += ' AND session_id = ?'
            issues_result = db.execute(issues_query, (session_id,)).fetchone()
        else:
            issues_result = db.execute(issues_query).fetchone()
        
        critical_issues = issues_result['critical_issues'] if issues_result else 0
        
        # Get active agents from current framework
        active_agents_count = 0
        if research_framework:
            if hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                try:
                    # Try to get current agents from virtual lab
                    current_agents = getattr(research_framework.virtual_lab, 'agents', {})
                    active_agents_count = len(current_agents)
                except:
                    active_agents_count = 0
        
        return {
            'total_agents': max(total_agents, active_agents_count),  # Use the higher count
            'avg_quality_score': round(avg_quality, 2),
            'critical_issues': critical_issues,
            'active_agents': active_agents_count
        }
        
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        return {
            'total_agents': 0,
            'avg_quality_score': 0.0,
            'critical_issues': 0,
            'active_agents': 0
        }

def store_metrics(session_id: Optional[str] = None):
    """Store system metrics in database."""
    try:
        with app.app_context():
            metrics = get_system_metrics()
            db = get_db()
            if db is None:
                logger.warning("Could not get database connection for storing metrics")
                return
                
            try:
                db.execute('''
                    INSERT INTO metrics (cpu_usage, memory_usage, active_agents, session_id)
                    VALUES (?, ?, ?, ?)
                ''', (metrics['cpu_usage'], metrics['memory_usage'], 
                      metrics['active_agents'], session_id))
                db.commit()
            finally:
                db.close()
    except Exception as e:
        logger.error(f"Error storing metrics: {e}")

def store_agent_performance(agent_id: str, session_id: str, task_completed: int, quality_score: float, status: str):
    """Store agent performance data in database."""
    try:
        with app.app_context():
            db = get_db()
            if db is None:
                logger.warning("Could not get database connection for storing agent performance")
                return
                
            try:
                db.execute('''
                    INSERT INTO agent_performance (agent_id, session_id, task_completed, quality_score, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (agent_id, session_id, task_completed, quality_score, status))
                db.commit()
            finally:
                db.close()
    except Exception as e:
        logger.error(f"Error storing agent performance: {e}")

# Background monitoring thread
def monitoring_thread():
    """Background thread for system monitoring."""
    while True:
        try:
            metrics = get_system_metrics()
            # Emit to all connected clients
            socketio.emit('system_metrics', metrics, namespace='/')
            
            # Store in database with app context
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

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors gracefully."""
    # Check if it's a static file request
    if request.path.startswith('/static/') or request.path.endswith('.map'):
        # Log missing static files at debug level only
        logger.debug(f"Missing static file: {request.path}")
        return '', 404
    
    # For other 404s, return JSON error
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration (excluding sensitive data)."""
    api_keys = system_config.get('api_keys', {})
    search_api_keys = system_config.get('search_api_keys', {})
    safe_config = {
        'system': system_config.get('system', {}),
        'framework': {k: v for k, v in system_config.get('framework', {}).items() 
                     if 'key' not in k.lower()},
        'api_keys_configured': {
            'openai': bool(api_keys.get('openai')),
            'anthropic': bool(api_keys.get('anthropic')),
            'gemini': bool(api_keys.get('gemini')),
            'huggingface': bool(api_keys.get('huggingface')),
            'ollama': bool(api_keys.get('ollama_endpoint'))
        },
        'search_api_keys_configured': {
            'google_search': bool(search_api_keys.get('google_search')),
            'google_search_engine_id': bool(search_api_keys.get('google_search_engine_id')),
            'serpapi': bool(search_api_keys.get('serpapi')),
            'semantic_scholar': bool(search_api_keys.get('semantic_scholar')),
            'openalex': bool(search_api_keys.get('openalex_email')),
            'core': bool(search_api_keys.get('core'))
        },
        'available_providers': [
            'openai' if api_keys.get('openai') else None,
            'anthropic' if api_keys.get('anthropic') else None,
            'gemini' if api_keys.get('gemini') else None,
            'huggingface' if api_keys.get('huggingface') else None,
            'ollama' if api_keys.get('ollama_endpoint') else None
        ],
        'free_options': {
            'enable_free_search': system_config.get('framework', {}).get('enable_free_search', True),
            'enable_mock_responses': system_config.get('framework', {}).get('enable_mock_responses', True)
        }
    }
    # Remove None values from available_providers
    safe_config['available_providers'] = [p for p in safe_config['available_providers'] if p]
    
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
            
        if 'search_api_keys' in data:
            system_config.setdefault('search_api_keys', {}).update(data['search_api_keys'])
        
        save_system_config()
        
        # Reinitialize framework if needed
        if 'api_keys' in data or 'search_api_keys' in data or 'framework' in data:
            initialize_framework()
        
        # Return updated status information
        api_keys = system_config.get('api_keys', {})
        search_api_keys = system_config.get('search_api_keys', {})
        response_data = {
            'success': True, 
            'message': 'Configuration updated',
            'api_keys_configured': {
                'openai': bool(api_keys.get('openai')),
                'anthropic': bool(api_keys.get('anthropic')),
                'gemini': bool(api_keys.get('gemini')),
                'huggingface': bool(api_keys.get('huggingface')),
                'ollama': bool(api_keys.get('ollama_endpoint'))
            },
            'search_api_keys_configured': {
                'google_search': bool(search_api_keys.get('google_search')),
                'google_search_engine_id': bool(search_api_keys.get('google_search_engine_id')),
                'serpapi': bool(search_api_keys.get('serpapi')),
                'semantic_scholar': bool(search_api_keys.get('semantic_scholar')),
                'openalex': bool(search_api_keys.get('openalex_email')),
                'core': bool(search_api_keys.get('core'))
            },
            'available_providers': [
                'openai' if api_keys.get('openai') else None,
                'anthropic' if api_keys.get('anthropic') else None,
                'gemini' if api_keys.get('gemini') else None,
                'huggingface' if api_keys.get('huggingface') else None,
                'ollama' if api_keys.get('ollama_endpoint') else None
            ],
            'free_options': {
                'enable_free_search': system_config.get('framework', {}).get('enable_free_search', True),
                'enable_mock_responses': system_config.get('framework', {}).get('enable_mock_responses', True)
            }
        }
        # Remove None values from available_providers
        response_data['available_providers'] = [p for p in response_data['available_providers'] if p]
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/test', methods=['POST'])
def test_api_key():
    """Test API key connectivity."""
    try:
        data = request.get_json()
        provider = data.get('provider')
        api_key = data.get('api_key')
        
        if not provider or not api_key:
            return jsonify({'error': 'Provider and API key are required'}), 400
        
        # Simple validation based on provider
        validation_rules = {
            'openai': lambda key: key.startswith('sk-'),
            'anthropic': lambda key: key.startswith('sk-ant-'),
            'gemini': lambda key: key.startswith('AIza'),
            'huggingface': lambda key: key.startswith('hf_'),
            'ollama': lambda key: key.startswith('http')
        }
        
        if provider in validation_rules:
            if not validation_rules[provider](api_key):
                return jsonify({
                    'valid': False,
                    'message': f'Invalid {provider} API key format'
                })
        
        # For now, just validate format. In a real implementation,
        # you would make actual API calls to test connectivity
        return jsonify({
            'valid': True,
            'message': f'{provider} API key format is valid'
        })
        
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
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
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        research_question = data.get('research_question', '').strip()
        
        # Input validation and sanitization
        if not research_question:
            return jsonify({'error': 'Research question is required'}), 400
            
        if len(research_question) > 1000:
            return jsonify({'error': 'Research question too long (max 1000 characters)'}), 400
            
        # Sanitize inputs
        import html
        research_question = html.escape(research_question)
        
        # Validate constraints
        constraints = {}
        if data.get('budget'):
            try:
                budget = float(data['budget'])
                if budget < 0 or budget > 1000000:
                    return jsonify({'error': 'Invalid budget value'}), 400
                constraints['budget'] = budget
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid budget format'}), 400
                
        if data.get('timeline'):
            try:
                timeline = int(data['timeline'])
                if timeline < 1 or timeline > 52:
                    return jsonify({'error': 'Invalid timeline (1-52 weeks)'}), 400
                constraints['timeline_weeks'] = timeline
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid timeline format'}), 400
                
        if data.get('max_agents'):
            try:
                max_agents = int(data['max_agents'])
                if max_agents < 1 or max_agents > 20:
                    return jsonify({'error': 'Invalid max agents (1-20)'}), 400
                constraints['team_size_max'] = max_agents
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid max agents format'}), 400
        
        # Prepare context
        context = {}
        if data.get('domain'):
            context['domain'] = data['domain']
        if data.get('priority'):
            context['priority'] = data['priority']
        
        # Start research in background thread
        def research_worker():
            try:
                global current_session
                session_id = f'session_{int(time.time())}'
                
                # Initialize current_session immediately
                global current_session
                current_session = {
                    'session_id': session_id,
                    'status': 'running',
                    'research_question': research_question,
                    'start_time': time.time()
                }
                
                # Emit status update
                socketio.emit('research_status', {
                    'status': 'starting',
                    'message': 'Initializing research session...'
                }, namespace='/')
                
                # Add initial chat log entry
                socketio.emit('chat_log', {
                    'session_id': session_id,
                    'type': 'system',
                    'author': 'System',
                    'message': f'Research session started: {research_question}',
                    'timestamp': time.time()
                }, namespace='/')
                
                # Conduct research with activity tracking
                def emit_agent_activity(agent_id, activity_type, message, status='active'):
                    socketio.emit('agent_activity', {
                        'session_id': session_id,
                        'agent_id': agent_id,
                        'type': activity_type,
                        'message': message,
                        'status': status,
                        'timestamp': time.time()
                    }, namespace='/')
                
                def emit_chat_log(log_type, author, message):
                    socketio.emit('chat_log', {
                        'session_id': session_id,
                        'type': log_type,
                        'author': author,
                        'message': message,
                        'timestamp': time.time()
                    }, namespace='/')
                
                # Set up activity tracking hooks
                def track_agent_activity(agent_id, activity_type, message, metadata=None):
                    emit_agent_activity(agent_id, activity_type, message)
                    # Log to framework
                    if hasattr(research_framework, 'log_agent_activity'):
                        research_framework.log_agent_activity(agent_id, activity_type, message, session_id, metadata)
                    elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                        research_framework.virtual_lab.log_agent_activity(agent_id, activity_type, message, session_id, metadata)
                
                def track_chat_message(log_type, author, message, metadata=None):
                    emit_chat_log(log_type, author, message)
                    # Log to framework
                    if hasattr(research_framework, 'log_chat_message'):
                        research_framework.log_chat_message(log_type, author, message, session_id, metadata)
                    elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                        research_framework.virtual_lab.log_chat_message(log_type, author, message, session_id, metadata)
                
                # Override framework methods to ensure logging
                original_log_agent_activity = getattr(research_framework, 'log_agent_activity', None)
                original_log_chat_message = getattr(research_framework, 'log_chat_message', None)
                
                def enhanced_log_agent_activity(agent_id, activity_type, message, session_id=None, metadata=None):
                    track_agent_activity(agent_id, activity_type, message, metadata)
                    if original_log_agent_activity:
                        return original_log_agent_activity(agent_id, activity_type, message, session_id, metadata)
                    return None
                
                def enhanced_log_chat_message(log_type, author, message, session_id=None, metadata=None):
                    track_chat_message(log_type, author, message, metadata)
                    if original_log_chat_message:
                        return original_log_chat_message(log_type, author, message, session_id, metadata)
                    return None
                
                # Store original conduct method
                original_conduct = getattr(research_framework, 'conduct_virtual_lab_research', None)
                
                def tracked_conduct_research(research_question, constraints=None, context=None):
                    # Emit initial status
                    emit_chat_log('system', 'System', 'Starting Virtual Lab research session')
                    
                    # Set up activity tracking hooks
                    def track_agent_activity(agent_id, activity_type, message, metadata=None):
                        emit_agent_activity(agent_id, activity_type, message)
                        # Also log to framework if available
                        if hasattr(research_framework, 'log_agent_activity'):
                            research_framework.log_agent_activity(agent_id, activity_type, message, session_id, metadata)
                        elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                            research_framework.virtual_lab.log_agent_activity(agent_id, activity_type, message, session_id, metadata)
                    
                    def track_chat_message(log_type, author, message, metadata=None):
                        emit_chat_log(log_type, author, message)
                        # Also log to framework if available
                        if hasattr(research_framework, 'log_chat_message'):
                            research_framework.log_chat_message(log_type, author, message, session_id, metadata)
                        elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                            research_framework.virtual_lab.log_chat_message(log_type, author, message, session_id, metadata)
                    
                    # Call the original method if it exists
                    if original_conduct:
                        results = original_conduct(research_question, constraints, context)
                    else:
                        # Fallback to virtual lab if available
                        if hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                            results = research_framework.virtual_lab.conduct_research_session(
                                research_question, constraints, context
                            )
                        else:
                            results = {'error': 'No research method available'}
                    
                    # Emit phase updates based on actual results
                    if results and 'phases' in results:
                        for i, (phase_name, phase_result) in enumerate(results['phases'].items()):
                            track_chat_message('thought', 'System', f'Completed phase: {phase_name.replace("_", " ").title()}')
                            socketio.emit('phase_update', {
                                'phase': i + 1,
                                'phase_name': phase_name,
                                'status': 'completed' if phase_result.get('success') else 'failed'
                            }, namespace='/')
                            
                            # Emit agent activity for successful phases
                            if phase_result.get('success'):
                                if 'hired_agents' in phase_result:
                                    for expertise, agent_info in phase_result['hired_agents'].get('hired_agents', {}).items():
                                        track_agent_activity(
                                            agent_info.get('agent_id', expertise),
                                            'meeting',
                                            f'Participated in {phase_name} phase',
                                            'active'
                                        )
                    
                    return results
                
                # Temporarily replace the method
                research_framework.conduct_virtual_lab_research = tracked_conduct_research
                
                # Conduct research with enhanced tracking
                try:
                    results = research_framework.conduct_virtual_lab_research(
                        research_question=research_question,
                        constraints=constraints,
                        context=context
                    )
                    
                    # Emit phase completion updates
                    if results and isinstance(results, dict):
                        if 'phase_results' in results:
                            for phase_name, phase_data in results['phase_results'].items():
                                phase_num = {
                                    'team_selection': 1,
                                    'literature_review': 2,
                                    'project_specification': 3,
                                    'tools_selection': 4,
                                    'tools_implementation': 5,
                                    'workflow_design': 6,
                                    'execution': 7,
                                    'synthesis': 8
                                }.get(phase_name, 0)
                                
                                socketio.emit('phase_update', {
                                    'phase': phase_num,
                                    'phase_name': phase_name.replace('_', ' ').title(),
                                    'status': 'completed' if phase_data.get('success') else 'failed',
                                    'progress': (phase_num / 8) * 100
                                }, namespace='/')
                except Exception as research_error:
                    logger.error(f"Research execution error: {research_error}")
                    results = {
                        'status': 'error',
                        'error': str(research_error),
                        'phase_results': {}
                    }
                
                # Restore original method
                research_framework.conduct_virtual_lab_research = original_conduct
                
                # Make results JSON serializable
                def make_json_serializable(obj):
                    """Recursively convert objects to JSON-serializable format."""
                    if hasattr(obj, 'to_dict'):
                        return obj.to_dict()
                    elif isinstance(obj, dict):
                        return {key: make_json_serializable(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(item) for item in obj]
                    elif isinstance(obj, set):
                        return [make_json_serializable(item) for item in obj]
                    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'MeetingRecord':
                        # Handle MeetingRecord objects specifically
                        return {
                            'meeting_id': getattr(obj, 'meeting_id', 'unknown'),
                            'meeting_type': getattr(obj, 'meeting_type', 'unknown'),
                            'phase': getattr(obj, 'phase', 'unknown'),
                            'participants': getattr(obj, 'participants', []),
                            'agenda': make_json_serializable(getattr(obj, 'agenda', {})),
                            'discussion_transcript': getattr(obj, 'discussion_transcript', []),
                            'outcomes': getattr(obj, 'outcomes', {}),
                            'decisions': getattr(obj, 'decisions', []),
                            'action_items': getattr(obj, 'action_items', []),
                            'start_time': getattr(obj, 'start_time', 0.0),
                            'end_time': getattr(obj, 'end_time', 0.0),
                            'success': getattr(obj, 'success', False)
                        }
                    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'MeetingAgenda':
                        # Handle MeetingAgenda objects specifically
                        return {
                            'meeting_id': getattr(obj, 'meeting_id', 'unknown'),
                            'meeting_type': getattr(obj, 'meeting_type', 'unknown'),
                            'phase': getattr(obj, 'phase', 'unknown'),
                            'objectives': getattr(obj, 'objectives', []),
                            'participants': getattr(obj, 'participants', []),
                            'discussion_topics': getattr(obj, 'discussion_topics', []),
                            'expected_outcomes': getattr(obj, 'expected_outcomes', []),
                            'duration_minutes': getattr(obj, 'duration_minutes', 10)
                        }
                    elif hasattr(obj, '__class__') and obj.__class__.__name__ in ['GeneralExpertAgent', 'BaseAgent', 'SimpleAgent']:
                        # Handle agent objects that might not have to_dict
                        return {
                            'agent_id': getattr(obj, 'agent_id', 'unknown'),
                            'role': getattr(obj, 'role', 'unknown'),
                            'expertise': getattr(obj, 'expertise', []),
                            'agent_type': obj.__class__.__name__
                        }
                    elif hasattr(obj, '__dict__'):
                        return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
                    elif hasattr(obj, 'value'):
                        # Handle Enum objects
                        return obj.value
                    else:
                        return obj
                
                # Convert results to JSON serializable format
                try:
                    serializable_results = make_json_serializable(results)
                except Exception as e:
                    logger.error(f"Error serializing results: {e}")
                    # Fallback to basic serialization
                    serializable_results = {
                        'status': 'error',
                        'error': f'Serialization error: {str(e)}',
                        'raw_results': str(results)
                    }
                
                current_session = {
                    'session_id': session_id,
                    'status': serializable_results.get('status', 'completed'),
                    'research_question': research_question,
                    'results': serializable_results,
                    'start_time': time.time(),
                    'end_time': time.time()
                }
                
                # Store in database with app context
                with app.app_context():
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
    """Get list of all research sessions."""
    try:
        db = get_db()
        if db is None:
            return jsonify({
                'sessions': [],
                'total': 0
            })
        
        try:
            sessions = db.execute('''
                SELECT DISTINCT session_id, 
                       MIN(timestamp) as start_time,
                       MAX(timestamp) as end_time,
                       COUNT(*) as activity_count
                FROM agent_activity 
                GROUP BY session_id
                ORDER BY start_time DESC
            ''').fetchall()
            
            return jsonify({
                'sessions': [dict(session) for session in sessions],
                'total': len(sessions)
            })
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        # Return empty sessions list instead of error
        return jsonify({
            'sessions': [],
            'total': 0
        })

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get detailed information about a specific session."""
    try:
        db = get_db()
        
        # Get session metadata
        session_info = db.execute('''
            SELECT session_id, 
                   MIN(timestamp) as start_time,
                   MAX(timestamp) as end_time,
                   COUNT(*) as activity_count
            FROM agent_activity 
            WHERE session_id = ?
            GROUP BY session_id
        ''', (session_id,)).fetchone()
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get agent activity for this session
        activities = db.execute('''
            SELECT * FROM agent_activity 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (session_id,)).fetchall()
        
        # Get chat logs for this session
        chat_logs = db.execute('''
            SELECT * FROM chat_logs 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (session_id,)).fetchall()
        
        # Get meetings for this session
        meetings = db.execute('''
            SELECT * FROM meetings 
            WHERE session_id = ?
            ORDER BY timestamp DESC
        ''', (session_id,)).fetchall()
        
        return jsonify({
            'session': dict(session_info),
            'activities': [dict(activity) for activity in activities],
            'chat_logs': [dict(log) for log in chat_logs],
            'meetings': [dict(meeting) for meeting in meetings]
        })
        
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics."""
    try:
        timeframe = request.args.get('timeframe', 'session')
        
        # Get current metrics (this doesn't require database)
        current_metrics = get_system_metrics()
        
        # Try to get database metrics, but don't fail if database is closed
        try:
            db = get_db()
            if db is None:
                # Return metrics without database data
                return jsonify({
                    'current': current_metrics,
                    'history': [],
                    'session_stats': {'total_sessions': 0, 'successful_sessions': 0},
                    'agent_stats': {'active_agents': 0, 'avg_quality_score': 0.0, 'critical_issues': 0, 'total_agents': 0}
                })
            
            try:
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
                
                # Get session statistics
                session_stats = db.execute('''
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_sessions
                    FROM sessions
                ''').fetchone()
                
                # Get agent stats
                agent_stats = get_agent_stats(db, current_session)
                
                return jsonify({
                    'current': current_metrics,
                    'history': [dict(m) for m in metrics],
                    'session_stats': dict(session_stats) if session_stats else {},
                    'agent_stats': agent_stats
                })
                
            finally:
                db.close()
                
        except Exception as db_error:
            logger.warning(f"Database error in metrics endpoint: {db_error}")
            # Return metrics without database data
            return jsonify({
                'current': current_metrics,
                'history': [],
                'session_stats': {'total_sessions': 0, 'successful_sessions': 0},
                'agent_stats': {'active_agents': 0, 'avg_quality_score': 0.0, 'critical_issues': 0, 'total_agents': 0}
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
        
        # Store in database
        if current_session:
            session_id = current_session.get('session_id')
            db = get_db()
            db.execute('''
                INSERT INTO agent_activity (session_id, agent_id, activity_type, message, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, 'human', 'intervention', message, 'active'))
            db.commit()
        
        # Emit to all clients
        socketio.emit('activity_log', activity, namespace='/')
        
        # If research is active, potentially influence it
        if current_session and research_framework:
            logger.info(f"Human intervention: {message}")
        
        return jsonify({'success': True, 'message': 'Intervention sent'})
        
    except Exception as e:
        logger.error(f"Error sending intervention: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent-activity', methods=['GET'])
def get_agent_activity():
    """Get agent activity for a session."""
    try:
        session_id = request.args.get('session_id')
        agent_id = request.args.get('agent_id')
        limit = int(request.args.get('limit', 50))
        
        # Try to get from framework first
        activities = []
        if research_framework:
            if hasattr(research_framework, 'get_agent_activity_log'):
                activities = research_framework.get_agent_activity_log(session_id)
            elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                activities = research_framework.virtual_lab.get_agent_activity_log(session_id)
        
        # Filter by agent_id if specified
        if agent_id:
            activities = [a for a in activities if a.get('agent_id') == agent_id]
        
        # Limit results
        activities = activities[-limit:] if activities else []
        
        return jsonify({
            'activities': activities,
            'total': len(activities)
        })
        
    except Exception as e:
        logger.error(f"Error getting agent activity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-logs', methods=['GET'])
def get_chat_logs():
    """Get chat logs for a session."""
    try:
        session_id = request.args.get('session_id')
        log_type = request.args.get('type')  # thought, choice, communication, tool_call, system
        limit = int(request.args.get('limit', 100))
        
        # Try to get from framework first
        logs = []
        if research_framework:
            if hasattr(research_framework, 'get_chat_logs'):
                logs = research_framework.get_chat_logs(session_id)
            elif hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
                logs = research_framework.virtual_lab.get_chat_logs(session_id)
        
        # Filter by log_type if specified
        if log_type:
            logs = [log for log in logs if log.get('log_type') == log_type]
        
        # Limit results
        logs = logs[-limit:] if logs else []
        
        return jsonify({
            'logs': logs,
            'total': len(logs)
        })
        
    except Exception as e:
        logger.error(f"Error getting chat logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-logs', methods=['POST'])
def add_chat_log():
    """Add a new chat log entry."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        log_type = data.get('type')
        author = data.get('author', 'System')
        message = data.get('message', '')
        metadata = data.get('metadata', '{}')
        
        if not session_id or not log_type or not message:
            return jsonify({'error': 'session_id, type, and message are required'}), 400
        
        db = get_db()
        db.execute('''
            INSERT INTO chat_logs (session_id, log_type, author, message, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, log_type, author, message, metadata))
        db.commit()
        
        # Emit to all clients
        socketio.emit('chat_log', {
            'session_id': session_id,
            'type': log_type,
            'author': author,
            'message': message,
            'timestamp': time.time()
        }, namespace='/')
        
        return jsonify({'success': True, 'message': 'Chat log added'})
        
    except Exception as e:
        logger.error(f"Error adding chat log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent-activity', methods=['POST'])
def add_agent_activity():
    """Add a new agent activity entry."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        agent_id = data.get('agent_id')
        activity_type = data.get('type')
        message = data.get('message', '')
        status = data.get('status', 'active')
        metadata = data.get('metadata', '{}')
        
        if not session_id or not agent_id or not activity_type:
            return jsonify({'error': 'session_id, agent_id, and type are required'}), 400
        
        db = get_db()
        db.execute('''
            INSERT INTO agent_activity (session_id, agent_id, activity_type, message, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, agent_id, activity_type, message, status, metadata))
        db.commit()
        
        # Emit to all clients
        socketio.emit('agent_activity', {
            'session_id': session_id,
            'agent_id': agent_id,
            'type': activity_type,
            'message': message,
            'status': status,
            'timestamp': time.time()
        }, namespace='/')
        
        return jsonify({'success': True, 'message': 'Agent activity added'})
        
    except Exception as e:
        logger.error(f"Error adding agent activity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    """Get list of meetings."""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 50))
        
        db = get_db()
        query = '''
            SELECT * FROM meetings 
            WHERE 1=1
        '''
        params = []
        
        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        meetings = db.execute(query, params).fetchall()
        
        # If no meetings found and no session_id specified, return empty list
        # instead of example data
        return jsonify({
            'meetings': [dict(meeting) for meeting in meetings],
            'total': len(meetings)
        })
        
    except Exception as e:
        logger.error(f"Error getting meetings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings', methods=['POST'])
def add_meeting():
    """Add a new meeting entry."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        meeting_id = data.get('meeting_id')
        participants = data.get('participants', '[]')
        topic = data.get('topic', '')
        duration = data.get('duration', 0)
        outcome = data.get('outcome', '')
        transcript = data.get('transcript', '')
        
        if not session_id or not meeting_id:
            return jsonify({'error': 'session_id and meeting_id are required'}), 400
        
        db = get_db()
        db.execute('''
            INSERT INTO meetings (session_id, meeting_id, participants, topic, duration, outcome, transcript)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, meeting_id, participants, topic, duration, outcome, transcript))
        db.commit()
        
        # Emit to all clients
        socketio.emit('meeting', {
            'session_id': session_id,
            'meeting_id': meeting_id,
            'participants': participants,
            'topic': topic,
            'duration': duration,
            'outcome': outcome,
            'timestamp': time.time()
        }, namespace='/')
        
        return jsonify({'success': True, 'message': 'Meeting added'})
        
    except Exception as e:
        logger.error(f"Error adding meeting: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get all agents from the framework."""
    try:
        agents = []
        added_agent_ids = set()
        
        if research_framework and hasattr(research_framework, 'virtual_lab') and research_framework.virtual_lab:
            # Get agents from virtual lab if available
            all_agents = research_framework.virtual_lab.get_active_agents()
            
            # Add PI agent
            if hasattr(research_framework.virtual_lab, 'pi_agent'):
                pi_agent = research_framework.virtual_lab.pi_agent
                if pi_agent.agent_id not in added_agent_ids:
                    agents.append({
                        'id': pi_agent.agent_id,
                        'name': pi_agent.role,
                        'expertise': ', '.join(pi_agent.expertise),
                        'status': 'active' if hasattr(pi_agent, 'is_active') and pi_agent.is_active() else 'active',
                        'is_active': hasattr(pi_agent, 'is_active') and pi_agent.is_active(),
                        'current_task': getattr(pi_agent, 'current_task', {}).get('id') if hasattr(pi_agent, 'current_task') and getattr(pi_agent, 'current_task') else None,
                        'performance': getattr(pi_agent, 'performance_metrics', {
                            'contributions': 0,
                            'meetings_attended': 0,
                            'tools_used': 0,
                            'phases_completed': 0
                        }),
                        'recent_activities': [],
                        'agent_type': 'Principal Investigator'
                    })
                    added_agent_ids.add(pi_agent.agent_id)
            
            # Add Scientific Critic
            if hasattr(research_framework.virtual_lab, 'scientific_critic'):
                critic_agent = research_framework.virtual_lab.scientific_critic
                if critic_agent.agent_id not in added_agent_ids:
                    agents.append({
                        'id': critic_agent.agent_id,
                        'name': critic_agent.role,
                        'expertise': ', '.join(critic_agent.expertise),
                        'status': 'active' if hasattr(critic_agent, 'is_active') and critic_agent.is_active() else 'active',
                        'is_active': hasattr(critic_agent, 'is_active') and critic_agent.is_active(),
                        'current_task': getattr(critic_agent, 'current_task', {}).get('id') if hasattr(critic_agent, 'current_task') and getattr(critic_agent, 'current_task') else None,
                        'performance': getattr(critic_agent, 'performance_metrics', {
                            'contributions': 0,
                            'meetings_attended': 0,
                            'tools_used': 0,
                            'phases_completed': 0
                        }),
                        'recent_activities': [],
                        'agent_type': 'Scientific Critic'
                    })
                    added_agent_ids.add(critic_agent.agent_id)
            
            # Add other agents (excluding PI and Critic which are already added)
            for agent in all_agents:
                # Skip if this agent is already added
                if agent.agent_id in added_agent_ids:
                    continue
                # Get recent activities from database
                db = get_db()
                recent_activities = []
                if db:
                    try:
                        activities = db.execute('''
                            SELECT activity_type, message, timestamp 
                            FROM agent_activity 
                            WHERE agent_id = ? 
                            ORDER BY timestamp DESC 
                            LIMIT 5
                        ''', (agent.agent_id,)).fetchall()
                        recent_activities = [{
                            'activity': activity['message'],
                            'type': activity['activity_type'],
                            'timestamp': activity['timestamp']
                        } for activity in activities]
                    except:
                        pass
                
                agents.append({
                    'id': agent.agent_id,
                    'name': agent.role,
                    'expertise': ', '.join(agent.expertise),
                    'status': 'active' if hasattr(agent, 'is_active') and agent.is_active() else 'idle',
                    'is_active': hasattr(agent, 'is_active') and agent.is_active(),
                    'current_task': getattr(agent, 'current_task', {}).get('id') if hasattr(agent, 'current_task') and getattr(agent, 'current_task') else None,
                    'performance': getattr(agent, 'performance_metrics', {
                        'contributions': 0,
                        'meetings_attended': 0,
                        'tools_used': 0,
                        'phases_completed': 0
                    }),
                    'recent_activities': recent_activities,
                    'agent_type': agent.__class__.__name__
                })
                added_agent_ids.add(agent.agent_id)
        elif hasattr(research_framework, 'get_active_agents'):
            all_agents = research_framework.get_active_agents()
            for agent in all_agents:
                agents.append({
                    'id': agent.agent_id,
                    'name': agent.role,
                    'expertise': ', '.join(agent.expertise),
                    'status': 'active' if hasattr(agent, 'is_active') and agent.is_active() else 'idle',
                    'is_active': hasattr(agent, 'is_active') and agent.is_active(),
                    'current_task': getattr(agent, 'current_task', {}).get('id') if hasattr(agent, 'current_task') and getattr(agent, 'current_task') else None,
                    'performance': getattr(agent, 'performance_metrics', {
                        'contributions': 0,
                        'meetings_attended': 0,
                        'tools_used': 0,
                        'phases_completed': 0
                    }),
                    'recent_activities': [],
                    'agent_type': agent.__class__.__name__
                })
        
        return jsonify({'agents': agents})
        
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        return jsonify({'agents': [], 'error': str(e)})

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

@socketio.on('heartbeat')
def handle_heartbeat():
    """Handle client heartbeat to keep connection alive."""
    client_id = request.sid
    if client_id in active_connections:
        active_connections[client_id]['last_seen'] = time.time()
    
    emit('heartbeat_ack', {'timestamp': time.time()})

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
    
    # Exempt API routes from CSRF protection
    exempt_api_routes()
    
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