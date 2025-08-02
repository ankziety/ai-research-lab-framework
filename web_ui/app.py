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
    # Determine if running in development mode
    flask_env = os.environ.get('FLASK_ENV', '').lower()
    flask_debug = os.environ.get('FLASK_DEBUG', '').lower()
    if flask_env in ('development', 'debug') or flask_debug in ('1', 'true', 'yes'):
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

# Thread-local storage for database connections
_thread_local = threading.local()

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
    """Get thread-local database connection."""
    if not hasattr(_thread_local, 'db'):
        _thread_local.db = sqlite3.connect('research_sessions.db')
        _thread_local.db.row_factory = sqlite3.Row
    return _thread_local.db

def close_db_connection():
    """Close thread-local database connection."""
    if hasattr(_thread_local, 'db'):
        _thread_local.db.close()
        delattr(_thread_local, 'db')

@app.teardown_appcontext
def close_db(error):
    close_db_connection()

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
        with app.app_context():
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
                session_id = f'session_{int(time.time())}'
                
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
                
                # Store the original method before replacing it
                original_conduct = research_framework.conduct_virtual_lab_research
                
                # Call the actual virtual lab research
                def tracked_conduct_research(research_question, constraints=None, context=None):
                    # Emit initial status
                    emit_chat_log('system', 'System', 'Starting Virtual Lab research session')
                    
                    # Call the original method (not the replaced one)
                    results = original_conduct(research_question, constraints, context)
                    
                    # Emit phase updates based on actual results
                    if results and 'phases' in results:
                        for i, (phase_name, phase_result) in enumerate(results['phases'].items()):
                            emit_chat_log('thought', 'System', f'Completed phase: {phase_name.replace("_", " ").title()}')
                            socketio.emit('phase_update', {
                                'phase': i + 1,
                                'phase_name': phase_name,
                                'status': 'completed' if phase_result.get('success') else 'failed'
                            }, namespace='/')
                            
                            # Emit agent activity for successful phases
                            if phase_result.get('success'):
                                if 'hired_agents' in phase_result:
                                    for expertise, agent_info in phase_result['hired_agents'].get('hired_agents', {}).items():
                                        emit_agent_activity(
                                            agent_info.get('agent_id', expertise),
                                            'meeting',
                                            f'Participated in {phase_name} phase',
                                            'active'
                                        )
                    
                    # Extract and store meeting records from results
                    if results and 'phases' in results:
                        for phase_name, phase_result in results['phases'].items():
                            if phase_result and 'meeting_record' in phase_result:
                                meeting_record = phase_result['meeting_record']
                                if hasattr(meeting_record, 'to_dict'):
                                    meeting_data = meeting_record.to_dict()
                                    
                                    # Store meeting in database
                                    db = get_db()
                                    db.execute('''
                                        INSERT INTO meetings (session_id, meeting_id, participants, topic, duration, outcome, transcript)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                    ''', (
                                        session_id,
                                        meeting_data.get('meeting_id', f'{phase_name}_meeting'),
                                        json.dumps(meeting_data.get('participants', [])),
                                        f'{phase_name.replace("_", " ").title()} Meeting',
                                        meeting_data.get('end_time', 0) - meeting_data.get('start_time', 0),
                                        json.dumps(meeting_data.get('outcomes', {})),
                                        json.dumps(meeting_data.get('discussion_transcript', []))
                                    ))
                                    db.commit()
                                    
                                    # Emit meeting to UI
                                    socketio.emit('meeting', {
                                        'session_id': session_id,
                                        'meeting_id': meeting_data.get('meeting_id', f'{phase_name}_meeting'),
                                        'participants': json.dumps(meeting_data.get('participants', [])),
                                        'topic': f'{phase_name.replace("_", " ").title()} Meeting',
                                        'duration': meeting_data.get('end_time', 0) - meeting_data.get('start_time', 0),
                                        'outcome': json.dumps(meeting_data.get('outcomes', {})),
                                        'timestamp': meeting_data.get('start_time', time.time())
                                    }, namespace='/')
                    
                    return results
                
                # Temporarily replace the method
                research_framework.conduct_virtual_lab_research = tracked_conduct_research
                
                # Conduct research
                results = research_framework.conduct_virtual_lab_research(
                    research_question=research_question,
                    constraints=constraints,
                    context=context
                )
                
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
                    'start_time': time.time()
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

@app.route('/api/chat-logs', methods=['GET'])
def get_chat_logs():
    """Get chat logs for a session."""
    try:
        session_id = request.args.get('session_id')
        log_type = request.args.get('type')  # thought, choice, communication, tool_call, system
        limit = int(request.args.get('limit', 100))
        
        db = get_db()
        
        query = '''
            SELECT * FROM chat_logs 
            WHERE 1=1
        '''
        params = []
        
        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        if log_type:
            query += ' AND log_type = ?'
            params.append(log_type)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        logs = db.execute(query, params).fetchall()
        
        return jsonify({
            'logs': [dict(log) for log in logs],
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

@app.route('/api/agent-activity', methods=['GET'])
def get_agent_activity():
    """Get agent activity for a session."""
    try:
        session_id = request.args.get('session_id')
        agent_id = request.args.get('agent_id')
        limit = int(request.args.get('limit', 50))
        
        db = get_db()
        
        query = '''
            SELECT * FROM agent_activity 
            WHERE 1=1
        '''
        params = []
        
        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        if agent_id:
            query += ' AND agent_id = ?'
            params.append(agent_id)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        activities = db.execute(query, params).fetchall()
        
        return jsonify({
            'activities': [dict(activity) for activity in activities],
            'total': len(activities)
        })
        
    except Exception as e:
        logger.error(f"Error getting agent activity: {e}")
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
    """Get meetings for a session."""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 20))
        
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