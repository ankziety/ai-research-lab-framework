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
from pathlib import Path

# Add parent directory to path to import the research framework
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import using absolute imports instead of relative imports
from core.ai_research_lab import create_framework
from core.virtual_lab import ResearchPhase, MeetingRecord, MeetingAgenda
from core.multi_agent_framework import MultiAgentResearchFramework
from web_ui.data_manager import DataManager
from web_ui.data_migration import DataMigration

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
    is_production = os.environ.get('PRODUCTION', '').lower() in ('1', 'true', 'yes')
    
    # Default to development mode if no explicit production flag is set
    if not is_production or flask_env in ('development', 'debug') or flask_debug in ('1', 'true', 'yes'):
        # Generate a random secret key for development
        secret_key = secrets.token_urlsafe(32)
        logging.warning("SECRET_KEY not set, using a random key for development. Do not use this in production!")
    else:
        raise RuntimeError("SECRET_KEY environment variable must be set in production. Set PRODUCTION=true if running in production.")

# Configure Flask for better WebSocket support
app.config['SECRET_KEY'] = secret_key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SOCKETIO_ASYNC_MODE'] = 'threading'
app.config['SOCKETIO_PING_TIMEOUT'] = 60
app.config['SOCKETIO_PING_INTERVAL'] = 25

# SocketIO setup for real-time communication
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=True, 
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    allow_upgrades=True
)

# Global state
research_framework: Optional[MultiAgentResearchFramework] = None
current_session: Optional[Dict[str, Any]] = None
system_config: Dict[str, Any] = {}
active_connections: Dict[str, Any] = {}

# Initialize data manager
data_manager: Optional[DataManager] = None

# Thread-local storage for database connections
_thread_local = threading.local()

def get_db():
    """Get thread-local database connection."""
    if not hasattr(_thread_local, 'db'):
        if data_manager:
            _thread_local.db = data_manager.get_db_connection()
        else:
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

# Utility functions
def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format with improved error handling."""
    try:
        if obj is None:
            return None
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {str(key): make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'MeetingRecord':
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
            return {
                'agent_id': getattr(obj, 'agent_id', 'unknown'),
                'role': getattr(obj, 'role', 'unknown'),
                'expertise': getattr(obj, 'expertise', []),
                'agent_type': obj.__class__.__name__
            }
        elif hasattr(obj, '__dict__'):
            return {str(key): make_json_serializable(value) for key, value in obj.__dict__.items()}
        elif hasattr(obj, 'value'):
            return obj.value
        else:
            return str(obj)
    except Exception as e:
        logger.warning(f"Failed to serialize object {type(obj)}: {e}")
        return f"<Serialization failed: {type(obj).__name__}>"

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
        # Use non-blocking CPU measurement to avoid websocket timeout issues
        cpu_percent = psutil.cpu_percent(interval=None)
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
            if data_manager:
                data_manager.persist_metrics(
                    session_id=session_id,
                    cpu_usage=metrics['cpu_usage'],
                    memory_usage=metrics['memory_usage'],
                    active_agents=metrics['active_agents']
                )
            else:
                # Fallback to direct database access
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
            
            # Also emit agent statistics
            agent_stats = get_agent_statistics()
            socketio.emit('agent_statistics', agent_stats, namespace='/')
            
            # Store in database with app context
            session_id = current_session.get('session_id') if current_session else None
            store_metrics(session_id)
            
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")
            time.sleep(10)

def get_agent_statistics():
    """Calculate real agent statistics from the research framework."""
    try:
        if not research_framework:
            return {
                'total_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0
            }
        
        # Get marketplace statistics
        marketplace_stats = research_framework.agent_marketplace.get_marketplace_statistics()
        total_agents = marketplace_stats.get('total_agents', 0)
        
        # Calculate average quality score from all agents
        avg_quality_score = 0.0
        if total_agents > 0:
            quality_scores = []
            for agent in research_framework.agent_marketplace.agent_registry.values():
                if hasattr(agent, 'performance_metrics') and agent.performance_metrics:
                    score = agent.performance_metrics.get('average_quality_score', 0.0)
                    if score > 0:
                        quality_scores.append(score)
            
            if quality_scores:
                avg_quality_score = sum(quality_scores) / len(quality_scores)
        
        # Get critical issues from scientific critic
        critical_issues = 0
        if hasattr(research_framework, 'scientific_critic') and research_framework.scientific_critic:
            critic = research_framework.scientific_critic
            if hasattr(critic, 'critique_history') and critic.critique_history:
                critical_issues = sum(1 for critique in critic.critique_history 
                                   if critique.get('critical_issues', []))
        
        return {
            'total_agents': total_agents,
            'avg_quality_score': round(avg_quality_score, 2),
            'critical_issues': critical_issues
        }
        
    except Exception as e:
        logger.error(f"Error calculating agent statistics: {e}")
        return {
            'total_agents': 0,
            'avg_quality_score': 0.0,
            'critical_issues': 0
        }

# Routes
@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/test-page')
def test_page():
    """Serve the test page."""
    return render_template('test.html')

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'socketio_available': True
    })

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify the web UI is working."""
    return jsonify({
        'status': 'ok',
        'message': 'Web UI is working correctly',
        'timestamp': time.time(),
        'framework_initialized': research_framework is not None
    })

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
            'enable_free_search': system_config.get('framework', {}).get('enable_free_search', False),
            'enable_mock_responses': system_config.get('framework', {}).get('enable_mock_responses', False)
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
        
        # Generate session_id once
        session_id = f'session_{int(time.time())}'
        
        # Start research in background thread
        def research_worker():
            try:
                global current_session
                
                # Emit status update
                socketio.emit('research_status', {
                    'status': 'starting',
                    'message': 'Initializing research session...'
                }, namespace='/')
                
                # Add initial chat log entry
                initial_log = {
                    'session_id': session_id,
                    'type': 'system',
                    'author': 'System',
                    'message': f'Research session started: {research_question}',
                    'timestamp': time.time()
                }
                socketio.emit('chat_log', initial_log, namespace='/')
                
                # Persist the initial log
                if data_manager:
                    data_manager.persist_chat_log(
                        session_id=session_id,
                        log_type='system',
                        author='System',
                        message=f'Research session started: {research_question}'
                    )
                
                # Conduct research with activity tracking
                def emit_agent_activity(agent_id, activity_type, message, status='active'):
                    activity_data = {
                        'session_id': session_id,
                        'agent_id': agent_id,
                        'type': activity_type,
                        'message': message,
                        'status': status,
                        'timestamp': time.time()
                    }
                    socketio.emit('agent_activity', activity_data, namespace='/')
                    
                    # Persist to database
                    if data_manager:
                        data_manager.persist_agent_activity(
                            session_id=session_id,
                            agent_id=agent_id,
                            activity_type=activity_type,
                            message=message,
                            status=status
                        )
                
                def emit_chat_log(log_type, author, message):
                    chat_data = {
                        'session_id': session_id,
                        'type': log_type,
                        'author': author,
                        'message': message,
                        'timestamp': time.time()
                    }
                    socketio.emit('chat_log', chat_data, namespace='/')
                    
                    # Persist to database
                    if data_manager:
                        data_manager.persist_chat_log(
                            session_id=session_id,
                            log_type=log_type,
                            author=author,
                            message=message
                        )
                
                # Store the original method before replacing it
                original_conduct = research_framework.conduct_virtual_lab_research
                
                # Call the actual virtual lab research
                def tracked_conduct_research(research_question, constraints=None, context=None):
                    # Emit initial status
                    emit_chat_log('system', 'System', 'Starting Virtual Lab research session')
                    
                    # Call the original method (not the replaced one) with session_id
                    results = original_conduct(research_question, constraints, context, session_id)
                    
                    # Check if results is valid
                    if not isinstance(results, dict):
                        emit_chat_log('system', 'System', f'Research completed with non-dict results: {type(results)}')
                        logger.warning(f"Research returned non-dict results: {type(results)} - {str(results)[:200]}")
                    
                    # Emit phase updates based on actual results
                    if results and isinstance(results, dict) and 'phases' in results:
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
                    if results and isinstance(results, dict) and 'phases' in results:
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
                    if data_manager:
                        data_manager.persist_session(
                            session_id=current_session['session_id'],
                            research_question=research_question,
                            status=current_session['status'],
                            config={'constraints': constraints, 'context': context},
                            results=serializable_results
                        )
                    else:
                        # Fallback to direct database access
                        db = get_db()
                        db.execute('''
                            INSERT OR REPLACE INTO sessions 
                            (id, research_question, config, results, status)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (current_session['session_id'], research_question,
                              json.dumps({'constraints': constraints, 'context': context}),
                              json.dumps(serializable_results), current_session['status']))
                        db.commit()
                
                # Emit completion
                socketio.emit('research_complete', current_session, namespace='/')
                
            except Exception as e:
                logger.error(f"Research error: {e}")
                current_session = {
                    'session_id': session_id,
                    'status': 'failed',
                    'research_question': research_question,
                    'results': {
                        'success': False,
                        'error': str(e),
                        'research_question': research_question,
                        'timestamp': time.time()
                    },
                    'start_time': time.time(),
                    'end_time': time.time()
                }
                socketio.emit('research_error', {'error': str(e)}, namespace='/')
        
        thread = threading.Thread(target=research_worker)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Research session started',
            'session_id': session_id
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
        if data_manager:
            sessions = data_manager.get_all_sessions(limit=50)
            return jsonify(sessions)
        else:
            # Fallback to direct database access
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
        if data_manager:
            session = data_manager.get_session(session_id)
            if session:
                return jsonify(session)
            else:
                return jsonify({'error': 'Session not found'}), 404
        else:
            # Fallback to direct database access
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
        
        if data_manager:
            if timeframe == 'session' and current_session:
                # Get metrics for current session
                session_id = current_session.get('session_id')
                metrics = data_manager.get_metrics(session_id=session_id)
            else:
                # Get recent metrics
                hours = {'24h': 24, '7d': 168, '30d': 720}.get(timeframe, 1)
                metrics = data_manager.get_metrics(hours=hours)
            
            # Get summary statistics
            current_metrics = get_system_metrics()
            
            # Get session statistics
            sessions = data_manager.get_all_sessions(limit=1000)
            total_sessions = len(sessions)
            successful_sessions = len([s for s in sessions if s.get('status') == 'completed'])
            
            return jsonify({
                'current': current_metrics,
                'history': metrics,
                'session_stats': {
                    'total_sessions': total_sessions,
                    'successful_sessions': successful_sessions
                },
                'agent_stats': get_agent_statistics()
            })
        else:
            # Fallback to direct database access
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
                'agent_stats': get_agent_statistics()
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
        
        if data_manager:
            logs = data_manager.get_chat_logs(
                session_id=session_id,
                log_type=log_type,
                limit=limit
            )
            return jsonify({
                'logs': logs,
                'total': len(logs)
            })
        else:
            # Fallback to direct database access
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
        metadata = data.get('metadata', {})
        
        if not session_id or not log_type or not message:
            return jsonify({'error': 'session_id, type, and message are required'}), 400
        
        # Persist to database
        if data_manager:
            success = data_manager.persist_chat_log(
                session_id=session_id,
                log_type=log_type,
                author=author,
                message=message,
                metadata=metadata
            )
        else:
            # Fallback to direct database access
            db = get_db()
            db.execute('''
                INSERT INTO chat_logs (session_id, log_type, author, message, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, log_type, author, message, json.dumps(metadata)))
            db.commit()
            success = True
        
        if success:
            # Emit to all clients
            socketio.emit('chat_log', {
                'session_id': session_id,
                'type': log_type,
                'author': author,
                'message': message,
                'timestamp': time.time()
            }, namespace='/')
            
            return jsonify({'success': True, 'message': 'Chat log added'})
        else:
            return jsonify({'error': 'Failed to persist chat log'}), 500
        
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
        
        if data_manager:
            activities = data_manager.get_agent_activity(
                session_id=session_id,
                agent_id=agent_id,
                limit=limit
            )
            return jsonify({
                'activities': activities,
                'total': len(activities)
            })
        else:
            # Fallback to direct database access
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
        metadata = data.get('metadata', {})
        
        if not session_id or not agent_id or not activity_type:
            return jsonify({'error': 'session_id, agent_id, and type are required'}), 400
        
        # Persist to database
        if data_manager:
            success = data_manager.persist_agent_activity(
                session_id=session_id,
                agent_id=agent_id,
                activity_type=activity_type,
                message=message,
                status=status,
                metadata=metadata
            )
        else:
            # Fallback to direct database access
            db = get_db()
            db.execute('''
                INSERT INTO agent_activity (session_id, agent_id, activity_type, message, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, agent_id, activity_type, message, status, json.dumps(metadata)))
            db.commit()
            success = True
        
        if success:
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
        else:
            return jsonify({'error': 'Failed to persist agent activity'}), 500
        
    except Exception as e:
        logger.error(f"Error adding agent activity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    """Get meetings for a session."""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 20))
        
        if data_manager:
            meetings = data_manager.get_meetings(
                session_id=session_id,
                limit=limit
            )
            return jsonify({
                'meetings': meetings,
                'total': len(meetings)
            })
        else:
            # Fallback to direct database access
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
        participants = data.get('participants', [])
        topic = data.get('topic', '')
        agenda = data.get('agenda', {})
        transcript = data.get('transcript', [])
        outcomes = data.get('outcomes', {})
        metadata = data.get('metadata', {})
        
        if not session_id or not meeting_id:
            return jsonify({'error': 'session_id and meeting_id are required'}), 400
        
        # Persist to database
        if data_manager:
            success = data_manager.persist_meeting(
                session_id=session_id,
                meeting_id=meeting_id,
                participants=participants,
                topic=topic,
                agenda=agenda,
                transcript=transcript,
                outcomes=outcomes,
                metadata=metadata
            )
        else:
            # Fallback to direct database access
            db = get_db()
            db.execute('''
                INSERT INTO meetings (session_id, meeting_id, participants, topic, agenda, transcript, outcomes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, meeting_id, json.dumps(participants), topic, 
                  json.dumps(agenda), json.dumps(transcript), json.dumps(outcomes), json.dumps(metadata)))
            db.commit()
            success = True
        
        if success:
            # Emit to all clients
            socketio.emit('meeting', {
                'session_id': session_id,
                'meeting_id': meeting_id,
                'participants': participants,
                'topic': topic,
                'timestamp': time.time()
            }, namespace='/')
            
            return jsonify({'success': True, 'message': 'Meeting added'})
        else:
            return jsonify({'error': 'Failed to persist meeting'}), 500
        
    except Exception as e:
        logger.error(f"Error adding meeting: {e}")
        return jsonify({'error': str(e)}), 500

# Data management endpoints
@app.route('/api/data/backup', methods=['POST'])
def create_backup():
    """Create a database backup."""
    try:
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        backup_path = data_manager.backup_database()
        return jsonify({
            'message': 'Backup created successfully',
            'backup_path': backup_path
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/restore', methods=['POST'])
def restore_backup():
    """Restore database from backup."""
    try:
        data = request.get_json()
        if not data or 'backup_path' not in data:
            return jsonify({'error': 'Backup path required'}), 400
        
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        success = data_manager.restore_database(data['backup_path'])
        if success:
            return jsonify({'message': 'Database restored successfully'}), 200
        else:
            return jsonify({'error': 'Failed to restore database'}), 500
        
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/export', methods=['POST'])
def export_data():
    """Export data to archive."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        export_path = data_manager.export_data(session_id)
        return jsonify({
            'message': 'Data exported successfully',
            'export_path': export_path
        }), 200
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/import', methods=['POST'])
def import_data():
    """Import data from archive."""
    try:
        data = request.get_json()
        if not data or 'import_path' not in data:
            return jsonify({'error': 'Import path required'}), 400
        
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        success = data_manager.import_data(data['import_path'])
        if success:
            return jsonify({'message': 'Data imported successfully'}), 200
        else:
            return jsonify({'error': 'Failed to import data'}), 500
        
    except Exception as e:
        logger.error(f"Error importing data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/integrity', methods=['GET'])
def validate_data_integrity():
    """Validate data integrity."""
    try:
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        integrity_results = data_manager.validate_data_integrity()
        return jsonify(integrity_results), 200
        
    except Exception as e:
        logger.error(f"Error validating data integrity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/cleanup', methods=['POST'])
def cleanup_data():
    """Clean up old data."""
    try:
        data = request.get_json() or {}
        days_to_keep = data.get('days_to_keep', 30)
        
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        success = data_manager.cleanup_old_data(days_to_keep)
        return jsonify({
            'message': 'Data cleanup completed',
            'results': success
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/info', methods=['GET'])
def get_data_info():
    """Get data directory information."""
    try:
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        info = data_manager.get_data_directory_info()
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/history', methods=['GET'])
def get_historical_data():
    """Get historical data for the app startup."""
    try:
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        # Get recent sessions
        sessions = data_manager.get_all_sessions(limit=10)
        
        # Get recent chat logs
        chat_logs = data_manager.get_chat_logs(limit=50)
        
        # Get recent agent activity
        agent_activity = data_manager.get_agent_activity(limit=30)
        
        # Get recent meetings
        meetings = data_manager.get_meetings(limit=10)
        
        # Get active sessions
        active_sessions = data_manager.get_active_sessions()
        
        return jsonify({
            'sessions': sessions,
            'chat_logs': chat_logs,
            'agent_activity': agent_activity,
            'meetings': meetings,
            'active_sessions': active_sessions,
            'total_sessions': len(sessions),
            'total_chat_logs': len(chat_logs),
            'total_agent_activities': len(agent_activity),
            'total_meetings': len(meetings)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/migrate', methods=['POST'])
def migrate_data():
    """Migrate existing data to new structure."""
    try:
        if not data_manager:
            return jsonify({'error': 'Data manager not initialized'}), 500
        
        # Get current directory (web_ui)
        current_dir = Path(__file__).parent
        
        # Create migration instance
        migration = DataMigration(str(current_dir), data_manager)
        
        # Run migration
        results = migration.run_full_migration()
        
        return jsonify({
            'message': 'Data migration completed',
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error migrating data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent-statistics', methods=['GET'])
def get_agent_statistics_endpoint():
    """Get detailed agent statistics."""
    try:
        stats = get_agent_statistics()
        
        # Add additional details if research framework is available
        if research_framework and hasattr(research_framework, 'agent_marketplace'):
            marketplace = research_framework.agent_marketplace
            
            # Get agent details
            agent_details = []
            for agent_id, agent in marketplace.agent_registry.items():
                if hasattr(agent, 'performance_metrics'):
                    agent_details.append({
                        'agent_id': agent_id,
                        'role': getattr(agent, 'role', 'Unknown'),
                        'expertise': getattr(agent, 'expertise', []),
                        'quality_score': agent.performance_metrics.get('average_quality_score', 0.0),
                        'success_rate': agent.performance_metrics.get('success_rate', 0.0),
                        'total_tasks': agent.performance_metrics.get('total_tasks', 0),
                        'is_hired': agent_id in marketplace.hired_agents
                    })
            
            stats['agent_details'] = agent_details
            stats['hired_agents'] = len(marketplace.hired_agents)
            stats['available_agents'] = len(marketplace.available_agents)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting agent statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/<agent_id>/performance', methods=['GET'])
def get_agent_performance(agent_id):
    """Get detailed performance information for a specific agent."""
    try:
        if not research_framework or not hasattr(research_framework, 'agent_marketplace'):
            return jsonify({'error': 'Research framework not available'}), 404
        
        marketplace = research_framework.agent_marketplace
        agent = marketplace.get_agent_by_id(agent_id)
        
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404
        
        # Get performance metrics
        performance = {
            'agent_id': agent_id,
            'role': getattr(agent, 'role', 'Unknown'),
            'expertise': getattr(agent, 'expertise', []),
            'is_hired': agent_id in marketplace.hired_agents,
            'is_available': agent_id in marketplace.available_agents,
            'performance_metrics': getattr(agent, 'performance_metrics', {}),
            'current_task': getattr(agent, 'current_task', None),
            'is_active': getattr(agent, 'is_active', lambda: False)()
        }
        
        # Add conversation history if available
        if hasattr(agent, 'get_conversation_history'):
            performance['conversation_history'] = agent.get_conversation_history(limit=50)
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/critic/history', methods=['GET'])
def get_critic_history():
    """Get scientific critic's critique history and critical issues."""
    try:
        if not research_framework or not hasattr(research_framework, 'scientific_critic'):
            return jsonify({'error': 'Scientific critic not available'}), 404
        
        critic = research_framework.scientific_critic
        
        if not hasattr(critic, 'critique_history'):
            return jsonify({'error': 'Critique history not available'}), 404
        
        # Get critique history
        critique_history = []
        for critique in critic.critique_history:
            critique_history.append({
                'timestamp': critique.get('timestamp', 0),
                'quality_score': critique.get('quality_score', 0.0),
                'critical_issues': critique.get('critical_issues', []),
                'feedback': critique.get('feedback', ''),
                'recommendations': critique.get('recommendations', [])
            })
        
        # Calculate summary statistics
        total_critiques = len(critique_history)
        critical_issues_count = sum(1 for c in critique_history if c['critical_issues'])
        avg_quality_score = sum(c['quality_score'] for c in critique_history) / max(1, total_critiques)
        
        return jsonify({
            'total_critiques': total_critiques,
            'critical_issues_count': critical_issues_count,
            'avg_quality_score': round(avg_quality_score, 2),
            'critique_history': critique_history
        })
        
    except Exception as e:
        logger.error(f"Error getting critic history: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    try:
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
    except Exception as e:
        logger.error(f"Error in handle_connect: {e}")
        emit('error', {'message': 'Connection error'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    try:
        client_id = request.sid
        if client_id in active_connections:
            del active_connections[client_id]
        
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in handle_disconnect: {e}")

@socketio.on('join_session')
def handle_join_session(data):
    """Handle client joining a session room."""
    try:
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            emit('joined_session', {'session_id': session_id})
    except Exception as e:
        logger.error(f"Error in handle_join_session: {e}")
        emit('error', {'message': 'Failed to join session'})

@socketio.on('leave_session')
def handle_leave_session(data):
    """Handle client leaving a session room."""
    try:
        session_id = data.get('session_id')
        if session_id:
            leave_room(session_id)
            emit('left_session', {'session_id': session_id})
    except Exception as e:
        logger.error(f"Error in handle_leave_session: {e}")
        emit('error', {'message': 'Failed to leave session'})

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
    # Initialize data manager
    data_manager = DataManager()
    # Initialize system
    load_system_config()
    initialize_framework()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitoring_thread)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run the app
    logger.info("Starting AI Research Lab Web Interface...")
    logger.info(f"Data directory: {data_manager.base_dir}")
    
    # Configure debug and allow_unsafe_werkzeug based on environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    allow_unsafe_werkzeug = os.environ.get('ALLOW_UNSAFE_WERKZEUG', '0') == '1'
    
    try:
        # Use threading mode for WebSocket support
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=debug_mode,
            allow_unsafe_werkzeug=allow_unsafe_werkzeug,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start Socket.IO server: {e}")
        # Fallback to regular Flask run without debug mode to avoid environment issues
        app.run(host='0.0.0.0', port=5000, debug=False)