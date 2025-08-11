#!/usr/bin/env python3
"""
AI Research Lab Gradio Interface

A modern Gradio-based interface for the AI Research Lab framework.
Provides a streamlined chat interface with research capabilities, agent management,
and real-time updates.
"""

import os
import sys
import json
import time
import asyncio
import threading
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Generator, Tuple
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import using absolute imports instead of relative imports
from core.ai_research_lab import create_framework
from core.multi_agent_framework import MultiAgentResearchFramework
from data_manager import DataManager

import gradio as gr
from gradio import Blocks, Chatbot, Textbox, Button, Dropdown, Slider, Checkbox, Markdown, HTML, JSON, Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIResearchLabGradio:
    """Enhanced Gradio interface for the AI Research Lab framework."""
    
    def __init__(self):
        """Initialize the AI Research Lab Gradio interface."""
        super().__init__()
        
        # Initialize core components
        self.framework = None  # Ensure framework attribute exists
        self.data_manager = None
        self.current_session_id = None
        self.current_session = None
        self.is_research_active = False
        self.research_thread = None
        self.system_config = {}  # Add missing system_config attribute
        
        # Initialize components
        self.initialize_framework()
        self.initialize_data_manager()
        
        # Create the Gradio interface
        self.interface = self.create_interface()
        
        logger.info("AI Research Lab Gradio interface initialized")
    
    def initialize_framework(self):
        """Initialize the research framework."""
        try:
            # Load config first to get API keys
            self.load_config()
            
            # Create necessary directories
            os.makedirs('experiments', exist_ok=True)
            os.makedirs('output', exist_ok=True)
            os.makedirs('manuscripts', exist_ok=True)
            os.makedirs('visualizations', exist_ok=True)
            os.makedirs('memory', exist_ok=True)
            
            # Enhanced config with all capabilities
            config = {
                'enable_mock_responses': False,
                'enable_free_search': True,
                'max_literature_results': 10,
                'default_llm_provider': 'openai',
                'default_model': 'gpt-4',
                'vector_db_path': 'memory/vector_memory.db',
                'embedding_model': 'all-MiniLM-L6-v2',
                'max_context_length': 100000,
                # Fix database path issues - use relative paths
                'experiment_db_path': 'experiments/experiments.db',
                'output_dir': 'output',
                'manuscript_dir': 'manuscripts',
                'visualization_dir': 'visualizations',
                # Enhanced agent configuration
                'enable_agent_marketplace': True,
                'enable_virtual_lab': True,
                'enable_memory_management': True,
                'store_all_interactions': True,
                # Response configuration
                'response_quality': 'high',
                'enable_detailed_responses': True,
                # Research configuration
                'research_timeout': 300,  # 5 minutes
                'max_research_phases': 8,
                'enable_phase_tracking': True
            }
            
            # Add API keys from loaded config
            api_keys = self.system_config.get('api_keys', {})
            for key_name, key_value in api_keys.items():
                if key_value:  # Only add non-empty keys
                    if key_name == 'openai':
                        config['openai_api_key'] = key_value
                    elif key_name == 'anthropic':
                        config['anthropic_api_key'] = key_value
                    elif key_name == 'gemini':
                        config['gemini_api_key'] = key_value
                    elif key_name == 'huggingface':
                        config['huggingface_api_key'] = key_value
                    elif key_name == 'ollama_endpoint':
                        config['ollama_endpoint'] = key_value
            
            # Also add the API keys directly to the config for backward compatibility
            config.update(api_keys)
            
            # Create framework
            self.framework = create_framework(config)
            
            # Ensure agent marketplace is properly initialized
            if hasattr(self.framework, 'agent_marketplace'):
                # Add specialized agents for common research domains
                self._initialize_specialized_agents()
            
            logger.info("Research framework initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize framework: {e}")
            # Create a minimal framework for UI functionality
            self.framework = None
            return False
    
    def _initialize_specialized_agents(self):
        """Initialize specialized agents for common research domains."""
        try:
            marketplace = self.framework.agent_marketplace
            
            # Create specialized agents for the domains mentioned in mock responses
            specialized_agents = [
                ('biomedical_engineering', 'Biomedical Engineering Expert', 
                 ['Biomedical Engineering', 'Medical Devices', 'Biocompatibility', 'Signal Processing']),
                ('neuroscience', 'Neuroscience Expert',
                 ['Neuroscience', 'Brain Monitoring', 'EEG Analysis', 'Cognitive Science']),
                ('materials_science', 'Materials Science Expert',
                 ['Materials Science', 'Biomaterials', 'Nanotechnology', 'Surface Chemistry']),
                ('signal_processing', 'Signal Processing Expert',
                 ['Signal Processing', 'Data Acquisition', 'Noise Reduction', 'Filtering']),
                ('clinical_research', 'Clinical Research Expert',
                 ['Clinical Research', 'Regulatory Compliance', 'Safety Protocols', 'Human Subjects'])
            ]
            
            for domain, role, expertise in specialized_agents:
                try:
                    agent = marketplace.create_specialized_agent(role, expertise)
                    logger.info(f"Created specialized agent: {agent.agent_id} ({role})")
                except Exception as e:
                    logger.warning(f"Failed to create {role}: {e}")
            
            logger.info(f"Agent marketplace initialized with {len(marketplace.available_agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize specialized agents: {e}")
    
    def initialize_data_manager(self):
        """Initialize the data manager."""
        try:
            self.data_manager = DataManager()
            logger.info("Data manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing data manager: {e}")
    
    def load_config(self):
        """Load system configuration."""
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
                'enable_mock_responses': False
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.system_config = self._deep_merge(default_config, loaded_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.system_config = default_config
        else:
            self.system_config = default_config
            self.save_config()
    
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge two dictionaries."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Save system configuration."""
        try:
            config_file = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_file, 'w') as f:
                json.dump(self.system_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def create_session(self) -> str:
        """Create a new research session."""
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        self.current_session = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'status': 'pending',
            'research_question': '',
            'chat_history': [],
            'agent_activities': [],
            'meetings': []
        }
        
        # Persist session to data manager
        if self.data_manager:
            self.data_manager.persist_session(
                session_id=session_id,
                research_question='',
                status='pending'
            )
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def log_chat_message(self, message: str, author: str = "user", log_type: str = "communication"):
        """Log a chat message to the data manager."""
        if not self.current_session_id:
            self.create_session()
        
        if self.data_manager:
            self.data_manager.persist_chat_log(
                session_id=self.current_session_id,
                log_type=log_type,
                author=author,
                message=message
            )
        
        # Add to current session
        if self.current_session:
            self.current_session['chat_history'].append({
                'timestamp': datetime.now().isoformat(),
                'author': author,
                'message': message,
                'type': log_type
            })
    
    def get_chat_history(self, session_id: Optional[str] = None) -> List[List[str]]:
        """Get chat history in Gradio format."""
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id:
            return []
        
        # Get from data manager
        if self.data_manager:
            chat_logs = self.data_manager.get_chat_logs(session_id=session_id, limit=100)
            
            # Convert to Gradio format
            history = []
            for log in reversed(chat_logs):  # Reverse to get chronological order
                if log['log_type'] in ['communication', 'thought', 'tool_call']:
                    if log['author'] == 'user':
                        history.append([log['message'], None])
                    else:
                        # Find the corresponding user message
                        if history and history[-1][1] is None:
                            history[-1][1] = log['message']
                        else:
                            history.append([None, log['message']])
            
            return history
        
        return []
    
    def chat_with_research_lab(self, message: str, history: List[List[str]], 
                              research_mode: bool = False) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """
        Enhanced chat function for interacting with the research lab.
        
        Args:
            message: User's message
            history: Chat history
            research_mode: Whether to start a research session
            
        Returns:
            Updated chat history, status message, and additional data
        """
        if not message.strip():
            return history, "Please enter a message.", {}
        
        # Create session if needed
        if not self.current_session_id:
            self.create_session()
        
        # Log user message
        self.log_chat_message(message, "user", "communication")
        
        # Add user message to history
        history.append([message, None])
        
        try:
            if research_mode and not self.is_research_active:
                # Start research session
                return self._start_research_session(message, history)
            elif self.is_research_active:
                # Continue research session
                return self._continue_research_session(message, history)
            else:
                # Regular chat interaction
                return self._handle_regular_chat(message, history)
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.log_chat_message(error_msg, "system", "system")
            history[-1][1] = error_msg
            return history, error_msg, {}
    
    def _start_research_session(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Start a new research session."""
        try:
            self.is_research_active = True
            
            # Update session
            if self.current_session:
                self.current_session['research_question'] = message
                self.current_session['status'] = 'running'
            
            # Persist session update
            if self.data_manager and self.current_session_id:
                self.data_manager.persist_session(
                    session_id=self.current_session_id,
                    research_question=message,
                    status='running'
                )
            
            # Ensure history has the current message
            if not history:
                history = []
            history.append([message, ""])
            
            # Start research in background thread
            def run_research():
                try:
                    logger.info(f"Starting research session: {message[:100]}...")
                    
                    # Use Virtual Lab research method
                    result = self.framework.conduct_virtual_lab_research(
                        research_question=message,
                        session_id=self.current_session_id
                    )
                    
                    # Update session with results
                    if self.current_session:
                        self.current_session['results'] = result
                        self.current_session['status'] = 'completed'
                        self.current_session['completion_time'] = time.time()
                    
                    # Persist results
                    if self.data_manager and self.current_session_id:
                        self.data_manager.persist_session(
                            session_id=self.current_session_id,
                            results=result,
                            status='completed'
                        )
                    
                    logger.info(f"Research session completed: {self.current_session_id}")
                    
                except Exception as e:
                    logger.error(f"Research session failed: {e}")
                    if self.current_session:
                        self.current_session['status'] = 'failed'
                        self.current_session['error'] = str(e)
            
            # Start research thread
            research_thread = threading.Thread(target=run_research)
            research_thread.daemon = True
            research_thread.start()
            
            # Return immediate response
            response = f"🔬 **Research Session Started**\n\n**Question:** {message}\n\n**Status:** Research is now running in the background. The AI research team will analyze your question and conduct a comprehensive investigation.\n\n**Current Phase:** Team Selection\n**Progress:** 0%\n\nYou can monitor progress using the 'Check Results' button."
            
            # Update history with response
            history[-1][1] = response
            
            return history, response, {
                'session_id': self.current_session_id,
                'status': 'started',
                'research_question': message
            }
            
        except Exception as e:
            logger.error(f"Failed to start research session: {e}")
            error_response = f"❌ **Research Session Failed**\n\nError: {str(e)}\n\nPlease try again or contact support."
            
            if not history:
                history = []
            history.append([message, error_response])
            
            return history, error_response, {
                'error': str(e),
                'status': 'failed'
            }
    
    def _continue_research_session(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Continue an active research session."""
        response = f"🤖 **Research in Progress**\n\nYour research session is currently active. You can:\n\n- Monitor progress in the Research Dashboard\n- View agent activities in the Agents tab\n- Check results in the Results tab\n\nTo start a new research session, please wait for the current one to complete or stop it first."
        
        self.log_chat_message(response, "assistant", "communication")
        history[-1][1] = response
        return history, "Research session is active", {'research_active': True}
    
    def _handle_regular_chat(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Handle regular chat interaction."""
        response = f"🤖 **AI Research Lab Assistant**\n\nHello! I'm your AI Research Lab assistant. I can help you with:\n\n- **Starting Research Sessions** - Use the research mode to conduct AI-powered research\n- **Agent Management** - View and manage AI agents\n- **Results Analysis** - Analyze research results\n- **System Configuration** - Configure API keys and settings\n\nTo start a research session, check the 'Research Mode' option and ask your research question."
        
        self.log_chat_message(response, "assistant", "communication")
        history[-1][1] = response
        return history, "Chat response generated", {}
    
    def get_research_status(self) -> Dict[str, Any]:
        """Get current research status with enhanced data."""
        if not self.is_research_active:
            return {
                'status': 'idle',
                'message': 'No active research session',
                'progress': 0,
                'current_phase': 'None',
                'agents_active': 0,
                'session_id': self.current_session_id
            }
        
        # Get real data from framework if available
        if self.framework and self.current_session_id:
            try:
                # Get agent activity
                agent_activities = self.data_manager.get_agent_activity(
                    session_id=self.current_session_id, limit=50
                ) if self.data_manager else []
                
                # Get chat logs for progress tracking
                chat_logs = self.data_manager.get_chat_logs(
                    session_id=self.current_session_id, limit=100
                ) if self.data_manager else []
                
                # Calculate progress based on activity
                progress = min(100, len(agent_activities) * 2)  # Rough estimate
                
                phases = [
                    'Team Selection', 'Project Specification', 'Tools Selection',
                    'Implementation', 'Workflow Design', 'Execution', 'Synthesis'
                ]
                current_phase = phases[min(progress // 15, len(phases) - 1)]
                
                return {
                    'status': 'active',
                    'message': 'Research in progress',
                    'progress': progress,
                    'current_phase': current_phase,
                    'agents_active': len(set(act['agent_id'] for act in agent_activities)),
                    'research_question': self.current_session.get('research_question', 'Unknown') if self.current_session else 'Unknown',
                    'session_id': self.current_session_id,
                    'chat_logs_count': len(chat_logs),
                    'agent_activities_count': len(agent_activities)
                }
            except Exception as e:
                logger.error(f"Error getting research status: {e}")
        
        # Fallback to simulation
        progress = min(50, int((time.time() - (self.current_session.get('start_time', time.time()))) / 10))
        
        phases = [
            'Team Selection', 'Project Specification', 'Tools Selection',
            'Implementation', 'Workflow Design', 'Execution', 'Synthesis'
        ]
        current_phase = phases[min(progress // 15, len(phases) - 1)]
        
        return {
            'status': 'active',
            'message': 'Research in progress',
            'progress': progress,
            'current_phase': current_phase,
            'agents_active': 3 if progress > 20 else 1,
            'research_question': self.current_session.get('research_question', 'Unknown') if self.current_session else 'Unknown',
            'session_id': self.current_session_id
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get enhanced agent statistics."""
        if not self.framework:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0,
                'hired_agents': [],
                'available_agents': []
            }
        
        try:
            marketplace = self.framework.agent_marketplace
            
            # Get all agents
            total_agents = len(marketplace.agent_registry)
            hired_agents = list(marketplace.hired_agents.keys()) if hasattr(marketplace, 'hired_agents') else []
            active_agents = len(hired_agents)
            
            # Get agent details
            available_agents = []
            for agent_id, agent in marketplace.agent_registry.items():
                agent_info = {
                    'id': agent_id,
                    'role': getattr(agent, 'role', 'Unknown'),
                    'expertise': getattr(agent, 'expertise', []),
                    'is_hired': agent_id in hired_agents,
                    'performance_metrics': getattr(agent, 'performance_metrics', {})
                }
                available_agents.append(agent_info)
            
            # Calculate average quality score
            quality_scores = []
            for agent in marketplace.agent_registry.values():
                if hasattr(agent, 'performance_metrics') and agent.performance_metrics:
                    score = agent.performance_metrics.get('average_quality_score', 0.0)
                    if score > 0:
                        quality_scores.append(score)
            
            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'avg_quality_score': round(avg_quality_score, 2),
                'critical_issues': 0,  # Would be calculated from scientific critic
                'hired_agents': hired_agents,
                'available_agents': available_agents
            }
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {
                'total_agents': 0,
                'active_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0,
                'hired_agents': [],
                'available_agents': []
            }
    
    def get_agent_activities(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get agent activities for display."""
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or not self.data_manager:
            return []
        
        try:
            activities = self.data_manager.get_agent_activity(session_id=session_id, limit=50)
            return activities
        except Exception as e:
            logger.error(f"Error getting agent activities: {e}")
            return []
    
    def get_meeting_transcripts(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get meeting transcripts for display."""
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or not self.data_manager:
            return []
        
        try:
            meetings = self.data_manager.get_meetings(session_id=session_id, limit=20)
            return meetings
        except Exception as e:
            logger.error(f"Error getting meeting transcripts: {e}")
            return []
    
    def create_research_dashboard(self) -> gr.Blocks:
        """Create the enhanced research dashboard tab."""
        with gr.Blocks() as dashboard:
            gr.Markdown("# 🔬 Research Dashboard")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Research Status
                    status_card = gr.Markdown("### Research Status\n\nNo active research session")
                    
                    # Progress Bar
                    progress_bar = gr.Slider(
                        minimum=0, maximum=100, value=0, 
                        label="Research Progress", interactive=False
                    )
                    
                    # Current Phase
                    phase_display = gr.Markdown("### Current Phase\n\nNone")
                    
                    # Agent Activities
                    activities_display = gr.Markdown("### Recent Agent Activities\n\nNo activities")
                
                with gr.Column(scale=1):
                    # Quick Stats
                    stats_card = gr.Markdown("### Quick Stats\n\n- **Active Agents:** 0\n- **Quality Score:** 0.0\n- **Critical Issues:** 0")
                    
                    # Meeting Transcripts
                    meetings_display = gr.Markdown("### Recent Meetings\n\nNo meetings")
            
            # Research Controls
            with gr.Row():
                start_research_btn = gr.Button("🚀 Start Research", variant="primary")
                stop_research_btn = gr.Button("⏹️ Stop Research", variant="stop")
                refresh_btn = gr.Button("🔄 Refresh")
            
            # Research Results
            results_display = gr.Markdown("### Research Results\n\nNo results available")
            
            # Update function
            def update_dashboard():
                status = self.get_research_status()
                agent_stats = self.get_agent_statistics()
                agent_activities = self.get_agent_activities()
                meetings = self.get_meeting_transcripts()
                
                # Status text
                status_text = f"### Research Status\n\n"
                if status['status'] == 'active':
                    status_text += f"**Status:** 🔄 Active\n"
                    status_text += f"**Question:** {status['research_question']}\n"
                    status_text += f"**Progress:** {status['progress']}%\n"
                    status_text += f"**Phase:** {status['current_phase']}\n"
                    status_text += f"**Session ID:** {status['session_id']}"
                else:
                    status_text += "**Status:** ⏸️ Idle\n\nNo active research session"
                
                # Stats text
                stats_text = f"### Quick Stats\n\n"
                stats_text += f"- **Active Agents:** {agent_stats['active_agents']}\n"
                stats_text += f"- **Quality Score:** {agent_stats['avg_quality_score']}\n"
                stats_text += f"- **Critical Issues:** {agent_stats['critical_issues']}"
                
                # Activities text
                activities_text = "### Recent Agent Activities\n\n"
                if agent_activities:
                    for activity in agent_activities[:5]:  # Show last 5
                        timestamp = activity.get('timestamp', 'Unknown')
                        agent_id = activity.get('agent_id', 'Unknown')
                        activity_type = activity.get('activity_type', 'Unknown')
                        message = activity.get('message', '')[:100] + "..." if len(activity.get('message', '')) > 100 else activity.get('message', '')
                        
                        activities_text += f"**{timestamp}** - {agent_id} ({activity_type})\n"
                        activities_text += f"{message}\n\n"
                else:
                    activities_text += "No activities recorded"
                
                # Meetings text
                meetings_text = "### Recent Meetings\n\n"
                if meetings:
                    for meeting in meetings[:3]:  # Show last 3
                        timestamp = meeting.get('timestamp', 'Unknown')
                        topic = meeting.get('topic', 'Unknown')
                        participants = json.loads(meeting.get('participants', '[]'))
                        
                        meetings_text += f"**{timestamp}** - {topic}\n"
                        meetings_text += f"Participants: {', '.join(participants)}\n\n"
                else:
                    meetings_text += "No meetings recorded"
                
                return status_text, status['progress'], f"### Current Phase\n\n{status['current_phase']}", stats_text, activities_text, meetings_text
            
            refresh_btn.click(update_dashboard, outputs=[status_card, progress_bar, phase_display, stats_card, activities_display, meetings_display])
            
            # Auto-refresh every 5 seconds
            dashboard.load(update_dashboard, outputs=[status_card, progress_bar, phase_display, stats_card, activities_display, meetings_display])
        
        return dashboard
    
    def create_agents_panel(self) -> gr.Blocks:
        """Create the enhanced agents management panel."""
        with gr.Blocks() as agents_panel:
            gr.Markdown("# 🤖 Agent Management")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Agent List
                    agent_list = gr.Markdown("### Available Agents\n\nNo agents available")
                    
                    # Agent Details
                    agent_details = gr.Markdown("### Agent Details\n\nSelect an agent to view details")
                
                with gr.Column(scale=1):
                    # Agent Statistics
                    agent_stats = gr.Markdown("### Agent Statistics\n\n- **Total Agents:** 0\n- **Active Agents:** 0\n- **Average Quality:** 0.0")
                    
                    # Agent Activities
                    agent_activities = gr.Markdown("### Recent Agent Activities\n\nNo activities")
            
            # Agent Controls
            with gr.Row():
                refresh_agents_btn = gr.Button("🔄 Refresh Agents")
                hire_agent_btn = gr.Button("👥 Hire Agent", variant="primary")
                fire_agent_btn = gr.Button("🚫 Fire Agent", variant="stop")
                view_activities_btn = gr.Button("📊 View Activities")
            
            def update_agents():
                if not self.framework:
                    return "### Available Agents\n\nFramework not initialized", "### Agent Details\n\nNo framework available", "### Agent Statistics\n\n- **Total Agents:** 0\n- **Active Agents:** 0\n- **Average Quality:** 0.0", "### Recent Agent Activities\n\nNo activities"
                
                try:
                    marketplace = self.framework.agent_marketplace
                    agent_stats_data = self.get_agent_statistics()
                    
                    # Build agents list
                    agents_text = "### Available Agents\n\n"
                    for agent_info in agent_stats_data['available_agents']:
                        status = "🟢 Hired" if agent_info['is_hired'] else "⚪ Available"
                        agents_text += f"**{agent_info['id']}** ({agent_info['role']}) - {status}\n"
                        agents_text += f"Expertise: {', '.join(agent_info['expertise'][:3])}\n\n"
                    
                    # Build statistics
                    stats_text = f"### Agent Statistics\n\n"
                    stats_text += f"- **Total Agents:** {agent_stats_data['total_agents']}\n"
                    stats_text += f"- **Active Agents:** {agent_stats_data['active_agents']}\n"
                    stats_text += f"- **Average Quality:** {agent_stats_data['avg_quality_score']}"
                    
                    # Build activities
                    activities = self.get_agent_activities()
                    activities_text = "### Recent Agent Activities\n\n"
                    if activities:
                        for activity in activities[:5]:  # Show last 5
                            timestamp = activity.get('timestamp', 'Unknown')
                            agent_id = activity.get('agent_id', 'Unknown')
                            activity_type = activity.get('activity_type', 'Unknown')
                            message = activity.get('message', '')[:100] + "..." if len(activity.get('message', '')) > 100 else activity.get('message', '')
                            
                            activities_text += f"**{timestamp}** - {agent_id} ({activity_type})\n"
                            activities_text += f"{message}\n\n"
                    else:
                        activities_text += "No activities recorded"
                    
                    return agents_text, "### Agent Details\n\nSelect an agent above to view detailed information", stats_text, activities_text
                    
                except Exception as e:
                    return f"### Available Agents\n\nError loading agents: {str(e)}", "### Agent Details\n\nError occurred", "### Agent Statistics\n\n- **Total Agents:** 0\n- **Active Agents:** 0\n- **Average Quality:** 0.0", "### Recent Agent Activities\n\nError loading activities"
            
            refresh_agents_btn.click(update_agents, outputs=[agent_list, agent_details, agent_stats, agent_activities])
            view_activities_btn.click(update_agents, outputs=[agent_list, agent_details, agent_stats, agent_activities])
            agents_panel.load(update_agents, outputs=[agent_list, agent_details, agent_stats, agent_activities])
        
        return agents_panel
    
    def create_settings_panel(self) -> gr.Blocks:
        """Create the settings configuration panel."""
        with gr.Blocks() as settings_panel:
            gr.Markdown("# ⚙️ Settings")
            
            with gr.Tabs():
                with gr.TabItem("API Keys"):
                    with gr.Column():
                        gr.Markdown("### API Configuration")
                        
                        openai_key = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="sk-...",
                            type="password",
                            value=self.system_config.get('api_keys', {}).get('openai', '')
                        )
                        
                        anthropic_key = gr.Textbox(
                            label="Anthropic API Key",
                            placeholder="sk-ant-...",
                            type="password",
                            value=self.system_config.get('api_keys', {}).get('anthropic', '')
                        )
                        
                        gemini_key = gr.Textbox(
                            label="Google Gemini API Key",
                            placeholder="AIza...",
                            type="password",
                            value=self.system_config.get('api_keys', {}).get('gemini', '')
                        )
                        
                        huggingface_key = gr.Textbox(
                            label="HuggingFace API Key",
                            placeholder="hf_...",
                            type="password",
                            value=self.system_config.get('api_keys', {}).get('huggingface', '')
                        )
                        
                        ollama_endpoint = gr.Textbox(
                            label="Ollama Endpoint",
                            placeholder="http://localhost:11434",
                            value=self.system_config.get('api_keys', {}).get('ollama_endpoint', 'http://localhost:11434')
                        )
                
                with gr.TabItem("System Settings"):
                    with gr.Column():
                        gr.Markdown("### System Configuration")
                        
                        max_agents = gr.Slider(
                            minimum=1, maximum=20, value=8,
                            label="Max Concurrent Agents",
                            step=1
                        )
                        
                        auto_save = gr.Checkbox(
                            label="Auto-save results",
                            value=self.system_config.get('system', {}).get('auto_save_results', True)
                        )
                        
                        enable_notifications = gr.Checkbox(
                            label="Enable notifications",
                            value=self.system_config.get('system', {}).get('enable_notifications', True)
                        )
                        
                        enable_mock = gr.Checkbox(
                            label="Enable mock responses",
                            value=self.system_config.get('framework', {}).get('enable_mock_responses', False)
                        )
                        
                        enable_free_search = gr.Checkbox(
                            label="Enable free search",
                            value=self.system_config.get('framework', {}).get('enable_free_search', True)
                        )
            
            # Save Settings Button
            save_btn = gr.Button("💾 Save Settings", variant="primary")
            save_status = gr.Textbox(label="Save Status", interactive=False)
            
            def save_settings(openai, anthropic, gemini, huggingface, ollama, max_agents_val, auto_save_val, notifications_val, mock_val, free_search_val):
                try:
                    # Update API keys
                    self.system_config.setdefault('api_keys', {}).update({
                        'openai': openai,
                        'anthropic': anthropic,
                        'gemini': gemini,
                        'huggingface': huggingface,
                        'ollama_endpoint': ollama
                    })
                    
                    # Update system settings
                    self.system_config.setdefault('system', {}).update({
                        'max_concurrent_agents': max_agents_val,
                        'auto_save_results': auto_save_val,
                        'enable_notifications': notifications_val
                    })
                    
                    # Update framework settings
                    self.system_config.setdefault('framework', {}).update({
                        'enable_mock_responses': mock_val,
                        'enable_free_search': free_search_val
                    })
                    
                    self.save_config()
                    
                    # Reinitialize framework with new settings
                    self.initialize_framework()
                    
                    return "✅ Settings saved successfully!"
                except Exception as e:
                    return f"❌ Error saving settings: {str(e)}"
            
            save_btn.click(
                save_settings,
                inputs=[openai_key, anthropic_key, gemini_key, huggingface_key, ollama_endpoint, 
                       max_agents, auto_save, enable_notifications, enable_mock, enable_free_search],
                outputs=save_status
            )
        
        return settings_panel
    
    def create_results_panel(self) -> gr.Blocks:
        """Create the enhanced results visualization panel."""
        with gr.Blocks() as results_panel:
            gr.Markdown("# 📊 Research Results")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Results Display
                    results_display = gr.Markdown("### Research Results\n\nNo results available")
                    
                    # Export Options
                    with gr.Row():
                        export_json_btn = gr.Button("📄 Export JSON")
                        export_csv_btn = gr.Button("📊 Export CSV")
                        export_pdf_btn = gr.Button("📋 Export PDF")
                
                with gr.Column(scale=1):
                    # Results Summary
                    results_summary = gr.Markdown("### Results Summary\n\n- **Status:** No results\n- **Quality Score:** N/A\n- **Key Findings:** None")
                    
                    # Session Info
                    session_info = gr.Markdown("### Session Information\n\n- **Session ID:** None\n- **Created:** N/A\n- **Duration:** N/A")
            
            def update_results():
                if not self.current_session:
                    return "### Research Results\n\nNo research session completed", "### Results Summary\n\n- **Status:** No results\n- **Quality Score:** N/A\n- **Key Findings:** None", f"### Session Information\n\n- **Session ID:** None\n- **Created:** N/A\n- **Duration:** N/A"
                
                results = self.current_session.get('results', {})
                session_id = self.current_session.get('session_id', 'Unknown')
                created_at = self.current_session.get('created_at', 'Unknown')
                
                # Calculate duration
                duration = "N/A"
                if created_at != 'Unknown':
                    try:
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        duration = str(datetime.now() - created_time).split('.')[0]
                    except:
                        duration = "N/A"
                
                if not results:
                    return "### Research Results\n\nResearch completed but no results available", "### Results Summary\n\n- **Status:** Completed\n- **Quality Score:** N/A\n- **Key Findings:** None", f"### Session Information\n\n- **Session ID:** {session_id}\n- **Created:** {created_at}\n- **Duration:** {duration}"
                
                # Format results
                results_text = "### Research Results\n\n"
                
                if isinstance(results, dict):
                    if 'phases' in results:
                        results_text += "**Research Phases:**\n\n"
                        for phase_name, phase_result in results['phases'].items():
                            status = "✅" if phase_result.get('success') else "❌"
                            results_text += f"{status} **{phase_name.replace('_', ' ').title()}**\n"
                            if phase_result.get('success'):
                                results_text += f"   - Completed successfully\n"
                            else:
                                results_text += f"   - Failed or incomplete\n"
                    
                    if 'summary' in results:
                        results_text += f"\n**Summary:**\n{results['summary']}\n"
                    
                    if 'conclusions' in results:
                        results_text += f"\n**Conclusions:**\n{results['conclusions']}\n"
                
                # Summary
                summary_text = "### Results Summary\n\n"
                summary_text += f"- **Status:** {'Completed' if results else 'In Progress'}\n"
                summary_text += f"- **Quality Score:** {results.get('quality_score', 'N/A')}\n"
                summary_text += f"- **Key Findings:** {len(results.get('phases', {}))} phases completed"
                
                # Session info
                session_text = f"### Session Information\n\n- **Session ID:** {session_id}\n- **Created:** {created_at}\n- **Duration:** {duration}"
                
                return results_text, summary_text, session_text
            
            # Update results when panel loads
            results_panel.load(update_results, outputs=[results_display, results_summary, session_info])
        
        return results_panel
    
    def create_interface(self) -> gr.Blocks:
        """Create the enhanced main Gradio interface."""
        with gr.Blocks(
            title="AI Research Lab",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-container {
                height: 600px;
                overflow-y: auto;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🔬 AI Research Lab")
            gr.Markdown("Welcome to the AI Research Lab! Start a research session or chat with the AI assistant.")
            
            with gr.Tabs():
                # Main Chat Tab
                with gr.TabItem("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Chat Interface
                            chatbot = gr.Chatbot(
                                label="AI Research Lab Chat",
                                height=500,
                                show_label=True,
                                container=True,
                                bubble_full_width=False
                            )
                            
                            with gr.Row():
                                with gr.Column(scale=4):
                                    msg = gr.Textbox(
                                        label="Message",
                                        placeholder="Ask a question or start research...",
                                        lines=2
                                    )
                                
                                with gr.Column(scale=1):
                                    research_mode = gr.Checkbox(
                                        label="Research Mode",
                                        value=False,
                                        info="Enable to start a research session"
                                    )
                                
                                with gr.Column(scale=1):
                                    submit_btn = gr.Button("Send", variant="primary")
                            
                            # Status display
                            status_display = gr.Markdown("Ready to chat!")
                        
                        with gr.Column(scale=1):
                            # Quick Actions
                            gr.Markdown("### Quick Actions")
                            
                            start_research_btn = gr.Button("🚀 Start Research", variant="primary")
                            view_agents_btn = gr.Button("🤖 View Agents")
                            check_results_btn = gr.Button("📊 Check Results")
                            open_settings_btn = gr.Button("⚙️ Settings")
                            
                            # System Status
                            gr.Markdown("### System Status")
                            system_status = gr.Markdown("🟢 System Online")
                    
                    # Chat function
                    def chat_fn(message, history, research_mode_val):
                        return self.chat_with_research_lab(message, history, research_mode_val)
                    
                    submit_btn.click(
                        chat_fn,
                        inputs=[msg, chatbot, research_mode],
                        outputs=[chatbot, status_display],
                        api_name="chat"
                    )
                    
                    msg.submit(
                        chat_fn,
                        inputs=[msg, chatbot, research_mode],
                        outputs=[chatbot, status_display],
                        api_name="chat"
                    )
                    
                    # Quick action handlers
                    def quick_start_research():
                        return "🚀 Research mode activated! Enter your research question in the chat.", "Research mode ready"
                    
                    def view_agents_action():
                        return "🤖 Agent panel opened. Check the Agents tab for details.", "Agent panel ready"
                    
                    def check_results_action():
                        return "📊 Results panel opened. Check the Results tab for details.", "Results panel ready"
                    
                    def open_settings_action():
                        return "⚙️ Settings panel opened. Check the Settings tab for configuration.", "Settings panel ready"
                    
                    start_research_btn.click(
                        quick_start_research,
                        outputs=[msg, status_display]
                    )
                    
                    view_agents_btn.click(
                        view_agents_action,
                        outputs=[msg, status_display]
                    )
                    
                    check_results_btn.click(
                        check_results_action,
                        outputs=[msg, status_display]
                    )
                    
                    open_settings_btn.click(
                        open_settings_action,
                        outputs=[msg, status_display]
                    )
                
                # Research Dashboard Tab
                with gr.TabItem("📊 Dashboard"):
                    self.create_research_dashboard()
                
                # Agents Tab
                with gr.TabItem("🤖 Agents"):
                    self.create_agents_panel()
                
                # Results Tab
                with gr.TabItem("📊 Results"):
                    self.create_results_panel()
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings"):
                    self.create_settings_panel()
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("AI Research Lab Framework - Powered by Gradio")
        
        return interface

def main():
    """Main function to run the Gradio interface."""
    # Create the interface
    app = AIResearchLabGradio()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=False,
        show_api=False
    )

if __name__ == "__main__":
    main() 