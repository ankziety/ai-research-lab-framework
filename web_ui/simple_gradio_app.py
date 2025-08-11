#!/usr/bin/env python3
"""
Simple AI Research Lab Gradio Interface

A streamlined Gradio-based interface for the AI Research Lab framework.
"""

import os
import sys
import json
import time
import threading
import logging
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

class SimpleAIResearchLabGradio:
    """Simplified Gradio interface for the AI Research Lab framework."""
    
    def __init__(self):
        self.research_framework: Optional[MultiAgentResearchFramework] = None
        self.data_manager: Optional[DataManager] = None
        self.current_session: Optional[Dict[str, Any]] = None
        self.system_config: Dict[str, Any] = {}
        self.is_research_active = False
        self.research_thread: Optional[threading.Thread] = None
        
        # Initialize components
        self.initialize_framework()
        self.initialize_data_manager()
        self.load_config()
        
    def initialize_framework(self):
        """Initialize the research framework."""
        try:
            # Basic config for now - will be enhanced with settings
            config = {
                'enable_mock_responses': True,
                'enable_free_search': True,
                'max_literature_results': 10,
                'default_llm_provider': 'openai',
                'default_model': 'gpt-4'
            }
            
            self.research_framework = create_framework(config)
            logger.info("Research framework initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing framework: {e}")
    
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
                'max_agents': 8,
                'auto_save': True,
                'notifications': True
            },
            'framework': {
                'enable_mock_responses': True,
                'enable_free_search': True,
                'max_literature_results': 10
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.system_config = self._deep_merge(default_config, loaded_config)
            else:
                self.system_config = default_config
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.system_config = default_config
    
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
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_file, 'w') as f:
                json.dump(self.system_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def chat_with_research_lab(self, message: str, history: List[List[str]], 
                              research_mode: bool = False) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Main chat function for the Gradio interface."""
        if not message.strip():
            return history, "", {}
        
        # Add user message to history
        history.append([message, None])
        
        try:
            if research_mode:
                return self._start_research_session(message, history)
            else:
                return self._handle_regular_chat(message, history)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = f"Sorry, I encountered an error: {str(e)}"
            history[-1][1] = error_response
            return history, error_response, {}
    
    def _start_research_session(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Start a research session."""
        try:
            def research_worker():
                try:
                    # Start research using the framework
                    if self.research_framework:
                        result = self.research_framework.conduct_research(
                            research_question=message,
                            constraints={
                                'max_agents': 6,
                                'budget': 50000,
                                'timeline': 12
                            }
                        )
                        
                        # Update the last message with research results
                        if history and len(history) > 0:
                            history[-1][1] = f"Research completed! Here are the key findings:\n\n{result.get('summary', 'Research in progress...')}"
                    else:
                        history[-1][1] = "Research framework not available. Please check the system configuration."
                except Exception as e:
                    logger.error(f"Research error: {e}")
                    if history and len(history) > 0:
                        history[-1][1] = f"Research error: {str(e)}"
            
            # Start research in background thread
            self.research_thread = threading.Thread(target=research_worker)
            self.research_thread.daemon = True
            self.research_thread.start()
            
            # Return immediate response
            response = "🚀 Research session started! I'm gathering information and analyzing your question. This may take a few moments..."
            history[-1][1] = response
            
            return history, response, {'status': 'research_started'}
            
        except Exception as e:
            logger.error(f"Error starting research: {e}")
            error_response = f"Sorry, I couldn't start the research session: {str(e)}"
            history[-1][1] = error_response
            return history, error_response, {}
    
    def _continue_research_session(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Continue an existing research session."""
        response = "Continuing research session..."
        history[-1][1] = response
        return history, response, {}
    
    def _handle_regular_chat(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Handle regular chat messages."""
        # Simple response for now
        response = f"I understand you said: '{message}'. This is a regular chat response. Enable research mode for more advanced capabilities."
        history[-1][1] = response
        return history, response, {}
    
    def get_research_status(self) -> Dict[str, Any]:
        """Get current research status."""
        return {
            'is_active': self.is_research_active,
            'current_phase': 'initialization',
            'progress': 0,
            'agents_working': 0,
            'estimated_completion': 'Unknown'
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        if self.research_framework:
            try:
                agents = self.research_framework.get_agents()
                return {
                    'total_agents': len(agents) if agents else 0,
                    'active_agents': len([a for a in agents if a.get('status') == 'active']) if agents else 0,
                    'agent_types': list(set([a.get('type', 'unknown') for a in agents])) if agents else [],
                    'system_status': 'operational'
                }
            except Exception as e:
                logger.error(f"Error getting agent statistics: {e}")
                return {
                    'total_agents': 0,
                    'active_agents': 0,
                    'agent_types': [],
                    'system_status': 'error'
                }
        else:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'agent_types': [],
                'system_status': 'not_initialized'
            }
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="AI Research Lab - Chat Interface",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            .chat-container {
                height: 600px;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🤖 AI Research Lab - Chat Interface
            
            Welcome to the AI Research Lab! Chat with your AI research team and start research projects.
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Chat with AI Research Lab",
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
                    
                    # Agent Statistics
                    gr.Markdown("### Agent Statistics")
                    agent_stats = gr.JSON(value=self.get_agent_statistics())
            
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
            
            start_research_btn.click(
                quick_start_research,
                outputs=[msg, status_display]
            )
            
            def update_agent_stats():
                return self.get_agent_statistics()
            
            view_agents_btn.click(
                update_agent_stats,
                outputs=[agent_stats]
            )
            
            def check_results():
                status = self.get_research_status()
                return f"Research Status: {'Active' if status['is_active'] else 'Inactive'}\nProgress: {status['progress']}%"
            
            check_results_btn.click(
                check_results,
                outputs=[status_display]
            )
        
        return interface

def main():
    """Main function to run the Gradio interface."""
    # Create the interface
    app = SimpleAIResearchLabGradio()
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
