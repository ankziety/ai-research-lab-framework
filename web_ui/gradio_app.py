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
    """Gradio interface for the AI Research Lab framework."""
    
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
    
    def chat_with_research_lab(self, message: str, history: List[List[str]], 
                              research_mode: bool = False) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """
        Main chat function for interacting with the research lab.
        
        Args:
            message: User's message
            history: Chat history
            research_mode: Whether to start a research session
            
        Returns:
            Updated chat history, status message, and additional data
        """
        if not message.strip():
            return history, "Please enter a message.", {}
        
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
            history[-1][1] = error_msg
            return history, error_msg, {}
    
    def _start_research_session(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Start a new research session."""
        try:
            self.is_research_active = True
            
            # Start research in background thread
            def research_worker():
                try:
                    # Conduct research
                    results = self.research_framework.conduct_virtual_lab_research(
                        research_question=message,
                        constraints={},
                        context={}
                    )
                    
                    # Update session
                    self.current_session = {
                        'research_question': message,
                        'results': results,
                        'start_time': time.time(),
                        'status': 'completed'
                    }
                    
                    self.is_research_active = False
                    
                except Exception as e:
                    logger.error(f"Research error: {e}")
                    self.is_research_active = False
            
            self.research_thread = threading.Thread(target=research_worker)
            self.research_thread.daemon = True
            self.research_thread.start()
            
            response = f"🔬 **Research Session Started**\n\n**Question:** {message}\n\nI'm now conducting research on this topic. The research framework will:\n\n1. **Team Selection** - Identify relevant AI agents\n2. **Project Specification** - Define research scope\n3. **Tools Selection** - Choose appropriate tools\n4. **Implementation** - Execute research plan\n5. **Workflow Design** - Create research workflow\n6. **Execution** - Run the research\n7. **Synthesis** - Compile results\n\nYou can monitor progress in the Research Dashboard tab."
            
            history[-1][1] = response
            
            return history, "Research session started successfully!", {
                'research_active': True,
                'research_question': message
            }
            
        except Exception as e:
            error_msg = f"Failed to start research session: {str(e)}"
            history[-1][1] = error_msg
            return history, error_msg, {}
    
    def _continue_research_session(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Continue an active research session."""
        response = f"🤖 **Research in Progress**\n\nYour research session is currently active. You can:\n\n- Monitor progress in the Research Dashboard\n- View agent activities in the Agents tab\n- Check results in the Results tab\n\nTo start a new research session, please wait for the current one to complete or stop it first."
        
        history[-1][1] = response
        return history, "Research session is active", {'research_active': True}
    
    def _handle_regular_chat(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str, Dict[str, Any]]:
        """Handle regular chat interaction."""
        # Simple response for now - can be enhanced with LLM integration
        response = f"🤖 **AI Research Lab Assistant**\n\nHello! I'm your AI Research Lab assistant. I can help you with:\n\n- **Starting Research Sessions** - Use the research mode to conduct AI-powered research\n- **Agent Management** - View and manage AI agents\n- **Results Analysis** - Analyze research results\n- **System Configuration** - Configure API keys and settings\n\nTo start a research session, check the 'Research Mode' option and ask your research question."
        
        history[-1][1] = response
        return history, "Chat response generated", {}
    
    def get_research_status(self) -> Dict[str, Any]:
        """Get current research status."""
        if not self.is_research_active:
            return {
                'status': 'idle',
                'message': 'No active research session',
                'progress': 0,
                'current_phase': 'None',
                'agents_active': 0
            }
        
        # Simulate progress for demo
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
            'research_question': self.current_session.get('research_question', 'Unknown')
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        if not self.research_framework:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0
            }
        
        try:
            marketplace = self.research_framework.agent_marketplace
            total_agents = len(marketplace.agent_registry)
            active_agents = len(marketplace.hired_agents) if hasattr(marketplace, 'hired_agents') else 0
            
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
                'critical_issues': 0  # Would be calculated from scientific critic
            }
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {
                'total_agents': 0,
                'active_agents': 0,
                'avg_quality_score': 0.0,
                'critical_issues': 0
            }
    
    def create_research_dashboard(self) -> gr.Blocks:
        """Create the research dashboard tab."""
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
                
                with gr.Column(scale=1):
                    # Quick Stats
                    stats_card = gr.Markdown("### Quick Stats\n\n- **Active Agents:** 0\n- **Quality Score:** 0.0\n- **Critical Issues:** 0")
            
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
                
                status_text = f"### Research Status\n\n"
                if status['status'] == 'active':
                    status_text += f"**Status:** 🔄 Active\n"
                    status_text += f"**Question:** {status['research_question']}\n"
                    status_text += f"**Progress:** {status['progress']}%\n"
                    status_text += f"**Phase:** {status['current_phase']}"
                else:
                    status_text += "**Status:** ⏸️ Idle\n\nNo active research session"
                
                stats_text = f"### Quick Stats\n\n"
                stats_text += f"- **Active Agents:** {agent_stats['active_agents']}\n"
                stats_text += f"- **Quality Score:** {agent_stats['avg_quality_score']}\n"
                stats_text += f"- **Critical Issues:** {agent_stats['critical_issues']}"
                
                return status_text, status['progress'], f"### Current Phase\n\n{status['current_phase']}", stats_text
            
            refresh_btn.click(update_dashboard, outputs=[status_card, progress_bar, phase_display, stats_card])
            
            # Auto-refresh every 5 seconds
            dashboard.load(update_dashboard, outputs=[status_card, progress_bar, phase_display, stats_card])
        
        return dashboard
    
    def create_agents_panel(self) -> gr.Blocks:
        """Create the agents management panel."""
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
            
            # Agent Controls
            with gr.Row():
                refresh_agents_btn = gr.Button("🔄 Refresh Agents")
                hire_agent_btn = gr.Button("👥 Hire Agent", variant="primary")
                fire_agent_btn = gr.Button("🚫 Fire Agent", variant="stop")
            
            def update_agents():
                if not self.research_framework:
                    return "### Available Agents\n\nFramework not initialized", "### Agent Details\n\nNo framework available", "### Agent Statistics\n\n- **Total Agents:** 0\n- **Active Agents:** 0\n- **Average Quality:** 0.0"
                
                try:
                    marketplace = self.research_framework.agent_marketplace
                    agents_text = "### Available Agents\n\n"
                    
                    for agent_id, agent in marketplace.agent_registry.items():
                        role = getattr(agent, 'role', 'Unknown')
                        expertise = getattr(agent, 'expertise', [])
                        is_hired = agent_id in getattr(marketplace, 'hired_agents', set())
                        
                        status = "🟢 Hired" if is_hired else "⚪ Available"
                        agents_text += f"**{agent_id}** ({role}) - {status}\n"
                        agents_text += f"Expertise: {', '.join(expertise[:3])}\n\n"
                    
                    agent_stats = self.get_agent_statistics()
                    stats_text = f"### Agent Statistics\n\n"
                    stats_text += f"- **Total Agents:** {agent_stats['total_agents']}\n"
                    stats_text += f"- **Active Agents:** {agent_stats['active_agents']}\n"
                    stats_text += f"- **Average Quality:** {agent_stats['avg_quality_score']}"
                    
                    return agents_text, "### Agent Details\n\nSelect an agent above to view detailed information", stats_text
                    
                except Exception as e:
                    return f"### Available Agents\n\nError loading agents: {str(e)}", "### Agent Details\n\nError occurred", "### Agent Statistics\n\n- **Total Agents:** 0\n- **Active Agents:** 0\n- **Average Quality:** 0.0"
            
            refresh_agents_btn.click(update_agents, outputs=[agent_list, agent_details, agent_stats])
            agents_panel.load(update_agents, outputs=[agent_list, agent_details, agent_stats])
        
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
                            value=self.system_config.get('framework', {}).get('enable_mock_responses', True)
                        )
                        
                        enable_free_search = gr.Checkbox(
                            label="Enable free search",
                            value=self.system_config.get('framework', {}).get('enable_free_search', True)
                        )
            
            # Save Settings Button
            save_btn = gr.Button("💾 Save Settings", variant="primary")
            
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
                outputs=gr.Textbox(label="Save Status")
            )
        
        return settings_panel
    
    def create_results_panel(self) -> gr.Blocks:
        """Create the results visualization panel."""
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
            
            def update_results():
                if not self.current_session:
                    return "### Research Results\n\nNo research session completed", "### Results Summary\n\n- **Status:** No results\n- **Quality Score:** N/A\n- **Key Findings:** None"
                
                results = self.current_session.get('results', {})
                
                if not results:
                    return "### Research Results\n\nResearch completed but no results available", "### Results Summary\n\n- **Status:** Completed\n- **Quality Score:** N/A\n- **Key Findings:** None"
                
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
                
                return results_text, summary_text
            
            # Update results when panel loads
            results_panel.load(update_results, outputs=[results_display, results_summary])
        
        return results_panel
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
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
                    
                    start_research_btn.click(
                        quick_start_research,
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
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 