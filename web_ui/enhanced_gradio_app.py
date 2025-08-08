#!/usr/bin/env python3
"""
Enhanced AI Research Lab Gradio Interface

A modern, comprehensive Gradio-based interface for the AI Research Lab framework
with integrated physics ecosystem, real-time research capabilities, and advanced
visualization features.
"""

import os
import sys
import json
import time
import asyncio
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Generator, Tuple, Union
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import gradio as gr
from gradio import Blocks, Chatbot, Textbox, Button, Dropdown, Slider, Checkbox, Markdown, HTML, JSON, Plot, DataFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from physics_ecosystem import PhysicsEcosystem, PhysicsToolType, PhysicsEngineStatus
from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAIResearchLabGradio:
    """Enhanced Gradio interface for the AI Research Lab framework with physics ecosystem."""
    
    def __init__(self):
        self.physics_ecosystem: Optional[PhysicsEcosystem] = None
        self.data_manager: Optional[DataManager] = None
        self.current_session: Optional[Dict[str, Any]] = None
        self.system_config: Dict[str, Any] = {}
        self.is_research_active = False
        self.research_thread: Optional[threading.Thread] = None
        self.research_history: List[Dict[str, Any]] = []
        
        # Initialize components
        self.initialize_physics_ecosystem()
        self.initialize_data_manager()
        self.load_config()
        
    def initialize_physics_ecosystem(self):
        """Initialize the physics ecosystem."""
        try:
            self.physics_ecosystem = PhysicsEcosystem()
            logger.info("Physics ecosystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing physics ecosystem: {e}")
    
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
            'physics_ecosystem': {
                'enable_engine_enhancement': True,
                'auto_fallback': True,
                'performance_tracking': True
            },
            'ui_settings': {
                'theme': 'soft',
                'auto_save': True,
                'notifications': True,
                'real_time_updates': True
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
    
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def save_config(self):
        """Save system configuration."""
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_file, 'w') as f:
                json.dump(self.system_config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def chat_with_research_lab(self, message: str, history: List[Dict[str, str]], 
                              research_mode: bool = False, physics_mode: bool = False) -> Tuple[List[Dict[str, str]], str]:
        """Enhanced chat function with physics ecosystem integration."""
        if not message.strip():
            return history, "Please enter a message.", {}
        
        try:
            if physics_mode and self.physics_ecosystem:
                return self._handle_physics_research(message, history)
            elif research_mode:
                return self._start_research_session(message, history)
            else:
                return self._handle_regular_chat(message, history)
        except Exception as e:
            logger.error(f"Error in chat function: {e}")
            return history, f"An error occurred: {str(e)}"
    
    def _handle_physics_research(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """Handle physics research using the ecosystem."""
        # Extract parameters from message (simple parsing for demo)
        parameters = self._extract_physics_parameters(message)
        
        # Execute physics research
        result = self.physics_ecosystem.execute_physics_research(message, parameters)
        
        if "error" in result:
            response = f"Physics research error: {result['error']}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, f"Physics research failed: {result['error']}"
        else:
            agent_role = result["result"]["agent_role"]
            tool_used = result["result"]["tool_used"]
            accuracy = result["result"]["result"].get("accuracy", 0.8)
            
            response = f"🔬 **Physics Research Completed**\n\n"
            response += f"**Agent:** {agent_role}\n"
            response += f"**Tool Used:** {tool_used}\n"
            response += f"**Accuracy:** {accuracy:.2%}\n"
            response += f"**Engine Enhanced:** {result['result']['result'].get('engine_enhanced', False)}\n\n"
            
            # Add calculation results
            calc_result = result["result"]["result"]["result"]
            if "energy_levels" in calc_result:
                response += f"**Energy Levels:** {calc_result['energy_levels']} eV\n"
            elif "trajectory" in calc_result:
                response += f"**Simulation Steps:** {len(calc_result['trajectory']['time'])}\n"
            elif "field_components" in calc_result:
                response += f"**Field Components:** Bx, By, Bz calculated\n"
        
        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Store research result
        self.research_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": message,
            "result": result,
            "mode": "physics"
        })
        
        return history, f"Physics research completed. {len(self.research_history)} total research sessions."
    
    def _extract_physics_parameters(self, message: str) -> Dict[str, Any]:
        """Extract physics parameters from message."""
        parameters = {}
        message_lower = message.lower()
        
        # Extract atomic number
        if "hydrogen" in message_lower or "h" in message_lower:
            parameters["atomic_number"] = 1
        elif "helium" in message_lower or "he" in message_lower:
            parameters["atomic_number"] = 2
        
        # Extract energy level
        if "ground state" in message_lower or "n=1" in message_lower:
            parameters["energy_level"] = 1
        elif "first excited" in message_lower or "n=2" in message_lower:
            parameters["energy_level"] = 2
        
        # Extract temperature for MD
        if "temperature" in message_lower:
            # Simple extraction - in real implementation would use regex
            if "300" in message_lower:
                parameters["temperature"] = 300
            elif "500" in message_lower:
                parameters["temperature"] = 500
        
        # Extract field strength for EM
        if "field" in message_lower and "strength" in message_lower:
            if "1" in message_lower:
                parameters["field_strength"] = 1.0
            elif "0.5" in message_lower:
                parameters["field_strength"] = 0.5
        
        return parameters
    
    def _start_research_session(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """Start a research session."""
        if self.is_research_active:
            return history, "Research session already in progress. Please wait for completion."
        
        self.is_research_active = True
        
        def research_worker():
            try:
                # Simulate research process
                time.sleep(2)
                
                # Mock research results
                research_result = {
                    "status": "completed",
                    "findings": [
                        "Initial hypothesis validated",
                        "Data analysis completed",
                        "Statistical significance confirmed"
                    ],
                    "confidence": 0.85,
                    "next_steps": [
                        "Conduct follow-up experiments",
                        "Validate results with larger dataset",
                        "Prepare manuscript for publication"
                    ]
                }
                
                self.current_session = {
                    "question": message,
                    "result": research_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.is_research_active = False
                
            except Exception as e:
                logger.error(f"Error in research worker: {e}")
                self.is_research_active = False
        
        self.research_thread = threading.Thread(target=research_worker)
        self.research_thread.start()
        
        response = "🚀 **Research Session Started**\n\n"
        response += f"**Question:** {message}\n"
        response += "**Status:** Initializing research framework...\n"
        response += "**Estimated Time:** 2-5 minutes\n\n"
        response += "Research is now running in the background. You can continue chatting or check the dashboard for updates."
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, "Research session started successfully."
    
    def _handle_regular_chat(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """Handle regular chat messages."""
        response = f"🤖 **AI Assistant Response**\n\n"
        response += f"I received your message: '{message}'\n\n"
        response += "**Available Modes:**\n"
        response += "• **Research Mode:** Enable for comprehensive research sessions\n"
        response += "• **Physics Mode:** Enable for physics-specific calculations\n"
        response += "• **Regular Chat:** General conversation and assistance\n\n"
        response += "What would you like to explore today?"
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, "Chat response generated."
    
    def get_physics_ecosystem_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the physics ecosystem."""
        if not self.physics_ecosystem:
            return {"error": "Physics ecosystem not initialized"}
        
        return self.physics_ecosystem.get_ecosystem_status()
    
    def get_research_status(self) -> Dict[str, Any]:
        """Get current research status."""
        return {
            "is_active": self.is_research_active,
            "current_session": self.current_session,
            "total_sessions": len(self.research_history),
            "last_session": self.research_history[-1] if self.research_history else None
        }
    
    def create_physics_dashboard(self) -> gr.Blocks:
        """Create physics ecosystem dashboard."""
        with gr.Blocks(title="Physics Ecosystem Dashboard") as dashboard:
            gr.Markdown("# 🔬 Physics Ecosystem Dashboard")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Ecosystem Status
                    gr.Markdown("### 🚀 Ecosystem Status")
                    ecosystem_status = gr.JSON(label="Ecosystem Status")
                    
                    def update_ecosystem_status():
                        return self.get_physics_ecosystem_status()
                    
                    update_btn = gr.Button("🔄 Update Status")
                    update_btn.click(update_ecosystem_status, outputs=[ecosystem_status])
                
                with gr.Column(scale=1):
                    # Quick Stats
                    gr.Markdown("### 📊 Quick Stats")
                    stats_display = gr.Markdown("Loading stats...")
                    
                    def update_stats():
                        status = self.get_physics_ecosystem_status()
                        if "error" in status:
                            return "❌ Physics ecosystem not available"
                        
                        engines = status.get("engines", {})
                        tools = status.get("tools", {})
                        agents = status.get("agents", {})
                        
                        stats = f"""
                        **Engines:** {len(engines)} available
                        **Tools:** {len(tools)} registered
                        **Agents:** {len(agents)} active
                        **Research Sessions:** {len(self.research_history)}
                        """
                        return stats
                    
                    update_btn.click(update_stats, outputs=[stats_display])
            
            # Physics Research Interface
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🔬 Physics Research")
                    
                    research_input = gr.Textbox(
                        label="Research Question",
                        placeholder="e.g., Solve Schrödinger equation for hydrogen atom ground state",
                        lines=3
                    )
                    
                    with gr.Row():
                        atomic_number = gr.Slider(
                            minimum=1, maximum=10, value=1, step=1,
                            label="Atomic Number"
                        )
                        energy_level = gr.Slider(
                            minimum=1, maximum=5, value=1, step=1,
                            label="Energy Level"
                        )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=100, maximum=1000, value=300, step=50,
                            label="Temperature (K)"
                        )
                        field_strength = gr.Slider(
                            minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                            label="Field Strength (T)"
                        )
                    
                    research_btn = gr.Button("🚀 Execute Physics Research", variant="primary")
                    research_output = gr.JSON(label="Research Results")
                    
                    def execute_physics_research(question, atomic_num, energy_lvl, temp, field_str):
                        parameters = {
                            "atomic_number": atomic_num,
                            "energy_level": energy_lvl,
                            "temperature": temp,
                            "field_strength": field_str
                        }
                        
                        if self.physics_ecosystem:
                            result = self.physics_ecosystem.execute_physics_research(question, parameters)
                            return result
                        else:
                            return {"error": "Physics ecosystem not available"}
                    
                    research_btn.click(
                        execute_physics_research,
                        inputs=[research_input, atomic_number, energy_level, temperature, field_strength],
                        outputs=[research_output]
                    )
                
                with gr.Column(scale=1):
                    # Visualization
                    gr.Markdown("### 📈 Visualization")
                    plot_output = gr.Plot(label="Physics Visualization")
                    
                    def create_sample_plot():
                        # Create a sample physics plot
                        fig = go.Figure()
                        
                        # Sample energy levels
                        levels = [-13.6, -3.4, -1.51, -0.85]
                        labels = ["n=1", "n=2", "n=3", "n=4"]
                        
                        fig.add_trace(go.Scatter(
                            x=[0, 0, 0, 0],
                            y=levels,
                            mode='markers+text',
                            text=labels,
                            textposition="middle right",
                            name="Energy Levels"
                        ))
                        
                        fig.update_layout(
                            title="Hydrogen Atom Energy Levels",
                            xaxis_title="",
                            yaxis_title="Energy (eV)",
                            showlegend=False
                        )
                        
                        return fig
                    
                    plot_btn = gr.Button("📊 Generate Sample Plot")
                    plot_btn.click(create_sample_plot, outputs=[plot_output])
        
        return dashboard
    
    def create_agents_panel(self) -> gr.Blocks:
        """Create enhanced agents panel with physics agents."""
        with gr.Blocks(title="AI Agents") as agents_panel:
            gr.Markdown("# 🤖 AI Agents Dashboard")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Agent Selection
                    gr.Markdown("### 🎯 Agent Selection")
                    
                    agent_dropdown = gr.Dropdown(
                        choices=["quantum_physics", "computational_physics", "electromagnetic_physics"],
                        label="Select Physics Agent",
                        value="quantum_physics"
                    )
                    
                    agent_info = gr.JSON(label="Agent Information")
                    
                    def get_agent_info(agent_id):
                        if not self.physics_ecosystem:
                            return {"error": "Physics ecosystem not available"}
                        
                        agents = self.physics_ecosystem.agent_manager.agents
                        if agent_id in agents:
                            agent = agents[agent_id]
                            return {
                                "agent_id": agent.agent_id,
                                "role": agent.role,
                                "expertise": agent.expertise,
                                "active_tools": agent.active_tools,
                                "performance": agent.performance_metrics
                            }
                        else:
                            return {"error": f"Agent {agent_id} not found"}
                    
                    agent_dropdown.change(get_agent_info, inputs=[agent_dropdown], outputs=[agent_info])
                
                with gr.Column(scale=1):
                    # Agent Performance
                    gr.Markdown("### 📊 Performance Metrics")
                    performance_display = gr.Markdown("Select an agent to view performance...")
                    
                    def update_performance(agent_id):
                        if not self.physics_ecosystem:
                            return "Physics ecosystem not available"
                        
                        agents = self.physics_ecosystem.agent_manager.agents
                        if agent_id in agents:
                            agent = agents[agent_id]
                            metrics = agent.performance_metrics
                            
                            return f"""
                            **Tasks Completed:** {metrics['tasks_completed']}
                            **Success Rate:** {metrics['success_rate']:.2%}
                            **Average Accuracy:** {metrics['average_accuracy']:.2%}
                            **Last Active:** {agent.last_active.strftime('%Y-%m-%d %H:%M') if agent.last_active else 'Never'}
                            """
                        else:
                            return f"Agent {agent_id} not found"
                    
                    agent_dropdown.change(update_performance, inputs=[agent_dropdown], outputs=[performance_display])
            
            # Agent Tools
            with gr.Row():
                gr.Markdown("### 🛠️ Available Tools")
                tools_display = gr.JSON(label="Agent Tools")
                
                def get_agent_tools(agent_id):
                    if not self.physics_ecosystem:
                        return {"error": "Physics ecosystem not available"}
                    
                    tools = self.physics_ecosystem.agent_manager.get_agent_tools(agent_id)
                    return [{"name": tool.name, "type": tool.tool_type.value, "accuracy": tool.accuracy} for tool in tools]
                
                agent_dropdown.change(get_agent_tools, inputs=[agent_dropdown], outputs=[tools_display])
        
        return agents_panel
    
    def create_enhanced_settings_panel(self) -> gr.Blocks:
        """Create enhanced settings panel with physics ecosystem configuration."""
        with gr.Blocks(title="Settings") as settings_panel:
            gr.Markdown("# ⚙️ Enhanced Settings")
            
            with gr.Tabs():
                # API Keys Tab
                with gr.TabItem("🔑 API Keys"):
                    with gr.Row():
                        with gr.Column():
                            openai_key = gr.Textbox(
                                label="OpenAI API Key",
                                type="password",
                                value=self.system_config.get('api_keys', {}).get('openai', '')
                            )
                            anthropic_key = gr.Textbox(
                                label="Anthropic API Key",
                                type="password",
                                value=self.system_config.get('api_keys', {}).get('anthropic', '')
                            )
                        
                        with gr.Column():
                            gemini_key = gr.Textbox(
                                label="Gemini API Key",
                                type="password",
                                value=self.system_config.get('api_keys', {}).get('gemini', '')
                            )
                            huggingface_key = gr.Textbox(
                                label="HuggingFace API Key",
                                type="password",
                                value=self.system_config.get('api_keys', {}).get('huggingface', '')
                            )
                
                # Physics Ecosystem Tab
                with gr.TabItem("🔬 Physics Ecosystem"):
                    with gr.Row():
                        with gr.Column():
                            enable_engine_enhancement = gr.Checkbox(
                                label="Enable Engine Enhancement",
                                value=self.system_config.get('physics_ecosystem', {}).get('enable_engine_enhancement', True)
                            )
                            auto_fallback = gr.Checkbox(
                                label="Auto Fallback",
                                value=self.system_config.get('physics_ecosystem', {}).get('auto_fallback', True)
                            )
                        
                        with gr.Column():
                            performance_tracking = gr.Checkbox(
                                label="Performance Tracking",
                                value=self.system_config.get('physics_ecosystem', {}).get('performance_tracking', True)
                            )
                
                # UI Settings Tab
                with gr.TabItem("🎨 UI Settings"):
                    with gr.Row():
                        with gr.Column():
                            theme_dropdown = gr.Dropdown(
                                choices=["soft", "default", "glass", "monochrome"],
                                label="Theme",
                                value=self.system_config.get('ui_settings', {}).get('theme', 'soft')
                            )
                            auto_save = gr.Checkbox(
                                label="Auto Save",
                                value=self.system_config.get('ui_settings', {}).get('auto_save', True)
                            )
                        
                        with gr.Column():
                            notifications = gr.Checkbox(
                                label="Notifications",
                                value=self.system_config.get('ui_settings', {}).get('notifications', True)
                            )
                            real_time_updates = gr.Checkbox(
                                label="Real-time Updates",
                                value=self.system_config.get('ui_settings', {}).get('real_time_updates', True)
                            )
            
            # Save Button
            save_btn = gr.Button("💾 Save Settings", variant="primary")
            save_status = gr.Markdown("Settings saved successfully!")
            
            def save_settings(openai, anthropic, gemini, huggingface, 
                            engine_enhancement, fallback, tracking,
                            theme, auto_save_val, notifications_val, real_time):
                # Update configuration
                self.system_config['api_keys']['openai'] = openai
                self.system_config['api_keys']['anthropic'] = anthropic
                self.system_config['api_keys']['gemini'] = gemini
                self.system_config['api_keys']['huggingface'] = huggingface
                
                self.system_config['physics_ecosystem']['enable_engine_enhancement'] = engine_enhancement
                self.system_config['physics_ecosystem']['auto_fallback'] = fallback
                self.system_config['physics_ecosystem']['performance_tracking'] = tracking
                
                self.system_config['ui_settings']['theme'] = theme
                self.system_config['ui_settings']['auto_save'] = auto_save_val
                self.system_config['ui_settings']['notifications'] = notifications_val
                self.system_config['ui_settings']['real_time_updates'] = real_time
                
                # Save to file
                self.save_config()
                
                return "✅ Settings saved successfully!"
            
            save_btn.click(
                save_settings,
                inputs=[openai_key, anthropic_key, gemini_key, huggingface_key,
                       enable_engine_enhancement, auto_fallback, performance_tracking,
                       theme_dropdown, auto_save, notifications, real_time_updates],
                outputs=[save_status]
            )
        
        return settings_panel
    
    def create_enhanced_interface(self) -> gr.Blocks:
        """Create the enhanced main Gradio interface."""
        with gr.Blocks(
            title="Enhanced AI Research Lab",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            .chat-container {
                height: 600px;
                overflow-y: auto;
            }
            .physics-dashboard {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🔬 Enhanced AI Research Lab")
            gr.Markdown("Welcome to the enhanced AI Research Lab with integrated physics ecosystem!")
            
            with gr.Tabs():
                # Enhanced Chat Tab
                with gr.TabItem("💬 Enhanced Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Enhanced Chat Interface
                            chatbot = gr.Chatbot(
                                label="AI Research Lab Chat",
                                height=500,
                                show_label=True,
                                container=True,
                                type="messages"
                            )
                            
                            with gr.Row():
                                with gr.Column(scale=4):
                                    msg = gr.Textbox(
                                        label="Message",
                                        placeholder="Ask a question, start research, or explore physics...",
                                        lines=2
                                    )
                                
                                with gr.Column(scale=1):
                                    research_mode = gr.Checkbox(
                                        label="Research Mode",
                                        value=False,
                                        info="Enable for comprehensive research sessions"
                                    )
                                
                                with gr.Column(scale=1):
                                    physics_mode = gr.Checkbox(
                                        label="Physics Mode",
                                        value=False,
                                        info="Enable for physics-specific calculations"
                                    )
                                
                                with gr.Column(scale=1):
                                    submit_btn = gr.Button("Send", variant="primary")
                            
                            # Status display
                            status_display = gr.Markdown("Ready to chat!")
                        
                        with gr.Column(scale=1):
                            # Quick Actions
                            gr.Markdown("### Quick Actions")
                            
                            start_research_btn = gr.Button("🚀 Start Research", variant="primary")
                            start_physics_btn = gr.Button("🔬 Physics Research", variant="primary")
                            view_agents_btn = gr.Button("🤖 View Agents")
                            check_results_btn = gr.Button("📊 Check Results")
                            open_settings_btn = gr.Button("⚙️ Settings")
                            
                            # System Status
                            gr.Markdown("### System Status")
                            system_status = gr.Markdown("🟢 System Online")
                    
                    # Enhanced chat function
                    def enhanced_chat_fn(message, history, research_mode_val, physics_mode_val):
                        return self.chat_with_research_lab(message, history, research_mode_val, physics_mode_val)
                    
                    submit_btn.click(
                        enhanced_chat_fn,
                        inputs=[msg, chatbot, research_mode, physics_mode],
                        outputs=[chatbot, status_display],
                        api_name="enhanced_chat"
                    )
                    
                    msg.submit(
                        enhanced_chat_fn,
                        inputs=[msg, chatbot, research_mode, physics_mode],
                        outputs=[chatbot, status_display],
                        api_name="enhanced_chat"
                    )
                    
                    # Quick action handlers
                    def quick_start_research():
                        return "🚀 Research mode activated! Enter your research question in the chat.", "Research mode ready"
                    
                    def quick_start_physics():
                        return "🔬 Physics mode activated! Ask physics questions or calculations.", "Physics mode ready"
                    
                    start_research_btn.click(
                        quick_start_research,
                        outputs=[msg, status_display]
                    )
                    
                    start_physics_btn.click(
                        quick_start_physics,
                        outputs=[msg, status_display]
                    )
                
                # Physics Dashboard Tab
                with gr.TabItem("🔬 Physics Dashboard"):
                    self.create_physics_dashboard()
                
                # Enhanced Agents Tab
                with gr.TabItem("🤖 Enhanced Agents"):
                    self.create_agents_panel()
                
                # Research Results Tab
                with gr.TabItem("📊 Research Results"):
                    with gr.Blocks():
                        gr.Markdown("### 📊 Research History")
                        
                        results_display = gr.JSON(label="Research Results")
                        
                        def update_results():
                            return {
                                "total_sessions": len(self.research_history),
                                "recent_sessions": self.research_history[-5:] if self.research_history else [],
                                "current_status": self.get_research_status()
                            }
                        
                        update_results_btn = gr.Button("🔄 Update Results")
                        update_results_btn.click(update_results, outputs=[results_display])
                
                # Enhanced Settings Tab
                with gr.TabItem("⚙️ Enhanced Settings"):
                    self.create_enhanced_settings_panel()
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("Enhanced AI Research Lab Framework - Powered by Gradio with Physics Ecosystem")
        
        return interface

def main():
    """Main function to run the enhanced Gradio interface."""
    # Create the enhanced interface
    app = EnhancedAIResearchLabGradio()
    interface = app.create_enhanced_interface()
    
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