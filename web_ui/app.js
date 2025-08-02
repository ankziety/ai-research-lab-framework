/**
 * AI Research Lab Web Interface
 * 
 * Frontend JavaScript application that provides interactive components
 * for the AI Research Lab framework with real-time updates and animations.
 */

class AIResearchLabApp {
    constructor() {
        this.socket = null;
        this.currentSession = null;
        this.systemConfig = {};
        this.agents = new Map();
        this.currentPhase = 0;
        this.isResearchActive = false;
        
        this.init();
    }
    
    init() {
        this.initializeUI();
        this.initializeWebSocket();
        this.loadConfiguration();
        this.startSystemMonitoring();
        
        console.log('AI Research Lab Interface initialized');
    }
    
    // ========================================
    // WebSocket Connection
    // ========================================
    
    initializeWebSocket() {
        this.socket = io();
        this.setupWebSocketEventHandlers();
    }

    setupWebSocketEventHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateSystemStatus('online');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateSystemStatus('offline');
        });

        this.socket.on('system_status', (data) => {
            this.handleSystemStatus(data);
        });

        this.socket.on('system_metrics', (data) => {
            this.updateSystemMetrics(data);
        });

        this.socket.on('research_status', (data) => {
            this.handleResearchStatus(data);
        });

        this.socket.on('research_complete', (data) => {
            this.handleResearchComplete(data);
        });

        this.socket.on('research_error', (data) => {
            this.handleResearchError(data);
        });

        this.socket.on('phase_update', (data) => {
            this.updateResearchPhase(data);
        });

        this.socket.on('agent_activity', (data) => {
            this.updateAgentActivity(data);
        });
        this.socket.on('activity_log', (data) => {
            this.addActivityLogEntry(data);
        });
    }
    
    // ========================================
    // UI Initialization
    // ========================================
    
    initializeUI() {
        this.initializeNavigation();
        this.initializeResearchConfig();
        this.initializeAgentVisualization();
        this.initializeProgressTracking();
        this.initializeActivityLog();
        this.initializeSettings();
        this.initializeConstraints();
    }
    
    initializeNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const panels = document.querySelectorAll('.panel');
        
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const panelId = item.dataset.panel;
                
                // Update active nav item
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
                
                // Show corresponding panel
                panels.forEach(panel => panel.classList.remove('active'));
                document.getElementById(`${panelId}-panel`).classList.add('active');
            });
        });
    }
    
    initializeResearchConfig() {
        const startBtn = document.getElementById('startResearchBtn');
        const stopBtn = document.getElementById('stopResearchBtn');
        
        startBtn.addEventListener('click', () => {
            this.startResearch();
        });
        
        stopBtn.addEventListener('click', () => {
            this.stopResearch();
        });
    }
    
    initializeConstraints() {
        const addConstraintBtn = document.getElementById('addConstraintBtn');
        const constraintsList = document.getElementById('constraintsList');
        
        addConstraintBtn.addEventListener('click', () => {
            this.addConstraintField();
        });
    }
    
    initializeAgentVisualization() {
        this.agentsContainer = document.getElementById('agentsContainer');
        this.agentsList = document.getElementById('agentsList');
        this.meetingIndicator = document.querySelector('.meeting-indicator');
    }
    
    initializeProgressTracking() {
        const phases = document.querySelectorAll('.phase');
        phases.forEach(phase => {
            phase.addEventListener('click', () => {
                this.showPhaseDetails(phase.dataset.phase);
            });
        });
    }
    
    initializeActivityLog() {
        const interventionInput = document.getElementById('interventionInput');
        const sendBtn = document.getElementById('sendInterventionBtn');
        const clearBtn = document.getElementById('clearLogsBtn');
        const exportBtn = document.getElementById('exportLogsBtn');
        
        sendBtn.addEventListener('click', () => {
            this.sendIntervention();
        });
        
        interventionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendIntervention();
            }
        });
        
        clearBtn.addEventListener('click', () => {
            this.clearActivityLog();
        });
        
        exportBtn.addEventListener('click', () => {
            this.exportActivityLog();
        });
    }
    
    initializeSettings() {
        const settingsBtn = document.getElementById('settingsBtn');
        const closeBtn = document.getElementById('closeSettingsBtn');
        const cancelBtn = document.getElementById('cancelSettingsBtn');
        const saveBtn = document.getElementById('saveSettingsBtn');
        const modal = document.getElementById('settingsModal');
        
        settingsBtn.addEventListener('click', () => {
            this.showSettingsModal();
        });
        
        closeBtn.addEventListener('click', () => {
            this.hideSettingsModal();
        });
        
        cancelBtn.addEventListener('click', () => {
            this.hideSettingsModal();
        });
        
        saveBtn.addEventListener('click', () => {
            this.saveSettings();
        });
        
        // Password visibility toggles
        const toggleButtons = document.querySelectorAll('.toggle-key');
        toggleButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetId = btn.dataset.target;
                const input = document.getElementById(targetId);
                const icon = btn.querySelector('i');
                
                if (input.type === 'password') {
                    input.type = 'text';
                    icon.className = 'fas fa-eye-slash';
                } else {
                    input.type = 'password';
                    icon.className = 'fas fa-eye';
                }
            });
        });
        
        // Modal click outside to close
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideSettingsModal();
            }
        });
    }
    
    // ========================================
    // Research Control
    // ========================================
    
    async startResearch() {
        const researchQuestion = document.getElementById('researchQuestion').value.trim();
        
        if (!researchQuestion) {
            this.showNotification('error', 'Error', 'Please enter a research question');
            return;
        }
        
        const config = {
            research_question: researchQuestion,
            domain: document.getElementById('domain').value,
            priority: document.getElementById('priority').value,
            budget: parseInt(document.getElementById('budget').value) || null,
            timeline: parseInt(document.getElementById('timeline').value) || null,
            max_agents: parseInt(document.getElementById('maxAgents').value) || null
        };
        
        // Add custom constraints
        const constraints = this.getCustomConstraints();
        Object.assign(config, constraints);
        
        try {
            this.isResearchActive = true;
            this.updateResearchButtons();
            
            const response = await fetch('/api/research/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('success', 'Research Started', result.message);
                this.updateSystemStatus('running');
            } else {
                this.showNotification('error', 'Error', result.error);
                this.isResearchActive = false;
                this.updateResearchButtons();
            }
        } catch (error) {
            console.error('Error starting research:', error);
            this.showNotification('error', 'Error', 'Failed to start research session');
            this.isResearchActive = false;
            this.updateResearchButtons();
        }
    }
    
    async stopResearch() {
        try {
            const response = await fetch('/api/research/stop', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isResearchActive = false;
                this.updateResearchButtons();
                this.updateSystemStatus('online');
                this.showNotification('warning', 'Research Stopped', result.message);
            }
        } catch (error) {
            console.error('Error stopping research:', error);
            this.showNotification('error', 'Error', 'Failed to stop research session');
        }
    }
    
    updateResearchButtons() {
        const startBtn = document.getElementById('startResearchBtn');
        const stopBtn = document.getElementById('stopResearchBtn');
        
        if (this.isResearchActive) {
            startBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');
        } else {
            startBtn.classList.remove('hidden');
            stopBtn.classList.add('hidden');
        }
    }
    
    // ========================================
    // Agent Visualization
    // ========================================
    
    createAgentAvatar(agent) {
        const avatar = document.createElement('div');
        avatar.className = 'agent-avatar';
        avatar.id = `agent-${agent.id}`;
        avatar.innerHTML = `
            ${this.getAgentInitials(agent.name)}
            <div class="agent-tooltip">${agent.name} - ${agent.expertise}</div>
        `;
        
        // Position agent around the meeting table
        this.positionAgentAvatar(avatar, agent.id);
        
        return avatar;
    }
    
    positionAgentAvatar(avatar, agentId) {
        const containerRect = this.agentsContainer.getBoundingClientRect();
        const centerX = containerRect.width / 2;
        const centerY = containerRect.height / 2;
        const radius = 150;
        
        // Calculate position around circle
        const angle = (agentId * 2 * Math.PI) / 6; // Assuming max 6 agents
        const x = centerX + radius * Math.cos(angle) - 30; // 30 = half avatar width
        const y = centerY + radius * Math.sin(angle) - 30; // 30 = half avatar height
        
        avatar.style.left = `${x}px`;
        avatar.style.top = `${y}px`;
    }
    
    updateAgentActivity(data) {
        const agentId = data.agent_id;
        let agent = this.agents.get(agentId);
        
        if (!agent) {
            // Create new agent
            agent = {
                id: agentId,
                name: data.name || `Agent ${agentId}`,
                expertise: data.expertise || 'General',
                status: data.status || 'idle',
                activity: data.activity || ''
            };
            this.agents.set(agentId, agent);
            
            // Add to visualization
            const avatar = this.createAgentAvatar(agent);
            this.agentsContainer.appendChild(avatar);
            
            // Add to agents list
            this.updateAgentsList();
        } else {
            // Update existing agent
            agent.status = data.status || agent.status;
            agent.activity = data.activity || agent.activity;
        }
        
        // Update avatar appearance
        this.updateAgentAvatar(agent);
        
        // Show speech bubble if agent is speaking
        if (data.message) {
            this.showAgentSpeech(agentId, data.message);
        }
        
        // Update activity log
        this.addActivityLogEntry({
            type: 'agent_activity',
            author: agent.name,
            message: data.activity || data.message,
            timestamp: Date.now() / 1000
        });
    }
    
    updateAgentAvatar(agent) {
        const avatar = document.getElementById(`agent-${agent.id}`);
        if (!avatar) return;
        
        // Remove existing status classes
        avatar.classList.remove('thinking', 'speaking', 'idle');
        
        // Add current status class
        avatar.classList.add(agent.status);
        
        // Update tooltip
        const tooltip = avatar.querySelector('.agent-tooltip');
        if (tooltip) {
            tooltip.textContent = `${agent.name} - ${agent.activity || agent.expertise}`;
        }
    }
    
    showAgentSpeech(agentId, message) {
        const avatar = document.getElementById(`agent-${agentId}`);
        if (!avatar) return;
        
        // Remove existing speech bubbles
        const existingBubble = avatar.querySelector('.speech-bubble');
        if (existingBubble) {
            existingBubble.remove();
        }
        
        // Create speech bubble
        const bubble = document.createElement('div');
        bubble.className = 'speech-bubble';
        bubble.textContent = message.length > 100 ? message.substring(0, 100) + '...' : message;
        
        // Position bubble relative to avatar
        bubble.style.bottom = '70px';
        bubble.style.left = '50%';
        bubble.style.transform = 'translateX(-50%)';
        
        avatar.appendChild(bubble);
        
        // Show bubble with animation
        setTimeout(() => bubble.classList.add('show'), 100);
        
        // Hide bubble after 3 seconds
        setTimeout(() => {
            bubble.classList.remove('show');
            setTimeout(() => bubble.remove(), 300);
        }, 3000);
    }
    
    updateAgentsList() {
        const agentsList = document.getElementById('agentsList');
        agentsList.innerHTML = '';
        
        this.agents.forEach(agent => {
            const card = document.createElement('div');
            card.className = 'agent-card';
            card.innerHTML = `
                <div class="agent-card-header">
                    <div class="agent-card-avatar">${this.getAgentInitials(agent.name)}</div>
                    <div class="agent-card-info">
                        <h4>${agent.name}</h4>
                        <p>${agent.expertise}</p>
                    </div>
                </div>
                <div class="agent-card-status">
                    <div class="agent-status-dot"></div>
                    <span>${agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
                </div>
                <div class="agent-card-metrics">
                    <div class="agent-metric">
                        <div class="agent-metric-value">0</div>
                        <div class="agent-metric-label">Tasks</div>
                    </div>
                    <div class="agent-metric">
                        <div class="agent-metric-value">0.0</div>
                        <div class="agent-metric-label">Quality</div>
                    </div>
                </div>
            `;
            
            agentsList.appendChild(card);
        });
        
        // Update team stats
        document.getElementById('activeAgents').textContent = this.agents.size;
        document.getElementById('teamEfficiency').textContent = '0%'; // Calculate actual efficiency
    }
    
    getAgentInitials(name) {
        return name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2);
    }
    
    // ========================================
    // Progress Tracking
    // ========================================
    
    updateResearchPhase(data) {
        this.currentPhase = data.phase;
        const progressFill = document.getElementById('progressFill');
        const currentPhaseDisplay = document.getElementById('currentPhase');
        const overallProgress = document.getElementById('overallProgress');
        
        // Update progress line
        progressFill.style.width = `${data.progress}%`;
        
        // Update current phase display
        currentPhaseDisplay.textContent = data.name;
        overallProgress.textContent = `${Math.round(data.progress)}%`;
        
        // Update phase indicators
        const phases = document.querySelectorAll('.phase');
        phases.forEach((phase, index) => {
            phase.classList.remove('current', 'completed');
            
            if (index + 1 < data.phase) {
                phase.classList.add('completed');
            } else if (index + 1 === data.phase) {
                phase.classList.add('current');
            }
        });
        
        // Show phase details
        this.showPhaseDetails(data.phase);
        
        // Update meeting indicator if there's a team meeting
        if (data.meeting_active) {
            this.meetingIndicator.classList.add('active');
            this.meetingIndicator.innerHTML = `
                <i class="fas fa-users"></i>
                <span>Team Meeting: ${data.name}</span>
            `;
        }
    }
    
    showPhaseDetails(phaseNumber) {
        const phaseDetails = document.getElementById('phaseDetails');
        const phases = [
            {
                name: 'Team Selection',
                description: 'The Principal Investigator analyzes the research requirements and selects appropriate expert agents for the team.'
            },
            {
                name: 'Project Specification',
                description: 'Team meeting to define project objectives, scope, methodology, and success criteria.'
            },
            {
                name: 'Tools Selection',
                description: 'Collaborative brainstorming to identify and select the most suitable computational tools and methods.'
            },
            {
                name: 'Implementation',
                description: 'Individual meetings where agents implement selected tools and develop necessary components.'
            },
            {
                name: 'Workflow Design',
                description: 'Principal Investigator designs an integrated workflow using the implemented tools.'
            },
            {
                name: 'Execution',
                description: 'Agents execute the designed workflow with cross-agent collaboration and critique.'
            },
            {
                name: 'Synthesis',
                description: 'Final team meeting to synthesize findings, conduct scientific critique, and formulate conclusions.'
            }
        ];
        
        const phase = phases[phaseNumber - 1];
        if (phase) {
            phaseDetails.innerHTML = `
                <h3>${phase.name}</h3>
                <p>${phase.description}</p>
            `;
        }
    }
    
    // ========================================
    // Activity Log
    // ========================================
    
    addActivityLogEntry(data) {
        const activityStream = document.getElementById('activityStream');
        const entry = document.createElement('div');
        entry.className = `activity-item ${data.type || 'system'}`;
        
        const timeString = new Date(data.timestamp * 1000).toLocaleTimeString();
        
        entry.innerHTML = `
            <div class="activity-avatar">
                ${this.getActivityIcon(data.type || 'system')}
            </div>
            <div class="activity-content">
                <div class="activity-header">
                    <span class="activity-author">${data.author || 'System'}</span>
                    <span class="activity-time">${timeString}</span>
                </div>
                <div class="activity-message">${data.message}</div>
            </div>
        `;
        
        activityStream.appendChild(entry);
        activityStream.scrollTop = activityStream.scrollHeight;
        
        // Remove old entries if too many
        while (activityStream.children.length > 100) {
            activityStream.removeChild(activityStream.firstChild);
        }
    }
    
    getActivityIcon(type) {
        const icons = {
            system: '<i class="fas fa-cog"></i>',
            agent: '<i class="fas fa-robot"></i>',
            human: '<i class="fas fa-user"></i>',
            research: '<i class="fas fa-search"></i>',
            meeting: '<i class="fas fa-users"></i>'
        };
        
        return icons[type] || icons.system;
    }
    
    async sendIntervention() {
        const input = document.getElementById('interventionInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        try {
            const response = await fetch('/api/intervention', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
            
            const result = await response.json();
            
            if (result.success) {
                input.value = '';
                this.showNotification('success', 'Intervention Sent', 'Your message has been sent to the research team');
            } else {
                this.showNotification('error', 'Error', result.error);
            }
        } catch (error) {
            console.error('Error sending intervention:', error);
            this.showNotification('error', 'Error', 'Failed to send intervention');
        }
    }
    
    clearActivityLog() {
        const activityStream = document.getElementById('activityStream');
        activityStream.innerHTML = '';
    }
    
    exportActivityLog() {
        const activities = Array.from(document.querySelectorAll('.activity-item')).map(item => {
            const author = item.querySelector('.activity-author').textContent;
            const time = item.querySelector('.activity-time').textContent;
            const message = item.querySelector('.activity-message').textContent;
            return `[${time}] ${author}: ${message}`;
        });
        
        const blob = new Blob([activities.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `activity_log_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    // ========================================
    // Configuration Management
    // ========================================
    
    async loadConfiguration() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            this.systemConfig = config;
            this.populateConfigForm(config);
        } catch (error) {
            console.error('Error loading configuration:', error);
        }
    }
    
    populateConfigForm(config) {
        // Populate system settings
        if (config.system) {
            document.getElementById('outputDir').value = config.system.output_dir || '';
            document.getElementById('maxConcurrentAgents').value = config.system.max_concurrent_agents || 8;
            document.getElementById('autoSaveResults').checked = config.system.auto_save_results || false;
            document.getElementById('enableNotifications').checked = config.system.enable_notifications || false;
        }
        
        // Show API key status
        if (config.api_keys_configured) {
            const openaiIndicator = document.createElement('span');
            openaiIndicator.textContent = config.api_keys_configured.openai ? ' ✓' : ' ✗';
            openaiIndicator.style.color = config.api_keys_configured.openai ? 'green' : 'red';
            
            const anthropicIndicator = document.createElement('span');
            anthropicIndicator.textContent = config.api_keys_configured.anthropic ? ' ✓' : ' ✗';
            anthropicIndicator.style.color = config.api_keys_configured.anthropic ? 'green' : 'red';
        }
    }
    
    showSettingsModal() {
        const modal = document.getElementById('settingsModal');
        modal.classList.add('show');
    }
    
    hideSettingsModal() {
        const modal = document.getElementById('settingsModal');
        modal.classList.remove('show');
    }
    
    async saveSettings() {
        const config = {
            api_keys: {
                openai: document.getElementById('openaiKey').value,
                anthropic: document.getElementById('anthropicKey').value
            },
            system: {
                output_dir: document.getElementById('outputDir').value,
                max_concurrent_agents: parseInt(document.getElementById('maxConcurrentAgents').value),
                auto_save_results: document.getElementById('autoSaveResults').checked,
                enable_notifications: document.getElementById('enableNotifications').checked
            }
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.hideSettingsModal();
                this.showNotification('success', 'Settings Saved', result.message);
                this.systemConfig = { ...this.systemConfig, ...config };
            } else {
                this.showNotification('error', 'Error', result.error);
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showNotification('error', 'Error', 'Failed to save settings');
        }
    }
    
    // ========================================
    // Constraints Management
    // ========================================
    
    addConstraintField() {
        const constraintsList = document.getElementById('constraintsList');
        const item = document.createElement('div');
        item.className = 'constraint-item';
        item.innerHTML = `
            <input type="text" placeholder="Constraint key" class="constraint-key">
            <input type="text" placeholder="Constraint value" class="constraint-value">
            <button type="button" class="btn btn-ghost" onclick="this.parentNode.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        constraintsList.appendChild(item);
    }
    
    getCustomConstraints() {
        const constraints = {};
        const items = document.querySelectorAll('.constraint-item');
        
        items.forEach(item => {
            const key = item.querySelector('.constraint-key').value.trim();
            const value = item.querySelector('.constraint-value').value.trim();
            
            if (key && value) {
                constraints[key] = value;
            }
        });
        
        return constraints;
    }
    
    // ========================================
    // System Monitoring
    // ========================================
    
    updateSystemStatus(status) {
        const statusIndicator = document.getElementById('systemStatus');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('span');
        
        statusDot.className = `status-dot ${status}`;
        
        const statusTexts = {
            offline: 'Offline',
            online: 'Online',
            running: 'Research Active'
        };
        
        statusText.textContent = statusTexts[status] || status;
    }
    
    updateSystemMetrics(metrics) {
        // Update CPU usage
        document.getElementById('cpuUsage').textContent = `${Math.round(metrics.cpu_usage)}%`;
        
        // Update memory info if available
        if (metrics.memory_usage !== undefined) {
            const memoryCard = document.querySelector('.metric-card .metric-content');
            if (memoryCard) {
                memoryCard.innerHTML += `<div class="metric-sub">Memory: ${Math.round(metrics.memory_usage)}%</div>`;
            }
        }
        
        // Update active agents count
        if (metrics.active_agents !== undefined) {
            document.getElementById('activeAgentsCount').textContent = metrics.active_agents;
        }
    }
    
    async startSystemMonitoring() {
        setInterval(async () => {
            try {
                const response = await fetch('/api/metrics?timeframe=session');
                const data = await response.json();
                
                if (data.current) {
                    this.updateSystemMetrics(data.current);
                }
                
                if (data.session_stats) {
                    document.getElementById('totalSessions').textContent = data.session_stats.total_sessions || 0;
                    document.getElementById('successfulSessions').textContent = data.session_stats.successful_sessions || 0;
                }
                
            } catch (error) {
                console.error('Error fetching metrics:', error);
            }
        }, 30000); // Update every 30 seconds
    }
    
    // ========================================
    // Event Handlers
    // ========================================
    
    handleSystemStatus(data) {
        if (data.framework_initialized) {
            this.updateSystemStatus('online');
        }
        
        if (data.current_session) {
            this.currentSession = data.current_session;
            this.isResearchActive = data.current_session.status === 'running';
            this.updateResearchButtons();
        }
        
        if (data.system_metrics) {
            this.updateSystemMetrics(data.system_metrics);
        }
    }
    
    handleResearchStatus(data) {
        this.addActivityLogEntry({
            type: 'research',
            author: 'System',
            message: data.message,
            timestamp: Date.now() / 1000
        });
    }
    
    handleResearchComplete(data) {
        this.currentSession = data;
        this.isResearchActive = false;
        this.updateResearchButtons();
        this.updateSystemStatus('online');
        
        this.showNotification('success', 'Research Complete', 
            `Research session ${data.session_id} completed successfully`);
        
        this.addActivityLogEntry({
            type: 'research',
            author: 'System',
            message: 'Research session completed successfully',
            timestamp: Date.now() / 1000
        });
    }
    
    handleResearchError(data) {
        this.isResearchActive = false;
        this.updateResearchButtons();
        this.updateSystemStatus('online');
        
        this.showNotification('error', 'Research Error', data.error);
        
        this.addActivityLogEntry({
            type: 'system',
            author: 'System',
            message: `Research error: ${data.error}`,
            timestamp: Date.now() / 1000
        });
    }
    
    // ========================================
    // Notifications
    // ========================================
    
    showNotification(type, title, message) {
        const container = document.getElementById('notificationsContainer');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        notification.innerHTML = `
            <div class="notification-header">
                <span class="notification-title">${title}</span>
                <button class="notification-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="notification-message">${message}</div>
        `;
        
        container.appendChild(notification);
        
        // Show notification with animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
        
        // Close button handler
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiResearchLab = new AIResearchLabApp();
});

// Demo functions for testing
function simulateAgentActivity() {
    const agents = ['agent_1', 'agent_2', 'agent_3'];
    const activities = ['analyzing data', 'generating hypothesis', 'reviewing literature', 'conducting experiment'];
    const statuses = ['thinking', 'speaking', 'idle'];
    
    setInterval(() => {
        const agent = agents[Math.floor(Math.random() * agents.length)];
        const activity = activities[Math.floor(Math.random() * activities.length)];
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        
        window.aiResearchLab.updateAgentActivity({
            agent_id: agent,
            name: `Agent ${agent.split('_')[1]}`,
            expertise: 'AI Research',
            activity: activity,
            status: status
        });
    }, 3000);
}

function simulateResearchProgress() {
    let currentPhase = 1;
    const phases = [
        'Team Selection',
        'Project Specification', 
        'Tools Selection',
        'Implementation',
        'Workflow Design',
        'Execution',
        'Synthesis'
    ];
    
    const interval = setInterval(() => {
        if (currentPhase <= phases.length) {
            window.aiResearchLab.updateResearchPhase({
                phase: currentPhase,
                name: phases[currentPhase - 1],
                progress: (currentPhase / phases.length) * 100,
                meeting_active: Math.random() > 0.7
            });
            currentPhase++;
        } else {
            clearInterval(interval);
        }
    }, 5000);
}

// Make demo functions available globally
window.simulateAgentActivity = simulateAgentActivity;
window.simulateResearchProgress = simulateResearchProgress;