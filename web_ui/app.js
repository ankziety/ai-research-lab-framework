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
        this.intervals = new Set();
        this.eventListeners = new Map();
        
        this.init();
    }
    
    init() {
        this.initializeUI();
        this.initializeWebSocket();
        this.loadConfiguration();
        this.startSystemMonitoring();
        this.startHeartbeat();
        
        console.log('AI Research Lab Interface initialized');
    }
    
    // ========================================
    // WebSocket Connection
    // ========================================
    
    initializeWebSocket() {
        this.socket = io({
            transports: ['polling', 'websocket'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 20000
        });
        this.setupWebSocketEventHandlers();
    }

    setupWebSocketEventHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateSystemStatus('online');
            this.showNotification('success', 'Connected', 'Successfully connected to the research framework');
        });

        this.socket.on('disconnect', (reason) => {
            console.log('Disconnected from server:', reason);
            this.updateSystemStatus('offline');
            this.showNotification('warning', 'Disconnected', 'Connection lost. Attempting to reconnect...');
        });

        this.socket.on('reconnect', (attemptNumber) => {
            console.log('Reconnected to server after', attemptNumber, 'attempts');
            this.updateSystemStatus('online');
            this.showNotification('success', 'Reconnected', 'Successfully reconnected to the research framework');
        });

        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log('Reconnection attempt:', attemptNumber);
            this.updateSystemStatus('connecting');
        });

        this.socket.on('reconnect_error', (error) => {
            console.log('Reconnection error:', error);
            this.updateSystemStatus('error');
            this.showNotification('error', 'Connection Error', 'Failed to reconnect. Please refresh the page.');
        });

        this.socket.on('reconnect_failed', () => {
            console.log('Reconnection failed');
            this.updateSystemStatus('error');
            this.showNotification('error', 'Connection Failed', 'Unable to reconnect. Please refresh the page.');
        });

        // Heartbeat to keep connection alive
        this.socket.on('heartbeat_ack', (data) => {
            console.log('Heartbeat acknowledged:', data);
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
            this.addActivityLogEntry(data);
        });
        this.socket.on('activity_log', (data) => {
            this.addActivityLogEntry(data);
        });
        this.socket.on('chat_log', (data) => {
            this.addChatLogEntry(data);
        });
        this.socket.on('meeting', (data) => {
            this.addMeetingEntry(data);
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
        this.initializeChatLogs();
        this.initializeMeetings();
        this.initializeSettings();
        this.initializeConstraints();
        this.initializeHistory();
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
        
        // Load all agents from the framework
        this.loadAllAgents();
        
        // Add refresh button event listener
        document.getElementById('refreshAgentsBtn').addEventListener('click', () => {
            this.loadAllAgents();
        });
    }
    
    async loadAllAgents() {
        try {
            const response = await fetch('/api/agents');
            const data = await response.json();
            
            if (data.agents) {
                // Clear existing agents
                this.agents.clear();
                this.agentsContainer.innerHTML = '';
                
                // Add all agents to the meeting table
                data.agents.forEach(agentData => {
                    const agent = {
                        id: agentData.id,
                        name: agentData.name,
                        expertise: agentData.expertise,
                        status: agentData.status,
                        is_active: agentData.is_active,
                        current_task: agentData.current_task,
                        performance: agentData.performance || {
                            contributions: 0,
                            meetings_attended: 0,
                            tools_used: 0,
                            phases_completed: 0
                        },
                        recent_activities: agentData.recent_activities || [],
                        agent_type: agentData.agent_type || 'Unknown'
                    };
                    
                    this.agents.set(agent.id, agent);
                    
                    // Create avatar for meeting table
                    const avatar = this.createAgentAvatar(agent);
                    this.agentsContainer.appendChild(avatar);
                });
                
                // Update agents list
                this.updateAgentsList();
            }
        } catch (error) {
            console.error('Error loading agents:', error);
        }
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
    
    initializeChatLogs() {
        const clearBtn = document.getElementById('clearChatLogsBtn');
        const exportBtn = document.getElementById('exportChatLogsBtn');
        const typeFilter = document.getElementById('chatLogType');
        
        clearBtn.addEventListener('click', () => {
            this.clearChatLogs();
        });
        
        exportBtn.addEventListener('click', () => {
            this.exportChatLogs();
        });
        
        typeFilter.addEventListener('change', () => {
            this.filterChatLogs(typeFilter.value);
        });
        
        // Load existing chat logs
        this.loadChatLogs();
    }
    
    initializeHistory() {
        this.loadHistory();
        
        // Set up event handlers for history controls
        document.getElementById('clearHistoryBtn').addEventListener('click', () => {
            this.clearHistory();
        });
        
        document.getElementById('exportHistoryBtn').addEventListener('click', () => {
            this.exportHistory();
        });
        
        document.getElementById('sessionFilter').addEventListener('change', (e) => {
            this.filterHistory(e.target.value);
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
        
        // API key test buttons
        const testButtons = document.querySelectorAll('.test-key');
        testButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.testApiKey(btn);
            });
        });
        
        // Modal click outside to close
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideSettingsModal();
            }
        });
    }
    
    async testApiKey(button) {
        const provider = button.dataset.provider;
        const inputId = this.getInputIdForProvider(provider);
        const input = document.getElementById(inputId);
        const apiKey = input.value.trim();
        
        if (!apiKey) {
            this.showNotification('error', 'Error', `Please enter a ${provider} API key first`);
            return;
        }
        
        // Update button state
        button.classList.add('testing');
        const icon = button.querySelector('i');
        icon.className = 'fas fa-spinner fa-spin';
        
        try {
            const response = await fetch('/api/config/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    provider: provider,
                    api_key: apiKey
                })
            });
            
            const result = await response.json();
            
            if (result.valid) {
                button.classList.remove('testing');
                button.classList.add('valid');
                icon.className = 'fas fa-check';
                this.showNotification('success', 'API Key Valid', result.message);
            } else {
                button.classList.remove('testing');
                button.classList.add('invalid');
                icon.className = 'fas fa-times';
                this.showNotification('error', 'Invalid API Key', result.message);
            }
            
            // Reset button after 3 seconds
            setTimeout(() => {
                button.classList.remove('valid', 'invalid');
                icon.className = 'fas fa-check';
            }, 3000);
            
        } catch (error) {
            console.error('Error testing API key:', error);
            button.classList.remove('testing');
            button.classList.add('invalid');
            icon.className = 'fas fa-times';
            this.showNotification('error', 'Error', 'Failed to test API key');
            
            // Reset button after 3 seconds
            setTimeout(() => {
                button.classList.remove('invalid');
                icon.className = 'fas fa-check';
            }, 3000);
        }
    }
    
    getInputIdForProvider(provider) {
        const mapping = {
            'openai': 'openaiKey',
            'anthropic': 'anthropicKey',
            'gemini': 'geminiKey',
            'huggingface': 'huggingfaceKey',
            'ollama': 'ollamaEndpoint',
            'google_search': 'googleSearchKey',
            'google_search_engine': 'googleSearchEngineId',
            'serpapi': 'serpapiKey',
            'semantic_scholar': 'semanticScholarKey',
            'openalex': 'openalexEmail',
            'core': 'coreKey'
        };
        return mapping[provider];
    }
    
    // ========================================
    // Research Control
    // ========================================
    
    async startResearch() {
        try {
            const researchQuestion = this.sanitizeInput(document.getElementById('researchQuestion').value.trim());
            
            if (!researchQuestion) {
                this.showNotification('error', 'Error', 'Please enter a research question');
                return;
            }
            
            // Input validation
            if (researchQuestion.length > 1000) {
                this.showNotification('error', 'Error', 'Research question too long (max 1000 characters)');
                return;
            }
            
            const config = {
                research_question: researchQuestion,
                domain: this.sanitizeInput(document.getElementById('domain').value),
                priority: this.sanitizeInput(document.getElementById('priority').value),
                budget: this.validateNumber(document.getElementById('budget').value, 0, 1000000),
                timeline: this.validateNumber(document.getElementById('timeline').value, 1, 52),
                max_agents: this.validateNumber(document.getElementById('maxAgents').value, 1, 20)
            };
            
            // Add custom constraints
            const constraints = this.getCustomConstraints();
            Object.assign(config, constraints);
            
            this.isResearchActive = true;
            this.updateResearchButtons();
            
            const response = await fetch('/api/research/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('success', 'Research Started', result.message);
                this.updateSystemStatus('running');
            } else {
                this.showNotification('error', 'Error', result.error || 'Unknown error occurred');
                this.isResearchActive = false;
                this.updateResearchButtons();
            }
        } catch (error) {
            console.error('Error starting research:', error);
            this.showNotification('error', 'Error', `Failed to start research session: ${error.message}`);
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
        
        // Create circular avatar with initials
        const initials = this.getAgentInitials(agent.name);
        const color = this.getAgentColor(agent.id);
        
        avatar.innerHTML = `
            <div class="agent-circle" style="background-color: ${color}">
                <span class="agent-initials">${initials}</span>
            </div>
            <div class="agent-tooltip">${agent.name} - ${agent.expertise}</div>
            <div class="agent-status-indicator"></div>
        `;
        
        // Position agent around the table
        this.positionAgentAvatar(avatar, agent.id);
        
        return avatar;
    }
    
    getAgentColor(agentId) {
        // Generate consistent colors for agents
        const colors = [
            '#4A90E2', '#7ED321', '#F5A623', '#D0021B', 
            '#9013FE', '#50E3C2', '#F8E71C', '#BD10E0',
            '#4A4A4A', '#9B9B9B', '#D8D8D8', '#FFFFFF'
        ];
        const index = agentId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
        return colors[index];
    }
    
    positionAgentAvatar(avatar, agentId) {
        // Position agents in a circle around the meeting table
        const container = this.agentsContainer;
        const containerRect = container.getBoundingClientRect();
        const centerX = containerRect.width / 2;
        const centerY = containerRect.height / 2;
        const radius = Math.min(centerX, centerY) * 0.6;
        
        // Calculate position based on agent ID
        const agentIndex = this.agents.size - 1;
        const angle = (agentIndex * 2 * Math.PI) / Math.max(this.agents.size, 1);
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        avatar.style.position = 'absolute';
        avatar.style.left = `${x - 25}px`;
        avatar.style.top = `${y - 25}px`;
        avatar.style.zIndex = '10';
        
        // Add animation for new agents
        avatar.style.opacity = '0';
        avatar.style.transform = 'scale(0)';
        setTimeout(() => {
            avatar.style.transition = 'all 0.5s ease-out';
            avatar.style.opacity = '1';
            avatar.style.transform = 'scale(1)';
        }, 100);
    }
    
    updateAgentActivity(data) {
        const agentId = data.agent_id;
        let agent = this.agents.get(agentId);
        
        if (!agent) {
            // Reload all agents to get the new one
            this.loadAllAgents();
            return;
        } else {
            // Update existing agent
            agent.status = data.status || agent.status;
            agent.activity = data.activity || agent.activity;
            
            // Update performance metrics if provided
            if (data.metadata) {
                if (data.metadata.word_count) {
                    agent.performance.total_words += data.metadata.word_count;
                    agent.performance.contributions += 1;
                    agent.performance.avg_sentence_length = 
                        agent.performance.total_words / agent.performance.contributions;
                }
                if (data.metadata.meeting_attended) {
                    agent.performance.meetings_attended += 1;
                }
                if (data.metadata.tool_used) {
                    agent.performance.tools_used += 1;
                }
                if (data.metadata.phase_completed) {
                    agent.performance.phases_completed += 1;
                }
            }
            
            // Add to recent activities
            agent.recent_activities.unshift({
                activity: data.activity || data.message,
                timestamp: Date.now() / 1000,
                type: data.activity_type || 'general'
            });
            
            // Keep only last 10 activities
            if (agent.recent_activities.length > 10) {
                agent.recent_activities = agent.recent_activities.slice(0, 10);
            }
        }
        
        // Update avatar appearance
        this.updateAgentAvatar(agent);
        
        // Show speech bubble if agent is speaking
        if (data.message) {
            this.showAgentSpeech(agentId, data.message);
        }
        
        // Update activity log with enhanced information
        this.addActivityLogEntry({
            type: 'agent_activity',
            author: agent.name,
            message: data.activity || data.message,
            timestamp: Date.now() / 1000,
            metadata: {
                agent_id: agentId,
                activity_type: data.activity_type,
                expertise: agent.expertise,
                status: agent.status,
                performance: agent.performance
            }
        });
        
        // Update agents list to reflect changes
        this.updateAgentsList();
        
        // Update meeting indicator if this is a meeting activity
        if (data.activity_type === 'meeting' || data.activity?.includes('meeting')) {
            this.updateMeetingIndicator(true);
        }
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
        
        // Create animated speech bubble
        const bubble = document.createElement('div');
        bubble.className = 'speech-bubble';
        bubble.innerHTML = `
            <div class="speech-content">
                <div class="speech-text">${message.length > 100 ? message.substring(0, 100) + '...' : message}</div>
                <div class="speech-tail"></div>
            </div>
        `;
        
        // Position bubble above the agent
        bubble.style.position = 'absolute';
        bubble.style.bottom = '60px';
        bubble.style.left = '50%';
        bubble.style.transform = 'translateX(-50%)';
        bubble.style.zIndex = '20';
        
        avatar.appendChild(bubble);
        
        // Show bubble with animation
        setTimeout(() => {
            bubble.style.transition = 'all 0.3s ease-out';
            bubble.style.opacity = '1';
            bubble.style.transform = 'translateX(-50%) scale(1)';
        }, 100);
        
        // Hide bubble after 4 seconds
        setTimeout(() => {
            bubble.style.transition = 'all 0.3s ease-in';
            bubble.style.opacity = '0';
            bubble.style.transform = 'translateX(-50%) scale(0.8)';
            setTimeout(() => bubble.remove(), 300);
        }, 4000);
    }
    
    updateAgentsList() {
        const agentsList = document.getElementById('agentsList');
        agentsList.innerHTML = '';
        
        this.agents.forEach(agent => {
            const card = document.createElement('div');
            card.className = 'agent-card';
            
            // Get recent activity for display
            const recentActivity = agent.recent_activities.length > 0 ? 
                agent.recent_activities[0].activity : 'No recent activity';
            
            // Determine status display
            const statusText = agent.is_active ? 'Active' : 'Idle';
            const statusClass = agent.is_active ? 'active' : 'idle';
            
            card.innerHTML = `
                <div class="agent-card-header">
                    <div class="agent-card-avatar" style="background-color: ${this.getAgentColor(agent.id)}">
                        ${this.getAgentInitials(agent.name)}
                    </div>
                    <div class="agent-card-info">
                        <h4>${agent.name}</h4>
                        <p>${agent.expertise}</p>
                        <div class="agent-status">
                            <span class="status-dot ${statusClass}"></span>
                            <span class="status-text">${statusText}</span>
                        </div>
                    </div>
                </div>
                <div class="agent-card-content">
                    <div class="agent-recent-activity">
                        <strong>Recent Activity:</strong>
                        <p>${recentActivity}</p>
                    </div>
                    <div class="agent-card-metrics">
                        <div class="agent-metric">
                            <div class="agent-metric-value">${agent.performance.contributions || 0}</div>
                            <div class="agent-metric-label">Contributions</div>
                        </div>
                        <div class="agent-metric">
                            <div class="agent-metric-value">${agent.performance.meetings_attended || 0}</div>
                            <div class="agent-metric-label">Meetings</div>
                        </div>
                        <div class="agent-metric">
                            <div class="agent-metric-value">${agent.performance.tools_used || 0}</div>
                            <div class="agent-metric-label">Tools Used</div>
                        </div>
                        <div class="agent-metric">
                            <div class="agent-metric-value">${agent.performance.phases_completed || 0}</div>
                            <div class="agent-metric-label">Phases</div>
                        </div>
                    </div>
                    <div class="agent-activities-list">
                        <strong>Recent Activities:</strong>
                        <div class="activities-scroll">
                            ${agent.recent_activities.slice(0, 3).map(activity => `
                                <div class="activity-item-mini">
                                    <span class="activity-time-mini">${new Date(activity.timestamp * 1000).toLocaleTimeString()}</span>
                                    <span class="activity-text-mini">${activity.activity}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
            
            agentsList.appendChild(card);
        });
        
        // Update team stats - show total agents present
        document.getElementById('activeAgents').textContent = this.agents.size;
        
        // Calculate team efficiency
        let totalContributions = 0;
        let totalMeetings = 0;
        let totalTools = 0;
        let totalPhases = 0;
        
        this.agents.forEach(agent => {
            totalContributions += agent.performance.contributions || 0;
            totalMeetings += agent.performance.meetings_attended || 0;
            totalTools += agent.performance.tools_used || 0;
            totalPhases += agent.performance.phases_completed || 0;
        });
        
        const efficiency = this.agents.size > 0 ? 
            Math.round((totalContributions + totalMeetings + totalTools + totalPhases) / this.agents.size) : 0;
        
        document.getElementById('teamEfficiency').textContent = `${efficiency}%`;
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
        
        // Calculate progress if not provided
        let progress = data.progress;
        if (!progress && data.phase) {
            const totalPhases = 7; // Number of phases in the research process
            progress = (data.phase / totalPhases) * 100;
        }
        
        // Update progress line
        if (progressFill) {
            progressFill.style.width = `${progress || 0}%`;
        }
        
        // Update current phase display
        if (currentPhaseDisplay) {
            currentPhaseDisplay.textContent = data.name || data.phase_name || `Phase ${data.phase}`;
        }
        if (overallProgress) {
            overallProgress.textContent = `${Math.round(progress || 0)}%`;
        }
        
        // Update phase indicators
        const phases = document.querySelectorAll('.phase');
        phases.forEach((phase, index) => {
            phase.classList.remove('current', 'completed', 'failed');
            
            if (index + 1 < data.phase) {
                phase.classList.add('completed');
            } else if (index + 1 === data.phase) {
                phase.classList.add('current');
                if (data.status === 'failed') {
                    phase.classList.add('failed');
                }
            }
        });
        
        // Show phase details
        this.showPhaseDetails(data.phase);
        
        // Update meeting indicator if there's a team meeting
        if (data.meeting_active) {
            this.updateMeetingIndicator(true);
        } else {
            this.updateMeetingIndicator(false);
        }
        
        // Add activity log entry for phase updates
        this.addActivityLogEntry({
            type: 'system',
            author: 'System',
            message: `Phase ${data.phase}: ${data.name || data.phase_name} - ${data.status || 'in progress'}`,
            timestamp: Date.now() / 1000
        });
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
    // Chat Logs
    // ========================================
    
    async loadChatLogs() {
        try {
            const response = await fetch('/api/chat-logs');
            const data = await response.json();
            
            if (data.logs) {
                data.logs.forEach(log => {
                    this.addChatLogEntry(log);
                });
            }
        } catch (error) {
            console.error('Error loading chat logs:', error);
        }
    }
    
    addChatLogEntry(data) {
        const chatStream = document.getElementById('chatStream');
        const entry = document.createElement('div');
        entry.className = `chat-item ${data.log_type || data.type || 'system'}`;
        
        const timeString = new Date(data.timestamp * 1000).toLocaleTimeString();
        
        // Calculate text metrics only for non-system messages
        let textMetrics = data.metadata || {};
        const isSystemMessage = (data.log_type || data.type || 'system') === 'system';
        
        if (!textMetrics.word_count && data.message && !isSystemMessage) {
            const words = data.message.split(' ');
            const sentences = data.message.split('.').filter(s => s.trim());
            textMetrics = {
                word_count: words.length,
                sentence_count: sentences.length,
                avg_sentence_length: sentences.length > 0 ? words.length / sentences.length : 0
            };
        }
        
        entry.innerHTML = `
            <div class="chat-avatar">
                ${this.getChatIcon(data.log_type || data.type || 'system')}
            </div>
            <div class="chat-content">
                <div class="chat-header">
                    <span class="chat-author">${data.author || 'System'}</span>
                    <span class="chat-type">${data.log_type || data.type || 'system'}</span>
                    <span class="chat-time">${timeString}</span>
                </div>
                <div class="chat-message">${this.renderMarkdown(data.message)}</div>
                ${!isSystemMessage && textMetrics.word_count ? `
                <div class="chat-metrics">
                    <span class="metric">Words: ${textMetrics.word_count}</span>
                    <span class="metric">Sentences: ${textMetrics.sentence_count}</span>
                    <span class="metric">Avg Length: ${textMetrics.avg_sentence_length ? textMetrics.avg_sentence_length.toFixed(1) : '0'}</span>
                </div>
                ` : ''}
            </div>
        `;
        
        chatStream.appendChild(entry);
        chatStream.scrollTop = chatStream.scrollHeight;
        
        // Remove old entries if too many
        while (chatStream.children.length > 200) {
            chatStream.removeChild(chatStream.firstChild);
        }
    }
    
    getChatIcon(type) {
        const icons = {
            thought: '<i class="fas fa-brain"></i>',
            choice: '<i class="fas fa-check-circle"></i>',
            communication: '<i class="fas fa-comments"></i>',
            tool_call: '<i class="fas fa-wrench"></i>',
            system: '<i class="fas fa-cog"></i>'
        };
        
        return icons[type] || icons.system;
    }
    
    renderMarkdown(text) {
        if (!text) return '';
        
        // Configure marked options for better rendering
        marked.setOptions({
            breaks: true,
            gfm: true,
            tables: true,
            smartLists: true,
            smartypants: true,
            highlight: function(code, lang) {
                // Basic syntax highlighting
                return `<pre><code class="language-${lang}">${this.escapeHtml(code)}</code></pre>`;
            }.bind(this)
        });
        
        // Parse markdown to HTML
        try {
            const html = marked.parse(text);
            return html;
        } catch (e) {
            console.error('Markdown parsing error:', e);
            // Fallback to plain text with basic line breaks
            return text.replace(/\n/g, '<br>');
        }
    }
    
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
    
    sanitizeInput(input) {
        if (!input) return '';
        return this.escapeHtml(input.toString().trim());
    }
    
    validateNumber(value, min, max) {
        if (!value) return null;
        const num = parseInt(value);
        if (isNaN(num) || num < min || num > max) {
            return null;
        }
        return num;
    }
    
    clearChatLogs() {
        const chatStream = document.getElementById('chatStream');
        chatStream.innerHTML = '';
    }
    
    exportChatLogs() {
        const logs = Array.from(document.querySelectorAll('.chat-item')).map(item => {
            const author = item.querySelector('.chat-author').textContent;
            const type = item.querySelector('.chat-type').textContent;
            const time = item.querySelector('.chat-time').textContent;
            const message = item.querySelector('.chat-message').textContent;
            return `[${time}] [${type}] ${author}: ${message}`;
        });
        
        const blob = new Blob([logs.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_logs_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    filterChatLogs(type) {
        const chatItems = document.querySelectorAll('.chat-item');
        chatItems.forEach(item => {
            if (!type || item.classList.contains(type)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    }
    
    // ========================================
    // Meetings
    // ========================================
    
    initializeMeetings() {
        this.loadMeetings();
        
        // Set up event handlers for meetings controls
        document.getElementById('clearMeetingsBtn').addEventListener('click', () => {
            this.clearMeetings();
        });
        
        document.getElementById('exportMeetingsBtn').addEventListener('click', () => {
            this.exportMeetings();
        });
        
        document.getElementById('meetingFilter').addEventListener('change', (e) => {
            this.filterMeetings(e.target.value);
        });
    }
    
    async loadMeetings() {
        try {
            const response = await fetch('/api/meetings');
            const data = await response.json();
            
            if (data.meetings) {
                data.meetings.forEach(meeting => {
                    this.displayMeetingEntry(meeting);
                });
            }
        } catch (error) {
            console.error('Error loading meetings:', error);
        }
    }
    
    addMeetingEntry(data) {
        // Handle nested meeting_data structure from virtual_lab
        const meetingData = data.meeting_data || data;
        const eventType = data.event_type || 'meeting_update';
        
        // Extract meeting information
        const meeting = {
            meeting_id: meetingData.meeting_id || 'unknown',
            topic: meetingData.topic || meetingData.phase || 'Research Meeting',
            participants: meetingData.participants || [],
            meeting_type: meetingData.meeting_type || 'team_meeting',
            phase: meetingData.phase || 'unknown',
            timestamp: data.timestamp || Date.now() / 1000,
            duration: meetingData.duration || 0,
            outcome: meetingData.outcome || '',
            transcript: meetingData.transcript || ''
        };
        
        // Handle different event types
        if (eventType === 'meeting_start') {
            // Update meeting indicator
            this.updateMeetingIndicator(true, meeting.topic);
            
            // Add to activity log
            this.addActivityLogEntry({
                type: 'meeting',
                author: 'System',
                message: `Meeting started: ${meeting.topic}`,
                timestamp: meeting.timestamp
            });
            
            // Add to chat log
            this.addChatLogEntry({
                type: 'communication',
                author: 'Meeting',
                message: `Meeting started: ${meeting.topic} with ${meeting.participants.length} participants`,
                timestamp: meeting.timestamp
            });
        } else if (eventType === 'meeting_end') {
            // Update meeting indicator
            this.updateMeetingIndicator(false);
            
            // Store meeting in database
            this.storeMeetingRecord(meeting);
            
            // Add to meetings panel
            this.displayMeetingEntry(meeting);
        }
    }
    
    displayMeetingEntry(data) {
        const meetingsList = document.getElementById('meetingsList');
        const meetingElement = document.createElement('div');
        meetingElement.className = 'meeting-item';
        
        // Parse participants if it's a string
        let participants = data.participants;
        if (typeof participants === 'string') {
            try {
                participants = JSON.parse(participants);
            } catch (e) {
                participants = [participants];
            }
        }
        if (!Array.isArray(participants)) {
            participants = [participants];
        }
        
        const participantsList = participants.join(', ');
        const timeStr = data.timestamp ? new Date(data.timestamp * 1000).toLocaleString() : 'Unknown time';
        
        meetingElement.innerHTML = `
            <div class="meeting-header">
                <div class="meeting-info">
                    <h4>${data.topic || data.meeting_id || 'Research Meeting'}</h4>
                    <span class="meeting-type">${this.getMeetingType(data.topic)}</span>
                </div>
                <span class="meeting-time">${timeStr}</span>
            </div>
            <div class="meeting-details">
                <div class="meeting-participants">
                    <span class="participants-label">Participants:</span>
                    <span class="participants-list">${participantsList}</span>
                </div>
                <div class="meeting-duration">
                    <span class="duration-label">Duration:</span>
                    <span class="duration-value">${data.duration || 0} minutes</span>
                </div>
                <div class="meeting-outcome">
                    ${data.outcome || 'Meeting in progress...'}
                </div>
                ${data.transcript ? `<div class="meeting-transcript">
                    <details>
                        <summary>View Transcript</summary>
                        <pre>${data.transcript}</pre>
                    </details>
                </div>` : ''}
            </div>
        `;
        
        // Insert at the top (after the welcome message)
        const firstChild = meetingsList.firstElementChild;
        if (firstChild && firstChild.classList.contains('system')) {
            meetingsList.insertBefore(meetingElement, firstChild.nextSibling);
        } else {
            meetingsList.appendChild(meetingElement);
        }
    }
    
    getMeetingType(topic) {
        if (!topic) return 'general';
        
        const topicLower = topic.toLowerCase();
        if (topicLower.includes('individual')) return 'individual';
        if (topicLower.includes('team')) return 'team';
        if (topicLower.includes('review') || topicLower.includes('critique')) return 'review';
        return 'general';
    }
    
    async storeMeetingRecord(meeting) {
        try {
            const response = await fetch('/api/meetings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSessionId || 'default',
                    meeting_id: meeting.meeting_id,
                    participants: JSON.stringify(meeting.participants),
                    topic: meeting.topic,
                    duration: meeting.duration,
                    outcome: meeting.outcome,
                    transcript: meeting.transcript
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to store meeting record');
            }
        } catch (error) {
            console.error('Error storing meeting record:', error);
        }
    }
    
    updateMeetingIndicator(isActive, topic = '') {
        const indicator = document.querySelector('.meeting-indicator');
        if (indicator) {
            const icon = indicator.querySelector('i');
            const text = indicator.querySelector('span');
            
            if (isActive) {
                indicator.classList.add('active');
                icon.className = 'fas fa-comments fa-pulse';
                text.textContent = topic || 'Meeting in progress';
            } else {
                indicator.classList.remove('active');
                icon.className = 'fas fa-comments';
                text.textContent = 'No active meeting';
            }
        }
    }
    
    clearMeetings() {
        const meetingsList = document.getElementById('meetingsList');
        const systemMessage = meetingsList.querySelector('.meeting-item.system');
        meetingsList.innerHTML = '';
        if (systemMessage) {
            meetingsList.appendChild(systemMessage);
        }
    }
    
    filterMeetings(type) {
        const meetings = document.querySelectorAll('#meetingsList .meeting-item:not(.system)');
        meetings.forEach(meeting => {
            const meetingType = meeting.querySelector('.meeting-type').textContent;
            if (!type || meetingType === type) {
                meeting.style.display = 'block';
            } else {
                meeting.style.display = 'none';
            }
        });
    }
    
    exportMeetings() {
        const meetings = Array.from(document.querySelectorAll('#meetingsList .meeting-item:not(.system)'));
        const meetingsData = meetings.map(meeting => {
            const topic = meeting.querySelector('h4').textContent;
            const type = meeting.querySelector('.meeting-type').textContent;
            const time = meeting.querySelector('.meeting-time').textContent;
            const participants = meeting.querySelector('.participants-list').textContent;
            const duration = meeting.querySelector('.duration-value')?.textContent || '';
            const outcome = meeting.querySelector('.meeting-outcome').textContent;
            
            return {
                topic,
                type,
                time,
                participants,
                duration,
                outcome
            };
        });
        
        const blob = new Blob([JSON.stringify(meetingsData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'meetings-export.json';
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
        // Populate API keys
        if (config.api_keys) {
            document.getElementById('openaiKey').value = config.api_keys.openai || '';
            document.getElementById('anthropicKey').value = config.api_keys.anthropic || '';
            document.getElementById('geminiKey').value = config.api_keys.gemini || '';
            document.getElementById('huggingfaceKey').value = config.api_keys.huggingface || '';
            document.getElementById('ollamaEndpoint').value = config.api_keys.ollama_endpoint || 'http://localhost:11434';
        }
        
        // Populate search API keys
        if (config.search_api_keys) {
            document.getElementById('googleSearchKey').value = config.search_api_keys.google_search || '';
            document.getElementById('googleSearchEngineId').value = config.search_api_keys.google_search_engine_id || '';
            document.getElementById('serpapiKey').value = config.search_api_keys.serpapi || '';
            document.getElementById('semanticScholarKey').value = config.search_api_keys.semantic_scholar || '';
            document.getElementById('openalexEmail').value = config.search_api_keys.openalex_email || '';
            document.getElementById('coreKey').value = config.search_api_keys.core || '';
        }
        
        // Populate framework settings
        if (config.framework) {
            document.getElementById('defaultProvider').value = config.framework.default_llm_provider || 'openai';
            document.getElementById('defaultModel').value = config.framework.default_model || 'gpt-4';
        }
        
        // Populate system settings
        if (config.system) {
            document.getElementById('outputDir').value = config.system.output_dir || '';
            document.getElementById('maxConcurrentAgents').value = config.system.max_concurrent_agents || 8;
            document.getElementById('autoSaveResults').checked = config.system.auto_save_results || false;
            document.getElementById('enableNotifications').checked = config.system.enable_notifications || false;
        }
        
        // Populate free options
        if (config.free_options) {
            document.getElementById('enableFreeSearch').checked = config.free_options.enable_free_search !== false;
            document.getElementById('enableMockResponses').checked = config.free_options.enable_mock_responses !== false;
        }
        
        // Update API key status indicators
        this.updateApiKeyStatuses(config.api_keys_configured);
        this.updateSearchApiKeyStatuses(config.search_api_keys_configured);
        
        // Update available providers
        this.updateAvailableProviders(config.available_providers || []);
    }
    
    updateApiKeyStatuses(apiKeysConfigured) {
        const statusElements = {
            openai: document.getElementById('openaiStatus'),
            anthropic: document.getElementById('anthropicStatus'),
            gemini: document.getElementById('geminiStatus'),
            huggingface: document.getElementById('huggingfaceStatus'),
            ollama: document.getElementById('ollamaStatus')
        };
        
        Object.entries(statusElements).forEach(([provider, element]) => {
            if (element) {
                const isConfigured = apiKeysConfigured[provider];
                element.textContent = isConfigured ? '✓ Configured' : '✗ Not configured';
                element.className = `key-status ${isConfigured ? 'valid' : 'invalid'}`;
            }
        });
    }
    
    updateSearchApiKeyStatuses(searchApiKeysConfigured) {
        const statusElements = {
            google_search: document.getElementById('googleSearchStatus'),
            google_search_engine_id: document.getElementById('googleSearchEngineIdStatus'),
            serpapi: document.getElementById('serpapiStatus'),
            semantic_scholar: document.getElementById('semanticScholarStatus'),
            openalex_email: document.getElementById('openalexStatus'),
            core: document.getElementById('coreStatus')
        };
        
        Object.entries(statusElements).forEach(([provider, element]) => {
            if (element) {
                const isConfigured = searchApiKeysConfigured[provider];
                element.textContent = isConfigured ? '✓ Configured' : '✗ Not configured';
                element.className = `key-status ${isConfigured ? 'valid' : 'invalid'}`;
            }
        });
    }
    
    updateAvailableProviders(providers) {
        const defaultProviderSelect = document.getElementById('defaultProvider');
        const currentValue = defaultProviderSelect.value;
        
        // Clear existing options
        defaultProviderSelect.innerHTML = '';
        
        // Add available providers
        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider;
            option.textContent = this.getProviderDisplayName(provider);
            defaultProviderSelect.appendChild(option);
        });
        
        // Restore previous selection if still available
        if (providers.includes(currentValue)) {
            defaultProviderSelect.value = currentValue;
        } else if (providers.length > 0) {
            defaultProviderSelect.value = providers[0];
        }
    }
    
    getProviderDisplayName(provider) {
        const names = {
            openai: 'OpenAI (GPT-4)',
            anthropic: 'Anthropic (Claude)',
            gemini: 'Google (Gemini)',
            huggingface: 'HuggingFace',
            ollama: 'Ollama (Local)'
        };
        return names[provider] || provider;
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
                anthropic: document.getElementById('anthropicKey').value,
                gemini: document.getElementById('geminiKey').value,
                huggingface: document.getElementById('huggingfaceKey').value,
                ollama_endpoint: document.getElementById('ollamaEndpoint').value
            },
            search_api_keys: {
                google_search: document.getElementById('googleSearchKey').value,
                google_search_engine_id: document.getElementById('googleSearchEngineId').value,
                serpapi: document.getElementById('serpapiKey').value,
                semantic_scholar: document.getElementById('semanticScholarKey').value,
                openalex_email: document.getElementById('openalexEmail').value,
                core: document.getElementById('coreKey').value
            },
            framework: {
                default_llm_provider: document.getElementById('defaultProvider').value,
                default_model: document.getElementById('defaultModel').value
            },
            system: {
                output_dir: document.getElementById('outputDir').value,
                max_concurrent_agents: parseInt(document.getElementById('maxConcurrentAgents').value),
                auto_save_results: document.getElementById('autoSaveResults').checked,
                enable_notifications: document.getElementById('enableNotifications').checked
            },
            free_options: {
                enable_free_search: document.getElementById('enableFreeSearch').checked,
                enable_mock_responses: document.getElementById('enableMockResponses').checked
            }
        };
        
        // Validate API keys
        const validationErrors = this.validateApiKeys(config.api_keys);
        if (validationErrors.length > 0) {
            this.showNotification('error', 'Validation Error', validationErrors.join('\n'));
            return;
        }

        // Validate search API keys
        const searchValidationErrors = this.validateSearchApiKeys(config.search_api_keys);
        if (searchValidationErrors.length > 0) {
            this.showNotification('error', 'Validation Error', searchValidationErrors.join('\n'));
            return;
        }
        
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
                this.updateApiKeyStatuses(result.api_keys_configured);
                this.updateSearchApiKeyStatuses(result.search_api_keys_configured);
                this.updateAvailableProviders(result.available_providers || []);
            } else {
                this.showNotification('error', 'Error', result.error);
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showNotification('error', 'Error', 'Failed to save settings');
        }
    }
    
    validateApiKeys(apiKeys) {
        const errors = [];
        
        // Validate OpenAI key format
        if (apiKeys.openai && !apiKeys.openai.startsWith('sk-')) {
            errors.push('OpenAI API key should start with "sk-"');
        }
        
        // Validate Anthropic key format
        if (apiKeys.anthropic && !apiKeys.anthropic.startsWith('sk-ant-')) {
            errors.push('Anthropic API key should start with "sk-ant-"');
        }
        
        // Validate Gemini key format
        if (apiKeys.gemini && !apiKeys.gemini.startsWith('AIza')) {
            errors.push('Google Gemini API key should start with "AIza"');
        }
        
        // Validate HuggingFace key format
        if (apiKeys.huggingface && !apiKeys.huggingface.startsWith('hf_')) {
            errors.push('HuggingFace API key should start with "hf_"');
        }
        
        // Validate Ollama endpoint format
        if (apiKeys.ollama_endpoint && !apiKeys.ollama_endpoint.startsWith('http')) {
            errors.push('Ollama endpoint should be a valid HTTP URL');
        }
        
        return errors;
    }

    validateSearchApiKeys(searchApiKeys) {
        const errors = [];

        if (searchApiKeys.google_search && !searchApiKeys.google_search.startsWith('AIza')) {
            errors.push('Google Search API key should start with "AIza"');
        }

        if (searchApiKeys.google_search_engine_id && !searchApiKeys.google_search_engine_id.startsWith('01')) {
            errors.push('Google Search Engine ID should start with "01"');
        }

        if (searchApiKeys.serpapi && searchApiKeys.serpapi.length < 10) {
            errors.push('SerpAPI key should be at least 10 characters long');
        }

        if (searchApiKeys.semantic_scholar && !searchApiKeys.semantic_scholar.startsWith('hf_')) {
            errors.push('Semantic Scholar API key should start with "hf_"');
        }

        if (searchApiKeys.openalex_email && !searchApiKeys.openalex_email.includes('@')) {
            errors.push('OpenAlex email should be a valid email address');
        }

        if (searchApiKeys.core && !searchApiKeys.core.startsWith('sk-')) {
            errors.push('Core API key should start with "sk-"');
        }

        return errors;
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
        // Debounce frequent updates
        if (this.metricsUpdateTimeout) {
            clearTimeout(this.metricsUpdateTimeout);
        }
        
        this.metricsUpdateTimeout = setTimeout(() => {
            // Update CPU usage
            const cpuElement = document.getElementById('cpuUsage');
            if (cpuElement && metrics.cpu_usage !== undefined) {
                cpuElement.textContent = `${Math.round(metrics.cpu_usage)}%`;
            }
            
            // Update memory info if available
            if (metrics.memory_usage !== undefined) {
                const memoryCard = document.querySelector('.metric-card .metric-content');
                if (memoryCard) {
                    const existingMemory = memoryCard.querySelector('.metric-sub');
                    if (existingMemory) {
                        existingMemory.textContent = `Memory: ${Math.round(metrics.memory_usage)}%`;
                    } else {
                        memoryCard.innerHTML += `<div class="metric-sub">Memory: ${Math.round(metrics.memory_usage)}%</div>`;
                    }
                }
            }
            
            // Update active agents count
            const agentsElement = document.getElementById('activeAgentsCount');
            if (agentsElement && metrics.active_agents !== undefined) {
                agentsElement.textContent = metrics.active_agents;
            }
        }, 100); // Debounce to 100ms
    }
    
    async startSystemMonitoring() {
        // Start monitoring system metrics
        const monitorMetrics = async () => {
            try {
                const response = await fetch('/api/metrics?timeframe=session');
                const data = await response.json();
                
                if (data.current) {
                    this.updateSystemMetrics(data.current);
                }
                
                if (data.agent_stats) {
                    // Update agent statistics
                    document.getElementById('avgAgentScore').textContent = 
                        data.agent_stats.avg_quality_score || '0.0';
                    document.getElementById('activeAgentsCount').textContent = 
                        data.agent_stats.active_agents || '0';
                    document.getElementById('criticalIssues').textContent = 
                        data.agent_stats.critical_issues || '0';
                }
                
                if (data.session_stats) {
                    document.getElementById('totalSessions').textContent = 
                        data.session_stats.total_sessions || '0';
                    document.getElementById('successfulSessions').textContent = 
                        data.session_stats.successful_sessions || '0';
                }
                
            } catch (error) {
                console.error('Error monitoring metrics:', error);
            }
        };
        
        // Start monitoring
        monitorMetrics();
        this.intervals.add(setInterval(monitorMetrics, 10000)); // Every 10 seconds
        
        // Refresh agents periodically
        const refreshAgents = () => {
            this.loadAllAgents();
        };
        
        refreshAgents();
        this.intervals.add(setInterval(refreshAgents, 30000)); // Every 30 seconds
    }
    
    refreshCurrentPanelData() {
        // Get currently active panel
        const activePanel = document.querySelector('.panel.active');
        if (!activePanel) return;
        
        const panelId = activePanel.id;
        
        switch (panelId) {
            case 'agents-panel':
                this.updateAgentsList();
                break;
            case 'history-panel':
                this.loadHistory();
                break;
            case 'meetings-panel':
                this.loadMeetings();
                break;
            case 'chat-logs-panel':
                this.loadChatLogs();
                break;
            case 'metrics-panel':
                // Metrics already refreshed by startSystemMonitoring
                break;
            default:
                // For other panels, no specific refresh needed
                break;
        }
    }

    startHeartbeat() {
        // Send heartbeat every 30 seconds to keep connection alive
        const heartbeatInterval = setInterval(() => {
            if (this.socket && this.socket.connected) {
                this.socket.emit('heartbeat');
            }
        }, 30000);
        
        this.intervals.add(heartbeatInterval);
    }
    
    cleanup() {
        // Clear all intervals
        this.intervals.forEach(interval => clearInterval(interval));
        this.intervals.clear();
        
        // Remove all event listeners
        this.eventListeners.forEach((listener, element) => {
            element.removeEventListener(listener.type, listener.handler);
        });
        this.eventListeners.clear();
        
        // Disconnect socket
        if (this.socket) {
            this.socket.disconnect();
        }
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

    updateMeetingIndicator(activeMeeting = false) {
        const indicator = this.meetingIndicator;
        if (!indicator) return;
        
        if (activeMeeting) {
            indicator.innerHTML = `
                <i class="fas fa-comments"></i>
                <span>Active Meeting</span>
                <div class="meeting-pulse"></div>
            `;
            indicator.classList.add('active');
        } else {
            indicator.innerHTML = `
                <i class="fas fa-comments"></i>
                <span>No active meeting</span>
            `;
            indicator.classList.remove('active');
        }
    }

    async loadHistory() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            
            if (data.sessions) {
                data.sessions.forEach(session => {
                    this.displayHistoryEntry(session);
                });
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }
    
    displayHistoryEntry(session) {
        const historyList = document.getElementById('historyList');
        const historyElement = document.createElement('div');
        historyElement.className = 'history-item';
        historyElement.dataset.sessionId = session.session_id;
        
        const startTime = new Date(session.start_time * 1000);
        const endTime = session.end_time ? new Date(session.end_time * 1000) : null;
        const duration = endTime ? Math.round((endTime - startTime) / 1000 / 60) : 'Ongoing';
        
        historyElement.innerHTML = `
            <div class="history-header">
                <div class="history-info">
                    <h4>Session ${session.session_id}</h4>
                    <span class="history-type">research</span>
                </div>
                <span class="history-time">${startTime.toLocaleString()}</span>
            </div>
            <div class="history-details">
                <div class="history-stats">
                    <div class="history-stat">
                        <span class="stat-label">Duration:</span>
                        <span class="stat-value">${duration} min</span>
                    </div>
                    <div class="history-stat">
                        <span class="stat-label">Activities:</span>
                        <span class="stat-value">${session.activity_count}</span>
                    </div>
                </div>
                <div class="history-actions">
                    <button class="btn btn-ghost btn-sm" onclick="app.viewSession('${session.session_id}')">
                        <i class="fas fa-eye"></i>
                        View Details
                    </button>
                    <button class="btn btn-ghost btn-sm" onclick="app.exportSession('${session.session_id}')">
                        <i class="fas fa-download"></i>
                        Export
                    </button>
                </div>
            </div>
        `;
        
        // Insert at the top (after the welcome message)
        const firstChild = historyList.firstElementChild;
        if (firstChild && firstChild.classList.contains('system')) {
            historyList.insertBefore(historyElement, firstChild.nextSibling);
        } else {
            historyList.appendChild(historyElement);
        }
    }
    
    async viewSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            const data = await response.json();
            
            if (data.error) {
                this.showNotification('error', 'Error', data.error);
                return;
            }
            
            // Show session details in a modal or new panel
            this.showSessionDetails(data);
        } catch (error) {
            console.error('Error viewing session:', error);
            this.showNotification('error', 'Error', 'Failed to load session details');
        }
    }
    
    showSessionDetails(sessionData) {
        // Create a modal to show session details
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h2>Session ${sessionData.session.session_id}</h2>
                    <button class="btn btn-ghost" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="session-info">
                        <h3>Session Information</h3>
                        <p><strong>Start Time:</strong> ${new Date(sessionData.session.start_time * 1000).toLocaleString()}</p>
                        <p><strong>End Time:</strong> ${sessionData.session.end_time ? new Date(sessionData.session.end_time * 1000).toLocaleString() : 'Ongoing'}</p>
                        <p><strong>Activities:</strong> ${sessionData.session.activity_count}</p>
                    </div>
                    <div class="session-activities">
                        <h3>Activities (${sessionData.activities.length})</h3>
                        <div class="activity-list">
                            ${sessionData.activities.map(activity => `
                                <div class="activity-item">
                                    <span class="activity-time">${new Date(activity.timestamp * 1000).toLocaleTimeString()}</span>
                                    <span class="activity-agent">${activity.agent_id}</span>
                                    <span class="activity-message">${activity.activity}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div class="session-chat">
                        <h3>Chat Log (${sessionData.chat_logs.length})</h3>
                        <div class="chat-list">
                            ${sessionData.chat_logs.map(log => `
                                <div class="chat-item ${log.log_type}">
                                    <span class="chat-time">${new Date(log.timestamp * 1000).toLocaleTimeString()}</span>
                                    <span class="chat-author">${log.author}</span>
                                    <span class="chat-message">${log.message}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('show'), 10);
    }
    
    async exportSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            const data = await response.json();
            
            if (data.error) {
                this.showNotification('error', 'Error', data.error);
                return;
            }
            
            // Create export data
            const exportData = {
                session: data.session,
                activities: data.activities,
                chat_logs: data.chat_logs,
                meetings: data.meetings,
                export_time: new Date().toISOString()
            };
            
            // Download as JSON
            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `session_${sessionId}_${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            this.showNotification('success', 'Exported', 'Session data exported successfully');
        } catch (error) {
            console.error('Error exporting session:', error);
            this.showNotification('error', 'Error', 'Failed to export session');
        }
    }
    
    clearHistory() {
        const historyList = document.getElementById('historyList');
        const systemMessage = historyList.querySelector('.history-item.system');
        historyList.innerHTML = '';
        if (systemMessage) {
            historyList.appendChild(systemMessage);
        }
    }
    
    filterHistory(filter) {
        const historyItems = document.querySelectorAll('#historyList .history-item:not(.system)');
        historyItems.forEach(item => {
            const sessionId = item.dataset.sessionId;
            const startTime = new Date(sessionId.split('_')[1] * 1000);
            const now = new Date();
            
            let show = true;
            switch (filter) {
                case 'recent':
                    show = (now - startTime) < 24 * 60 * 60 * 1000; // 24 hours
                    break;
                case 'week':
                    show = (now - startTime) < 7 * 24 * 60 * 60 * 1000; // 7 days
                    break;
                case 'month':
                    show = (now - startTime) < 30 * 24 * 60 * 60 * 1000; // 30 days
                    break;
                default:
                    show = true;
            }
            
            item.style.display = show ? 'block' : 'none';
        });
    }
    
    exportHistory() {
        const historyItems = Array.from(document.querySelectorAll('#historyList .history-item:not(.system)'));
        const exportData = historyItems.map(item => {
            const sessionId = item.dataset.sessionId;
            const title = item.querySelector('h4').textContent;
            const time = item.querySelector('.history-time').textContent;
            const stats = Array.from(item.querySelectorAll('.history-stat')).map(stat => {
                const label = stat.querySelector('.stat-label').textContent;
                const value = stat.querySelector('.stat-value').textContent;
                return `${label} ${value}`;
            }).join(', ');
            
            return `[${time}] ${title} - ${stats}`;
        });
        
        const blob = new Blob([exportData.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session_history_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiResearchLab = new AIResearchLabApp();
    // Create global alias for easier access
    window.app = window.aiResearchLab;
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