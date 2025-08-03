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
        this.meetings = [];
        this.currentPhase = 0;
        this.isResearchActive = false;
        
        this.init();
    }
    
    init() {
        try {
            // Add a simple status indicator to show the app is working
            this.addStatusIndicator();
            
            this.initializeUI();
            this.initializeWebSocket();
            this.loadConfiguration();
            this.startSystemMonitoring();
            
            // Load initial data
            this.loadInitialData();
            
            console.log('AI Research Lab Interface initialized');
        } catch (error) {
            console.error('Error initializing AI Research Lab Interface:', error);
            this.showNotification('error', 'Initialization Error', 'Failed to initialize the interface');
        }
    }
    
    addStatusIndicator() {
        // Add a simple status indicator to the header
        const headerControls = document.querySelector('.header-controls');
        if (headerControls) {
            const statusDiv = document.createElement('div');
            statusDiv.className = 'status-indicator';
            statusDiv.id = 'appStatus';
            statusDiv.innerHTML = `
                <div class="status-dot online"></div>
                <span>App Ready</span>
            `;
            headerControls.appendChild(statusDiv);
        }
    }
    
    async loadInitialData() {
        try {
            // Load historical data from persistent storage
            await this.loadHistoricalData();
            
            // Load initial chat logs
            await this.loadChatLogs();
            
            // Load initial meetings
            await this.loadMeetings();
            
            // Load initial agent activity
            await this.refreshAgents();
            
            // Load metrics
            await this.loadMetrics();
            
            // Show welcome notification
            this.showNotification('success', 'Welcome', 'AI Research Lab Interface loaded successfully');
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('error', 'Data Load Error', 'Some data failed to load. Check console for details.');
        }
    }
    
    async loadHistoricalData() {
        try {
            const response = await fetch('/api/data/history');
            const data = await response.json();
            
            if (data.error) {
                console.warn('Could not load historical data:', data.error);
                return;
            }
            
            console.log('Loaded historical data:', {
                sessions: data.total_sessions,
                chatLogs: data.total_chat_logs,
                agentActivity: data.total_agent_activities,
                meetings: data.total_meetings
            });
            
            // Load recent sessions into the UI
            if (data.sessions && data.sessions.length > 0) {
                this.loadSessionsIntoUI(data.sessions);
            }
            
            // Load recent chat logs
            if (data.chat_logs && data.chat_logs.length > 0) {
                this.loadChatLogsIntoUI(data.chat_logs);
            }
            
            // Load recent agent activity
            if (data.agent_activity && data.agent_activity.length > 0) {
                this.loadAgentActivityIntoUI(data.agent_activity);
            }
            
            // Load recent meetings
            if (data.meetings && data.meetings.length > 0) {
                this.loadMeetingsIntoUI(data.meetings);
            }
            
            // Handle active sessions
            if (data.active_sessions && data.active_sessions.length > 0) {
                this.handleActiveSessions(data.active_sessions);
            }
            
        } catch (error) {
            console.error('Error loading historical data:', error);
        }
    }
    
    loadSessionsIntoUI(sessions) {
        // Update session list if it exists
        const sessionList = document.getElementById('sessionList');
        if (sessionList) {
            sessions.forEach(session => {
                const sessionElement = this.createSessionElement(session);
                sessionList.appendChild(sessionElement);
            });
        }
    }
    
    createSessionElement(session) {
        const div = document.createElement('div');
        div.className = 'session-item';
        div.innerHTML = `
            <div class="session-header">
                <span class="session-id">${session.id}</span>
                <span class="session-status ${session.status}">${session.status}</span>
            </div>
            <div class="session-question">${session.research_question || 'No question'}</div>
            <div class="session-date">${new Date(session.created_at).toLocaleString()}</div>
        `;
        return div;
    }
    
    loadChatLogsIntoUI(chatLogs) {
        const chatContainer = document.getElementById('chatLogs');
        if (chatContainer) {
            chatLogs.forEach(log => {
                this.addChatLogEntry(log);
            });
        }
    }
    
    loadAgentActivityIntoUI(activities) {
        activities.forEach(activity => {
            this.updateAgentActivity(activity);
        });
    }
    
    loadMeetingsIntoUI(meetings) {
        const meetingsContainer = document.getElementById('meetingsList');
        if (meetingsContainer) {
            meetings.forEach(meeting => {
                this.createMeetingEntry(meeting);
            });
        }
    }
    
    handleActiveSessions(activeSessions) {
        if (activeSessions.length > 0) {
            // If there are active sessions, show a notification
            this.showNotification('info', 'Active Sessions Found', 
                `Found ${activeSessions.length} active research session(s). You can continue from where you left off.`);
        }
    }
    
    async loadMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            if (data.current) {
                this.updateSystemMetrics(data.current);
            }
            
            if (data.session_stats) {
                const totalSessions = document.getElementById('totalSessions');
                const successfulSessions = document.getElementById('successfulSessions');
                
                if (totalSessions) {
                    totalSessions.textContent = data.session_stats.total_sessions || 0;
                }
                if (successfulSessions) {
                    successfulSessions.textContent = data.session_stats.successful_sessions || 0;
                }
            }
            
        } catch (error) {
            console.error('Error loading metrics:', error);
        }
    }
    
    showSkeletonLoading(containerId, type = 'default') {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const skeletonTemplates = {
            'activity': `
                <div class="skeleton-item">
                    <div class="skeleton-avatar"></div>
                    <div class="skeleton-content">
                        <div class="skeleton-line skeleton-title"></div>
                        <div class="skeleton-line skeleton-text"></div>
                    </div>
                </div>
            `,
            'chat': `
                <div class="skeleton-chat-item">
                    <div class="skeleton-avatar"></div>
                    <div class="skeleton-content">
                        <div class="skeleton-line skeleton-header"></div>
                        <div class="skeleton-line skeleton-message"></div>
                    </div>
                </div>
            `,
            'meeting': `
                <div class="skeleton-meeting-item">
                    <div class="skeleton-line skeleton-title"></div>
                    <div class="skeleton-line skeleton-info"></div>
                    <div class="skeleton-line skeleton-status"></div>
                </div>
            `,
            'agent': `
                <div class="skeleton-agent-item">
                    <div class="skeleton-avatar"></div>
                    <div class="skeleton-content">
                        <div class="skeleton-line skeleton-name"></div>
                        <div class="skeleton-line skeleton-expertise"></div>
                    </div>
                </div>
            `,
            'metric': `
                <div class="skeleton-metric-item">
                    <div class="skeleton-line skeleton-value"></div>
                    <div class="skeleton-line skeleton-label"></div>
                </div>
            `
        };
        
        const template = skeletonTemplates[type] || skeletonTemplates['default'];
        const skeletonCount = type === 'metric' ? 4 : 5;
        
        container.innerHTML = '';
        for (let i = 0; i < skeletonCount; i++) {
            container.innerHTML += template;
        }
    }
    
    hideSkeletonLoading(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Remove skeleton elements
        const skeletons = container.querySelectorAll('.skeleton-item, .skeleton-chat-item, .skeleton-meeting-item, .skeleton-agent-item, .skeleton-metric-item');
        skeletons.forEach(skeleton => skeleton.remove());
    }
    
    // ========================================
    // WebSocket Connection
    // ========================================
    
    initializeWebSocket() {
        console.log('Initializing WebSocket connection...');
        
        this.socket = io({
            timeout: 60000,           // 60 second connection timeout
            reconnection: true,       // Enable reconnection
            reconnectionDelay: 1000,  // Initial delay for reconnection
            reconnectionDelayMax: 5000, // Max delay for reconnection
            reconnectionAttempts: 10, // Max reconnection attempts
            forceNew: false,         // Don't force new connection if one exists
            transports: ['websocket', 'polling'] // Allow both transports
        });
        
        console.log('WebSocket object created, setting up event handlers...');
        this.setupWebSocketEventHandlers();
    }

    setupWebSocketEventHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateSystemStatus('online');
            this.showNotification('success', 'Connected', 'Successfully connected to server');
        });

        this.socket.on('disconnect', (reason) => {
            console.log('Disconnected from server:', reason);
            this.updateSystemStatus('offline');
            if (reason === 'io server disconnect') {
                // Server initiated disconnect, try to reconnect
                this.socket.connect();
            }
            this.showNotification('warning', 'Disconnected', `Connection lost: ${reason}`);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateSystemStatus('error');
            this.showNotification('error', 'Connection Error', 'Failed to connect to server');
        });

        this.socket.on('reconnect', (attemptNumber) => {
            console.log('Reconnected after', attemptNumber, 'attempts');
            this.updateSystemStatus('online');
            this.showNotification('success', 'Reconnected', `Reconnected after ${attemptNumber} attempts`);
        });

        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log('Attempting to reconnect...', attemptNumber);
            this.updateSystemStatus('reconnecting');
        });

        this.socket.on('reconnect_error', (error) => {
            console.error('Reconnection error:', error);
            this.updateSystemStatus('error');
        });

        this.socket.on('reconnect_failed', () => {
            console.error('Failed to reconnect');
            this.updateSystemStatus('failed');
            this.showNotification('error', 'Connection Failed', 'Unable to reconnect to server');
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
        console.log('Initializing UI components...');
        
        try {
            // Check if required elements exist
            const requiredElements = [
                'researchQuestion',
                'startResearchBtn',
                'agentsContainer',
                'activityStream',
                'chatStream',
                'notificationsContainer'
            ];
            
            const missingElements = requiredElements.filter(id => !document.getElementById(id));
            if (missingElements.length > 0) {
                console.error('Missing required elements:', missingElements);
                throw new Error(`Missing required elements: ${missingElements.join(', ')}`);
            }
            
            this.initializeNavigation();
            console.log('Navigation initialized');
            
            this.initializeResearchConfig();
            console.log('Research config initialized');
            
            this.initializeAgentVisualization();
            console.log('Agent visualization initialized');
            
            this.initializeProgressTracking();
            console.log('Progress tracking initialized');
            
            this.initializeActivityLog();
            console.log('Activity log initialized');
            
            this.initializeChatLogs();
            console.log('Chat logs initialized');
            
            this.initializeMeetings();
            console.log('Meetings initialized');
            
            this.initializeSettings();
            console.log('Settings initialized');
            
            this.initializeConstraints();
            console.log('Constraints initialized');
            
            console.log('All UI components initialized successfully');
        } catch (error) {
            console.error('Error initializing UI components:', error);
            this.showNotification('error', 'UI Error', 'Failed to initialize some UI components: ' + error.message);
        }
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
        
        // Add refresh button event handler
        const refreshBtn = document.getElementById('refreshAgentsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshAgents();
            });
        }
    }
    
    async refreshAgents() {
        try {
            // Show loading state
            const refreshBtn = document.getElementById('refreshAgentsBtn');
            const icon = refreshBtn.querySelector('i');
            icon.className = 'fas fa-spinner fa-spin';
            
            // Show skeleton loading for agents list
            this.showSkeletonLoading('agentsList', 'agent');
            
            // Fetch current agent activity
            const response = await fetch('/api/agent-activity');
            const data = await response.json();
            
            // Hide skeleton loading
            this.hideSkeletonLoading('agentsList');
            
            if (data.activities && data.activities.length > 0) {
                // Clear existing agents
                this.agents.clear();
                this.agentsContainer.innerHTML = '';
                
                // Process each activity to rebuild agent list
                data.activities.forEach(activity => {
                    this.updateAgentActivity({
                        agent_id: activity.agent_id,
                        name: activity.agent_id,
                        expertise: 'Research Agent',
                        status: activity.status || 'idle',
                        activity: activity.message || '',
                        type: activity.activity_type
                    });
                });
            } else {
                // Show empty state
                const agentsList = document.getElementById('agentsList');
                if (agentsList) {
                    agentsList.innerHTML = '<div class="empty-state">No agents available</div>';
                }
                const agentsContainer = document.getElementById('agentsContainer');
                if (agentsContainer) {
                    agentsContainer.innerHTML = '<div class="empty-state">No agents active</div>';
                }
            }
            
            // Reset button state
            icon.className = 'fas fa-sync-alt';
            this.showNotification('success', 'Agents Refreshed', 'Agent list updated successfully');
            
        } catch (error) {
            console.error('Error refreshing agents:', error);
            this.hideSkeletonLoading('agentsList');
            this.showNotification('error', 'Refresh Error', 'Failed to refresh agents');
            
            // Reset button state
            const refreshBtn = document.getElementById('refreshAgentsBtn');
            const icon = refreshBtn.querySelector('i');
            icon.className = 'fas fa-sync-alt';
            
            // Show error state
            const agentsList = document.getElementById('agentsList');
            if (agentsList) {
                agentsList.innerHTML = '<div class="empty-state">Failed to load agents</div>';
            }
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

        // Determine this agent's index among currently known agents to place them evenly in a circle
        const agentIds = Array.from(this.agents.keys());
        const index = agentIds.indexOf(agentId);
        const total = agentIds.length || 1;

        // Calculate position around circle based on index (avoids NaN for string IDs)
        const angle = (index * 2 * Math.PI) / total;
        const x = centerX + radius * Math.cos(angle) - 30; // 30 = half avatar width
        const y = centerY + radius * Math.sin(angle) - 30; // 30 = half avatar height

        avatar.style.left = `${x}px`;
        avatar.style.top = `${y}px`;
    }
    
    updateAgentActivity(data) {
        const agentId = data.agent_id;
        let agent = this.agents.get(agentId);
        
        if (!agent) {
            // Create new agent with better naming
            const agentName = this.getAgentDisplayName(agentId);
            const agentExpertise = this.getAgentExpertise(agentId);
            
            agent = {
                id: agentId,
                name: agentName,
                expertise: agentExpertise,
                status: data.status || 'idle',
                activity: data.activity || data.type || ''
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
            agent.activity = data.activity || data.type || agent.activity;
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
    
    getAgentDisplayName(agentId) {
        const nameMap = {
            'principal_investigator': 'Principal Investigator',
            'test_agent': 'Research Assistant',
            'scientific_critic': 'Scientific Critic',
            'domain_expert': 'Domain Expert',
            'data_analyst': 'Data Analyst',
            'literature_reviewer': 'Literature Reviewer'
        };
        
        return nameMap[agentId] || `Agent ${agentId.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}`;
    }
    
    getAgentExpertise(agentId) {
        const expertiseMap = {
            'principal_investigator': 'Research Leadership',
            'test_agent': 'General Research',
            'scientific_critic': 'Scientific Review',
            'domain_expert': 'Specialized Knowledge',
            'data_analyst': 'Data Analysis',
            'literature_reviewer': 'Literature Reviewer'
        };
        
        return expertiseMap[agentId] || 'Research Support';
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
    // Chat Logs
    // ========================================
    
    async loadChatLogs() {
        try {
            // Show skeleton loading
            this.showSkeletonLoading('chatStream', 'chat');
            
            const response = await fetch('/api/chat-logs');
            const data = await response.json();
            
            // Hide skeleton loading
            this.hideSkeletonLoading('chatStream');
            
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(log => {
                    this.addChatLogEntry(log);
                });
            } else {
                // Show empty state
                const chatStream = document.getElementById('chatStream');
                if (chatStream) {
                    chatStream.innerHTML = '<div class="empty-state">No chat logs available</div>';
                }
            }
        } catch (error) {
            console.error('Error loading chat logs:', error);
            this.hideSkeletonLoading('chatStream');
            const chatStream = document.getElementById('chatStream');
            if (chatStream) {
                chatStream.innerHTML = '<div class="empty-state">Failed to load chat logs</div>';
            }
        }
    }
    
    addChatLogEntry(data) {
        const chatStream = document.getElementById('chatStream');
        const entry = document.createElement('div');
        entry.className = `chat-item ${data.type || 'system'}`;
        
        const timeString = new Date(data.timestamp * 1000).toLocaleTimeString();
        
        entry.innerHTML = `
            <div class="chat-avatar">
                ${this.getChatIcon(data.type || 'system')}
            </div>
            <div class="chat-content">
                <div class="chat-header">
                    <span class="chat-author">${data.author || 'System'}</span>
                    <span class="chat-type">${data.type || 'system'}</span>
                    <span class="chat-time">${timeString}</span>
                </div>
                <div class="chat-message">${data.message}</div>
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
        this.setupMeetingsEventHandlers();
    }
    
    setupMeetingsEventHandlers() {
        // Handle meeting selection
        document.addEventListener('click', (e) => {
            if (e.target.closest('.meeting-entry')) {
                const meetingId = e.target.closest('.meeting-entry').dataset.meetingId;
                this.showMeetingDetails(meetingId);
            }
        });
    }
    
    async loadMeetings() {
        try {
            // Show skeleton loading
            this.showSkeletonLoading('meetingsList', 'meeting');
            
            const response = await fetch('/api/meetings');
            const data = await response.json();
            
            // Hide skeleton loading
            this.hideSkeletonLoading('meetingsList');
            
            if (data.meetings && data.meetings.length > 0) {
                this.meetings = data.meetings;
                this.updateMeetingsList(data.meetings);
                this.updateMeetingsStats(data.meetings);
            } else {
                // Show empty state
                const meetingsList = document.getElementById('meetingsList');
                if (meetingsList) {
                    meetingsList.innerHTML = '<div class="empty-state">No meetings recorded yet</div>';
                }
            }
        } catch (error) {
            console.error('Error loading meetings:', error);
            this.hideSkeletonLoading('meetingsList');
            const meetingsList = document.getElementById('meetingsList');
            if (meetingsList) {
                meetingsList.innerHTML = '<div class="empty-state">Failed to load meetings</div>';
            }
        }
    }
    
    updateMeetingsList(meetings) {
        const meetingsList = document.getElementById('meetingsList');
        if (!meetingsList) return;
        
        meetingsList.innerHTML = '';
        
        if (!meetings || meetings.length === 0) {
            meetingsList.innerHTML = '<div class="empty-state">No meetings recorded yet</div>';
            return;
        }
        
        meetings.forEach(meeting => {
            const meetingEntry = this.createMeetingEntry(meeting);
            meetingsList.appendChild(meetingEntry);
        });
    }
    
    createMeetingEntry(meeting) {
        const entry = document.createElement('div');
        entry.className = 'meeting-entry';
        entry.dataset.meetingId = meeting.meeting_id;
        
        const duration = this.formatDuration(meeting.duration);
        const timestamp = new Date(meeting.timestamp * 1000).toLocaleString();
        
        entry.innerHTML = `
            <div class="meeting-header">
                <h4>${meeting.topic}</h4>
                <span class="meeting-time">${timestamp}</span>
            </div>
            <div class="meeting-info">
                <span class="meeting-duration">${duration}</span>
                <span class="meeting-participants">${meeting.participants} participants</span>
            </div>
            <div class="meeting-outcome">
                ${meeting.outcome ? '✓ Completed' : '⏳ In Progress'}
            </div>
        `;
        
        return entry;
    }
    
    updateMeetingsStats(meetings) {
        const totalMeetings = document.getElementById('totalMeetings');
        const avgDuration = document.getElementById('avgMeetingDuration');
        
        if (totalMeetings) {
            totalMeetings.textContent = meetings.length;
        }
        
        if (avgDuration && meetings.length > 0) {
            const totalDuration = meetings.reduce((sum, meeting) => sum + (meeting.duration || 0), 0);
            const averageDuration = totalDuration / meetings.length;
            avgDuration.textContent = this.formatDuration(averageDuration);
        }
    }
    
    showMeetingDetails(meetingId) {
        // Find the meeting data
        const meeting = this.meetings.find(m => m.meeting_id === meetingId);
        if (!meeting) return;
        
        const detailsContainer = document.getElementById('meetingDetails');
        if (!detailsContainer) return;
        
        const duration = this.formatDuration(meeting.duration);
        const timestamp = new Date(meeting.timestamp * 1000).toLocaleString();
        
        // Safely parse JSON data with error handling
        let outcomeContent = 'No outcomes recorded';
        let transcriptContent = 'No transcript available';
        
        try {
            if (meeting.outcome && meeting.outcome !== '') {
                const parsedOutcome = JSON.parse(meeting.outcome);
                outcomeContent = typeof parsedOutcome === 'object' ? 
                    JSON.stringify(parsedOutcome, null, 2) : parsedOutcome;
            }
        } catch (e) {
            outcomeContent = meeting.outcome || 'No outcomes recorded';
        }
        
        try {
            if (meeting.transcript && meeting.transcript !== '') {
                const parsedTranscript = JSON.parse(meeting.transcript);
                transcriptContent = typeof parsedTranscript === 'object' ? 
                    JSON.stringify(parsedTranscript, null, 2) : parsedTranscript;
            }
        } catch (e) {
            transcriptContent = meeting.transcript || 'No transcript available';
        }
        
        detailsContainer.innerHTML = `
            <h3>${meeting.topic}</h3>
            <div class="meeting-detail-info">
                <p><strong>Date:</strong> ${timestamp}</p>
                <p><strong>Duration:</strong> ${duration}</p>
                <p><strong>Participants:</strong> ${meeting.participants}</p>
            </div>
            <div class="meeting-outcome-details">
                <h4>Outcomes</h4>
                <div class="outcome-content">
                    ${outcomeContent}
                </div>
            </div>
            <div class="meeting-transcript">
                <h4>Discussion</h4>
                <div class="transcript-content">
                    ${transcriptContent}
                </div>
            </div>
        `;
    }
    
    formatDuration(seconds) {
        if (!seconds) return '0m';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds}s`;
    }
    
    addMeetingEntry(data) {
        // Add to meetings list
        this.loadMeetings();
        
        // Add to activity log
        this.addActivityLogEntry({
            type: 'meeting',
            author: 'System',
            message: `Meeting: ${data.topic} (${data.participants} participants)`,
            timestamp: data.timestamp
        });
        
        // Add to chat log
        this.addChatLogEntry({
            type: 'communication',
            author: 'Meeting',
            message: `Meeting started: ${data.topic} with ${data.participants} participants`,
            timestamp: data.timestamp
        });
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
            running: 'Research Active',
            error: 'Connection Error',
            reconnecting: 'Reconnecting...',
            failed: 'Connection Failed'
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
    console.log('DOM loaded, initializing AI Research Lab App...');
    
    try {
        window.aiResearchLab = new AIResearchLabApp();
        console.log('AI Research Lab App initialized successfully');
    } catch (error) {
        console.error('Error initializing AI Research Lab App:', error);
        
        // Show error notification
        const container = document.getElementById('notificationsContainer');
        if (container) {
            const notification = document.createElement('div');
            notification.className = 'notification error';
            notification.innerHTML = `
                <div class="notification-header">
                    <span class="notification-title">Initialization Error</span>
                    <button class="notification-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="notification-message">Failed to initialize the application: ${error.message}</div>
            `;
            container.appendChild(notification);
            setTimeout(() => notification.classList.add('show'), 100);
        }
    }
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