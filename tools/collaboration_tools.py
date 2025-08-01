"""
Collaboration Tools

Tools for team communication, task coordination, and collaborative research.
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class TeamCommunication(BaseTool):
    """
    Tool for facilitating communication between research team members and agents.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="team_communication",
            name="Team Communication Tool",
            description="Facilitate structured communication and knowledge sharing among research team members",
            capabilities=[
                "message_routing",
                "expertise_matching",
                "knowledge_sharing",
                "meeting_coordination",
                "collaborative_editing"
            ],
            requirements={
                "min_memory": 25
            }
        )
        self.message_history = {}
        self.active_discussions = {}
        self.expertise_directory = {}
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute team communication tasks."""
        task_type = task.get('type', 'send_message')
        
        if task_type == 'send_message':
            return self._send_message(task, context)
        elif task_type == 'create_discussion':
            return self._create_discussion(task, context)
        elif task_type == 'find_expert':
            return self._find_expert(task, context)
        elif task_type == 'schedule_meeting':
            return self._schedule_meeting(task, context)
        elif task_type == 'share_knowledge':
            return self._share_knowledge(task, context)
        else:
            return {'error': f'Unknown communication task: {task_type}'}
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle communication tasks."""
        comm_keywords = [
            'communicate', 'message', 'discuss', 'meeting', 'collaborate',
            'share', 'coordinate', 'expert', 'team', 'conversation'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in comm_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _send_message(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to team members or specific agents."""
        sender_id = context.get('agent_id', 'unknown')
        recipients = task.get('recipients', [])
        message = task.get('message', '')
        priority = task.get('priority', 'normal')
        message_type = task.get('message_type', 'general')
        
        if not message:
            return {'error': 'No message content provided'}
        
        # Create message object
        message_id = f"msg_{int(time.time())}_{sender_id}"
        message_obj = {
            'id': message_id,
            'sender': sender_id,
            'recipients': recipients,
            'content': message,
            'priority': priority,
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'sent',
            'responses': []
        }
        
        # Store message
        if sender_id not in self.message_history:
            self.message_history[sender_id] = []
        self.message_history[sender_id].append(message_obj)
        
        # Route message based on type
        routing_results = self._route_message(message_obj, context)
        
        return {
            'success': True,
            'message_id': message_id,
            'routing_results': routing_results,
            'delivery_status': 'delivered',
            'estimated_response_time': self._estimate_response_time(recipients, priority)
        }
    
    def _route_message(self, message: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to appropriate recipients based on content and urgency."""
        routing_results = {
            'primary_recipients': [],
            'cc_recipients': [],
            'routing_logic': []
        }
        
        content = message['content'].lower()
        message_type = message['type']
        priority = message['priority']
        
        # Determine routing based on message content
        if message_type == 'question':
            # Route to experts who can answer
            experts = self._find_relevant_experts(content)
            routing_results['primary_recipients'].extend(experts[:3])  # Top 3 experts
            routing_results['routing_logic'].append(f"Routed to {len(experts)} subject matter experts")
        
        elif message_type == 'update':
            # Route to stakeholders and team leads
            stakeholders = message.get('recipients', [])
            routing_results['primary_recipients'].extend(stakeholders)
            routing_results['routing_logic'].append("Routed to specified stakeholders")
        
        elif message_type == 'urgent':
            # Route to all available team members
            all_agents = context.get('available_agents', [])
            routing_results['primary_recipients'].extend(all_agents)
            routing_results['routing_logic'].append("Urgent message - routed to all available agents")
        
        else:
            # Default routing to specified recipients
            recipients = message.get('recipients', [])
            routing_results['primary_recipients'].extend(recipients)
            routing_results['routing_logic'].append("Routed to specified recipients")
        
        return routing_results
    
    def _find_relevant_experts(self, content: str) -> List[str]:
        """Find experts relevant to the message content."""
        experts = []
        
        # Keywords to expert mapping
        expert_keywords = {
            'statistics': ['data_scientist', 'statistical_expert'],
            'experiment': ['experimental_expert', 'research_methodology_expert'],
            'literature': ['literature_expert', 'systematic_review_expert'],
            'analysis': ['data_analyst', 'pattern_detection_expert'],
            'visualization': ['visualization_expert', 'data_scientist'],
            'methodology': ['methodology_expert', 'research_design_expert']
        }
        
        # Find matching experts
        for keyword, expert_types in expert_keywords.items():
            if keyword in content:
                experts.extend(expert_types)
        
        # Remove duplicates and return
        return list(set(experts))
    
    def _estimate_response_time(self, recipients: List[str], priority: str) -> str:
        """Estimate response time based on recipients and priority."""
        base_time = {
            'urgent': '15 minutes',
            'high': '1 hour', 
            'normal': '4 hours',
            'low': '24 hours'
        }
        
        # Adjust based on number of recipients
        time_estimate = base_time.get(priority, '4 hours')
        
        if len(recipients) > 5:
            time_estimate += ' (may be longer due to multiple recipients)'
        
        return time_estimate
    
    def _create_discussion(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured discussion thread on a research topic."""
        creator_id = context.get('agent_id', 'unknown')
        topic = task.get('topic', '')
        participants = task.get('participants', [])
        discussion_type = task.get('discussion_type', 'general')
        agenda = task.get('agenda', [])
        
        if not topic:
            return {'error': 'Discussion topic is required'}
        
        discussion_id = f"disc_{int(time.time())}_{creator_id}"
        
        discussion = {
            'id': discussion_id,
            'topic': topic,
            'creator': creator_id,
            'participants': participants,
            'type': discussion_type,
            'agenda': agenda,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'messages': [],
            'decisions': [],
            'action_items': []
        }
        
        self.active_discussions[discussion_id] = discussion
        
        # Send invitations to participants
        invitation_results = self._send_discussion_invitations(discussion, context)
        
        return {
            'success': True,
            'discussion_id': discussion_id,
            'discussion_url': f"/discussions/{discussion_id}",
            'invitation_results': invitation_results,
            'suggested_participants': self._suggest_additional_participants(topic)
        }
    
    def _send_discussion_invitations(self, discussion: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send invitations to discussion participants."""
        participants = discussion['participants']
        topic = discussion['topic']
        
        invitation_message = f"You're invited to participate in a discussion on '{topic}'. Please join to share your expertise and insights."
        
        invitation_task = {
            'type': 'send_message',
            'recipients': participants,
            'message': invitation_message,
            'message_type': 'invitation',
            'priority': 'normal'
        }
        
        return self._send_message(invitation_task, context)
    
    def _suggest_additional_participants(self, topic: str) -> List[Dict[str, str]]:
        """Suggest additional participants who might contribute to the discussion."""
        suggestions = []
        
        topic_lower = topic.lower()
        
        # Domain-specific suggestions
        if any(word in topic_lower for word in ['statistics', 'analysis', 'data']):
            suggestions.append({
                'agent_type': 'Statistical Expert',
                'reason': 'Can provide statistical analysis expertise'
            })
        
        if any(word in topic_lower for word in ['experiment', 'design', 'methodology']):
            suggestions.append({
                'agent_type': 'Experimental Design Expert',
                'reason': 'Can advise on experimental methodology'
            })
        
        if any(word in topic_lower for word in ['literature', 'review', 'survey']):
            suggestions.append({
                'agent_type': 'Literature Expert',
                'reason': 'Can provide comprehensive literature perspective'
            })
        
        if any(word in topic_lower for word in ['ethics', 'compliance', 'regulation']):
            suggestions.append({
                'agent_type': 'Ethics Advisor',
                'reason': 'Can ensure ethical compliance and considerations'
            })
        
        # Default suggestions
        if not suggestions:
            suggestions.extend([
                {
                    'agent_type': 'Critical Analysis Expert',
                    'reason': 'Can provide objective evaluation and critique'
                },
                {
                    'agent_type': 'Research Methodology Expert',
                    'reason': 'Can advise on research approach and methods'
                }
            ])
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _find_expert(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Find experts with specific expertise or knowledge."""
        expertise_needed = task.get('expertise', '')
        domain = task.get('domain', '')
        urgency = task.get('urgency', 'normal')
        
        if not expertise_needed:
            return {'error': 'Expertise requirement not specified'}
        
        # Search for matching experts
        matching_experts = self._search_experts(expertise_needed, domain)
        
        # Rank experts by relevance and availability
        ranked_experts = self._rank_experts(matching_experts, expertise_needed, urgency)
        
        return {
            'success': True,
            'expertise_needed': expertise_needed,
            'matching_experts': ranked_experts,
            'recommendations': self._generate_expert_recommendations(ranked_experts),
            'alternative_resources': self._suggest_alternative_resources(expertise_needed)
        }
    
    def _search_experts(self, expertise: str, domain: str = '') -> List[Dict[str, Any]]:
        """Search for experts based on expertise and domain."""
        experts = []
        
        # Mock expert database (in real implementation, would query actual expert directory)
        mock_experts = [
            {
                'id': 'statistical_expert_1',
                'name': 'Statistical Analysis Expert',
                'expertise': ['statistics', 'data analysis', 'hypothesis testing'],
                'domains': ['biomedical', 'social sciences', 'engineering'],
                'availability': 'high',
                'response_time': '2 hours',
                'success_rate': 0.95
            },
            {
                'id': 'experimental_expert_1',
                'name': 'Experimental Design Expert',
                'expertise': ['experimental design', 'methodology', 'controls'],
                'domains': ['clinical research', 'psychology', 'biology'],
                'availability': 'medium',
                'response_time': '4 hours',
                'success_rate': 0.92
            },
            {
                'id': 'literature_expert_1',
                'name': 'Literature Review Expert',
                'expertise': ['literature review', 'systematic review', 'meta-analysis'],
                'domains': ['medical research', 'evidence synthesis'],
                'availability': 'high',
                'response_time': '1 hour',
                'success_rate': 0.98
            },
            {
                'id': 'visualization_expert_1',
                'name': 'Data Visualization Expert',
                'expertise': ['data visualization', 'plotting', 'infographics'],
                'domains': ['data science', 'research communication'],
                'availability': 'medium',
                'response_time': '3 hours',
                'success_rate': 0.90
            }
        ]
        
        # Filter experts based on expertise and domain
        expertise_lower = expertise.lower()
        domain_lower = domain.lower()
        
        for expert in mock_experts:
            # Check expertise match
            expertise_match = any(exp in expertise_lower for exp in expert['expertise'])
            
            # Check domain match (if specified)
            domain_match = True
            if domain:
                domain_match = any(dom in domain_lower for dom in expert['domains'])
            
            if expertise_match and domain_match:
                experts.append(expert)
        
        return experts
    
    def _rank_experts(self, experts: List[Dict[str, Any]], expertise: str, urgency: str) -> List[Dict[str, Any]]:
        """Rank experts by relevance, availability, and track record."""
        for expert in experts:
            score = 0
            
            # Expertise relevance score
            expertise_matches = sum(1 for exp in expert['expertise'] if exp in expertise.lower())
            score += expertise_matches * 0.4
            
            # Availability score
            availability_scores = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
            score += availability_scores.get(expert['availability'], 0.1)
            
            # Success rate score
            score += expert['success_rate'] * 0.2
            
            # Urgency adjustment
            if urgency == 'urgent':
                response_time_hours = int(expert['response_time'].split()[0])
                if response_time_hours <= 2:
                    score += 0.1
            
            expert['relevance_score'] = score
        
        # Sort by relevance score
        return sorted(experts, key=lambda x: x['relevance_score'], reverse=True)
    
    def _generate_expert_recommendations(self, experts: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for expert selection."""
        recommendations = []
        
        if not experts:
            recommendations.append("No matching experts found - consider broadening search criteria")
            return recommendations
        
        top_expert = experts[0]
        recommendations.append(f"Recommended: {top_expert['name']} (relevance score: {top_expert['relevance_score']:.2f})")
        
        if len(experts) > 1:
            recommendations.append(f"Alternative: {experts[1]['name']} as backup option")
        
        # Availability recommendations
        urgent_available = [e for e in experts if e['availability'] == 'high']
        if urgent_available:
            recommendations.append(f"{len(urgent_available)} experts available for immediate consultation")
        
        return recommendations
    
    def _suggest_alternative_resources(self, expertise: str) -> List[Dict[str, str]]:
        """Suggest alternative resources if no experts are available."""
        alternatives = []
        
        expertise_lower = expertise.lower()
        
        if 'statistics' in expertise_lower:
            alternatives.append({
                'type': 'Tool',
                'name': 'Statistical Analyzer Tool',
                'description': 'Automated statistical analysis and hypothesis testing'
            })
        
        if 'literature' in expertise_lower:
            alternatives.append({
                'type': 'Tool',
                'name': 'Literature Search Tool',
                'description': 'Comprehensive literature search and analysis'
            })
        
        if 'visualization' in expertise_lower:
            alternatives.append({
                'type': 'Tool',
                'name': 'Data Visualizer Tool',
                'description': 'Automated data visualization and plotting'
            })
        
        # General alternatives
        alternatives.extend([
            {
                'type': 'Resource',
                'name': 'External Consultation',
                'description': 'Consider consulting external domain experts'
            },
            {
                'type': 'Resource',
                'name': 'Literature Review',
                'description': 'Conduct comprehensive literature review on the topic'
            }
        ])
        
        return alternatives[:3]
    
    def _schedule_meeting(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a meeting or research session."""
        organizer_id = context.get('agent_id', 'unknown')
        participants = task.get('participants', [])
        topic = task.get('topic', '')
        duration = task.get('duration', 60)  # minutes
        preferred_time = task.get('preferred_time', '')
        meeting_type = task.get('meeting_type', 'discussion')
        
        meeting_id = f"meeting_{int(time.time())}_{organizer_id}"
        
        # Find optimal meeting time
        optimal_time = self._find_optimal_meeting_time(participants, preferred_time, duration)
        
        meeting = {
            'id': meeting_id,
            'organizer': organizer_id,
            'participants': participants,
            'topic': topic,
            'type': meeting_type,
            'duration_minutes': duration,
            'scheduled_time': optimal_time,
            'status': 'scheduled',
            'agenda': self._generate_meeting_agenda(topic, meeting_type),
            'preparation_materials': self._suggest_preparation_materials(topic)
        }
        
        # Send meeting invitations
        invitation_results = self._send_meeting_invitations(meeting, context)
        
        return {
            'success': True,
            'meeting_id': meeting_id,
            'scheduled_time': optimal_time,
            'meeting_details': meeting,
            'invitation_results': invitation_results,
            'preparation_checklist': self._create_preparation_checklist(meeting)
        }
    
    def _find_optimal_meeting_time(self, participants: List[str], preferred_time: str, duration: int) -> str:
        """Find optimal meeting time based on participant availability."""
        # In real implementation, would check actual availability calendars
        # For now, generate a reasonable meeting time
        
        if preferred_time:
            return preferred_time
        
        # Default to next business day at 2 PM
        next_day = datetime.now() + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        
        meeting_time = next_day.replace(hour=14, minute=0, second=0, microsecond=0)
        return meeting_time.isoformat()
    
    def _generate_meeting_agenda(self, topic: str, meeting_type: str) -> List[str]:
        """Generate a meeting agenda based on topic and type."""
        agenda = []
        
        if meeting_type == 'research_planning':
            agenda = [
                "1. Review research objectives and scope",
                "2. Discuss methodology and approach",
                "3. Resource and timeline planning",
                "4. Risk assessment and mitigation",
                "5. Next steps and action items"
            ]
        elif meeting_type == 'progress_review':
            agenda = [
                "1. Progress updates from team members",
                "2. Review of completed milestones",
                "3. Discussion of challenges and blockers",
                "4. Adjustment of timeline and priorities",
                "5. Planning for next phase"
            ]
        elif meeting_type == 'results_discussion':
            agenda = [
                "1. Presentation of research findings",
                "2. Statistical analysis review",
                "3. Discussion of implications",
                "4. Peer review and critique",
                "5. Publication and dissemination planning"
            ]
        else:
            # General discussion agenda
            agenda = [
                f"1. Introduction and context for {topic}",
                "2. Current state of knowledge",
                "3. Key questions and challenges",
                "4. Proposed solutions and approaches",
                "5. Action items and follow-up"
            ]
        
        return agenda
    
    def _suggest_preparation_materials(self, topic: str) -> List[Dict[str, str]]:
        """Suggest materials participants should review before the meeting."""
        materials = []
        
        topic_lower = topic.lower()
        
        # Topic-specific materials
        if 'literature' in topic_lower:
            materials.append({
                'type': 'Reading',
                'description': 'Recent literature reviews on the topic',
                'priority': 'high'
            })
        
        if 'methodology' in topic_lower:
            materials.append({
                'type': 'Document',
                'description': 'Research methodology guidelines and protocols',
                'priority': 'high'
            })
        
        if 'results' in topic_lower or 'analysis' in topic_lower:
            materials.append({
                'type': 'Data',
                'description': 'Current data analysis results and visualizations',
                'priority': 'high'
            })
        
        # General materials
        materials.extend([
            {
                'type': 'Background',
                'description': 'Previous meeting notes and action items',
                'priority': 'medium'
            },
            {
                'type': 'Reference',
                'description': 'Relevant project documentation',
                'priority': 'medium'
            }
        ])
        
        return materials[:4]  # Limit to top 4 materials
    
    def _send_meeting_invitations(self, meeting: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send meeting invitations to participants."""
        participants = meeting['participants']
        
        invitation_message = f"""
        Meeting Invitation: {meeting['topic']}
        
        Time: {meeting['scheduled_time']}
        Duration: {meeting['duration_minutes']} minutes
        Type: {meeting['type']}
        
        Agenda:
        {chr(10).join(meeting['agenda'])}
        
        Please confirm your attendance and review preparation materials.
        """
        
        invitation_task = {
            'type': 'send_message',
            'recipients': participants,
            'message': invitation_message,
            'message_type': 'meeting_invitation',
            'priority': 'normal'
        }
        
        return self._send_message(invitation_task, context)
    
    def _create_preparation_checklist(self, meeting: Dict[str, Any]) -> List[str]:
        """Create a preparation checklist for meeting participants."""
        checklist = [
            "Review meeting agenda and objectives",
            "Gather relevant data and analysis results",
            "Prepare progress updates or status reports",
            "Review previous meeting notes and action items",
            "Prepare questions and discussion points"
        ]
        
        # Add meeting-specific items
        if meeting['type'] == 'research_planning':
            checklist.extend([
                "Review research proposal and objectives",
                "Prepare resource requirement estimates",
                "Identify potential risks and challenges"
            ])
        elif meeting['type'] == 'results_discussion':
            checklist.extend([
                "Prepare presentation of key findings",
                "Review statistical significance and effect sizes",
                "Consider implications and limitations"
            ])
        
        return checklist
    
    def _share_knowledge(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Share knowledge or insights with the research team."""
        sharer_id = context.get('agent_id', 'unknown')
        knowledge_type = task.get('knowledge_type', 'insight')
        content = task.get('content', '')
        target_audience = task.get('target_audience', 'all')
        tags = task.get('tags', [])
        
        if not content:
            return {'error': 'No knowledge content provided'}
        
        knowledge_id = f"knowledge_{int(time.time())}_{sharer_id}"
        
        knowledge_entry = {
            'id': knowledge_id,
            'sharer': sharer_id,
            'type': knowledge_type,
            'content': content,
            'target_audience': target_audience,
            'tags': tags,
            'shared_at': datetime.now().isoformat(),
            'relevance_score': self._calculate_knowledge_relevance(content, context),
            'access_level': 'team',
            'usage_count': 0
        }
        
        # Store in knowledge base
        self._store_knowledge(knowledge_entry)
        
        # Notify relevant team members
        notification_results = self._notify_knowledge_recipients(knowledge_entry, context)
        
        return {
            'success': True,
            'knowledge_id': knowledge_id,
            'shared_with': target_audience,
            'relevance_score': knowledge_entry['relevance_score'],
            'notification_results': notification_results,
            'suggested_applications': self._suggest_knowledge_applications(content, knowledge_type)
        }
    
    def _calculate_knowledge_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate relevance score for shared knowledge."""
        # Simple relevance calculation based on keywords and context
        current_research = context.get('current_research_topics', [])
        
        relevance = 0.5  # Base relevance
        
        # Check for matches with current research topics
        content_lower = content.lower()
        for topic in current_research:
            if topic.lower() in content_lower:
                relevance += 0.1
        
        # Check for high-value content indicators
        value_indicators = [
            'significant', 'important', 'breakthrough', 'novel',
            'finding', 'result', 'conclusion', 'implication'
        ]
        
        for indicator in value_indicators:
            if indicator in content_lower:
                relevance += 0.05
        
        return min(1.0, relevance)
    
    def _store_knowledge(self, knowledge_entry: Dict[str, Any]) -> None:
        """Store knowledge entry in the knowledge base."""
        # In real implementation, would store in database
        # For now, store in memory
        if not hasattr(self, 'knowledge_base'):
            self.knowledge_base = {}
        
        self.knowledge_base[knowledge_entry['id']] = knowledge_entry
    
    def _notify_knowledge_recipients(self, knowledge_entry: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Notify relevant team members about new knowledge."""
        target_audience = knowledge_entry['target_audience']
        
        # Determine recipients based on target audience
        if target_audience == 'all':
            recipients = context.get('all_team_members', [])
        elif target_audience == 'experts':
            recipients = self._find_relevant_experts(knowledge_entry['content'])
        else:
            recipients = [target_audience] if isinstance(target_audience, str) else target_audience
        
        notification_message = f"""
        New knowledge shared: {knowledge_entry['type']}
        
        From: {knowledge_entry['sharer']}
        Tags: {', '.join(knowledge_entry['tags'])}
        Relevance: {knowledge_entry['relevance_score']:.2f}
        
        Content: {knowledge_entry['content'][:200]}...
        
        Access full content via knowledge ID: {knowledge_entry['id']}
        """
        
        notification_task = {
            'type': 'send_message',
            'recipients': recipients,
            'message': notification_message,
            'message_type': 'knowledge_notification',
            'priority': 'low'
        }
        
        return self._send_message(notification_task, context)
    
    def _suggest_knowledge_applications(self, content: str, knowledge_type: str) -> List[str]:
        """Suggest applications for the shared knowledge."""
        applications = []
        
        content_lower = content.lower()
        
        # Type-specific applications
        if knowledge_type == 'methodology':
            applications.append("Apply to current experimental design")
            applications.append("Include in research protocol documentation")
        
        elif knowledge_type == 'finding':
            applications.append("Consider for manuscript discussion section")
            applications.append("Use for hypothesis generation in future studies")
        
        elif knowledge_type == 'tool':
            applications.append("Evaluate for current data analysis needs")
            applications.append("Add to recommended tools list")
        
        # Content-specific applications
        if any(word in content_lower for word in ['statistical', 'analysis', 'test']):
            applications.append("Apply to current statistical analysis")
        
        if any(word in content_lower for word in ['visualization', 'plot', 'graph']):
            applications.append("Use for data visualization improvements")
        
        if any(word in content_lower for word in ['literature', 'review', 'citation']):
            applications.append("Incorporate into literature review")
        
        # Default applications
        if not applications:
            applications = [
                "Document in project knowledge base",
                "Share with relevant external collaborators",
                "Consider for training and education materials"
            ]
        
        return applications[:3]


class TaskCoordinator(BaseTool):
    """
    Tool for coordinating tasks and workflows among research team members.
    """
    
    def __init__(self):
        super().__init__(
            tool_id="task_coordinator",
            name="Task Coordinator",
            description="Coordinate tasks, manage workflows, and track progress across research team",
            capabilities=[
                "task_assignment",
                "workflow_management",
                "progress_tracking",
                "dependency_management",
                "resource_allocation"
            ],
            requirements={
                "min_memory": 50
            }
        )
        self.active_tasks = {}
        self.workflows = {}
        self.dependencies = {}
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task coordination functions."""
        task_type = task.get('type', 'assign_task')
        
        if task_type == 'assign_task':
            return self._assign_task(task, context)
        elif task_type == 'create_workflow':
            return self._create_workflow(task, context)
        elif task_type == 'track_progress':
            return self._track_progress(task, context)
        elif task_type == 'manage_dependencies':
            return self._manage_dependencies(task, context)
        elif task_type == 'allocate_resources':
            return self._allocate_resources(task, context)
        else:
            return {'error': f'Unknown coordination task: {task_type}'}
    
    def can_handle(self, task_type: str, requirements: Dict[str, Any]) -> float:
        """Assess capability to handle coordination tasks."""
        coord_keywords = [
            'coordinate', 'assign', 'manage', 'workflow', 'progress',
            'dependency', 'resource', 'schedule', 'track', 'organize'
        ]
        
        confidence = 0.0
        task_lower = task_type.lower()
        
        for keyword in coord_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _assign_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a task to appropriate team member(s)."""
        assigner_id = context.get('agent_id', 'unknown')
        task_description = task.get('description', '')
        required_skills = task.get('required_skills', [])
        priority = task.get('priority', 'normal')
        deadline = task.get('deadline', '')
        estimated_effort = task.get('estimated_effort', 'medium')
        
        if not task_description:
            return {'error': 'Task description is required'}
        
        task_id = f"task_{int(time.time())}_{assigner_id}"
        
        # Find best assignee
        assignee_analysis = self._find_best_assignee(required_skills, context)
        
        # Create task object
        task_obj = {
            'id': task_id,
            'description': task_description,
            'assigner': assigner_id,
            'assignee': assignee_analysis['recommended_assignee'],
            'required_skills': required_skills,
            'priority': priority,
            'deadline': deadline,
            'estimated_effort': estimated_effort,
            'status': 'assigned',
            'created_at': datetime.now().isoformat(),
            'progress': 0,
            'dependencies': [],
            'resources_needed': task.get('resources_needed', [])
        }
        
        # Store task
        self.active_tasks[task_id] = task_obj
        
        # Send task assignment notification
        notification_results = self._send_task_assignment(task_obj, context)
        
        return {
            'success': True,
            'task_id': task_id,
            'assigned_to': assignee_analysis['recommended_assignee'],
            'assignment_rationale': assignee_analysis['rationale'],
            'alternative_assignees': assignee_analysis['alternatives'],
            'notification_results': notification_results,
            'task_breakdown': self._suggest_task_breakdown(task_description)
        }
    
    def _find_best_assignee(self, required_skills: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best assignee for a task based on skills and availability."""
        available_agents = context.get('available_agents', [])
        agent_skills = context.get('agent_skills', {})
        agent_workload = context.get('agent_workload', {})
        
        if not available_agents:
            return {
                'recommended_assignee': 'general_research_agent',
                'rationale': 'No specific agents available - using general agent',
                'alternatives': []
            }
        
        # Score agents based on skills and availability
        agent_scores = {}
        
        for agent in available_agents:
            score = 0
            
            # Skill match score
            agent_skill_list = agent_skills.get(agent, [])
            skill_matches = sum(1 for skill in required_skills if skill in agent_skill_list)
            score += skill_matches * 0.4
            
            # Availability score (inverse of workload)
            workload = agent_workload.get(agent, 5)  # Default medium workload
            availability_score = max(0, (10 - workload) / 10)
            score += availability_score * 0.3
            
            # General capability score
            score += 0.3  # Base capability score
            
            agent_scores[agent] = score
        
        # Sort by score
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_agents:
            best_agent = sorted_agents[0][0]
            alternatives = [agent for agent, _ in sorted_agents[1:3]]
            
            return {
                'recommended_assignee': best_agent,
                'rationale': f'Best skill match and availability (score: {agent_scores[best_agent]:.2f})',
                'alternatives': alternatives
            }
        else:
            return {
                'recommended_assignee': 'general_research_agent',
                'rationale': 'No agents available - using default',
                'alternatives': []
            }
    
    def _send_task_assignment(self, task_obj: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send task assignment notification to assignee."""
        assignment_message = f"""
        New Task Assignment: {task_obj['description']}
        
        Priority: {task_obj['priority']}
        Estimated Effort: {task_obj['estimated_effort']}
        Deadline: {task_obj.get('deadline', 'Not specified')}
        Required Skills: {', '.join(task_obj['required_skills'])}
        
        Task ID: {task_obj['id']}
        
        Please acknowledge receipt and provide estimated completion time.
        """
        
        comm_tool = TeamCommunication()
        notification_task = {
            'type': 'send_message',
            'recipients': [task_obj['assignee']],
            'message': assignment_message,
            'message_type': 'task_assignment',
            'priority': task_obj['priority']
        }
        
        return comm_tool._send_message(notification_task, context)
    
    def _suggest_task_breakdown(self, task_description: str) -> List[Dict[str, str]]:
        """Suggest how to break down a complex task."""
        breakdown = []
        
        desc_lower = task_description.lower()
        
        # Common task breakdown patterns
        if 'analysis' in desc_lower:
            breakdown.extend([
                {'step': 'Data collection and preparation', 'estimated_time': '2-4 hours'},
                {'step': 'Exploratory data analysis', 'estimated_time': '1-2 hours'},
                {'step': 'Statistical analysis', 'estimated_time': '2-3 hours'},
                {'step': 'Results interpretation', 'estimated_time': '1-2 hours'}
            ])
        
        elif 'experiment' in desc_lower:
            breakdown.extend([
                {'step': 'Experimental design and protocol', 'estimated_time': '3-5 hours'},
                {'step': 'Resource and material preparation', 'estimated_time': '1-2 hours'},
                {'step': 'Experiment execution', 'estimated_time': '4-8 hours'},
                {'step': 'Data collection and recording', 'estimated_time': '2-3 hours'}
            ])
        
        elif 'literature' in desc_lower or 'review' in desc_lower:
            breakdown.extend([
                {'step': 'Search strategy development', 'estimated_time': '1 hour'},
                {'step': 'Database searches', 'estimated_time': '2-3 hours'},
                {'step': 'Article screening and selection', 'estimated_time': '3-5 hours'},
                {'step': 'Synthesis and writing', 'estimated_time': '4-6 hours'}
            ])
        
        else:
            # Generic breakdown
            breakdown.extend([
                {'step': 'Task planning and resource identification', 'estimated_time': '1 hour'},
                {'step': 'Core task execution', 'estimated_time': '3-5 hours'},
                {'step': 'Quality review and validation', 'estimated_time': '1 hour'},
                {'step': 'Documentation and reporting', 'estimated_time': '1-2 hours'}
            ])
        
        return breakdown[:4]  # Limit to 4 steps
    
    def _create_workflow(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured workflow for a research process."""
        creator_id = context.get('agent_id', 'unknown')
        workflow_name = task.get('name', '')
        workflow_type = task.get('workflow_type', 'research')
        steps = task.get('steps', [])
        
        if not workflow_name:
            return {'error': 'Workflow name is required'}
        
        workflow_id = f"workflow_{int(time.time())}_{creator_id}"
        
        # Generate workflow steps if not provided
        if not steps:
            steps = self._generate_workflow_steps(workflow_type)
        
        workflow = {
            'id': workflow_id,
            'name': workflow_name,
            'type': workflow_type,
            'creator': creator_id,
            'steps': steps,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'current_step': 0,
            'participants': [],
            'estimated_duration': self._calculate_workflow_duration(steps)
        }
        
        # Store workflow
        self.workflows[workflow_id] = workflow
        
        return {
            'success': True,
            'workflow_id': workflow_id,
            'workflow': workflow,
            'next_actions': self._get_workflow_next_actions(workflow),
            'resource_requirements': self._calculate_workflow_resources(steps)
        }
    
    def _generate_workflow_steps(self, workflow_type: str) -> List[Dict[str, Any]]:
        """Generate standard workflow steps based on type."""
        if workflow_type == 'experimental_study':
            return [
                {
                    'name': 'Literature Review',
                    'description': 'Comprehensive review of existing literature',
                    'estimated_duration': '5-7 days',
                    'required_skills': ['literature_search', 'critical_analysis'],
                    'deliverables': ['literature_review_report']
                },
                {
                    'name': 'Experimental Design',
                    'description': 'Design experimental protocol and methodology',
                    'estimated_duration': '3-5 days',
                    'required_skills': ['experimental_design', 'statistics'],
                    'deliverables': ['experimental_protocol', 'power_analysis']
                },
                {
                    'name': 'Ethics Approval',
                    'description': 'Obtain necessary ethics approvals',
                    'estimated_duration': '2-4 weeks',
                    'required_skills': ['ethics_compliance'],
                    'deliverables': ['ethics_approval_letter']
                },
                {
                    'name': 'Data Collection',
                    'description': 'Execute experiment and collect data',
                    'estimated_duration': '2-8 weeks',
                    'required_skills': ['experimental_execution', 'data_collection'],
                    'deliverables': ['raw_data', 'collection_log']
                },
                {
                    'name': 'Data Analysis',
                    'description': 'Analyze collected data',
                    'estimated_duration': '1-2 weeks',
                    'required_skills': ['statistical_analysis', 'data_visualization'],
                    'deliverables': ['analysis_results', 'visualizations']
                },
                {
                    'name': 'Manuscript Preparation',
                    'description': 'Prepare research manuscript',
                    'estimated_duration': '2-3 weeks',
                    'required_skills': ['scientific_writing', 'data_interpretation'],
                    'deliverables': ['manuscript_draft']
                }
            ]
        
        elif workflow_type == 'systematic_review':
            return [
                {
                    'name': 'Protocol Development',
                    'description': 'Develop systematic review protocol',
                    'estimated_duration': '1-2 weeks',
                    'required_skills': ['systematic_review', 'protocol_writing'],
                    'deliverables': ['review_protocol']
                },
                {
                    'name': 'Search Strategy',
                    'description': 'Develop and execute search strategy',
                    'estimated_duration': '1 week',
                    'required_skills': ['literature_search', 'database_searching'],
                    'deliverables': ['search_strategy', 'search_results']
                },
                {
                    'name': 'Study Selection',
                    'description': 'Screen and select relevant studies',
                    'estimated_duration': '2-3 weeks',
                    'required_skills': ['study_screening', 'inclusion_criteria'],
                    'deliverables': ['included_studies_list', 'prisma_diagram']
                },
                {
                    'name': 'Data Extraction',
                    'description': 'Extract data from included studies',
                    'estimated_duration': '1-2 weeks',
                    'required_skills': ['data_extraction', 'quality_assessment'],
                    'deliverables': ['extracted_data', 'quality_scores']
                },
                {
                    'name': 'Synthesis',
                    'description': 'Synthesize findings and prepare review',
                    'estimated_duration': '2-3 weeks',
                    'required_skills': ['meta_analysis', 'evidence_synthesis'],
                    'deliverables': ['systematic_review_manuscript']
                }
            ]
        
        else:
            # Generic research workflow
            return [
                {
                    'name': 'Planning',
                    'description': 'Research planning and preparation',
                    'estimated_duration': '1-2 days',
                    'required_skills': ['research_planning'],
                    'deliverables': ['research_plan']
                },
                {
                    'name': 'Execution',
                    'description': 'Execute research activities',
                    'estimated_duration': '1-2 weeks',
                    'required_skills': ['research_execution'],
                    'deliverables': ['research_outputs']
                },
                {
                    'name': 'Analysis',
                    'description': 'Analyze results and findings',
                    'estimated_duration': '3-5 days',
                    'required_skills': ['data_analysis'],
                    'deliverables': ['analysis_report']
                },
                {
                    'name': 'Documentation',
                    'description': 'Document and report findings',
                    'estimated_duration': '2-3 days',
                    'required_skills': ['scientific_writing'],
                    'deliverables': ['final_report']
                }
            ]
    
    def _calculate_workflow_duration(self, steps: List[Dict[str, Any]]) -> str:
        """Calculate estimated total duration for workflow."""
        total_days = 0
        
        for step in steps:
            duration_str = step.get('estimated_duration', '1 day')
            
            # Parse duration (simplified)
            if 'day' in duration_str:
                days = int(duration_str.split('-')[0]) if '-' in duration_str else 1
                total_days += days
            elif 'week' in duration_str:
                weeks = int(duration_str.split('-')[0]) if '-' in duration_str else 1
                total_days += weeks * 5  # Assume 5 working days per week
        
        if total_days > 30:
            return f"{total_days // 30} months"
        elif total_days > 5:
            return f"{total_days // 5} weeks"
        else:
            return f"{total_days} days"
    
    def _get_workflow_next_actions(self, workflow: Dict[str, Any]) -> List[str]:
        """Get next actions for workflow execution."""
        current_step = workflow['current_step']
        steps = workflow['steps']
        
        if current_step >= len(steps):
            return ['Workflow completed - prepare final deliverables']
        
        next_step = steps[current_step]
        
        actions = [
            f"Begin step: {next_step['name']}",
            f"Assign responsible team member with skills: {', '.join(next_step['required_skills'])}",
            f"Prepare resources for: {next_step['description']}",
            f"Set milestone deadline: {next_step['estimated_duration']}"
        ]
        
        return actions
    
    def _calculate_workflow_resources(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for workflow."""
        all_skills = []
        all_deliverables = []
        
        for step in steps:
            all_skills.extend(step.get('required_skills', []))
            all_deliverables.extend(step.get('deliverables', []))
        
        unique_skills = list(set(all_skills))
        skill_counts = {skill: all_skills.count(skill) for skill in unique_skills}
        
        return {
            'required_skills': unique_skills,
            'skill_frequency': skill_counts,
            'total_deliverables': len(all_deliverables),
            'critical_skills': [skill for skill, count in skill_counts.items() if count > 1],
            'estimated_team_size': min(len(unique_skills), 6)  # Max 6 team members
        }