"""
Principal Investigator Agent - Coordinates research and manages team of expert agents.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PrincipalInvestigatorAgent(BaseAgent):
    """
    Principal Investigator (PI) Agent that coordinates research activities
    and dynamically hires domain expert agents based on research needs.
    """
    
    def __init__(self, agent_id: str = "PI", model_config: Optional[Dict[str, Any]] = None, cost_manager=None):
        super().__init__(
            agent_id=agent_id,
            role="Principal Investigator",
            expertise=["Research Coordination", "Task Decomposition", "Team Management"],
            model_config=model_config,
            cost_manager=cost_manager
        )
        self.active_research_sessions = {}
        self.hired_agents = {}
        self.research_history = []
        self.current_research_context = ""  # Track current research problem for dynamic agent creation
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'active_research_sessions': list(self.active_research_sessions.keys()),
            'hired_agents_count': len(self.hired_agents),
            'research_history_count': len(self.research_history),
            'current_research_context': self.current_research_context
        })
        return base_dict
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate PI response with focus on coordination and synthesis.
        """
        specialized_prompt = f"""
        You are a Principal Investigator (PI) leading a research team with expertise in coordinating interdisciplinary research projects.
        Analyze the following research request and provide a comprehensive coordination plan:
        
        {prompt}
        
        As the PI, focus on:
        - Breaking down the research problem into manageable components
        - Identifying what types of expertise are needed
        - Developing a research coordination strategy
        - Outlining task assignment and timeline
        - Considering methodological approaches and integration points
        - Identifying potential challenges and mitigation strategies
        
        Provide a clear, actionable research coordination plan.
        """
        
        return self.llm_client.generate_response(
            specialized_prompt,
            context,
            agent_role=self.role
        )
    
    def assess_task_relevance(self, task_description: str) -> float:
        """
        PI is relevant to all coordination and management tasks.
        """
        coordination_keywords = [
            'coordinate', 'manage', 'overview', 'synthesize', 
            'integrate', 'plan', 'strategy', 'workflow', 
        ]
        
        task_lower = task_description.lower()
        relevance = sum(1 for keyword in coordination_keywords if keyword in task_lower)
        return min(1.0, relevance * 0.2)  # Max relevance for coordination tasks
    
    
    def analyze_research_problem(self, problem_description: str, 
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a research problem and determine required expertise using AI.
        
        Args:
            problem_description: Description of the research problem
            constraints: Optional constraints (budget, time, etc.)
            
        Returns:
            Analysis including required expertise and task breakdown
        """
        logger.info(f"PI analyzing research problem: {problem_description[:100]}...")
        
        # Use LLM for sophisticated analysis
        analysis_prompt = f"""
        As a Principal Investigator, analyze this research problem and determine what types of expertise are needed:
        
        Research Problem: {problem_description}
        
        Please provide:
        1. List of required expertise domains (e.g., biology, chemistry, data_science, psychology, etc.)
        2. Complexity assessment (scale 1-10)
        3. Estimated number of expert agents needed (1-8)
        4. Priority ranking of expertise domains
        5. Key research questions to address
        
        Format your response as:
        DOMAINS: [domain1, domain2, domain3]
        COMPLEXITY: [number]
        AGENTS_NEEDED: [number]
        PRIORITIES: [domain1, domain2, domain3]
        QUESTIONS: [question1 | question2 | question3]
        """
        
        llm_response = self.llm_client.generate_response(
            analysis_prompt, 
            {'constraints': constraints or {}}, 
            agent_role=self.role,
        )
        
        # Parse LLM response
        required_expertise, complexity_score, estimated_agents, priority_domains, key_questions = self._parse_analysis_response(llm_response)
        
        # Fallback to keyword-based analysis if LLM parsing fails
        if not required_expertise:
            required_expertise, complexity_score, estimated_agents, priority_domains = self._fallback_keyword_analysis(problem_description)
        
        # Always include literature research for comprehensive analysis
        if 'literature' not in required_expertise and 'literature_research' not in required_expertise:
            required_expertise.append('literature_research')
        
        analysis = {
            'problem_id': f"research_{int(time.time())}",
            'description': problem_description,
            'required_expertise': required_expertise,
            'constraints': constraints or {},
            'complexity_score': complexity_score,
            'estimated_agents_needed': estimated_agents,
            'priority_domains': priority_domains,
            'key_questions': key_questions,
            'analysis_timestamp': time.time()
        }
        
        logger.info(f"Analysis complete. Required expertise: {required_expertise}")
        return analysis
    
    def _parse_analysis_response(self, response: str) -> tuple:
        """Parse the LLM analysis response."""
        import re
        
        try:
            # Extract domains
            domains_match = re.search(r'DOMAINS:\s*\[([^\]]+)\]', response)
            domains = []
            if domains_match:
                domains_str = domains_match.group(1)
                domains = [d.strip().strip('"\'') for d in domains_str.split(',')]
            
            # Extract complexity
            complexity_match = re.search(r'COMPLEXITY:\s*(\d+)', response)
            complexity = 5  # default
            if complexity_match:
                complexity = min(10, max(1, int(complexity_match.group(1))))
            
            # Extract agents needed
            agents_match = re.search(r'AGENTS_NEEDED:\s*(\d+)', response)
            agents_needed = min(8, max(1, complexity))  # default based on complexity
            if agents_match:
                agents_needed = min(8, max(1, int(agents_match.group(1))))
            
            # Extract priorities
            priorities_match = re.search(r'PRIORITIES:\s*\[([^\]]+)\]', response)
            priorities = domains[:3]  # default to first 3 domains
            if priorities_match:
                priorities_str = priorities_match.group(1)
                priorities = [p.strip().strip('"\'') for p in priorities_str.split(',')][:3]
            
            # Extract questions
            questions_match = re.search(r'QUESTIONS:\s*\[([^\]]+)\]', response)
            questions = []
            if questions_match:
                questions_str = questions_match.group(1)
                questions = [q.strip().strip('"\'') for q in questions_str.split('|')]
            
            return domains, complexity / 10.0, agents_needed, priorities, questions
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM analysis response: {e}")
            return [], 0.5, 3, [], []
    
    def _fallback_keyword_analysis(self, problem_description: str) -> tuple:
        """Fallback keyword-based analysis if LLM parsing fails."""
        # General domain mapping (not specific to ophthalmology)
        expertise_mapping = {
            'biology': ['biology', 'biological', 'organism', 'cellular', 'molecular', 'genetic'],
            'chemistry': ['chemistry', 'chemical', 'compound', 'reaction', 'molecular'],
            'physics': ['physics', 'physical', 'quantum', 'energy', 'force', 'mechanics'],
            'medicine': ['medicine', 'medical', 'clinical', 'patient', 'treatment', 'diagnosis', 'health'],
            'psychology': ['psychology', 'mental', 'behavior', 'cognitive', 'anxiety', 'depression', 'emotional'],
            'neuroscience': ['brain', 'neural', 'neuroscience', 'neurological', 'cortex', 'neuron'],
            'data_science': ['data', 'analysis', 'machine learning', 'statistics', 'model', 'algorithm', 'computational'],
            'engineering': ['engineering', 'design', 'technical', 'system', 'manufacturing'],
            'environmental': ['environment', 'environmental', 'climate', 'ecology', 'sustainability'],
            'social_science': ['social', 'society', 'community', 'cultural', 'anthropology', 'sociology'],
            'economics': ['economic', 'financial', 'market', 'business', 'economy']
        }
        
        problem_lower = problem_description.lower()
        required_expertise = []
        
        for domain, keywords in expertise_mapping.items():
            if any(keyword in problem_lower for keyword in keywords):
                required_expertise.append(domain)
        
        # If no domains found, add general research capability
        if not required_expertise:
            required_expertise = ['general_research', 'data_science']
        
        complexity_score = min(1.0, len(required_expertise) * 0.15 + 0.3)
        estimated_agents = min(6, max(2, len(required_expertise)))
        priority_domains = required_expertise[:3]
        
        return required_expertise, complexity_score, estimated_agents, priority_domains
    
    def hire_agents(self, marketplace, required_expertise: List[str], 
                   constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hire agents from marketplace or create new experts based on required expertise.
        
        Args:
            marketplace: AgentMarketplace instance
            required_expertise: List of required expertise domains
            constraints: Optional hiring constraints
            
        Returns:
            Dictionary mapping expertise to hired agent instances
        """
        logger.info(f"PI hiring agents for expertise: {required_expertise}")
        
        hired_agents = {}
        hiring_decisions = []
        
        for expertise in required_expertise:
            # Try to get available agents for this expertise
            available_agents = marketplace.get_agents_by_expertise(expertise)
            
            # If no specific agents available, try to find general agents that could adapt
            if not available_agents:
                available_agents = marketplace.get_agents_by_expertise('general_research')
                
            # If still no agents, create a new expert for this domain
            if not available_agents:
                logger.info(f"Creating new expert for domain: {expertise}")
                new_agent = marketplace.create_expert_for_domain(expertise, self.current_research_context)
                available_agents = [new_agent]
            
            # Select best agent based on performance and relevance
            best_agent = self._select_best_agent(available_agents, expertise, constraints)
            
            if best_agent:
                hired_agents[expertise] = best_agent
                hiring_decisions.append({
                    'expertise': expertise,
                    'agent_id': best_agent.agent_id,
                    'agent_role': best_agent.role,
                    'performance_score': best_agent.performance_metrics['average_quality_score'],
                    'relevance_score': best_agent.assess_task_relevance(f"Research in {expertise}"),
                    'hire_time': time.time(),
                    'created_new': best_agent.agent_id not in marketplace.available_agents
                })
                
                # Mark agent as hired in marketplace
                marketplace.hire_agent(best_agent.agent_id)
                
                logger.info(f"Hired {best_agent.agent_id} ({best_agent.role}) for {expertise}")
        
        # Store hiring decisions
        self.hired_agents.update(hired_agents)
        
        return {
            'hired_agents': hired_agents,
            'hiring_decisions': hiring_decisions,
            'total_hired': len(hired_agents),
            'new_agents_created': sum(1 for decision in hiring_decisions if decision.get('created_new', False))
        }
    
    def _select_best_agent(self, available_agents: List[BaseAgent], 
                          expertise_domain: str,
                          constraints: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        """
        Select the best agent from available options for a specific expertise domain.
        
        Args:
            available_agents: List of available agents
            expertise_domain: The specific expertise domain needed
            constraints: Optional selection constraints
            
        Returns:
            Best agent or None if no suitable agent found
        """
        if not available_agents:
            return None
        
        # Score agents based on performance metrics and relevance
        scored_agents = []
        for agent in available_agents:
            metrics = agent.performance_metrics
            
            # Assess relevance to the specific expertise domain
            relevance_score = agent.assess_task_relevance(f"Research tasks in {expertise_domain}")
            
            # Calculate composite score
            score = (
                metrics['average_quality_score'] * 0.3 +
                metrics['success_rate'] * 0.2 +
                relevance_score * 0.4 +  # Relevance is most important
                (1.0 if metrics['tasks_completed'] > 0 else 0.5) * 0.1
            )
            
            scored_agents.append((score, agent))
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        
        return scored_agents[0][1]
    
    def decompose_research_task(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose research problem into specific tasks using LLM analysis.
        
        Args:
            problem_analysis: Analysis from analyze_research_problem
            
        Returns:
            List of task dictionaries
        """
        problem_desc = problem_analysis['description']
        required_expertise = problem_analysis['required_expertise']
        
        # Use LLM to decompose tasks
        decomposition_prompt = f"""
        As a Principal Investigator, decompose this research problem into specific tasks for expert agents:
        
        Research Problem: {problem_desc}
        Required Expertise: {required_expertise}
        
        For each expertise domain, define:
        1. Specific research task description
        2. Expected outputs/deliverables
        3. Priority level (high/medium/low)
        
        Format each task as:
        DOMAIN: [expertise_domain]
        TASK: [detailed task description]
        OUTPUTS: [expected output 1 | expected output 2 | expected output 3]
        PRIORITY: [high/medium/low]
        ---
        """
        
        llm_response = self.llm_client.generate_response(
            decomposition_prompt, 
            {'problem_analysis': problem_analysis}, 
            agent_role=self.role
        )
        
        # Parse LLM response into tasks
        tasks = self._parse_task_decomposition(llm_response, problem_analysis)
        
        # If LLM parsing fails, use fallback method
        if not tasks:
            tasks = self._fallback_task_decomposition(problem_analysis)
        
        logger.info(f"Decomposed research into {len(tasks)} tasks")
        return tasks
    
    def _parse_task_decomposition(self, response: str, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LLM task decomposition response."""
        tasks = []
        task_blocks = response.split('---')
        task_id_base = problem_analysis['problem_id']
        
        import re
        
        for i, block in enumerate(task_blocks):
            if not block.strip():
                continue
                
            try:
                # Extract domain
                domain_match = re.search(r'DOMAIN:\s*\[([^\]]+)\]', block)
                if not domain_match:
                    continue
                domain = domain_match.group(1).strip()
                
                # Extract task description
                task_match = re.search(r'TASK:\s*\[([^\]]+)\]', block)
                if not task_match:
                    continue
                task_desc = task_match.group(1).strip()
                
                # Extract expected outputs
                outputs_match = re.search(r'OUTPUTS:\s*\[([^\]]+)\]', block)
                outputs = []
                if outputs_match:
                    outputs_str = outputs_match.group(1)
                    outputs = [o.strip() for o in outputs_str.split('|')]
                
                # Extract priority
                priority_match = re.search(r'PRIORITY:\s*\[([^\]]+)\]', block)
                priority = 'medium'  # default
                if priority_match:
                    priority = priority_match.group(1).strip().lower()
                
                task = {
                    'id': f"{task_id_base}_task_{i+1}",
                    'expertise_required': domain,
                    'description': task_desc,
                    'expected_outputs': outputs,
                    'priority': priority,
                    'dependencies': [],
                    'deadline': None
                }
                tasks.append(task)
                
            except Exception as e:
                logger.warning(f"Failed to parse task block: {e}")
                continue
        
        return tasks
    
    def _fallback_task_decomposition(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback task decomposition method."""
        problem_desc = problem_analysis['description']
        required_expertise = problem_analysis['required_expertise']
        
        tasks = []
        task_id_base = problem_analysis['problem_id']
        
        # Generic task templates based on expertise type
        for i, expertise in enumerate(required_expertise):
            task = {
                'id': f"{task_id_base}_task_{i+1}",
                'expertise_required': expertise,
                'description': f"Analyze {expertise} aspects of: {problem_desc}",
                'expected_outputs': [
                    f"{expertise.title()} analysis",
                    "Key findings and insights",
                    "Recommendations for further research"
                ],
                'priority': 'high' if expertise in problem_analysis.get('priority_domains', []) else 'medium',
                'dependencies': [],
                'deadline': None
            }
            tasks.append(task)
        
        return tasks
    
    def coordinate_research_session(self, problem_description: str, 
                                  marketplace, constraints: Optional[Dict[str, Any]] = None,
                                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Coordinate a complete research session.
        
        Args:
            problem_description: Research problem description
            marketplace: AgentMarketplace instance
            constraints: Optional constraints
            session_id: Optional session ID to use (if not provided, will generate one)
            
        Returns:
            Complete research session results
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        logger.info(f"Starting research session: {session_id}")
        
        # Set current research context for dynamic agent creation
        self.current_research_context = problem_description
        
        session_data = {
            'session_id': session_id,
            'start_time': time.time(),
            'problem_description': problem_description,
            'status': 'running'
        }
        
        try:
            # Step 1: Analyze problem
            analysis = self.analyze_research_problem(problem_description, constraints)
            session_data['analysis'] = analysis
            
            # Step 2: Hire required agents
            hiring_result = self.hire_agents(marketplace, analysis['required_expertise'], constraints)
            
            # Convert agent objects to dictionaries for JSON serialization
            serializable_hiring_result = {
                'hired_agents': {expertise: agent.to_dict() for expertise, agent in hiring_result['hired_agents'].items()},
                'hiring_decisions': hiring_result['hiring_decisions'],
                'total_hired': hiring_result['total_hired'],
                'new_agents_created': hiring_result['new_agents_created']
            }
            session_data['hired_agents'] = serializable_hiring_result
            
            # Step 3: Decompose into tasks
            tasks = self.decompose_research_task(analysis)
            session_data['tasks'] = tasks
            
            # Step 4: Assign tasks to agents
            task_assignments = self._assign_tasks_to_agents(tasks, hiring_result['hired_agents'])
            session_data['task_assignments'] = task_assignments
            
            # Step 5: Execute tasks (enhanced implementation)
            results = self._execute_research_tasks(task_assignments)
            session_data['results'] = results
            
            # Step 6: Synthesize findings
            synthesis = self._synthesize_research_findings(results)
            session_data['synthesis'] = synthesis
            
            session_data['status'] = 'completed'
            session_data['end_time'] = time.time()
            
        except Exception as e:
            session_data['status'] = 'failed'
            session_data['error'] = str(e)
            logger.error(f"Research session {session_id} failed: {e}")
        finally:
            # Clear research context
            self.current_research_context = ""
        
        # Store session
        self.active_research_sessions[session_id] = session_data
        self.research_history.append(session_data)
        
        return session_data
    
    def _assign_tasks_to_agents(self, tasks: List[Dict[str, Any]], 
                               hired_agents: Dict[str, Any]) -> Dict[str, Any]:
        """Assign tasks to hired agents."""
        assignments = {}
        
        # Extract agent dictionaries from hired_agents structure
        agent_dicts = hired_agents.get('hired_agents', {})
        
        for task in tasks:
            expertise = task['expertise_required']
            if expertise in agent_dicts:
                agent_dict = agent_dicts[expertise]
                # Create a simple agent object for task assignment
                simple_agent = self._create_simple_agent_from_dict(agent_dict)
                if simple_agent.assign_task(task):
                    assignments[task['id']] = {
                        'agent_id': simple_agent.agent_id,
                        'task': task,
                        'status': 'assigned'
                    }
                    logger.info(f"Assigned task {task['id']} to {simple_agent.agent_id}")
        
        return assignments
    
    def _create_simple_agent_from_dict(self, agent_dict: Dict[str, Any]) -> BaseAgent:
        """Create a simple agent object from dictionary for task assignment."""
        from agents.base_agent import BaseAgent
        
        class SimpleAgent(BaseAgent):
            def __init__(self, agent_dict):
                super().__init__(
                    agent_id=agent_dict['agent_id'],
                    role=agent_dict['role'],
                    expertise=agent_dict['expertise']
                )
            
            def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
                """Generate a simple response based on role and expertise."""
                return f"Agent {self.agent_id} ({self.role}) response: {prompt[:100]}..."
        
        return SimpleAgent(agent_dict)
    
    def _execute_research_tasks(self, task_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned research tasks with real agent interaction."""
        results = {}
        
        for task_id, assignment in task_assignments.items():
            try:
                # Get the assigned agent from our hired agents
                agent_id = assignment['agent_id']
                agent = None
                
                # Find the agent instance from our stored hired agents
                for hired_agent in self.hired_agents.values():
                    if hired_agent.agent_id == agent_id:
                        agent = hired_agent
                        break
                
                # If not found in stored agents, create a simple agent for execution
                if not agent:
                    logger.warning(f"Agent {agent_id} not found in stored agents, creating simple agent")
                    # We would need to get the agent dict from the session data
                    # For now, create a simple agent
                    agent = self._create_simple_agent_from_dict({
                        'agent_id': agent_id,
                        'role': 'Unknown Agent',
                        'expertise': ['general']
                    })
                
                # Assign task to agent
                task_data = assignment['task']
                agent.assign_task(task_data)
                
                # Generate task-specific prompt
                task_prompt = f"""
                Task: {task_data['description']}
                Expected Outputs: {', '.join(task_data['expected_outputs'])}
                Priority: {task_data['priority']}
                
                Please provide a comprehensive analysis addressing all expected outputs.
                """
                
                # Get agent response
                agent_response = agent.generate_response(task_prompt, {'task': task_data})
                
                # Complete the task
                completion_data = agent.complete_task(agent_response)
                
                results[task_id] = {
                    'task_id': task_id,
                    'agent_id': agent_id,
                    'agent_role': agent.role,
                    'status': 'completed',
                    'output': agent_response,
                    'completion_data': completion_data,
                    'completion_time': time.time()
                }
                
                logger.info(f"Task {task_id} completed by {agent_id}")
                
            except Exception as e:
                logger.error(f"Failed to execute task {task_id}: {e}")
                results[task_id] = {
                    'task_id': task_id,
                    'agent_id': assignment.get('agent_id', 'unknown'),
                    'status': 'failed',
                    'error': str(e),
                    'completion_time': time.time()
                }
        
        return results
    
    def _synthesize_research_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings from all research tasks using LLM analysis."""
        
        # Collect all outputs from agents
        agent_outputs = []
        for task_id, result in results.items():
            if result['status'] == 'completed':
                agent_outputs.append({
                    'task_id': task_id,
                    'agent_role': result.get('agent_role', 'Unknown'),
                    'output': result['output']
                })
        
        # Use LLM to synthesize findings
        synthesis_prompt = f"""
        As a Principal Investigator, synthesize the findings from multiple expert agents into a coherent research summary:
        
        Expert Outputs:
        {self._format_agent_outputs_for_synthesis(agent_outputs)}
        
        Please provide:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (3-5 main points)
        3. Cross-domain Insights (connections between different expert perspectives)
        4. Research Recommendations (next steps)
        5. Confidence Assessment (scale 1-10 with justification)
        6. Identified Knowledge Gaps
        
        Format as:
        SUMMARY: [executive summary]
        FINDINGS: [finding1 | finding2 | finding3]
        INSIGHTS: [insight1 | insight2]
        RECOMMENDATIONS: [rec1 | rec2 | rec3]
        CONFIDENCE: [number] - [justification]
        GAPS: [gap1 | gap2]
        """
        
        llm_response = self.llm_client.generate_response(
            synthesis_prompt, 
            {'agent_outputs': agent_outputs}, 
            agent_role=self.role
        )
        
        # Parse synthesis response
        synthesis = self._parse_synthesis_response(llm_response)
        
        # Add metadata
        synthesis.update({
            'synthesis_time': time.time(),
            'num_agent_contributions': len(agent_outputs),
            'successful_tasks': len([r for r in results.values() if r['status'] == 'completed']),
            'failed_tasks': len([r for r in results.values() if r['status'] == 'failed'])
        })
        
        return synthesis
    
    def _format_agent_outputs_for_synthesis(self, agent_outputs: List[Dict[str, Any]]) -> str:
        """Format agent outputs for synthesis prompt."""
        formatted = []
        for i, output in enumerate(agent_outputs, 1):
            formatted.append(f"""
            {i}. {output['agent_role']} Analysis:
            {output['output'][:500]}{'...' if len(output['output']) > 500 else ''}
            """)
        return '\n'.join(formatted)
    
    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM synthesis response."""
        import re
        
        synthesis = {
            'summary': "Research synthesis completed",
            'key_findings': [],
            'cross_domain_insights': [],
            'recommendations': [],
            'confidence_score': 0.7,
            'confidence_justification': "Default assessment",
            'knowledge_gaps': []
        }
        
        try:
            # Extract summary
            summary_match = re.search(r'SUMMARY:\s*([^\n]+)', response)
            if summary_match:
                synthesis['summary'] = summary_match.group(1).strip()
            
            # Extract findings
            findings_match = re.search(r'FINDINGS:\s*([^\n]+)', response)
            if findings_match:
                findings_str = findings_match.group(1)
                synthesis['key_findings'] = [f.strip() for f in findings_str.split('|')]
            
            # Extract insights
            insights_match = re.search(r'INSIGHTS:\s*([^\n]+)', response)
            if insights_match:
                insights_str = insights_match.group(1)
                synthesis['cross_domain_insights'] = [i.strip() for i in insights_str.split('|')]
            
            # Extract recommendations
            recs_match = re.search(r'RECOMMENDATIONS:\s*([^\n]+)', response)
            if recs_match:
                recs_str = recs_match.group(1)
                synthesis['recommendations'] = [r.strip() for r in recs_str.split('|')]
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)\s*-\s*([^\n]+)', response)
            if conf_match:
                synthesis['confidence_score'] = int(conf_match.group(1)) / 10.0
                synthesis['confidence_justification'] = conf_match.group(2).strip()
            
            # Extract gaps
            gaps_match = re.search(r'GAPS:\s*([^\n]+)', response)
            if gaps_match:
                gaps_str = gaps_match.group(1)
                synthesis['knowledge_gaps'] = [g.strip() for g in gaps_str.split('|')]
                
        except Exception as e:
            logger.warning(f"Failed to parse synthesis response: {e}")
        
        return synthesis
    
    def get_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific research session by ID."""
        return self.active_research_sessions.get(session_id)
    
    def list_research_sessions(self) -> List[str]:
        """List all active research session IDs."""
        return list(self.active_research_sessions.keys())
    
    def get_research_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get research history."""
        if limit:
            return self.research_history[-limit:]
        return self.research_history.copy()