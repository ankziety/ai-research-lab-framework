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
    
    def __init__(self, agent_id: str = "PI", model_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            role="Principal Investigator",
            expertise=["Research Coordination", "Task Decomposition", "Team Management"],
            model_config=model_config
        )
        self.active_research_sessions = {}
        self.hired_agents = {}
        self.research_history = []
        
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
            'integrate', 'plan', 'strategy', 'workflow'
        ]
        
        task_lower = task_description.lower()
        relevance = sum(1 for keyword in coordination_keywords if keyword in task_lower)
        return min(1.0, relevance * 0.2)  # Max relevance for coordination tasks
    
    def analyze_research_problem(self, problem_description: str, 
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a research problem and determine required expertise.
        
        Args:
            problem_description: Description of the research problem
            constraints: Optional constraints (budget, time, etc.)
            
        Returns:
            Analysis including required expertise and task breakdown
        """
        logger.info(f"PI analyzing research problem: {problem_description[:100]}...")
        
        # Simple keyword-based expertise detection
        # In a real implementation, this would use sophisticated NLP/LLM analysis
        expertise_mapping = {
            'ophthalmology': ['eye', 'vision', 'retina', 'glaucoma', 'ophthalmology', 'visual'],
            'psychology': ['mental health', 'psychology', 'behavior', 'cognitive', 'anxiety', 'depression'],
            'neuroscience': ['brain', 'neural', 'neuroscience', 'neurological', 'cortex', 'neuron'],
            'data_science': ['data', 'analysis', 'machine learning', 'statistics', 'model', 'algorithm'],
            'literature': ['literature', 'review', 'papers', 'research', 'publications', 'studies']
        }
        
        problem_lower = problem_description.lower()
        required_expertise = []
        
        for domain, keywords in expertise_mapping.items():
            if any(keyword in problem_lower for keyword in keywords):
                required_expertise.append(domain)
        
        # Always include literature researcher for context
        if 'literature' not in required_expertise:
            required_expertise.append('literature')
        
        analysis = {
            'problem_id': f"research_{int(time.time())}",
            'description': problem_description,
            'required_expertise': required_expertise,
            'constraints': constraints or {},
            'complexity_score': len(required_expertise) * 0.2,
            'estimated_agents_needed': min(len(required_expertise), 5),
            'priority_domains': required_expertise[:3]  # Top 3 most relevant
        }
        
        logger.info(f"Analysis complete. Required expertise: {required_expertise}")
        return analysis
    
    def hire_agents(self, marketplace, required_expertise: List[str], 
                   constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hire agents from the marketplace based on required expertise.
        
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
            # Get available agents for this expertise
            available_agents = marketplace.get_agents_by_expertise(expertise)
            
            if not available_agents:
                logger.warning(f"No agents available for expertise: {expertise}")
                continue
            
            # Select best agent based on performance metrics
            best_agent = self._select_best_agent(available_agents, constraints)
            
            if best_agent:
                hired_agents[expertise] = best_agent
                hiring_decisions.append({
                    'expertise': expertise,
                    'agent_id': best_agent.agent_id,
                    'performance_score': best_agent.performance_metrics['average_quality_score'],
                    'hire_time': time.time()
                })
                
                # Mark agent as hired in marketplace
                marketplace.hire_agent(best_agent.agent_id)
                
                logger.info(f"Hired {best_agent.agent_id} for {expertise}")
        
        # Store hiring decisions
        self.hired_agents.update(hired_agents)
        
        return {
            'hired_agents': hired_agents,
            'hiring_decisions': hiring_decisions,
            'total_hired': len(hired_agents)
        }
    
    def _select_best_agent(self, available_agents: List[BaseAgent], 
                          constraints: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        """
        Select the best agent from available options.
        
        Args:
            available_agents: List of available agents
            constraints: Optional selection constraints
            
        Returns:
            Best agent or None if no suitable agent found
        """
        if not available_agents:
            return None
        
        # Score agents based on performance metrics
        scored_agents = []
        for agent in available_agents:
            metrics = agent.performance_metrics
            
            # Calculate composite score
            score = (
                metrics['average_quality_score'] * 0.4 +
                metrics['success_rate'] * 0.3 +
                (1.0 if metrics['tasks_completed'] > 0 else 0.5) * 0.3
            )
            
            scored_agents.append((score, agent))
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        
        return scored_agents[0][1]
    
    def decompose_research_task(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down research problem into specific tasks for expert agents.
        
        Args:
            problem_analysis: Analysis from analyze_research_problem
            
        Returns:
            List of task dictionaries
        """
        problem_desc = problem_analysis['description']
        required_expertise = problem_analysis['required_expertise']
        
        tasks = []
        task_id_base = problem_analysis['problem_id']
        
        # Generate domain-specific tasks
        task_templates = {
            'ophthalmology': {
                'description': f"Analyze ophthalmological aspects of: {problem_desc}",
                'expected_outputs': ['clinical assessment', 'diagnostic considerations', 'treatment options']
            },
            'psychology': {
                'description': f"Evaluate psychological factors in: {problem_desc}",
                'expected_outputs': ['mental health correlations', 'behavioral patterns', 'psychological mechanisms']
            },
            'neuroscience': {
                'description': f"Examine neurological components of: {problem_desc}",
                'expected_outputs': ['neural mechanisms', 'brain regions involved', 'neurological pathways']
            },
            'data_science': {
                'description': f"Perform data analysis for: {problem_desc}",
                'expected_outputs': ['statistical analysis', 'model development', 'data insights']
            },
            'literature': {
                'description': f"Conduct literature review on: {problem_desc}",
                'expected_outputs': ['relevant publications', 'research summary', 'knowledge gaps']
            }
        }
        
        for i, expertise in enumerate(required_expertise):
            if expertise in task_templates:
                task = {
                    'id': f"{task_id_base}_task_{i+1}",
                    'expertise_required': expertise,
                    'description': task_templates[expertise]['description'],
                    'expected_outputs': task_templates[expertise]['expected_outputs'],
                    'priority': 'high' if expertise in problem_analysis['priority_domains'] else 'medium',
                    'dependencies': [],
                    'deadline': None
                }
                tasks.append(task)
        
        logger.info(f"Decomposed research into {len(tasks)} tasks")
        return tasks
    
    def coordinate_research_session(self, problem_description: str, 
                                  marketplace, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Coordinate a complete research session from problem analysis to completion.
        
        Args:
            problem_description: Research problem to investigate
            marketplace: AgentMarketplace instance
            constraints: Optional constraints
            
        Returns:
            Complete research session results
        """
        session_id = f"session_{int(time.time())}"
        logger.info(f"Starting research session: {session_id}")
        
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
            session_data['hired_agents'] = hiring_result
            
            # Step 3: Decompose into tasks
            tasks = self.decompose_research_task(analysis)
            session_data['tasks'] = tasks
            
            # Step 4: Assign tasks to agents
            task_assignments = self._assign_tasks_to_agents(tasks, hiring_result['hired_agents'])
            session_data['task_assignments'] = task_assignments
            
            # Step 5: Execute tasks (simplified - would involve agent coordination)
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
        
        # Store session
        self.active_research_sessions[session_id] = session_data
        self.research_history.append(session_data)
        
        return session_data
    
    def _assign_tasks_to_agents(self, tasks: List[Dict[str, Any]], 
                               hired_agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """Assign tasks to hired agents."""
        assignments = {}
        
        for task in tasks:
            expertise = task['expertise_required']
            if expertise in hired_agents:
                agent = hired_agents[expertise]
                if agent.assign_task(task):
                    assignments[task['id']] = {
                        'agent_id': agent.agent_id,
                        'task': task,
                        'status': 'assigned'
                    }
                    logger.info(f"Assigned task {task['id']} to {agent.agent_id}")
        
        return assignments
    
    def _execute_research_tasks(self, task_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned research tasks (simplified implementation)."""
        results = {}
        
        for task_id, assignment in task_assignments.items():
            # Simplified execution - in real implementation, this would involve
            # complex agent interaction and actual research work
            results[task_id] = {
                'task_id': task_id,
                'agent_id': assignment['agent_id'],
                'status': 'completed',
                'output': f"Research output for {assignment['task']['description']}",
                'completion_time': time.time()
            }
        
        return results
    
    def _synthesize_research_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings from all research tasks."""
        synthesis = {
            'summary': "Integrated research findings from expert agents",
            'key_findings': [],
            'recommendations': [],
            'confidence_score': 0.8,
            'synthesis_time': time.time()
        }
        
        # Extract key findings from each result
        for task_id, result in results.items():
            synthesis['key_findings'].append({
                'source_task': task_id,
                'finding': f"Key finding from {result['agent_id']}"
            })
        
        synthesis['recommendations'].append("Continue research with additional data")
        synthesis['recommendations'].append("Validate findings through experimental studies")
        
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