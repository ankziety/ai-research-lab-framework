"""
Principal Investigator (PI) Orchestration layer for research automation.

This module implements the main orchestration logic for coordinating specialist agents,
managing research tasks, and interfacing with memory systems.
"""
import logging
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime
import uuid


class PIOrchestrator:
    """
    Principal Investigator orchestration layer for research automation.
    
    Responsibilities:
    - Register specialist agents by role/name and callable handle
    - Accept user research requests and decompose them into subtasks
    - Assign subtasks to registered specialists and aggregate results
    - Interface with vector memory for context storage/retrieval
    - Log provenance for every decision and call
    """
    
    def __init__(self):
        """Initialize the PI orchestrator."""
        self._specialists: Dict[str, Callable] = {}
        self._memory = None
        self._provenance_log: List[Dict[str, Any]] = []
        
        # Setup logging
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
    
    def register_specialist(self, role_name: str, handler: Callable) -> None:
        """
        Register a specialist agent by role name and callable handler.
        
        Args:
            role_name: The name/role of the specialist (e.g., "literature_reviewer", "data_analyst")
            handler: Callable that handles tasks for this specialist
        """
        self._specialists[role_name] = handler
        
        # Log the registration
        log_entry = {
            'action': 'register_specialist',
            'role_name': role_name,
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        self._provenance_log.append(log_entry)
        self._logger.info(f"Registered specialist: {role_name}")
        
        # Store in memory if available
        if self._memory:
            self._memory.store_context(
                f"Registered specialist {role_name}",
                {'action': 'registration', 'role': role_name}
            )
    
    def set_memory(self, memory_instance) -> None:
        """
        Set the vector memory instance for context storage/retrieval.
        
        Args:
            memory_instance: Instance of vector memory class (e.g., VectorMemory)
        """
        self._memory = memory_instance
        
        log_entry = {
            'action': 'set_memory',
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        self._provenance_log.append(log_entry)
        self._logger.info("Vector memory instance set")
    
    def run_research_task(self, request: str) -> Dict[str, Any]:
        """
        Accept a user research request, decompose it into subtasks, 
        assign to specialists, and aggregate results.
        
        Args:
            request: User research request as a string
            
        Returns:
            Dictionary containing aggregated results and metadata
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Log the start of task processing
        log_entry = {
            'action': 'start_research_task',
            'task_id': task_id,
            'request': request,
            'timestamp': start_time.isoformat(),
            'id': str(uuid.uuid4())
        }
        self._provenance_log.append(log_entry)
        self._logger.info(f"Starting research task {task_id}: {request}")
        
        # Store request in memory if available
        if self._memory:
            self._memory.store_context(
                f"Research request: {request}",
                {'task_id': task_id, 'type': 'request'}
            )
        
        try:
            # Decompose the request into subtasks
            subtasks = self._decompose_request(request, task_id)
            
            # Assign subtasks to specialists and collect results
            specialist_results = {}
            for subtask in subtasks:
                role = subtask['assigned_role']
                if role in self._specialists:
                    result = self._execute_subtask(subtask, task_id)
                    specialist_results[role] = result
                else:
                    self._logger.warning(f"No specialist registered for role: {role}")
                    specialist_results[role] = {
                        'error': f"No specialist available for role: {role}",
                        'subtask': subtask
                    }
            
            # Aggregate results
            aggregated_result = self._aggregate_results(specialist_results, request, task_id)
            
            # Log completion
            end_time = datetime.now()
            completion_log = {
                'action': 'complete_research_task',
                'task_id': task_id,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'timestamp': end_time.isoformat(),
                'id': str(uuid.uuid4())
            }
            self._provenance_log.append(completion_log)
            self._logger.info(f"Completed research task {task_id}")
            
            return aggregated_result
            
        except Exception as e:
            # Log error
            error_log = {
                'action': 'error_research_task',
                'task_id': task_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'id': str(uuid.uuid4())
            }
            self._provenance_log.append(error_log)
            self._logger.error(f"Error in research task {task_id}: {e}")
            
            return {
                'task_id': task_id,
                'status': 'error',
                'error': str(e),
                'request': request
            }
    
    def _decompose_request(self, request: str, task_id: str) -> List[Dict[str, Any]]:
        """
        Decompose a research request into subtasks.
        
        Args:
            request: The original research request
            task_id: Unique identifier for this task
            
        Returns:
            List of subtask dictionaries
        """
        # Simple decomposition logic - in a real implementation this could be more sophisticated
        subtasks = []
        
        # Determine which types of specialists are needed based on keywords
        request_lower = request.lower()
        
        if any(keyword in request_lower for keyword in ['literature', 'papers', 'research', 'review']):
            subtasks.append({
                'id': str(uuid.uuid4()),
                'task_id': task_id,
                'type': 'literature_review',
                'assigned_role': 'literature_reviewer',
                'description': f"Review literature for: {request}",
                'input': request
            })
        
        if any(keyword in request_lower for keyword in ['analyze', 'data', 'statistics', 'results']):
            subtasks.append({
                'id': str(uuid.uuid4()),
                'task_id': task_id,
                'type': 'data_analysis',
                'assigned_role': 'data_analyst',
                'description': f"Analyze data for: {request}",
                'input': request
            })
        
        if any(keyword in request_lower for keyword in ['write', 'draft', 'manuscript', 'paper']):
            subtasks.append({
                'id': str(uuid.uuid4()),
                'task_id': task_id,
                'type': 'manuscript_writing',
                'assigned_role': 'manuscript_writer',
                'description': f"Write manuscript for: {request}",
                'input': request
            })
        
        # If no specific subtasks identified, create a general research subtask
        if not subtasks:
            subtasks.append({
                'id': str(uuid.uuid4()),
                'task_id': task_id,
                'type': 'general_research',
                'assigned_role': 'general_researcher',
                'description': f"General research for: {request}",
                'input': request
            })
        
        # Log decomposition
        decomposition_log = {
            'action': 'decompose_request',
            'task_id': task_id,
            'subtasks_count': len(subtasks),
            'subtask_types': [st['type'] for st in subtasks],
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        self._provenance_log.append(decomposition_log)
        
        return subtasks
    
    def _execute_subtask(self, subtask: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """
        Execute a subtask by calling the appropriate specialist.
        
        Args:
            subtask: The subtask to execute
            task_id: Parent task ID
            
        Returns:
            Result from the specialist
        """
        role = subtask['assigned_role']
        specialist = self._specialists[role]
        
        # Log subtask execution start
        execution_log = {
            'action': 'execute_subtask',
            'task_id': task_id,
            'subtask_id': subtask['id'],
            'role': role,
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        self._provenance_log.append(execution_log)
        self._logger.info(f"Executing subtask {subtask['id']} with specialist {role}")
        
        try:
            # Call the specialist handler
            result = specialist(subtask['input'])
            
            # Store result in memory if available
            if self._memory:
                self._memory.store_context(
                    f"Specialist {role} result: {str(result)}",
                    {
                        'task_id': task_id,
                        'subtask_id': subtask['id'],
                        'role': role,
                        'type': 'specialist_result'
                    }
                )
            
            return {
                'status': 'success',
                'result': result,
                'subtask': subtask,
                'specialist': role
            }
            
        except Exception as e:
            self._logger.error(f"Error executing subtask {subtask['id']}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'subtask': subtask,
                'specialist': role
            }
    
    def _aggregate_results(self, specialist_results: Dict[str, Any], 
                          original_request: str, task_id: str) -> Dict[str, Any]:
        """
        Aggregate results from multiple specialists.
        
        Args:
            specialist_results: Results from each specialist
            original_request: The original research request
            task_id: Task identifier
            
        Returns:
            Aggregated results dictionary
        """
        # Log aggregation start
        aggregation_log = {
            'action': 'aggregate_results',
            'task_id': task_id,
            'specialist_count': len(specialist_results),
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        self._provenance_log.append(aggregation_log)
        
        # Simple aggregation - collect all successful results
        successful_results = {}
        errors = {}
        
        for role, result in specialist_results.items():
            if result.get('status') == 'success':
                successful_results[role] = result['result']
            else:
                errors[role] = result.get('error', 'Unknown error')
        
        aggregated = {
            'task_id': task_id,
            'status': 'completed',
            'request': original_request,
            'results': successful_results,
            'errors': errors if errors else None,
            'specialist_count': len(successful_results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store aggregated result in memory if available
        if self._memory:
            self._memory.store_context(
                f"Aggregated research results: {str(aggregated)}",
                {'task_id': task_id, 'type': 'aggregated_result'}
            )
        
        return aggregated
    
    def get_provenance_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete provenance log of all actions.
        
        Returns:
            List of log entries with timestamps and action details
        """
        return self._provenance_log.copy()
    
    def get_registered_specialists(self) -> List[str]:
        """
        Get list of currently registered specialist roles.
        
        Returns:
            List of specialist role names
        """
        return list(self._specialists.keys())
    
    def clear_provenance_log(self) -> None:
        """Clear the provenance log (useful for testing)."""
        self._provenance_log.clear()