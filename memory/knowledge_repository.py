"""
Knowledge Repository for managing validated research findings and insights.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from .vector_database import VectorDatabase

logger = logging.getLogger(__name__)


class KnowledgeRepository:
    """
    Repository for managing validated research findings, agent performance,
    and institutional knowledge accumulated over time.
    """
    
    def __init__(self, vector_db: VectorDatabase):
        """
        Initialize knowledge repository.
        
        Args:
            vector_db: VectorDatabase instance for storage/retrieval
        """
        self.vector_db = vector_db
        self.validated_findings = {}
        self.agent_performance_history = {}
        self.research_patterns = {}
        
        logger.info("KnowledgeRepository initialized")
    
    def add_validated_finding(self, finding_text: str, research_domain: str,
                            confidence_score: float, evidence_sources: List[str],
                            validating_agents: List[str], session_id: str,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a validated research finding to the repository.
        
        Args:
            finding_text: The research finding
            research_domain: Domain of research (e.g., 'ophthalmology', 'psychology')
            confidence_score: Confidence in the finding (0.0-1.0)
            evidence_sources: List of evidence sources
            validating_agents: List of agents that validated this finding
            session_id: Research session ID
            metadata: Additional metadata
            
        Returns:
            Finding ID
        """
        finding_id = f"finding_{int(time.time())}_{hash(finding_text) % 10000}"
        
        finding_data = {
            'finding_id': finding_id,
            'finding_text': finding_text,
            'research_domain': research_domain,
            'confidence_score': confidence_score,
            'evidence_sources': evidence_sources,
            'validating_agents': validating_agents,
            'session_id': session_id,
            'timestamp': time.time(),
            'metadata': metadata or {},
            'citation_count': 0,
            'last_cited': None
        }
        
        self.validated_findings[finding_id] = finding_data
        
        # Store in vector database with high importance
        self.vector_db.store_content(
            content=finding_text,
            content_type="validated_finding",
            session_id=session_id,
            importance_score=0.9,  # High importance for validated findings
            metadata={
                'finding_id': finding_id,
                'research_domain': research_domain,
                'confidence_score': confidence_score,
                'validating_agents': validating_agents
            }
        )
        
        logger.info(f"Added validated finding: {finding_id} (confidence: {confidence_score})")
        return finding_id
    
    def search_findings(self, query: str, research_domain: Optional[str] = None,
                       min_confidence: float = 0.5, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant validated findings.
        
        Args:
            query: Search query
            research_domain: Optional domain filter
            min_confidence: Minimum confidence score
            limit: Maximum number of results
            
        Returns:
            List of matching findings
        """
        # Use vector database for semantic search
        vector_results = self.vector_db.search_similar(
            query=query,
            limit=limit * 2,  # Get more results for filtering
            content_type="validated_finding",
            min_importance=0.8  # Only high-importance validated findings
        )
        
        # Filter and enhance results
        findings = []
        for result in vector_results:
            finding_metadata = result.get('metadata', {})
            finding_id = finding_metadata.get('finding_id')
            
            if finding_id and finding_id in self.validated_findings:
                finding_data = self.validated_findings[finding_id].copy()
                
                # Apply filters
                if research_domain and finding_data['research_domain'] != research_domain:
                    continue
                if finding_data['confidence_score'] < min_confidence:
                    continue
                
                finding_data['similarity_score'] = result['similarity_score']
                findings.append(finding_data)
                
                if len(findings) >= limit:
                    break
        
        logger.debug(f"Found {len(findings)} validated findings for query: {query}")
        return findings
    
    def cite_finding(self, finding_id: str, citing_session: str) -> bool:
        """
        Record that a finding has been cited/used.
        
        Args:
            finding_id: ID of the finding being cited
            citing_session: Session that cited the finding
            
        Returns:
            True if citation recorded successfully
        """
        if finding_id not in self.validated_findings:
            logger.warning(f"Finding {finding_id} not found for citation")
            return False
        
        finding = self.validated_findings[finding_id]
        finding['citation_count'] += 1
        finding['last_cited'] = time.time()
        
        # Store citation metadata
        if 'citations' not in finding:
            finding['citations'] = []
        
        finding['citations'].append({
            'citing_session': citing_session,
            'citation_time': time.time()
        })
        
        logger.debug(f"Recorded citation for finding {finding_id} (total: {finding['citation_count']})")
        return True
    
    def record_agent_performance(self, agent_id: str, task_type: str,
                               performance_score: float, task_context: Dict[str, Any]):
        """
        Record agent performance for a specific task.
        
        Args:
            agent_id: ID of the agent
            task_type: Type of task performed
            performance_score: Performance score (0.0-1.0)
            task_context: Context about the task
        """
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []
        
        performance_record = {
            'task_type': task_type,
            'performance_score': performance_score,
            'task_context': task_context,
            'timestamp': time.time()
        }
        
        self.agent_performance_history[agent_id].append(performance_record)
        
        # Store in vector database for retrieval
        self.vector_db.store_content(
            content=f"Agent {agent_id} performed {task_type} with score {performance_score}",
            content_type="performance_record",
            agent_id=agent_id,
            importance_score=0.6,
            metadata=performance_record
        )
        
        logger.debug(f"Recorded performance for agent {agent_id}: {task_type} = {performance_score}")
    
    def get_agent_performance_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Performance summary or None if no data
        """
        if agent_id not in self.agent_performance_history:
            return None
        
        records = self.agent_performance_history[agent_id]
        if not records:
            return None
        
        # Calculate aggregated metrics
        total_tasks = len(records)
        scores = [r['performance_score'] for r in records]
        avg_score = sum(scores) / total_tasks
        
        # Performance by task type
        by_task_type = {}
        for record in records:
            task_type = record['task_type']
            if task_type not in by_task_type:
                by_task_type[task_type] = []
            by_task_type[task_type].append(record['performance_score'])
        
        # Calculate averages by task type
        task_type_averages = {
            task_type: sum(scores) / len(scores)
            for task_type, scores in by_task_type.items()
        }
        
        # Recent performance (last 10 tasks)
        recent_records = records[-10:]
        recent_avg = sum(r['performance_score'] for r in recent_records) / len(recent_records)
        
        return {
            'agent_id': agent_id,
            'total_tasks': total_tasks,
            'average_performance': avg_score,
            'recent_performance': recent_avg,
            'performance_by_task_type': task_type_averages,
            'performance_trend': self._calculate_trend(scores),
            'last_task_time': max(r['timestamp'] for r in records)
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate performance trend from scores."""
        if len(scores) < 3:
            return "insufficient_data"
        
        # Compare first and last thirds
        first_third = scores[:len(scores)//3]
        last_third = scores[-len(scores)//3:]
        
        first_avg = sum(first_third) / len(first_third)
        last_avg = sum(last_third) / len(last_third)
        
        diff = last_avg - first_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def discover_research_patterns(self, min_occurrences: int = 3) -> Dict[str, Any]:
        """
        Discover patterns in research activities and findings.
        
        Args:
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            Dictionary of discovered patterns
        """
        patterns = {
            'successful_agent_combinations': {},
            'effective_research_domains': {},
            'common_finding_themes': {},
            'performance_correlations': {}
        }
        
        # Analyze successful agent combinations
        for finding in self.validated_findings.values():
            if finding['confidence_score'] >= 0.8:  # High-confidence findings
                agents = tuple(sorted(finding['validating_agents']))
                if len(agents) > 1:  # Multi-agent validation
                    patterns['successful_agent_combinations'][agents] = \
                        patterns['successful_agent_combinations'].get(agents, 0) + 1
        
        # Filter patterns by minimum occurrences
        patterns['successful_agent_combinations'] = {
            agents: count for agents, count in patterns['successful_agent_combinations'].items()
            if count >= min_occurrences
        }
        
        # Analyze effective research domains
        domain_success = {}
        for finding in self.validated_findings.values():
            domain = finding['research_domain']
            confidence = finding['confidence_score']
            
            if domain not in domain_success:
                domain_success[domain] = []
            domain_success[domain].append(confidence)
        
        patterns['effective_research_domains'] = {
            domain: {
                'average_confidence': sum(scores) / len(scores),
                'finding_count': len(scores)
            }
            for domain, scores in domain_success.items()
            if len(scores) >= min_occurrences
        }
        
        logger.info(f"Discovered {len(patterns['successful_agent_combinations'])} agent combination patterns")
        return patterns
    
    def get_knowledge_recommendations(self, current_research: str,
                                    current_agents: List[str]) -> Dict[str, Any]:
        """
        Get recommendations based on accumulated knowledge.
        
        Args:
            current_research: Description of current research
            current_agents: List of currently involved agents
            
        Returns:
            Recommendations dictionary
        """
        recommendations = {
            'relevant_findings': [],
            'suggested_agents': [],
            'research_approaches': [],
            'potential_pitfalls': []
        }
        
        # Find relevant validated findings
        relevant_findings = self.search_findings(current_research, limit=3)
        recommendations['relevant_findings'] = [
            {
                'finding_text': f['finding_text'],
                'confidence_score': f['confidence_score'],
                'citation_count': f['citation_count']
            }
            for f in relevant_findings
        ]
        
        # Analyze patterns for agent recommendations
        patterns = self.discover_research_patterns()
        successful_combinations = patterns['successful_agent_combinations']
        
        # Suggest additional agents based on successful combinations
        for agent_combo, success_count in successful_combinations.items():
            overlap = set(agent_combo) & set(current_agents)
            if overlap and len(overlap) < len(agent_combo):
                missing_agents = set(agent_combo) - set(current_agents)
                recommendations['suggested_agents'].extend(list(missing_agents))
        
        # Remove duplicates and limit suggestions
        recommendations['suggested_agents'] = list(set(recommendations['suggested_agents']))[:3]
        
        return recommendations
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get knowledge repository statistics."""
        total_findings = len(self.validated_findings)
        
        if total_findings == 0:
            return {
                'total_findings': 0,
                'total_agents_tracked': len(self.agent_performance_history),
                'vector_db_stats': self.vector_db.get_stats()
            }
        
        # Finding statistics
        confidence_scores = [f['confidence_score'] for f in self.validated_findings.values()]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        domains = [f['research_domain'] for f in self.validated_findings.values()]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        citation_counts = [f['citation_count'] for f in self.validated_findings.values()]
        total_citations = sum(citation_counts)
        
        # Agent performance statistics
        total_agents_tracked = len(self.agent_performance_history)
        total_performance_records = sum(len(records) for records in self.agent_performance_history.values())
        
        return {
            'total_findings': total_findings,
            'average_confidence': avg_confidence,
            'findings_by_domain': domain_counts,
            'total_citations': total_citations,
            'most_cited_finding': max(citation_counts) if citation_counts else 0,
            'total_agents_tracked': total_agents_tracked,
            'total_performance_records': total_performance_records,
            'vector_db_stats': self.vector_db.get_stats()
        }
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """
        Export the complete knowledge base for backup or analysis.
        
        Returns:
            Complete knowledge base data
        """
        return {
            'validated_findings': self.validated_findings,
            'agent_performance_history': self.agent_performance_history,
            'export_timestamp': time.time(),
            'repository_stats': self.get_repository_stats()
        }
    
    def import_knowledge_base(self, knowledge_data: Dict[str, Any]) -> bool:
        """
        Import knowledge base data.
        
        Args:
            knowledge_data: Knowledge base data to import
            
        Returns:
            True if import successful
        """
        try:
            if 'validated_findings' in knowledge_data:
                self.validated_findings.update(knowledge_data['validated_findings'])
            
            if 'agent_performance_history' in knowledge_data:
                for agent_id, records in knowledge_data['agent_performance_history'].items():
                    if agent_id not in self.agent_performance_history:
                        self.agent_performance_history[agent_id] = []
                    self.agent_performance_history[agent_id].extend(records)
            
            logger.info("Knowledge base import completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge base import failed: {e}")
            return False