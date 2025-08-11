"""
LeReT-based Retrieval Optimization System

Implements multi-hop retrieval with prompt-driven diverse query generation,
preference-based reinforcement learning (IPO) for query refinement, and
iterative fine-tuning loops to improve retrieval over time.

Based on the LeReT (Learning to Retrieve) framework for advanced information retrieval.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import random

from .literature_retriever import LiteratureRetriever

logger = logging.getLogger(__name__)


@dataclass
class QueryGenerationStrategy:
    """Strategy for generating diverse queries."""
    name: str
    description: str
    generation_function: Callable
    diversity_score: float = 1.0


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    documents: List[Dict[str, Any]]
    relevance_scores: List[float]
    strategy_used: str
    timestamp: float
    metadata: Dict[str, Any]


class LeReTRetrievalOptimizer:
    """
    LeReT-based retrieval optimization system.
    
    Implements:
    - Multi-hop retrieval with diverse query generation
    - Preference-based reinforcement learning for query refinement
    - Iterative fine-tuning loops
    - Direct and indirect supervision support
    """
    
    def __init__(self, literature_retriever: LiteratureRetriever, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize LeReT retrieval optimizer.
        
        Args:
            literature_retriever: Base literature retriever
            config: Configuration dictionary
        """
        self.literature_retriever = literature_retriever
        self.config = config or {}
        
        # Query generation strategies
        self.query_strategies = self._initialize_query_strategies()
        
        # Preference learning components
        self.preference_model = self._initialize_preference_model()
        self.query_history = []
        self.feedback_history = []
        
        # Multi-hop retrieval state
        self.hop_results = {}
        self.query_chains = {}
        
        # Fine-tuning state
        self.performance_metrics = {}
        self.optimization_iterations = 0
        
        # Supervision data
        self.direct_supervision_data = []
        self.indirect_supervision_data = []
        
        logger.info("LeReT Retrieval Optimizer initialized")
    
    def _initialize_query_strategies(self) -> Dict[str, QueryGenerationStrategy]:
        """Initialize diverse query generation strategies."""
        strategies = {}
        
        # Strategy 1: Keyword expansion
        strategies['keyword_expansion'] = QueryGenerationStrategy(
            name="Keyword Expansion",
            description="Expand query with related terms and synonyms",
            generation_function=self._generate_keyword_expansion_queries,
            diversity_score=0.8
        )
        
        # Strategy 2: Question reformulation
        strategies['question_reformulation'] = QueryGenerationStrategy(
            name="Question Reformulation",
            description="Reformulate query as different question types",
            generation_function=self._generate_question_reformulation_queries,
            diversity_score=0.9
        )
        
        # Strategy 3: Technical specification
        strategies['technical_specification'] = QueryGenerationStrategy(
            name="Technical Specification",
            description="Add technical details and specifications",
            generation_function=self._generate_technical_queries,
            diversity_score=0.7
        )
        
        # Strategy 4: Contextual broadening
        strategies['contextual_broadening'] = QueryGenerationStrategy(
            name="Contextual Broadening",
            description="Broaden query with related contexts",
            generation_function=self._generate_contextual_queries,
            diversity_score=0.6
        )
        
        # Strategy 5: Methodological focus
        strategies['methodological_focus'] = QueryGenerationStrategy(
            name="Methodological Focus",
            description="Focus on specific methodologies and approaches",
            generation_function=self._generate_methodological_queries,
            diversity_score=0.8
        )
        
        return strategies
    
    def _initialize_preference_model(self) -> Dict[str, Any]:
        """Initialize preference-based reinforcement learning model."""
        return {
            'query_embeddings': {},
            'preference_weights': {},
            'learning_rate': self.config.get('learning_rate', 0.01),
            'exploration_rate': self.config.get('exploration_rate', 0.1),
            'temperature': self.config.get('temperature', 0.7)
        }
    
    def multi_hop_retrieval(self, initial_query: str, max_hops: int = 3, 
                           max_results_per_hop: int = 10) -> Dict[str, Any]:
        """
        Perform multi-hop retrieval with diverse query generation.
        
        Args:
            initial_query: Initial search query
            max_hops: Maximum number of retrieval hops
            max_results_per_hop: Maximum results per hop
            
        Returns:
            Multi-hop retrieval results with query chains and documents
        """
        logger.info(f"Starting multi-hop retrieval: {initial_query}")
        
        session_id = f"multihop_{int(time.time())}"
        current_query = initial_query
        all_results = []
        query_chain = [initial_query]
        
        for hop in range(max_hops):
            logger.info(f"Executing hop {hop + 1}/{max_hops}")
            
            # Generate diverse queries for current hop
            diverse_queries = self._generate_diverse_queries(current_query, hop)
            
            # Execute retrieval with preference-based selection
            hop_results = self._execute_hop_retrieval(
                diverse_queries, max_results_per_hop, hop
            )
            
            # Store hop results
            self.hop_results[f"{session_id}_hop_{hop}"] = hop_results
            all_results.extend(hop_results['documents'])
            
            # Generate next hop query based on retrieved documents
            if hop < max_hops - 1 and hop_results['documents']:
                current_query = self._generate_next_hop_query(
                    hop_results['documents'], initial_query, hop
                )
                query_chain.append(current_query)
            else:
                break
        
        # Store query chain
        self.query_chains[session_id] = query_chain
        
        # Remove duplicates and rank by relevance
        unique_results = self._remove_duplicates(all_results)
        ranked_results = self._rank_by_relevance(unique_results, initial_query)
        
        return {
            'session_id': session_id,
            'query_chain': query_chain,
            'documents': ranked_results[:max_results_per_hop * max_hops],
            'hop_results': self.hop_results,
            'total_hops': len(query_chain),
            'total_documents': len(ranked_results)
        }
    
    def _generate_diverse_queries(self, base_query: str, hop: int) -> List[str]:
        """Generate diverse queries using multiple strategies."""
        queries = [base_query]  # Always include original query
        
        # Select strategies based on hop and diversity requirements
        strategy_names = list(self.query_strategies.keys())
        num_strategies = min(3, len(strategy_names))  # Use 3 strategies per hop
        
        selected_strategies = random.sample(strategy_names, num_strategies)
        
        for strategy_name in selected_strategies:
            strategy = self.query_strategies[strategy_name]
            try:
                generated_queries = strategy.generation_function(base_query, hop)
                queries.extend(generated_queries)
            except Exception as e:
                logger.warning(f"Failed to generate queries with strategy {strategy_name}: {e}")
        
        # Apply preference-based selection if we have too many queries
        if len(queries) > 5:
            queries = self._select_preferred_queries(queries, base_query)
        
        return queries[:5]  # Limit to 5 queries per hop
    
    def _generate_keyword_expansion_queries(self, base_query: str, hop: int) -> List[str]:
        """Generate queries by expanding keywords."""
        # Simple keyword expansion - in production, use more sophisticated NLP
        keywords = base_query.lower().split()
        expanded_terms = {
            'neuron': ['neural', 'neuronal', 'neurobiology', 'neuroscience'],
            'simulation': ['modeling', 'computational', 'numerical', 'simulation'],
            'physics': ['physical', 'mechanical', 'dynamics', 'kinematics'],
            'research': ['study', 'investigation', 'analysis', 'experiment'],
            'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
            'agent': ['autonomous', 'intelligent', 'robotic', 'automated']
        }
        
        expanded_queries = []
        for keyword in keywords:
            if keyword in expanded_terms:
                for expansion in expanded_terms[keyword]:
                    new_query = base_query.replace(keyword, expansion)
                    expanded_queries.append(new_query)
        
        return expanded_queries[:3]  # Limit expansions
    
    def _generate_question_reformulation_queries(self, base_query: str, hop: int) -> List[str]:
        """Generate queries by reformulating as different question types."""
        reformulations = []
        
        # What questions
        if not base_query.lower().startswith('what'):
            reformulations.append(f"What is {base_query}?")
            reformulations.append(f"What are the {base_query}?")
        
        # How questions
        if not base_query.lower().startswith('how'):
            reformulations.append(f"How to {base_query}?")
            reformulations.append(f"How does {base_query} work?")
        
        # Why questions
        if not base_query.lower().startswith('why'):
            reformulations.append(f"Why is {base_query} important?")
        
        return reformulations[:3]
    
    def _generate_technical_queries(self, base_query: str, hop: int) -> List[str]:
        """Generate technical specification queries."""
        technical_terms = [
            'algorithm', 'methodology', 'framework', 'architecture',
            'implementation', 'optimization', 'performance', 'scalability'
        ]
        
        technical_queries = []
        for term in technical_terms:
            technical_queries.append(f"{base_query} {term}")
        
        return technical_queries[:3]
    
    def _generate_contextual_queries(self, base_query: str, hop: int) -> List[str]:
        """Generate contextually broadened queries."""
        contexts = [
            'in neuroscience', 'in physics', 'in AI research',
            'for autonomous systems', 'in computational modeling'
        ]
        
        contextual_queries = []
        for context in contexts:
            contextual_queries.append(f"{base_query} {context}")
        
        return contextual_queries[:3]
    
    def _generate_methodological_queries(self, base_query: str, hop: int) -> List[str]:
        """Generate methodology-focused queries."""
        methodologies = [
            'simulation methods', 'experimental design', 'data analysis',
            'validation techniques', 'benchmarking approaches'
        ]
        
        methodological_queries = []
        for methodology in methodologies:
            methodological_queries.append(f"{base_query} {methodology}")
        
        return methodological_queries[:3]
    
    def _select_preferred_queries(self, queries: List[str], base_query: str) -> List[str]:
        """Select preferred queries using preference-based RL."""
        if not self.preference_model['query_embeddings']:
            # No preference model yet, use diversity-based selection
            return self._diversity_based_selection(queries)
        
        # Calculate preference scores
        preference_scores = []
        for query in queries:
            score = self._calculate_preference_score(query, base_query)
            preference_scores.append(score)
        
        # Select queries with highest preference scores
        query_score_pairs = list(zip(queries, preference_scores))
        query_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [query for query, _ in query_score_pairs[:5]]
    
    def _diversity_based_selection(self, queries: List[str]) -> List[str]:
        """Select diverse queries based on content diversity."""
        # Simple diversity: prefer queries with different starting words
        selected = [queries[0]]  # Always include first query
        
        for query in queries[1:]:
            if len(selected) >= 5:
                break
            
            # Check if query starts with different words
            query_start = query.split()[0].lower()
            is_diverse = True
            
            for selected_query in selected:
                selected_start = selected_query.split()[0].lower()
                if query_start == selected_start:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(query)
        
        return selected
    
    def _calculate_preference_score(self, query: str, base_query: str) -> float:
        """Calculate preference score for a query."""
        # Simple preference scoring - in production, use learned embeddings
        base_terms = set(base_query.lower().split())
        query_terms = set(query.lower().split())
        
        # Term overlap
        overlap = len(base_terms.intersection(query_terms))
        total_terms = len(base_terms.union(query_terms))
        
        if total_terms == 0:
            return 0.0
        
        overlap_score = overlap / total_terms
        
        # Length penalty (prefer queries of similar length)
        length_diff = abs(len(query) - len(base_query))
        length_penalty = max(0, 1 - length_diff / 100)
        
        # Combine scores
        preference_score = 0.7 * overlap_score + 0.3 * length_penalty
        
        return preference_score
    
    def _execute_hop_retrieval(self, queries: List[str], max_results: int, 
                              hop: int) -> Dict[str, Any]:
        """Execute retrieval for a single hop."""
        all_documents = []
        query_results = {}
        
        for query in queries:
            try:
                # Use base literature retriever
                results = self.literature_retriever.search(
                    query=query,
                    max_results=max_results // len(queries),
                    sources=['pubmed', 'arxiv', 'semantic_scholar']
                )
                
                query_results[query] = results
                all_documents.extend(results)
                
                # Store query for preference learning
                self.query_history.append({
                    'query': query,
                    'hop': hop,
                    'results_count': len(results),
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Failed to execute query '{query}': {e}")
                query_results[query] = []
        
        return {
            'queries': queries,
            'query_results': query_results,
            'documents': all_documents,
            'hop': hop,
            'timestamp': time.time()
        }
    
    def _generate_next_hop_query(self, documents: List[Dict[str, Any]], 
                                initial_query: str, hop: int) -> str:
        """Generate query for next hop based on retrieved documents."""
        if not documents:
            return initial_query
        
        # Extract key terms from top documents
        key_terms = []
        for doc in documents[:3]:  # Use top 3 documents
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            
            # Simple term extraction (in production, use NLP)
            text = f"{title} {abstract}".lower()
            words = text.split()
            
            # Filter for meaningful terms
            meaningful_terms = [word for word in words 
                              if len(word) > 3 and word.isalpha()]
            
            key_terms.extend(meaningful_terms[:5])  # Top 5 terms per document
        
        # Select most common terms
        term_counts = {}
        for term in key_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        # Get top terms
        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        selected_terms = [term for term, _ in top_terms[:3]]
        
        # Generate next hop query
        if selected_terms:
            next_query = f"{initial_query} {' '.join(selected_terms)}"
        else:
            next_query = initial_query
        
        return next_query
    
    def _remove_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on title and authors."""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            title = doc.get('title', '').lower()
            authors = ' '.join(doc.get('authors', [])).lower()
            
            # Create unique identifier
            doc_id = f"{title}_{authors}"
            
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _rank_by_relevance(self, documents: List[Dict[str, Any]], 
                          query: str) -> List[Dict[str, Any]]:
        """Rank documents by relevance to query."""
        if not documents:
            return []
        
        # Calculate relevance scores
        scored_docs = []
        query_terms = set(query.lower().split())
        
        for doc in documents:
            title = doc.get('title', '').lower()
            abstract = doc.get('abstract', '').lower()
            
            # Title relevance (higher weight)
            title_terms = set(title.split())
            title_overlap = len(query_terms.intersection(title_terms))
            title_score = title_overlap * 0.6
            
            # Abstract relevance
            abstract_terms = set(abstract.split())
            abstract_overlap = len(query_terms.intersection(abstract_terms))
            abstract_score = abstract_overlap * 0.4
            
            # Publication year bonus
            year = doc.get('publication_year')
            year_bonus = 0.0
            if year and isinstance(year, int):
                if year >= 2020:
                    year_bonus = 0.2
                elif year >= 2015:
                    year_bonus = 0.1
            
            # Citation bonus
            citations = doc.get('citation_count', 0)
            citation_bonus = min(0.1, citations / 1000) if citations else 0.0
            
            # Total score
            total_score = title_score + abstract_score + year_bonus + citation_bonus
            
            scored_docs.append((doc, total_score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]
    
    def update_preferences(self, feedback: Dict[str, Any]):
        """
        Update preference model with user feedback.
        
        Args:
            feedback: Dictionary containing feedback data
                - query: The query that was evaluated
                - documents: List of documents with relevance scores
                - user_ratings: Optional user ratings for documents
        """
        query = feedback.get('query')
        documents = feedback.get('documents', [])
        user_ratings = feedback.get('user_ratings', {})
        
        if not query or not documents:
            return
        
        # Store feedback for learning
        self.feedback_history.append({
            'query': query,
            'documents': documents,
            'user_ratings': user_ratings,
            'timestamp': time.time()
        })
        
        # Update preference weights (simplified - in production, use proper RL)
        self._update_preference_weights(query, documents, user_ratings)
        
        logger.info(f"Updated preferences for query: {query}")
    
    def _update_preference_weights(self, query: str, documents: List[Dict[str, Any]], 
                                 user_ratings: Dict[str, float]):
        """Update preference weights based on feedback."""
        # Simple weight update - in production, use proper reinforcement learning
        query_terms = query.lower().split()
        
        for term in query_terms:
            if term not in self.preference_model['preference_weights']:
                self.preference_model['preference_weights'][term] = 1.0
            
            # Calculate average document relevance
            if documents:
                avg_relevance = sum(doc.get('relevance_score', 0) for doc in documents) / len(documents)
                
                # Update weight based on relevance
                current_weight = self.preference_model['preference_weights'][term]
                learning_rate = self.preference_model['learning_rate']
                
                new_weight = current_weight + learning_rate * (avg_relevance - 0.5)
                self.preference_model['preference_weights'][term] = max(0.1, new_weight)
    
    def iterative_fine_tuning(self, training_queries: List[str], 
                            validation_queries: List[str]) -> Dict[str, Any]:
        """
        Perform iterative fine-tuning of the retrieval system.
        
        Args:
            training_queries: List of training queries
            validation_queries: List of validation queries
            
        Returns:
            Fine-tuning results and performance metrics
        """
        logger.info("Starting iterative fine-tuning")
        
        iteration = 0
        max_iterations = self.config.get('max_fine_tuning_iterations', 10)
        performance_history = []
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Fine-tuning iteration {iteration}/{max_iterations}")
            
            # Execute retrieval on training queries
            training_results = []
            for query in training_queries:
                result = self.multi_hop_retrieval(query, max_hops=2, max_results_per_hop=5)
                training_results.append(result)
            
            # Evaluate performance on validation queries
            validation_results = []
            for query in validation_queries:
                result = self.multi_hop_retrieval(query, max_hops=2, max_results_per_hop=5)
                validation_results.append(result)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(validation_results)
            performance_history.append(metrics)
            
            # Store metrics
            self.performance_metrics[f"iteration_{iteration}"] = metrics
            
            # Check for convergence
            if len(performance_history) >= 2:
                recent_improvement = (
                    performance_history[-1]['avg_relevance'] - 
                    performance_history[-2]['avg_relevance']
                )
                
                if recent_improvement < 0.01:  # Small improvement threshold
                    logger.info(f"Convergence reached at iteration {iteration}")
                    break
            
            # Update preference model based on training results
            self._update_model_from_training(training_results)
        
        self.optimization_iterations = iteration
        
        return {
            'iterations': iteration,
            'performance_history': performance_history,
            'final_metrics': performance_history[-1] if performance_history else {},
            'converged': iteration < max_iterations
        }
    
    def _calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics for retrieval results."""
        if not results:
            return {'avg_relevance': 0.0, 'avg_documents': 0.0, 'avg_hops': 0.0}
        
        total_relevance = 0.0
        total_documents = 0
        total_hops = 0
        
        for result in results:
            documents = result.get('documents', [])
            total_documents += len(documents)
            total_hops += result.get('total_hops', 0)
            
            # Calculate average relevance for documents
            if documents:
                doc_relevance = sum(doc.get('relevance_score', 0) for doc in documents)
                total_relevance += doc_relevance / len(documents)
        
        avg_relevance = total_relevance / len(results)
        avg_documents = total_documents / len(results)
        avg_hops = total_hops / len(results)
        
        return {
            'avg_relevance': avg_relevance,
            'avg_documents': avg_documents,
            'avg_hops': avg_hops
        }
    
    def _update_model_from_training(self, training_results: List[Dict[str, Any]]):
        """Update preference model based on training results."""
        # Simple model update - in production, use proper machine learning
        for result in training_results:
            documents = result.get('documents', [])
            if documents:
                # Update preference weights based on document relevance
                avg_relevance = sum(doc.get('relevance_score', 0) for doc in documents) / len(documents)
                
                # Adjust exploration rate based on performance
                if avg_relevance > 0.7:
                    self.preference_model['exploration_rate'] *= 0.95  # Reduce exploration
                else:
                    self.preference_model['exploration_rate'] *= 1.05  # Increase exploration
                
                # Clamp exploration rate
                self.preference_model['exploration_rate'] = max(0.01, min(0.5, 
                    self.preference_model['exploration_rate']))
    
    def add_direct_supervision(self, query: str, relevant_documents: List[Dict[str, Any]]):
        """Add direct supervision data (known-good documents)."""
        self.direct_supervision_data.append({
            'query': query,
            'relevant_documents': relevant_documents,
            'timestamp': time.time()
        })
        
        logger.info(f"Added direct supervision for query: {query}")
    
    def add_indirect_supervision(self, query: str, feedback_signal: float):
        """Add indirect supervision data (feedback signals)."""
        self.indirect_supervision_data.append({
            'query': query,
            'feedback_signal': feedback_signal,
            'timestamp': time.time()
        })
        
        logger.info(f"Added indirect supervision for query: {query}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics."""
        return {
            'total_queries': len(self.query_history),
            'total_feedback': len(self.feedback_history),
            'optimization_iterations': self.optimization_iterations,
            'direct_supervision_count': len(self.direct_supervision_data),
            'indirect_supervision_count': len(self.indirect_supervision_data),
            'performance_metrics': self.performance_metrics,
            'preference_model_stats': {
                'total_weights': len(self.preference_model['preference_weights']),
                'avg_weight': np.mean(list(self.preference_model['preference_weights'].values())) 
                    if self.preference_model['preference_weights'] else 0.0
            }
        }
