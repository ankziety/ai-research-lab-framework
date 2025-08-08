#!/usr/bin/env python3
"""
Accuracy Evaluation System

Provides comprehensive evaluation capabilities for AI models and physics simulations.
Follows behavioral/black-box testing philosophy with public API focus.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.base import BaseEstimator


class EvaluationType(Enum):
    """Types of evaluation supported by the system."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    PHYSICS_SIMULATION = "physics_simulation"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    physics_energy_error: Optional[float] = None
    physics_force_error: Optional[float] = None
    custom_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    evaluation_type: EvaluationType
    metrics: EvaluationMetrics
    evaluation_time: float
    additional_info: Optional[Dict[str, Any]] = None


class AccuracyEvaluator:
    """
    Comprehensive accuracy evaluation system for AI models and physics simulations.
    
    Provides behavioral/black-box evaluation focusing on public API and expected outputs.
    """
    
    def __init__(self):
        """Initialize the accuracy evaluator."""
        self.evaluation_history: List[EvaluationResult] = []
    
    def evaluate_model(
        self,
        model: BaseEstimator,
        test_data: Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float, None],
        evaluation_type: EvaluationType,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a model using the specified evaluation type.
        
        Args:
            model: The model to evaluate
            test_data: Tuple of (X_test, y_test) data, or single value for physics simulations
            evaluation_type: Type of evaluation to perform
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with metrics and timing information
        """
        start_time = time.time()
        
        # Check for invalid evaluation type first
        if not isinstance(evaluation_type, EvaluationType):
            raise ValueError(f"Unsupported evaluation type: {evaluation_type}")
        
        # Handle different test_data formats
        if evaluation_type == EvaluationType.PHYSICS_SIMULATION:
            # For physics simulations, test_data might be a single value
            if isinstance(test_data, (int, float)):
                X_test = np.array([test_data])
                y_test = np.array([test_data])  # Use same value for expected
            elif test_data is None:
                X_test = np.array([])
                y_test = np.array([])
            else:
                X_test, y_test = test_data
        else:
            # For classification/regression, expect tuple
            if not isinstance(test_data, tuple) or len(test_data) != 2:
                raise ValueError(f"test_data must be a tuple of (X_test, y_test) for {evaluation_type}")
            X_test, y_test = test_data
        
        if evaluation_type == EvaluationType.CLASSIFICATION:
            metrics = self._evaluate_classification(model, X_test, y_test)
        elif evaluation_type == EvaluationType.REGRESSION:
            metrics = self._evaluate_regression(model, X_test, y_test)
        elif evaluation_type == EvaluationType.PHYSICS_SIMULATION:
            metrics = self._evaluate_physics_simulation(model, X_test, y_test, **kwargs)
        else:
            raise ValueError(f"Unsupported evaluation type: {evaluation_type}")
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            evaluation_type=evaluation_type,
            metrics=metrics,
            evaluation_time=evaluation_time
        )
        
        self.evaluation_history.append(result)
        return result
    
    def _evaluate_classification(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate classification model."""
        y_pred = model.predict(X_test)
        
        return EvaluationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            f1_score=f1_score(y_test, y_pred, average='weighted')
        )
    
    def _evaluate_regression(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate regression model."""
        y_pred = model.predict(X_test)
        
        return EvaluationMetrics(
            mse=mean_squared_error(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
            r2_score=r2_score(y_test, y_pred)
        )
    
    def _evaluate_physics_simulation(
        self,
        model: Optional[BaseEstimator],
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> EvaluationMetrics:
        """Evaluate physics simulation model."""
        # Handle case where no model is provided (e.g., analytical validation)
        if model is None:
            # For physics simulations without a model, use the test data as prediction
            y_pred = y_test
        else:
            y_pred = model.predict(X_test)
        
        # Calculate physics-specific metrics
        energy_error = np.mean(np.abs(y_test - y_pred)) if len(y_test) > 0 else 0.0
        force_error = np.std(y_test - y_pred) if len(y_test) > 0 else 0.0
        
        # Calculate custom metrics for physics simulations
        custom_metrics = {}
        if len(y_test) > 0:
            calculated_value = float(y_test[0]) if len(y_test) == 1 else float(np.mean(y_test))
            expected_value = float(y_pred[0]) if len(y_pred) == 1 else float(np.mean(y_pred))
            
            absolute_error = abs(calculated_value - expected_value)
            relative_error = absolute_error / abs(expected_value) if expected_value != 0 else 0.0
            
            custom_metrics = {
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'expected_value': expected_value,
                'calculated_value': calculated_value
            }
        
        # Calculate accuracy for physics simulations (inverse of error)
        accuracy = max(0.0, 1.0 - energy_error) if energy_error <= 1.0 else 0.0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            physics_energy_error=energy_error,
            physics_force_error=force_error,
            mse=mean_squared_error(y_test, y_pred) if len(y_test) > 0 else 0.0,
            mae=mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0.0,
            custom_metrics=custom_metrics
        )
    
    def get_accuracy_summary(self, results: Optional[List[EvaluationResult]] = None) -> Dict[str, Any]:
        """
        Get a summary of accuracy evaluations performed.
        
        Args:
            results: Optional list of results to summarize. If None, uses internal history.
            
        Returns:
            Dictionary containing summary statistics
        """
        if results is None:
            results = self.evaluation_history
            
        if not results:
            return {
                "error": "No evaluation results provided"
            }
        
        # Count evaluation types
        type_counts = {}
        accuracies = []
        total_time = 0.0
        detailed_results = []
        
        for result in results:
            eval_type = result.evaluation_type.value
            type_counts[eval_type] = type_counts.get(eval_type, 0) + 1
            total_time += result.evaluation_time
            
            # Extract accuracy from metrics
            accuracy = None
            if result.metrics.accuracy is not None:
                accuracy = result.metrics.accuracy
            elif result.metrics.r2_score is not None:
                accuracy = result.metrics.r2_score
            elif result.metrics.physics_energy_error is not None:
                # For physics simulations, use inverse of error as accuracy
                accuracy = max(0.0, 1.0 - result.metrics.physics_energy_error)
            
            if accuracy is not None:
                accuracies.append(accuracy)
            
            # Add to detailed results
            detailed_results.append({
                "type": eval_type,
                "time": result.evaluation_time,
                "accuracy": accuracy,
                "metrics": {
                    k: v for k, v in result.metrics.__dict__.items() 
                    if v is not None
                }
            })
        
        # Calculate accuracy statistics
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            accuracy_std = np.std(accuracies)
            best_accuracy = np.max(accuracies)
            worst_accuracy = np.min(accuracies)
        else:
            avg_accuracy = 0.0
            accuracy_std = 0.0
            best_accuracy = 0.0
            worst_accuracy = 0.0
        
        return {
            "total_evaluations": len(results),
            "evaluation_types": type_counts,
            "average_accuracy": avg_accuracy,
            "accuracy_std": accuracy_std,
            "best_accuracy": best_accuracy,
            "worst_accuracy": worst_accuracy,
            "total_evaluation_time": total_time,
            "detailed_results": detailed_results
        }
    
    def clear_history(self):
        """Clear the evaluation history."""
        self.evaluation_history.clear()
    
    def get_evaluation_history(self) -> List[EvaluationResult]:
        """Get the complete evaluation history."""
        return self.evaluation_history.copy()


# Convenience functions for common evaluation tasks
def evaluate_classification_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> EvaluationResult:
    """Convenience function for classification evaluation."""
    evaluator = AccuracyEvaluator()
    return evaluator.evaluate_model(
        model=model,
        test_data=(X_test, y_test),
        evaluation_type=EvaluationType.CLASSIFICATION
    )


def evaluate_regression_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> EvaluationResult:
    """Convenience function for regression evaluation."""
    evaluator = AccuracyEvaluator()
    return evaluator.evaluate_model(
        model=model,
        test_data=(X_test, y_test),
        evaluation_type=EvaluationType.REGRESSION
    )


def evaluate_physics_simulation(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs
) -> EvaluationResult:
    """Convenience function for physics simulation evaluation."""
    evaluator = AccuracyEvaluator()
    return evaluator.evaluate_model(
        model=model,
        test_data=(X_test, y_test),
        evaluation_type=EvaluationType.PHYSICS_SIMULATION,
        **kwargs
    ) 