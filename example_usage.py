#!/usr/bin/env python3
"""
Example usage of the results_visualizer module.

This script demonstrates how to use the visualize function with different types
of experiment results.
"""

from results_visualizer import visualize

def main():
    """Demonstrate the results visualizer with sample data."""
    
    # Example 1: Machine learning experiment results
    ml_results = [
        {
            'accuracy': 0.85,
            'loss': 0.15,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'status': 'completed',
            'model': 'resnet50',
            'dataset': 'cifar10',
            'epochs': 100,
            'batch_size': 32
        },
        {
            'accuracy': 0.87,
            'loss': 0.13,
            'precision': 0.84,
            'recall': 0.86,
            'f1_score': 0.85,
            'status': 'completed',
            'model': 'resnet50',
            'dataset': 'cifar10',
            'epochs': 150,
            'batch_size': 64
        },
        {
            'accuracy': 0.89,
            'loss': 0.11,
            'precision': 0.86,
            'recall': 0.90,
            'f1_score': 0.88,
            'status': 'completed',
            'model': 'resnet50',
            'dataset': 'cifar10',
            'epochs': 200,
            'batch_size': 32
        },
        {
            'accuracy': 0.83,
            'loss': 0.17,
            'precision': 0.80,
            'recall': 0.85,
            'f1_score': 0.82,
            'status': 'failed',
            'model': 'vgg16',
            'dataset': 'cifar10',
            'epochs': 100,
            'batch_size': 32
        }
    ]
    
    # Example 2: Scientific experiment results
    science_results = [
        {
            'temperature': 25.5,
            'pressure': 101.3,
            'humidity': 60.2,
            'reaction_time': 45.3,
            'yield_percentage': 78.5,
            'status': 'completed',
            'experiment_type': 'catalysis',
            'catalyst': 'platinum'
        },
        {
            'temperature': 30.0,
            'pressure': 102.1,
            'humidity': 58.7,
            'reaction_time': 38.9,
            'yield_percentage': 82.1,
            'status': 'completed',
            'experiment_type': 'catalysis',
            'catalyst': 'platinum'
        },
        {
            'temperature': 35.2,
            'pressure': 103.5,
            'humidity': 55.3,
            'reaction_time': 32.1,
            'yield_percentage': 85.7,
            'status': 'completed',
            'experiment_type': 'catalysis',
            'catalyst': 'palladium'
        }
    ]
    
    # Example 3: Mixed data with some failed experiments
    mixed_results = [
        {
            'accuracy': 0.75,
            'training_time': 120.5,
            'memory_usage': 2048,
            'status': 'completed',
            'model': 'transformer',
            'dataset': 'text_corpus'
        },
        {
            'accuracy': 0.78,
            'training_time': 145.2,
            'memory_usage': 3072,
            'status': 'completed',
            'model': 'transformer',
            'dataset': 'text_corpus'
        },
        {
            'accuracy': 0.72,
            'training_time': 95.8,
            'memory_usage': 1536,
            'status': 'failed',
            'model': 'lstm',
            'dataset': 'text_corpus'
        }
    ]
    
    # Generate visualizations
    print("Generating visualizations...")
    
    try:
        # Create ML results visualization
        visualize(ml_results, 'ml_experiment_results.png')
        print("✓ ML experiment results saved to 'ml_experiment_results.png'")
        
        # Create science results visualization
        visualize(science_results, 'science_experiment_results.png')
        print("✓ Science experiment results saved to 'science_experiment_results.png'")
        
        # Create mixed results visualization
        visualize(mixed_results, 'mixed_experiment_results.png')
        print("✓ Mixed experiment results saved to 'mixed_experiment_results.png'")
        
        print("\nAll visualizations generated successfully!")
        print("Check the generated PNG files to see the plots.")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    main()