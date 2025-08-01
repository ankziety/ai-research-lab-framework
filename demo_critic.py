#!/usr/bin/env python3
"""
Demo script showing the Critic module functionality.

This script demonstrates how to use the Critic class to review
research outputs and get structured feedback.
"""

from critic import Critic


def demo_critic():
    """Demonstrate the Critic module with various examples."""
    critic = Critic()
    
    print("=" * 60)
    print("CRITIC MODULE DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Good research content
    print("\n📊 EXAMPLE 1: Well-structured research content")
    print("-" * 50)
    
    good_research = """
    Introduction
    
    This study investigates the effectiveness of deep learning approaches in natural language processing tasks. Previous research has shown promising results, but comparative analysis remains limited [1, 2].
    
    Methodology
    
    We employed a systematic approach using three machine learning algorithms: Support Vector Machines, Random Forest, and Convolutional Neural Networks. Each method was evaluated using 10-fold cross-validation on a dataset of 15,000 text samples. Statistical significance was assessed using paired t-tests with α = 0.05.
    
    Results
    
    The CNN approach achieved 94.2% accuracy (σ = 1.3%), significantly outperforming Random Forest (87.8%, σ = 2.1%) and SVM (83.4%, σ = 1.9%). These results demonstrate substantial improvements over baseline methods reported in literature [3, 4].
    
    Discussion
    
    Our findings suggest that deep learning architectures provide significant advantages for text classification tasks. The performance gains justify the increased computational requirements, particularly for large-scale applications.
    
    Conclusion
    
    This research contributes empirical evidence supporting the adoption of CNN-based approaches for NLP tasks and provides a foundation for future comparative studies in the field.
    """
    
    result = critic.review(good_research)
    print_critique(result)
    
    # Example 2: Poor research content
    print("\n❌ EXAMPLE 2: Poor quality content")
    print("-" * 50)
    
    poor_research = """
    AI is really good. It helps with many things. AI can solve problems. AI is the future. Everyone should use AI. AI makes work easier. AI is important for business. AI will change everything. AI helps people. AI is beneficial.
    """
    
    result = critic.review(poor_research)
    print_critique(result)
    
    # Example 3: Medium quality content
    print("\n⚠️ EXAMPLE 3: Medium quality content")
    print("-" * 50)
    
    medium_research = """
    This research examines machine learning applications in healthcare. We used data from 500 patients to train predictive models. The analysis shows that machine learning can improve diagnostic accuracy by 15% compared to traditional methods. However, more research is needed to validate these findings across different medical conditions.
    
    Our methodology involved collecting patient data and applying various algorithms including decision trees and neural networks. The results indicate promising potential for clinical implementation.
    """
    
    result = critic.review(medium_research)
    print_critique(result)
    
    print("\n" + "=" * 60)
    print("END OF DEMONSTRATION")
    print("=" * 60)


def print_critique(result):
    """Print a formatted critique result."""
    print(f"📈 Overall Score: {result['overall_score']}/100")
    
    if result['strengths']:
        print("\n✅ STRENGTHS:")
        for strength in result['strengths']:
            print(f"   • {strength}")
    
    if result['weaknesses']:
        print("\n❌ WEAKNESSES:")
        for weakness in result['weaknesses']:
            print(f"   • {weakness}")
    
    if result['suggestions']:
        print("\n💡 SUGGESTIONS:")
        for suggestion in result['suggestions']:
            print(f"   • {suggestion}")
    
    print()


if __name__ == "__main__":
    demo_critic()