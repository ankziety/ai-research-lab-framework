#!/usr/bin/env python3
"""
Test script to validate OpenAI pricing and cost estimates.
"""

import os
import sys
from typing import Dict, Any

def test_pricing():
    """Test the pricing validation functionality."""
    try:
        # Add the project root to the path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from data.cost_manager import CostManager
        
        # Initialize cost manager
        config = {
            'budget_limit': 30.0,
            'cost_optimization': True
        }
        
        cost_manager = CostManager(30.0, config)
        
        print("=== OpenAI Pricing Validation ===\n")
        
        # Check current pricing
        print("1. Checking current OpenAI pricing...")
        pricing_info = cost_manager.check_openai_pricing()
        
        if 'error' in pricing_info:
            print(f"❌ Error: {pricing_info['error']}")
        else:
            print("✅ Successfully fetched pricing information")
            print(f"   Available models: {len(pricing_info.get('available_models', []))}")
            print(f"   Pricing source: {pricing_info.get('pricing_source', 'Unknown')}")
            
            if 'current_pricing' in pricing_info:
                print("\n   Current pricing (per 1K tokens):")
                for model, costs in pricing_info['current_pricing'].items():
                    print(f"   - {model}: ${costs['input']:.4f} input, ${costs['output']:.4f} output")
        
        # Validate cost estimates
        print("\n2. Validating our cost estimates...")
        validation = cost_manager.validate_cost_estimates()
        
        if 'error' in validation:
            print(f"❌ Error: {validation['error']}")
        else:
            print(f"✅ Validated {len(validation.get('models_checked', []))} models")
            
            if validation.get('discrepancies'):
                print(f"⚠️  Found {len(validation['discrepancies'])} pricing discrepancies:")
                for disc in validation['discrepancies']:
                    print(f"   - {disc['model']}: Input diff ${disc['input_diff']:.4f}, Output diff ${disc['output_diff']:.4f}")
            else:
                print("✅ All pricing estimates are current")
            
            if validation.get('recommendations'):
                print("\n   Recommendations:")
                for rec in validation['recommendations']:
                    print(f"   - {rec}")
        
        # Test model selection
        print("\n3. Testing model selection...")
        test_budgets = [50.0, 10.0, 5.0, 1.0, 0.5]
        
        for budget in test_budgets:
            model = cost_manager.optimize_model_selection(
                task_complexity='medium',
                budget_remaining=budget,
                required_capabilities=['reasoning', 'analysis']
            )
            print(f"   Budget ${budget}: Selected {model}")
        
        # Test cost estimation
        print("\n4. Testing cost estimation...")
        test_models = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4']
        tokens_input = 1000
        tokens_output = 500
        
        for model in test_models:
            if model in cost_manager.model_costs:
                cost = cost_manager.estimate_cost(model, tokens_input, tokens_output)
                print(f"   {model}: ${cost:.4f} for {tokens_input + tokens_output} tokens")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pricing()
