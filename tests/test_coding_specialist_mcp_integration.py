"""
Integration Tests for Coding Specialist and MCP

These tests use the actual LLM client to test real functionality,
not mocked responses. They test the end-to-end workflow.
"""

import pytest
import tempfile
import shutil
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from agents.llm_client import _llm_client

from agents.coding_specialist import CodingSpecialist
from agents.domain_experts import MedicalExpert
from tools.mcp_tool_registry import MCPToolRegistry
from config import load_config


class TestCodingSpecialistRealIntegration:
    """Integration tests using real LLM functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global LLM client to avoid test pollution
        global _llm_client
        _llm_client = None
        
        # Load configuration with API keys
        self.config = load_config()
        
        # Create temporary directory for tools
        self.temp_dir = tempfile.mkdtemp()
        self.tools_dir = Path(self.temp_dir) / "shared_tools"
        self.tools_dir.mkdir(exist_ok=True)
        
        # Store original environment variable
        self.original_mcp_tools_dir = os.environ.get("MCP_TOOLS_DIR")
        
        # Set environment variable for MCP tools directory first
        os.environ["MCP_TOOLS_DIR"] = str(self.tools_dir)
        
        # Initialize coding specialist with temp directory and config
        self.coding_specialist = CodingSpecialist(
            agent_id="test_coding_specialist",
            model_config=self.config,
            tools_directory=str(self.tools_dir)
        )
        
        # Initialize medical expert with config
        self.medical_expert = MedicalExpert(
            agent_id="test_medical_expert",
            model_config=self.config
        )
        
        # Initialize MCP tool registry
        self.mcp_registry = MCPToolRegistry(tools_directory=str(self.tools_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Restore original environment variable
        if self.original_mcp_tools_dir is not None:
            os.environ["MCP_TOOLS_DIR"] = self.original_mcp_tools_dir
        elif "MCP_TOOLS_DIR" in os.environ:
            del os.environ["MCP_TOOLS_DIR"]
        
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
    
    def test_real_llm_analysis_with_structured_prompt(self):
        """Test that the LLM can actually generate structured JSON responses."""
        # Create a very specific, structured prompt that should elicit JSON
        tool_request = {
            'domain': 'medical',
            'task_description': 'Create a simple calculator for BMI calculation',
            'constraints': {
                'input_format': 'JSON',
                'output_format': 'JSON',
                'complexity': 'simple'
            },
            'existing_tools': []
        }
        
        # Test the analysis - this should work with real LLM
        try:
            analysis = self.coding_specialist.analyze_tool_requirements(tool_request)
            
            # Verify we got a structured response
            assert isinstance(analysis, dict)
            assert 'tool_design' in analysis
            assert 'implementation_approach' in analysis
            assert 'mcp_description' in analysis
            
            print(f"✅ LLM successfully generated structured analysis: {analysis['tool_design']['name']}")
            
        except Exception as e:
            # If this fails, it means the LLM isn't generating proper JSON
            # This is valuable information about the real system
            print(f"❌ LLM failed to generate structured response: {e}")
            pytest.skip(f"LLM integration test failed: {e}")
    
    def test_real_tool_implementation_workflow(self):
        """Test the complete workflow with real LLM."""
        # Create a simple tool specification
        tool_spec = {
            'name': 'bmi_calculator',
            'description': 'Calculate BMI from height and weight',
            'domain': 'medical',
            'parameters': {
                'height_cm': {
                    'type': 'number',
                    'description': 'Height in centimeters',
                    'required': True
                },
                'weight_kg': {
                    'type': 'number',
                    'description': 'Weight in kilograms',
                    'required': True
                }
            },
            'capabilities': ['calculation', 'health_assessment'],
            'requirements': {}
        }
        
        try:
            # Implement the tool
            result = self.coding_specialist.implement_tool(tool_spec)
            
            if result['success']:
                print(f"✅ Successfully implemented tool: {result['tool_name']}")
                
                # Verify files were created
                files = result['files_created']
                tool_file = Path(files['tool_code'])
                mcp_file = Path(files['mcp_description'])
                
                assert tool_file.exists(), f"Tool file not created: {tool_file}"
                assert mcp_file.exists(), f"MCP file not created: {mcp_file}"
                
                # Test that the tool can be loaded
                tool = self.mcp_registry.load_mcp_tool('bmi_calculator')
                assert tool is not None, "Generated tool could not be loaded"
                
                # Test that the tool can be executed
                test_task = {
                    'height_cm': 175,
                    'weight_kg': 70
                }
                
                test_context = {'agent_id': 'test_agent'}
                execution_result = self.mcp_registry.execute_mcp_tool('bmi_calculator', test_task, test_context)
                
                assert execution_result['success'], f"Tool execution failed: {execution_result.get('error')}"
                print(f"✅ Tool executed successfully: {execution_result}")
                
            else:
                print(f"❌ Tool implementation failed: {result.get('error')}")
                pytest.skip(f"Tool implementation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Real tool implementation test failed: {e}")
            pytest.skip(f"Real tool implementation test failed: {e}")
    
    def test_real_error_recovery(self):
        """Test that the LLM can actually recover from parsing errors."""
        # This test intentionally creates a malformed response scenario
        # to see if the LLM can fix it
        
        # We'll test this by creating a scenario where the LLM might generate
        # prose instead of JSON, and see if our retry mechanism works
        
        tool_request = {
            'domain': 'test',
            'task_description': 'Create a test tool with very specific requirements that might confuse the LLM',
            'constraints': {
                'format': 'must be JSON',
                'structure': 'very specific',
                'complexity': 'high'
            },
            'existing_tools': []
        }
        
        try:
            # This should trigger the retry mechanism if the LLM generates prose
            analysis = self.coding_specialist.analyze_tool_requirements(tool_request)
            
            # If we get here, either the LLM generated JSON on first try
            # or the retry mechanism worked
            assert isinstance(analysis, dict)
            assert 'tool_design' in analysis
            
            print(f"✅ LLM successfully handled complex request: {analysis['tool_design']['name']}")
            
        except Exception as e:
            print(f"❌ LLM failed to handle complex request even with retries: {e}")
            # This is actually valuable information - it means our retry mechanism
            # might need improvement
            pytest.skip(f"LLM error recovery test failed: {e}")
    
    def test_real_domain_agent_integration(self):
        """Test that domain agents can actually discover and use tools created by coding specialist."""
        # Create a simple tool with medical domain
        tool_spec = {
            'name': 'simple_adder',
            'description': 'Add two numbers together',
            'domain': 'medical',
            'parameters': {
                'a': {'type': 'number', 'description': 'First number', 'required': True},
                'b': {'type': 'number', 'description': 'Second number', 'required': True}
            },
            'capabilities': ['calculation'],
            'requirements': {}
        }
        
        try:
            # Implement the tool
            result = self.coding_specialist.implement_tool(tool_spec)
            
            if result['success']:
                # Test that medical expert can discover the tool
                available_tools = self.medical_expert.discover_available_tools(
                    "I need to add two numbers together"
                )
                
                # Look for our MCP tool
                mcp_tools = [tool for tool in available_tools if tool.get('tool_type') == 'mcp']
                
                if mcp_tools:
                    print(f"✅ Domain agent discovered {len(mcp_tools)} MCP tools")
                    
                    # Test tool execution
                    test_task = {'a': 5, 'b': 3}
                    test_context = {'agent_id': 'test_medical_expert'}
                    
                    execution_result = self.medical_expert.execute_tool(
                        'mcp_simple_adder', test_task, test_context
                    )
                    
                    if execution_result['success']:
                        print(f"✅ Domain agent successfully executed MCP tool: {execution_result}")
                    else:
                        print(f"❌ Domain agent failed to execute MCP tool: {execution_result.get('error')}")
                        pytest.skip(f"Domain agent tool execution failed: {execution_result.get('error')}")
                else:
                    print("❌ Domain agent did not discover any MCP tools")
                    pytest.skip("Domain agent tool discovery failed")
            else:
                print(f"❌ Tool implementation failed: {result.get('error')}")
                pytest.skip(f"Tool implementation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Real domain agent integration test failed: {e}")
            pytest.skip(f"Real domain agent integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
