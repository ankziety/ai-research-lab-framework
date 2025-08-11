"""
Test Coding Specialist and MCP Integration

Tests the workflow where domain agents can request tools from the coding specialist
and then use those tools via MCP-compatible interfaces.
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from agents.llm_client import _llm_client

from agents.coding_specialist import CodingSpecialist
from agents.domain_experts import MedicalExpert
from tools.mcp_tool_registry import MCPToolRegistry


class TestCodingSpecialistMCPIntegration:
    """Test the integration between coding specialist and MCP tool registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global LLM client to avoid test pollution
        global _llm_client
        _llm_client = None
        
        # Create temporary directory for tools
        self.temp_dir = tempfile.mkdtemp()
        self.tools_dir = Path(self.temp_dir) / "shared_tools"
        self.tools_dir.mkdir(exist_ok=True)
        
        # Store original environment variable
        self.original_mcp_tools_dir = os.environ.get("MCP_TOOLS_DIR")
        
        # Set environment variable for MCP tools directory
        os.environ["MCP_TOOLS_DIR"] = str(self.tools_dir)
        
        # Initialize coding specialist with temp directory
        self.coding_specialist = CodingSpecialist(
            agent_id="test_coding_specialist",
            tools_directory=str(self.tools_dir)
        )
        
        # Initialize medical expert
        self.medical_expert = MedicalExpert(agent_id="test_medical_expert")
        
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
    
    def test_coding_specialist_analyzes_tool_requirements(self):
        """Test that coding specialist can analyze tool requirements."""
        # Mock the LLM client to return proper JSON response
        with patch.object(self.coding_specialist, 'generate_response') as mock_generate:
            mock_generate.return_value = json.dumps({
                'tool_design': {
                    'name': 'medical_risk_analyzer',
                    'description': 'Analyze patient data for risk factors',
                    'parameters': {
                        'patient_data': {'type': 'string', 'description': 'CSV patient data'},
                        'risk_threshold': {'type': 'number', 'description': 'Risk threshold'}
                    },
                    'return_type': 'dict'
                },
                'implementation_approach': 'Statistical analysis with risk scoring algorithms',
                'mcp_description': {
                    'name': 'medical_risk_analyzer',
                    'description': 'Analyze patient data for risk factors',
                    'inputSchema': {'type': 'object', 'properties': {}},
                    'outputSchema': {'type': 'object', 'properties': {}}
                },
                'integration_plan': 'Register with tool registry and provide MCP interface',
                'testing_strategy': 'Unit tests with medical data validation',
                'documentation': 'Comprehensive documentation with usage examples'
            })
            
            # Medical expert requests a tool
            tool_request = {
                'domain': 'medical',
                'task_description': 'Analyze patient data for risk factors',
                'constraints': {
                    'input_format': 'CSV',
                    'output_format': 'JSON',
                    'performance': 'fast'
                },
                'existing_tools': ['data_visualizer', 'statistical_analyzer']
            }
            
            # Coding specialist analyzes requirements
            analysis = self.coding_specialist.analyze_tool_requirements(tool_request)
            
            # Verify analysis structure
            assert 'tool_design' in analysis
            assert 'implementation_approach' in analysis
            assert 'mcp_description' in analysis
            assert 'integration_plan' in analysis
            assert 'testing_strategy' in analysis
            assert 'documentation' in analysis
            
            # Verify tool design
            tool_design = analysis['tool_design']
            assert 'name' in tool_design
            assert 'description' in tool_design
            assert 'parameters' in tool_design
            assert 'return_type' in tool_design
    
    def test_coding_specialist_handles_parsing_errors(self):
        """Test that coding specialist handles parsing errors with retry mechanism."""
        # Mock the LLM client to return malformed responses
        with patch.object(self.coding_specialist, 'generate_response') as mock_generate:
            # First call returns malformed response, second call returns corrected response
            mock_generate.side_effect = [
                "This is not valid JSON",
                json.dumps({
                    'tool_design': {
                        'name': 'test_tool',
                        'description': 'Test tool',
                        'parameters': {},
                        'return_type': 'dict'
                    },
                    'implementation_approach': 'Test implementation',
                    'mcp_description': {
                        'name': 'test_tool',
                        'description': 'Test tool',
                        'inputSchema': {'type': 'object', 'properties': {}},
                        'outputSchema': {'type': 'object', 'properties': {}}
                    },
                    'integration_plan': 'Test integration',
                    'testing_strategy': 'Test strategy',
                    'documentation': 'Test documentation'
                })
            ]
            
            tool_request = {
                'domain': 'test',
                'task_description': 'Test tool',
                'constraints': {},
                'existing_tools': []
            }
            
            # Should succeed after retry
            analysis = self.coding_specialist.analyze_tool_requirements(tool_request)
            
            # Verify the retry mechanism was used
            assert mock_generate.call_count == 2
            assert 'tool_design' in analysis
            assert analysis['tool_design']['name'] == 'test_tool'
    
    def test_coding_specialist_handles_implementation_errors(self):
        """Test that coding specialist handles implementation errors with retry mechanism."""
        # Mock the LLM client to return corrected specifications
        with patch.object(self.coding_specialist, 'generate_response') as mock_generate:
            # Mock implementation correction
            mock_generate.return_value = json.dumps({
                'name': 'corrected_tool',
                'description': 'Corrected tool',
                'domain': 'test',
                'parameters': {},
                'capabilities': ['testing'],
                'requirements': {}
            })
            
            # Mock file operations to cause an error on first attempt
            with patch('builtins.open', side_effect=[OSError("Permission denied"), None, None, None]):
                # Create a tool specification
                tool_spec = {
                    'name': 'test_tool',
                    'description': 'Test tool',
                    'domain': 'test',
                    'parameters': {
                        'param1': {
                            'type': 'string',
                            'description': 'Test parameter'
                        }
                    },
                    'capabilities': ['testing'],
                    'requirements': {}
                }
                
                # Should succeed after retry with corrected specification
                result = self.coding_specialist.implement_tool(tool_spec, max_retries=1)
                
                # Verify the retry mechanism was used
                assert mock_generate.called
                assert result['success'] is True
    
    def test_coding_specialist_implements_tool(self):
        """Test that coding specialist can implement a tool."""
        # Mock the LLM client to return proper JSON response for implementation correction
        with patch.object(self.coding_specialist, 'generate_response') as mock_generate:
            mock_generate.return_value = json.dumps({
                'name': 'patient_risk_analyzer',
                'description': 'Analyze patient data for risk factors',
                'domain': 'medical',
                'parameters': {
                    'patient_data': {
                        'type': 'string',
                        'description': 'CSV patient data',
                        'required': True
                    },
                    'risk_threshold': {
                        'type': 'number',
                        'description': 'Risk threshold for analysis',
                        'required': False,
                        'default': 0.5
                    }
                },
                'capabilities': ['risk_analysis', 'data_processing', 'medical_assessment'],
                'requirements': {'min_memory': 50}
            })
            
            # Tool specification
            tool_spec = {
                'name': 'patient_risk_analyzer',
                'description': 'Analyze patient data for risk factors',
                'domain': 'medical',
                'parameters': {
                    'patient_data': {
                        'type': 'string',
                        'description': 'CSV patient data',
                        'required': True
                    },
                    'risk_threshold': {
                        'type': 'number',
                        'description': 'Risk threshold for analysis',
                        'required': False,
                        'default': 0.5
                    }
                },
                'capabilities': ['risk_analysis', 'data_processing', 'medical_assessment'],
                'requirements': {'min_memory': 50}
            }
            
            # Implement tool
            result = self.coding_specialist.implement_tool(tool_spec)
            
            # Verify implementation
            assert result['success'] is True
            assert result['tool_name'] == 'patient_risk_analyzer'
            assert 'files_created' in result
            
            files = result['files_created']
            assert 'tool_code' in files
            assert 'mcp_description' in files
            assert 'tests' in files
            
            # Verify files exist
            tool_file = Path(files['tool_code'])
            mcp_file = Path(files['mcp_description'])
            test_file = Path(files['tests'])
            
            assert tool_file.exists()
            assert mcp_file.exists()
            assert test_file.exists()
    
    def test_mcp_tool_discovery(self):
        """Test that MCP tools can be discovered."""
        # First implement a tool
        tool_spec = {
            'name': 'test_tool',
            'description': 'Test tool for discovery',
            'domain': 'medical',
            'parameters': {},
            'capabilities': ['testing'],
            'requirements': {}
        }
        
        self.coding_specialist.implement_tool(tool_spec)
        
        # Discover tools
        discovered_tools = self.mcp_registry.discover_mcp_tools()
        
        # Verify discovery
        assert len(discovered_tools) >= 1
        
        tool_info = discovered_tools[0]
        assert tool_info['name'] == 'test_tool'
        assert tool_info['domain'] == 'medical'
        assert 'mcp_description' in tool_info
        assert 'file_path' in tool_info
        assert 'mcp_path' in tool_info
    
    def test_mcp_tool_loading(self):
        """Test that MCP tools can be loaded."""
        # Implement a tool
        tool_spec = {
            'name': 'loadable_tool',
            'description': 'Tool for testing loading',
            'domain': 'medical',
            'parameters': {
                'input_data': {
                    'type': 'string',
                    'description': 'Input data',
                    'required': True
                }
            },
            'capabilities': ['data_processing'],
            'requirements': {}
        }
        
        self.coding_specialist.implement_tool(tool_spec)
        
        # Load the tool
        tool = self.mcp_registry.load_mcp_tool('loadable_tool')
        
        # Verify tool loaded
        assert tool is not None
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'tool_id')
        assert tool.tool_id == 'loadable_tool'
    
    def test_domain_agent_tool_discovery(self):
        """Test that domain agents can discover MCP tools."""
        # Implement a medical tool
        tool_spec = {
            'name': 'medical_analyzer',
            'description': 'Medical data analysis tool',
            'domain': 'medical',
            'parameters': {
                'patient_data': {
                    'type': 'string',
                    'description': 'Patient data',
                    'required': True
                }
            },
            'capabilities': ['medical_analysis', 'data_processing'],
            'requirements': {}
        }
        
        self.coding_specialist.implement_tool(tool_spec)
        
        # Medical expert discovers tools
        available_tools = self.medical_expert.discover_available_tools(
            "Analyze patient data for medical insights"
        )
        
        # Verify MCP tools are discovered
        mcp_tools = [tool for tool in available_tools if tool.get('tool_type') == 'mcp']
        assert len(mcp_tools) >= 1
        
        medical_tool = mcp_tools[0]
        assert medical_tool['name'] == 'medical_analyzer'
        assert medical_tool['tool_type'] == 'mcp'
        assert 'mcp_description' in medical_tool
        assert medical_tool['confidence'] > 0
    
    def test_domain_agent_tool_execution(self):
        """Test that domain agents can execute MCP tools."""
        # Implement a simple tool
        tool_spec = {
            'name': 'simple_calculator',
            'description': 'Simple calculation tool',
            'domain': 'general',
            'parameters': {
                'operation': {
                    'type': 'string',
                    'description': 'Mathematical operation',
                    'required': True
                },
                'a': {
                    'type': 'number',
                    'description': 'First number',
                    'required': True
                },
                'b': {
                    'type': 'number',
                    'description': 'Second number',
                    'required': True
                }
            },
            'capabilities': ['calculation'],
            'requirements': {}
        }
        
        self.coding_specialist.implement_tool(tool_spec)
        
        # Medical expert executes the tool
        task = {
            'operation': 'add',
            'a': 5,
            'b': 3
        }
        
        context = {
            'agent_id': 'test_medical_expert',
            'agent_role': 'Medical Expert'
        }
        
        result = self.medical_expert.execute_tool('mcp_simple_calculator', task, context)
        
        # Verify execution
        assert result['success'] is True
        assert 'tool_name' in result
        assert result['tool_name'] == 'simple_calculator'
    
    def test_mcp_tool_validation(self):
        """Test MCP tool input/output validation."""
        # Implement tool with specific schema
        tool_spec = {
            'name': 'validated_tool',
            'description': 'Tool with validation',
            'domain': 'medical',
            'parameters': {
                'patient_id': {
                    'type': 'string',
                    'description': 'Patient ID',
                    'required': True
                },
                'age': {
                    'type': 'number',
                    'description': 'Patient age',
                    'required': True
                }
            },
            'output_properties': {
                'risk_score': {
                    'type': 'number',
                    'description': 'Calculated risk score'
                },
                'recommendations': {
                    'type': 'array',
                    'description': 'Medical recommendations'
                }
            },
            'capabilities': ['validation'],
            'requirements': {}
        }
        
        self.coding_specialist.implement_tool(tool_spec)
        
        # Test valid input
        valid_task = {
            'patient_id': 'P12345',
            'age': 45
        }
        
        context = {'agent_id': 'test_agent'}
        
        result = self.mcp_registry.execute_mcp_tool('validated_tool', valid_task, context)
        assert result['success'] is True
        
        # Test invalid input (missing required field)
        invalid_task = {
            'patient_id': 'P12345'
            # Missing 'age' field
        }
        
        result = self.mcp_registry.execute_mcp_tool('validated_tool', invalid_task, context)
        assert result['success'] is False
        assert 'validation' in result['error'].lower() or 'required' in result['error'].lower()
    
    def test_tool_recommendation_system(self):
        """Test tool recommendation system."""
        # Implement multiple tools
        tools = [
            {
                'name': 'medical_analyzer',
                'description': 'Medical data analysis tool',
                'domain': 'medical',
                'parameters': {},
                'capabilities': ['medical_analysis'],
                'requirements': {}
            },
            {
                'name': 'data_visualizer',
                'description': 'Data visualization tool',
                'domain': 'general',
                'parameters': {},
                'capabilities': ['visualization'],
                'requirements': {}
            },
            {
                'name': 'statistical_analyzer',
                'description': 'Statistical analysis tool',
                'domain': 'data_science',
                'parameters': {},
                'capabilities': ['statistical_analysis'],
                'requirements': {}
            }
        ]
        
        for tool_spec in tools:
            self.coding_specialist.implement_tool(tool_spec)
        
        # Get recommendations for medical task
        recommendations = self.mcp_registry.recommend_tools_for_task(
            "Analyze patient medical data for risk assessment",
            domain="medical"
        )
        
        # Verify recommendations
        assert len(recommendations) >= 1
        
        # Medical analyzer should be top recommendation
        top_recommendation = recommendations[0]
        assert top_recommendation['name'] == 'medical_analyzer'
        assert top_recommendation['confidence'] > 0.1
        assert 'relevance_reason' in top_recommendation
    
    def test_registry_statistics(self):
        """Test MCP registry statistics."""
        # Implement a few tools
        for i in range(3):
            tool_spec = {
                'name': f'test_tool_{i}',
                'description': f'Test tool {i}',
                'domain': 'general',
                'parameters': {},
                'capabilities': ['testing'],
                'requirements': {}
            }
            self.coding_specialist.implement_tool(tool_spec)
        
        # Get statistics
        stats = self.mcp_registry.get_registry_stats()
        
        # Verify statistics
        assert stats['total_tools_discovered'] >= 3
        assert stats['total_tools_loaded'] >= 0
        assert 'tools_by_domain' in stats
        assert 'tools_by_capability' in stats
        assert 'registry_health' in stats
        
        # Verify domain breakdown
        domain_stats = stats['tools_by_domain']
        assert 'general' in domain_stats
        assert domain_stats['general'] >= 3


if __name__ == "__main__":
    pytest.main([__file__])
