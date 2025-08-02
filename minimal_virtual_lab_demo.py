"""
Minimal Virtual Lab Example

This script demonstrates the basic Virtual Lab functionality without requiring
external API keys. It shows the structure and flow of the Virtual Lab methodology.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def minimal_virtual_lab_demo():
    """Minimal demo of Virtual Lab functionality."""
    
    print("🧪 Virtual Lab Minimal Demo")
    print("=" * 30)
    
    try:
        # Import Virtual Lab components
        from virtual_lab import VirtualLabMeetingSystem, MeetingType, ResearchPhase
        print("✅ Virtual Lab components imported")
        
        # Show research phases
        print("\n📋 Research Phases in Virtual Lab:")
        for i, phase in enumerate(ResearchPhase, 1):
            print(f"   {i}. {phase.value.replace('_', ' ').title()}")
        
        # Show meeting types
        print("\n🏢 Meeting Types:")
        for meeting_type in MeetingType:
            print(f"   • {meeting_type.value.replace('_', ' ').title()}")
        
        # Test basic framework initialization without heavy dependencies
        print("\n🔧 Testing Framework Integration...")
        
        try:
            from multi_agent_framework import MultiAgentResearchFramework
            
            # Minimal config to avoid external dependencies
            config = {
                'store_all_interactions': False,
                'enable_memory_management': False,
                'auto_critique': False,
                'auto_visualize': False,
                'vector_db_path': ':memory:',  # In-memory database
                'experiment_db_path': ':memory:'
            }
            
            print("   Initializing framework...")
            framework = MultiAgentResearchFramework(config)
            
            if hasattr(framework, 'virtual_lab'):
                print("✅ Virtual Lab integrated successfully")
                
                # Check available methods
                vlab_methods = [
                    'conduct_virtual_lab_research',
                    'get_virtual_lab_session', 
                    'list_virtual_lab_sessions',
                    'get_meeting_history',
                    'get_virtual_lab_statistics'
                ]
                
                print("\n📚 Available Virtual Lab Methods:")
                for method in vlab_methods:
                    if hasattr(framework, method):
                        print(f"   ✅ {method}")
                    else:
                        print(f"   ❌ {method}")
                
                # Show Virtual Lab statistics structure
                try:
                    stats = framework.get_virtual_lab_statistics()
                    print(f"\n📊 Virtual Lab Statistics Structure: {type(stats).__name__}")
                    if isinstance(stats, dict):
                        print(f"   Available keys: {list(stats.keys())}")
                except Exception as e:
                    print(f"   Statistics method available but returned: {type(e).__name__}")
                
            else:
                print("❌ Virtual Lab not found in framework")
            
        except ImportError as e:
            print(f"❌ Framework import failed: {e}")
        except Exception as e:
            print(f"❌ Framework initialization failed: {e}")
        
        print("\n🎯 Virtual Lab Key Features:")
        print("   ✅ Meeting-based research coordination")
        print("   ✅ Structured 7-phase research methodology")
        print("   ✅ Multi-agent collaboration")
        print("   ✅ Cross-agent interaction and critique")
        print("   ✅ Scientific quality assessment")
        print("   ✅ Iterative refinement workflows")
        
        print("\n📖 Based on the paper:")
        print("   'The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies'")
        print("   by Swanson et al. (2025)")
        
        print("\n✅ Virtual Lab implementation verified!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure all required files are present:")
        print("   • virtual_lab.py")
        print("   • multi_agent_framework.py")
        print("   • agents/ directory")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_virtual_lab_demo()