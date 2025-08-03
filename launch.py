#!/usr/bin/env python3
"""
AI Research Lab Framework Launcher

This script provides multiple ways to launch the AI Research Lab Framework:
1. Web UI - Interactive web interface
2. CLI - Command line interface
3. Python API - Direct Python usage
4. Virtual Lab - Virtual lab meeting system
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def install_requirements():
    """Install required dependencies."""
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "web_ui/requirements.txt"])

def launch_web_ui():
    """Launch the web interface."""
    print("Starting AI Research Lab Web Interface...")
    print("The web interface will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Change to web_ui directory and run Flask app
    os.chdir("web_ui")
    subprocess.run([sys.executable, "app.py"])

def launch_cli():
    """Launch the command line interface."""
    print("Starting AI Research Lab CLI...")
    print("Available commands:")
    print("  python -m data.cli --help")
    print("  python -m data.cli virtual-lab-research --question 'Your research question'")
    print("  python -m data.cli run-experiment --params 'param1=value1' 'param2=value2'")
    print("  python -m data.cli draft-manuscript --results-file results.json")
    
    # Run CLI with help
    subprocess.run([sys.executable, "-m", "data.cli", "--help"])

def run_python_example():
    """Run a Python example."""
    print("Running Python API example...")
    
    example_code = '''
from ai_research_lab import create_framework

# Initialize the framework
framework = create_framework({
    'openai_api_key': 'your-api-key',  # Optional for basic functionality
    'max_agents_per_research': 3,
    'budget_limit': 50.0
})

# Conduct research using Virtual Lab methodology
results = framework.conduct_virtual_lab_research(
    research_question="How can we improve machine learning model interpretability?",
    constraints={'budget': 25.0, 'timeline_weeks': 1},
    context={'domain': 'computer_science', 'priority': 'high'}
)

print(f"Research completed: {results['status']}")
print(f"Key findings: {results.get('key_findings', [])}")
'''
    
    print("Example Python code:")
    print(example_code)
    print("\nTo run this example, create a file with the above code and execute it.")

def show_usage():
    """Show usage information."""
    print("""
AI Research Lab Framework - Usage Options

1. Web Interface (Recommended for beginners):
   python launch.py --web
   
2. Command Line Interface:
   python launch.py --cli
   
3. Python API Example:
   python launch.py --example
   
4. Install Dependencies:
   python launch.py --install
   
5. All-in-one (install + web):
   python launch.py --all

Examples:
   # Start web interface
   python launch.py --web
   
   # Run CLI with research question
   python -m data.cli virtual-lab-research --question "What are the latest developments in AI?"
   
   # Run experiment
   python -m data.cli run-experiment --params "model=transformer" "dataset=imagenet"
   
   # Draft manuscript
   python -m data.cli draft-manuscript --results-file results.json --output manuscript.md
""")

def main():
    parser = argparse.ArgumentParser(description="AI Research Lab Framework Launcher")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--cli", action="store_true", help="Launch command line interface")
    parser.add_argument("--example", action="store_true", help="Show Python API example")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--all", action="store_true", help="Install dependencies and launch web interface")
    
    args = parser.parse_args()
    
    if args.install or args.all:
        install_requirements()
    
    if args.web or args.all:
        launch_web_ui()
    elif args.cli:
        launch_cli()
    elif args.example:
        run_python_example()
    else:
        show_usage()

if __name__ == "__main__":
    main() 