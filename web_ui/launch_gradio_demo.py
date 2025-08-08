#!/usr/bin/env python3
"""
AI Research Lab Gradio Demo Launcher

Comprehensive launcher for the AI Research Lab Gradio interface and demo system.
Provides options for running different types of demos and evaluations.
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioDemoLauncher:
    """Launcher for the Gradio demo system."""
    
    def __init__(self):
        self.gradio_process = None
        self.demo_process = None
        self.results_dir = Path("demo_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def start_gradio_interface(self):
        """Start the Gradio interface."""
        try:
            logger.info("Starting Gradio interface...")
            
            # Change to web_ui directory
            os.chdir("web_ui")
            
            # Start Gradio app
            self.gradio_process = subprocess.Popen(
                [sys.executable, "gradio_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Gradio interface started with PID: {self.gradio_process.pid}")
            logger.info("Interface available at: http://localhost:7860")
            
            # Wait for interface to start
            time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Gradio interface: {e}")
            return False
    
    def run_demo_system(self):
        """Run the demo system."""
        try:
            logger.info("Running demo system...")
            
            # Import and run demo system
            from demo_system import DemoSystem
            
            demo_system = DemoSystem()
            evaluation_result = demo_system.run_comprehensive_evaluation()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"demo_evaluation_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(evaluation_result, f, indent=2)
            
            logger.info(f"Demo evaluation completed. Results saved to: {results_file}")
            
            # Print summary
            print("\n" + "="*50)
            print("📊 DEMO EVALUATION RESULTS")
            print("="*50)
            print(f"Total evaluation time: {evaluation_result['total_evaluation_time']:.2f} seconds")
            print(f"Success rate: {evaluation_result['overall_metrics']['success_rate']:.2%}")
            print(f"Successful demos: {evaluation_result['overall_metrics']['successful_demos']}/3")
            print(f"Average execution time: {evaluation_result['overall_metrics']['average_execution_time']:.2f} seconds")
            
            print("\n💡 Recommendations:")
            for i, rec in enumerate(evaluation_result['recommendations'], 1):
                print(f"  {i}. {rec}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Demo system failed: {e}")
            return None
    
    async def run_playwright_demo(self):
        """Run the Playwright demo."""
        try:
            logger.info("Running Playwright demo...")
            
            # Import and run Playwright demo
            from playwright_demo import PlaywrightDemo
            
            playwright_demo = PlaywrightDemo()
            evaluation_result = await playwright_demo.run_comprehensive_demo_evaluation()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"playwright_evaluation_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(evaluation_result, f, indent=2)
            
            logger.info(f"Playwright evaluation completed. Results saved to: {results_file}")
            
            # Print summary
            print("\n" + "="*50)
            print("🎭 PLAYWRIGHT EVALUATION RESULTS")
            print("="*50)
            print(f"Total evaluation time: {evaluation_result['total_evaluation_time']:.2f} seconds")
            print(f"Success rate: {evaluation_result['overall_metrics']['success_rate']:.2%}")
            print(f"Total screenshots: {evaluation_result['overall_metrics']['total_screenshots']}")
            print(f"Total steps: {evaluation_result['overall_metrics']['total_steps']}")
            
            print("\n💡 Recommendations:")
            for i, rec in enumerate(evaluation_result['recommendations'], 1):
                print(f"  {i}. {rec}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Playwright demo failed: {e}")
            return None
    
    def run_comprehensive_evaluation(self):
        """Run a comprehensive evaluation including both demos."""
        logger.info("Starting comprehensive evaluation...")
        
        # Start Gradio interface
        if not self.start_gradio_interface():
            logger.error("Failed to start Gradio interface")
            return None
        
        # Wait for interface to be ready
        time.sleep(10)
        
        # Run demo system
        demo_results = self.run_demo_system()
        
        # Run Playwright demo
        playwright_results = asyncio.run(self.run_playwright_demo())
        
        # Combine results
        comprehensive_results = {
            "evaluation_type": "comprehensive",
            "start_time": datetime.now().isoformat(),
            "demo_system_results": demo_results,
            "playwright_results": playwright_results,
            "overall_metrics": {
                "demo_success_rate": demo_results['overall_metrics']['success_rate'] if demo_results else 0.0,
                "playwright_success_rate": playwright_results['overall_metrics']['success_rate'] if playwright_results else 0.0,
                "combined_success_rate": 0.0
            }
        }
        
        if demo_results and playwright_results:
            comprehensive_results['overall_metrics']['combined_success_rate'] = (
                demo_results['overall_metrics']['success_rate'] + 
                playwright_results['overall_metrics']['success_rate']
            ) / 2
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_evaluation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to: {results_file}")
        
        return comprehensive_results
    
    def stop_services(self):
        """Stop all running services."""
        if self.gradio_process:
            logger.info("Stopping Gradio interface...")
            self.gradio_process.terminate()
            self.gradio_process.wait()
        
        if self.demo_process:
            logger.info("Stopping demo process...")
            self.demo_process.terminate()
            self.demo_process.wait()
    
    def print_menu(self):
        """Print the main menu."""
        print("\n" + "="*60)
        print("🤖 AI RESEARCH LAB GRADIO DEMO LAUNCHER")
        print("="*60)
        print("1. Start Gradio Interface Only")
        print("2. Run Demo System Only")
        print("3. Run Playwright Demo Only")
        print("4. Run Comprehensive Evaluation")
        print("5. Exit")
        print("="*60)

def main():
    """Main function."""
    launcher = GradioDemoLauncher()
    
    try:
        while True:
            launcher.print_menu()
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Gradio interface...")
                if launcher.start_gradio_interface():
                    print("✅ Gradio interface started successfully!")
                    print("🌐 Access the interface at: http://localhost:7860")
                    input("\nPress Enter to stop the interface...")
                    launcher.stop_services()
                else:
                    print("❌ Failed to start Gradio interface")
            
            elif choice == "2":
                print("\n🔬 Running demo system...")
                results = launcher.run_demo_system()
                if results:
                    print("✅ Demo system completed successfully!")
                else:
                    print("❌ Demo system failed")
            
            elif choice == "3":
                print("\n🎭 Running Playwright demo...")
                results = asyncio.run(launcher.run_playwright_demo())
                if results:
                    print("✅ Playwright demo completed successfully!")
                else:
                    print("❌ Playwright demo failed")
            
            elif choice == "4":
                print("\n📊 Running comprehensive evaluation...")
                results = launcher.run_comprehensive_evaluation()
                if results:
                    print("✅ Comprehensive evaluation completed successfully!")
                else:
                    print("❌ Comprehensive evaluation failed")
            
            elif choice == "5":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter a number between 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        launcher.stop_services()

if __name__ == "__main__":
    main() 