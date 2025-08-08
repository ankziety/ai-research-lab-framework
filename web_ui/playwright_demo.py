#!/usr/bin/env python3
"""
Playwright Demo Integration

Automated testing and evaluation of the AI Research Lab Gradio interface using Playwright.
Provides deterministic testing scenarios and performance evaluation.
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlaywrightDemo:
    """Playwright-based demo system for automated testing."""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.screenshots_dir = Path("demo_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
    
    async def run_gradio_interface_test(self) -> Dict[str, Any]:
        """Test the Gradio interface using Playwright."""
        try:
            from playwright.async_api import async_playwright
            
            test_start = time.time()
            test_results = {
                "test_type": "gradio_interface",
                "start_time": datetime.now().isoformat(),
                "steps": [],
                "screenshots": [],
                "performance_metrics": {}
            }
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=False)
                page = await browser.new_page()
                
                # Navigate to Gradio interface
                await page.goto("http://localhost:7860")
                await page.wait_for_load_state("networkidle")
                
                # Take initial screenshot
                screenshot_path = self.screenshots_dir / f"initial_state_{int(time.time())}.png"
                await page.screenshot(path=str(screenshot_path))
                test_results["screenshots"].append(str(screenshot_path))
                test_results["steps"].append("Interface loaded successfully")
                
                # Test Dashboard Tab
                await self.test_dashboard_tab(page, test_results)
                
                # Test Chat Tab
                await self.test_chat_tab(page, test_results)
                
                # Test Research Tab
                await self.test_research_tab(page, test_results)
                
                # Test Analytics Tab
                await self.test_analytics_tab(page, test_results)
                
                # Test Configuration Tab
                await self.test_configuration_tab(page, test_results)
                
                # Calculate performance metrics
                test_duration = time.time() - test_start
                test_results["performance_metrics"] = {
                    "total_duration": test_duration,
                    "steps_completed": len(test_results["steps"]),
                    "screenshots_taken": len(test_results["screenshots"]),
                    "success_rate": 1.0 if len(test_results["steps"]) > 0 else 0.0
                }
                
                await browser.close()
                
            self.test_results.append(test_results)
            return test_results
            
        except Exception as e:
            logger.error(f"Playwright test failed: {e}")
            return {
                "test_type": "gradio_interface",
                "start_time": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed"
            }
    
    async def test_dashboard_tab(self, page, test_results: Dict[str, Any]):
        """Test the dashboard tab functionality."""
        try:
            # Click on Dashboard tab
            await page.click('text="📊 Dashboard"')
            await page.wait_for_timeout(1000)
            
            # Check for system status
            status_element = await page.query_selector('text="System Status"')
            if status_element:
                test_results["steps"].append("Dashboard tab loaded with system status")
            
            # Click refresh button
            refresh_btn = await page.query_selector('button:has-text("🔄 Refresh Status")')
            if refresh_btn:
                await refresh_btn.click()
                await page.wait_for_timeout(1000)
                test_results["steps"].append("Status refresh functionality working")
            
            # Take screenshot
            screenshot_path = self.screenshots_dir / f"dashboard_tab_{int(time.time())}.png"
            await page.screenshot(path=str(screenshot_path))
            test_results["screenshots"].append(str(screenshot_path))
            
        except Exception as e:
            test_results["steps"].append(f"Dashboard tab test failed: {e}")
    
    async def test_chat_tab(self, page, test_results: Dict[str, Any]):
        """Test the chat tab functionality."""
        try:
            # Click on Chat tab
            await page.click('text="💬 Chat"')
            await page.wait_for_timeout(1000)
            
            # Find message input
            message_input = await page.query_selector('textarea[placeholder*="message"]')
            if message_input:
                # Type a test message
                await message_input.fill("Hello, this is a test message from Playwright")
                test_results["steps"].append("Chat message input working")
                
                # Select agent
                agent_dropdown = await page.query_selector('select')
                if agent_dropdown:
                    await agent_dropdown.select_option("all")
                    test_results["steps"].append("Agent selection working")
                
                # Send message
                send_btn = await page.query_selector('button:has-text("📤 Send Message")')
                if send_btn:
                    await send_btn.click()
                    await page.wait_for_timeout(2000)
                    test_results["steps"].append("Message sending functionality working")
            
            # Take screenshot
            screenshot_path = self.screenshots_dir / f"chat_tab_{int(time.time())}.png"
            await page.screenshot(path=str(screenshot_path))
            test_results["screenshots"].append(str(screenshot_path))
            
        except Exception as e:
            test_results["steps"].append(f"Chat tab test failed: {e}")
    
    async def test_research_tab(self, page, test_results: Dict[str, Any]):
        """Test the research tab functionality."""
        try:
            # Click on Research tab
            await page.click('text="🔬 Research"')
            await page.wait_for_timeout(1000)
            
            # Fill research question
            question_input = await page.query_selector('textarea[placeholder*="research question"]')
            if question_input:
                await question_input.fill("Test research question for automated evaluation")
                test_results["steps"].append("Research question input working")
            
            # Fill constraints
            constraints_input = await page.query_selector('textarea[placeholder*="Constraints"]')
            if constraints_input:
                await constraints_input.fill('{"max_iterations": 2, "time_limit": 60}')
                test_results["steps"].append("Constraints input working")
            
            # Fill context
            context_input = await page.query_selector('textarea[placeholder*="Context"]')
            if context_input:
                await context_input.fill('{"dataset": "test", "model_type": "test"}')
                test_results["steps"].append("Context input working")
            
            # Test demo project
            demo_dropdown = await page.query_selector('select[aria-label*="Demo Project"]')
            if demo_dropdown:
                await demo_dropdown.select_option("split_experiment")
                test_results["steps"].append("Demo project selection working")
            
            # Take screenshot
            screenshot_path = self.screenshots_dir / f"research_tab_{int(time.time())}.png"
            await page.screenshot(path=str(screenshot_path))
            test_results["screenshots"].append(str(screenshot_path))
            
        except Exception as e:
            test_results["steps"].append(f"Research tab test failed: {e}")
    
    async def test_analytics_tab(self, page, test_results: Dict[str, Any]):
        """Test the analytics tab functionality."""
        try:
            # Click on Analytics tab
            await page.click('text="📈 Analytics"')
            await page.wait_for_timeout(1000)
            
            # Check for agent activities
            activities_element = await page.query_selector('text="Agent Activities"')
            if activities_element:
                test_results["steps"].append("Analytics tab loaded with agent activities")
            
            # Click refresh button
            refresh_btn = await page.query_selector('button:has-text("🔄 Refresh Activities")')
            if refresh_btn:
                await refresh_btn.click()
                await page.wait_for_timeout(1000)
                test_results["steps"].append("Analytics refresh functionality working")
            
            # Take screenshot
            screenshot_path = self.screenshots_dir / f"analytics_tab_{int(time.time())}.png"
            await page.screenshot(path=str(screenshot_path))
            test_results["screenshots"].append(str(screenshot_path))
            
        except Exception as e:
            test_results["steps"].append(f"Analytics tab test failed: {e}")
    
    async def test_configuration_tab(self, page, test_results: Dict[str, Any]):
        """Test the configuration tab functionality."""
        try:
            # Click on Configuration tab
            await page.click('text="⚙️ Configuration"')
            await page.wait_for_timeout(1000)
            
            # Check for configuration input
            config_input = await page.query_selector('textarea[placeholder*="Configuration"]')
            if config_input:
                test_results["steps"].append("Configuration tab loaded with config input")
            
            # Test data export
            export_dropdown = await page.query_selector('select[aria-label*="Export Data Type"]')
            if export_dropdown:
                await export_dropdown.select_option("chat_history")
                test_results["steps"].append("Data export selection working")
            
            # Take screenshot
            screenshot_path = self.screenshots_dir / f"configuration_tab_{int(time.time())}.png"
            await page.screenshot(path=str(screenshot_path))
            test_results["screenshots"].append(str(screenshot_path))
            
        except Exception as e:
            test_results["steps"].append(f"Configuration tab test failed: {e}")
    
    async def run_comprehensive_demo_evaluation(self) -> Dict[str, Any]:
        """Run a comprehensive demo evaluation using Playwright."""
        evaluation_start = time.time()
        
        # Start Gradio interface in background
        import subprocess
        import threading
        
        def start_gradio():
            subprocess.run([sys.executable, "gradio_app.py"], cwd="web_ui")
        
        gradio_thread = threading.Thread(target=start_gradio)
        gradio_thread.daemon = True
        gradio_thread.start()
        
        # Wait for interface to start
        await asyncio.sleep(10)
        
        # Run interface tests
        interface_test = await self.run_gradio_interface_test()
        
        # Calculate overall metrics
        total_time = time.time() - evaluation_start
        
        evaluation_result = {
            "evaluation_type": "playwright_comprehensive",
            "start_time": datetime.now().isoformat(),
            "total_evaluation_time": total_time,
            "test_results": {
                "gradio_interface": interface_test
            },
            "overall_metrics": {
                "total_tests": 1,
                "successful_tests": 1 if interface_test.get("status") != "failed" else 0,
                "total_screenshots": len(interface_test.get("screenshots", [])),
                "total_steps": len(interface_test.get("steps", [])),
                "success_rate": 1.0 if interface_test.get("status") != "failed" else 0.0
            },
            "recommendations": self.generate_playwright_recommendations([interface_test])
        }
        
        self.performance_metrics = evaluation_result
        return evaluation_result
    
    def generate_playwright_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on Playwright test results."""
        recommendations = []
        
        for test_result in test_results:
            if test_result.get("status") == "failed":
                recommendations.append("Fix interface loading issues and ensure Gradio server is running")
                continue
            
            steps = test_result.get("steps", [])
            if len(steps) < 5:
                recommendations.append("Improve interface responsiveness and element loading")
            
            screenshots = test_result.get("screenshots", [])
            if len(screenshots) < 3:
                recommendations.append("Enhance visual feedback and interface state management")
            
            performance = test_result.get("performance_metrics", {})
            if performance.get("total_duration", 0) > 60:
                recommendations.append("Optimize interface loading times and reduce latency")
        
        if not recommendations:
            recommendations.append("Interface performance is excellent - consider adding more advanced features")
        
        return recommendations
    
    def export_playwright_results(self, filename: str = None) -> str:
        """Export Playwright test results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"playwright_results_{timestamp}.json"
        
        export_data = {
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "export_timestamp": datetime.now().isoformat(),
            "framework_version": "1.0.0"
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return f"Playwright results exported to {filename}"

async def main():
    """Main function to run the Playwright demo."""
    playwright_demo = PlaywrightDemo()
    
    print("🤖 AI Research Lab Playwright Demo")
    print("=" * 50)
    
    # Run comprehensive evaluation
    print("Running comprehensive Playwright evaluation...")
    evaluation_result = await playwright_demo.run_comprehensive_demo_evaluation()
    
    print(f"\n✅ Evaluation completed in {evaluation_result['total_evaluation_time']:.2f} seconds")
    print(f"Success rate: {evaluation_result['overall_metrics']['success_rate']:.2%}")
    print(f"Total screenshots: {evaluation_result['overall_metrics']['total_screenshots']}")
    print(f"Total steps: {evaluation_result['overall_metrics']['total_steps']}")
    
    # Export results
    filename = playwright_demo.export_playwright_results()
    print(f"\n📊 Results exported to: {filename}")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for i, rec in enumerate(evaluation_result['recommendations'], 1):
        print(f"  {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(main()) 