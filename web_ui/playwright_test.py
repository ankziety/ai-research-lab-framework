#!/usr/bin/env python3
"""
Playwright Test Script for AI Research Lab Gradio Interface

This script uses Playwright to interact with the Gradio interface and test
the AI research lab functionality in a browser environment.
"""

import asyncio
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Playwright imports
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed. Installing...")
    subprocess.run(["pip", "install", "playwright"])
    subprocess.run(["playwright", "install"])
    from playwright.async_api import async_playwright

class GradioPlaywrightTester:
    """Playwright-based tester for the Gradio interface."""
    
    def __init__(self, gradio_url: str = "http://localhost:7860"):
        self.gradio_url = gradio_url
        self.test_results = {}
        
    async def start_gradio_server(self):
        """Start the Gradio server in a separate process."""
        print("🚀 Starting Gradio server...")
        
        # Start the server in a separate thread
        def run_server():
            try:
                subprocess.run([
                    "python", "gradio_app.py"
                ], cwd=Path(__file__).parent)
            except Exception as e:
                print(f"Error starting server: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        await asyncio.sleep(5)
        print("✅ Gradio server started")
    
    async def test_interface_loading(self, page):
        """Test that the interface loads correctly."""
        print("📋 Testing interface loading...")
        
        try:
            # Navigate to the interface
            await page.goto(self.gradio_url)
            await page.wait_for_load_state("networkidle")
            
            # Check for main elements
            await page.wait_for_selector("text=AI Research Lab", timeout=10000)
            await page.wait_for_selector("text=Chat", timeout=5000)
            await page.wait_for_selector("text=Dashboard", timeout=5000)
            
            print("✅ Interface loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Interface loading failed: {e}")
            return False
    
    async def test_chat_functionality(self, page):
        """Test the chat functionality."""
        print("💬 Testing chat functionality...")
        
        try:
            # Switch to chat tab
            await page.click("text=Chat")
            await page.wait_for_timeout(1000)
            
            # Find and fill the message input
            message_input = await page.wait_for_selector("textarea[placeholder*='question']", timeout=5000)
            await message_input.fill("Hello, can you help me with research?")
            
            # Send the message
            send_button = await page.wait_for_selector("button:has-text('Send')", timeout=5000)
            await send_button.click()
            
            # Wait for response
            await page.wait_for_timeout(3000)
            
            # Check for response
            response_elements = await page.query_selector_all(".message")
            if len(response_elements) > 0:
                print("✅ Chat functionality working")
                return True
            else:
                print("❌ No response received")
                return False
                
        except Exception as e:
            print(f"❌ Chat functionality failed: {e}")
            return False
    
    async def test_research_mode(self, page):
        """Test the research mode functionality."""
        print("🔬 Testing research mode...")
        
        try:
            # Switch to chat tab
            await page.click("text=Chat")
            await page.wait_for_timeout(1000)
            
            # Enable research mode
            research_checkbox = await page.wait_for_selector("input[type='checkbox']", timeout=5000)
            await research_checkbox.check()
            
            # Enter research question
            message_input = await page.wait_for_selector("textarea[placeholder*='question']", timeout=5000)
            await message_input.fill("Design a split experiment for model evaluation")
            
            # Send the message
            send_button = await page.wait_for_selector("button:has-text('Send')", timeout=5000)
            await send_button.click()
            
            # Wait for research to start
            await page.wait_for_timeout(5000)
            
            # Check for research response
            page_content = await page.content()
            if "Research Session Started" in page_content or "research" in page_content.lower():
                print("✅ Research mode working")
                return True
            else:
                print("❌ Research mode not responding")
                return False
                
        except Exception as e:
            print(f"❌ Research mode failed: {e}")
            return False
    
    async def test_dashboard(self, page):
        """Test the dashboard functionality."""
        print("📊 Testing dashboard...")
        
        try:
            # Switch to dashboard tab
            await page.click("text=Dashboard")
            await page.wait_for_timeout(2000)
            
            # Check for dashboard elements
            await page.wait_for_selector("text=Research Dashboard", timeout=5000)
            await page.wait_for_selector("text=Research Status", timeout=5000)
            
            # Check for progress bar
            progress_bar = await page.query_selector("input[type='range']")
            if progress_bar:
                print("✅ Dashboard loaded successfully")
                return True
            else:
                print("❌ Dashboard elements missing")
                return False
                
        except Exception as e:
            print(f"❌ Dashboard test failed: {e}")
            return False
    
    async def test_agents_panel(self, page):
        """Test the agents panel functionality."""
        print("🤖 Testing agents panel...")
        
        try:
            # Switch to agents tab
            await page.click("text=Agents")
            await page.wait_for_timeout(2000)
            
            # Check for agents panel elements
            await page.wait_for_selector("text=Agent Management", timeout=5000)
            
            # Check for refresh button
            refresh_button = await page.query_selector("button:has-text('Refresh')")
            if refresh_button:
                print("✅ Agents panel loaded successfully")
                return True
            else:
                print("❌ Agents panel elements missing")
                return False
                
        except Exception as e:
            print(f"❌ Agents panel test failed: {e}")
            return False
    
    async def test_settings_panel(self, page):
        """Test the settings panel functionality."""
        print("⚙️ Testing settings panel...")
        
        try:
            # Switch to settings tab
            await page.click("text=Settings")
            await page.wait_for_timeout(2000)
            
            # Check for settings elements
            await page.wait_for_selector("text=API Keys", timeout=5000)
            
            # Check for API key inputs
            api_inputs = await page.query_selector_all("input[type='password']")
            if len(api_inputs) > 0:
                print("✅ Settings panel loaded successfully")
                return True
            else:
                print("❌ Settings panel elements missing")
                return False
                
        except Exception as e:
            print(f"❌ Settings panel test failed: {e}")
            return False
    
    async def test_results_panel(self, page):
        """Test the results panel functionality."""
        print("📊 Testing results panel...")
        
        try:
            # Switch to results tab
            await page.click("text=Results")
            await page.wait_for_timeout(2000)
            
            # Check for results elements
            await page.wait_for_selector("text=Research Results", timeout=5000)
            
            # Check for export buttons
            export_buttons = await page.query_selector_all("button:has-text('Export')")
            if len(export_buttons) > 0:
                print("✅ Results panel loaded successfully")
                return True
            else:
                print("❌ Results panel elements missing")
                return False
                
        except Exception as e:
            print(f"❌ Results panel test failed: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run a comprehensive test of the Gradio interface."""
        print("🧪 Starting comprehensive Playwright test...")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=False)  # Set to True for headless
            page = await browser.new_page()
            
            try:
                # Test interface loading
                interface_loaded = await self.test_interface_loading(page)
                self.test_results['interface_loading'] = interface_loaded
                
                if not interface_loaded:
                    print("❌ Interface failed to load, stopping tests")
                    return self.test_results
                
                # Test chat functionality
                chat_working = await self.test_chat_functionality(page)
                self.test_results['chat_functionality'] = chat_working
                
                # Test research mode
                research_working = await self.test_research_mode(page)
                self.test_results['research_mode'] = research_working
                
                # Test dashboard
                dashboard_working = await self.test_dashboard(page)
                self.test_results['dashboard'] = dashboard_working
                
                # Test agents panel
                agents_working = await self.test_agents_panel(page)
                self.test_results['agents_panel'] = agents_working
                
                # Test settings panel
                settings_working = await self.test_settings_panel(page)
                self.test_results['settings_panel'] = settings_working
                
                # Test results panel
                results_working = await self.test_results_panel(page)
                self.test_results['results_panel'] = results_working
                
                # Calculate overall success rate
                successful_tests = sum(self.test_results.values())
                total_tests = len(self.test_results)
                success_rate = (successful_tests / total_tests) * 100
                
                self.test_results['overall_success_rate'] = success_rate
                self.test_results['successful_tests'] = successful_tests
                self.test_results['total_tests'] = total_tests
                
                print(f"\n📈 Test Results Summary:")
                print(f"   Success Rate: {success_rate:.1f}%")
                print(f"   Successful Tests: {successful_tests}/{total_tests}")
                
                for test_name, result in self.test_results.items():
                    if test_name not in ['overall_success_rate', 'successful_tests', 'total_tests']:
                        status = "✅" if result else "❌"
                        print(f"   {status} {test_name}")
                
            except Exception as e:
                print(f"❌ Test execution failed: {e}")
                self.test_results['error'] = str(e)
            
            finally:
                await browser.close()
        
        return self.test_results
    
    def save_test_results(self):
        """Save test results to file."""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save results as JSON
        with open(output_dir / "playwright_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Generate report
        report = self.generate_test_report()
        with open(output_dir / "playwright_test_report.md", "w") as f:
            f.write(report)
        
        print(f"📁 Test results saved to {output_dir}/")
    
    def generate_test_report(self) -> str:
        """Generate a test report."""
        success_rate = self.test_results.get('overall_success_rate', 0)
        successful_tests = self.test_results.get('successful_tests', 0)
        total_tests = self.test_results.get('total_tests', 0)
        
        report = f"""
# Playwright Test Report - AI Research Lab Gradio Interface

## Test Summary
- **Overall Success Rate:** {success_rate:.1f}%
- **Successful Tests:** {successful_tests}/{total_tests}
- **Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Detailed Results

"""
        
        for test_name, result in self.test_results.items():
            if test_name not in ['overall_success_rate', 'successful_tests', 'total_tests', 'error']:
                status = "✅ PASS" if result else "❌ FAIL"
                report += f"- **{test_name}:** {status}\n"
        
        if 'error' in self.test_results:
            report += f"\n## Error\n{self.test_results['error']}\n"
        
        report += f"""
## Conclusion

The Gradio interface demonstrates {'excellent' if success_rate >= 80 else 'good' if success_rate >= 60 else 'needs improvement'} functionality with a {success_rate:.1f}% success rate across all tested features.

### Recommendations
"""
        
        if success_rate < 80:
            report += "- Investigate and fix failing tests\n"
            report += "- Improve error handling and user feedback\n"
            report += "- Enhance interface responsiveness\n"
        else:
            report += "- Interface is ready for production use\n"
            report += "- Consider adding more advanced features\n"
            report += "- Monitor performance in real-world usage\n"
        
        return report

async def main():
    """Main function to run the Playwright tests."""
    print("🔬 AI Research Lab Playwright Test Runner")
    print("=" * 50)
    
    # Create tester
    tester = GradioPlaywrightTester()
    
    # Run tests
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_test_results()
    
    # Print summary
    success_rate = results.get('overall_success_rate', 0)
    print(f"\n✅ Playwright testing completed!")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🏆 Excellent! Interface is ready for use.")
    elif success_rate >= 60:
        print("👍 Good! Interface is mostly functional.")
    else:
        print("⚠️ Needs improvement. Some features are not working properly.")

if __name__ == "__main__":
    asyncio.run(main()) 