#!/usr/bin/env python3
"""
Launch script for the Secure Crew Feedback Parser Web UI
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import openai
        print("✅ Required packages found")
        return True
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install requirements:")
        print("pip install -r web_requirements.txt")
        print("pip install -r requirements.txt")
        return False

def main():
    """Main function to launch the secure web UI"""
    print("🚢 Secure Crew Feedback Parser Web UI")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists('secure_web_ui.py'):
        print("❌ secure_web_ui.py not found. Please run from the project root directory.")
        sys.exit(1)
    
    print("\n🔐 Security Features:")
    print("• Users must provide their own OpenAI API key")
    print("• API keys are validated before use")
    print("• Keys stored in secure server sessions only")
    print("• No API keys saved to disk or logs")
    print("• Automatic file cleanup after processing")
    
    print("\n🌐 Starting web server...")
    print("Access the application at: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the secure web UI
        from secure_web_ui import app
        app.run(debug=False, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()