#!/usr/bin/env python3
"""
🎓 Yazd University Intelligent Assistant - Startup Script
========================================================

This script provides an easy way to run the Yazd University Intelligent Assistant
with different options and configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("📝 Creating .env file with default configuration...")
        create_env_file()
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected!")
        print("💡 Please activate your virtual environment first:")
        print("   source .venv/bin/activate  # On macOS/Linux")
        print("   .venv\\Scripts\\activate     # On Windows")
        return False
    
    # Check if requirements are installed
    try:
        import fastapi
        import langchain
        import torch
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Please install requirements: pip install -r requirements.txt")
        return False

def create_env_file():
    """Create a default .env file."""
    env_content = """# API Configuration
AVALAI_API_KEY=your_avalai_api_key_here

# Application Configuration
DEBUG=False
HOST=0.0.0.0
PORT=8000

# Model Configuration
EMBEDDINGS_MODEL=heydariAI/persian-embeddings
LLM_MODEL=gpt-4.1
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print("✅ .env file created successfully")
    print("⚠️  Please update AVALAI_API_KEY with your actual API key")

def check_database():
    """Check if the database exists."""
    print("🔍 Checking database...")
    
    if Path("chroma_db").exists():
        print("✅ Database found")
        return True
    else:
        print("❌ Database not found")
        return False

def create_database():
    """Create the database."""
    print("🗄️  Creating database...")
    try:
        subprocess.run([sys.executable, "create_database.py"], check=True)
        print("✅ Database created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create database: {e}")
        return False

def start_backend():
    """Start the backend server."""
    print("🚀 Starting backend server...")
    try:
        subprocess.run([sys.executable, "backend.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")

def start_frontend():
    """Start the frontend server."""
    print("🚀 Starting frontend server...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Yazd University Intelligent Assistant")
    parser.add_argument("--backend", action="store_true", help="Start backend server only")
    parser.add_argument("--frontend", action="store_true", help="Start frontend server only")
    parser.add_argument("--create-db", action="store_true", help="Create database only")
    parser.add_argument("--check", action="store_true", help="Check environment and database only")
    
    args = parser.parse_args()
    
    print("🎓 Yazd University Intelligent Assistant")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    # Check database
    if not check_database():
        print("💡 Would you like to create the database? (y/n): ", end="")
        if input().lower() == 'y':
            if not create_database():
                return
        else:
            print("❌ Cannot run without database")
            return
    
    # Handle different modes
    if args.check:
        print("✅ Environment check completed")
        return
    elif args.create_db:
        if create_database():
            print("✅ Database creation completed")
        return
    elif args.backend:
        start_backend()
    elif args.frontend:
        start_frontend()
    else:
        # Default: start both
        print("🚀 Starting both backend and frontend...")
        print("💡 Backend will be available at: http://localhost:8000")
        print("💡 Frontend will be available at: http://localhost:8501")
        print("💡 Press Ctrl+C to stop both servers")
        
        try:
            # Start backend in background
            backend_process = subprocess.Popen([sys.executable, "backend.py"])
            
            # Wait a moment for backend to start
            import time
            time.sleep(3)
            
            # Start frontend
            frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend.py"])
            
            # Wait for either process to finish
            backend_process.wait()
            frontend_process.terminate()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Servers stopped")

if __name__ == "__main__":
    main() 