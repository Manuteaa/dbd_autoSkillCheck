#!/usr/bin/env python3
"""
Setup script for DBD Auto Skill Check.

This script handles installation of dependencies and initial setup.
"""

import sys
import subprocess
import os
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")


def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("Error: requirements.txt not found!")
        return False
    
    try:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["models", "logs", "saved_images"]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")


def check_model_files():
    """Check if model files are present."""
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.onnx")) + list(models_dir.glob("*.trt"))
    
    if not model_files:
        print("⚠️  No model files found in models/ directory")
        print("   Please download model.onnx from the releases page:")
        print("   https://github.com/Manuteaa/dbd_autoSkillCheck/releases")
        return False
    else:
        print(f"✓ Found {len(model_files)} model file(s):")
        for model in model_files:
            print(f"   - {model.name}")
        return True


def setup_git_hooks():
    """Setup git hooks for development (optional)."""
    git_dir = Path(".git")
    if not git_dir.exists():
        return
    
    hooks_dir = git_dir / "hooks"
    pre_commit_hook = hooks_dir / "pre-commit"
    
    if not pre_commit_hook.exists():
        hook_content = """#!/bin/sh
# Pre-commit hook for code formatting and linting

echo "Running code quality checks..."

# Check if black is available for formatting
if command -v black >/dev/null 2>&1; then
    echo "Formatting Python code with black..."
    black --check --diff .
fi

# Check if flake8 is available for linting
if command -v flake8 >/dev/null 2>&1; then
    echo "Linting Python code with flake8..."
    flake8 --max-line-length=100 --ignore=E203,W503 .
fi

echo "Pre-commit checks completed."
"""
        try:
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)
            pre_commit_hook.chmod(0o755)
            print("✓ Git pre-commit hook installed")
        except Exception as e:
            print(f"⚠️  Could not install git hooks: {e}")


def create_launcher_scripts():
    """Create launcher scripts for different platforms."""
    
    # Windows batch file
    bat_content = """@echo off
echo Starting DBD Auto Skill Check...
python app.py
pause
"""
    
    with open("run_app.bat", 'w') as f:
        f.write(bat_content)
    print("✓ Created Windows launcher: run_app.bat")
    
    # Unix shell script
    sh_content = """#!/bin/bash
echo "Starting DBD Auto Skill Check..."
python3 app.py
"""
    
    with open("run_app.sh", 'w') as f:
        f.write(sh_content)
    
    # Make shell script executable
    try:
        os.chmod("run_app.sh", 0o755)
        print("✓ Created Unix launcher: run_app.sh")
    except Exception:
        print("⚠️  Created Unix launcher but could not set executable permission")


def run_tests():
    """Run basic tests to verify installation."""
    print("\nRunning installation tests...")
    
    try:
        # Test imports
        import numpy
        import PIL
        import gradio
        import onnxruntime
        import mss
        print("✓ All required packages can be imported")
        
        # Test model loading (if available)
        from dbd.AI_model import AI_model
        print("✓ AI_model class can be imported")
        
        # Test utilities
        from dbd.utils.monitor import get_monitors
        monitors = get_monitors()
        print(f"✓ Detected {len(monitors)} monitor(s)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False


def main():
    """Main setup function."""
    print("DBD Auto Skill Check - Setup Script")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed: Could not install requirements")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check for model files
    has_models = check_model_files()
    
    # Setup development tools (optional)
    setup_git_hooks()
    
    # Create launcher scripts
    create_launcher_scripts()
    
    # Run tests
    if run_tests():
        print("\n" + "=" * 40)
        print("✓ Setup completed successfully!")
        
        if has_models:
            print("\nYou can now run the application with:")
            print("  python app.py")
            print("  or double-click run_app.bat (Windows)")
            print("  or ./run_app.sh (Unix/Linux/Mac)")
        else:
            print("\n⚠️  Please download model files before running the application")
            
        print("\nFor more information, visit:")
        print("https://github.com/Manuteaa/dbd_autoSkillCheck")
        
    else:
        print("\n✗ Setup completed with errors")
        print("Please check the error messages above and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()