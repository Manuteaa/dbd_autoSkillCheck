"""
Code formatting and quality tools for DBD Auto Skill Check.

This module provides utilities for code formatting, linting, and quality checks.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional


def check_tool_available(tool_name: str) -> bool:
    """Check if a development tool is available."""
    try:
        subprocess.run([tool_name, '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_dev_tools() -> bool:
    """Install development tools if not available."""
    tools = ['black', 'flake8', 'mypy', 'isort']
    missing_tools = [tool for tool in tools if not check_tool_available(tool)]
    
    if not missing_tools:
        print("✓ All development tools are already installed")
        return True
    
    print(f"Installing missing tools: {missing_tools}")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install'
        ] + missing_tools, check=True)
        print("✓ Development tools installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install tools: {e}")
        return False


def format_code(directory: str = ".") -> Tuple[bool, List[str]]:
    """Format code using black and isort."""
    results = []
    success = True
    
    # Format with black
    if check_tool_available('black'):
        try:
            result = subprocess.run([
                'black', '--line-length', '100', directory
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                results.append("✓ Code formatted with black")
            else:
                results.append(f"⚠️ Black formatting issues: {result.stderr}")
                success = False
        except Exception as e:
            results.append(f"✗ Black formatting failed: {e}")
            success = False
    else:
        results.append("⚠️ Black not available - install with: pip install black")
    
    # Sort imports with isort
    if check_tool_available('isort'):
        try:
            result = subprocess.run([
                'isort', '--profile', 'black', '--line-length', '100', directory
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                results.append("✓ Imports sorted with isort")
            else:
                results.append(f"⚠️ Import sorting issues: {result.stderr}")
                success = False
        except Exception as e:
            results.append(f"✗ Import sorting failed: {e}")
            success = False
    else:
        results.append("⚠️ isort not available - install with: pip install isort")
    
    return success, results


def lint_code(directory: str = ".") -> Tuple[bool, List[str]]:
    """Lint code using flake8."""
    results = []
    success = True
    
    if not check_tool_available('flake8'):
        results.append("⚠️ flake8 not available - install with: pip install flake8")
        return False, results
    
    try:
        result = subprocess.run([
            'flake8', 
            '--max-line-length', '100',
            '--ignore', 'E203,W503,E501',  # Ignore conflicts with black
            '--exclude', '__pycache__,venv,env,build,dist',
            directory
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            results.append("✓ No linting issues found")
        else:
            results.append("⚠️ Linting issues found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    results.append(f"    {line}")
            success = False
            
    except Exception as e:
        results.append(f"✗ Linting failed: {e}")
        success = False
    
    return success, results


def check_types(directory: str = ".") -> Tuple[bool, List[str]]:
    """Check types using mypy."""
    results = []
    success = True
    
    if not check_tool_available('mypy'):
        results.append("⚠️ mypy not available - install with: pip install mypy")
        return False, results
    
    try:
        # Create mypy config if it doesn't exist
        mypy_config = Path(directory) / 'mypy.ini'
        if not mypy_config.exists():
            config_content = """[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
ignore_missing_imports = True
"""
            with open(mypy_config, 'w') as f:
                f.write(config_content)
            results.append("✓ Created mypy.ini configuration")
        
        result = subprocess.run([
            'mypy', directory
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            results.append("✓ No type checking issues found")
        else:
            results.append("⚠️ Type checking issues found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('Success'):
                    results.append(f"    {line}")
            # Don't mark as failure for type issues, just warnings
            
    except Exception as e:
        results.append(f"✗ Type checking failed: {e}")
        success = False
    
    return success, results


def run_security_check(directory: str = ".") -> Tuple[bool, List[str]]:
    """Run basic security checks."""
    results = []
    success = True
    
    # Check for common security issues
    python_files = list(Path(directory).rglob("*.py"))
    
    security_patterns = [
        (r'eval\s*\(', "Avoid using eval() - security risk"),
        (r'exec\s*\(', "Avoid using exec() - security risk"),
        (r'subprocess\.call\([^)]*shell=True', "Avoid shell=True in subprocess calls"),
        (r'os\.system\s*\(', "Avoid os.system() - use subprocess instead"),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
    ]
    
    import re
    
    issues_found = False
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern, message in security_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    results.append(f"⚠️ {file_path.relative_to(directory)}: {message}")
                    issues_found = True
                    
        except Exception as e:
            results.append(f"✗ Could not check {file_path}: {e}")
            success = False
    
    if not issues_found and success:
        results.append("✓ No obvious security issues found")
    
    return success, results


def generate_quality_report(directory: str = ".") -> str:
    """Generate a comprehensive code quality report."""
    report = []
    report.append("=" * 60)
    report.append("CODE QUALITY REPORT")
    report.append("=" * 60)
    
    # Format code
    report.append("\n🎨 CODE FORMATTING:")
    format_success, format_results = format_code(directory)
    report.extend(format_results)
    
    # Lint code
    report.append("\n🔍 LINTING:")
    lint_success, lint_results = lint_code(directory)
    report.extend(lint_results)
    
    # Check types
    report.append("\n📝 TYPE CHECKING:")
    type_success, type_results = check_types(directory)
    report.extend(type_results)
    
    # Security check
    report.append("\n🔒 SECURITY CHECK:")
    security_success, security_results = run_security_check(directory)
    report.extend(security_results)
    
    # Overall assessment
    report.append("\n🏆 OVERALL ASSESSMENT:")
    
    passed_checks = sum([format_success, lint_success, type_success, security_success])
    total_checks = 4
    
    if passed_checks == total_checks:
        report.append("✅ All quality checks passed!")
    elif passed_checks >= total_checks * 0.75:
        report.append("🟡 Most quality checks passed - minor issues to address")
    else:
        report.append("🔴 Several quality issues found - improvement needed")
    
    report.append(f"Score: {passed_checks}/{total_checks}")
    
    return "\n".join(report)


def main():
    """Main function for running code quality checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code quality tools for DBD Auto Skill Check")
    parser.add_argument('--install', action='store_true', help='Install development tools')
    parser.add_argument('--format', action='store_true', help='Format code')
    parser.add_argument('--lint', action='store_true', help='Lint code')
    parser.add_argument('--types', action='store_true', help='Check types')
    parser.add_argument('--security', action='store_true', help='Security check')
    parser.add_argument('--all', action='store_true', help='Run all checks')
    parser.add_argument('--report', action='store_true', help='Generate quality report')
    parser.add_argument('directory', nargs='?', default='.', help='Directory to check')
    
    args = parser.parse_args()
    
    if args.install:
        install_dev_tools()
        return
    
    if args.all or args.report:
        report = generate_quality_report(args.directory)
        print(report)
        
        # Save report
        with open('quality_report.txt', 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to quality_report.txt")
        return
    
    if args.format:
        success, results = format_code(args.directory)
        for result in results:
            print(result)
    
    if args.lint:
        success, results = lint_code(args.directory)
        for result in results:
            print(result)
    
    if args.types:
        success, results = check_types(args.directory)
        for result in results:
            print(result)
    
    if args.security:
        success, results = run_security_check(args.directory)
        for result in results:
            print(result)
    
    if not any([args.format, args.lint, args.types, args.security, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()