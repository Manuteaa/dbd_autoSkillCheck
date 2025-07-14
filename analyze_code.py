#!/usr/bin/env python3
"""
Static code analysis for DBD Auto Skill Check.

This script analyzes the code without requiring external dependencies,
focusing on code quality, structure, and potential issues.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Set


class CodeAnalyzer:
    """Analyzes Python code for quality and potential issues."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.issues = []
        self.metrics = {
            'total_files': 0,
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'files_with_docstrings': 0,
            'functions_with_docstrings': 0,
            'functions_with_type_hints': 0,
            'error_handling_blocks': 0,
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            analysis = {
                'path': file_path,
                'lines': len(content.splitlines()),
                'has_imports': self._check_imports(tree),
                'has_docstring': self._check_module_docstring(tree),
                'functions': self._analyze_functions(tree),
                'classes': self._analyze_classes(tree),
                'error_handling': self._count_error_handling(tree),
                'complexity_issues': self._check_complexity(tree, content),
            }
            
            self._update_metrics(analysis)
            return analysis
            
        except Exception as e:
            self.issues.append(f"Failed to analyze {file_path}: {e}")
            return {'path': file_path, 'error': str(e)}
    
    def _check_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Check import statements."""
        imports = {'standard': [], 'third_party': [], 'local': []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['standard'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith('.') or node.module.startswith('dbd'):
                        imports['local'].append(node.module)
                    else:
                        imports['third_party'].append(node.module)
        
        return imports
    
    def _check_module_docstring(self, tree: ast.AST) -> bool:
        """Check if module has a docstring."""
        return (isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str))
    
    def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze functions in the module."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'lines': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
                    'has_docstring': self._has_docstring(node),
                    'has_type_hints': self._has_type_hints(node),
                    'complexity': self._calculate_complexity(node),
                    'parameters': len(node.args.args),
                }
                functions.append(func_info)
        
        return functions
    
    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze classes in the module."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'has_docstring': self._has_docstring(node),
                    'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    'has_init': any(n.name == '__init__' for n in node.body if isinstance(n, ast.FunctionDef)),
                }
                classes.append(class_info)
        
        return classes
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a function or class has a docstring."""
        return (node.body and 
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str))
    
    def _has_type_hints(self, node: ast.FunctionDef) -> bool:
        """Check if a function has type hints."""
        has_return_annotation = node.returns is not None
        has_arg_annotations = any(arg.annotation is not None for arg in node.args.args)
        return has_return_annotation or has_arg_annotations
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _count_error_handling(self, tree: ast.AST) -> int:
        """Count error handling blocks."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                count += 1
        return count
    
    def _check_complexity(self, tree: ast.AST, content: str) -> List[str]:
        """Check for complexity issues."""
        issues = []
        lines = content.splitlines()
        
        # Check for long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append(f"Line {i}: Line too long ({len(line)} chars)")
        
        # Check for deep nesting
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                nesting = self._calculate_nesting_depth(node)
                if nesting > 4:
                    issues.append(f"Function {node.name}: Deep nesting ({nesting} levels)")
        
        return issues
    
    def _calculate_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _update_metrics(self, analysis: Dict[str, Any]) -> None:
        """Update overall metrics."""
        if 'error' in analysis:
            return
            
        self.metrics['total_files'] += 1
        self.metrics['total_lines'] += analysis['lines']
        self.metrics['total_functions'] += len(analysis['functions'])
        self.metrics['total_classes'] += len(analysis['classes'])
        
        if analysis['has_docstring']:
            self.metrics['files_with_docstrings'] += 1
        
        for func in analysis['functions']:
            if func['has_docstring']:
                self.metrics['functions_with_docstrings'] += 1
            if func['has_type_hints']:
                self.metrics['functions_with_type_hints'] += 1
        
        self.metrics['error_handling_blocks'] += analysis['error_handling']
    
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the entire project."""
        python_files = list(self.root_dir.rglob("*.py"))
        
        # Filter out __pycache__ and other build directories
        python_files = [f for f in python_files if '__pycache__' not in str(f)]
        
        results = []
        for file_path in python_files:
            analysis = self.analyze_file(file_path)
            results.append(analysis)
        
        return {
            'files': results,
            'metrics': self.metrics,
            'issues': self.issues,
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 60)
        report.append("DBD Auto Skill Check - Code Analysis Report")
        report.append("=" * 60)
        
        # Overall metrics
        metrics = results['metrics']
        report.append("\n📊 PROJECT METRICS:")
        report.append(f"  • Total Python files: {metrics['total_files']}")
        report.append(f"  • Total lines of code: {metrics['total_lines']}")
        report.append(f"  • Total functions: {metrics['total_functions']}")
        report.append(f"  • Total classes: {metrics['total_classes']}")
        
        # Code quality metrics
        if metrics['total_files'] > 0:
            docstring_coverage = (metrics['files_with_docstrings'] / metrics['total_files']) * 100
            report.append(f"\n📚 DOCUMENTATION COVERAGE:")
            report.append(f"  • Files with docstrings: {docstring_coverage:.1f}%")
        
        if metrics['total_functions'] > 0:
            func_docstring_coverage = (metrics['functions_with_docstrings'] / metrics['total_functions']) * 100
            type_hint_coverage = (metrics['functions_with_type_hints'] / metrics['total_functions']) * 100
            report.append(f"  • Functions with docstrings: {func_docstring_coverage:.1f}%")
            report.append(f"  • Functions with type hints: {type_hint_coverage:.1f}%")
        
        report.append(f"\n🛡️ ERROR HANDLING:")
        report.append(f"  • Error handling blocks: {metrics['error_handling_blocks']}")
        
        # File-by-file analysis
        report.append(f"\n📁 FILE ANALYSIS:")
        for file_analysis in results['files']:
            if 'error' in file_analysis:
                continue
                
            file_path = file_analysis['path']
            rel_path = file_path.relative_to(self.root_dir)
            
            report.append(f"\n  📄 {rel_path}")
            report.append(f"     Lines: {file_analysis['lines']}")
            report.append(f"     Functions: {len(file_analysis['functions'])}")
            report.append(f"     Classes: {len(file_analysis['classes'])}")
            report.append(f"     Error handling: {file_analysis['error_handling']} blocks")
            
            # Function details
            if file_analysis['functions']:
                complex_functions = [f for f in file_analysis['functions'] if f['complexity'] > 10]
                if complex_functions:
                    report.append(f"     ⚠️  Complex functions: {[f['name'] for f in complex_functions]}")
            
            # Complexity issues
            if file_analysis['complexity_issues']:
                report.append(f"     ⚠️  Issues: {len(file_analysis['complexity_issues'])}")
        
        # Issues summary
        if results['issues']:
            report.append(f"\n⚠️ ISSUES FOUND:")
            for issue in results['issues']:
                report.append(f"  • {issue}")
        
        # Quality assessment
        report.append(f"\n✅ QUALITY ASSESSMENT:")
        
        quality_score = 0
        max_score = 5
        
        # Documentation score
        if metrics['total_functions'] > 0:
            func_doc_ratio = metrics['functions_with_docstrings'] / metrics['total_functions']
            type_hint_ratio = metrics['functions_with_type_hints'] / metrics['total_functions']
            
            if func_doc_ratio > 0.8:
                quality_score += 1
                report.append("  ✓ Good documentation coverage")
            elif func_doc_ratio > 0.5:
                report.append("  ⚠️  Moderate documentation coverage")
            else:
                report.append("  ❌ Poor documentation coverage")
            
            if type_hint_ratio > 0.8:
                quality_score += 1
                report.append("  ✓ Excellent type hint coverage")
            elif type_hint_ratio > 0.5:
                report.append("  ⚠️  Moderate type hint coverage")
            else:
                report.append("  ❌ Poor type hint coverage")
        
        # Error handling score
        if metrics['error_handling_blocks'] > metrics['total_functions'] * 0.3:
            quality_score += 1
            report.append("  ✓ Good error handling coverage")
        else:
            report.append("  ⚠️  Could improve error handling")
        
        # Code organization score
        if metrics['total_classes'] > 0 and metrics['total_functions'] > 0:
            quality_score += 1
            report.append("  ✓ Well-structured code organization")
        
        # File organization score
        if any('utils' in str(f['path']) for f in results['files']):
            quality_score += 1
            report.append("  ✓ Good file organization with utilities")
        
        report.append(f"\n🏆 OVERALL QUALITY SCORE: {quality_score}/{max_score}")
        
        if quality_score >= 4:
            report.append("  🌟 Excellent code quality!")
        elif quality_score >= 3:
            report.append("  👍 Good code quality")
        elif quality_score >= 2:
            report.append("  📈 Moderate code quality - room for improvement")
        else:
            report.append("  🔧 Needs significant improvement")
        
        return "\n".join(report)


def main():
    """Main analysis function."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting static code analysis...")
    analyzer = CodeAnalyzer(root_dir)
    results = analyzer.analyze_project()
    
    report = analyzer.generate_report(results)
    print(report)
    
    # Save report to file
    with open('code_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: code_analysis_report.txt")
    
    # Return success/failure based on quality
    quality_score = 0
    metrics = results['metrics']
    
    if metrics['total_functions'] > 0:
        if metrics['functions_with_docstrings'] / metrics['total_functions'] > 0.5:
            quality_score += 1
        if metrics['functions_with_type_hints'] / metrics['total_functions'] > 0.5:
            quality_score += 1
    
    return quality_score >= 1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)