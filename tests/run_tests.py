"""
Comprehensive Test Runner for TDD Framework.

This module provides utilities for running tests with different configurations,
generating test reports, and managing test execution.

Author: Research Team
"""

import pytest
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import subprocess
import json
from datetime import datetime


class TestRunner:
    """Comprehensive test runner for the loudspeaker-jax project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_unit_tests(self, verbose: bool = True, coverage: bool = True) -> int:
        """Run unit tests only."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml"
            ])
        
        cmd.extend([
            "-m", "unit",
            "--tb=short"
        ])
        
        return subprocess.call(cmd)
    
    def run_integration_tests(self, verbose: bool = True) -> int:
        """Run integration tests only."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "-m", "integration",
            "--tb=short"
        ])
        
        return subprocess.call(cmd)
    
    def run_property_tests(self, verbose: bool = True) -> int:
        """Run property-based tests only."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "-m", "property",
            "--tb=short",
            "--hypothesis-show-statistics"
        ])
        
        return subprocess.call(cmd)
    
    def run_performance_tests(self, verbose: bool = True) -> int:
        """Run performance tests only."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "-m", "performance",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-save=performance_results"
        ])
        
        return subprocess.call(cmd)
    
    def run_mathematical_tests(self, verbose: bool = True) -> int:
        """Run mathematical correctness tests only."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "-m", "mathematical",
            "--tb=short"
        ])
        
        return subprocess.call(cmd)
    
    def run_physical_tests(self, verbose: bool = True) -> int:
        """Run physical constraint tests only."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "-m", "physical",
            "--tb=short"
        ])
        
        return subprocess.call(cmd)
    
    def run_all_tests(self, verbose: bool = True, coverage: bool = True, 
                     parallel: bool = False) -> int:
        """Run all tests."""
        cmd = ["pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml"
            ])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        cmd.extend([
            "--tb=short",
            "--hypothesis-show-statistics"
        ])
        
        return subprocess.call(cmd)
    
    def run_tdd_cycle(self, test_file: str, verbose: bool = True) -> int:
        """Run TDD cycle for a specific test file."""
        print(f"Running TDD cycle for {test_file}")
        
        # Run tests to see failures
        cmd = ["pytest", str(self.test_dir / test_file)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "--maxfail=1"  # Stop on first failure
        ])
        
        return subprocess.call(cmd)
    
    def generate_test_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive test report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.reports_dir / f"test_report_{timestamp}.json"
        
        # Run tests with JSON output
        cmd = [
            "pytest", str(self.test_dir),
            "--json-report",
            f"--json-report-file={output_file}",
            "--cov=src",
            "--cov-report=json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Test report generated: {output_file}")
        else:
            print(f"Test execution failed: {result.stderr}")
        
        return str(output_file)
    
    def run_benchmark_comparison(self) -> int:
        """Run benchmark comparison tests."""
        cmd = [
            "pytest", str(self.test_dir),
            "-m", "performance",
            "--benchmark-compare",
            "--benchmark-compare-fail=mean:10%"
        ]
        
        return subprocess.call(cmd)
    
    def check_test_coverage(self, min_coverage: float = 90.0) -> bool:
        """Check if test coverage meets minimum requirements."""
        cmd = [
            "pytest", str(self.test_dir),
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-fail-under", str(min_coverage)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def run_smoke_tests(self) -> int:
        """Run smoke tests for basic functionality."""
        cmd = [
            "pytest", str(self.test_dir),
            "-m", "smoke",
            "--tb=short"
        ]
        
        return subprocess.call(cmd)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Test runner for loudspeaker-jax project")
    parser.add_argument("--test-type", choices=[
        "unit", "integration", "property", "performance", 
        "mathematical", "physical", "all", "smoke"
    ], default="all", help="Type of tests to run")
    
    parser.add_argument("--tdd-file", help="Run TDD cycle for specific test file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")
    parser.add_argument("--coverage-check", type=float, help="Check minimum coverage percentage")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.tdd_file:
        return runner.run_tdd_cycle(args.tdd_file, args.verbose)
    
    if args.report:
        report_file = runner.generate_test_report()
        print(f"Test report saved to: {report_file}")
        return 0
    
    if args.benchmark:
        return runner.run_benchmark_comparison()
    
    if args.coverage_check is not None:
        success = runner.check_test_coverage(args.coverage_check)
        if success:
            print(f"Coverage check passed (>= {args.coverage_check}%)")
        else:
            print(f"Coverage check failed (< {args.coverage_check}%)")
        return 0 if success else 1
    
    # Run tests based on type
    coverage = not args.no_coverage
    
    if args.test_type == "unit":
        return runner.run_unit_tests(args.verbose, coverage)
    elif args.test_type == "integration":
        return runner.run_integration_tests(args.verbose)
    elif args.test_type == "property":
        return runner.run_property_tests(args.verbose)
    elif args.test_type == "performance":
        return runner.run_performance_tests(args.verbose)
    elif args.test_type == "mathematical":
        return runner.run_mathematical_tests(args.verbose)
    elif args.test_type == "physical":
        return runner.run_physical_tests(args.verbose)
    elif args.test_type == "smoke":
        return runner.run_smoke_tests()
    else:  # all
        return runner.run_all_tests(args.verbose, coverage, args.parallel)


if __name__ == "__main__":
    sys.exit(main())
