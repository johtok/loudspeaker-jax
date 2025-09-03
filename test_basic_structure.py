#!/usr/bin/env python3
"""
Basic structure test to verify the framework works without JAX dependencies.

This script tests the basic structure and logic without requiring JAX installation.
"""

import sys
import os
from pathlib import Path
import numpy as np

def test_basic_imports():
    """Test basic imports that should work."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("  ✓ NumPy imported successfully")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print("  ✓ SciPy imported successfully")
    except ImportError as e:
        print(f"  ✗ SciPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        print("  ✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "src/ground_truth_model.py",
        "src/comprehensive_testing.py", 
        "src/dynax_identification.py",
        "src/run_comprehensive_tests.py",
        "tests/test_comprehensive_framework.py",
        "run_tests.py",
        "docs/testing_framework_guide.md",
        "docs/implementation_plan.md",
        "docs/technical_roadmap.md",
        "docs/mathematical_foundations.md",
        "docs/tdd_methodology.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_basic_logic():
    """Test basic logic without JAX dependencies."""
    print("\nTesting basic logic...")
    
    # Test numpy operations
    try:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.1, 2.1, 3.1])
        
        # Test loss calculation
        loss = np.mean((x - y) ** 2)
        print(f"  ✓ Loss calculation: {loss:.6f}")
        
        # Test R² calculation
        ss_res = np.sum((x - y) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"  ✓ R² calculation: {r2:.4f}")
        
        # Test NRMSE calculation
        rmse = np.sqrt(np.mean((x - y) ** 2))
        nrmse = rmse / np.std(x)
        print(f"  ✓ NRMSE calculation: {nrmse:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic logic test failed: {e}")
        return False

def test_ground_truth_structure():
    """Test ground truth model structure without JAX."""
    print("\nTesting ground truth model structure...")
    
    try:
        # Read the ground truth model file
        with open("src/ground_truth_model.py", "r") as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            "class GroundTruthLoudspeakerModel",
            "def force_factor",
            "def stiffness", 
            "def inductance",
            "def dynamics",
            "def output",
            "def simulate",
            "def generate_synthetic_data",
            "create_standard_ground_truth",
            "create_highly_nonlinear_ground_truth",
            "create_linear_ground_truth"
        ]
        
        all_present = True
        for component in required_components:
            if component in content:
                print(f"  ✓ {component}")
            else:
                print(f"  ✗ {component} - MISSING")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"  ✗ Ground truth structure test failed: {e}")
        return False

def test_comprehensive_testing_structure():
    """Test comprehensive testing framework structure."""
    print("\nTesting comprehensive testing framework structure...")
    
    try:
        # Read the comprehensive testing file
        with open("src/comprehensive_testing.py", "r") as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            "class TestResult",
            "class ComparisonResult", 
            "class ComprehensiveTester",
            "def test_method",
            "def compare_methods",
            "def generate_report",
            "def _calculate_loss",
            "def _calculate_r2",
            "def _calculate_nrmse",
            "def _calculate_mae",
            "def _calculate_correlation"
        ]
        
        all_present = True
        for component in required_components:
            if component in content:
                print(f"  ✓ {component}")
            else:
                print(f"  ✗ {component} - MISSING")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"  ✗ Comprehensive testing structure test failed: {e}")
        return False

def test_dynax_structure():
    """Test Dynax identification structure."""
    print("\nTesting Dynax identification structure...")
    
    try:
        # Read the Dynax identification file
        with open("src/dynax_identification.py", "r") as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            "class DynaxLoudspeakerModel",
            "class DynaxSystemIdentifier",
            "def identify_linear_parameters",
            "def identify_nonlinear_parameters",
            "def dynax_identification_method",
            "def dynax_linear_only_method"
        ]
        
        all_present = True
        for component in required_components:
            if component in content:
                print(f"  ✓ {component}")
            else:
                print(f"  ✗ {component} - MISSING")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"  ✗ Dynax structure test failed: {e}")
        return False

def main():
    """Run all basic structure tests."""
    print("=" * 60)
    print("BASIC STRUCTURE TESTING (No JAX Dependencies)")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_file_structure,
        test_basic_logic,
        test_ground_truth_structure,
        test_comprehensive_testing_structure,
        test_dynax_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✅ ALL BASIC STRUCTURE TESTS PASSED!")
        print("\nThe framework structure is correct!")
        print("To run full tests with JAX:")
        print("1. Install JAX: pip install jax jaxlib")
        print("2. Install additional packages: pip install diffrax equinox jaxopt")
        print("3. Run: python run_tests.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the missing components above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
