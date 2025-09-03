"""
Setup Results Directory and Run Comprehensive Analysis.

This script renames demo_results to results and runs the comprehensive analysis
with complex tone signals, spectrograms, and R² comparisons.

Author: Research Team
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def rename_demo_results_to_results():
    """Rename demo_results directory to results."""
    print("📁 Setting up results directory...")
    
    # Check if demo_results exists
    if os.path.exists("demo_results"):
        # If results already exists, merge them
        if os.path.exists("results"):
            print("  📂 Merging demo_results into existing results directory...")
            for item in os.listdir("demo_results"):
                src = os.path.join("demo_results", item)
                dst = os.path.join("results", item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            shutil.rmtree("demo_results")
            print("  ✅ Merged demo_results into results")
        else:
            # Simply rename demo_results to results
            os.rename("demo_results", "results")
            print("  ✅ Renamed demo_results to results")
    else:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        print("  ✅ Created results directory")
    
    # Create subdirectories
    os.makedirs("results/spectrograms", exist_ok=True)
    os.makedirs("results/comparisons", exist_ok=True)
    print("  ✅ Created subdirectories: spectrograms/, comparisons/")


def run_comprehensive_analysis():
    """Run the comprehensive results analysis."""
    print("\n🚀 Running comprehensive analysis...")
    
    try:
        # Import and run the comprehensive analysis
        from comprehensive_results_comparison import ComprehensiveResultsAnalyzer
        
        # Create analyzer
        analyzer = ComprehensiveResultsAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        print("\n✅ Comprehensive analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running comprehensive analysis: {str(e)}")
        return False


def main():
    """Main function to setup and run analysis."""
    print("🎯 SETTING UP RESULTS AND RUNNING COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Change to src directory if not already there
    if not os.path.exists("comprehensive_results_comparison.py"):
        if os.path.exists("src/comprehensive_results_comparison.py"):
            os.chdir("src")
            print("📁 Changed to src directory")
    
    # Step 1: Rename demo_results to results
    rename_demo_results_to_results()
    
    # Step 2: Run comprehensive analysis
    success = run_comprehensive_analysis()
    
    if success:
        print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ demo_results renamed to results")
        print("✅ Comprehensive analysis completed")
        print("✅ Complex tone (15Hz + 600Hz) analysis done")
        print("✅ Pink noise analysis completed")
        print("✅ Spectrograms created for both signals")
        print("✅ R² comparison across all methods")
        print("✅ Error timeseries analysis completed")
        print("✅ All results saved to results/ directory")
        print("=" * 80)
        
        # List results directory contents
        print("\n📁 Results directory contents:")
        for root, dirs, files in os.walk("results"):
            level = root.replace("results", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("\n❌ Analysis failed. Check error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
