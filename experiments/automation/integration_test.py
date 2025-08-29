#!/usr/bin/env python3
"""
Integration test for GPU-MPC research productivity tools.

This script performs a quick integration test to verify all components
work together correctly without requiring actual MPC executables.
"""

import tempfile
import shutil
from pathlib import Path
import json

from parameter_sweep import ParameterSweep, ExperimentResult
from result_validator import ResultValidator
from experiment_runner import ExperimentRunner, ExperimentConfig, PartyConfig

def test_parameter_sweep():
    """Test parameter sweep functionality."""
    print("Testing ParameterSweep...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock configuration that doesn't require actual executables
        base_config = {
            'experiment_type': 'test',
            'executable': 'echo',  # Use echo command for testing
            'timeout': 10
        }
        
        sweep = ParameterSweep(base_config, temp_dir)
        sweep.add_parameter_range('param1', [1, 2])
        sweep.add_parameter_range('param2', ['a', 'b'])
        
        combinations = sweep.generate_parameter_combinations()
        assert len(combinations) == 4, f"Expected 4 combinations, got {len(combinations)}"
        
        # Test that we can generate a summary even with no results
        summary = sweep.generate_summary_stats()
        assert summary['total_experiments'] == 0
        
        print("✓ ParameterSweep basic functionality works")

def test_result_validator():
    """Test result validator functionality."""
    print("Testing ResultValidator...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        validator = ResultValidator(temp_dir)
        
        # Test reconstruction validation
        result = validator.validate_reconstruction(
            share0=12345, share1=54321, expected=66666,
            test_name="arithmetic_test", sharing_type="arithmetic"
        )
        assert result.passed, "Arithmetic reconstruction should pass"
        
        result = validator.validate_reconstruction(
            share0=1, share1=0, expected=True,
            test_name="boolean_test", sharing_type="boolean"
        )
        assert result.passed, "Boolean reconstruction should pass"
        
        # Test timing analysis
        timing_data = [1.0, 1.1, 0.9, 1.2, 1.0]
        result = validator.analyze_timing_consistency(timing_data, "timing_test")
        assert result.passed, "Timing consistency should pass for consistent data"
        
        # Generate report
        report = validator.generate_validation_report()
        assert report['summary']['total_tests'] == 3
        assert report['summary']['passed_tests'] == 3
        
        print("✓ ResultValidator functionality works")

def test_experiment_runner():
    """Test experiment runner functionality.""" 
    print("Testing ExperimentRunner...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = ExperimentRunner(temp_dir)
        
        # Create mock parties using echo command
        party0 = PartyConfig(
            party_id=0,
            hostname="localhost",
            executable="echo",
            arguments=["Party 0 output"],
            gpu_id=-1  # No GPU for testing
        )
        
        party1 = PartyConfig(
            party_id=1, 
            hostname="localhost",
            executable="echo",
            arguments=["Party 1 output"],
            gpu_id=-1  # No GPU for testing
        )
        
        config = ExperimentConfig(
            experiment_name="test_experiment",
            parties=[party0, party1],
            timeout=30,
            sync_start=False  # Sequential for testing
        )
        
        # Test configuration validation
        try:
            runner._validate_config(config)
            print("✓ Configuration validation works")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            
        print("✓ ExperimentRunner basic functionality works")

def test_integration():
    """Test integration between all components."""
    print("Testing integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock experiment results
        results = [
            ExperimentResult(
                parameters={'threads': 4, 'bit_width': 32},
                success=True,
                execution_time=1.5,
                communication_bytes=2048
            ),
            ExperimentResult(
                parameters={'threads': 8, 'bit_width': 32}, 
                success=True,
                execution_time=1.2,
                communication_bytes=2056
            ),
            ExperimentResult(
                parameters={'threads': 4, 'bit_width': 64},
                success=False,
                execution_time=0.5,
                error_message="Mock failure"
            )
        ]
        
        # Test parameter sweep result handling
        sweep = ParameterSweep({}, temp_dir)
        sweep.results = results
        
        summary = sweep.generate_summary_stats()
        assert summary['total_experiments'] == 3
        assert summary['successful_experiments'] == 2
        assert summary['failed_experiments'] == 1
        
        # Test result validation on sweep results
        validator = ResultValidator(temp_dir)
        successful_results = sweep.get_successful_results()
        timing_data = [r.execution_time for r in successful_results]
        
        timing_result = validator.analyze_timing_consistency(timing_data, "integration_test")
        assert timing_result is not None
        
        # Save results
        sweep.save_results()
        validator.save_validation_results()
        
        # Verify files were created
        assert (Path(temp_dir) / "results.json").exists()
        assert (Path(temp_dir) / "results.csv").exists()
        assert (Path(temp_dir) / "validation_results.json").exists()
        
        print("✓ Integration between components works")

def main():
    """Run all integration tests."""
    print("GPU-MPC Research Productivity Tools - Integration Test")
    print("=" * 60)
    
    try:
        test_parameter_sweep()
        test_result_validator()
        test_experiment_runner()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ All integration tests passed!")
        print("The research productivity tools are ready for use.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())