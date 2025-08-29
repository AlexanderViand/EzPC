#!/usr/bin/env python3
# Author: AI Assistant / Claude
# Copyright:
# 
# Copyright (c) 2024 Microsoft Research
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Research Productivity Tool: Example Parameter Sweep for GPU-MPC

This script demonstrates how to use the automation tools for systematic
parameter exploration and result validation in MPC experiments.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add automation tools to path
sys.path.append(str(Path(__file__).parent))

from parameter_sweep import ParameterSweep, ExperimentResult
from result_validator import ResultValidator, ValidationResult
from experiment_runner import ExperimentRunner, ExperimentConfig, PartyConfig

def run_dcf_parameter_sweep(output_dir: str = "dcf_sweep_results"):
    """
    Example: DCF protocol parameter sweep.
    
    Sweeps over bit widths, thread counts, and element counts to analyze
    performance characteristics of the DCF protocol.
    """
    print("=== DCF Protocol Parameter Sweep ===")
    
    # Base configuration for DCF benchmark
    base_config = {
        'experiment_type': 'mpc_benchmark',
        'executable': './build/benchmarks/mpc_benchmark',
        'task': 'dcf',
        'party': 0,  # This will be overridden for each party
        'peer': '127.0.0.1',
        'timeout': 1800,  # 30 minutes per experiment
        'working_dir': '.'
    }
    
    # Create parameter sweep
    sweep = ParameterSweep(base_config, output_dir)
    
    # Add parameter ranges
    sweep.add_bit_width_range([32, 48, 64])  # Different input bit widths
    sweep.add_parameter_range('threads', [1, 2, 4, 8])  # Thread count scaling
    sweep.add_parameter_range('element_count', [1000, 10000, 100000])  # Data size scaling
    
    print(f"Configured parameter sweep with {len(sweep.generate_parameter_combinations())} combinations")
    
    # Run the sweep
    results = sweep.run_sweep(resume=True)
    
    # Generate summary statistics
    summary = sweep.generate_summary_stats()
    
    print(f"\nSweep completed:")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Successful: {summary['successful_experiments']}")
    print(f"  Failed: {summary['failed_experiments']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Average execution time: {summary.get('average_execution_time', 0):.2f}s")
    
    if summary.get('average_communication_bytes'):
        print(f"  Average communication: {summary['average_communication_bytes']:,} bytes")
        
    return results, summary

def run_multi_protocol_comparison(output_dir: str = "protocol_comparison"):
    """
    Example: Compare multiple MPC protocols.
    
    Runs DCF, SCMP, and TwoMax protocols with identical parameters
    to enable fair performance comparison.
    """
    print("\n=== Multi-Protocol Performance Comparison ===")
    
    # Base configuration
    base_config = {
        'experiment_type': 'mpc_benchmark',
        'executable': './build/benchmarks/mpc_benchmark',
        'party': 0,
        'peer': '127.0.0.1',
        'threads': 4,
        'bit_width': 64,
        'timeout': 900,  # 15 minutes per experiment
        'working_dir': '.'
    }
    
    sweep = ParameterSweep(base_config, output_dir)
    
    # Compare different protocols
    sweep.add_parameter_range('task', ['dcf', 'scmp', 'twomax'])
    
    # Multiple runs for statistical significance
    sweep.add_parameter_range('run_id', list(range(5)))  # 5 runs per protocol
    
    print(f"Configured protocol comparison with {len(sweep.generate_parameter_combinations())} experiments")
    
    results = sweep.run_sweep(resume=True)
    
    # Analyze results by protocol
    protocol_stats = {}
    for result in sweep.get_successful_results():
        protocol = result.parameters['task']
        if protocol not in protocol_stats:
            protocol_stats[protocol] = []
        protocol_stats[protocol].append(result.execution_time)
        
    print(f"\nProtocol performance comparison:")
    for protocol, times in protocol_stats.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"  {protocol.upper()}: {avg_time:.2f}s avg (min: {min_time:.2f}s, max: {max_time:.2f}s)")
            
    return results, protocol_stats

def run_validation_example(experiment_results: list):
    """
    Example: Validate experiment results.
    
    Demonstrates how to use the result validator to check correctness
    and analyze timing consistency.
    """
    print("\n=== Result Validation Example ===")
    
    validator = ResultValidator("validation_example")
    
    # Example 1: Validate reconstruction correctness
    # In a real scenario, these would come from actual MPC outputs
    validator.validate_reconstruction(
        share0=0x12345678, share1=0x87654321, expected=True,
        test_name="dcf_correctness_test", sharing_type="boolean"
    )
    
    # Example 2: Validate timing consistency
    successful_results = [r for r in experiment_results if r.success]
    if successful_results:
        timing_data = [r.execution_time for r in successful_results[:10]]  # Use first 10 results
        validator.analyze_timing_consistency(timing_data, "dcf_timing")
        
    # Example 3: Validate communication efficiency
    comm_data = [2048, 2056, 2044, 2052, 2048]  # Example communication bytes
    validator.validate_communication_efficiency(comm_data, 4, "dcf_communication")
    
    # Example 4: Validate experiment directory structure
    validator.validate_experiment_directory("dcf_sweep_results", 
                                          expected_files=["results.json", "results.csv"])
    
    # Save validation results
    validator.save_validation_results()
    
    # Generate validation report
    report = validator.generate_validation_report()
    
    print(f"Validation completed:")
    print(f"  Total tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed_tests']}")
    print(f"  Failed: {report['summary']['failed_tests']}")
    print(f"  Success rate: {report['summary']['success_rate']:.1%}")
    
    if report['failed_tests']:
        print(f"\nFailed tests:")
        for failed_test in report['failed_tests']:
            print(f"  - {failed_test['test_name']}: {failed_test['error_message']}")
            
    return report

def run_distributed_experiment_example():
    """
    Example: Run distributed multi-party experiment.
    
    Demonstrates how to use the experiment runner for coordinated
    multi-party execution.
    """
    print("\n=== Distributed Experiment Example ===")
    
    runner = ExperimentRunner("distributed_example")
    
    # Configure parties (in real scenario, these would be on different machines)
    party0 = PartyConfig(
        party_id=0,
        hostname="localhost",  # In production: actual hostname/IP
        executable="./build/benchmarks/mpc_benchmark",
        arguments=["--task", "dcf", "--party", "0", "--peer", "127.0.0.1", "--threads", "4"],
        gpu_id=0
    )
    
    party1 = PartyConfig(
        party_id=1,
        hostname="localhost",  # In production: actual hostname/IP
        executable="./build/benchmarks/mpc_benchmark",
        arguments=["--task", "dcf", "--party", "1", "--peer", "127.0.0.1", "--threads", "4"],
        gpu_id=0
    )
    
    config = ExperimentConfig(
        experiment_name="dcf_distributed_test",
        parties=[party0, party1],
        timeout=600,  # 10 minutes
        sync_start=True,
        collect_party_outputs=True
    )
    
    print("Running distributed DCF experiment...")
    result = runner.run_experiment(config)
    
    print(f"Distributed experiment: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    if result.success:
        print(f"Output files: {len(result.output_files)}")
        print(f"Party results: {len(result.party_results)} parties")
    else:
        print(f"Error: {result.error_message}")
        
    return result

def generate_comprehensive_report(sweep_results, protocol_stats, validation_report, distributed_result):
    """
    Generate a comprehensive research report.
    
    Combines results from parameter sweeps, protocol comparisons,
    validation, and distributed experiments into a unified report.
    """
    print("\n=== Generating Comprehensive Report ===")
    
    report = {
        "experiment_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(sweep_results),
            "protocols_tested": list(protocol_stats.keys()) if protocol_stats else [],
            "validation_success_rate": validation_report['summary']['success_rate'],
            "distributed_test_success": distributed_result.success if distributed_result else False
        },
        "parameter_sweep_results": {
            "successful_experiments": len([r for r in sweep_results if r.success]),
            "failed_experiments": len([r for r in sweep_results if not r.success]),
            "performance_insights": {}
        },
        "protocol_comparison": protocol_stats,
        "validation_summary": validation_report['summary'],
        "distributed_execution": {
            "success": distributed_result.success if distributed_result else False,
            "execution_time": distributed_result.execution_time if distributed_result else 0,
            "party_count": len(distributed_result.party_results) if distributed_result and distributed_result.party_results else 0
        },
        "research_insights": {
            "performance_trends": "Analyze parameter impact on execution time",
            "scalability_analysis": "Thread count vs performance relationship",
            "protocol_efficiency": "Communication overhead comparison",
            "reliability_metrics": "Success rate and consistency analysis"
        }
    }
    
    # Save comprehensive report
    report_file = Path("automation_example_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Comprehensive report saved to: {report_file}")
    
    # Print key insights
    print("\nKey Research Insights:")
    print(f"  • Tested {report['experiment_summary']['total_experiments']} parameter combinations")
    print(f"  • Compared {len(report['experiment_summary']['protocols_tested'])} MPC protocols")
    print(f"  • Achieved {report['experiment_summary']['validation_success_rate']:.1%} validation success rate")
    print(f"  • Distributed execution: {'✓' if report['distributed_execution']['success'] else '✗'}")
    
    return report

def main():
    """Main function demonstrating the automation tools."""
    parser = argparse.ArgumentParser(description="GPU-MPC Automation Tools Example")
    parser.add_argument("--skip-sweep", action="store_true", 
                       help="Skip parameter sweep (for faster testing)")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip protocol comparison")
    parser.add_argument("--skip-distributed", action="store_true",
                       help="Skip distributed experiment")
    parser.add_argument("--output-dir", default="automation_example_output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("GPU-MPC Research Productivity Tools - Example Usage")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory for relative paths
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Example 1: Parameter sweep
        sweep_results = []
        if not args.skip_sweep:
            sweep_results, sweep_summary = run_dcf_parameter_sweep()
        
        # Example 2: Protocol comparison  
        protocol_stats = {}
        if not args.skip_comparison:
            comparison_results, protocol_stats = run_multi_protocol_comparison()
            sweep_results.extend(comparison_results)
            
        # Example 3: Result validation
        validation_report = run_validation_example(sweep_results)
        
        # Example 4: Distributed experiment
        distributed_result = None
        if not args.skip_distributed:
            distributed_result = run_distributed_experiment_example()
            
        # Example 5: Comprehensive reporting
        final_report = generate_comprehensive_report(
            sweep_results, protocol_stats, validation_report, distributed_result
        )
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print(f"Results saved in: {output_dir.absolute()}")
        print("\nResearch Productivity Benefits:")
        print("  ✓ Automated parameter exploration")
        print("  ✓ Multi-protocol performance comparison")
        print("  ✓ Systematic result validation")
        print("  ✓ Distributed experiment coordination")
        print("  ✓ Comprehensive result reporting")
        
    except Exception as e:
        print(f"\nExample failed with error: {str(e)}")
        return 1
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        
    return 0

if __name__ == "__main__":
    exit(main())