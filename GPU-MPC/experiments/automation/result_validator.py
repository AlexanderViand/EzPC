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
Research Productivity Tool: MPC Result Validation Framework for GPU-MPC

This module provides comprehensive validation capabilities for MPC experiment results,
including correctness verification, statistical analysis, and error reporting.
"""

import json
import csv
import logging
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import math

@dataclass
class ValidationResult:
    """Container for validation results."""
    test_name: str
    passed: bool
    error_message: str = ""
    expected_value: Any = None
    actual_value: Any = None
    tolerance: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class TimingStats:
    """Container for timing statistics."""
    mean: float
    median: float
    std_dev: float
    min_time: float
    max_time: float
    percentile_95: float
    percentile_99: float
    count: int

@dataclass
class CommunicationStats:
    """Container for communication statistics."""
    total_bytes: int
    total_rounds: int
    avg_bytes_per_round: float
    max_message_size: int
    min_message_size: int

class ResultValidator:
    """
    Comprehensive validation framework for MPC experiment results.
    
    Supports validation of:
    - MPC reconstruction correctness
    - Timing result consistency  
    - Communication efficiency
    - Statistical significance
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        """
        Initialize result validator.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_results: List[ValidationResult] = []
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "validation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def validate_reconstruction(self, share0: Union[int, float], share1: Union[int, float], 
                              expected: Union[int, float, bool], 
                              test_name: str, sharing_type: str = "arithmetic",
                              tolerance: float = 0.0) -> ValidationResult:
        """
        Validate MPC reconstruction result.
        
        Args:
            share0: First party's share
            share1: Second party's share  
            expected: Expected reconstructed value
            test_name: Name of the test
            sharing_type: Type of sharing ("arithmetic", "boolean", "xor")
            tolerance: Tolerance for floating point comparisons
            
        Returns:
            ValidationResult with validation outcome
        """
        try:
            if sharing_type == "arithmetic":
                reconstructed = share0 + share1
            elif sharing_type == "boolean" or sharing_type == "xor":
                reconstructed = bool((share0 ^ share1) & 1)
            else:
                raise ValueError(f"Unknown sharing type: {sharing_type}")
                
            # Compare with tolerance for floating point values
            if isinstance(expected, float) or isinstance(reconstructed, float):
                passed = abs(reconstructed - expected) <= tolerance
            else:
                passed = (reconstructed == expected)
                
            result = ValidationResult(
                test_name=test_name,
                passed=passed,
                expected_value=expected,
                actual_value=reconstructed,
                tolerance=tolerance,
                error_message="" if passed else f"Reconstruction failed: expected {expected}, got {reconstructed}"
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name=test_name, 
                passed=False,
                error_message=f"Validation error: {str(e)}"
            )
            
        self.validation_results.append(result)
        self.logger.info(f"Validation {test_name}: {'PASSED' if result.passed else 'FAILED'}")
        return result
        
    def validate_party_outputs(self, party0_file: str, party1_file: str, 
                             test_name: str, sharing_type: str = "arithmetic") -> List[ValidationResult]:
        """
        Validate outputs from two-party MPC by comparing party output files.
        
        Args:
            party0_file: Path to party 0's output file
            party1_file: Path to party 1's output file
            test_name: Name of the test
            sharing_type: Type of sharing used
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        try:
            # Parse party output files
            party0_data = self._parse_output_file(party0_file)
            party1_data = self._parse_output_file(party1_file)
            
            # Find matching share pairs
            share_pairs = self._match_shares(party0_data, party1_data)
            
            for share_name, (share0, share1, expected) in share_pairs.items():
                result = self.validate_reconstruction(
                    share0, share1, expected, 
                    f"{test_name}_{share_name}", sharing_type
                )
                results.append(result)
                
        except Exception as e:
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=f"Failed to validate party outputs: {str(e)}"
            )
            results.append(result)
            self.validation_results.append(result)
            
        return results
        
    def _parse_output_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse MPC output file to extract shares and expected values.
        
        Args:
            file_path: Path to output file
            
        Returns:
            Dictionary with parsed data
        """
        data = {}
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Output file not found: {file_path}")
            
        content = file_path.read_text()
        
        # Parse JSON output if available
        try:
            json_data = json.loads(content)
            return json_data
        except json.JSONDecodeError:
            pass
            
        # Parse text output with common patterns
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for share patterns like "share_name: value"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # Try to parse as number
                    try:
                        if '.' in value_str:
                            data[key] = float(value_str)
                        else:
                            data[key] = int(value_str)
                    except ValueError:
                        data[key] = value_str
                        
            # Look for result patterns
            result_match = re.search(r'Result:\s*(\d+)', line)
            if result_match:
                data['result'] = int(result_match.group(1))
                
            # Look for timing patterns
            timing_match = re.search(r'Time:\s*([\d.]+)', line)
            if timing_match:
                data['execution_time'] = float(timing_match.group(1))
                
            # Look for communication patterns
            comm_match = re.search(r'Communication:\s*(\d+)', line)
            if comm_match:
                data['communication_bytes'] = int(comm_match.group(1))
                
        return data
        
    def _match_shares(self, party0_data: Dict, party1_data: Dict) -> Dict[str, Tuple[Any, Any, Any]]:
        """
        Match shares between party outputs and determine expected values.
        
        Args:
            party0_data: Party 0's parsed data
            party1_data: Party 1's parsed data
            
        Returns:
            Dictionary mapping share names to (share0, share1, expected) tuples
        """
        share_pairs = {}
        
        # Find common keys that represent shares
        common_keys = set(party0_data.keys()) & set(party1_data.keys())
        
        for key in common_keys:
            if key.startswith('share_') or key.endswith('_share'):
                share0 = party0_data[key]
                share1 = party1_data[key]
                
                # Try to find expected value
                expected_key = key.replace('share_', 'expected_').replace('_share', '_expected')
                expected = party0_data.get(expected_key, party1_data.get(expected_key))
                
                if expected is not None:
                    share_pairs[key] = (share0, share1, expected)
                    
        return share_pairs
        
    def analyze_timing_consistency(self, timing_data: List[float], 
                                 test_name: str, max_cv: float = 0.3) -> ValidationResult:
        """
        Analyze timing results for consistency and outliers.
        
        Args:
            timing_data: List of timing measurements
            test_name: Name of the test
            max_cv: Maximum acceptable coefficient of variation
            
        Returns:
            ValidationResult indicating timing consistency
        """
        if len(timing_data) < 3:
            result = ValidationResult(
                test_name=f"{test_name}_timing",
                passed=False,
                error_message="Insufficient timing data for analysis"
            )
            self.validation_results.append(result)
            return result
            
        try:
            stats = self.compute_timing_stats(timing_data)
            
            # Check coefficient of variation
            cv = stats.std_dev / stats.mean if stats.mean > 0 else float('inf')
            
            # Check for outliers (values beyond 3 standard deviations)
            outliers = [t for t in timing_data if abs(t - stats.mean) > 3 * stats.std_dev]
            
            passed = cv <= max_cv and len(outliers) <= len(timing_data) * 0.05  # Allow 5% outliers
            
            error_msg = ""
            if cv > max_cv:
                error_msg += f"High coefficient of variation: {cv:.3f} > {max_cv}. "
            if len(outliers) > len(timing_data) * 0.05:
                error_msg += f"Too many outliers: {len(outliers)}/{len(timing_data)}. "
                
            result = ValidationResult(
                test_name=f"{test_name}_timing",
                passed=passed,
                error_message=error_msg.strip(),
                actual_value={"cv": cv, "outliers": len(outliers), "stats": asdict(stats)}
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name=f"{test_name}_timing",
                passed=False,
                error_message=f"Timing analysis error: {str(e)}"
            )
            
        self.validation_results.append(result)
        return result
        
    def compute_timing_stats(self, timing_data: List[float]) -> TimingStats:
        """
        Compute comprehensive timing statistics.
        
        Args:
            timing_data: List of timing measurements
            
        Returns:
            TimingStats object with computed statistics
        """
        if not timing_data:
            raise ValueError("Empty timing data")
            
        sorted_data = sorted(timing_data)
        
        return TimingStats(
            mean=statistics.mean(timing_data),
            median=statistics.median(timing_data),
            std_dev=statistics.stdev(timing_data) if len(timing_data) > 1 else 0.0,
            min_time=min(timing_data),
            max_time=max(timing_data),
            percentile_95=np.percentile(sorted_data, 95),
            percentile_99=np.percentile(sorted_data, 99),
            count=len(timing_data)
        )
        
    def validate_communication_efficiency(self, comm_data: List[int], 
                                        expected_rounds: int, test_name: str,
                                        max_overhead: float = 0.2) -> ValidationResult:
        """
        Validate communication efficiency metrics.
        
        Args:
            comm_data: List of communication byte counts
            expected_rounds: Expected number of communication rounds
            test_name: Name of the test
            max_overhead: Maximum acceptable overhead ratio
            
        Returns:
            ValidationResult indicating communication efficiency
        """
        try:
            if not comm_data:
                result = ValidationResult(
                    test_name=f"{test_name}_communication",
                    passed=False,
                    error_message="No communication data available"
                )
                self.validation_results.append(result)
                return result
                
            total_bytes = sum(comm_data)
            avg_bytes = total_bytes / len(comm_data)
            
            # Estimate theoretical minimum based on data size
            # This is a rough heuristic - real analysis would need protocol specifics
            estimated_min = expected_rounds * 32  # Assume 32 bytes minimum per round
            overhead_ratio = (total_bytes - estimated_min) / estimated_min if estimated_min > 0 else 0
            
            passed = overhead_ratio <= max_overhead
            
            stats = CommunicationStats(
                total_bytes=total_bytes,
                total_rounds=len(comm_data),
                avg_bytes_per_round=avg_bytes,  
                max_message_size=max(comm_data),
                min_message_size=min(comm_data)
            )
            
            result = ValidationResult(
                test_name=f"{test_name}_communication",
                passed=passed,
                error_message="" if passed else f"High communication overhead: {overhead_ratio:.2%}",
                actual_value=asdict(stats)
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name=f"{test_name}_communication",
                passed=False,
                error_message=f"Communication analysis error: {str(e)}"
            )
            
        self.validation_results.append(result)
        return result
        
    def validate_experiment_directory(self, experiment_dir: str, 
                                    expected_files: List[str] = None) -> List[ValidationResult]:
        """
        Validate experiment output directory structure and files.
        
        Args:
            experiment_dir: Path to experiment directory
            expected_files: List of expected output files
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        exp_path = Path(experiment_dir)
        
        if not exp_path.exists():
            result = ValidationResult(
                test_name="directory_exists",
                passed=False,
                error_message=f"Experiment directory not found: {experiment_dir}"
            )
            results.append(result)
            self.validation_results.append(result)
            return results
            
        # Check for expected files
        if expected_files:
            for expected_file in expected_files:
                file_path = exp_path / expected_file
                passed = file_path.exists()
                
                result = ValidationResult(
                    test_name=f"file_exists_{expected_file}",
                    passed=passed,
                    error_message="" if passed else f"Expected file not found: {expected_file}"
                )
                results.append(result)
                self.validation_results.append(result)
                
        # Check for common output patterns
        output_files = list(exp_path.glob("*.log")) + list(exp_path.glob("*.json")) + list(exp_path.glob("*.txt"))
        
        if not output_files:
            result = ValidationResult(
                test_name="has_output_files",
                passed=False,
                error_message="No output files found in experiment directory"
            )
            results.append(result)
            self.validation_results.append(result)
        else:
            result = ValidationResult(
                test_name="has_output_files",
                passed=True,
                actual_value=len(output_files)
            )
            results.append(result)
            self.validation_results.append(result)
            
        return results
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary containing validation summary and statistics
        """
        if not self.validation_results:
            return {"error": "No validation results available"}
            
        passed_tests = [r for r in self.validation_results if r.passed]
        failed_tests = [r for r in self.validation_results if not r.passed]
        
        report = {
            "summary": {
                "total_tests": len(self.validation_results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.validation_results),
                "generation_time": datetime.now().isoformat()
            },
            "failed_tests": [
                {
                    "test_name": r.test_name,
                    "error_message": r.error_message,
                    "expected": r.expected_value,
                    "actual": r.actual_value
                }
                for r in failed_tests
            ],
            "test_categories": self._categorize_tests()
        }
        
        return report
        
    def _categorize_tests(self) -> Dict[str, Dict[str, int]]:
        """Categorize tests by type for reporting."""
        categories = {
            "reconstruction": {"passed": 0, "failed": 0},
            "timing": {"passed": 0, "failed": 0},
            "communication": {"passed": 0, "failed": 0},
            "files": {"passed": 0, "failed": 0},
            "other": {"passed": 0, "failed": 0}
        }
        
        for result in self.validation_results:
            category = "other"
            if "timing" in result.test_name:
                category = "timing"
            elif "communication" in result.test_name:
                category = "communication"
            elif "file" in result.test_name or "directory" in result.test_name:
                category = "files"
            elif any(x in result.test_name for x in ["reconstruction", "share", "expected"]):
                category = "reconstruction"
                
            if result.passed:
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
                
        return categories
        
    def save_validation_results(self):
        """Save validation results to JSON and CSV files."""
        # Save detailed results to JSON
        json_file = self.output_dir / "validation_results.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in self.validation_results], f, indent=2)
            
        # Save summary to CSV
        csv_file = self.output_dir / "validation_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test_name', 'passed', 'error_message', 'expected_value', 
                           'actual_value', 'tolerance', 'timestamp'])
            
            for result in self.validation_results:
                writer.writerow([
                    result.test_name, result.passed, result.error_message,
                    result.expected_value, result.actual_value, 
                    result.tolerance, result.timestamp
                ])
                
        # Save validation report
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.generate_validation_report(), f, indent=2)
            
        self.logger.info(f"Validation results saved to {self.output_dir}")
        
    def clear_results(self):
        """Clear all validation results."""
        self.validation_results.clear()
        self.logger.info("Validation results cleared")


if __name__ == "__main__":
    # Example usage
    validator = ResultValidator("example_validation")
    
    # Example: Validate DCF reconstruction
    validator.validate_reconstruction(
        share0=12345, share1=54321, expected=True,
        test_name="dcf_comparison", sharing_type="boolean"
    )
    
    # Example: Validate timing consistency  
    timing_data = [1.23, 1.25, 1.21, 1.28, 1.19, 1.31, 1.26]
    validator.analyze_timing_consistency(timing_data, "dcf_benchmark")
    
    # Example: Validate communication efficiency
    comm_data = [2048, 2056, 2044, 2052]
    validator.validate_communication_efficiency(comm_data, 4, "dcf_protocol")
    
    # Generate and save results
    validator.save_validation_results()
    
    report = validator.generate_validation_report()
    print(f"Validation completed: {report['summary']['passed_tests']}/{report['summary']['total_tests']} tests passed")