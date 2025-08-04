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
Research Productivity Tool: Parameter Sweep Automation for GPU-MPC

This module provides automated parameter sweeping capabilities for MPC experiments,
allowing researchers to systematically evaluate protocols across different configurations.
"""

import json
import csv
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import itertools
import copy

@dataclass 
class ParameterRange:
    """Represents a range of parameter values to sweep."""
    name: str
    values: List[Any]
    description: str = ""

@dataclass
class ExperimentResult:
    """Container for experiment results."""
    parameters: Dict[str, Any]
    success: bool
    execution_time: float
    communication_bytes: Optional[int] = None
    error_message: str = ""
    output_files: List[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class ParameterSweep:
    """
    Automates running experiments with different parameter combinations.
    
    Supports both MPC benchmark tasks and higher-level protocol experiments
    (Orca/SIGMA) with configurable parameter ranges and result collection.
    """
    
    def __init__(self, base_config: Dict[str, Any], output_dir: str = "parameter_sweep_results"):
        """
        Initialize parameter sweep.
        
        Args:
            base_config: Base configuration dictionary
            output_dir: Directory to save results
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parameter_ranges: List[ParameterRange] = []
        self.results: List[ExperimentResult] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "parameter_sweep.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def add_parameter_range(self, name: str, values: List[Any], description: str = ""):
        """
        Add a parameter range to sweep.
        
        Args:
            name: Parameter name (dot-notation supported, e.g., 'mpc.bit_width')
            values: List of values to test
            description: Optional description of the parameter
        """
        param_range = ParameterRange(name, values, description)
        self.parameter_ranges.append(param_range)
        self.logger.info(f"Added parameter range: {name} with {len(values)} values")
        
    def add_bit_width_range(self, bit_widths: List[int]):
        """Convenience method to add bit width parameter range."""
        self.add_parameter_range("bit_width", bit_widths, "Input bit width for MPC operations")
        
    def add_party_config_range(self, party_configs: List[Dict[str, Any]]):
        """Convenience method to add party configuration range.""" 
        self.add_parameter_range("party_config", party_configs, "Party configuration settings")
        
    def add_model_range(self, models: List[str]):
        """Convenience method to add model range for SIGMA experiments."""
        self.add_parameter_range("model", models, "Neural network models to evaluate")
        
    def add_sequence_length_range(self, seq_lengths: List[int]):
        """Convenience method to add sequence length range."""
        self.add_parameter_range("sequence_length", seq_lengths, "Input sequence lengths")
        
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from the defined ranges.
        
        Returns:
            List of parameter dictionaries
        """
        if not self.parameter_ranges:
            return [{}]
            
        # Extract parameter names and value lists
        param_names = [pr.name for pr in self.parameter_ranges]
        param_values = [pr.values for pr in self.parameter_ranges]
        
        # Generate Cartesian product
        combinations = []
        for value_combo in itertools.product(*param_values):
            combo_dict = dict(zip(param_names, value_combo))
            combinations.append(combo_dict)
            
        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
        
    def _apply_parameters_to_config(self, base_config: Dict[str, Any], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter values to base configuration.
        
        Args:
            base_config: Base configuration dictionary
            parameters: Parameter values to apply
            
        Returns:
            Updated configuration dictionary
        """
        config = copy.deepcopy(base_config)
        
        for param_name, param_value in parameters.items():
            # Support dot notation for nested parameters
            keys = param_name.split('.')
            current = config
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
                
            # Set the final value
            current[keys[-1]] = param_value
            
        return config
        
    def _build_command(self, config: Dict[str, Any]) -> str:
        """
        Build command string from configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Command string to execute
        """
        experiment_type = config.get('experiment_type', 'mpc_benchmark')
        
        if experiment_type == 'mpc_benchmark':
            return self._build_mpc_benchmark_command(config)
        elif experiment_type == 'orca':
            return self._build_orca_command(config)
        elif experiment_type == 'sigma':
            return self._build_sigma_command(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
            
    def _build_mpc_benchmark_command(self, config: Dict[str, Any]) -> str:
        """Build MPC benchmark command."""
        cmd_parts = [config.get('executable', './build/benchmarks/mpc_benchmark')]
        
        # Add standard MPC benchmark arguments
        if 'task' in config:
            cmd_parts.extend(['--task', str(config['task'])])
        if 'party' in config:
            cmd_parts.extend(['--party', str(config['party'])])
        if 'peer' in config:
            cmd_parts.extend(['--peer', str(config['peer'])])
        if 'threads' in config:
            cmd_parts.extend(['--threads', str(config['threads'])])
        if 'bit_width' in config:
            cmd_parts.extend(['--bin', str(config['bit_width'])])
            
        return ' '.join(cmd_parts)
        
    def _build_orca_command(self, config: Dict[str, Any]) -> str:
        """Build Orca experiment command."""
        experiment = config.get('experiment', 'CNN2')
        party = config.get('party', 0)
        gpu = config.get('gpu', 0)
        key_dir = config.get('key_dir', './keys/')
        
        if config.get('role') == 'dealer':
            return f"CUDA_VISIBLE_DEVICES={gpu} ./orca_dealer {party} {experiment} {key_dir}"
        else:
            peer_ip = config.get('peer_ip', 'localhost')
            return f"CUDA_VISIBLE_DEVICES={gpu} ./orca_evaluator {party} {peer_ip} {experiment} {key_dir}"
            
    def _build_sigma_command(self, config: Dict[str, Any]) -> str:
        """Build SIGMA experiment command."""
        model = config.get('model', 'gpt2')
        sequence_length = config.get('sequence_length', 128)
        party = config.get('party', 0)
        peer_ip = config.get('peer_ip', 'localhost')
        threads = config.get('threads', 64)
        gpu = config.get('gpu', 0)
        
        return f"CUDA_VISIBLE_DEVICES={gpu} ./sigma {model} {sequence_length} {party} {peer_ip} {threads}"
        
    def _execute_experiment(self, config: Dict[str, Any], parameters: Dict[str, Any]) -> ExperimentResult:
        """
        Execute a single experiment configuration.
        
        Args:
            config: Experiment configuration
            parameters: Parameter values for this run
            
        Returns:
            ExperimentResult containing execution results
        """
        start_time = time.time()
        
        try:
            # Build command
            command = self._build_command(config)
            self.logger.info(f"Executing: {command}")
            
            # Create output directory for this parameter combination
            param_str = "_".join([f"{k}={v}" for k, v in parameters.items()])
            param_dir = self.output_dir / "runs" / param_str
            param_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute command
            with open(param_dir / "stdout.log", "w") as stdout_file, \
                 open(param_dir / "stderr.log", "w") as stderr_file:
                
                result = subprocess.run(
                    command,
                    shell=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd=config.get('working_dir', '.'),
                    timeout=config.get('timeout', 3600)  # 1 hour default timeout
                )
                
            execution_time = time.time() - start_time
            
            # Collect output files
            output_files = []
            if param_dir.exists():
                output_files = [str(f) for f in param_dir.glob("*") if f.is_file()]
                
            # Parse results if successful
            comm_bytes = None
            if result.returncode == 0:
                comm_bytes = self._parse_communication_stats(param_dir)
                
            return ExperimentResult(
                parameters=parameters,
                success=(result.returncode == 0),
                execution_time=execution_time,
                communication_bytes=comm_bytes,
                error_message="" if result.returncode == 0 else f"Process exited with code {result.returncode}",
                output_files=output_files
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ExperimentResult(
                parameters=parameters,
                success=False,
                execution_time=execution_time,
                error_message="Experiment timed out"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExperimentResult(
                parameters=parameters,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def _parse_communication_stats(self, output_dir: Path) -> Optional[int]:
        """
        Parse communication statistics from experiment output.
        
        Args:
            output_dir: Directory containing experiment output
            
        Returns:
            Communication bytes if found, None otherwise
        """
        try:
            # Try to find communication stats in various output files
            stat_files = ['stdout.log', 'stderr.log', 'dealer.txt', 'evaluator.txt']
            
            for stat_file in stat_files:
                file_path = output_dir / stat_file
                if file_path.exists():
                    content = file_path.read_text()
                    
                    # Look for common communication reporting patterns
                    if 'Communication:' in content:
                        for line in content.split('\n'):
                            if 'Communication:' in line:
                                # Extract number (assuming format like "Communication: 1234567 bytes")
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part.isdigit():
                                        return int(part)
                                        
        except Exception as e:
            self.logger.warning(f"Failed to parse communication stats: {e}")
            
        return None
        
    def run_sweep(self, max_parallel: int = 1, resume: bool = True) -> List[ExperimentResult]:
        """
        Execute parameter sweep.
        
        Args:
            max_parallel: Maximum number of parallel experiments (currently only supports 1)
            resume: Whether to resume from previous run
            
        Returns:
            List of experiment results
        """
        self.logger.info("Starting parameter sweep")
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations()
        
        # Load previous results if resuming
        if resume:
            self._load_previous_results()
            
        # Filter out already completed combinations
        completed_params = {frozenset(r.parameters.items()) for r in self.results if r.success}
        remaining_combinations = [
            combo for combo in combinations
            if frozenset(combo.items()) not in completed_params
        ]
        
        self.logger.info(f"Running {len(remaining_combinations)} experiments "
                        f"({len(combinations) - len(remaining_combinations)} already completed)")
        
        # Execute experiments
        for i, parameters in enumerate(remaining_combinations):
            self.logger.info(f"Running experiment {i+1}/{len(remaining_combinations)}: {parameters}")
            
            # Apply parameters to base config
            config = self._apply_parameters_to_config(self.base_config, parameters)
            
            # Execute experiment
            result = self._execute_experiment(config, parameters)
            self.results.append(result)
            
            # Save intermediate results
            if (i + 1) % 10 == 0 or i == len(remaining_combinations) - 1:
                self.save_results()
                
            if result.success:
                self.logger.info(f"Experiment completed successfully in {result.execution_time:.2f}s")
            else:
                self.logger.error(f"Experiment failed: {result.error_message}")
                
        self.logger.info("Parameter sweep completed")
        return self.results
        
    def _load_previous_results(self):
        """Load results from previous runs."""
        results_file = self.output_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    self.results = [ExperimentResult(**r) for r in data]
                    self.logger.info(f"Loaded {len(self.results)} previous results")
            except Exception as e:
                self.logger.warning(f"Failed to load previous results: {e}")
                
    def save_results(self):
        """Save results to JSON and CSV files."""
        # Save to JSON
        json_file = self.output_dir / "results.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
            
        # Save to CSV
        csv_file = self.output_dir / "results.csv"
        if self.results:
            with open(csv_file, 'w', newline='') as f:
                # Collect all parameter names
                all_params = set()
                for result in self.results:
                    all_params.update(result.parameters.keys())
                all_params = sorted(all_params)
                
                # Write CSV header
                fieldnames = all_params + ['success', 'execution_time', 'communication_bytes', 'error_message', 'timestamp']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write results
                for result in self.results:
                    row = result.parameters.copy()
                    row.update({
                        'success': result.success,
                        'execution_time': result.execution_time,
                        'communication_bytes': result.communication_bytes,
                        'error_message': result.error_message,
                        'timestamp': result.timestamp
                    })
                    writer.writerow(row)
                    
        self.logger.info(f"Results saved to {json_file} and {csv_file}")
        
    def get_successful_results(self) -> List[ExperimentResult]:
        """Get only successful experiment results."""
        return [r for r in self.results if r.success]
        
    def get_failed_results(self) -> List[ExperimentResult]:
        """Get only failed experiment results."""
        return [r for r in self.results if not r.success]
        
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        successful = self.get_successful_results()
        failed = self.get_failed_results()
        
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful),
            'failed_experiments': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'total_execution_time': sum(r.execution_time for r in self.results),
            'average_execution_time': sum(r.execution_time for r in successful) / len(successful) if successful else 0
        }
        
        if successful:
            execution_times = [r.execution_time for r in successful]
            summary.update({
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'median_execution_time': sorted(execution_times)[len(execution_times)//2]
            })
            
        # Communication statistics
        comm_results = [r for r in successful if r.communication_bytes is not None]
        if comm_results:
            comm_bytes = [r.communication_bytes for r in comm_results]
            summary.update({
                'min_communication_bytes': min(comm_bytes),
                'max_communication_bytes': max(comm_bytes),
                'average_communication_bytes': sum(comm_bytes) / len(comm_bytes)
            })
            
        return summary


if __name__ == "__main__":
    # Example usage
    base_config = {
        'experiment_type': 'mpc_benchmark',
        'executable': './build/benchmarks/mpc_benchmark',
        'party': 0,
        'peer': 'localhost',
        'threads': 4,
        'timeout': 1800  # 30 minutes
    }
    
    sweep = ParameterSweep(base_config, "example_sweep_results")
    
    # Add parameter ranges
    sweep.add_parameter_range('task', ['dcf', 'scmp', 'twomax'])
    sweep.add_bit_width_range([32, 48, 64])
    sweep.add_parameter_range('threads', [1, 2, 4, 8])
    
    # Run sweep
    results = sweep.run_sweep()
    
    # Generate summary
    summary = sweep.generate_summary_stats()
    print(f"Completed {summary['successful_experiments']}/{summary['total_experiments']} experiments")
    print(f"Average execution time: {summary['average_execution_time']:.2f}s")