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
Research Productivity Tool: Multi-Party Experiment Runner for GPU-MPC

This module provides orchestration capabilities for running multi-party MPC experiments,
including automatic process management, result collection, and distributed execution support.
"""

import json
import subprocess
import logging
import time
import signal
import socket
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
import copy

@dataclass
class PartyConfig:
    """Configuration for a single MPC party."""
    party_id: int
    hostname: str
    port: int = 12345
    gpu_id: int = 0
    executable: str = ""
    arguments: List[str] = None
    working_dir: str = "."
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.arguments is None:
            self.arguments = []
        if self.env_vars is None:
            self.env_vars = {}

@dataclass
class ExperimentConfig:
    """Configuration for a multi-party experiment."""
    experiment_name: str
    parties: List[PartyConfig]
    timeout: int = 3600  # 1 hour default
    max_retries: int = 2
    output_dir: str = "experiment_results"
    collect_party_outputs: bool = True
    sync_start: bool = True
    cleanup_on_failure: bool = True

@dataclass
class ExperimentResult:
    """Result from running a multi-party experiment."""
    experiment_name: str
    success: bool
    execution_time: float
    party_results: Dict[int, Dict[str, Any]]
    error_message: str = ""
    output_files: List[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class ExperimentRunner:
    """
    Orchestrates multi-party MPC experiments with automatic process management.
    
    Supports both local and distributed execution with:
    - Synchronized party startup
    - Real-time output monitoring  
    - Automatic error handling and cleanup
    - Result collection and aggregation
    """
    
    def __init__(self, output_dir: str = "experiment_runner_results"):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.active_processes: Dict[int, subprocess.Popen] = {}
        self.process_threads: Dict[int, threading.Thread] = {}
        self.party_outputs: Dict[int, Dict[str, Any]] = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "experiment_runner.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup_processes()
        
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a multi-party experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult with execution results
        """
        self.logger.info(f"Starting experiment: {config.experiment_name}")
        start_time = time.time()
        
        # Create experiment output directory
        exp_dir = self.output_dir / config.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous state
        self.active_processes.clear()
        self.process_threads.clear()
        self.party_outputs.clear()
        
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Setup party environments
            self._setup_party_environments(config)
            
            # Start parties
            if config.sync_start:
                success = self._start_parties_synchronized(config, exp_dir)
            else:
                success = self._start_parties_sequential(config, exp_dir)
                
            if not success:
                raise RuntimeError("Failed to start all parties")
                
            # Monitor execution
            success = self._monitor_execution(config)
            
            # Collect results
            if config.collect_party_outputs:
                self._collect_party_outputs(exp_dir)
                
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                experiment_name=config.experiment_name,
                success=success,
                execution_time=execution_time,
                party_results=copy.deepcopy(self.party_outputs),
                output_files=self._get_output_files(exp_dir)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Experiment failed: {str(e)}")
            
            if config.cleanup_on_failure:
                self.cleanup_processes()
                
            return ExperimentResult(
                experiment_name=config.experiment_name,
                success=False,
                execution_time=execution_time,
                party_results=copy.deepcopy(self.party_outputs),
                error_message=str(e)
            )
        finally:
            # Always cleanup processes
            self.cleanup_processes()
            
    def _validate_config(self, config: ExperimentConfig):
        """Validate experiment configuration."""
        if len(config.parties) < 2:
            raise ValueError("At least 2 parties required for MPC experiment")
            
        party_ids = {p.party_id for p in config.parties}  
        if len(party_ids) != len(config.parties):
            raise ValueError("Duplicate party IDs found")
            
        for party in config.parties:
            if not party.executable:
                raise ValueError(f"No executable specified for party {party.party_id}")
                
    def _setup_party_environments(self, config: ExperimentConfig):
        """Setup environment variables and working directories for parties."""
        for party in config.parties:
            # Create party-specific output directory
            party_dir = self.output_dir / config.experiment_name / f"party_{party.party_id}"
            party_dir.mkdir(parents=True, exist_ok=True)
            
            # Add party output directory to environment
            party.env_vars['OUTPUT_DIR'] = str(party_dir)
            party.env_vars['PARTY_ID'] = str(party.party_id)
            
            # Set CUDA_VISIBLE_DEVICES if specified
            if party.gpu_id >= 0:
                party.env_vars['CUDA_VISIBLE_DEVICES'] = str(party.gpu_id)
                
    def _start_parties_synchronized(self, config: ExperimentConfig, exp_dir: Path) -> bool:
        """Start all parties simultaneously with synchronization."""
        self.logger.info("Starting parties with synchronization")
        
        # Use threading to start parties simultaneously
        start_threads = []
        start_results = {}
        
        def start_party_thread(party):
            try:
                process = self._start_single_party(party, config, exp_dir)
                start_results[party.party_id] = process
            except Exception as e:
                self.logger.error(f"Failed to start party {party.party_id}: {str(e)}")
                start_results[party.party_id] = None
                
        # Start all parties
        for party in config.parties:
            thread = threading.Thread(target=start_party_thread, args=(party,))
            thread.start()
            start_threads.append(thread)
            
        # Wait for all to complete startup
        for thread in start_threads:
            thread.join(timeout=30)  # 30 second startup timeout
            
        # Check results
        failed_parties = []
        for party_id, process in start_results.items():
            if process is None:
                failed_parties.append(party_id)
            else:
                self.active_processes[party_id] = process
                
        if failed_parties:
            self.logger.error(f"Failed to start parties: {failed_parties}")
            return False
            
        self.logger.info(f"Successfully started {len(self.active_processes)} parties")
        return True
        
    def _start_parties_sequential(self, config: ExperimentConfig, exp_dir: Path) -> bool:
        """Start parties sequentially."""
        self.logger.info("Starting parties sequentially")
        
        for party in config.parties:
            try:
                process = self._start_single_party(party, config, exp_dir)
                self.active_processes[party.party_id] = process
                time.sleep(1)  # Small delay between party starts
            except Exception as e:
                self.logger.error(f"Failed to start party {party.party_id}: {str(e)}")
                return False
                
        self.logger.info(f"Successfully started {len(self.active_processes)} parties")
        return True
        
    def _start_single_party(self, party: PartyConfig, config: ExperimentConfig, exp_dir: Path) -> subprocess.Popen:
        """Start a single party process."""
        self.logger.info(f"Starting party {party.party_id} on {party.hostname}")
        
        # Build command
        cmd_parts = [party.executable] + party.arguments
        
        # Setup environment
        env = dict(os.environ)
        env.update(party.env_vars)
        
        # Setup output files
        party_dir = exp_dir / f"party_{party.party_id}"
        stdout_file = party_dir / "stdout.log"
        stderr_file = party_dir / "stderr.log"
        
        # Start process
        if party.hostname == "localhost" or party.hostname == "127.0.0.1":
            # Local execution
            process = subprocess.Popen(
                cmd_parts,
                stdout=open(stdout_file, 'w'),
                stderr=open(stderr_file, 'w'),
                cwd=party.working_dir,
                env=env,
                preexec_fn=os.setsid  # Create new process group for cleanup
            )
        else:
            # Remote execution via SSH
            ssh_cmd = self._build_ssh_command(party, cmd_parts, env)
            process = subprocess.Popen(
                ssh_cmd,
                stdout=open(stdout_file, 'w'),
                stderr=open(stderr_file, 'w'),
                shell=True,
                preexec_fn=os.setsid
            )
            
        self.logger.info(f"Started party {party.party_id} with PID {process.pid}")
        return process
        
    def _build_ssh_command(self, party: PartyConfig, cmd_parts: List[str], env: Dict[str, str]) -> str:
        """Build SSH command for remote execution."""
        # Build environment variable string
        env_str = " ".join([f"{k}={v}" for k, v in env.items()])
        
        # Build remote command
        remote_cmd = " ".join(cmd_parts)
        
        # Build SSH command
        ssh_cmd = f"ssh {party.hostname} 'cd {party.working_dir} && {env_str} {remote_cmd}'"
        
        return ssh_cmd
        
    def _monitor_execution(self, config: ExperimentConfig) -> bool:
        """Monitor party execution until completion or timeout."""
        self.logger.info(f"Monitoring execution (timeout: {config.timeout}s)")
        
        start_time = time.time()
        check_interval = min(10, config.timeout // 10)  # Check every 10s or 1/10 of timeout
        
        while self.active_processes:
            # Check for timeout
            if time.time() - start_time > config.timeout:
                self.logger.error("Experiment timed out")
                return False
                
            # Check process status
            completed_parties = []
            failed_parties = []
            
            for party_id, process in self.active_processes.items():
                returncode = process.poll()
                if returncode is not None:
                    if returncode == 0:
                        completed_parties.append(party_id)
                        self.logger.info(f"Party {party_id} completed successfully")
                    else:
                        failed_parties.append(party_id)
                        self.logger.error(f"Party {party_id} failed with return code {returncode}")
                        
            # Remove completed/failed processes
            for party_id in completed_parties + failed_parties:
                del self.active_processes[party_id]
                
            # If any party failed, consider experiment failed
            if failed_parties:
                self.logger.error(f"Parties {failed_parties} failed, stopping experiment")
                return False
                
            # If all parties completed, experiment succeeded
            if not self.active_processes:
                self.logger.info("All parties completed successfully")
                return True
                
            # Wait before next check
            time.sleep(check_interval)
            
        return True
        
    def _collect_party_outputs(self, exp_dir: Path):
        """Collect and parse outputs from all parties."""
        self.logger.info("Collecting party outputs")
        
        for party_dir in exp_dir.glob("party_*"):
            party_id = int(party_dir.name.split("_")[1])
            
            party_output = {
                "party_id": party_id,
                "output_dir": str(party_dir),
                "files": []
            }
            
            # Collect output files
            for output_file in party_dir.glob("*"):
                if output_file.is_file():
                    party_output["files"].append(str(output_file))
                    
            # Parse logs for key information
            stdout_file = party_dir / "stdout.log"
            stderr_file = party_dir / "stderr.log"
            
            if stdout_file.exists():
                party_output["stdout"] = self._parse_log_file(stdout_file)
                
            if stderr_file.exists():
                party_output["stderr"] = self._parse_log_file(stderr_file)
                
            self.party_outputs[party_id] = party_output
            
    def _parse_log_file(self, log_file: Path) -> Dict[str, Any]:
        """Parse log file for structured information."""
        parsed_data = {"raw_content": ""}
        
        try:
            content = log_file.read_text()
            parsed_data["raw_content"] = content
            
            # Extract timing information
            timing_matches = re.findall(r'Time:\s*([\d.]+)', content)
            if timing_matches:
                parsed_data["timing"] = [float(t) for t in timing_matches]
                
            # Extract communication information
            comm_matches = re.findall(r'Communication:\s*(\d+)', content)
            if comm_matches:
                parsed_data["communication_bytes"] = [int(c) for c in comm_matches]
                
            # Extract JSON results
            json_matches = re.findall(r'\{[^{}]*\}', content)
            for json_str in json_matches:
                try:
                    json_data = json.loads(json_str)
                    if "results" not in parsed_data:
                        parsed_data["results"] = []
                    parsed_data["results"].append(json_data)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            parsed_data["parse_error"] = str(e)
            
        return parsed_data
        
    def _get_output_files(self, exp_dir: Path) -> List[str]:
        """Get list of all output files from experiment."""
        output_files = []
        for file_path in exp_dir.rglob("*"):
            if file_path.is_file():
                output_files.append(str(file_path))
        return output_files
        
    def cleanup_processes(self):
        """Cleanup all active processes."""
        if not self.active_processes:
            return
            
        self.logger.info("Cleaning up active processes")
        
        # Send SIGTERM first
        for party_id, process in self.active_processes.items():
            try:
                if process.poll() is None:  # Process still running
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    self.logger.info(f"Sent SIGTERM to party {party_id}")
            except Exception as e:
                self.logger.warning(f"Failed to send SIGTERM to party {party_id}: {e}")
                
        # Wait for graceful shutdown
        time.sleep(5)
        
        # Force kill remaining processes
        for party_id, process in self.active_processes.items():
            try:
                if process.poll() is None:  # Process still running
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    self.logger.warning(f"Force killed party {party_id}")
            except Exception as e:
                self.logger.warning(f"Failed to kill party {party_id}: {e}")
                
        self.active_processes.clear()
        
    def run_parameter_sweep(self, base_config: ExperimentConfig, 
                          parameter_combinations: List[Dict[str, Any]]) -> List[ExperimentResult]:
        """
        Run parameter sweep using experiment runner.
        
        Args:
            base_config: Base experiment configuration
            parameter_combinations: List of parameter combinations to test
            
        Returns:
            List of experiment results
        """
        results = []
        
        for i, params in enumerate(parameter_combinations):
            self.logger.info(f"Running parameter combination {i+1}/{len(parameter_combinations)}: {params}")
            
            # Apply parameters to configuration
            config = self._apply_parameters_to_config(base_config, params)
            config.experiment_name = f"{base_config.experiment_name}_param_{i}"
            
            # Run experiment
            result = self.run_experiment(config)
            results.append(result)
            
            if result.success:
                self.logger.info(f"Parameter combination {i+1} completed successfully")
            else:
                self.logger.error(f"Parameter combination {i+1} failed: {result.error_message}")
                
        return results
        
    def _apply_parameters_to_config(self, base_config: ExperimentConfig, 
                                   parameters: Dict[str, Any]) -> ExperimentConfig:
        """Apply parameter values to experiment configuration."""
        config = copy.deepcopy(base_config)
        
        # Apply parameters to party configurations
        for party in config.parties:
            for param_name, param_value in parameters.items():
                if param_name == "threads" and "--threads" not in party.arguments:
                    party.arguments.extend(["--threads", str(param_value)])
                elif param_name == "bit_width" and "--bin" not in party.arguments:
                    party.arguments.extend(["--bin", str(param_value)])
                elif param_name == "task" and "--task" not in party.arguments:
                    party.arguments.extend(["--task", str(param_value)])
                    
        return config
        
    def save_results(self, results: List[ExperimentResult], filename: str = "experiment_results.json"):
        """Save experiment results to file."""
        results_file = self.output_dir / filename
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        self.logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    import os
    import re
    
    # Example usage
    runner = ExperimentRunner("example_experiments")
    
    # Create example configuration for DCF benchmark
    party0 = PartyConfig(
        party_id=0,
        hostname="localhost",
        executable="./build/benchmarks/mpc_benchmark",
        arguments=["--task", "dcf", "--party", "0", "--peer", "127.0.0.1", "--threads", "4"],
        gpu_id=0
    )
    
    party1 = PartyConfig(
        party_id=1,
        hostname="localhost", 
        executable="./build/benchmarks/mpc_benchmark",
        arguments=["--task", "dcf", "--party", "1", "--peer", "127.0.0.1", "--threads", "4"],
        gpu_id=0
    )
    
    config = ExperimentConfig(
        experiment_name="dcf_test",
        parties=[party0, party1],
        timeout=600,  # 10 minutes
        sync_start=True
    )
    
    # Run single experiment
    result = runner.run_experiment(config)
    
    print(f"Experiment completed: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    if not result.success:
        print(f"Error: {result.error_message}")
        
    # Save results
    runner.save_results([result])