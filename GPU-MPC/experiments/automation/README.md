# GPU-MPC Research Productivity Tools

This directory contains automation tools designed to improve research productivity for the GPU-MPC codebase. These tools enable systematic parameter exploration, result validation, and experiment orchestration for MPC protocols.

## Tools Overview

### 1. Parameter Sweep (`parameter_sweep.py`)
Automates systematic exploration of protocol parameters to analyze performance characteristics.

**Features:**
- Support for multiple parameter types (bit widths, thread counts, element counts)
- JSON and CSV output for easy analysis
- Resume capability for long-running sweeps
- Integration with existing benchmark executables
- Statistical summaries and reporting

**Example Usage:**
```python
from parameter_sweep import ParameterSweep

# Configure base experiment
base_config = {
    'experiment_type': 'mpc_benchmark',
    'executable': './build/benchmarks/mpc_benchmark',
    'task': 'dcf',
    'party': 0,
    'peer': '127.0.0.1'
}

# Create sweep and add parameter ranges
sweep = ParameterSweep(base_config, "dcf_sweep_results")
sweep.add_bit_width_range([32, 48, 64])
sweep.add_parameter_range('threads', [1, 2, 4, 8])

# Run sweep
results = sweep.run_sweep()
```

### 2. Result Validator (`result_validator.py`)
Comprehensive validation framework for MPC experiment results.

**Features:**
- MPC reconstruction correctness validation
- Timing result consistency analysis
- Communication efficiency metrics
- Statistical analysis of experimental data
- Automated error detection and reporting

**Example Usage:**
```python
from result_validator import ResultValidator

validator = ResultValidator("validation_results")

# Validate MPC reconstruction
validator.validate_reconstruction(
    share0=12345, share1=54321, expected=True,
    test_name="dcf_comparison", sharing_type="boolean"
)

# Analyze timing consistency
timing_data = [1.23, 1.25, 1.21, 1.28, 1.19]
validator.analyze_timing_consistency(timing_data, "dcf_benchmark")

# Generate validation report
report = validator.generate_validation_report()
```

### 3. Experiment Runner (`experiment_runner.py`)
Orchestrates multi-party MPC experiments with automatic process management.

**Features:**
- Synchronized multi-party execution
- Both local and distributed execution support
- Real-time process monitoring
- Automatic error handling and cleanup
- Result collection and aggregation

**Example Usage:**
```python
from experiment_runner import ExperimentRunner, ExperimentConfig, PartyConfig

# Configure parties
party0 = PartyConfig(
    party_id=0, hostname="localhost",
    executable="./build/benchmarks/mpc_benchmark",
    arguments=["--task", "dcf", "--party", "0", "--peer", "127.0.0.1"]
)

party1 = PartyConfig(
    party_id=1, hostname="localhost",
    executable="./build/benchmarks/mpc_benchmark", 
    arguments=["--task", "dcf", "--party", "1", "--peer", "127.0.0.1"]
)

# Run experiment
config = ExperimentConfig("dcf_test", [party0, party1])
runner = ExperimentRunner()
result = runner.run_experiment(config)
```

### 4. Example Script (`example_sweep.py`)
Comprehensive example demonstrating all automation tools.

**Features:**
- DCF protocol parameter sweep
- Multi-protocol performance comparison
- Result validation workflows
- Distributed experiment coordination
- Comprehensive reporting

## Quick Start

1. **Run the complete example:**
   ```bash
   cd experiments/automation
   python example_sweep.py
   ```

2. **Run specific components:**
   ```bash
   # Skip time-consuming sweeps for testing
   python example_sweep.py --skip-sweep --skip-comparison
   
   # Custom output directory
   python example_sweep.py --output-dir my_experiments
   ```

3. **Integrate with existing experiments:**
   ```python
   # Import tools in your experiment scripts
   from automation.parameter_sweep import ParameterSweep
   from automation.result_validator import ResultValidator
   from automation.experiment_runner import ExperimentRunner
   ```

## Research Productivity Benefits

### 1. **Systematic Parameter Exploration**
- Automated testing of parameter combinations
- Consistent experimental methodology
- Statistical significance through multiple runs
- Performance trend analysis

### 2. **Result Validation and Quality Assurance**
- Automated correctness verification
- Statistical consistency checking
- Error detection and reporting
- Reproducibility validation

### 3. **Experiment Orchestration**
- Coordinated multi-party execution
- Distributed computing support
- Automatic resource management
- Standardized result collection

### 4. **Data Analysis and Reporting**
- Structured result formats (JSON, CSV)
- Statistical summaries
- Performance comparisons
- Comprehensive research reports

## Integration with Existing Workflows

### MPC Benchmark Integration
The tools work seamlessly with existing benchmarks:
```bash
# Traditional manual approach
./build/benchmarks/mpc_benchmark --task dcf --party 0 --peer IP --threads 4
./build/benchmarks/mpc_benchmark --task dcf --party 1 --peer IP --threads 4

# Automated approach
python -c "
from automation.experiment_runner import *
# ... configuration ...
runner.run_experiment(config)
"
```

### Orca/SIGMA Integration
Support for high-level protocol experiments:
```python
# Orca experiment configuration
config = {
    'experiment_type': 'orca',
    'role': 'dealer',
    'experiment': 'CNN2',
    'party': 0,
    'gpu': 0
}

sweep = ParameterSweep(config)
sweep.add_parameter_range('experiment', ['CNN2', 'ResNet18', 'VGG16'])
results = sweep.run_sweep()
```

## Output Structure

All tools generate structured outputs for easy analysis:

```
experiment_results/
├── parameter_sweep_results/
│   ├── results.json           # Detailed results
│   ├── results.csv            # Tabular format  
│   ├── parameter_sweep.log    # Execution log
│   └── runs/                  # Individual run outputs
├── validation_results/
│   ├── validation_results.json
│   ├── validation_summary.csv
│   ├── validation_report.json
│   └── validation.log
└── experiment_runner_results/
    ├── experiment_name/
    │   ├── party_0/
    │   │   ├── stdout.log
    │   │   └── stderr.log
    │   └── party_1/
    │       ├── stdout.log
    │       └── stderr.log
    └── experiment_runner.log
```

## Best Practices

### 1. **Parameter Sweep Design**
- Start with small parameter ranges for initial exploration
- Use resume functionality for long-running sweeps
- Include multiple runs for statistical significance
- Document parameter choices and rationale

### 2. **Result Validation**
- Always validate reconstruction correctness
- Check timing consistency across runs
- Monitor communication efficiency
- Set appropriate tolerance levels

### 3. **Experiment Organization**
- Use descriptive experiment names
- Organize results by research question
- Maintain experiment documentation
- Archive important results

### 4. **Resource Management**
- Monitor system resources during sweeps
- Use appropriate timeouts
- Clean up failed experiments
- Balance parallelism with system capacity

## Advanced Usage

### Custom Parameter Types
```python
# Add custom parameter validation
sweep.add_parameter_range('custom_param', [1, 2, 3], 
                         description="Custom experimental parameter")

# Apply parameters with custom logic
def apply_custom_params(config, params):
    if 'custom_param' in params:
        config['special_flag'] = params['custom_param'] > 2
    return config
```

### Distributed Execution
```python
# Configure remote parties
party_remote = PartyConfig(
    party_id=1,
    hostname="remote-server.example.com",
    executable="./build/benchmarks/mpc_benchmark",
    arguments=["--task", "dcf", "--party", "1"]
)

# Automatic SSH execution
config = ExperimentConfig("distributed_dcf", [party_local, party_remote])
result = runner.run_experiment(config)
```

### Custom Validation Rules
```python
# Add domain-specific validation
def validate_dcf_properties(validator, results):
    for result in results:
        # Check DCF-specific properties
        if result.task == 'dcf':
            validator.validate_reconstruction(
                result.share0, result.share1, result.expected,
                f"dcf_{result.id}", "boolean"
            )

validator = ResultValidator()
validate_dcf_properties(validator, experiment_results)
```

## Troubleshooting

### Common Issues
1. **Process hanging:** Check network connectivity and firewall settings
2. **Memory errors:** Reduce parameter ranges or increase system memory
3. **Timeout errors:** Increase timeout values for complex experiments
4. **Permission errors:** Ensure executable permissions on benchmark files

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tools with debug output
sweep = ParameterSweep(config)
# ... detailed logs will be generated
```

## Contributing

To extend the automation tools:

1. **Add new parameter types:** Extend `ParameterRange` class
2. **Add validation rules:** Extend `ResultValidator` class  
3. **Add experiment types:** Extend `ExperimentRunner` command building
4. **Add analysis functions:** Extend reporting capabilities

Example extension:
```python
# Add new experiment type
def _build_custom_command(self, config):
    # Custom command building logic
    return command_string

# Register in ExperimentRunner
ExperimentRunner._build_custom_command = _build_custom_command
```

These tools significantly improve research productivity by automating repetitive tasks, ensuring result quality, and enabling systematic exploration of the MPC protocol design space.