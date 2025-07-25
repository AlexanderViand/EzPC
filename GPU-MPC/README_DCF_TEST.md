# DCF Test - Real Network MPC

This directory contains a **real network-based MPC implementation** for DCF (Distributed Comparison Function) using the same infrastructure as the SIGMA protocol.

## Overview

The `make test` target creates a DCF-based comparison tool that performs **actual MPC over the network** between two parties, just like `./sigma` does. This is **NOT** a simulation - it requires two separate processes/machines to run.

## Key Features

✅ **Real Network MPC** - Uses actual network communication between parties  
✅ **Two-phase Protocol** - Offline key generation + Online MPC evaluation  
✅ **Same CLI as Sigma** - Identical command-line interface  
✅ **DCF-based Comparison** - Securely compares two elements using DCF  
✅ **Performance Metrics** - Measures timing and communication costs  

## Usage

### Step 1: Compile
```bash
make test
```

### Step 2: Run Two-Party MPC

**IMPORTANT**: This requires TWO separate terminal sessions or machines!

#### Terminal 1 (Party 0 - Server):
```bash
./test dcf-test 128 0 <client_ip> 64
```

#### Terminal 2 (Party 1 - Client):  
```bash
./test dcf-test 128 1 <server_ip> 64
```

**For localhost testing:**
```bash
# Terminal 1:
./test dcf-test 128 0 127.0.0.1 64

# Terminal 2 (after server starts):
./test dcf-test 128 1 127.0.0.1 64
```

### Command Line Arguments
```
./test <model> <sequence_length> <party=0/1> <peer_ip> <cpu_threads>
```

- `model`: Use "dcf-test" for DCF comparison
- `sequence_length`: Sequence length (e.g., 128) 
- `party`: 0 for server, 1 for client
- `peer_ip`: IP address of the other party
- `cpu_threads`: Number of CPU threads to use

## How It Works

### Phase 1: Key Generation (Offline)
- Each party generates DCF keys independently
- Keys are stored in memory for the online phase
- No network communication required

### Phase 2: Network MPC (Online)  
- Parties establish network connection using LLAMA protocol
- Synchronize before starting computation
- Perform DCF evaluation with network communication
- Generate secret shares of the comparison result

### Test Data
The implementation compares two fixed elements:
- **Element 1**: 42
- **Element 2**: 35  
- **Expected Result**: 42 < 35 = false

## Output

Results are saved to `output/P<party>/`:
- `keygen.txt` - Key generation statistics
- `mpc.txt` - MPC communication and timing statistics

## Network Requirements

- Both parties must be able to reach each other over the network
- No firewall blocking the connection
- For localhost testing: Use 127.0.0.1 as peer IP for both parties

## Troubleshooting

**Program hangs after showing configuration:**
- Make sure both parties are running
- Check network connectivity between parties
- Verify no firewall is blocking the connection

**Connection timeout:**
- Ensure the server (party 0) starts first
- Check that IP addresses are correct
- Try using localhost (127.0.0.1) for testing

## Comparison with Sigma

This DCF test follows the same pattern as the SIGMA protocol:

| Feature | SIGMA | DCF Test |
|---------|-------|----------|
| Network MPC | ✅ | ✅ |
| Two-phase protocol | ✅ | ✅ |
| CLI interface | ✅ | ✅ |
| Performance metrics | ✅ | ✅ |
| Model complexity | High (Transformers) | Simple (DCF) |

The DCF test provides a simpler way to test and understand the MPC infrastructure without the complexity of transformer models. 