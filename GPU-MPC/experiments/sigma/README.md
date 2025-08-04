
# SIGMA: Secure GPT Inference with Function Secret Sharing

Implementation of protocols from the paper [SIGMA](https://eprint.iacr.org/2023/1269).

**Warning**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Build

This project requires NVIDIA GPUs, and assumes that GPU drivers and the [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/) are already installed. The following has been tested on Ubuntu 20.04 with CUDA 11.7, CMake 3.27.2 and g++-9. 

GPU-MPC requires CMake version >= 3.18 for the modern build system.

To build SIGMA, use one of the following CMake presets from the root directory:

```bash
# Build with SIGMA enabled (~10 min)
cmake --preset test-only
cmake --build build --parallel
```

or manually configure:

```bash
cmake -B build -S . \
    -DGPU_MPC_BUILD_SIGMA=ON \
    -DGPU_MPC_BUILD_TESTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build --parallel
```

Note: `matplotlib` is needed for generating Figure 10. Install it with:
```bash
pip install matplotlib
```

## Run SIGMA

### Prerequisites and caveats

1. Since FSS generates large keys, writing keys to disk and reading keys from disk can take a long time. To ensure that the artifact runs in a reasonable amount of time, we avoid going to disk and instead have the dealer generate keys in CPU memory. These keys are then used by the evaluator. Please make sure that the CPU memory is large enough to support the key size of the model being run. Key sizes can be estimated from Table 9 of the paper.

3. Currently, we only support sequence lengths that are powers-of-2.


### Run standalone

The build produces the `sigma` executable which is in `build/experiments/sigma/`. Each party (the server and the client) needs to run this executable. The executable requires the user to specify the model, sequence length, party number (0 for the server/1 for the client), the IP address of the other party, and the number of CPU threads to use for computation.

The syntax is 
```javascript
./sigma <model name> <sequence length> <party=0/1 (server/client)> <peer IP> <CPU threads>
```

We currently support the following models: `bert-tiny, bert-base, bert-large, gpt2, llama-7b, llama-13b`.

**Example:** To run GPT2, the server will run:
```javascript
./build/experiments/sigma/sigma gpt2 128 0 <client IP> 64
```

The client will run (_on a different machine_):
```javascript
./build/experiments/sigma/sigma gpt2 128 1 <server IP> 64
```

Results are stored in the `output/P<party number>/models/<model name>-<sequence length>/` folder.

### Running the artifact

Before the artifact can be run, we need to configure it via `config.json`. 

For the server(=P0), `config.json` looks like:
```javascript
{
    "P0": {
            "gpu": <The ID of the GPU to use>,
            "peer": <The IP address of the remote peer>,
            "cpu_threads": <The number of CPU threads to use for computation>
    }
}
```

For the client(=P1), `config.json` looks exactly the same, only the arguments are specified under the key "P1".

A sample `config.json` file can be found in the `experiments/sigma` folder.

Once `config.json` has been filled, the script `run_experiment.py` can be used to reproduce the tables and figures in the paper. Here are the relevant options:

```
usage: python run_experiment.py [-h] [--perf true] [--n_seq true] [--all true] --party 0/1

optional arguments:
  --perf true      Generate Tables 3, 5, 9, and Figure 10.
  --n_seq true     Generate Table 8.
  --all true       Run all the experiments.
```

Table 7 can be reproduced by throttling the network bandwidth (with `tc`, for example) and re-running `python run_experiment.py --perf true` to generate Table 5. 

Results are stored in `output/P<party-number>/Table<table-number>.json` or `output/P<party-number>/Fig<figure-number>.png`. 

Log files (which might help with debugging) can be found in the `output/P<party number>/models/<model name>-<sequence length>/logs.txt` file.


## Citation

You can cite the paper using the following BibTeX entry:

```

@inproceedings{sigma,
author = {Kanav Gupta and Neha Jawalkar and Ananta Mukherjee and Nishanth Chandran and Divya Gupta and Ashish Panwar and Rahul Sharma},
year = {2024},
title = {SIGMA: Secure GPT Inference with Function Secret Sharing},
booktitle = {Proc. Priv. Enhancing Technol.}
}

```

