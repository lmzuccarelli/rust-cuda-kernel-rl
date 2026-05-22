# Overview

A complete cuda kernel reinforcement workflow written in Rust

This heavily based on [KernelBench](https://github.com/ScalingIntelligence/KernelBench)

Used the project [KernelBlaster](https://github.com/NVlabs/KernelBlaster) as inspiration for this project

## Usage

clone this repo

```
git clone https://github.com/lmzuccarelli/rust-cuda_kernel-rl

cd rust-cuda-kernel-rl
```

### Build binary

This assumes you have installed Rust


```
make build
```

## Prerequisites and dependencies(cuda13.2 specific)

- python
- libtorch
- cuda 13.2 and cuda 13.2 driver
- nvshmem-cuda-13
- cusparselt
- libnccl-2.30.3-1+cuda13.2 
- libnccl-devel-2.30.3-1+cuda13.2 
- libnccl-static-2.30.3-1+cuda13.2
- cudnn9-cuda-13
- nvidia nsights (ncu)


## Workflow

There are basically 3 type of endpoint servers

- controller
- compiler
- gpu

The controller can be executed on any server (no gpu required)
The compiler does not need a gpu but needs nvcc to be installed
The gpu service needs a gpu, this is where the compiled binary will be executed and profiled through ncu (nvidia nsights)

- The user will initiate a workflow via the controller api using a json payload to indicate the kernel/s to be profiled.
- The controller will then pass on the info to the compiler process and compile the cuda-kernel/s on success it will then ask the gpu service to execute the kernel to ensure it executes correctly, the seconf step is then to get
a simple baseline (executed only once for each cuda kernel). We use 'Elapsed Cycles' as the main profiling unit, it's also used towards our rewards functionality.
- The controller will then assemble a prompt for the LLM using the cuda kernel code, the results from the ncu insights (referencing the Speed of Light results), it uses a known database of problems and recommended optimisations.
- The controller then fires the prompt and askf for resultant optimized cuda kernel code. This process is repeated until the set amount of trajectories is reached.
- The trajectory with the highest (lowest Elapsed Cycles as a percentage from the baseline) score will be saved together with the releavnt cuda-kernel optimized code.

