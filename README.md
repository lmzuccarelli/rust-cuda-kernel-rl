# Overview

A complete cuda kernel reinforcement workflow written in Rust

This project is heavily based on [KernelBench](https://github.com/ScalingIntelligence/KernelBench) as well as [KernelBlaster](https://github.com/NVlabs/KernelBlaster) which was used as inspiration

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

- llm (for inferencing)
- compiler
- gpu

The llm service can be executed on any server (no gpu required). Can use openapi type llm service. It currently uses claude with opus4.6

The compiler does not need a gpu but needs nvcc (plus cudnn, cusparselt and other depenedencies) to be installed.

The gpu service needs a gpu, this is where the compiled binary will be executed and profiled through ncu (nvidia nsights).

### Basic Flow 

- The user will initiate a workflow via the various service endpoints (via a config file) using a json payload to indicate the kernel/s to be compiled, executed and profiled.
- The compiler process is used to compile the cuda-kernel/s.
- The gpu process is used to get an initial baseline by executing the initial kernel and then profiling it using ncu for the *Elpased Cycles*.
- The llm porcess will have all the results from the ncu insights (referencing the Speed of Light results), it uses a known database of problems and recommended optimisations with the cuda-kernel code as reference.
- This process is repeated until the set trajectory value is reached (max_rollout) .
- The trajectory with the best score (lowest Elapsed Cycles as a percentage from the baseline) will be saved together with the relevant cuda-kernel optimized code.
- A simple workflow cli is used for the workflow controll.


## Cavaets

### Update the sudo user in the /etc/sudoers file 

Example

```bash
myuser ALL=(ALL) NOPASSWD: ALL
```

This disables asking for a password (ncu needs elevated privileges) refer to [permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag)

### Ensure LD_LIBRARY_PATH envar is set 

Before executing the gpu or compiler services ensure the LD_LIBRARY_PATH envar is set correctly (link to cuda lib64 and CUPTI)

Example 

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-<version>/lib64:/usr/local/cuda/extras/CUPTI/lib64
```

## Troubleshooting

### Manaully test cuda kernels

Use the following cli in the specific "trajectory_XXX/step_XXX/build" directory on the gpu server (base directory is "out")

```bash
# change the GPU_ARCH_VERSION as needed
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch -DCMAKE_BUILD_TYPE=Release .. -DGPU_ARCH_VERSION=86

cmake --build . --config Release
```

