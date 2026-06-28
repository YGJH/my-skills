---
name: pytorch-cuda-extension
description: Use when setting up a PyTorch C++/CUDA extension to compile CUDA kernels as a Python module, when moving RL environment logic to GPU, or when the user asks about CUDAExtension, nvcc flags, pybind11 bindings for PyTorch, or fusing env step+obs into a single compiled extension call
---

# PyTorch CUDA Extension

## Overview

Compile C++/CUDA code into a Python-importable `.so` via `torch.utils.cpp_extension.CUDAExtension`. The compiled module can fuse multiple operations (step, obs build, action decode) into a single call, keeping all tensors on GPU with zero host-device copies in the inner loop.

## When to Use

- User wants to move Python env logic to GPU for speed
- User asks about `CUDAExtension`, `BuildExtension`, or `load_inline`
- Setting up `setup.py` with nvcc compilation flags
- Creating pybind11 bindings for CUDA kernels
- Fusing multiple GPU operations into one extension call

## Quick Reference

| Concern | Solution |
|---------|----------|
| Build system | `CUDAExtension` with `BuildExtension` as `cmdclass` |
| nvcc std | `-std=c++17` (not default C++14) |
| Position-independent code | `-Xcompiler=-fPIC` (NOT `--compiler-options '-fPIC'`) |
| Fast math | `--use_fast_math` (replaces div/trig with intrinsics) |
| Profiler line info | `-lineinfo` |
| OpenMP | `-fopenmp` for both cxx and linker |
| Stream | `at::cuda::getCurrentCUDAStream()` |
| Tensor check | `TORCH_CHECK(t.is_cuda(), "msg")` |
| State layout | Raw `uint8_t` bytes, `reinterpret_cast<EnvState*>` for zero-copy access |

## Project Structure

```
project/
├── setup.py                  # CUDAExtension definition
├── cpp/
│   ├── constants.hpp         # Shared constants, dims
│   ├── types.hpp             # State struct (POD, fixed-size arrays)
│   ├── env.hpp               # Public API declarations
│   ├── cpu_impl.cpp          # CPU-side: reset, map gen, RNG (OpenMP)
│   ├── cuda_kernels.cu       # GPU: step, obs, decode, forecast
│   └── bindings.cpp          # pybind11 module definition
├── package_name/
│   ├── __init__.py
│   └── env.py                # Python wrapper calling _C methods
```

## setup.py Template

```python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def _openmp_flags():
    if sys.platform.startswith("linux"):
        return ["-fopenmp"], ["-fopenmp"]
    return [], []

cxx_flags, link_flags = _openmp_flags()

setup(
    name="my_cuda_env",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="my_cuda_env._C",
            sources=[
                "cpp/bindings.cpp",
                "cpp/cpu_impl.cpp",
                "cpp/cuda_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", *cxx_flags],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-lineinfo",
                    "-Xcompiler=-fPIC",
                ],
            },
            extra_link_args=link_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

## Key Patterns

### 1. Zero-Copy State with Raw Bytes

Store the entire environment state as a flat `uint8_t` tensor. Cast to struct in the kernel — no serialization, no field-by-field copying:

```cpp
// types.hpp — POD struct with fixed-size arrays (no vector, no pointer)
struct EnvState {
    int32_t num_agents;
    int32_t step;
    int32_t done;
    float body_x[MAX_BODIES];
    float body_y[MAX_BODIES];
    int32_t body_owner[MAX_BODIES];
    // ... all fields use constexpr-sized arrays
};

// cuda_kernels.cu — reinterpret in kernel
__global__ void StepKernel(uint8_t* states_u8, ...) {
    int env_idx = blockIdx.x;
    EnvState& s = reinterpret_cast<EnvState*>(states_u8)[env_idx];
    // Direct struct field access — zero copy
    s.body_x[0] = new_val;
}
```

Why this matters: Complex envs have 100+ fields. `reinterpret_cast` avoids serializing each field into separate tensor arguments.

### 2. One-Block-Per-Env Grid

Use `blockIdx.x` as the env index. Thread 0 does all the work per env:

```cpp
__global__ void StepKernel(..., int num_envs) {
    int env_idx = blockIdx.x;
    if (env_idx >= num_envs) return;
    if (threadIdx.x != 0) return;  // Single-threaded per env
    // ... env logic ...
}

// Launch: N blocks, 1 thread each
StepKernel<<<num_envs, 1, 0, stream>>>(...);
```

For multi-agent obs: use 2D grid:
```cpp
dim3 grid(num_envs, max_agents);
BuildObsKernel<<<grid, 1, 0, stream>>>(...);
// blockIdx.x = env, blockIdx.y = player
```

### 3. Fused Step + Obs in One Call

Combine step and observation building in a single extension function to avoid a second Python→C++ call:

```cpp
std::vector<torch::Tensor> StepCuda(torch::Tensor states_u8, torch::Tensor actions) {
    // 1. Run step kernel (mutates states_u8 in-place)
    StepKernel<<<E, 1, 0, stream>>>(states_u8.data_ptr<uint8_t>(), ...);
    // 2. Allocate obs tensors
    auto obs = AllocateObs(states_u8);
    // 3. Run obs kernel (reads updated states_u8)
    LaunchBuildObs(states_u8, obs);
    // 4. Return obs + rewards + dones + metrics
    return {obs..., rewards, dones, invalid, metrics};
}
```

### 4. pybind11 Bindings

```cpp
// bindings.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Forward-declare functions defined elsewhere
namespace my_env {
    torch::Tensor ResetCpu(int num_envs, int num_agents, int episode_steps, uint64_t seed);
    std::vector<torch::Tensor> StepCuda(torch::Tensor states, torch::Tensor actions, ...);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reset_cpu", &my_env::ResetCpu, "CPU reset");
    m.def("step_cuda", &my_env::StepCuda, "CUDA step + obs");
    m.attr("MAX_BODIES") = my_env::MAX_BODIES;  // Expose constants
}
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `--compiler-options "'-fPIC'"` | Use `-Xcompiler=-fPIC` |
| Forgetting `-std=c++17` on nvcc | Add `-std=c++17` to nvcc flags |
| Serializing each field as a separate tensor arg | Use single `uint8_t` state tensor + `reinterpret_cast` |
| Thread-per-env grid (`blockIdx.x * blockDim.x + threadIdx.x`) | Use block-per-env (`blockIdx.x` as env index, single thread) |
| Separate `step()` and `build_obs()` calls from Python | Fuse in C++: `StepCuda` does both, returns all outputs |
| Using `std::vector` in device structs | Use constexpr-sized C arrays only |
| Missing `-Xcompiler=-fPIC` | Causes linker errors on Linux |
| `TORCH_CHECK` with wrong dtype | Check `is_cuda()` and dtype before `data_ptr()` |

## Real-World Impact

This pattern (Orbit Wars CUDA PPO) achieves 4096 parallel environments, each with complex physics (orbital mechanics, fleet combat, comet paths, action validation), all running entirely on GPU with step+obs fused in a single extension call. The Python wrapper (`env.py`) only calls `_C.reset_cpu()` and `_C.step_cuda()` — no per-field tensor marshaling.
