# GPU Batched Environment Simulation

## Overview

Move the entire RL environment simulator — physics, action validation/decoding, and observation encoding — into CUDA kernels. State lives as raw bytes on GPU. Actions from the neural network are decoded and validated directly in CUDA before being applied. Observations are built from raw state on GPU, including multi-step future simulation (forecast ledger) for richer features.

## When to Use

- Building RL training that needs thousands of parallel environments
- Current Python env is the bottleneck (not the neural network)
- Want to eliminate CPU↔GPU transfers in the training loop
- Need action validation/decoding that's more complex than simple bounds checking
- Want observation features that require forward simulation

## Core Architecture

```
GPU Memory (persistent):
├── states_u8: uint8 tensor [E, sizeof(EnvState)]  ← raw bytes, zero-copy
├── actions:   int32 tensor [E, A, B, 3]           ← from NN, decoded on GPU
├── rewards:   float32 tensor [E, A]
├── dones:     bool tensor [E]
└── observations: built on-the-fly, returned to Python

CPU only for:
├── Reset/map generation (OpenMP-parallel, result .to("cuda") once)
├── Training loop orchestration
└── Logging/checkpointing
```

## Key Patterns

### 1. Raw Bytes State (Zero-Copy Struct Access)

The environment state is a single `uint8_t` tensor. In the kernel, cast to the state struct:

```cpp
// types.hpp — POD only, no pointers, no std::vector
struct EnvState {
    int32_t num_agents, step, done;
    float body_x[MAX_BODIES];      // constexpr-sized C arrays
    float body_y[MAX_BODIES];
    int32_t body_owner[MAX_BODIES];
    int32_t body_active[MAX_BODIES];
    // fleets, comets, etc. — all fixed-size arrays
};

// In kernel:
__global__ void StepKernel(uint8_t* states_u8, ...) {
    int env_idx = blockIdx.x;
    EnvState& s = reinterpret_cast<EnvState*>(states_u8)[env_idx];
    // Direct access: s.body_x[i], s.body_owner[i], etc.
}
```

**Why this matters:** Complex envs have 100+ fields. You get all of them in one tensor argument with zero serialization overhead. The state struct MUST be POD (trivially copyable) — no vectors, no pointers, only constexpr-sized arrays.

### 2. Action Decoder on GPU

The NN outputs simple discrete choices. The CUDA kernel decodes them into physical actions with validation:

```cpp
// NN outputs: [launch_flag, target_body_slot, ship_bin]
// CUDA decodes to: {from_planet_id, angle, ships}

__device__ int DecodeAction(const EnvState& s, int player,
                            int from_slot, int to_slot, int ship_bin,
                            float& out_angle, int& out_eta) {
    // 1. Determine ship count from bin
    int ships = ShipsFromBin(s.body_ships[from_slot], ship_bin);

    // 2. Estimate flight angle and ETA (considering target motion)
    if (!EstimateAngleAndEta(s, player, from_slot, to_slot, ships,
                              out_angle, out_eta))
        return 0;  // no valid trajectory

    // 3. Validate: forward-simulate the flight, check collisions
    if (!ValidateLaunch(s, from_slot, to_slot, ships, out_angle, out_eta))
        return 0;  // would hit sun, bounds, or wrong target

    return ships;
}
```

The key insight: validation requires forward simulation (where will the target be? will the fleet hit the sun?), which MUST run on GPU to avoid CPU round-trips per action.

### 3. Forecast Ledger Observation

Beyond current-state features, pre-compute a multi-step future simulation for each player and encode it as observation features:

```cpp
__device__ void BuildForecastLedger(const EnvState& s, int player,
                                     float* ledger_features) {
    // Initialize a lightweight simulation copy
    SimBody bodies[MAX_BODIES];
    SimFleet fleets[MAX_FLEETS];
    InitSim(s, bodies, fleets);

    for (int turn = 0; turn <= FORECAST_HORIZON; ++turn) {
        if (turn > 0) {
            AdvanceSimOneTurn(s, bodies, fleets);
        }

        for (int b = 0; b < num_bodies; ++b) {
            int out = ledger_offset(b, turn);
            ledger[out+0] = bodies[b].active;           // still exists?
            ledger[out+1] = relative_owner(bodies[b]);  // mine/enemy/neutral
            ledger[out+2] = log_norm_ships(bodies[b]);  // ship count
            ledger[out+3] = log_norm(incoming_friendly); // reinforcements
            ledger[out+4] = log_norm(incoming_enemy);
            ledger[out+5] = holds_flag;                  // still control it?
            ledger[out+6] = first_enemy_arrival_turn;
            ledger[out+7] = fall_turn;                   // when lost
        }
    }
}
```

Output shape: `[E, A, B, HORIZON+1, LEDGER_DIM]` — the NN sees not just current state but a 16-step projection of how each body evolves.

### 4. Per-Agent Canonicalization

Rotate coordinates into each player's reference frame before encoding features:

```cpp
__device__ void CanonicalizePosition(float x, float y, int player,
                                      int num_agents, float& ox, float& oy) {
    float theta = PlayerRotation(player, num_agents);  // 0°, 90°, 180°, 270°
    float dx = x - CENTER, dy = y - CENTER;
    ox = CENTER + dx * cosf(theta) + dy * sinf(theta);
    oy = CENTER - dx * sinf(theta) + dy * cosf(theta);
}
```

This makes observations symmetric — each player sees the world from their own perspective, and the NN learns rotation-invariant policies.

### 5. Fused Step + Obs in One Call

Don't make separate C++ calls for step and observation building. Fuse them:

```cpp
std::vector<torch::Tensor> StepCuda(torch::Tensor states_u8,
                                     torch::Tensor actions, ...) {
    int E = states_u8.size(0);

    // Phase 1: Step kernel (mutates states_u8 in-place)
    StepKernel<<<E, 1, 0, stream>>>(states_u8.data_ptr(), actions.data_ptr(),
                                     rewards.data_ptr(), dones.data_ptr(), ...);

    // Phase 2: Observation kernel (reads updated states_u8)
    auto obs = AllocateObs(states_u8);
    dim3 grid(E, MAX_AGENTS);  // 2D grid: env × player
    BuildObsKernel<<<grid, 1, 0, stream>>>(states_u8.data_ptr(),
                                            obs[0].data_ptr(), ...);

    // Return everything in one go
    return {obs_tensors..., rewards, dones, invalid_counts, metrics};
}
```

The Python side makes exactly ONE call per step:
```python
out = _C.step_cuda(self.states, actions, ...)
obs = out[:7]        # body, ledger, global, source_mask, target_mask, ship_bin, dones
rewards = out[7]     # per-agent rewards
dones = out[8]       # terminal flags
invalid = out[9]     # invalid action counts per agent
```

### 6. One-Block-Per-Env Mapping

Simple and effective for heterogeneous env logic:

```cpp
// Step: 1 block per env, thread 0 does the work
StepKernel<<<num_envs, 1>>>();

// Obs: 2D grid — env × player
dim3 grid(num_envs, max_agents);
BuildObsKernel<<<grid, 1>>>();
// blockIdx.x = env index, blockIdx.y = player index
```

This avoids warp-divergence issues from different envs taking different code paths. The grid dimension limit (~65K for dim x) is the only constraint — for >65K envs, use 2D env grid.

## Design Decisions

| Decision | Recommendation | Why |
|----------|---------------|-----|
| State layout | Single `uint8_t` tensor with POD struct | Zero-copy, one argument, all fields accessible |
| Env-to-thread mapping | 1 block = 1 env, thread 0 only | Heterogeneous env logic, no divergence issues |
| Action decoding | On GPU, inside step kernel | Validates physical legality (collision, trajectory) |
| Observation building | On GPU, after step | Forecast ledger requires forward simulation |
| Step + obs fusion | Single extension call | Eliminates one Python→C++ transition |
| CPU role | Reset only (map gen, RNG) | One-time cost, amortized over episode |
| Agent canonicalization | In obs kernel, per-player rotation | Symmetric observations, rotation-invariant policy |
| Forecast horizon | 16 steps typical | Balances feature richness vs compute cost |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Serializing state field-by-field as separate tensor args | Single `uint8_t` state tensor + `reinterpret_cast` |
| Attempting to validate actions on CPU | Decode/validate in CUDA kernel — forward simulation is GPU work |
| Separate Python calls for step() and build_obs() | Fuse in C++: one `step_cuda()` returns everything |
| Storing state as AoS (struct of arrays in each env) | Use SoA within the POD struct for coalesced access where beneficial |
| Forgetting to `cudaDeviceSynchronize()` before reading results | Required before CPU reads any output tensor |
| Using `std::vector` or pointers in device struct | Only constexpr-sized C arrays; POD only |
| Skipping per-agent canonicalization | Each agent needs its own rotated reference frame |

## Reference: Orbit Wars CUDA PPO

This pattern was proven in the Orbit Wars CUDA PPO project. Key metrics:
- **4096 parallel environments** on a single GPU
- **80K lines of CUDA C++** (orbit_cuda.cu) implementing: orbital mechanics, fleet combat, comet spawning, full action decoder with trajectory validation, 16-step forecast ledger per player
- **Fused step+obs in one call** — Python calls `_C.step_cuda()` once per training step
- **Deterministic parity tests** against a pure Python reference implementation
- CPU only handles map generation (OpenMP-parallel reset), everything else on GPU
