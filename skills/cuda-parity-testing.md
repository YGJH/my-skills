# CUDA Parity Testing

## Overview

Verify that CUDA kernels produce identical results to a Python reference by using **deterministic fixture states**, **binary state dumps**, and **full snapshot comparison**. Unlike random-input testing, this approach creates hand-crafted initial states that test specific semantic scenarios, then compares every field of the mutated state between CUDA and a pure Python reimplementation of the same state machine.

## When to Use

- Porting an existing Python env to CUDA and need to verify correctness
- CUDA kernel produces different results than Python reference
- Need deterministic, reproducible tests that catch semantic bugs (not just numerical drift)
- Want tests that survive across CUDA toolkit and GPU architecture changes
- Building a test suite that documents what each kernel scenario covers

## Core Pattern

```
1. Create deterministic fixture state (CPU, known properties)
2. Dump state as structured Python objects (snapshot)
3. Run Python reference step on snapshot → reference result
4. Copy same fixture state to GPU
5. Run CUDA step (mutates GPU state in-place)
6. Dump GPU state back → CUDA result
7. Compare every field: bodies, fleets, comets, rewards, dones, metrics
```

## Key Techniques

### 1. Deterministic Fixture States

Instead of random inputs, create **named, hand-crafted** initial states that each test a specific scenario:

```cpp
// parity_test_utils.cpp
torch::Tensor MakeParityFixtureCpu(const std::string& fixture,
                                    int64_t num_agents,
                                    int64_t episode_steps) {
    if (fixture == "production_rotation") {
        // Two planets at known positions, one rotating, verify production + movement
    } else if (fixture == "fleet_capture_static") {
        // Fleet heading to static planet, verify capture mechanics
    } else if (fixture == "combat_tie") {
        // Equal-strength fleets from two players hit same planet
    } else if (fixture == "comet_spawn") {
        // Comet scheduled to spawn at step 50, verify path advancement
    } else if (fixture == "action_decode_static") {
        // Static target, verify action decoder computes correct angle/ships
    } else if (fixture == "action_decode_rotating_target") {
        // Rotating target, verify future-aware angle estimation
    } else if (fixture == "terminal_score") {
        // End-of-episode scoring scenario
    }
    // ... returns uint8 tensor with EnvState bytes
}
```

**Fixture naming convention:** `[mechanism]_[condition]` — e.g., `fleet_capture_static`, `action_decode_rotating_target`, `neutral_capture_midgame`.

### 2. Binary State Dumps

The extension exposes a dump function that converts raw bytes to structured Python dicts:

```cpp
// bindings.cpp — expose to Python
m.def("dump_states", &DumpStates, "Dump raw EnvState tensors for parity tests");

// parity_test_utils.cpp
pybind11::dict DumpStates(torch::Tensor states_u8) {
    // Cast raw bytes to EnvState structs
    // Convert every field to Python lists
    // Return as nested dict with keys: num_agents, bodies, fleets, comets, ...
}
```

This is superior to comparing output tensors because:
- Catches bugs in fields that aren't part of the normal output (e.g., internal `next_fleet_id`, `comet_path_index`)
- Reveals state corruption that produces correct-looking outputs for wrong reasons
- Enables field-by-field diff for debugging ("fleet 3 angle differs by 0.001")

### 3. Python Reference Reimplementation

Write a **pure Python** version of the state machine that mirrors the CUDA logic exactly:

```python
# tests/parity/python_reference.py
@dataclass
class Snapshot:
    """Full environment state as Python objects."""
    num_agents: int
    episode_steps: int
    step: int
    done: bool
    bodies: list[BodySnapshot]
    fleets: list[FleetSnapshot]
    comets: list[CometGroupSnapshot]

def snapshot_from_dump(dump: dict, env_index: int) -> Snapshot:
    """Convert extension dump to typed Snapshot."""
    ...

def step_reference(before: Snapshot, actions: torch.Tensor
                   ) -> tuple[Snapshot, list[float], bool, list[int]]:
    """Pure Python reimplementation of the CUDA step logic."""
    # Clone the snapshot
    # Apply production, movement, fleet advancement
    # Decode actions (same logic as CUDA DecodeShipsCuda)
    # Resolve combat (same logic as CUDA combat resolution)
    # Check terminal conditions
    # Return (after_snapshot, rewards, done, invalid_counts)
```

The reference must implement the **identical** algorithm as CUDA, not an optimized or simplified version. Use the same formulas, same iteration order, same edge case handling.

### 4. Full Snapshot Comparison

Compare every field individually for clear error messages:

```python
def _assert_snapshot_equal(ref: Snapshot, got: Snapshot, tol=1e-4):
    assert got.num_agents == ref.num_agents
    assert got.step == ref.step
    assert got.done == ref.done
    assert got.next_fleet_id == ref.next_fleet_id  # internal field!

    for i, (rb, gb) in enumerate(zip(ref.bodies, got.bodies)):
        assert gb.id == rb.id, f"body {i} id: {gb.id} != {rb.id}"
        assert gb.owner == rb.owner, f"body {i} owner"
        assert gb.ships == rb.ships, f"body {i} ships: {gb.ships} != {rb.ships}"
        assert gb.active == rb.active, f"body {i} active"
        assert math.isclose(gb.x, rb.x, rel_tol=tol), f"body {i} x"
        assert math.isclose(gb.y, rb.y, rel_tol=tol), f"body {i} y"

    for i, (rf, gf) in enumerate(zip(ref.fleets, got.fleets)):
        assert gf.id == rf.id, f"fleet {i} id"
        assert gf.ships == rf.ships, f"fleet {i} ships"
        assert math.isclose(gf.angle, rf.angle, rel_tol=tol), f"fleet {i} angle"

    for g, (rc, gc) in enumerate(zip(ref.comets, got.comets)):
        assert gc.spawned == rc.spawned, f"comet {g} spawned"
        assert gc.path_index == rc.path_index, f"comet {g} path_index"
        assert gc.body_slots == rc.body_slots, f"comet {g} body_slots"
```

This gives you a **precise** error message like `body 3 x: 45.001 != 45.000` instead of "tensors differ."

### 5. Action Decode Parity

The hardest part to test: the NN outputs `[launch, target_slot, ship_bin]`, but the CUDA kernel decodes this into physical parameters (angle, ships, ETA) using forward simulation. The Python reference must replicate this exactly:

```python
def test_cuda_action_decode_static_target():
    actions = empty_actions()
    actions[0, 0, 0, 0] = 1   # player 0, body slot 0, launch
    actions[0, 0, 0, 1] = 1   # target body slot 1
    actions[0, 0, 0, 2] = 1   # ship bin 1 (50%)

    before = snapshot_from_dump(cpu_fixture)
    ref_after, ref_rewards, ref_done, ref_invalid = step_reference(before, actions)
    cuda_after = snapshot_from_dump(cuda_result)
    _assert_snapshot_equal(ref_after, cuda_after)
    assert ref_rewards == cuda_rewards
    assert ref_invalid == cuda_invalid
```

### 6. Test Matrix: Scenario Coverage

```python
@pytest.mark.parametrize("fixture", [
    "production_rotation",       # basic physics
    "fleet_capture_static",      # combat + capture
    "combat_tie",                # equal-strength edge case
    "fleet_sun_and_bounds",      # boundary deaths
    "comet_spawn",               # dynamic entity lifecycle
    "terminal_score",            # end-of-episode logic
])
def test_cuda_step_matches_python_without_actions(fixture):
    """No-op actions: verify basic physics, movement, production."""

def test_cuda_action_static_target():
    """Action targeting a stationary planet."""

def test_cuda_action_rotating_target():
    """Action targeting a planet in orbit — tests future-aware aiming."""

def test_cuda_action_comet_target():
    """Action targeting an active comet — tests comet path prediction."""

def test_cuda_action_intercepted():
    """Action where fleet hits wrong body before reaching target."""

def test_cuda_action_sun_blocked():
    """Action where trajectory crosses the sun."""

def test_cuda_capture_plus_projected():
    """Capture-plus bin tests future target state prediction."""

def test_invalid_action_counts():
    """Invalid sources, non-owned sources, out-of-bounds targets."""
```

## Test Flow

```
┌──────────────────┐
│ Make fixture (CPU)│──► deterministic EnvState bytes
└────────┬─────────┘
         │
    ┌────▼────┐          ┌──────────────┐
    │ Dump to  │          │ Copy to GPU  │
    │ Snapshot │          │ (cuda_state) │
    └────┬────┘          └──────┬───────┘
         │                      │
    ┌────▼──────────┐    ┌──────▼───────────┐
    │ Python ref    │    │ CUDA step_cuda() │
    │ step(snapshot)│    │ (mutates in-place)│
    └────┬──────────┘    └──────┬───────────┘
         │                      │
    ┌────▼──────────┐    ┌──────▼───────────┐
    │ Ref snapshot  │    │ Dump GPU state   │
    │ (expected)    │    │ (actual)         │
    └────┬──────────┘    └──────┬───────────┘
         │                      │
         └──────────┬───────────┘
                    │
            ┌───────▼────────┐
            │ Field-by-field │
            │  comparison    │
            └────────────────┘
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using random inputs for parity tests | Use deterministic, named fixtures |
| Only comparing output tensors (rewards, dones) | Dump and compare the FULL state struct |
| Python reference uses different algorithms than CUDA | Reimplement the IDENTICAL state machine logic |
| Tolerance too tight for float32 math (expecting bit-exact) | Use `rel_tol=1e-4` for positions, exact for integers |
| Testing only the happy path | Create fixtures for edge cases: ties, sun deaths, interception |
| Not testing invalid action paths | Send invalid actions, verify invalid counts match |
| Skipping internal state fields | Compare `next_fleet_id`, `comet_path_index`, `fleet_remove` too |

## Why Random Inputs Are Insufficient

Random-input tests (generate random state, run both, compare) have fundamental problems:
1. **Coverage is probabilistic** — may never hit specific edge cases (equal-strength combat, sun-crossing trajectory)
2. **Flaky failures** — random state may trigger numerical edge cases that aren't real bugs
3. **Hard to debug** — "random state #47291 diverged at body 17" is useless for root-causing
4. **No semantic meaning** — you don't know WHAT is being tested

Named fixtures give you: deterministic reproducibility, known coverage, meaningful failure messages, and documentation of tested scenarios.

## Reference: Orbit Wars CUDA PPO Parity Suite

The Orbit Wars project uses this exact approach with:
- **6 fixture types** covering physics, combat, comets, scoring
- **5 action-decode fixtures** covering static/rotating/comet targets, interception, sun-blocking
- **1 invalid-action fixture** testing all rejection reasons
- **Full snapshot comparison** comparing 50+ fields across bodies, fleets, and comet groups
- **Pure Python reference** (`tests/parity/python_reference.py`) implementing the identical state machine

All tests are deterministic, run in milliseconds, and catch semantic bugs that random-input testing would miss.
