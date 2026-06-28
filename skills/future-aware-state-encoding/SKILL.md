---
name: future-aware-state-encoding
description: Use when an RL environment has delayed effects, travel times, or predictable future state, and the policy needs to reason about when actions will land or what the world will look like after a sequence of events.
---

# Future-Aware State Encoding

## Overview

Two complementary techniques for giving an RL policy visibility into future state without simulating entities as full tokens:

1. **Countdown bins** — Encode future events as per-entity countdown arrays that shift each timestep
2. **Ceasefire forecasts** — Simulate state evolution under a "no new actions" assumption, encode as auxiliary features

**Core principle:** The policy doesn't need to track individual in-flight entities. It needs to know, for each entity: "what will arrive here, and when?" The countdown bins answer this directly. The ceasefire forecast extends this to: "what will this entity's state be in N turns if I do nothing?"

## When to Use

**Countdown bins when:**
- Actions have delayed effects (travel time, build time, cooldown)
- The delay horizon is bounded and known (e.g., max 24 turns)
- Simulating in-flight entities as full tokens would be expensive
- You need per-entity, per-timestep arrival information

**Ceasefire forecasts when:**
- The policy makes poor timing decisions (launching too early or too late)
- Future state is predictable from current state (production, countdown arrivals)
- The policy needs to evaluate "what happens after my action lands?"
- You have spare compute for a lightweight forward simulation

**Don't use when:**
- Actions have immediate effects (no travel/cooldown)
- The future is unpredictable (stochastic transitions dominate)
- You can afford to simulate entities as full tokens
- The state space is small enough that a few frames of history suffice

## Pattern 1: Countdown Bins

### The Problem

Your environment has in-flight entities (missiles, fleets, packages, packets) traveling between locations. Simulating each one as a token in your entity-transformer is expensive — there could be hundreds. But the policy needs to know what's arriving where and when.

### The Solution

Replace in-flight entity tokens with per-destination countdown arrays:

```
Instead of:  Fleet tokens [F₁, F₂, F₃, ...] in the sequence
Use:         Per-planet arrays: incoming[planet, turn_bin] = net_effect
```

### Mechanism

Each destination entity has a fixed-size countdown buffer (e.g., 24 bins for the next 24 turns):

```python
# Environment: launching an entity writes to the destination's countdown
incoming[target, floor(eta)] += launched_ships

# Environment: each timestep, shift bins left by one
incoming[:, :-1] = incoming[:, 1:]
incoming[:, -1] = 0  # clear farthest bin
```

The policy sees the countdown as per-entity features:

```python
features[entity_idx, 8:8+MAX_BINS] = incoming[entity_idx, :] / SCALE
```

### Multi-Agent: Resolving Conflicts in Features

When multiple agents can send entities to the same destination, resolve the interaction before encoding:

```python
def resolve_incoming(incoming_per_agent):
    """For each (destination, arrival_turn), resolve multi-agent conflicts."""
    # Sort all incoming fleets to this destination at this turn by size
    top, second = top_two(incoming_per_agent)
    if top == second:      # mutual annihilation
        return 0
    else:                  # largest survives with (top - second) ships
        return sign(top.owner == ego) * (top.ships - second.ships) / SCALE
```

The sign encodes whether the survivor is friendly (+) or hostile (−). This single scalar per (entity, turn) replaces tracking every individual fleet.

### Multi-Agent: Adding Owner Identity

When knowing the net effect isn't enough (e.g., 3+ players where "who is arriving" matters), add a one-hot encoding:

```python
# For each (planet, turn_bin): encode survivor owner as one-hot
survivor_one_hot = one_hot(survivor_owner, num_players + 1)  # +1 for neutral
features[entity_idx, NET_END:NET_END + BINS * SLOTS] = survivor_one_hot.flatten()
```

This is needed when the policy cares about which opponent is attacking — useful for diplomacy, targeting priority, and threat assessment in games with >2 players.

### Benefits

| Aspect | Fleet tokens | Countdown bins |
|--------|-------------|----------------|
| Token count | Variable, up to hundreds | Fixed, zero extra tokens |
| Attention cost | O(L²) grows with fleets | O(1) per entity — just features |
| Information | Full fleet state (position, velocity) | Arrival time + net effect only |
| Missing | — | Fleet position mid-flight (usually unnecessary) |
| Memory | Per-fleet state tracking | Fixed-size per-entity array |

## Pattern 2: Ceasefire Forecast Features

### The Problem

The policy launches a fleet at turn T. It arrives at turn T + ETA. The policy needs to evaluate: "will this be a good attack when it lands?" But the target's state at T + ETA depends on production, other incoming fleets, and combat — all predictable from current state. Without this forecast, the policy sees only current garrison, leading to attacks that land too early (target hasn't grown) or too late (target already reinforced).

### The Solution

Simulate the ceasefire future — what happens if nobody launches anything new — and encode it as auxiliary features:

```python
def simulate_ceasefire_future(current_state, num_ticks=24):
    """Simulate state for next N ticks under no-new-actions assumption."""
    garrison = current_garrison
    owner = current_owner
    future = []
    for t in range(num_ticks):
        # Production: owned planets grow
        is_owned = owner > NEUTRAL
        garrison = where(is_owned, garrison + production, garrison)
        
        # Resolve incoming arrivals for this tick
        for each entity:
            arr_owner = incoming_owner[entity, t]
            arr_ships = incoming_ships[entity, t]
            garrison, owner = resolve_combat(garrison, owner, arr_owner, arr_ships)
        
        # Encode this tick's state
        future.append(cat([one_hot(owner, 5), garrison / 1000]))
    
    return cat(future)  # [num_ticks × features_per_tick]
```

### Two Ways to Use the Forecast

**1. As a trunk feature** — Add to every entity's embedding:

```python
future_features = simulate_ceasefire_future(state)
entity_embedding += future_proj(future_features)  # Linear(144, d_model)
```

This gives every transformer layer access to the future projection. Attention can compare "how will entity A and entity B evolve over the next 24 turns?"

**2. As target-head context** — Give the action head that evaluates specific action outcomes both the origin's and target's future:

```python
# When scoring "launch from origin to target with fleet_size arriving at eta":
origin_future = simulate_ceasefire_future(origin_state_with_ships_removed)
target_future = simulate_ceasefire_future(target_state_with_fleet_arriving_at_eta)
target_score = target_head(cat([target_hidden, fleet_context, origin_future, target_future]))
```

The target head now sees: "here's what happens to the origin after I remove these ships, and here's what happens to the target when my fleet arrives and fights whatever was already arriving."

### Injecting the Candidate Action

When computing the forecast for scoring an action, inject the candidate action's effects into the simulation:

```python
def target_future_with_launch(state, launch_ships, launch_eta):
    """Simulate target future as if my fleet arrives at launch_eta."""
    for t in range(num_ticks):
        # ... normal production and incoming resolution ...
        if t == launch_eta:
            # Inject my fleet as an additional arrival
            resolve_combat(garrison, owner, MY_OWNER, launch_ships)
    return future
```

This lets the target head accurately evaluate "will I capture this planet?" accounting for all other arriving fleets, production, and existing garrison.

### When This Matters Most

Ceasefire forecasts are most impactful when:
- **Production compounds over time** — a planet with 10 ships now might have 50 in 10 turns
- **Multiple fleets are already in-flight** — the target might be captured by someone else before your fleet arrives
- **Timing precision matters** — launching one turn too early sends too few ships; one turn too late and the target is gone
- **Opportunity cost matters** — removing ships from origin affects the origin's ability to defend or launch again

## Implementation Checklist

### Countdown bins
- [ ] Define the maximum horizon (how many bins?)
- [ ] Implement bin shifting in the environment step function
- [ ] Encode net effect per (entity, bin) as features (signed scalar, normalized)
- [ ] For 3+ agents: add survivor owner identity encoding
- [ ] Resolve multi-agent conflicts before encoding (largest-vs-second, or full resolution)
- [ ] Ensure launch writes to correct bin: `floor(eta)`, clamped to horizon

### Ceasefire forecasts
- [ ] Implement lightweight tick-by-tick simulation (production, arrivals, combat)
- [ ] Support injecting candidate actions into the forecast
- [ ] Decide: trunk feature only, target-head only, or both?
- [ ] Keep simulation in the model's framework (PyTorch/JAX) for gradient flow
- [ ] Mask inactive entities in forecast output
- [ ] Cache or vectorize — the simulation runs every forward pass

## Concrete Example: Orbit Wars

**Countdown bins:** 24-turn horizon. Fleets launched to a target write to `incoming_fleets[agent, target, floor(eta)]`. Each planet's observation includes a 24-element signed net effect (positive = friendly survivor, negative = hostile). In 4-player mode, an additional 24×5 = 120 one-hot dimensions encode survivor owner identity.

**Ceasefire forecast:** 24-tick simulation of garrison + owner evolution under no-new-launches. Used both as a trunk feature (projected via `future_feat_proj` into d_model) and as target-head context (origin future + target future with the candidate fleet injected at `t = eta`). This fixed the problem of policies launching attacks slightly too early, before the target had grown enough ships to be worth capturing.

**Result:** Without future features, the policy struggled with timing — launching too early (target hasn't produced enough value) or too late (target already captured). With them, launch timing became precise, and the target head could accurately rank destinations by post-arrival value.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Counting bins from 1 instead of 0 | `floor(eta)` should map to bin 0 for eta < 1.0. Verify off-by-one. |
| Not shifting bins every timestep | Bins must shift every env step. A frozen countdown gives stale info. |
| Encoding raw incoming fleets without conflict resolution | Two agents sending 100 ships each ≠ 200 arriving. Resolve combat first. |
| Ceasefire simulation too expensive for real-time use | Vectorize over entities. Cache results within a macro-turn (state unchanged). |
| Forgetting to mask inactive entities in forecast | Inactive entities should output zero future features, not stale data. |
| Using ceasefire forecast but never injecting the candidate action | Target head needs to see the effect of its own action, not just baseline future. |
| Sign ambiguity in net effect | Document clearly: positive = friendly, negative = hostile, zero = neutral/annihilated. |
