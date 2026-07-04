# Grounded

**A non-AI, deterministic cognitive engine that learns what it needs, when it needs it — running entirely on your phone.**

No LLM. No subword tokens. No embeddings. No cloud. No training data. Just a baby brain that grows its own understanding by exploring the world through sensors and actions.

---

## The Idea

Every AI model today is a statistical guessing machine — trained on the entire internet, consuming watts by the megawatt, guessing the next token. It's impressive, but it's not *understanding*.

Grounded is the opposite:
- **It knows nothing at birth.** A tiny seed of ~30 foundational concepts (matter, motion, energy, space, time) expressed as **5-dimensional PrimitiveVectors** — Mass, Velocity, Spatial, Valence, Temporal Frequency. New concepts are derived by combining base vectors: Momentum = Speed + Mass, Temperature = Hot + Cold, Texture = Surface + Mass.
- **It wakes up on its own.** The first tick injects activation into SELF — it "notices" its own sensors. After that, a Default Mode Network keeps it thinking even in silence, cycling through favorite concepts and exploring unfamiliar ones.
- **It learns relationally via CCG parsing.** Raw definitions are parsed by a stateless Combinatory Categorial Grammar parser — not 6 ad-hoc regex patterns. Tokens are classified into semantic categories (Entity, Action, Property, Relation) and reduced via shift-reduce rules: `Entity + Action + Entity → ActionFrame`, `Property + Entity → Attribution`. Unrecognized structures degrade gracefully to proximity edges.
- **It's grounded.** Sensors (accelerometer, light, proximity) feed real values into the graph. Actions produce real Android intents. No symbol floats free.
- **It's curious — and energy-aware.** When it encounters something it doesn't understand, it recursively resolves it against an offline knowledge base. But instead of a hard depth cap of 10, it uses a **CuriosityBudget** — an energy pool that depletes proportional to semantic distance from SELF, global arousal level, and structural error rate. High novelty lowers the halt threshold, allowing deeper exploration for novel concepts.
- **It understands via frame satisfaction.** Meaning is not flat edge count — a sentence is "understood" when its parsed roles satisfy a ConceptualFrame's mandatory slot requirements. `CommercialTransfer` requires Buyer, Seller, Goods, Money. `The pirate bought a hat` binds `pirate`→Buyer, `hat`→Goods.
- **Rendering is a motor system.** `DrawSkeleton`/`ApplyTransform` are Effector primitives in the graph. When an Effector node fires, it generates a RenderCommand — and the rendered output feeds back into predictive coding. Match → Reward spike. Mismatch → PredictionError + Novelty penalty.
- **It's honest.** When it doesn't know something, it *knows* it doesn't know — that's a structural error, and it hurts.
- **It's self-correcting.** Three layered guardrails — edge invariants, AST validation, runtime verification — prevent structural corruption before it accumulates.

This is not machine learning. This is *structure building*.

## How It Works

### PrimitiveMatrix — Algebraic Bootstrapping

Every grounded concept is a 5-dimensional vector:

| Dimension | Range | Examples |
|-----------|-------|---------|
| **Mass** | 0.0–1.0 | Matter=1.0, Light=0.0, Gas=0.1 |
| **Velocity** | 0.0–1.0 | Motion=1.0, Static=0.0 |
| **Spatial** | 0.0–1.0 | Dimension=1.0, Color=0.0 |
| **Valence** | -1.0–+1.0 | Color=+0.5, Cold=-0.2 |
| **Temporal** | 0.0–1.0 | Time=1.0, Never=0.0 |

Base primitives (Matter, Motion, Energy, Space, Time, etc.) are unit vectors. Complex concepts are derived by `combine()`: weighted addition of two parent vectors. Edge contracts between concepts are derived dynamically from dimensional overlap — no hardcoded contract table needed.

### Shift-Reduce CCG Parser

The 6 ad-hoc grammar patterns (`is_a`, `has`, `can`, `needs`, `causes`, `is_like`) are replaced by a proper Combinatory Categorial Grammar parser:

1. **Tokenize** — classify each word into a SemanticCategory (Entity, Action, Property, Relation, Modifier, Unknown)
2. **Shift** — push token onto parse stack
3. **Reduce** — apply combinatory rules in priority order:
   - `Entity + Action + Entity → ActionFrame`
   - `Entity + Action → ActionFrame (no object)`
   - `Property + Entity → Attribution`
   - `Entity + Relation + Entity → Relation`
   - `Entity + Unknown → proximity`
   - `Unknown + Entity → proximity`
4. **Fallback** — unrecognized structures degrade to pairwise `AssociatedWith` proximity edges

### CuriosityBudget — Energy-Aware Exploration

Replaces the hard `MAX_RECURSION_DEPTH=10` with a CuriosityBudget energy model:

```
E_consume = 0.3 * dist(SELF, concept) + 0.4 * arousal + 0.2 * error_rate
threshold = novelty * 0.15
halt when E_remaining < threshold
```

- High arousal → expensive search (clamped, focused)
- High novelty → cheap exploration (deeper search)
- Structural errors → more expensive branches
- The budget is carried through recursive resolution chains

### ConceptualFrame — Slot-Based Meaning

Frames define understanding structurally:

| Schema | Required Slots | Optional Slots |
|--------|---------------|----------------|
| CommercialTransfer | buyer, seller, goods | money |
| MotionEvent | actor, goal | source, instrument |
| SensoryEvent | observer, stimulus, channel | — |
| OwnershipChange | owner, possession | previous_owner |

A sentence is "understood" when its parsed roles bind to frame slots. Satisfaction = mandatory slots filled / total slots.

### Unified Cognitive-Render Loop

```rust
// MotorCommand nodes fire → generate RenderCommands
// Renderer executes → sends back actual_hash
// process_render_feedback() compares against prediction:
//   match → spike_reward(0.05)    ("I was right!")
//   mismatch → spike_novelty(...) ("I was wrong...")
```

Rendering is a motor system. `Effector` AST nodes map to `MotorCommand` grounding. The activation engine tracks `RenderPrediction` — expected render hash — and compares against actual render feedback. This closes the perception-action loop entirely within the graph.

### Spreading Activation

```
For each node, every 16ms tick (6-phase):
  Phase 1 — Neuromodulator decay:
    novelty/arousal/reward leak toward baseline.
    Compute global threshold_mod, plasticity_mod.

  Phase 2 — Decay + Injection + Precision-Weighted Prediction Error:
    activation *= node.decay * 0.97
    activation += sensor/intent energy
    if prediction exists:
      raw_error = |actual - expected| / max(expected, 0.001)
      precision = clamp(1.0 / (node.variance + 0.001), 0.1, 10.0)
      weighted_error = raw_error * precision
      update node.mean_error += 0.05 * (raw_error - mean_error)
      update node.variance += 0.05 * ((actual-expected)² - variance)
      if weighted_error > 30% → PredictionError → novelty spike

  Phase 3 — Sparse Spread + Firing:
    if activation > threshold * threshold_mod → fire, reset to 0
      push node index to FiredNodesBuffer
      if MotorCommand → push RenderCommand + RenderPrediction
    else:
      if edge_count > 64: sort by weight, truncate to top 64
      spread energy to neighbors (conservation)
      stop when energy < 0.005

  Phase 4 — Event-Driven STDP + pruning:
    for each node in FiredNodesBuffer:
      decay eligibility on its edges, LTP if co-fire
    for each pair in FiredNodesBuffer:
      check bidirectional edges, apply LTP/LTD
    drift toward default weight, prune if |weight| < 0.005
    clear FiredNodesBuffer

  Phase 5 — Valence update (preference formation):
    fire + prediction error → negative drift
    fire + no prediction error  → positive drift
    SELF → slow drift toward +0.5

  Phase 6 — Structural verification:
    energy conservation check (pre-spread vs post-spread)
    fired-chain path integrity (cached verify_path — skips
      per-edge contract checks for previously verified pairs)
    penalties: novelty spike, valence drop, edge prune marking
```

No allocations in the hot path. Double-buffered activation arrays flipped atomically. Zero-lock reads.

### Neuromodulation

Three global channels modulate behavior like brain chemistry:

| Channel | Spiked by | Effect |
|---------|-----------|--------|
| **Novelty** | Curiosity gaps, prediction errors, structural errors | Lowers thresholds (easier to fire), accelerates STDP, lowers CuriosityBudget halt threshold |
| **Arousal** | Rapid sensor deltas (>0.5g) | Lowers thresholds, increases CuriosityBudget step cost |
| **Reward** | Stable predictions, render feedback match | Solidifies recent edge changes |

Each decays naturally toward baseline every tick (novelty 8%, arousal 12%, reward 4%).

### Predictive Coding

Every activation tick computes a forward prediction: "what activation level do I expect next tick?" When sensor data violates this expectation, a Prediction Error signal is generated — injecting energetic novelty into the curiosity loop:

```
tick N: spread activation → compute prediction[node] = activation[node]
tick N+1: compare actual vs prediction
           raw_error = |actual - expected| / max(expected, 0.001)
           precision = clamp(1.0 / (node.variance + 0.001), 0.1, 10.0)
           weighted_error = raw_error * precision
           node.mean_error += 0.05 * (raw_error - mean_error)
           node.variance += 0.05 * ((actual-expected)² - variance)
           if weighted_error > 0.3 → spike novelty, inject into curiosity gap node
```

Each node tracks its own prediction reliability (`mean_error`, `variance` EMAs). High-variance nodes (noisy sensors) naturally self-attenuate — their precision approaches 0.1, reducing effective error up to 10×. This prevents runaway curiosity loops from sensor jitter.

Render feedback follows the same pattern: expected render hash vs actual render hash.

### Spike-Timing-Dependent Plasticity (STDP)

Edges learn from co-firing patterns. Each Edge carries:
- `dynamic_weight` — the effective weight used during spreading, modified by STDP
- `eligibility` — a Hebbian trace that decays (×0.9/tick), boosts (+1.0) when source fires, and is consumed (×0.5) when target fires to drive Long-Term Potentiation

Between LTP events, `dynamic_weight` slowly drifts toward the architecturally-intended default weight (LTD drift). Edges below `|0.005|` are pruned during consolidation.

### Sleep Consolidation

When neuromodulator levels are low (novelty < 0.1, arousal < 0.1) and 1000 ticks have elapsed (~16s), the engine runs an offline consolidation pass:
- **Edge GC** — removes pruned edges
- **Linear chain compression** — A→B→C where B is a pass-through node (indegree=1, outdegree=1) becomes direct edge A→C with combined weight
- **Autonomous category synthesis** — clusters nodes sharing ≥80% edge signature overlap into a parent `Concept_Cluster` node with `IsA` edges. Creates abstract categories without external supervision.

### Self-Healing Metacognition

When the engine detects that one of its own cognitive modules is underperforming — a parser that's too slow, a scheduler with too many failures — it can rewrite itself. The _metacognition_ crate implements a 5-phase deterministic pipeline that runs during the same idle consolidation window (novelty < 0.1, arousal < 0.1):

```
Idle tick → DeficiencyScanner.scan()
  ↕ constraint violated ≥5 consecutive ticks?
  Yes → pick most severe deficiency
  Phase 1 — Generate: translate deficiency into a CandidateModule DSL patch
  Phase 2 — Verify: validate type signatures against the targeted trait bound
  Phase 3 — Regression Test: run standard test inputs through the candidate
  Phase 4 — Benchmark: compare stock vs candidate on a sample loop
  Phase 5 — Hot-Swap: if ≥5% faster with >95% success rate, atomically flip the
            double-buffer SwapSlot → remedy node recorded in graph
```

**Three engine layers** enforced by trait + visibility boundaries:
- **Layer 0** (Physics) — immutable, no swap slot
- **Layer 1** (Cognitive Modules) — `CognitiveParser`, `FrameMatcher`, `CuriosityScheduler`, `GapDetectorModule` — self-healing via `SwapSlot<dyn Trait>`
- **Layer 2** (Strategies) — `ExplorationPolicy`, `InferenceOrder` — fully replaceable

The candidate patch is expressed as **DSL bytecode** (18 safe opcodes, `#![no_std]`-compatible, `[f64; 16]` fixed stack, 256-byte state buffer) and compiled by a `DslCompiler` that validates termination and bounds. Hot-swaps use a `SwapSlot<T>` lock-free double buffer with `AtomicU64` generation counter — readers never block.

When the DeficiencyScanner reports a persistent constraint violation, the **metacognitive curiosity divert** routes budget inward: `E_consume = 0.3·dist(SELF, concept) + 0.4·arousal + 0.2·error_rate + 0.1·deficiency_severity`. At severity > 0.3, at least 30% of the curiosity budget goes to internal optimization instead of external exploration.

### Episodic Memory — A Timeline of Lived Experience

Beyond the semantic graph's conceptual knowledge, the engine records a continuous timeline of events. Every tick, during Phase 6, it writes a 64-byte record into a lock-free SPSC ring buffer (1024 slots):

```
Fired Node A (activation 0.73, novelty 0.2, arousal 0.1)
  ↓
Prediction Error on Node B (magnitude 0.45, novelty spike 0.3)
  ↓
Sensor Reading "accelerometer[0]" = 0.82
  ↓
  ...
```

During idle consolidation (novelty < 0.1, arousal < 0.1), the ring buffer is drained and events are grouped into episode clusters (temporally adjacent events within 5 ticks of each other). Each cluster's **importance** is computed deterministically:

| Factor | Weight | Source |
|--------|--------|--------|
| Prediction error present | +0.40 | Surprise = important |
| Structural fault present | +0.50 | Failure = very important |
| Peak novelty | ×0.30 | Emotional salience |
| Peak arousal | ×0.20 | Physiological response |
| Peak reward | ×0.15 | Success signal |
| Event density (max 10) | ×0.15 | Richness of experience |

Clusters above `0.15` importance are promoted into the graph as `NodeType::Episode` nodes with `Grounding::Episode { tick, timestamp_ms, importance }`. SELF links to each episode via `Relation::Experienced`. Episodes are chained in temporal order via `Relation::Precedes`. Involved nodes receive `Relation::AssociatedWith` edges back to the episode.

The query API supports recollection:
- `query_recent(graph, 10)` — last 10 episodes
- `query_tick_range(graph, 1000, 2000)` — episodes within a tick window
- `query_by_node_label(graph, "concept_movement")` — episodes involving a specific node

This gives the engine a genuine, queryable past — not just conceptual knowledge, but *memory of what happened and when*.

### The Self Node

Node index 1 is always `SELF` — the engine's persistent "I". Pre-inserted with base activation 1.0 and decay 1.0, it never fades. Every experience attaches here:

- Sensor reading → `SELF --GroundedIn--> instrument`
- Intent received → `SELF --HasProperty--> object`
- Action fired → `SELF --CausedBy--> action_node`

`introspect()` returns the entire subgraph reachable from SELF. The engine can always answer "what do I know?" because every concept it's encountered has a relational path back to itself.

### Born Alive: Default Mode Network

The engine doesn't sit silent waiting for input. It has its own inner life:

- **First tick**: Inject SELF with activation → "I'm awake. I notice: accelerometer, proximity, light." Links SELF to every sensor it's born with.
- **~800ms idle**: Every ~50 ticks without external input, it spontaneously activates its highest-valence concept. It returns to things it likes, wonders about things it doesn't understand.
- **~3.2s idle**: When very idle, it seeks novelty — picks a poorly-connected node and injects activation into it, generating "I'm curious about X. What is it?"
- **Every ~8s active**: When processing external events, it periodically voices thoughts about whatever just fired.

### Valence: Learning to Like and Dislike

Every `GroundedNode` carries a `valence: f64` field (-1.0 to +1.0). After every tick:
- If a node fires AND had a prediction error → valence shifts negative (surprise = aversive)
- If a node fires with no prediction error → valence shifts positive (familiar = comfortable)
- SELF slowly drifts toward +0.5 (baseline contentment)

Over hours and days, the system develops genuine preferences — deterministically, from its own prediction history.

### Opinions, Mood, Interests

| Function | Returns | Example |
|----------|---------|---------|
| `get_opinion("accelerometer")` | "I like accelerometer. It makes me think of movement." | Synthesized from valence + neighbor traversal |
| `get_mood()` | "Curious and alert" / "Content" / "Calm" | From neuromodulator levels |
| `get_interests(5)` | `["self", "sensor_light", "concept_movement", ...]` | Top-N highest-valence nodes |

### The Curiosity Loop

When you say "cat dressed as a pirate walking on two legs":

1. **Split into words** → [cat, dressed, as, a, pirate, walking, on, two, legs]
2. **Gap detect** → cat has edges (is an animal, has fur). pirate has no edges → **GAP**
3. **Check CuriosityBudget** → `step_cost(semantic_distance, arousal, novelty)` → decrement budget
4. **Resolve** → Look up "pirate" in offline knowledge; CCG parse into relations
5. **Insert** → Create nodes + edges in the graph
6. **Recurse** → "tricorn hat" has no edges → **GAP** → carry budget forward → resolve
7. **Stop** when CuriosityBudget is exhausted (not at a hard depth 10)

### The Asset Pipeline

Compound prompts feed into the asset ingestor:

1. **Lexical pass** → role assignment (subjects: cat, predicates: dressed/walking, modifiers: two)
2. **Skeleton extraction** → cat → quadruped skeleton with 6 joints
3. **Transform cascade** → "walking on two legs" → quadruped→biped transform
4. **Render AST compilation** → `compile_to_ast()` converts ops into validated `RenderAst`
5. **AST validation** → includes `Effector` nodes (motor commands that map to `Grounding::MotorCommand`)
6. **Serialization** → `render_ast_to_json()` produces verified JSON for the GPU renderer

---

## Visual Effector Pipeline

The motor system bridges the cognitive daemon thread and the render thread through two lock-free channels:

```
Cognitive Daemon (Phase 6)                  Render Bridge Thread
┌────────────────────────────────────┐     ┌──────────────────────────┐
│ compute_effector_state()           │     │ loop {                  │
│   ↓ buffer.write(&state)          │ 1.  │   buffer.read(&state)   │
│                                    │────►│   backend.render(state) │
│ get_active_visual_primitives(acts) │ 2.  │   ring.pop(&payload)    │
│   if > threshold: ring.push()     │────►│   backend.render(payload)│
│                                    │     │   sleep(8ms)            │
│ SensorMapper (Phase 2):            │     │ }                       │
│   accel→RotationX/Y/Z injection   │     │                         │
│   light→ColorChroma + Scale inj.  │     │ AST Errors → PenalizeFn │
└────────────────────────────────────┘     │   update_valence(-0.05) │
                                          └──────────────────────────┘
 1. VisualEffectorBuffer (double-buffered, [AtomicU64×11], 22 f32 values)
 2. VisualPrimitiveRingBuffer (SPSC, [AtomicU64×384], 64 slots × 6 f64)
```

### Sensor-to-Image Transforms

Deterministic mapping from sensor readings to effector state:

| Sensor | Transform | Effector Fields |
|--------|-----------|-----------------|
| Light (lux) | `PaletteInterpolator::compute(lux, arousal)` → log-normalized interpolation | pal_coeff_0, pal_coeff_1, wireframe_flag |
| Accelerometer (gx, gy, gz) | `SkeletalTransformMatrix::from_gravity(g, activation)` → Euler angles | rot0..rot5 |
| Gravity (magnitude) | `rest_pose_adjustment(g)` → scale (upright=1.0, supine=0.6) | scale_x/y/z |
| Spatial activation | Color intensity = activation × 0.8 + 0.2 | color0_rgba, color1_rgba |

- **`TransformEngine`** — one per daemon, owns `PaletteInterpolator` + previous gravity vector
- **`light_to_palette_matrix()`** — pure function, side-effect-free palette interpolation
- **`accel_to_skeletal_rotation()`** — pure function, maps accel vector to Euler rotation angles
- **`gravitational_to_rest_pose()`** — pure function, maps gravity magnitude to scale modifiers

### RenderBridge (own OS thread)

Polls both the lock-free `VisualEffectorBuffer` and `VisualPrimitiveRingBuffer` at 8ms intervals:

- **`RenderBackend` trait** — abstract interface: `render(&state) → Result<u64, String>`
- **`NullRenderBackend`** — computes a hash from state without actual rendering
- **`WgpuRenderBackend`** — full wgpu state machine (requires `--features wgpu`), reads effector state → updates bone transforms, palette colors, wireframe toggle, encodes + submits passes
- **`VisualPrimitiveRingBuffer` polling** — reads `FixedVisualPayload` from SPSC ring buffer, packs into effector state array, sends to backend
- **`PenalizeFn`** — callback for `validate_ast()` error → `GraphArena::update_valence(-0.05, 0.1)` on all nodes
- Started/stopped alongside `CognitiveDaemon` from `CognitiveLifecycle`

---

## Three Structural Guardrails

### 1. Edge Invariants (compile-time / pre-execution)

Every `Edge` carries an optional `InvariantContract`. If not explicitly set, `effective_contract()` falls back to the `Relation::canonical_contract()`.

**`GraphArena::verify_path(path)`** validates a sequence of node IDs against their contracts. Uses a 16-slot incremental path cache (`parking_lot::Mutex<[Option<CachedPath>; 16]>`) — if `(source, dest)` was previously verified, per-edge contract checks are skipped. Cache is invalidated on any structural mutation.
**`GraphArena::find_path(start, end)`** uses BFS to find the shortest valid traversal.

### 2. AST-Driven Integrity (asset pipeline)

**`RenderAst`** enum: `Scene`, `DrawSkeleton`, `ApplyTransform`, `ApplyMesh`, `Composite`, `Effector`.
- `validate_ast()` checks scene structure, skeleton references, conflicting transforms, composite source ordering, **and Effector target references**.
- `compile_to_ast()` → `validate_ast()` → `render_ast_to_json()` enforced for all render ops.

### 3. Runtime Verification Loop (Phase 6)

After every tick's valence update:
- **Energy conservation**: discrepancy > 1% → `EnergyNonConservation` fault
- **Path integrity**: fired chain passes through `verify_path()` → `ContractMismatch` / `CyclicDependency` / `DeadNode`
- On fault: novelty spike, valence drop, edge pruning

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Grounded Engine                     │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Rust Runtime (native thread)          │ │
│  │                                                    │ │
│  │  ┌─────────────┐   ┌──────────────────────────┐  │ │
│  │  │ EventChannel│──►│ CognitiveDaemon (16ms)   │  │ │
│  │  │ (SPSC 128)  │   │ ├─ Phase 1-5 tick       │  │ │
│  │  │             │   │ ├─ Render feedback loop  │  │ │
│  │  │ Sensor data │   │ ├─ Phase 6: verify       │  │ │
│  │  │ Intent JSON │   │ └─ Output channel        │  │ │
│  │  └─────────────┘   └──────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Curiosity Harvester (Tokio async)        │  │ │
│  │  │ ├─ GapDetector → KnowledgeGap            │  │ │
│  │  │ ├─ CCG RelationalParser                  │  │ │
│  │  │ ├─ KnowledgeStore (~30 foundation cones) │  │ │
│  │  │ └─ CuriosityBudget (energy-aware search) │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Asset Ingestor                           │  │ │
│  │  │ ├─ ComponentExtractor                    │  │ │
│  │  │ ├─ TransformEngine                       │  │ │
│  │  │ ├─ RenderAst (incl. Effector nodes)      │  │ │
│  │  │ └─ compile/validate/serialize            │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Verification Core                        │  │ │
│  │  │ ├─ VerificationLoop (Phase 6)            │  │ │
│  │  │ ├─ verify_path() / find_path()           │  │ │
│  │  │ └─ StructuralError accumulator           │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ VisualEffectorBuffer (lock-free)         │  │ │
│  │  │ ├─ [AtomicU64; 11] × 2 double buffer    │  │ │
│  │  │ ├─ Written by Phase 6 (cognitive thread)│  │ │
│  │  │ └─ Read by RenderBridge (render thread)  │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ RenderBridge Thread (8ms poll)           │  │ │
│  │  │ ├─ Polls VisualEffectorBuffer            │  │ │
│  │  │ ├─ RenderBackend trait (wgpu/Null)       │  │ │
│  │  │ └─ validate_ast() error → penalize node  │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Sensor→Image Transforms (asset-ingestor) │  │ │
│  │  │ ├─ PaletteInterpolator (lux→palette)     │  │ │
│  │  │ ├─ SkeletalTransformMatrix (accel→rot)   │  │ │
│  │  │ └─ TransformEngine (sensor→effector)     │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ FrameSchemas + PrimitiveMatrix           │  │ │
│  │  │ ├─ 5-dimensional vector algebra          │  │ │
│  │  │ ├─ 31 base + derived primitives          │  │ │
│  │  │ ├─ 4 built-in frame schemas              │  │ │
│  │  │ └─ Frame satisfaction evaluation         │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Crates

| Crate | Purpose |
|-------|---------|
| `semantic-graph` | GroundedNode (valence, **mean_error**, **variance**, **epistemic_status**), GraphArena (**path_cache**: 16-slot `CachedPath` with interior mutability, **link_workspace()**, **garbage_collect_transient_edges()**), ActivationBuffer, Edge (STDP, contract, **TransientWorkspace**), FiringHistory, Neuromodulator, PrimitiveVector (5-d algebra), FrameSchema/FrameInstance (slot-based meaning), 12 Relation types (added **SupportsBelief**), 3 **EpistemicStatus** variants + **CognitiveMode** enum, MotorCommandType (effector commands), **VisualEffectorBuffer** (lock-free atomic double buffer), effector_state module (22-element pack/unpack), 31 base+derived primitives, **VisualPrimitiveType** (6 visual primitive variants), **Grounding::VisualPrimitive**, **FixedVisualPayload** (6 f64 fields), **VisualPrimitiveRingBuffer** (lock-free SPSC, 64 slots) |
| `semantic-parser` | Verb→CDAction table (30+), sensor parsing, Realizer, **CCG RelationalParser** (shift-reduce, semantic categories, 7 reduction rules, proximity fallback) |
| `cognitive-core` | ActivationEngine (6-phase tick: **precision-weighted prediction error**, **workspace resonance**, **epistemic gating**, **belief confidence tracking**, **sparse spread** with TRIM=64, **event-driven STDP** via `FiredNodesBuffer [u32; 1024]`, **counterfactual short-circuit**), VerificationLoop, CognitiveDaemon (**set_cognitive_mode()**, **set_self_healing_hook()**, **set_episodic_recorder()**, runs `SelfHealingHook` + `EpisodicRecorder` during idle consolidation), **WorkingMemoryWorkspace** (`[NodeId; 12]`, resonance +0.15), **RenderCommand/RenderPrediction feedback loop** (+**render_target: CognitiveMode**), Phase 6 VisualEffectorBuffer + **VisualPrimitiveRingBuffer push**, **SensorMapper** (stateless fixed-point accelerometer/light→visual primitive injection), **EpisodicEvent** enum (5 variants for hot-path recording), **EpisodicRecorder** trait (lock-free record + consolidate), EventChannel, Consolidation (**synthesize_categories()** v2: edge-signature + **predictive-role abstraction** with 2-hop profile clustering) |
| `hw-daemon` | Android lifecycle bridge, graph persistence, keepalive, modulate/consolidate bridge, creates `ModuleRegistry` + `SelfHealingPipeline` with default constraints, attaches to daemon via `set_self_healing_hook()`, **RenderBridge** (OS thread, dual buffer + ring polling, RenderBackend trait, Null/Wgpu backends), **PenalizeFn** (–0.05 valence on validate_ast() error) |
| `curiosity-core` | Gap detection, **CCG-based DefinitionResolver**, async harvester, **CuriosityBudget** (energy-aware, replaces depth 10) |
| `metacognition` | **Self-healing pipeline**: `DeficiencyScanner` (constraint violation detection), `CandidateModule` DSL (18 safe opcodes, bytecode interpreter), `SwapSlot` lock-free double-buffer hot-swap, 5-phase `SelfHealingPipeline` (Generation → Contract Verification → Regression Testing → Ecological Benchmarking → Hot-Swap), `MetacognitiveCuriosity` (internal budget routing) |
| `episodic-memory` | **Episodic timeline**: lock-free SPSC ring buffer (1024 × 64 bytes) for hot-path event recording, idle-cycle consolidation into `NodeType::Episode` graph nodes linked to SELF via `Relation::Experienced`, query API for tick-range / node-label / recent-N retrieval |
| `planning-core` | **Hierarchical planner + foresight + values + autonomy modules (A–D)**: `HierarchicalPlanner` (goal→subgoal decomposition via `SubGoalOf`, action-tree expansion, plan selection by cost/confidence), `ForesightEngine` (chain-strength analysis, per-action symbolic simulation, parallel `NodeType::Simulation` branches with confidence pruning), `ValueSystem` (6 hardwired drives + 6 value categories + long-term goals + `DriveHook` tick injection for persistent activation bias), **Module A — `GoalFormationEngine`** (autonomous goal creation from prediction error histories, priority formula `P = α·Eₚ + β·N + γ·D_dep + δ·S_ineff`), **Module B — `StrategicPlanner`** (MCTS with UCB1 selection, expected free energy minimization, DMN replan at 0.3 prediction error threshold), **Module C — `AffordanceRegistry`** (tool abstraction via 5-d cosine similarity signatures, graph materialization every 5000 ticks), **Module D — `CrossDomainEngine` + `PlanExecutionEngine`** (domain subspace projection with 5×5 learned matrices, step-by-step plan execution with parallel branch injection) |
| `asset-ingestor` | Prompt decomposition, quadruped→biped transform, RenderAst (incl. **Effector** variant), compile_to_ast/validate_ast/render_ast_to_json, **effector math** (GravityVector, PaletteInterpolator, SkeletalTransformMatrix), **TransformEngine** (sensor→effector) |
| `uniffi-exports` | 18-function UniFFI surface for Kotlin |

## Status

**Active development.** All crates have unit tests.

- [x] Semantic graph with arena allocation + double-buffered activation
- [x] 6-phase spreading activation (decay/inject/spread/fire/valence/verify)
- [x] 30+ verb→CD action mappings
- [x] Hebbian STDP: dynamic_edge_weight, eligibility traces, LTP/LTD, pruning
- [x] Neuromodulation: novelty, arousal, reward channels
- [x] Predictive coding: activation predictions, |error| > 30% novelty spike
- [x] Sleep consolidation: edge GC, linear chain compression
- [x] Lock-free SPSC event channel (128 slots)
- [x] Self node, self-linking, introspect(), valence, opinion/mood/interests
- [x] Default Mode Network: spontaneous inner activity when idle
- [x] **PrimitiveMatrix**: 5-d vector algebra, 28 base + 3 derived primitives, dynamic edge contract derivation
- [x] **CCG RelationalParser**: stateless shift-reduce, semantic categories, 7 reduction rules, proximity fallback
- [x] **CuriosityBudget**: energy-aware search, replaces depth-10 circuit breaker
- [x] **ConceptualFrame slots**: FrameSchema/SlotBinding/FrameInstance, 4 built-in schemas, satisfaction evaluation
- [x] **Unified cognitive-render loop**: MotorCommand grounding, Effector AST nodes, RenderCommand->prediction->feedback->reward/novelty
- [x] **VisualEffectorBuffer**: lock-free `[AtomicU64; 11]` × 2 double buffer in semantic-graph
- [x] **Effector math**: GravityVector, PaletteInterpolator, SkeletalTransformMatrix, 3 new derived primitives (kinetic_energy, spatial_bound, color_intensity)
- [x] **TransformEngine**: deterministic light→palette, accel→skeletal rotation, gravitational→rest-pose in asset-ingestor
- [x] **RenderBridge**: OS thread polling VisualEffectorBuffer at 8ms, RenderBackend trait (Null/Wgpu), validate_ast() error → penalize callback in hw-daemon
- [x] **Phase 6 effector push**: ActivationEngine.compute_effector_state() → VisualEffectorBuffer.write() after structural verification
- [x] **VisualPrimitiveType**: 6 concrete visual primitive variants (`SpatialScale`, `RotationX/Y/Z`, `ColorChroma`, `TopologyWireframe`) with fixed `NodeId` indices in GraphArena
- [x] **FixedVisualPayload**: 6-field stack struct with `pack()`/`unpack()` → `[AtomicU64; 6]`
- [x] **VisualPrimitiveRingBuffer**: lock-free SPSC ring buffer (64 slots, 384 atomic words) in semantic-graph
- [x] **SensorMapper**: stateless fixed-point accelerometer→rotation, light→chroma injection + variance error detection
- [x] **Phase 6 ring buffer push**: cluster activation > threshold → push to SPSC ring buffer alongside existing VisualEffectorBuffer push
- [x] **Render bridge ring polling**: reads FixedVisualPayload from ring buffer alongside existing effector buffer reads
- [x] **validate_ast() → –0.05 valence**: PenalizeFn deducts valence on structural errors detected during render
- [x] **Precision-weighted prediction error**: per-node `mean_error`/`variance` EMA, precision = 1/(variance+0.001) clamped [0.1, 10.0], weighted_error vs threshold
- [x] **Sparse spread**: sort+truncate to 64 edges for high-degree nodes, early termination at 0.005 remaining energy
- [x] **Event-driven STDP**: FiredNodesBuffer [u32; 1024], incident-only edge processing in Phase 4 (O(F·avg_degree) instead of O(N+E))
- [x] **Incremental path cache**: 16-slot `CachedPath` cache in GraphArena with interior mutability, invalidated on structural mutation
- [x] **Autonomous category synthesis**: isomorphism scanner during consolidation, ≥80% edge signature overlap → Concept_Cluster parent with IsA edges
- [x] **Epistemic node separation**: three-tier `TransientObservation → StableBelief → CoreConcept` with belief confidence tracking and Phase 3 spread gating
- [x] **Working memory workspace**: `[NodeId; 12]` stack-allocated attention buffer, `+0.15` resonance injection, `TransientWorkspace` contract edges (skipped in STDP and consolidation)
- [x] **Counterfactual simulation mode**: `CognitiveMode::Counterfactual` short-circuits Phase 4 (STDP) and Phase 5 (valence), `RenderCommand.render_target` routes to ImaginationBuffer
- [x] **Predictive-role abstraction**: two-pass category synthesis (edge-signature + 2-hop predictive profile clustering at 70% overlap, terminal node extraction)
- [x] **Self-healing metacognition**: DeficiencyScanner + 5-phase pipeline + DSL bytecode (18 opcodes) + SwapSlot lock-free hot-swap + metacognitive curiosity divert (wired end-to-end via `CuriosityHook` trait in cognitive-core, implemented by `MetacognitiveCuriosity`, called in idle consolidation and DMN curiosity drive)
- [x] **Unified episodic memory**: lock-free SPSC ring buffer (1024 × 64 bytes), 5 event types (firing/prediction-error/fault/sensor/intent), idle-cycle graph consolidation as `NodeType::Episode` nodes linked to SELF via `Relation::Experienced`, tick-range / node-label / recent-N query API
- [x] **Hierarchical planning engine**: `HierarchicalPlanner` in `planning-core` (goal decomposition, action-tree expansion, plan selection, mid-loop replanning on prediction error)
- [x] **Cross-modal sensor fusion**: `CrossModalRegistry` in cognitive-core binding sensor channels to semantic concept nodes; `SensorMapper::cross_modal_inject()` in Phase 2 of every tick
- [x] **Foresight engine**: `ForesightEngine` in `planning-core` (chain strength analysis, per-action symbolic simulation, parallel branches with confidence pruning)
- [x] **Intrinsic value system**: `ValueSystem` in `planning-core` (6 hardwired drives + 6 value categories + long-term goals + `DriveHook` tick injection)
- [x] **Autonomous goal formation (Module A)**: `GoalFormationEngine` with per-node error ring buffers, systemic efficiency tracker, autonomous goal node creation via priority formula
- [x] **Strategic planning with MCTS (Module B)**: `StrategicPlanner` with UCB1 tree search, expected free energy action selection, DMN replan trigger at 0.3 threshold
- [x] **Tool abstraction layer (Module C)**: `AffordanceRegistry` with 5-d affordance signatures, cosine similarity matching, automatic graph injection
- [x] **Cross-domain execution (Module D)**: `CrossDomainEngine` with 5×5 learned domain mappings and subspace projection; `PlanExecutionEngine` with step advancement, pause/resume, parallel branch injection
- [ ] `cargo test` pass (needs actual test environment)
- [ ] Integration: persist graph → survive restart → resume curiosity

## Why not just use an LLM?

| | LLM | Grounded |
|---|---|---|
| **Understanding** | Statistical token prediction | Relational graph traversal + frame satisfaction |
| **Grounding** | None (floating symbols) | Every node traces to sensor or action |
| **Determinism** | None (temperature sampling) | Deterministic per tick; graph grows over time |
| **Offline** | No (needs datacenter) | Yes (runs on a phone) |
| **Power** | Watts × 1000 | Milliwatts |
| **Learning** | Pre-trained on all of internet | Grows through interaction |
| **Size** | GB–TB | KB–MB |
| **Honesty** | Hallucinates constantly | Can only traverse what it knows |
| **Self-correction** | Fine-tuning / RLHF | Structural errors auto-penalize; energy-aware search |

Grounded doesn't guess. It traverses what it knows, understands via frame satisfaction, explores via energy budget, and closes the perception-action loop through render feedback.

## Build

```bash
# Check all crates
cargo check

# Run tests
cargo test
```

---

**Grounded is not an AI. It's something else entirely.**
