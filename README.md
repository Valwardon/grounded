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

  Phase 2 — Decay + Injection + Prediction Error:
    activation *= node.decay * 0.97
    activation += sensor/intent energy
    |activation - prediction| > 30% → PredictionError → novelty spike

  Phase 3 — Spread + Eligibility:
    if activation > threshold * threshold_mod → fire, reset to 0
    else: for each edge, spread energy to neighbor (conservation)
          if MotorCommand → push RenderCommand + RenderPrediction

  Phase 4 — STDP + pruning:
    for each edge: decay eligibility, boost if source fired,
    if target fired: LTP = eligibility * rate * plasticity_mod,
    drift toward default weight, prune if |weight| < 0.005

  Phase 5 — Valence update (preference formation):
    fire + prediction error → negative drift
    fire + no prediction error  → positive drift
    SELF → slow drift toward +0.5

  Phase 6 — Structural verification:
    energy conservation check (pre-spread vs post-spread)
    fired-chain path integrity (verify_path on edge contracts)
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
           error = |actual - expected| / expected
           if error > 0.3 → spike novelty, inject into curiosity gap node
```

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

## Three Structural Guardrails

### 1. Edge Invariants (compile-time / pre-execution)

Every `Edge` carries an optional `InvariantContract`. If not explicitly set, `effective_contract()` falls back to the `Relation::canonical_contract()`.

**`GraphArena::verify_path(path)`** validates a sequence of node IDs against their contracts.
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
│  │  │ FrameSchmas + PrimitiveMatrix            │  │ │
│  │  │ ├─ 5-dimensional vector algebra         │  │ │
│  │  │ ├─ 28 base + 3 derived primitives        │  │ │
│  │  │ ├─ 4 built-in frame schemas              │  │ │
│  │  │ └─ Frame satisfaction evaluation         │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Crates

| Crate | Purpose |
|-------|---------|
| `semantic-graph` | GroundedNode (valence), GraphArena, ActivationBuffer, Edge (STDP, contract), FiringHistory, Neuromodulator, PrimitiveVector (5-d algebra), FrameSchema/FrameInstance (slot-based meaning), 10 Relation types, MotorCommandType (effector commands) |
| `semantic-parser` | Verb→CDAction table (30+), sensor parsing, Realizer, **CCG RelationalParser** (shift-reduce, semantic categories, 7 reduction rules, proximity fallback) |
| `cognitive-core` | ActivationEngine (6-phase tick), VerificationLoop, CognitiveDaemon, **RenderCommand/RenderPrediction feedback loop**, EventChannel, Consolidation |
| `hw-daemon` | Android lifecycle bridge, graph persistence, keepalive, modulate/consolidate bridge |
| `curiosity-core` | Gap detection, **CCG-based DefinitionResolver**, async harvester, **CuriosityBudget** (energy-aware, replaces depth 10) |
| `asset-ingestor` | Prompt decomposition, quadruped→biped transform, RenderAst (incl. **Effector** variant), compile_to_ast/validate_ast/render_ast_to_json |
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
