# Grounded

**A non-AI, deterministic cognitive engine that learns what it needs, when it needs it — running entirely on your phone.**

No LLM. No subword tokens. No embeddings. No cloud. No training data. Just a baby brain that grows its own understanding by exploring the world through sensors and actions.

---

## The Idea

Every AI model today is a statistical guessing machine — trained on the entire internet, consuming watts by the megawatt, guessing the next token. It's impressive, but it's not *understanding*.

Grounded is the opposite:
- **It knows nothing at birth.** A tiny seed of ~30 foundational concepts (matter, motion, energy, space, time) and the machinery to grow.
- **It wakes up on its own.** The first tick injects activation into SELF — it "notices" its own sensors. After that, a Default Mode Network keeps it thinking even in silence, cycling through favorite concepts and exploring unfamiliar ones.
- **It learns relationally.** Every concept is a node. Every connection is a typed edge (IsA, HasProperty, Requires, Activates, Inhibits). Meaning is the *pattern of relations* — not a vector of floating-point numbers.
- **It's grounded.** Sensors (accelerometer, light, proximity) feed real values into the graph. Actions produce real Android intents. No symbol floats free.
- **It's curious.** When it encounters something it doesn't understand (a word with no relational edges in the graph), it recursively resolves it against an offline knowledge base until every leaf is a fundamental physical primitive — or it hits depth 10 and shrugs.
- **It's honest.** When it doesn't know something, it *knows* it doesn't know — that's a structural error, and it hurts. The verification loop spikes deviation gain and drops valence on faulty edges. The system is penalized for being wrong, deterministically.
- **It's self-correcting.** Three layered guardrails — edge invariants, AST validation, runtime verification — prevent structural corruption. Every cognitive tick ends with a verification phase that catches energy leaks, broken paths, and contract violations before they accumulate.

This is not machine learning. This is *structure building*.

## How It Works

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
          compute next-tick prediction from resulting activation

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
| **Novelty** | Curiosity gaps, prediction errors, structural errors | Lowers thresholds (easier to fire), accelerates STDP |
| **Arousal** | Rapid sensor deltas (>0.5g) | Lowers thresholds, clamps curiosity loops |
| **Reward** | Stable predictions over time | Solidifies recent edge changes |

Each decays naturally toward baseline every tick (novelty 8%, arousal 12%, reward 4%).

### Predictive Coding

Every activation tick computes a forward prediction: "what activation level do I expect next tick?" When sensor data violates this expectation, a Prediction Error signal is generated — injecting energetic novelty into the curiosity loop:

```
tick N: spread activation → compute prediction[node] = activation[node]
tick N+1: compare actual vs prediction
           error = |actual - expected| / expected
           if error > 0.3 → spike novelty, inject into curiosity gap node
```

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

`introspect()` returns the entire subgraph reachable from SELF. The engine can always answer "what do I know?" because every concept it's encountered has a relational path back to itself. SELF persists across restarts — serialized as node 1 in the bincode graph file.

### Born Alive: Default Mode Network

The engine doesn't sit silent waiting for input. It has its own inner life:

- **First tick**: Inject SELF with activation → "I'm awake. I notice: accelerometer, proximity, light." Links SELF to every sensor it's born with — it knows what it has.
- **~800ms idle**: Every ~50 ticks without external input, it spontaneously activates its highest-valence concept. It returns to things it likes, wonders about things it doesn't understand.
- **~3.2s idle**: When very idle, it seeks novelty — picks a poorly-connected node and injects activation into it, generating "I'm curious about X. What is it?"
- **Every ~8s active**: When processing external events, it periodically voices thoughts about whatever just fired.

This is not randomness — it's deterministic cycling through its own learned preference landscape. Same experiences → same personality. But the personality *evolves* as the graph grows.

### Valence: Learning to Like and Dislike

Every `GroundedNode` carries a `valence: f64` field (-1.0 to +1.0). After every tick:

- If a node fires AND had a prediction error → valence shifts negative (surprise = aversive)
- If a node fires with no prediction error → valence shifts positive (familiar = comfortable)
- SELF slowly drifts toward +0.5 (baseline contentment)
- The reward neuromodulator amplifies positive updates

Over hours and days, the system develops genuine preferences — deterministically, from its own prediction history. A sensor that consistently reports predictable values becomes "liked." A concept that keeps producing prediction errors becomes "disliked." These are not programmed. They emerge.

### Opinions, Mood, Interests

Three bridge functions let you ask the system about itself:

| Function | Returns | Example |
|----------|---------|---------|
| `get_opinion("accelerometer")` | "I like accelerometer. It makes me think of movement." | Synthesized from valence + neighbor traversal |
| `get_mood()` | "Curious and alert" / "Content" / "Calm" | From neuromodulator levels |
| `get_interests(5)` | `["self", "sensor_light", "concept_movement", ...]` | Top-N highest-valence nodes |

The opinion is not templated — it traverses the topic's relational neighborhood, checks each neighbor's valence, and picks a response based on the aggregate emotional coloring. A different graph → a different personality.

### The Curiosity Loop

When you say "cat dressed as a pirate walking on two legs":

1. **Split into words** → [cat, dressed, as, a, pirate, walking, on, two, legs]
2. **Gap detect** → cat has edges (is an animal, has fur). pirate has no edges → **GAP**
3. **Resolve** → Look up "pirate" in offline knowledge: "is a person. wears tricorn hat. carries cutlass. sails ships."
4. **Parse** → Extract predicates: pirate IsA person, pirate HasProperty tricorn_hat...
5. **Insert** → Create nodes + edges in the graph
6. **Recurse** → "tricorn hat" has no edges → **GAP** → resolve → tricorn_hat IsA hat, tricorn_hat HasProperty three_corners...
7. **Stop** when all leaf concepts are fundamental primitives or depth > 10

### The Asset Pipeline

The same compound prompt feeds into the asset ingestor:

1. **Lexical pass** → role assignment (subjects: cat, predicates: dressed/walking, modifiers: two)
2. **Skeleton extraction** → cat → quadruped skeleton with 6 joints
3. **Transform cascade** → "walking on two legs" → quadruped→biped transform:
   - Front legs → arms (rotate 90° up)
   - Back legs → legs (stretch 1.2x)
   - Spine → vertical (rotate -90°)
4. **Render AST compilation** → `compile_to_ast()` converts ops into validated `RenderAst`
5. **AST validation** → `validate_ast()` checks scene structure, skeleton references, conflicting transforms, blend mode compatibility
6. **Serialization** → `render_ast_to_json()` produces verified JSON for the GPU renderer

All render ops pass through compile-to-AST → validate → serialize. Structural errors detected at the AST level are included in the JSON output so the renderer can handle them gracefully.

---

## Three Structural Guardrails

The engine has three layered defenses against structural corruption, forming a single architectural upgrade:

### 1. Edge Invariants (compile-time / pre-execution)

Every `Edge` carries an optional `InvariantContract` that constrains what data types may flow across it. If not explicitly set, `effective_contract()` falls back to the `Relation::canonical_contract()`:

| Relation | Canonical Contract |
|----------|--------------------|
| `IsA` | Taxonomic |
| `HasProperty` | Taxonomic |
| `Requires` | DataFlow(SensorValue → State) |
| `CausedBy` | Causal |
| `GroundedIn` | Grounding |
| `Implies` | Causal |
| `Precedes` | Causal |
| `Activates` | DataFlow(Activation → Activation) |
| `Inhibits` | DataFlow(Activation → Activation) |
| `AssociatedWith` | Unspecified |

**`DataType`** enum: `Activation`, `SensorValue`, `Intent`, `State`, `Any`.

**`GraphArena::verify_path(path)`** validates a sequence of node IDs:
1. Checks source node is alive (not nulled / dead)
2. Checks edge exists between consecutive nodes
3. Resolves effective contract — verifies source output type matches target input type
4. Detects cycles (non-SELF duplicates)
5. Detects self-loops and invalid SELF references

Returns `Ok(())` or specific `StructuralError` variant.

**`GraphArena::find_path(start, end)`** uses BFS to find the shortest valid traversal between two nodes, returning the path for verification.

### 2. AST-Driven Integrity (asset pipeline)

The asset pipeline no longer serializes `RenderOp` directly. All output passes through a validated intermediate representation:

**`RenderAst`** enum:
- `Scene { label, children }` — root node
- `DrawSkeleton { label, bones, color_palette, wireframe, opacity, blend_mode }`
- `ApplyTransform { target_label, translate, rotate, scale }`
- `ApplyMesh { skeleton_label, mesh_label, color_palette, blend_mode }`
- `Composite { label, sources, blend_mode, opacity }`

**`compile_to_ast(prompt) -> RenderAst`** converts a `DecomposedPrompt` into a tree of `RenderAst` nodes.

**`validate_ast(ast) -> Result<(), StructuralError>`** checks:
- Scene must have at least one child
- `ApplyTransform` must reference an existing skeleton label
- `ApplyMesh` must reference an existing skeleton label
- No conflicting transforms on the same target
- `Composite` sources must reference labels defined earlier in the scene
- All blend modes are valid

**`render_ast_to_json(ast)`** serializes the validated AST into JSON for the wgpu renderer. If validation fails, the error is embedded in the JSON output so the renderer can decide how to handle it.

### 3. Runtime Verification Loop (Phase 6)

After every cognitive tick's valence update (Phase 5), a verification phase runs before the firing history advances:

**`VerificationLoop::verify(ctx, modulators, energy_before, energy_after, fired_chain) -> Vec<VerificationEvent>`**:

Checks performed:
- **Energy conservation**: Sum absolute activation before Phase 3 vs after Phase 3. If discrepancy > 1%, a `StructuralError::EnergyNonConservation` is emitted.
- **Path integrity**: The chain of nodes that fired this tick is passed through `GraphArena::verify_path()`. If it fails, a `StructuralError::ContractMismatch` (or other error) is emitted.

On structural fault:
- `modulators.spike_novelty(discrepancy * 0.5)` — novelty spikes proportional to the violation
- Valence of every node in the faulty chain drops by -0.05 per 1% energy discrepancy
- Faulty edges are marked for pruning (`dynamic_weight = 0.0`)
- The `ActivationEngine.structural_faults` Vec accumulates all `VerificationEvent`s for the daemon to log

The system hurts when it's wrong — deterministically. Structural errors are NOT silent failures. They propagate as novelty (making thresholds easier to fire), valence drops (making faulty concepts less liked), and edge pruning (removing corrupted connections).

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Android Phone                       │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  ForegroundService (START_STICKY + WakeLock)     │ │
│  │  ├─ Sensor listeners (accel, proximity, light)   │ │
│  │  ├─ Keepalive watchdog (every 5s)                │ │
│  │  └─ UniFFI bridge calls → Rust .so               │ │
│  └──────────────────────┬───────────────────────────┘ │
│                         │                              │
│  ┌──────────────────────▼───────────────────────────┐ │
│  │              Rust Runtime (native thread)         │ │
│  │                                                    │ │
│  │  ┌─────────────┐   ┌──────────────────────────┐  │ │
│  │  │ EventChannel│──►│ CognitiveDaemon (16ms)   │  │ │
│  │  │ (SPSC 128)  │   │ ├─ Phase 1-5 tick       │  │ │
│  │  │             │   │ ├─ Phase 6: verify       │  │ │
│  │  │ Sensor data │   │ ├─ Output channel        │  │ │
│  │  │ Intent JSON │   │ └─ Graph commands        │  │ │
│  │  └─────────────┘   └──────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Curiosity Harvester (Tokio async)        │  │ │
│  │  │ ├─ GapDetector → KnowledgeGap            │  │ │
│  │  │ ├─ DefinitionResolver (6 grammar rules)  │  │ │
│  │  │ ├─ KnowledgeStore (~30 foundation cones) │  │ │
│  │  │ └─ Circuit breaker at depth 10           │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Asset Ingestor                           │  │ │
│  │  │ ├─ ComponentExtractor                    │  │ │
│  │  │ ├─ TransformEngine                       │  │ │
│  │  │ ├─ compile_to_ast() → validate_ast()     │  │ │
│  │  │ └─ render_ast_to_json()                  │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Verification Core                        │  │ │
│  │  │ ├─ VerificationLoop (Phase 6)            │  │ │
│  │  │ ├─ verify_path() / find_path()           │  │ │
│  │  │ └─ StructuralError accumulator           │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Crates

| Crate | Purpose |
|-------|---------|
| `semantic-graph` | GroundedNode (valence), GraphArena (verify_path, find_path, get_valence, set_valence, nodes_with_highest_valence, find_by_label, label_of), ActivationBuffer, Edge (STDP, contract, effective_contract, canonical_contract), FiringHistory, Neuromodulator, PredictionError, DataType, InvariantContract, StructuralError, ConceptualFrame, 10 Relation types |
| `semantic-parser` | Verb→CDAction table (30+), sensor parsing, Realizer (JSON/text) |
| `cognitive-core` | ActivationEngine (6-phase tick: neuromodulator decay / prediction error / spread+eligibility / STDP+pruning / valence update / structural verification), VerificationLoop (energy conservation, path integrity, penalties), EventChannel, CognitiveDaemon, Consolidation |
| `hw-daemon` | Android lifecycle bridge, graph persistence, keepalive, modulate/consolidate bridge |
| `curiosity-core` | Gap detection, offline knowledge resolution, async harvester |
| `asset-ingestor` | Prompt decomposition, quadruped→biped transform, RenderAst (compile_to_ast / validate_ast / render_ast_to_json) |
| `uniffi-exports` | 18-function UniFFI surface for Kotlin |

## Status

**Active development.** All crates have unit tests. The architecture is solid, but this is version 0.1 — a baby learning to crawl.

- [x] Semantic graph with arena allocation + double-buffered activation
- [x] 6-phase spreading activation (decay/inject/spread/fire/valence/verify)
- [x] 30+ verb→CD action mappings
- [x] Hebbian STDP: dynamic_edge_weight, eligibility traces, LTP on co-firing, LTD drift, pruning
- [x] Neuromodulation: novelty, arousal, reward channels with per-tick decay and global threshold/plasticity modulation
- [x] Predictive coding: activation predictions, |error| > 30% novelty spike
- [x] Sleep consolidation: edge GC, linear chain compression
- [x] Lock-free SPSC event channel (128 slots)
- [x] Self node: `NodeId::SELF` (index 1) pre-inserted in every graph with base activation 1.0, decay 1.0 (never decays)
- [x] Self-linking: every sensor reading, intent, and fired action anchors to SELF via `link_to_self()`
- [x] `introspect()` — returns everything SELF is connected to
- [x] Self persists across restarts (serialized as node index 1 in bincode)
- [x] Default Mode Network: spontaneous inner activity when idle, curiosity drive, inner monologue
- [x] Valence system: nodes accumulate positive/negative experience from prediction success/failure
- [x] Preference formation: deterministic "likes" and "dislikes" emerge from prediction history
- [x] Opinion synthesis: `get_opinion(topic)` traverses the graph and produces contextual response
- [x] Mood query: `get_mood()` returns current state from neuromodulator levels
- [x] Interest query: `get_interests(count)` returns top-N highest-valence concepts
- [x] Birth signal: first tick wakes up SELF, links to sensors, announces awareness
- [x] Recursive curiosity harvester (Tokio, Semaphore(4), depth 10)
- [x] 6 grammar patterns for offline definition resolution
- [x] Quadruped→biped geometric transform
- [x] UniFFI bridge to Kotlin
- [x] Android ForegroundService with wakelock + keepalive watchdog
- [x] **Edge invariants**: `InvariantContract`, `DataType`, `StructuralError`, `Edge.contract`, `canonical_contract()`, `verify_path()`, `find_path()`
- [x] **AST-driven pipeline**: `RenderAst` enum, `compile_to_ast()`, `validate_ast()`, `render_ast_to_json()` — all render ops compile through validated AST
- [x] **Runtime verification**: `VerificationLoop` (Phase 6), energy conservation check, path integrity, novelty/valence/edge penalties on fault, `ActivationEngine.structural_faults`
- [ ] `cargo test` pass (needs actual test environment)
- [ ] Integration: persist graph → survive restart → resume curiosity with self node intact
- [ ] Integration: prompt → render ops → wgpu draws skeleton

## Why not just use an LLM?

| | LLM | Grounded |
|---|---|---|
| **Understanding** | Statistical token prediction | Relational graph traversal |
| **Grounding** | None (floating symbols) | Every node traces to sensor or action |
| **Determinism** | None (temperature sampling) | Deterministic per tick; graph grows over time |
| **Offline** | No (needs datacenter) | Yes (runs on a phone) |
| **Power** | Watts × 1000 | Milliwatts |
| **Learning** | Pre-trained on all of internet | Grows through interaction |
| **Size** | GB–TB | KB–MB |
| **Honesty** | Hallucinates constantly | Can only traverse what it knows |
| **Self-correction** | Fine-tuning / RLHF | Structural errors spike deviation gain and drop valence automatically |

Grounded doesn't guess. It traverses what it knows. When it doesn't know something, it *knows* it doesn't know — that's a knowledge gap, and it resolves it. When the graph violates its own structural rules, the verification loop makes it hurt. Ask it the same thing next week and it may have a better answer.

## Build

```bash
# Check all crates
cargo check

# Run tests
cargo test
```

---

**Grounded is not an AI. It's something else entirely.**
