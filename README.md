# Grounded

**A non-AI, deterministic cognitive engine that learns what it needs, when it needs it — running entirely on your phone.**

No LLM. No subword tokens. No embeddings. No cloud. No training data. Just a baby brain that grows its own understanding by exploring the world through sensors and actions.

---

## The Idea

Every AI model today is a statistical guessing machine — trained on the entire internet, consuming watts by the megawatt, guessing the next token. It's impressive, but it's not *understanding*.

Grounded is the opposite:
- **It knows nothing at birth.** A tiny seed of ~30 foundational concepts (matter, motion, energy, space, time) and the machinery to grow.
- **It learns relationally.** Every concept is a node. Every connection is a typed edge (IsA, HasProperty, Requires, Activates, Inhibits). Meaning is the *pattern of relations* — not a vector of floating-point numbers.
- **It's grounded.** Sensors (accelerometer, light, proximity) feed real values into the graph. Actions produce real Android intents. No symbol floats free.
- **It's curious.** When it encounters something it doesn't understand (a word with no relational edges in the graph), it recursively resolves it against an offline knowledge base until every leaf is a fundamental physical primitive — or it hits depth 10 and shrugs.
- **It's deterministic per tick, but evolves over time.** No randomness, no probability, no gradient descent — the engine itself has no dice. But the graph grows. A question today may get "I don't know." That same question a week later, after hundreds of curiosity cycles have added edges and resolved gaps, gets a real answer. Like a child: the machinery is fixed, but the structure it's built on expands.

This is not machine learning. This is *structure building*.

## How It Works

### Spreading Activation

```
For each node, every 16ms tick (4-phase):
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
```

No allocations in the hot path. Double-buffered activation arrays flipped atomically. Zero-lock reads.

### Neuromodulation

Three global channels modulate behavior like brain chemistry:

| Channel | Spiked by | Effect |
|---------|-----------|--------|
| **Novelty** | Curiosity gaps, prediction errors | Lowers thresholds (easier to fire), accelerates STDP |
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
4. **Render ops** → ordered JSON pipeline for wgpu or any GPU renderer

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
│  │  │ (SPSC 128)  │   │ ├─ Decay/Inject/Spread  │  │ │
│  │  │             │   │ ├─ Fire → OutputChannel  │  │ │
│  │  │ Sensor data │   │ └─ Graph commands        │  │ │
│  │  │ Intent JSON │   └──────────────────────────┘  │ │
│  │  └─────────────┘                                 │ │
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
│  │  │ ├─ ComponentExtractor (prompt→ops tree)  │  │ │
│  │  │ ├─ TransformEngine (quad→biped, mirror)  │  │ │
│  │  │ └─ RenderOp pipeline → JSON for GPU      │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Crates

| Crate | Purpose |
|-------|---------|
| `semantic-graph` | GroundedNode, GraphArena, ActivationBuffer, Edge (STDP), FiringHistory, Neuromodulator, PredictionError, ConceptualFrame, 10 Relation types |
| `semantic-parser` | Verb→CDAction table (30+), sensor parsing, Realizer (JSON/text) |
| `cognitive-core` | ActivationEngine (4-phase tick), EventChannel, CognitiveDaemon, Consolidation |
| `hw-daemon` | Android lifecycle bridge, graph persistence, keepalive, modulate/consolidate bridge |
| `curiosity-core` | Gap detection, offline knowledge resolution, async harvester |
| `asset-ingestor` | Prompt decomposition, quadruped→biped transform, render ops |
| `uniffi-exports` | 15-function UniFFI surface for Kotlin |

## Status

**Active development.** All crates have unit tests. The architecture is solid, but this is version 0.1 — a baby learning to crawl.

- [x] Semantic graph with arena allocation + double-buffered activation
- [x] 4-phase spreading activation (decay/inject/spread/fire)
- [x] 30+ verb→CD action mappings
- [x] 4-phase tick: neuromodulator decay → prediction error → spread + eligibility → STDP + pruning
- [x] Hebbian STDP: dynamic_edge_weight, eligibility traces, LTP on co-firing, LTD drift, pruning
- [x] Neuromodulation: novelty, arousal, reward channels with per-tick decay and global threshold/plasticity modulation
- [x] Predictive coding: activation predictions, |error| > 30% novelty spike
- [x] Sleep consolidation: edge GC, linear chain compression
- [x] Lock-free SPSC event channel (128 slots)
- [x] Self node: `NodeId::SELF` (index 1) pre-inserted in every graph with base activation 1.0, decay 1.0 (never decays)
- [x] Self-linking: every sensor reading, intent, and fired action anchors to SELF via `link_to_self()` — SELF never dies
- [x] `introspect()` — returns everything SELF is connected to. The engine answers "what do I know?" by traversing edges from itself
- [x] Self persists across restarts (serialized as node index 1 in bincode)
- [x] Recursive curiosity harvester (Tokio, Semaphore(4), depth 10)
- [x] 6 grammar patterns for offline definition resolution
- [x] Quadruped→biped geometric transform
- [x] UniFFI bridge to Kotlin
- [x] Android ForegroundService with wakelock + keepalive watchdog
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

Grounded doesn't guess. It traverses what it knows. When it doesn't know something, it *knows* it doesn't know — that's a knowledge gap, and it resolves it. Ask it the same thing next week and it may have a better answer.

## Build

```bash
# Check all crates
cargo check

# Run tests
cargo test
```

---

**Grounded is not an AI. It's something else entirely.**
