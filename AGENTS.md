# Grounded — Agent Instructions

## Vision

A non-AI, deterministic cognitive system that runs 100% on-device. No tokens, no embeddings, no LLMs. It builds a relational semantic graph grounded to real sensor values and real platform actions. Knowledge is acquired through recursive curiosity — it detects gaps in its own understanding and resolves them autonomously, bounded by a circuit breaker at depth 10.

## Core Principles

1. **Formal Semantics** — Every concept is a node in a directed graph. Every edge is a typed relation (IsA, HasProperty, Requires, Activates, Inhibits...). Meaning is the set of relations a node participates in.

2. **Symbol Grounding** — Sensor nodes map raw accelerometer/proximity/light values into activation energy via deterministic normalization. Action nodes produce real Android intents. No symbol floats free — every concept traces back to sensor or action via edges.

3. **Conceptual Dependency (CD)** — Schank's CD primitives (Atrans, Ptrans, Mtrans, Mbuild, Propel, etc.) are the atomic verbs. All complex actions decompose into these primitives.

4. **Recursive Curiosity** — The engine autonomously detects knowledge gaps (tokens with zero relational edges) and resolves them via an offline knowledge base, recursively until all leaf concepts are fundamental physical primitives. Circuit breaker at depth 10.

5. **Deterministic** — No random numbers. No probability. Same input → same output, always. Every algorithm is O(E) bounded-time, no fallbacks, no guesswork.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Grounded Engine                           │
│                                                              │
│  ┌─────────────┐   ┌──────────────────────────────────┐     │
│  │  Android     │   │         Rust Runtime              │     │
│  │  Foreground  │   │                                   │     │
│  │  Service     │   │  ┌────────────────────────────┐   │     │
│  │              │   │  │  Cognitive Daemon (16ms)   │   │     │
│  │  Wakelock    │◄──┤  │  ┌──────────────────────┐  │   │     │
│  │  Sensors     │   │  │  │ ActivationEngine     │  │   │     │
│  │  Notification│   │  │  │ ├─ Decay phase       │  │   │     │
│  │              │   │  │  │ ├─ Inject phase      │  │   │     │
│  │  UniFFI      ├───┤  │  │ ├─ Spread phase      │  │   │     │
│  │  Bridge      │   │  │  │ └─ Fire phase        │  │   │     │
│  └─────────────┘   │  │  └──────────────────────┘  │   │     │
│                     │  │                            │   │     │
│                     │  │  EventChannel (SPSC 128)   │   │     │
│                     │  │  OutputChannel (RwLock)    │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  Curiosity Harvester       │   │     │
│                     │  │  (Tokio async, 4 concurrent)│   │     │
│                     │  │  ├─ GapDetector            │   │     │
│                     │  │  ├─ DefinitionResolver     │   │     │
│                     │  │  ├─ KnowledgeStore (offline)│   │     │
│                     │  │  └─ Recursion breaker /10  │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  Asset Ingestor            │   │     │
│                     │  │  ├─ ComponentExtractor     │   │     │
│                     │  │  ├─ TransformEngine        │   │     │
│                     │  │  └─ RenderOp pipeline      │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  SemanticContext (Arc)      │   │     │
│                     │  │  ├─ GraphArena (Vec<RwLock>)│   │     │
│                     │  │  └─ ActivationBuffer       │   │     │
│                     │  │     (double-buffered f64[]) │   │     │
│                     │  └────────────────────────────┘   │     │
│                     └──────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## Core Neuroscience Enhancements

### 4-Phase Tick (16ms)

```
Phase 1 — Neuromodulator Decay (O(1)):
  novelty/arousal/reward leak toward baseline (8%/12%/4%)
  Compute threshold_mod = (1.0 - novelty*0.35 - arousal*0.20).clamp(0.4, 1.0)
  Compute plasticity_mod = (0.5 + novelty*0.4 + reward*0.3).clamp(0.5, 1.5)

Phase 2 — Decay + Injection + Prediction Error (O(N)):
  activation[i] *= node.decay * 0.97
  activation[i] += injection_queue[i]
  if |activation[i] - prediction[i]| / prediction[i] > 0.3:
    → PredictionError → spike novelty by error * 0.15

Phase 3 — Spread + Eligibility (O(N+E)):
  effective_threshold = node.threshold * threshold_mod
  if activation[i] > effective_threshold:
    fire(i), record in FiringHistory bitset, activation[i] = 0
    boost eligibility on all outgoing edges (+1.0)
  else: for each edge (i→j):
    spread = activation[i] * edge.dynamic_weight * 0.15
    conservation: activation[j] += spread, activation[i] -= spread
  prediction[i] = activation[i]  (for next tick's error check)

Phase 4 — STDP + Pruning (O(E)):
  for each edge (i→j):
    eligibility *= 0.9
    if i fired this tick: eligibility += 1.0
    if j fired this tick: LTP = eligibility * 0.008 * plasticity_mod
      dynamic_weight = (dynamic_weight + LTP).clamp(-1.0, 1.0)
      eligibility *= 0.5  (consumed)
    LTD drift: dynamic_weight += (default_weight - dynamic_weight) * 0.0005
    if |dynamic_weight| < 0.005: mark for pruning, set to 0.0
```

### FiringHistory Ring Buffer
- `LTP_WINDOW=4` ticks of bitset storage
- `record_fired()`, `fired_recently(id, lookback)`, `advance_tick()`
- Zero allocations in hot path — pre-sized to node_count

### Neuromodulator Channels
| Channel | Decay/tick | Spiked by | Effect on thresholds |
|---------|-----------|-----------|---------------------|
| novelty | 8% | Prediction errors, curiosity gaps | `threshold * (1.0 - novelty*0.35)` |
| arousal | 12% | Sensor deltas > 0.5 | `threshold * (1.0 - arousal*0.20)` |
| reward | 4% | 100 ticks without prediction errors | Accelerates plasticity |

### Edge STDP Fields
Every Edge now carries:
- `dynamic_weight: f64` — used during spread, modified by STDP
- `eligibility: f64` — Hebbian trace (decay 0.9, boost on source fire, consume on target fire)
- `default_weight()` — what dynamic_weight drifts toward between LTP events

### Consolidation (sleep loop)
Triggered every ~1000 ticks (~16s) when novelty < 0.1 AND arousal < 0.1:
- `GraphArena::garbage_collect_edges()` — remove edges below PRUNE_THRESHOLD (0.005)
- `compress_linear_chains()` — A→B→C where B has indegree=1, outdegree=1 → A→C with combined weight
- Marked B nodes have threshold set to MAX (never fire again)

## Crate Map

### Core (semantic-graph)
- `GroundedNode` — label, type (Entity/Concept/Action/Sensor/State/Frame), `Grounding` enum (Sensor/Action/Stored/HardwareQuery/Abstract), decay, threshold, edges
- `Edge` — relation, target, weight_override, `dynamic_weight` (STDP), `eligibility` (Hebbian trace), `default_weight()`, `effective_weight()` (reads dynamic_weight)
- `GraphArena` — `Vec<RwLock<GroundedNode>>` + label index. O(1) access by NodeId, `garbage_collect_edges()` for pruning
- `ActivationBuffer` — two `Box<[f64]>` arrays flipped atomically. Writer writes to back, flips, reader reads from front
- `FiringHistory` — ring buffer of `LTP_WINDOW=4` ticks, bitset storage, zero-alloc in hot path
- `Neuromodulator` — novelty/arousal/reward channels with decay, spike(), `threshold_modifier()`, `plasticity_modifier()`
- `PredictionError` — node_id, expected, actual, error_magnitude
- `SemanticContext` — graph + activation + tick counter (AtomicU64)
- `ConceptualFrame` — CD primitives: Atrans, Ptrans, Mtrans, Mbuild, Propel, Ingest, Expel, Move, Grasp, Speak, Attend, SystemAction, SensorReading
- `Relation` — IsA, HasProperty, Requires, CausedBy, Implies, GroundedIn, Precedes, Activates, Inhibits, AssociatedWith (each with a deterministic spread weight)

### Parser (semantic-parser)
- `parse_intent(json)` — verb→CDAction lookup (30+ verbs), Android intent→frame mapping
- `parse_sensor_event(sensor, channel, value)` — raw sensor→SensorReading frame
- `Realizer` — template-based frame→AndroidIntent JSON or display text

### Cognitive Core (cognitive-core)
- `ActivationEngine::tick()` — 4-phase tick (neuromodulator decay / prediction error / spread+eligibility / STDP+pruning)
- `ActivationEngine` — owns `Neuromodulator`, `FiringHistory`, predictions[] array, fired_this_tick bitset
- `EventChannel` — 128-slot SPSC ring buffer, lock-free
- `CognitiveDaemon` — background thread, 16ms tick, event drain→tick→dispatch, arousal spike on sensor delta, reward tonic on stable predictions, consolidation every ~1000 ticks, `read_modulators()`
- `consolidate()` — edge GC + linear chain compression (sleep loop)

### Hardware Daemon (hw-daemon)
- `CognitiveLifecycle` — `on_create()`/`start()`/`stop()`/`on_trim_memory()` + graph persistence via bincode, `read_modulators()`
- Global bridge: `init()`, `start()`, `stop()`, `feed_sensor()`, `feed_intent()`, `drain_outputs()`, `keepalive()`, `missed_heartbeats()`, `tick_count()`, `inspect_prompt()`, `modulate()`, `trigger_consolidation()`, `read_modulators()`
- Keepalive watchdog: checks tick counter advancing. 3 missed beats (~15s) → dead

### Curiosity Core (curiosity-core)
- `KnowledgeGap` — token + parent + recursion_depth + GapSource
- `GapDetector::detect(tokens)` — checks each token against graph edges. No edges = gap
- `DefinitionResolver::resolve(token, definition)` — 6 grammar patterns (is_a/has/can/needs/causes/is_like) → predicates → graph edges
- `KnowledgeStore` — offline foundation knowledge (~30 concepts: cat, dog, pirate, bipedal, etc.)
- `AutonomousHarvester` — Tokio async loop, bounded by Semaphore(4), circuit breaker at depth 10
- `CuriosityDaemon` — high-level `inspect_prompt()` → detect → emit gaps

### Asset Ingestor (asset-ingestor)
- `ComponentExtractor::decompose(prompt)` — lexical pass → role assignment → component extraction → render op tree
- `TransformEngine::quadruped_to_biped()` — joint remapping: front legs→arms, back legs→legs, spine rotate 90°
- `AssetPipeline::process()` — full prompt→render ops flow, `realize_to_render_json()` output
- BASE_SKELETONS: human, quadruped, bird, fish with joint names

### UniFFI Exports (uniffi-exports)
15 exported functions: `init`, `start`, `stop`, `feed_sensor`, `feed_intent`, `drain_outputs`, `is_running`, `trim_memory`, `keepalive`, `missed_heartbeats`, `inspect_prompt`, `tick_count`, `modulate`, `trigger_consolidation`, `read_modulators`

## File Layout

```
grounded/
├── Cargo.toml              ← workspace root
├── README.md               ← vision and architecture
├── AGENTS.md               ← this file
├── crates/
│   ├── semantic-graph/     ← GroundedNode, GraphArena, ActivationBuffer, CD primitives
│   ├── semantic-parser/    ← Intent/sensor parsing, Realizer
│   ├── cognitive-core/     ← ActivationEngine (4-phase), CognitiveDaemon, EventChannel, Consolidation
│   ├── hw-daemon/          ← Lifecycle, global bridge, keepalive
│   ├── curiosity-core/     ← Gap detection, offline resolver, async harvester
│   ├── asset-ingestor/     ← Multi-modal prompt decomposition, transform engine
│   └── uniffi-exports/     ← 12-export UniFFI surface
└── android/
    └── app/src/main/kotlin/com/grounded/engine/
        └── CognitiveForegroundService.kt
```

## Key Decisions

- **FSM is emergent from graph topology** — no match statement or enum to extend. New actions = insert Action node. New rules = insert edge. Everything is data, not code.
- **Arena allocation** — Vec<RwLock<GroundedNode>> indexed by NodeId. Slots never freed during hot loop. Compaction is explicit (persist/restart).
- **Double-buffer activation** — two Box<[f64]> arrays toggled by AtomicU8. Writer writes to back, flips. Reader reads from active. No locks in hot path.
- **Curiosity runs on Tokio** (separate async runtime) because it may do async I/O. Bounded by Semaphore(4) for battery.
- **Verb→CDAction via hardcoded table** — 30+ entries, no stemming/embedding. Pure string match.
- **Knowledge base is offline, compiled-in** — ~30 foundational concepts at install time, grown via recursive resolution at runtime.
- **STDP is NOT machine learning.** It is a deterministic O(E) edge-weight update based on co-firing within a 4-tick window. Same firing pattern → same weight change, always. No randomness, no gradient, no probability. Eligibility traces are exponential decay curves, not learned parameters.
- **Spreading activation is NOT machine learning.** Same input → same output, always. Energy decays to zero if no injection occurs. No gradients, no weights, no training.
- **Quadruped→biped transform is hardcoded matrix math** — front legs→arms (rotate 90°, attach upper torso), back legs→legs (stretch by 1.2x), spine rotate -90°→vertical. Pure geometry, no learning.

## Keepalive Design

The Kotlin `CognitiveForegroundService` starts a `java.util.Timer` that calls `keepalive()` every 5 seconds. The Rust side checks whether `SemanticContext.tick` has advanced since the last call. If the tick is stuck for 3 consecutive checks (~15s), the service `stop()`s and `start()`s the engine, also renewing the 4h PARTIAL_WAKE_LOCK. This ensures the cognitive daemon stays alive even if the thread crashes silently.

## Build

```bash
cargo check     # check all crates
cargo test      # run all tests
cargo build     # build debug
cargo build --release  # build release
```

## Tests

- **semantic-graph**: serialization roundtrip, arena operations
- **cognitive-core**: activation propagation, energy decay, activation reads, STDP edge reinforcement, prediction error novelty spike, neuromodulator threshold modulation
- **semantic-parser**: open camera intent, sensor event roundtrip, verb table
- **curiosity-core**: missing token detection, known token grounding, predicate parsing, dependency discovery
- **asset-ingestor**: pirate cat decomposition, quadruped→biped transform, render op generation
