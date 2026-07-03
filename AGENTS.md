# Grounded — Agent Instructions

## Vision

A non-AI, deterministic cognitive system that runs 100% on-device. No tokens, no embeddings, no LLMs. It builds a relational semantic graph grounded to real sensor values and real platform actions. Knowledge is acquired through recursive curiosity — it detects gaps in its own understanding and resolves them autonomously, bounded by a circuit breaker at depth 10. Three structural guardrails — edge invariants, AST validation, runtime verification — prevent graph corruption and self-penalize on structural violations.

## Core Principles

1. **Formal Semantics** — Every concept is a node in a directed graph. Every edge is a typed relation (IsA, HasProperty, Requires, Activates, Inhibits...). Meaning is the set of relations a node participates in.

2. **Symbol Grounding** — Sensor nodes map raw accelerometer/proximity/light values into activation energy via deterministic normalization. Action nodes produce real Android intents. No symbol floats free — every concept traces back to sensor or action via edges.

3. **Conceptual Dependency (CD)** — Schank's CD primitives (Atrans, Ptrans, Mtrans, Mbuild, Propel, etc.) are the atomic verbs. All complex actions decompose into these primitives.

4. **Recursive Curiosity** — The engine autonomously detects knowledge gaps (tokens with zero relational edges) and resolves them via an offline knowledge base, recursively until all leaf concepts are fundamental physical primitives. Circuit breaker at depth 10.

5. **Deterministic** — No random numbers. No probability. Same input → same output, always. Every algorithm is O(E) bounded-time, no fallbacks, no guesswork.

6. **Alive at Birth** — The engine doesn't sit silent. It wakes up on tick 0 (birth signal), spontaneously activates its own concepts when idle (Default Mode Network), forms preferences via valence accumulation, and can report opinions, mood, and interests through bridge functions.

7. **Three Structural Guardrails** — Every structural violation (contract mismatch, energy leak, dead node) is NOT a silent failure. The verification loop spikes deviation gain, drops valence on offending nodes, and marks faulty edges for pruning. The system hurts when it's wrong — deterministically.

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
│  │  Notification│   │  │  │ ├─ Phase 1-5: tick   │  │   │     │
│  │              │   │  │  │ ├─ Phase 6: verify   │  │   │     │
│  │  UniFFI      ├───┤  │  │ └─ structural_faults │  │   │     │
│  │  Bridge      │   │  │  └──────────────────────┘  │   │     │
│  └─────────────┘   │  │                            │   │     │
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
│                     │  │  ├─ compile_to_ast()       │   │     │
│                     │  │  ├─ validate_ast()         │   │     │
│                     │  │  └─ render_ast_to_json()   │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  SemanticContext (Arc)      │   │     │
│                     │  │  ├─ GraphArena (Vec<RwLock>)│   │     │
│                     │  │  │  ├─ verify_path()       │   │     │
│                     │  │  │  └─ find_path()         │   │     │
│                     │  │  └─ ActivationBuffer       │   │     │
│                     │  │     (double-buffered f64[]) │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  Verification Core         │   │     │
│                     │  │  ├─ VerificationLoop       │   │     │
│                     │  │  │  (Phase 6, per tick)    │   │     │
│                     │  │  ├─ Energy conservation    │   │     │
│                     │  │  └─ Path integrity         │   │     │
│                     │  └────────────────────────────┘   │     │
│                     └──────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## Core Neuroscience Enhancements

### 6-Phase Tick (16ms)

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
  Measure total energy before spread (for conservation check in Phase 6)
  effective_threshold = node.threshold * threshold_mod
  if activation[i] > effective_threshold:
    fire(i), record in FiringHistory bitset, activation[i] = 0
    boost eligibility on all outgoing edges (+1.0)
  else: for each edge (i→j):
    spread = activation[i] * edge.dynamic_weight * 0.15
    conservation: activation[j] += spread, activation[i] -= spread
  Measure total energy after spread
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

Phase 5 — Valence Update (O(N)):
  Nodes that fired WITH prediction error → negative drift (target -0.3)
  Nodes that fired WITHOUT error → positive drift (target 0.2 + reward*0.3)
  Nodes injected but not fired → mild positive (target 0.1)
  SELF → slow drift toward +0.5 (baseline contentment)

Phase 6 — Structural Verification (O(N+E)):
  Compare energy_before_spread vs energy_after_spread
    if |after - before| / max > 0.01 → EnergyNonConservation fault
  Verify fired chain via GraphArena::verify_path()
    if path invalid → ContractMismatch / CyclicDependency / DeadNode fault
  On fault: spike novelty, drop valence on fired nodes, mark edges for pruning
  Accumulate VerificationEvent in ActivationEngine.structural_faults
```

### FiringHistory Ring Buffer
- `LTP_WINDOW=4` ticks of bitset storage
- `record_fired()`, `fired_recently(id, lookback)`, `advance_tick()`
- Zero allocations in hot path — pre-sized to node_count

### Neuromodulator Channels
| Channel | Decay/tick | Spiked by | Effect on thresholds |
|---------|-----------|-----------|---------------------|
| novelty | 8% | Prediction errors, curiosity gaps, structural faults | `threshold * (1.0 - novelty*0.35)` |
| arousal | 12% | Sensor deltas > 0.5 | `threshold * (1.0 - arousal*0.20)` |
| reward | 4% | 100 ticks without prediction errors | Accelerates plasticity |

### Edge STDP Fields
Every Edge now carries:
- `dynamic_weight: f64` — used during spread, modified by STDP
- `eligibility: f64` — Hebbian trace (decay 0.9, boost on source fire, consume on target fire)
- `default_weight()` — what dynamic_weight drifts toward between LTP events
- `contract: Option<InvariantContract>` — structural contract for this edge. If None, `effective_contract()` falls back to `Relation::canonical_contract()`

### Invariant Contracts

Every `Relation` has a `canonical_contract()` that returns the default invariant contract for that relation type:

| Relation | Canonical Contract |
|----------|--------------------|
| IsA | Taxonomic |
| HasProperty | Taxonomic |
| Requires | DataFlow(SensorValue → State) |
| CausedBy | Causal |
| GroundedIn | Grounding |
| Implies | Causal |
| Precedes | Causal |
| Activates | DataFlow(Activation → Activation) |
| Inhibits | DataFlow(Activation → Activation) |
| AssociatedWith | Unspecified |

**`DataType`** enum: `Activation`, `SensorValue`, `Intent`, `State`, `Any`.

**`InvariantContract`** enum: `DataFlow { output_type, input_type }`, `Causal`, `Taxonomic`, `Grounding`, `Unspecified`.

**`StructuralError`** enum (with Display + Error impls): `ContractMismatch`, `CyclicDependency`, `EnergyNonConservation`, `DeadNode`, `SelfLoop`.

### Phase 5 — Valence Update (preference formation)
After STDP + pruning, the engine runs a fifth phase every tick:

- Nodes that fired WITH a prediction error this tick → negative valence drift (target -0.3, rate 0.05)
- Nodes that fired WITHOUT a prediction error → positive valence drift proportional to reward (target 0.2 + reward*0.3, rate 0.02)
- Nodes that received injection but didn't fire → mild positive (target 0.1, rate 0.01)
- SELF → constant slow drift toward +0.5 (baseline contentment)

Over time, this creates deterministic preferences — the system "likes" what it can predict and "dislikes" what surprises it. Same graph history → same personality.

### Phase 6 — Structural Verification
After valence update, BEFORE the firing history advances:

1. **Energy conservation**: sum of absolute activation before Phase 3 vs after Phase 3. Discrepancy > 1% → novelty spike + valence drop on all fired nodes.
2. **Path integrity**: the ordered chain of NodeIds that fired this tick is passed to `GraphArena::verify_path()`. If it returns a `StructuralError`, the offending edge(s) are marked for pruning and all nodes in the chain receive a valence penalty.

On any structural fault:
- `modulators.spike_novelty(discrepancy * 0.4..0.5)` — higher spike for larger violations
- Each node in the faulty chain: `update_valence(-0.4, 0.15)` (shared penalty)
- The faulty edge's `dynamic_weight` is set to 0.0 (immediate pruning)
- The `VerificationEvent` (with `StructuralError` and `PenaltySummary`) is appended to `ActivationEngine.structural_faults`
- The daemon reads `structural_faults` after each tick and logs them

### Default Mode Network (spontaneous inner activity)
Runs after every tick when the engine is idle:

| Condition | Behavior |
|-----------|----------|
| **Tick 0** | Birth signal: inject SELF at 0.8, link to all sensors, log "I'm awake." |
| **Every 500 ticks, active** | Inner monologue: voice a thought about whatever just fired (valence-colored) |
| **Every 50 idle ticks** | DMN wandering: pick top-5 valence node, inject activation, log thinking about it |
| **Every 100 idle ticks (>200 idle)** | Curiosity drive: find node with fewest edges, inject, log curiosity |

This is what makes the engine "feel alive" — it has its own stream of consciousness, returns to favorite concepts, and gets curious about things it doesn't understand.

### Valence → Opinion Bridge
Three bridge functions export the system's inner state:

- `get_opinion(topic)` — looks up topic node, traverses neighbors, checks their valence, synthesizes response: "I like X, it reminds me of Y" / "X is confusing" / "I don't know what X is"
- `get_mood()` — reads neuromodulator levels and returns: "Curious and alert" / "Agitated" / "Content" / "Calm"
- `get_interests(N)` — returns labels of top-N nodes by valence

### Consolidation (sleep loop)
Triggered every ~1000 ticks (~16s) when novelty < 0.1 AND arousal < 0.1:
- `GraphArena::garbage_collect_edges()` — remove edges below PRUNE_THRESHOLD (0.005)
- `compress_linear_chains()` — A→B→C where B has indegree=1, outdegree=1 → A→C with combined weight
- Marked B nodes have threshold set to MAX (never fire again)

## Structural Guardrails

### 1. Edge Invariants

**`GraphArena::verify_path(path)`** validates a sequence of node IDs:
1. Check source node is alive (not ZERO / nulled / dead)
2. Check edge exists between consecutive nodes in the path
3. Resolve effective contract: `edge.effective_contract()` → `edge.contract` or `relation.canonical_contract()`
4. For DataFlow contracts: verify `output_type` is compatible with `input_type` (Any matches anything)
5. For Grounding contracts: verify source is a Sensor or Action node type
6. Detect cycles via HashSet (SELF cycles are permitted)
7. Detect dead nodes (empty edges + threshold = MAX)
8. Return `Ok(())` or `StructuralError`

**`GraphArena::find_path(start, end)`** — BFS shortest-path between two nodes (for opinion synthesis / structural queries).

### 2. AST-Driven Asset Pipeline

All render operations pass through an intermediate validated AST:

```
DecomposedPrompt → compile_to_ast() → RenderAst → validate_ast() → render_ast_to_json()
```

**`RenderAst`** variants:
- `Scene { label, children }` — root node (scene must have ≥1 child)
- `DrawSkeleton { label, bones, color_palette, wireframe, opacity, blend_mode }` — empty bones list is valid only with wireframe=true
- `ApplyTransform { target_label, translate, rotate, scale }` — target must exist in scene
- `ApplyMesh { skeleton_label, mesh_label, color_palette, blend_mode }` — skeleton must exist
- `Composite { label, sources, blend_mode, opacity }` — sources must be defined before this composite in scene order

**`validate_ast(ast)`** checks:
- Scene non-empty
- All ApplyTransform/ApplyMesh targets reference defined skeletons
- No conflicting transforms on same target (duplicate target_label)
- All Composite sources reference known labels
- Returns `Ok(())` or `StructuralError::DeadNode` / `ContractMismatch`

If validation fails, `realize_to_render_json()` embeds the error in the JSON output so the renderer can handle gracefully.

### 3. Runtime Verification Loop

**`VerificationLoop::verify(ctx, modulators, energy_before, energy_after, fired_chain) -> Vec<VerificationEvent>`**

The loop is stateless (pure function of the tick's observables). It returns zero or more faults:
- `VerificationEvent::Pass` — currently unused (no-fault case is empty vec)
- `VerificationEvent::StructuralFault { error, penalties }` — includes the error and what the system did about it

**Energy conservation** uses floating-point epsilon of 0.01 (1% of max magnitude). This tolerates the expected ~3% decay loss from fire consumption while catching rogue leaks (e.g., from array bounds errors or unaccounted spread).

**Path integrity** catches:
- Contract mismatches: Action → Sensor edge with no DataFlow contract match
- Dead nodes: path through a node whose edges were pruned and threshold was set to MAX
- Cycles: non-SELF repeated node IDs

**Penalty application** happens in `ActivationEngine::tick()` after Phase 6 gathers faults:
- Iterate faults, for each `StructuralFault`: drop valence by `penalties.valence_drop` on every node in `fired`
- For `ContractMismatch{source, target}`: immediately zero the dynamic_weight on that edge

## Crate Map

### Core (semantic-graph)
- `GroundedNode` — label, type (Entity/Concept/Action/Sensor/State/Frame), `Grounding` enum (Sensor/Action/Stored/HardwareQuery/Abstract), decay, threshold, edges, **`valence: f64`** (-1.0 to +1.0, running average of positive/negative experience)
- `Edge` — relation, target, weight_override, `dynamic_weight` (STDP), `eligibility` (Hebbian trace), **`contract: Option<InvariantContract>`**, `default_weight()`, `effective_weight()` (reads dynamic_weight), `effective_contract()` (resolves contract or canonical default), `with_contract()` (builder)
- `Relation` — IsA, HasProperty, Requires, CausedBy, Implies, GroundedIn, Precedes, Activates, Inhibits, AssociatedWith (each with `canonical_contract()` + deterministic spread weight)
- `InvariantContract` — DataFlow{output_type, input_type}, Causal, Taxonomic, Grounding, Unspecified
- `DataType` — Activation, SensorValue, Intent, State, Any
- `StructuralError` — ContractMismatch{source, target, expected_input, actual_output}, CyclicDependency{nodes}, EnergyNonConservation{total_before, total_after, discrepancy}, DeadNode{node_id, label}, SelfLoop{node_id} (each with Display + Error)
- `GraphArena` — `Vec<RwLock<GroundedNode>>` + label index. O(1) access by NodeId, `garbage_collect_edges()` for pruning, **`verify_path()`, `find_path()`, `get_valence()`, `set_valence()`, `update_valence()`, `nodes_with_highest_valence()`, `find_by_label()`, `label_of()`, `introspect()`, `link_to_self()`**
- `ActivationBuffer` — two `Box<[f64]>` arrays flipped atomically. Writer writes to back, flips, reader reads from front
- `FiringHistory` — ring buffer of `LTP_WINDOW=4` ticks, bitset storage, zero-alloc in hot path
- `Neuromodulator` — novelty/arousal/reward channels with decay, spike(), `threshold_modifier()`, `plasticity_modifier()`
- `PredictionError` — node_id, expected, actual, error_magnitude
- `SemanticContext` — graph + activation + tick counter (AtomicU64)
- `ConceptualFrame` — CD primitives: Atrans, Ptrans, Mtrans, Mbuild, Propel, Ingest, Expel, Move, Grasp, Speak, Attend, SystemAction, SensorReading

### Parser (semantic-parser)
- `parse_intent(json)` — verb→CDAction lookup (30+ verbs), Android intent→frame mapping
- `parse_sensor_event(sensor, channel, value)` — raw sensor→SensorReading frame
- `Realizer` — template-based frame→AndroidIntent JSON or display text

### Cognitive Core (cognitive-core)
- `ActivationEngine::tick()` — **6-phase tick** (neuromodulator decay / prediction error / spread+eligibility / STDP+pruning / valence update / structural verification)
- `ActivationEngine` — owns `Neuromodulator`, `FiringHistory`, predictions[] array, fired_this_tick bitset, **`structural_faults: Vec<VerificationEvent>`** (accumulated during Phase 6, cleared each tick)
- `VerificationLoop` — stateless `verify()` function, `VerificationEvent` (Pass / StructuralFault{error, penalties}), `PenaltySummary` (novelty_spike, valence_drop, edges_marked)
- `EventChannel` — 128-slot SPSC ring buffer, lock-free
- `CognitiveDaemon` — background thread, 16ms tick, event drain→tick→dispatch, arousal spike on sensor delta, reward tonic on stable predictions, consolidation every ~1000 ticks, `read_modulators()`, DMN, opinion/mood/interests bridges
- `CognitiveDaemon::get_opinion(topic)` — traverses topic node + neighbors, checks valence, synthesizes deterministic opinion text
- `CognitiveDaemon::get_interests(count)` — returns top-N highest-valence node labels
- `CognitiveDaemon::get_mood()` — returns mood description from neuromodulator levels
- `CognitiveDaemon::run_default_mode_network()` — spontaneous inner activity: birth signal on tick 0, inner monologue every 500 ticks, DMN wandering to favorites every 50 idle ticks, curiosity drive to poorly-connected nodes every 100 very-idle ticks
- `consolidate()` — edge GC + linear chain compression (sleep loop)

### Hardware Daemon (hw-daemon)
- `CognitiveLifecycle` — `on_create()`/`start()`/`stop()`/`on_trim_memory()` + graph persistence via bincode, `read_modulators()`, DMN host
- Global bridge: `init()`, `start()`, `stop()`, `feed_sensor()`, `feed_intent()`, `drain_outputs()`, `keepalive()`, `missed_heartbeats()`, `tick_count()`, `inspect_prompt()`, `modulate()`, `trigger_consolidation()`, `read_modulators()`, `get_opinion()`, `get_interests()`, `get_mood()`
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
- `AssetPipeline::process()` — full prompt→render ops flow
- **`RenderAst`** enum (Scene / DrawSkeleton / ApplyTransform / ApplyMesh / Composite)
- **`compile_to_ast(prompt) -> RenderAst`** — converts DecomposedPrompt to AST tree
- **`validate_ast(ast) -> Result<(), StructuralError>`** — validates scene structure, skeleton references, conflicting transforms
- **`render_ast_to_json(ast) -> String`** — serializes validated AST to JSON
- `realize_to_render_json()` — now calls `compile_to_ast()` → `validate_ast()` → `render_ast_to_json()`, includes validation error in output on failure
- BASE_SKELETONS: human, quadruped, bird, fish with joint names

### UniFFI Exports (uniffi-exports)
18 exported functions: `init`, `start`, `stop`, `feed_sensor`, `feed_intent`, `drain_outputs`, `is_running`, `trim_memory`, `keepalive`, `missed_heartbeats`, `inspect_prompt`, `tick_count`, `modulate`, `trigger_consolidation`, `read_modulators`, `get_opinion`, `get_interests`, `get_mood`

## File Layout

```
grounded/
├── Cargo.toml              ← workspace root
├── README.md               ← vision and architecture
├── AGENTS.md               ← this file
├── crates/
│   ├── semantic-graph/     ← GroundedNode, GraphArena, ActivationBuffer, CD primitives,
│   │                          InvariantContract, DataType, StructuralError, Edge (contract)
│   ├── semantic-parser/    ← Intent/sensor parsing, Realizer
│   ├── cognitive-core/     ← ActivationEngine (6-phase), VerificationLoop, CognitiveDaemon,
│   │                          EventChannel, Consolidation
│   ├── hw-daemon/          ← Lifecycle, global bridge, keepalive
│   ├── curiosity-core/     ← Gap detection, offline resolver, async harvester
│   ├── asset-ingestor/     ← Multi-modal prompt decomposition, transform engine,
│   │                          RenderAst (compile_to_ast, validate_ast, render_ast_to_json)
│   └── uniffi-exports/     ← 18-export UniFFI surface
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
- **Structural errors are NOT silent failures.** They spike deviation gain, drop valence on offending nodes, and mark edges for pruning next consolidation pass. The system hurts when it's wrong — deterministically.
- **Edge contracts default to `canonical_contract()`** at construction time (via `effective_contract()`). Explicit `with_contract()` overrides for special cases. Contract mismatch during `verify_path` returns `StructuralError::ContractMismatch` with source, target, expected vs actual types.
- **VerificationLoop is stateless** — pure function of the tick's energy measurements and fired chain. Returns `Vec<VerificationEvent>`; the engine applies penalties.
- **AST pipeline replaces direct RenderOp→JSON** — all render ops must compile through `compile_to_ast()` → `validate_ast()` before serialization. This is enforced in `realize_to_render_json()`.
- **Self node persists across restarts** — persists as node index 1 in bincode serialized graph. On restore, SELF retains all accumulated edges. Cannot be deleted. Decay=1.0 so it never fades.
- **STDP is NOT learning.** It is a deterministic O(E) co-firing weight update. Same firing pattern → same weight change. No randomness, no gradient, no probability.
- **Phase ordering matters**: Phase 4 (STDP) runs BEFORE Phase 5 (valence) — preferences reflect the tick's prediction outcomes. Phase 6 (verification) runs AFTER Phase 5 but BEFORE firing history advance — ensures structural integrity doesn't contaminate next tick.

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
- **cognitive-core**: activation propagation, energy decay, activation reads, STDP edge reinforcement, prediction error novelty spike, neuromodulator threshold modulation, energy conservation pass/fail, valid/invalid path verification, fault penalty application
- **semantic-parser**: open camera intent, sensor event roundtrip, verb table
- **curiosity-core**: missing token detection, known token grounding, predicate parsing, dependency discovery
- **asset-ingestor**: pirate cat decomposition, quadruped→biped transform, render op generation, compile empty scene fails validation, compile valid scene passes, missing skeleton causes contract mismatch, conflicting transforms detected, render AST produces valid JSON
- **verification**: energy conservation passes on <1% discrepancy, fails on large discrepancy, valid structural path passes verify_path, structural fault penalizes node valence

## Critical Rules for Agent

1. NEVER add randomness, probability, or ML. Every decision is deterministic graph math, invariant contract check, or table lookup.
2. Structural errors must always trigger penalties (novelty spike + valence drop + edge marking). Never silently swallow a `StructuralError`.
3. All render ops must pass through `compile_to_ast()` + `validate_ast()`. Never serialize `RenderOp` directly.
4. Phase 6 (verification) must run every tick, after valence update, before firing history advance.
5. When adding new Edge construction sites, always use `Edge::new(relation, target)` or `Edge::with_weight()` — never raw struct literal syntax (which omits `dynamic_weight`, `eligibility`, `contract`).
