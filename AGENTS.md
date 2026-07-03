# Grounded — Agent Instructions

## Vision

A non-AI, deterministic cognitive system that runs 100% on-device. No tokens, no embeddings, no LLMs. It builds a relational semantic graph grounded to real sensor values and real platform actions. Knowledge is acquired through recursive curiosity — it detects gaps in its own understanding and resolves them autonomously, governed by a CuriosityBudget (energy-aware, no hard depth cap). Meaning is frame-satisfaction: a sentence is "understood" when its parsed roles fill a ConceptualFrame's mandatory slots. Rendering is a motor system — Effector AST nodes generate MotorCommand predictions, and render feedback closes the perception-action loop via reward/novelty spikes. Three structural guardrails — edge invariants, AST validation, runtime verification — prevent graph corruption.

## Core Principles

1. **PrimitiveMatrix Algebra** — Every concept is a 5-dimensional vector [Mass, Velocity, Spatial, Valence, Temporal]. Base primitives (Matter, Motion, Energy, etc.) are unit vectors. Complex concepts are derived by `combine()`. Edge contracts are derived dynamically from dimensional overlap.

2. **Formal Semantics** — Every concept is a node. Every edge is a typed relation (IsA, HasProperty, Requires, etc.). Meaning is frame-slot satisfaction — not flat edge count.

3. **CCG Parsing** — Raw definitions are parsed by a stateless Combinatory Categorial Grammar parser with 7 shift-reduce rules. Unrecognized structures degrade to proximity edges.

4. **Symbol Grounding** — Sensor nodes map raw values into activation. Action nodes produce Android intents. MotorCommand nodes produce RenderCommands. No symbol floats free.

5. **CuriosityBudget** — Replaces depth-10 circuit breaker with energy pool `E_curious`. Step cost = α·dist(SELF) + β·arousal + γ·error_rate. High novelty cheapens exploration.

6. **Deterministic** — No random numbers. No probability. Same input → same output, always. Every algorithm is O(E) bounded-time.

7. **Alive at Birth** — Birth signal on tick 0. Default Mode Network for spontaneous inner activity. Valence accumulation for preference formation. Opinion/mood/interests bridge functions.

8. **Three Structural Guardrails** — Edge invariants, AST validation, runtime verification. Violations spike deviation gain, drop valence, mark edges for pruning.

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
│  │  Notification│   │  │  │ ├─ Phase 1-6 tick    │  │   │     │
│  │              │   │  │  │ ├─ Render feedback   │  │   │     │
│  │  UniFFI      ├───┤  │  │ │   loop (Effector)  │  │   │     │
│  │  Bridge      │   │  │  │ └─ structural_faults │  │   │     │
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
│                     │  │  ├─ CCG RelationalParser   │   │     │
│                     │  │  ├─ KnowledgeStore (offline)│   │     │
│                     │  │  └─ CuriosityBudget        │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  SemanticContext (Arc)      │   │     │
│                     │  │  ├─ GraphArena             │   │     │
│                     │  │  │  ├─ PrimitiveMatrix     │   │     │
│                     │  │  │  ├─ FrameSchemas        │   │     │
│                     │  │  │  ├─ verify_path()       │   │     │
│                     │  │  │  └─ find_path()         │   │     │
│                     │  │  └─ ActivationBuffer       │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  Asset Ingestor            │   │     │
│                     │  │  ├─ ComponentExtractor     │   │     │
│                     │  │  ├─ TransformEngine        │   │     │
│                     │  │  ├─ RenderAst + Effector   │   │     │
│                     │  │  └─ compile/validate/json  │   │     │
│                     │  └────────────────────────────┘   │     │
│                     │                                   │     │
│                     │  ┌────────────────────────────┐   │     │
│                     │  │  Verification Core         │   │     │
│                     │  │  ├─ VerificationLoop       │   │     │
│                     │  │  ├─ Energy conservation    │  │     │
│                     │  │  ├─ Path integrity (cached)│  │     │
│                     │  └────────────────────────────┘   │     │
│                     └──────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## Core Systems

### PrimitiveMatrix (semantic-graph)

5-dimensional vector algebra for all concepts:

- **`PrimitiveVector { mass, velocity, spatial, valence, temporal }`** — fundamental representation
- **28 base primitives** — Matter, Motion, Dimension, Time, Energy, Force, Mass, Light, Sound, Solid, Liquid, Gas, Hot, Cold, Color, Shape, Body, Surface, Edge, Corner, Up, Down, Big, Small, Fast, Slow (all as `const` unit vectors)
- **3 derived** — momentum (Motion×Mass), temperature (Hot×Cold), texture (Surface×Mass)
- **`combine(&self, other)`** — weighted addition producing new primitives
- **`distance(&self, other)`** — Manhattan distance between vectors
- **`contract_between(src, tgt)`** — derives `InvariantContract` from dimensional overlap

### CCG RelationalParser (semantic-parser/src/relational.rs)

Replaces 6 ad-hoc grammar patterns with proper combinatoric categorial grammar:

- **`SemanticCategory`** enum: Entity, Action, Property, Relation, Modifier, Unknown
- **`RelationalParser::classify(label)`** — token → category via suffix analysis + verb table
- **`RelationalParser::parse(labels)`** — shift-reduce with 7 rules:
  1. Entity + Action + Entity → ActionFrame(actor, action, object)
  2. Entity + Action → ActionFrame(actor, action, None)
  3. Property + Entity → Attribution
  4. Entity + Relation + Entity → Relation
  5. Entity + Unknown → proximity (AssociatedWith)
  6. Unknown + Entity → proximity
  7. ActionFrame(actor, action, None) + Entity → add object
- **`RelationalParser::resolve_definition(token, raw)`** — sentence-split + parse + collect
- **Fallback** — unrecognized tokens get pairwise `AssociatedWith` proximity edges

### CuriosityBudget (curiosity-core/src/gap.rs)

Energy-aware exploration replacing hard `MAX_RECURSION_DEPTH=10`:

```rust
pub struct CuriosityBudget {
    total_energy: f64,
    remaining: f64,
    error_count: u64,
}

fn step_cost(semantic_distance, arousal, novelty) -> f64 {
    // cost = 0.3*dist + 0.4*arousal + 0.2*error_rate
    // discounted by novelty: * (1.0 - novelty * 0.3)
    // clamped to [0.05, 2.0]
}

fn halt_threshold(novelty) -> f64 {
    novelty * 0.15  // clamped to [0.01, 0.5]
}
```

- Budget carried through recursive resolution chain
- High arousal → expensive steps (focused, shallow search)
- High novelty → cheap steps + low threshold (deep exploration)
- Primitive tokens cost nothing (no further resolution needed)

### ConceptualFrame Slot System (semantic-graph)

Frames define mandatory relational slots:

| Schema | CDAction | Required | Optional |
|--------|----------|----------|----------|
| CommercialTransfer | Atrans | buyer, seller, goods | money |
| MotionEvent | Ptrans | actor, goal | source, instrument |
| SensoryEvent | Attend | observer, stimulus, channel | — |
| OwnershipChange | Grasp | owner, possession | previous_owner |

- **`FrameSchema`** — defines slots with name, expected NodeType, Relation, required flag
- **`FrameInstance`** — schema_name + frame_node + Vec<(slot_name, node_id)> + satisfaction score
- **`evaluate(edges, graph) -> (satisfaction, bindings)`** — finds edges matching slot relations, checks target node types, returns ratio of filled slots

### Unified Cognitive-Render Loop (asset-ingestor + cognitive-core)

- **`MotorCommandType`** — DrawSkeleton, ApplyTransform, ApplyMesh, Composite, Spawn, Despawn, QueryRenderState
- **`Grounding::MotorCommand { command_type, target, parameters }`** — new grounding variant
- **`RenderAst::Effector { label, command_type, target, parameters, expected_hash }`** — new AST variant
- **`ActivationEngine.process_render_feedback(effector_label, actual_hash)`** — compares actual render hash against prediction:
  - Match → `spike_reward(0.05)`
  - Mismatch → push `PredictionError` + `spike_novelty(error_mag * 0.15)`
- **`CognitiveOutput::RenderCommand`** — dispatched to external renderer
- **`CognitiveEvent::RenderFeedback`** — consumed from renderer to close the loop

## 6-Phase Tick (16ms)

```
Phase 1 — Neuromodulator Decay (O(1)):
  novelty/arousal/reward leak toward baseline (8%/12%/4%)

Phase 2 — Decay + Injection + Precision-Weighted Prediction Error (O(N)):
  activation[i] *= node.decay * 0.97 + injection[i]
  if prediction exists:
    raw_error = |actual - expected| / max(expected, 0.001)
    precision = clamp(1.0 / (variance + 0.001), 0.1, 10.0)
    weighted_error = raw_error * precision
    mean_error += 0.05 * (raw_error - mean_error)
    variance += 0.05 * ((actual - expected)^2 - variance)
    if weighted_error > PREDICTION_ERROR_THRESHOLD → novelty spike

Phase 3 — Sparse Spread + Firing + Eligibility (O(N+E)):
  if activation > threshold * threshold_mod:
    fire(i), push index to FiredNodesBuffer[count++]
    if MotorCommand → push RenderCommand + prediction
  else:
    if edge_count > SPARSE_SPREAD_MAX_EDGES (64):
      sort edges by dynamic_weight desc, truncate to 64
    spread energy to neighbors (conservation)
    terminate when remaining energy < SPARSE_SPREAD_MIN_ENERGY (0.005)

Phase 4 — Event-Driven STDP + Pruning (O(F·avg_degree)):
  for each fired node i in FiredNodesBuffer:
    for each edge of i: decay eligibility, LTP/LTD
    for each pair (i, j) in FiredNodesBuffer (j > i):
      if (i,j) not in processed_pairs:
        find edge i→j or j→i, apply LTP/LTD, mark processed
  clear FiredNodesBuffer (count = 0)

Phase 5 — Valence Update (O(N)):
  fire + error → negative; fire + no error → positive; SELF → +0.5 baseline

Phase 6 — Structural Verification + Visual Push (O(N+E)):
  energy conservation, path integrity (cached: cache_hit short-circuits
    per-edge contract checks for previously verified (source,dest) pairs),
    novelty/valence/edge penalties
  compute_effector_state() → VisualEffectorBuffer.write()
  get_active_visual_primitives(activations) → if > VISUAL_CLUSTER_THRESHOLD, VisualPrimitiveRingBuffer.push()
```

### Phase 2 Detail: Precision-Weighted Prediction Error

Each `GroundedNode` now carries a running estimate of its own prediction reliability:

| Node Field | Purpose | Update Rule |
|-----------|---------|-------------|
| `mean_error` | EMA of absolute prediction error | `+= 0.05 × (raw_error - mean_error)` |
| `variance` | EMA of squared error | `+= 0.05 × ((actual-expected)² - variance)` |
| `precision` | Derived (not stored) | `1.0 / (variance + 0.001)`, clamped [0.1, 10.0] |

High-variance nodes (noisy sensors, unpredictable concepts) naturally self-attenuate — their precision term approaches 0.1, reducing effective error by up to 10×. This prevents runaway curiosity loops from sensor jitter.

### Phase 3 Detail: Sparse Spread for High-Degree Nodes

When a node has more than `SPARSE_SPREAD_MAX_EDGES` (64) outbound edges, the spread phase sorts edges by `dynamic_weight` descending and propagates energy only to the top 64. Remaining energy below `SPARSE_SPREAD_MIN_ENERGY` (0.005) is discarded. This prevents hub nodes (e.g. SELF with thousands of edges) from O(E) dominating the tick budget.

### Phase 4 Detail: Event-Driven STDP

Phase 4 no longer iterates all graph edges globally. Instead:

1. Phase 3 populates `FiredNodesBuffer: [u32; 1024]` with indices of firing nodes
2. Phase 4 processes only edges incident to nodes in the buffer:
   - For each fired node `i`: iterate its outbound edges, decay eligibility, apply LTP if target also fired
   - For each unordered pair `(i, j)` with `i < j`: check for bidirectional edges via `processed_pairs: HashSet<(usize, usize)>`
3. `FiredNodesBuffer` is cleared at end of Phase 4 (`count = 0`)

Complexity reduces from O(N+E) to O(F·avg_degree) where F ≤ 1024 ≪ N.

### Incremental Path Cache (GraphArena)

`verify_path()` now checks a **16-slot path cache** before doing per-edge contract verification:

- **Cache hit**: `(source, dest)` pair found → return `Ok(())` immediately
- **Cache miss**: verify path, store `CachedPath { source, destination, relation_mask, path_weight }` on success
- **Invalidation**: any structural mutation (`insert`, `insert_at`, `link_to_self`, `garbage_collect_edges`) sets all slots to `None` via `.fill(None)`
- **Eviction**: random-slot replacement on full (hash-based: `source.0 * 2654435761 % 16`)
- **Storage**: `parking_lot::Mutex<[Option<CachedPath>; 16]>` for interior mutability (`verify_path` is `&self`)
- **`find_path()`** unchanged — still full BFS; cache avoids re-verifying same paths in Phase 6, not path discovery

### Autonomous Category Synthesis (Sleep Consolidation)

During low-activity periods (novelty < 0.1, arousal < 0.1) the consolidation pass runs an **isomorphism scanner** that detects structural clusters:

```
synthesize_categories():
  for each node i:
    signature_i = extract_edge_signature(i)
      → sorted Vec<(Relation::to_u8(), TargetNodeId)> of outbound edges
      → capped at SIGNATURE_MAX_EDGES (32)
  for each pair (i, j):
    overlap = signature_overlap(signature_i, signature_j)
      → Jaccard-like ratio on sorted signature tuples
  if cluster of ≥3 nodes with overlap ≥ 80%:
    1. Create parent node: label="Concept_Cluster_0x{first_child:X}"
    2. Compute intersection of all children's edge sets
    3. Migrate intersecting edges to parent
    4. Anchor each child with Relation::IsA → parent (weight 1.0)
    5. Set parent valence = average of children's valence
    6. categories_synthesized += 1
```

Parent nodes act as abstract categories — the system autonomously hoists recurring patterns into a superordinate concept without external supervision.

## Crate Map

### semantic-graph
- `PrimitiveVector` — 5-d vector algebra, `combine()`, `distance()`, `contract_between()`
- `base_primitives` — 28 unit vectors + 3 derived
- `primitive_for(label)` — lookup by name
- `FrameSchema`, `SlotBinding`, `FrameInstance` — slot-based meaning
- `frame_schemas` — 4 built-in: CommercialTransfer, MotionEvent, SensoryEvent, OwnershipChange
- `MotorCommandType` — 7 effector command types
- `Grounding::MotorCommand` — motor effector grounding
- `GroundedNode.mean_error`, `GroundedNode.variance` — EMA prediction statistics for precision-weighting
- `CachedPath { source, destination, relation_mask, path_weight }` — path cache entry
- `GraphArena::path_cache: Mutex<[Option<CachedPath>; 16]>` — incremental path cache with interior mutability
- `GraphArena::cache_hit()`, `cache_store()` — cache lookup/store helpers
- `GraphArena::verify_path()` — checks cache before per-edge traversal
- Cache invalidation on `insert`, `insert_at`, `link_to_self`, `garbage_collect_edges`

### semantic-parser
- `parse_intent()`, `parse_sensor_event()`, `Realizer`
- `relational::RelationalParser` — CCG shift-reduce, 7 rules, proximity fallback
- `relational::SemanticCategory` — Entity, Action, Property, Relation, Modifier, Unknown

### cognitive-core
- `ActivationEngine::tick()` — 6-phase tick (now includes render feedback loop)
- `RenderCommand`, `RenderPrediction` — motor command dispatch + feedback tracking
- `process_render_feedback()` — compares actual hash vs predicted, spikes reward/novelty
- `VerificationLoop`, `EventChannel`, `CognitiveDaemon`, Consolidation
- `CognitiveOutput::RenderCommand`, `CognitiveEvent::RenderFeedback`
- `FiredNodesBuffer: [u32; 1024]` — event-driven STDP tracking, populated Phase 3, consumed Phase 4
- `SPARSE_SPREAD_MAX_EDGES = 64` — sort+truncate for high-degree node spread
- `SPARSE_SPREAD_MIN_ENERGY = 0.005` — early termination threshold in spread
- Phase 2: precision-weighted prediction error (`mean_error`/`variance` EMA, precision clamping)
- Phase 3: sparse spread (sort by weight, truncate to 64)
- Phase 4: event-driven STDP (incident-only via FiredNodesBuffer + processed_pairs HashSet)
- `ConsolidationReport.categories_synthesized: usize` — category synthesis counter
- `consolidation.rs: synthesize_categories()` — cluster detection, signature extraction, edge migration, parent node creation

### curiosity-core
- `CuriosityBudget` — energy pool, `step_cost()`, `halt_threshold()`, `consume()`
- `KnowledgeGap.budget` replaces `recursion_depth`
- `DefinitionResolver::resolve()` now uses CCG `RelationalParser::resolve_definition()`
- Removed `MAX_RECURSION_DEPTH=10` — replaced by CuriosityBudget

### asset-ingestor
- `RenderAst::Effector` — new variant with command_type, target, parameters, expected_hash
- `validate_ast()` — validates Effector target references
- `render_ast_to_json()` — serializes Effector nodes

## Key Decisions

- **Primitives are not hardcoded enums** — they are vectors in a 5-d space. New primitives generated by combining base vectors. Edge contracts derived from dimensional composition, not a hardcoded table.
- **Curiosity no longer has a hard depth limit** — replaced by CuriosityBudget energy pool. Naturally truncates low-value branches while allowing deep search for novel concepts.
- **Meaning is frame-satisfaction** — not flat edge count. A sentence is "understood" when its parsed roles satisfy a ConceptualFrame's slot requirements.
- **Rendering is a motor system** — Effector AST nodes map to MotorCommand grounding. Render output feeds back into predictive coding. Match → Reward. Mismatch → Novelty spike.
- **Parsing is shift-reduce CCG** — stateless, deterministic, no regex. 6 ad-hoc grammar patterns replaced by 7 combinatory reduction rules. Unrecognized structures degrade to AssociatedWith proximity.
- **CuriosityBudget replaces recursion_depth** — `KnowledgeGap.budget: CuriosityBudget` instead of `recursion_depth: u8`. Budget is carried through recursive resolution chain.

## Visual Primitive Ring Buffer (Lock-Free SPSC)

Alongside the double-buffered `VisualEffectorBuffer`, there is a **lock-free SPSC ring buffer** (`VisualPrimitiveRingBuffer`) for streaming higher-level visual state from the cognitive daemon thread to the render bridge thread:

```
Cognitive Daemon (Phase 6)                  Render Bridge Thread
┌────────────────────────────────┐         ┌──────────────────────────┐
│                                │         │                          │
│  get_active_visual_            │  push   │  loop {                  │
│  primitives(activations)       │────────►│    ring.pop(&payload)   │
│    → FixedVisualPayload        │         │    backend.render(payload)│
│                                │  SPSC   │    sleep(8ms)           │
│  if cluster_activation >       │  Ring   │  }                      │
│     VISUAL_CLUSTER_THRESHOLD:  │  Buffer │                          │
│    ring.push(&payload)         │  (64)   │  FixedVisualPayload:    │
│                                │         │    spatial_scale: f64   │
│  SensorMapper (Phase 2)        │         │    rotation_x/y/z: f64  │
│    map_accelerometer(g) →      │         │    chroma_saturation: f64│
│      RotationX/Y/Z injection   │         │    wireframe: f64       │
│    map_light(lux) →            │         │                          │
│      ColorChroma + Scale       │         │  PenalizeFn on error:   │
│    variance_error(prev,curr)   │         │    -0.05 valence         │
│      → novelty spike           │         └──────────────────────────┘
└────────────────────────────────┘
```

### VisualPrimitiveType (semantic-graph)

6 concrete primitive types stored as `Grounding::VisualPrimitive`:

| Variant | Fixed NodeId | Maps From (SensorMapper) | Activation → Render |
|---------|-------------|--------------------------|---------------------|
| `SpatialScale` | 200 | Light sensor (inverse) | 0..1 → scale multiplier |
| `RotationX` | 201 | Accelerometer X | -1..1 → Euler angle rad |
| `RotationY` | 202 | Accelerometer Y | -1..1 → Euler angle rad |
| `RotationZ` | 203 | Accelerometer Z | -1..1 → Euler angle rad |
| `ColorChroma` | 204 | Light sensor (direct) | 0..1 → chroma saturation |
| `TopologyWireframe` | 205 | — | >0.5 → wireframe on |

Nodes occupy fixed indices in `GraphArena` via `insert_at()` (pads with empty slots to reach canonical position). `get_active_visual_primitives()` scans by `Grounding::VisualPrimitive`, not by index range.

### FixedVisualPayload (semantic-graph)

```rust
pub struct FixedVisualPayload {
    pub spatial_scale: f64,
    pub rotation_x: f64,
    pub rotation_y: f64,
    pub rotation_z: f64,
    pub chroma_saturation: f64,
    pub wireframe: f64,
}
```

- `wireframe` is binary: `if activation > 0.5 { 1.0 } else { 0.0 }`
- `cluster_activation() = sum of all 6 absolute values; must exceed VISUAL_CLUSTER_THRESHOLD (0.15 × 6 = 0.9 avg) to push
- Packed as `[AtomicU64; 6]` via `f64::to_bits()` inside the ring buffer

### VisualPrimitiveRingBuffer (semantic-graph)

Fixed-size SPSC ring buffer (no mutex, no RwLock):
- `VISUAL_RING_SIZE = 64` slots × 6 `AtomicU64` words = 384 atomic words
- **Writer** (cognitive daemon, Phase 6): relaxed stores → Release fence → `write_seq` Release
- **Reader** (render bridge): `read_seq` Relaxed → `write_seq` Acquire → Acquire fence → relaxed loads → `read_seq` Release
- `push()` returns false if full (caller retries next tick); `pop()` returns false if empty
- `Sync` because all fields are `AtomicU64`
- No spinning, no allocations, no locks

### SensorMapper (cognitive-core)

Stateless fixed-point sensor→activation translator:

| Function | Input | Output |
|----------|-------|--------|
| `map_accelerometer(gx, gy, gz)` | m/s² | `(RotationX, gx*0.05), (RotationY, gy*0.05), (RotationZ, gz*0.05)` each clamped to [-0.8, 0.8] |
| `map_light_sensor(lux)` | lux | `(ColorChroma, lux*0.001), (SpatialScale, lux/1000*0.5+0.2)` clamped to [0, 0.8] |
| `variance_error(prev, curr)` | — | `Some(ratio)` if `\|curr-prev\|/max(\|prev\|,0.001) > 0.3`, else `None` |

- Called during Phase 2 (Injection) in `CognitiveDaemon::handle_event()` on `SensorReading` events
- Returns `[(NodeId, f64); N]` — injection targets consumed by `ActivationEngine::inject()`
- Variance error beyond 30% also calls `spike_novelty(ratio * 0.3)`

### validate_ast() → ContractMismatch Valence Penalty

When the render bridge detects a structural error (contract mismatch during render):
1. `RenderBridge` calls `PenalizeFn` with error string
2. `PenalizeFn` calls `GraphArena::update_valence(node, -0.05, 0.1)` on all nodes
3. This feeds into Phase 5 of the next cognitive tick — lower valence on affected paths
4. The fault is also recorded in `ActivationEngine::structural_faults` for standard error handling

## Visual Effector Pipeline (Cross-Crate Integration)

```
Cognitive Daemon Thread (writer)           Render Bridge Thread (reader)
┌──────────────────────────────┐           ┌──────────────────────────┐
│  Phase 6 — Structural       │           │  wgpu/Null RenderBackend  │
│  Verification + Effector    │           │                          │
│  State Push                 │           │  loop {                  │
│                              │           │    buffer.read(&state)  │
│  compute_effector_state()   │  ┌──────┐ │    backend.render(state) │
│    ↓                        │  │Lock  │ │    sleep(8ms)           │
│  buffer.write(&state)       │──►Free  ◄──┤  }                      │
│                              │  │Buf   │ │                          │
│  VisualEffectorState:       │  └──────┘ │  AST Validation Errors   │
│    [0..5]   rot0..5         │           │    → penalize_fn()       │
│    [6..13]  color0/1 RGBA   │           └──────────────────────────┘
│    [14..16] scale_xyz       │
│    [17]     blend           │  Shared VisualEffectorBuffer
│    [18..19] pal_coeff_0/1   │  (AtomicU64 × 11, double-buffered)
│    [20]     bone_twist      │
│    [21]     wireframe_flag  │
└──────────────────────────────┘
```

### VisualEffectorBuffer (semantic-graph)

Lock-free double-buffered state array (no mutex, no RwLock in hot path):

- **22 f32 values** packed as 11 AtomicU64 words (2 f32 per word)
- **Double-buffered**: writer writes to back buffer then atomically flips readable generation (AtomicU8)
- **Reader** polls generation, reads both buffers, verifies generation unchanged (torn-read detection)
- **Writer** (cognitive daemon, Phase 6) calls `buffer.write(&state)` — zero allocation
- **Reader** (render bridge thread) calls `buffer.read(&mut state)` — returns false on torn read
- Stored as `Arc<VisualEffectorBuffer>` shared between CognitiveDaemon and RenderBridge

### Sensor-to-Image Transforms (asset-ingestor)

Deterministic sensor → effector state mapping, no random numbers:

| Sensor | Transform | Effector Fields |
|--------|-----------|-----------------|
| Light (lux) | `PaletteInterpolator::compute(lux, arousal)` → log-normalized interpolation | pal_coeff_0, pal_coeff_1, wireframe_flag |
| Accelerometer (gx, gy, gz) | `SkeletalTransformMatrix::from_gravity(g, activation)` → Euler angles | rot0..rot5 |
| Gravity (magnitude) | `rest_pose_adjustment(g)` → scale (upright=1.0, supine=0.6) | scale_x/y/z |
| Spatial activation | Color intensity = activation * 0.8 + 0.2 | color0_rgba, color1_rgba |

- **`TransformEngine`** — one per cognitive daemon, owns `PaletteInterpolator` + previous gravity
- **`light_to_palette_matrix()`** — pure function, side-effect-free palette interpolation
- **`accel_to_skeletal_rotation()`** — pure function, maps accel vector to Euler rotation angles
- **`gravitational_to_rest_pose()`** — pure function, maps gravity magnitude to scale modifiers

### RenderBridge (hw-daemon)

Own OS thread ("render-bridge") that polls VisualEffectorBuffer:

- **`RenderBackend` trait** — abstract wgpu/no-op backend, `render(&state) → Result<u64, String>`
- **`NullRenderBackend`** — computes hash from state, no actual rendering
- **`WgpuRenderBackend`** — full wgpu state machine (requires `--features wgpu`), reads effector state → updates skeletal bone transforms, palette colors, wireframe toggle, encodes + submits render pass
- **`PenalizeFn`** — callback for validate_ast() error → GraphArena penalization
- **8ms poll interval** (≈120Hz), sleeps between polls, no busy-wait
- Started/stopped alongside CognitiveDaemon from CognitiveLifecycle

## Crate Map (Updated)

### semantic-graph additions
- `VisualEffectorBuffer` — lock-free `[AtomicU64; 11]` double-buffered state
- `effector_state` module — helpers for packing/unpacking the 22-element state array
- `EFFECTOR_STATE_FLOATS` (22), `EFFECTOR_STATE_WORDS` (11) — layout constants
- `base_primitives::kinetic_energy()` — Motion × Mass, drives rotation scaling
- `base_primitives::spatial_bound()` — Dimension × Matter, drives color intensity
- `base_primitives::color_intensity()` — Light × Color, drives palette interpolation
- `VisualPrimitiveType` — `SpatialScale`, `RotationX/Y/Z`, `ColorChroma`, `TopologyWireframe` enum
- `Grounding::VisualPrimitive { primitive_type }` — new grounding variant
- `FixedVisualPayload` — 6 f64 fields (spatial_scale, rotation_x/y/z, chroma_saturation, wireframe)
- `VisualPrimitiveRingBuffer` — lock-free SPSC `[AtomicU64; 384]` ring buffer (64 slots × 6 words)
- `VISUAL_RING_SIZE` (64), `VISUAL_RING_MASK` (63), `VISUAL_CLUSTER_THRESHOLD` (0.15)
- `VISUAL_SPATIAL_SCALE` (200), `VISUAL_ROTATION_X/Y/Z` (201-203), `VISUAL_COLOR_CHROMA` (204), `VISUAL_TOPOLOGY_WIREFRAME` (205) — fixed canonical indices
- `GraphArena::insert_at()` — insert node at specific raw index, padding with empty slots
- `GraphArena::get_active_visual_primitives(&self, activations) -> FixedVisualPayload` — extraction method
- `GroundedNode.mean_error`, `GroundedNode.variance` — EMA prediction statistics for precision-weighted error
- `CachedPath { source, destination, relation_mask, path_weight }` — path cache entry (16 slots)
- `GraphArena::path_cache: parking_lot::Mutex<[Option<CachedPath>; 16]>` — interior mutability cache
- `GraphArena::cache_hit()`, `cache_store()` — cache operations, invalidated on structural mutation

### asset-ingestor additions
- `effector.rs` — `GravityVector`, `PaletteInterpolator`, `SkeletalTransformMatrix` (4×4 rotation, gravity decomposition, rest-pose adjustment)
- `sensor_transform.rs` — `TransformEngine` (light→palette, accel→skeletal, gravitational→rest-pose), `light_to_palette_matrix()`, `accel_to_skeletal_rotation()`, `gravitational_to_rest_pose()` pure functions

### hw-daemon additions
- `render_bridge.rs` — `RenderBridge` (own thread, buffer + ring polling, backend abstraction), `RenderBackend` trait, `NullRenderBackend`, `WgpuRenderBackend` (feature-gated `wgpu`), `PenalizeFn` callback
- `lifecycle.rs` — creates `VisualPrimitiveRingBuffer`, passes to both daemon and bridge; sets penalize callback with -0.05 valence deduction

### cognitive-core additions
- `ActivationEngine::effector_buffer: Option<Arc<VisualEffectorBuffer>>` — written to during Phase 6
- `ActivationEngine::visual_ring: Option<Arc<VisualPrimitiveRingBuffer>>` — written to during Phase 6
- `ActivationEngine::compute_effector_state()` — aggregates fired MotorCommand nodes into 22-element state array
- `CognitiveDaemon::new(ctx, effector_buffer, visual_ring)` — accepts both buffers, passes to ActivationEngine
- `SensorMapper` — stateless fixed-point sensor→visual primitive injection mapper
- `SensorMapper::map_accelerometer(gx, gy, gz) -> [(NodeId; f64); 3]` — accelerometer→rotation triad
- `SensorMapper::map_light_sensor(lux) -> [(NodeId; f64); 2]` — light→chroma + scale
- `SensorMapper::variance_error(prev, curr) -> Option<f64>` — >30% delta → prediction error magnitude
- `CognitiveDaemon::handle_event()` — SensorReading events call SensorMapper inject into visual primitives
- `FiredNodesBuffer: [u32; 1024]` with `fired_nodes_count: usize` — tracks firing nodes for Phase 4
- `SPARSE_SPREAD_MAX_EDGES = 64` — hub node spread truncation
- `SPARSE_SPREAD_MIN_ENERGY = 0.005` — spread termination threshold
- Phase 2: precision-weighted prediction error (mean_error/variance EMA updates)
- Phase 3: sparse spread (sort-by-weight truncation for high-degree nodes)
- Phase 4: event-driven STDP (incident-only via FiredNodesBuffer + processed_pairs)
- `ConsolidationReport.categories_synthesized: usize` — tracks synthesized categories
- `consolidation.rs: synthesize_categories()` — isomorphism scanner, cluster detection (3+ nodes ≥80% signature overlap), parent node creation with IsA anchoring

## Critical Rules for Agent

1. NEVER add randomness, probability, or ML. Every decision is deterministic graph math, CCG reduction, or table lookup.
2. Structural errors must always trigger penalties (novelty spike + valence drop + edge marking). Never silently swallow a `StructuralError`.
3. All render ops must pass through `compile_to_ast()` + `validate_ast()`. Never serialize `RenderOp` directly.
4. Phase 6 (verification) must run every tick, after valence update, before firing history advance.
5. When adding new Edge construction sites, always use builder methods — never raw struct literal syntax (which omits `dynamic_weight`, `eligibility`, `contract`).
6. CuriosityBudget replaces all hard depth limits. Never reintroduce `MAX_RECURSION_DEPTH`.
7. All definition resolution must go through CCG `RelationalParser`, not 6-grammar `parse_predicates()`.
8. Visual primitive nodes must always be inserted at their fixed canonical indices via `GraphArena::insert_at()` — never via `insert()` — so that `SensorMapper` injection targets and `get_active_visual_primitives()` extraction work correctly.
9. The SPSC ring buffer (`VisualPrimitiveRingBuffer`) must never be read/written from multiple threads — single producer (cognitive daemon Phase 6) and single consumer (render bridge thread) only.
10. `SensorMapper` is stateless — all sensor history for variance detection lives in `CognitiveDaemon::prev_sensor_values`, not in the mapper.
11. Precision-weighted prediction error: always compute `precision = 1.0 / (variance + 0.001)`, clamp [0.1, 10.0], and compare `weighted_error = raw_error * precision` against `PREDICTION_ERROR_THRESHOLD`. Never compare raw error against threshold.
12. `variance`/`mean_error` on `GroundedNode` must be updated every tick when a prediction exists: `mean_error += 0.05 * (raw_error - mean_error)`; `variance += 0.05 * ((actual - expected)^2 - variance)`.
13. Category synthesis must only run during quiescence (novelty < 0.1, arousal < 0.1). Never during active sensor processing.
14. Sparse spread: only sort edges when `edge_count > SPARSE_SPREAD_MAX_EDGES` (64). Normal-degree nodes iterate all edges without sorting.
15. `FiredNodesBuffer` must be cleared at end of Phase 4 (`fired_nodes_count = 0`). The raw buffer array is not zeroed — stale indices are overwritten by next Phase 3 push.
16. Path cache must be invalidated (`.fill(None)`) on every structural mutation: `insert`, `insert_at`, `link_to_self`, `garbage_collect_edges`. Never skip invalidation.
