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
│                     │  │  ├─ Energy conservation    │   │     │
│                     │  │  └─ Path integrity         │   │     │
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

Phase 2 — Decay + Injection + Prediction Error (O(N)):
  activation[i] *= node.decay * 0.97 + injection[i]
  if |actual - predicted| / predicted > 0.3 → PredictionError → novelty spike

Phase 3 — Spread + Eligibility (O(N+E)):
  if activation > threshold * threshold_mod:
    fire(i), if MotorCommand → push RenderCommand + prediction
  else: spread energy to neighbors (conservation)

Phase 4 — STDP + Pruning (O(E)):
  eligibility decay, boost on source fire, LTP on co-fire, LTD drift, prune

Phase 5 — Valence Update (O(N)):
  fire + error → negative; fire + no error → positive; SELF → +0.5 baseline

Phase 6 — Structural Verification (O(N+E)):
  energy conservation, path integrity, novelty/valence/edge penalties
```

## Crate Map

### semantic-graph
- `PrimitiveVector` — 5-d vector algebra, `combine()`, `distance()`, `contract_between()`
- `base_primitives` — 28 unit vectors + 3 derived
- `primitive_for(label)` — lookup by name
- `FrameSchema`, `SlotBinding`, `FrameInstance` — slot-based meaning
- `frame_schemas` — 4 built-in: CommercialTransfer, MotionEvent, SensoryEvent, OwnershipChange
- `MotorCommandType` — 7 effector command types
- `Grounding::MotorCommand` — motor effector grounding

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

### asset-ingestor additions
- `effector.rs` — `GravityVector`, `PaletteInterpolator`, `SkeletalTransformMatrix` (4×4 rotation, gravity decomposition, rest-pose adjustment)
- `sensor_transform.rs` — `TransformEngine` (light→palette, accel→skeletal, gravitational→rest-pose), `light_to_palette_matrix()`, `accel_to_skeletal_rotation()`, `gravitational_to_rest_pose()` pure functions

### hw-daemon additions
- `render_bridge.rs` — `RenderBridge` (own thread, buffer polling, backend abstraction), `RenderBackend` trait, `NullRenderBackend`, `WgpuRenderBackend` (feature-gated `wgpu`), `PenalizeFn` callback

### cognitive-core additions
- `ActivationEngine::effector_buffer: Option<Arc<VisualEffectorBuffer>>` — written to during Phase 6
- `ActivationEngine::compute_effector_state()` — aggregates fired MotorCommand nodes into 22-element state array
- `CognitiveDaemon::new(ctx, effector_buffer)` — accepts shared buffer, passes to ActivationEngine

## Critical Rules for Agent

1. NEVER add randomness, probability, or ML. Every decision is deterministic graph math, CCG reduction, or table lookup.
2. Structural errors must always trigger penalties (novelty spike + valence drop + edge marking). Never silently swallow a `StructuralError`.
3. All render ops must pass through `compile_to_ast()` + `validate_ast()`. Never serialize `RenderOp` directly.
4. Phase 6 (verification) must run every tick, after valence update, before firing history advance.
5. When adding new Edge construction sites, always use builder methods — never raw struct literal syntax (which omits `dynamic_weight`, `eligibility`, `contract`).
6. CuriosityBudget replaces all hard depth limits. Never reintroduce `MAX_RECURSION_DEPTH`.
7. All definition resolution must go through CCG `RelationalParser`, not 6-grammar `parse_predicates()`.
