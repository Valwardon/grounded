use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

// ────────────────────────────────────────────────────────────
//  Constants
// ────────────────────────────────────────────────────────────

/// Number of historical ticks tracked for STDP co-firing detection
pub const LTP_WINDOW: usize = 4;

/// Eligibility trace decay per tick (0.9 = retains ~35% over 10 ticks)
pub const ELIGIBILITY_DECAY: f64 = 0.9;

/// LTP weight increment per unit eligibility
pub const LTP_RATE: f64 = 0.008;

/// LTD drift toward default weight per tick
pub const DRIFT_RATE: f64 = 0.0005;

/// Minimum absolute weight before edge is pruned
pub const PRUNE_THRESHOLD: f64 = 0.005;

/// Prediction error threshold (fractional deviation) that triggers novelty spike
pub const PREDICTION_ERROR_THRESHOLD: f64 = 0.3;

// ────────────────────────────────────────────────────────────
//  Node identity
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

static NODE_COUNTER: AtomicU64 = AtomicU64::new(1);

impl NodeId {
    pub fn fresh() -> Self {
        NodeId(NODE_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn from_raw(raw: u64) -> Self {
        NodeId(raw)
    }

    pub const ZERO: Self = NodeId(0);
    /// The persistent self node — every experience is anchored to it.
    /// Inserted at index 1 in every GraphArena.
    pub const SELF: Self = NodeId(1);
}

// ────────────────────────────────────────────────────────────
//  Data-flow types for invariant contracts
// ────────────────────────────────────────────────────────────

/// The type of data a node produces or consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// Generic activation energy (spreading activation)
    Activation,
    /// Raw sensor reading (accelerometer, light, proximity)
    SensorValue,
    /// Android intent or system command
    Intent,
    /// Persistent state value
    State,
    /// Any type is compatible (universal adapter)
    Any,
}

/// Invariant contract between two nodes connected by an edge.
///
/// Defines what the source node guarantees to produce (post-condition)
/// and what the target node expects to consume (pre-condition).
/// Violations are detected by `verify_path()` and penalized by
/// the `VerificationLoop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InvariantContract {
    /// Data-flow: source output type → target input type must be compatible
    DataFlow { output_type: DataType, input_type: DataType },
    /// Causal: source event causes target state change
    Causal,
    /// Taxonomic: source classifies or categorizes target
    Taxonomic,
    /// Grounding: source is physically realized by target
    Grounding,
    /// No formal contract — reliance on STDP-learned weight only
    Unspecified,
}

/// Structural verification error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralError {
    /// Edge contract violation: source output type doesn't match target input type
    ContractMismatch {
        source: NodeId,
        target: NodeId,
        expected_input: DataType,
        actual_output: DataType,
    },
    /// Path contains a cycle that violates causality
    CyclicDependency {
        nodes: Vec<NodeId>,
    },
    /// Energy conservation violated during activation spread
    EnergyNonConservation {
        total_before: f64,
        total_after: f64,
        discrepancy: f64,
    },
    /// Node in path has been marked dead (threshold=MAX, edges=0)
    DeadNode {
        node_id: NodeId,
        label: String,
    },
    /// Traversal leads to SELF in an invalid position
    SelfLoop {
        node_id: NodeId,
    },
}

impl std::fmt::Display for StructuralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructuralError::ContractMismatch { source, target, expected_input, actual_output } => {
                write!(f, "Contract mismatch: node {} produces {:?} but node {} expects {:?}",
                    source.0, actual_output, target.0, expected_input)
            }
            StructuralError::CyclicDependency { nodes } => {
                write!(f, "Cyclic dependency detected: {:?}", nodes.iter().map(|n| n.0).collect::<Vec<_>>())
            }
            StructuralError::EnergyNonConservation { total_before, total_after, discrepancy } => {
                write!(f, "Energy non-conservation: before={:.6} after={:.6} delta={:.6}",
                    total_before, total_after, discrepancy)
            }
            StructuralError::DeadNode { node_id, label } => {
                write!(f, "Dead node in path: {} ({})", label, node_id.0)
            }
            StructuralError::SelfLoop { node_id } => {
                write!(f, "Self-loop detected at node {}", node_id.0)
            }
        }
    }
}

impl std::error::Error for StructuralError {}

// ────────────────────────────────────────────────────────────
//  Logical relation types (deterministic, closed set)
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Relation {
    IsA,
    HasProperty,
    Requires,
    CausedBy,
    Implies,
    GroundedIn,
    Precedes,
    Activates,
    Inhibits,
    AssociatedWith,
}

impl Relation {
    pub fn spread_weight(&self) -> f64 {
        match self {
            Relation::IsA => 0.9,
            Relation::HasProperty => 0.7,
            Relation::Requires => 0.8,
            Relation::CausedBy => 0.85,
            Relation::Implies => 0.75,
            Relation::GroundedIn => 1.0,
            Relation::Precedes => 0.5,
            Relation::Activates => 1.0,
            Relation::Inhibits => -0.6,
            Relation::AssociatedWith => 0.3,
        }
    }

    /// Return the default invariant contract for this relation type.
    /// Used at edge creation time when no explicit contract is provided.
    pub fn canonical_contract(&self) -> InvariantContract {
        match self {
            Relation::IsA => InvariantContract::Taxonomic,
            Relation::HasProperty => InvariantContract::Taxonomic,
            Relation::Requires => InvariantContract::DataFlow {
                output_type: DataType::Activation,
                input_type: DataType::Activation,
            },
            Relation::CausedBy => InvariantContract::Causal,
            Relation::Implies => InvariantContract::DataFlow {
                output_type: DataType::Activation,
                input_type: DataType::Activation,
            },
            Relation::GroundedIn => InvariantContract::Grounding,
            Relation::Precedes => InvariantContract::Causal,
            Relation::Activates => InvariantContract::DataFlow {
                output_type: DataType::Activation,
                input_type: DataType::Activation,
            },
            Relation::Inhibits => InvariantContract::DataFlow {
                output_type: DataType::Activation,
                input_type: DataType::Activation,
            },
            Relation::AssociatedWith => InvariantContract::Unspecified,
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Sensor normalization strategy (deterministic, serializable)
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorNorm {
    Linear { scale: f64, offset: f64 },
    Clamp { min: f64, max: f64 },
    Binary { threshold: f64 },
}

impl SensorNorm {
    pub fn apply(&self, raw: f32) -> f64 {
        match self {
            SensorNorm::Linear { scale, offset } => *raw as f64 * scale + offset,
            SensorNorm::Clamp { min, max } => (*raw as f64).clamp(*min, *max),
            SensorNorm::Binary { threshold } => {
                if *raw as f64 > *threshold { 1.0 } else { 0.0 }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Grounded concept mapping
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Grounding {
    Sensor { sensor_type: String, channel: u8, norm: SensorNorm },
    Action { intent_template: String },
    Stored { keyspace: String, key: String },
    HardwareQuery { query_type: String },
    /// Motor effector command: directly maps to a RenderAst effector node.
    /// The render pipeline executes this command and feeds visual results
    /// back into the prediction error system.
    MotorCommand {
        command_type: MotorCommandType,
        target: String,
        parameters: Vec<f64>,
    },
    Abstract,
}

/// Types of motor effector commands for the unified render loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MotorCommandType {
    DrawSkeleton,
    ApplyTransform,
    ApplyMesh,
    Composite,
    /// Spawn or reconfigure a visual element
    Spawn,
    /// Remove or disable a visual element
    Despawn,
    /// Query the current render state
    QueryRenderState,
}

impl MotorCommandType {
    pub fn label(&self) -> &'static str {
        match self {
            MotorCommandType::DrawSkeleton => "draw_skeleton",
            MotorCommandType::ApplyTransform => "apply_transform",
            MotorCommandType::ApplyMesh => "apply_mesh",
            MotorCommandType::Composite => "composite",
            MotorCommandType::Spawn => "spawn",
            MotorCommandType::Despawn => "despawn",
            MotorCommandType::QueryRenderState => "query_render_state",
        }
    }
}

// ────────────────────────────────────────────────────────────
//  The GroundedNode
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedNode {
    pub id: NodeId,
    pub label: String,
    pub node_type: NodeType,
    pub grounding: Grounding,
    pub decay: f64,
    pub threshold: f64,
    pub base_activation: f64,
    pub edges: Vec<Edge>,

    /// Valence: running average of positive/negative experience associated with this node.
    /// -1.0 (aversive) to +1.0 (attractive). Updated each tick based on co-occurrence
    /// with prediction errors (negative) or reward (positive).
    /// Drives preference formation — the system forms "likes" and "dislikes"
    /// deterministically from its own prediction history.
    pub valence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Entity, Concept, Action, Sensor, State, Frame,
}

// ────────────────────────────────────────────────────────────
//  Edge with STDP support
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub relation: Relation,
    pub target: NodeId,
    pub weight_override: Option<f64>,

    /// Mutable weight modified by STDP. Initialized from weight_override or
    /// relation.spread_weight(). Drifts toward default between LTP events.
    pub dynamic_weight: f64,

    /// Hebbian eligibility trace. Boosted when source fires, decays each tick,
    /// consumed when target fires to drive LTP.
    pub eligibility: f64,

    /// Invariant contract that constrains what data/types may flow across this edge.
    /// If None, defaults to `relation.canonical_contract()` at construction time.
    /// Verified at traversal time by `GraphArena::verify_path()`.
    pub contract: Option<InvariantContract>,
}

impl Edge {
    pub fn new(relation: Relation, target: NodeId) -> Self {
        let base = relation.spread_weight();
        Edge {
            relation, target, weight_override: None, dynamic_weight: base, eligibility: 0.0,
            contract: None,
        }
    }

    pub fn with_weight(relation: Relation, target: NodeId, override_weight: f64) -> Self {
        Edge {
            relation, target, weight_override: Some(override_weight), dynamic_weight: override_weight,
            eligibility: 0.0, contract: None,
        }
    }

    pub fn with_contract(mut self, contract: InvariantContract) -> Self {
        self.contract = Some(contract);
        self
    }

    /// The weight used during spreading activation (read from STDP-modified value).
    #[inline]
    pub fn effective_weight(&self) -> f64 {
        self.dynamic_weight
    }

    /// The architecturally intended weight — what dynamic_weight drifts toward.
    pub fn default_weight(&self) -> f64 {
        self.weight_override.unwrap_or_else(|| self.relation.spread_weight())
    }

    /// Resolve the effective contract: explicit override or canonical default.
    pub fn effective_contract(&self) -> InvariantContract {
        self.contract.unwrap_or_else(|| self.relation.canonical_contract())
    }
}

// ────────────────────────────────────────────────────────────
//  Semantic relation trait
// ────────────────────────────────────────────────────────────

pub trait SemanticRelation: Send + Sync {
    fn relation(&self) -> Relation;
    fn source(&self) -> NodeId;
    fn target(&self) -> NodeId;
    fn weight(&self) -> f64;
}

// ────────────────────────────────────────────────────────────
//  Arena-backed graph store
// ────────────────────────────────────────────────────────────

pub struct GraphArena {
    nodes: Vec<parking_lot::RwLock<GroundedNode>>,
    label_index: Vec<(String, NodeId)>,
}

impl GraphArena {
    pub fn with_capacity(cap: usize) -> Self {
        let mut arena = GraphArena {
            nodes: Vec::with_capacity(cap),
            label_index: Vec::with_capacity(cap),
        };
        // Index 0: null sentinel
        arena.nodes.push(parking_lot::RwLock::new(GroundedNode {
            id: NodeId::ZERO,
            label: String::from("<null>"),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: f64::MAX,
            base_activation: 0.0,
            edges: Vec::new(),
            valence: 0.0,
        }));
        // Index 1: the persistent self — every experience anchors here
        arena.nodes.push(parking_lot::RwLock::new(GroundedNode {
            id: NodeId::SELF,
            label: String::from("self"),
            node_type: NodeType::State,
            grounding: Grounding::Abstract,
            decay: 1.0,
            threshold: f64::MAX,
            base_activation: 1.0,
            edges: Vec::new(),
            valence: 0.5,
        }));
        arena.label_index.push(("self".to_string(), NodeId::SELF));
        arena
    }

    pub fn insert(&mut self, node: GroundedNode) -> NodeId {
        let id = NodeId::from_raw(self.nodes.len() as u64);
        self.label_index.push((node.label.clone(), id));
        self.nodes.push(parking_lot::RwLock::new(node));
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&parking_lot::RwLock<GroundedNode>> {
        self.nodes.get(id.0 as usize)
    }

    pub fn lookup(&self, label: &str) -> Option<NodeId> {
        self.label_index.iter().find(|(l, _)| l == label).map(|(_, id)| *id)
    }

    pub fn by_type(&self, ty: NodeType) -> Vec<NodeId> {
        self.nodes.iter().filter(|n| n.read().node_type == ty).map(|n| n.read().id).collect()
    }

    pub fn serialize(&self) -> Vec<u8> {
        let nodes: Vec<GroundedNode> = self.nodes.iter().map(|n| n.read().clone()).collect();
        bincode::serialize(&nodes).expect("graph serialization")
    }

    pub fn deserialize(data: &[u8]) -> Self {
        let nodes: Vec<GroundedNode> = bincode::deserialize(data).expect("graph deserialization");
        let count = nodes.len();
        let mut arena = GraphArena::with_capacity(count);
        arena.nodes = nodes.into_iter().map(parking_lot::RwLock::new).collect();
        for node in &arena.nodes {
            let n = node.read();
            arena.label_index.push((n.label.clone(), n.id));
        }
        arena
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Attach a relational edge from the self node to a target.
    pub fn link_to_self(&mut self, relation: Relation, target: NodeId) -> bool {
        if target.0 as usize >= self.nodes.len() || target == NodeId::ZERO || target == NodeId::SELF {
            return false;
        }
        let self_idx = NodeId::SELF.0 as usize;
        if self_idx >= self.nodes.len() {
            return false;
        }
        self.nodes[self_idx].write().edges.push(Edge::new(relation, target));
        true
    }

    /// Return all nodes directly connected to self.
    pub fn introspect(&self) -> Vec<(NodeId, String, Relation)> {
        let self_idx = NodeId::SELF.0 as usize;
        if self_idx >= self.nodes.len() {
            return Vec::new();
        }
        let self_node = self.nodes[self_idx].read();
        self_node.edges.iter().filter_map(|edge| {
            let idx = edge.target.0 as usize;
            if idx >= self.nodes.len() { return None; }
            let node = self.nodes[idx].read();
            Some((edge.target, node.label.clone(), edge.relation))
        }).collect()
    }

    /// Get the valence of a node.
    pub fn get_valence(&self, id: NodeId) -> Option<f64> {
        self.nodes.get(id.0 as usize).map(|n| n.read().valence)
    }

    /// Set the valence of a node.
    pub fn set_valence(&self, id: NodeId, v: f64) {
        if let Some(n) = self.nodes.get(id.0 as usize) {
            n.write().valence = v.clamp(-1.0, 1.0);
        }
    }

    /// Update valence toward a target (running average).
    /// valence += (target - valence) * rate
    pub fn update_valence(&self, id: NodeId, target: f64, rate: f64) {
        if let Some(n) = self.nodes.get(id.0 as usize) {
            let mut node = n.write();
            node.valence = (node.valence + (target - node.valence) * rate).clamp(-1.0, 1.0);
        }
    }

    /// Return up to `count` node IDs with the highest valence.
    /// Excludes NodeId::ZERO and optionally NodeId::SELF.
    pub fn nodes_with_highest_valence(&self, count: usize, include_self: bool) -> Vec<(NodeId, f64)> {
        let mut scored: Vec<(NodeId, f64)> = self.nodes.iter().enumerate()
            .filter(|(i, _)| *i > 0 && (include_self || *i as u64 != NodeId::SELF.0))
            .filter_map(|(i, n)| {
                let node = n.read();
                if node.valence.abs() > 0.01 { Some((NodeId::from_raw(i as u64), node.valence)) } else { None }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(count);
        scored
    }

    /// Find a node by label (exact match).
    pub fn find_by_label(&self, label: &str) -> Option<NodeId> {
        self.label_index.iter().find(|(l, _)| l == label).map(|(_, id)| *id)
    }

    /// Get label for a node ID.
    pub fn label_of(&self, id: NodeId) -> Option<String> {
        self.nodes.get(id.0 as usize).map(|n| n.read().label.clone())
    }

    /// Remove edges whose dynamic weight has been pruned.
    /// Called during off-tick consolidation, never in the hot path.
    pub fn garbage_collect_edges(&mut self) {
        for node_lock in &self.nodes {
            let mut node = node_lock.write();
            node.edges.retain(|e| e.dynamic_weight.abs() >= PRUNE_THRESHOLD);
        }
    }

    // ── Structural path verification ──────────────────────

    /// Verify that a sequence of node IDs forms a structurally valid path.
    ///
    /// For each consecutive pair (path[i], path[i+1]):
    ///   1. Check that the source node is alive (not dead / nulled).
    ///   2. Check that an edge exists between them.
    ///   3. Check the edge's invariant contract: the source's post-condition
    ///      (its grounding output type) must match the target's pre-condition
    ///      (its grounding input type).
    ///   4. Check for cycles (no node appears twice).
    ///   5. Check for invalid SELF references.
    ///
    /// Returns `Ok(())` if the entire path is structurally sound.
    pub fn verify_path(&self, path: &[NodeId]) -> Result<(), StructuralError> {
        if path.len() < 2 {
            return Ok(()); // single-node paths are trivially valid
        }

        // Check for duplicate nodes (cycles)
        let mut seen = std::collections::HashSet::new();
        for (i, &node_id) in path.iter().enumerate() {
            if node_id == NodeId::ZERO {
                return Err(StructuralError::DeadNode {
                    node_id, label: "<null>".into(),
                });
            }
            if !seen.insert(node_id) {
                // Cycle detected but it might be intentional — only reject
                // if the cycle doesn't pass through SELF
                if node_id != NodeId::SELF {
                    return Err(StructuralError::CyclicDependency {
                        nodes: path.iter().copied().collect(),
                    });
                }
            }
            // Check the node is alive
            if let Some(n) = self.get(node_id) {
                let node = n.read();
                if node.edges.is_empty() && node.threshold >= f64::MAX && node.decay.abs() < f32::EPSILON as f64 {
                    return Err(StructuralError::DeadNode {
                        node_id, label: node.label.clone(),
                    });
                }
            }

            // Check edge contract between this node and the next
            if i + 1 < path.len() {
                let next_id = path[i + 1];
                let edge = self.get(node_id)
                    .and_then(|n| n.read().edges.iter().find(|e| e.target == next_id));

                if let Some(edge) = edge {
                    let contract = edge.effective_contract();
                    match contract {
                        InvariantContract::DataFlow { output_type, input_type } => {
                            if !Self::types_compatible(output_type, input_type) {
                                return Err(StructuralError::ContractMismatch {
                                    source: node_id,
                                    target: next_id,
                                    expected_input: input_type,
                                    actual_output: output_type,
                                });
                            }
                        }
                        InvariantContract::Grounding => {
                            // Grounding requires source to be a Sensor or Action
                            // and target to be a Concept
                            if let Some(n) = self.get(node_id) {
                                let node = n.read();
                                if !matches!(node.node_type, NodeType::Sensor | NodeType::Action) {
                                    return Err(StructuralError::ContractMismatch {
                                        source: node_id,
                                        target: next_id,
                                        expected_input: DataType::Any,
                                        actual_output: DataType::Activation,
                                    });
                                }
                            }
                        }
                        _ => {} // Causal, Taxonomic, Unspecified are always accepted
                    }
                } else {
                    // No edge exists between consecutive nodes in path
                    // This is fine if the path was constructed from edges;
                    // if the caller provided an arbitrary sequence, they get an error
                    return Err(StructuralError::DeadNode {
                        node_id: next_id,
                        label: format!("no_edge_from_{}", node_id.0),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if a data type produced by a source can be consumed by a target.
    fn types_compatible(output: DataType, input: DataType) -> bool {
        match (output, input) {
            (_, DataType::Any) => true,
            (DataType::Any, _) => true,
            (a, b) if a == b => true,
            _ => false,
        }
    }

    /// Collect all node IDs along every unique traversal from `start` to `end`,
    /// respecting direction of edges. Returns shortest path found.
    pub fn find_path(&self, start: NodeId, end: NodeId) -> Option<Vec<NodeId>> {
        if start == end {
            return Some(vec![start]);
        }
        // BFS
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = std::collections::VecDeque::new();
        let mut parent = vec![NodeId::ZERO; self.nodes.len()];

        visited[start.0 as usize] = true;
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if let Some(node) = self.get(current) {
                for edge in &node.read().edges {
                    let next = edge.target;
                    let idx = next.0 as usize;
                    if idx < visited.len() && !visited[idx] {
                        visited[idx] = true;
                        parent[idx] = current;
                        if next == end {
                            // Reconstruct path
                            let mut path = vec![end, current];
                            let mut p = current;
                            while p != start {
                                p = parent[p.0 as usize];
                                path.push(p);
                            }
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(next);
                    }
                }
            }
        }
        None
    }
}

// ────────────────────────────────────────────────────────────
//  Lock-free activation double buffer + prediction array
// ────────────────────────────────────────────────────────────

pub struct ActivationBuffer {
    buffers: [Box<[f64]>; 2],
    active: AtomicU8,
    len: usize,
}

impl ActivationBuffer {
    pub fn new(len: usize) -> Self {
        let zero = vec![0.0; len].into_boxed_slice();
        ActivationBuffer { buffers: [zero.clone(), zero], active: AtomicU8::new(0), len }
    }

    #[inline]
    pub fn read(&self) -> &[f64] {
        let idx = self.active.load(Ordering::Acquire) as usize;
        &self.buffers[idx]
    }

    pub fn back_mut(&mut self) -> &mut [f64] {
        let idx = 1 - self.active.load(Ordering::Relaxed) as usize;
        &mut self.buffers[idx]
    }

    pub fn flip(&self) {
        let current = self.active.load(Ordering::Relaxed);
        self.active.store(1 - current, Ordering::Release);
    }

    pub fn resize(&mut self, new_len: usize) {
        for buf in &mut self.buffers {
            let mut v = buf.to_vec();
            v.resize(new_len, 0.0);
            *buf = v.into_boxed_slice();
        }
        self.len = new_len;
    }
}

// ────────────────────────────────────────────────────────────
//  Firing history ring buffer (bitset-based, no allocations in hot path)
// ────────────────────────────────────────────────────────────

pub struct FiringHistory {
    inner: Vec<u64>,
    write_ptr: usize,
    node_count: usize,
    words_per_tick: usize,
}

impl FiringHistory {
    pub fn new(node_count: usize) -> Self {
        let words = (node_count + 63) / 64;
        FiringHistory {
            inner: vec![0; LTP_WINDOW * words.max(1)],
            write_ptr: 0,
            node_count,
            words_per_tick: words.max(1),
        }
    }

    /// Record that a node fired this tick.
    #[inline]
    pub fn record_fired(&mut self, node_id: NodeId) {
        let idx = node_id.0 as usize;
        let word = self.write_ptr * self.words_per_tick + idx / 64;
        let bit = idx % 64;
        if word < self.inner.len() {
            self.inner[word] |= 1u64 << bit;
        }
    }

    /// Advance to the next tick slot (clears current tick's bits).
    #[inline]
    pub fn advance_tick(&mut self) {
        self.write_ptr = (self.write_ptr + 1) % LTP_WINDOW;
        let base = self.write_ptr * self.words_per_tick;
        for i in 0..self.words_per_tick {
            if base + i < self.inner.len() {
                self.inner[base + i] = 0;
            }
        }
    }

    /// Check if a node fired within the last `lookback` ticks (1 = last tick, etc.).
    #[inline]
    pub fn fired_recently(&self, node_id: NodeId, lookback: usize) -> bool {
        let idx = node_id.0 as usize;
        let word = idx / 64;
        let bit = idx % 64;
        let mask = 1u64 << bit;
        for back in 0..lookback.min(LTP_WINDOW) {
            let ptr = (self.write_ptr + LTP_WINDOW - back) % LTP_WINDOW;
            let offset = ptr * self.words_per_tick + word;
            if offset < self.inner.len() && (self.inner[offset] & mask) != 0 {
                return true;
            }
        }
        false
    }

    pub fn resize(&mut self, new_node_count: usize) {
        let new_words = ((new_node_count + 63) / 64).max(1);
        let new_total = LTP_WINDOW * new_words;
        let mut new_inner = vec![0; new_total];
        for t in 0..LTP_WINDOW.min(self.inner.len() / self.words_per_tick.max(1)) {
            let src_off = t * self.words_per_tick;
            let dst_off = t * new_words;
            for w in 0..self.words_per_tick.min(new_words) {
                if src_off + w < self.inner.len() && dst_off + w < new_total {
                    new_inner[dst_off + w] = self.inner[src_off + w];
                }
            }
        }
        self.inner = new_inner;
        self.node_count = new_node_count;
        self.words_per_tick = new_words;
        self.write_ptr = self.write_ptr % LTP_WINDOW;
    }
}

// ────────────────────────────────────────────────────────────
//  Neuromodulator channels
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Neuromodulator {
    /// Spiked by curiosity gaps and prediction errors.
    /// Lowers firing thresholds, accelerates STDP.
    pub novelty: f64,

    /// Spiked by violent sensor shifts or threat signals.
    /// Diverts energy to action paths, dampens curiosity.
    pub arousal: f64,

    /// Spiked when prediction matches outcome.
    /// Solidifies recent edge changes.
    pub reward: f64,
}

impl Neuromodulator {
    pub fn new() -> Self {
        Neuromodulator { novelty: 0.0, arousal: 0.0, reward: 0.0 }
    }

    /// Decay all channels toward baseline (0.0) each tick.
    /// Novelty decays fastest, reward slowest.
    #[inline]
    pub fn tick_decay(&mut self) {
        self.novelty += (0.0 - self.novelty) * 0.08;
        self.arousal += (0.0 - self.arousal) * 0.12;
        self.reward += (0.0 - self.reward) * 0.04;
    }

    pub fn spike_novelty(&mut self, amount: f64) {
        self.novelty = (self.novelty + amount).clamp(0.0, 1.0);
    }

    pub fn spike_arousal(&mut self, amount: f64) {
        self.arousal = (self.arousal + amount).clamp(0.0, 1.0);
    }

    pub fn spike_reward(&mut self, amount: f64) {
        self.reward = (self.reward + amount).clamp(0.0, 1.0);
    }

    /// Global threshold multiplier: high novelty/arousal → lower threshold → easier to fire.
    #[inline]
    pub fn threshold_modifier(&self) -> f64 {
        (1.0 - self.novelty * 0.35 - self.arousal * 0.20).clamp(0.4, 1.0)
    }

    /// STDP plasticity multiplier: novelty + reward accelerate learning.
    #[inline]
    pub fn plasticity_modifier(&self) -> f64 {
        (0.5 + self.novelty * 0.4 + self.reward * 0.3).clamp(0.5, 1.5)
    }
}

// ────────────────────────────────────────────────────────────
//  Prediction error signal
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PredictionError {
    pub node_id: NodeId,
    pub expected: f64,
    pub actual: f64,
    pub error_magnitude: f64,
}

// ────────────────────────────────────────────────────────────
//  Graph + activation = context
// ────────────────────────────────────────────────────────────

pub struct SemanticContext {
    pub graph: RwLock<GraphArena>,
    pub activation: RwLock<ActivationBuffer>,
    pub tick: AtomicU64,
}

impl SemanticContext {
    pub fn new(graph: GraphArena) -> Arc<Self> {
        let len = graph.len();
        Arc::new(SemanticContext {
            graph: RwLock::new(graph),
            activation: RwLock::new(ActivationBuffer::new(len)),
            tick: AtomicU64::new(0),
        })
    }

    pub fn introspect(&self) -> Vec<(NodeId, String, Relation)> {
        self.graph.read().introspect()
    }

    pub fn link_to_self(&self, relation: Relation, target: NodeId) -> bool {
        self.graph.write().link_to_self(relation, target)
    }
}

// ────────────────────────────────────────────────────────────
//  Conceptual Dependency frame
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualFrame {
    pub frame_type: CDType,
    pub actor: Option<NodeId>,
    pub action: CDAction,
    pub object: Option<NodeId>,
    pub recipient: Option<NodeId>,
    pub instrument: Option<NodeId>,
    pub source: Option<NodeId>,
    pub goal: Option<NodeId>,
    pub time: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CDType {
    Action, StateChange, MentalEvent, PhysicalTransfer,
    OwnershipTransfer, SensorEvent, SystemCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CDAction {
    Atrans, Ptrans, Mtrans, Mbuild, Propel,
    Ingest, Expel, Move, Grasp, Speak, Attend,
    SystemAction, SensorReading,
}

impl ConceptualFrame {
    pub fn new(action: CDAction) -> Self {
        ConceptualFrame {
            frame_type: CDType::Action,
            actor: None, action, object: None, recipient: None,
            instrument: None, source: None, goal: None, time: None,
        }
    }

    pub fn injection_targets(&self, base_energy: f64) -> Vec<(NodeId, f64)> {
        let mut targets = Vec::with_capacity(6);
        if let Some(a) = self.actor { targets.push((a, base_energy * 0.8)); }
        if let Some(o) = self.object { targets.push((o, base_energy * 1.0)); }
        if let Some(r) = self.recipient { targets.push((r, base_energy * 0.6)); }
        if let Some(inst) = self.instrument { targets.push((inst, base_energy * 0.4)); }
        if let Some(s) = self.source { targets.push((s, base_energy * 0.5)); }
        if let Some(g) = self.goal { targets.push((g, base_energy * 0.7)); }
        targets
    }
}

// ────────────────────────────────────────────────────────────
//  PrimitiveMatrix — algebraic bootstrapping schema
//
//  Every grounded concept is a 5-dimensional vector:
//  [Mass, Velocity, Spatial, Valence, TemporalFrequency]
//
//  Base primitives (Matter, Motion, Energy, etc.) are pure
//  unit vectors. Complex concepts are derived by vector
//  combination: Speed + Mass = Momentum, Motion + Valence = Mood.
//
//  Edge contracts are derived dynamically from the dimensional
//  composition of source and target vectors.
// ────────────────────────────────────────────────────────────

/// The 5 fundamental physical dimensions of any concept.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveVector {
    /// Mass / substance / material density (0.0 = immaterial, 1.0 = solid)
    pub mass: f64,
    /// Velocity / speed / rate of change (0.0 = static, 1.0 = fast)
    pub velocity: f64,
    /// Spatial extent / dimensionality (0.0 = point, 1.0 = voluminous)
    pub spatial: f64,
    /// Valence / affective charge (-1.0 = aversive, +1.0 = attractive)
    pub valence: f64,
    /// Temporal frequency / recurrence rate (0.0 = never, 1.0 = constant)
    pub temporal: f64,
}

impl PrimitiveVector {
    pub const fn new(mass: f64, velocity: f64, spatial: f64, valence: f64, temporal: f64) -> Self {
        PrimitiveVector { mass, velocity, spatial, valence, temporal }
    }

    pub fn zero() -> Self {
        PrimitiveVector { mass: 0.0, velocity: 0.0, spatial: 0.0, valence: 0.0, temporal: 0.0 }
    }

    /// Combine two vectors via weighted addition. The result inherits
    /// dimensional properties from both parents.
    pub fn combine(&self, other: &PrimitiveVector) -> Self {
        PrimitiveVector {
            mass: (self.mass + other.mass * 0.5).clamp(0.0, 1.0),
            velocity: (self.velocity + other.velocity * 0.5).clamp(0.0, 1.0),
            spatial: (self.spatial + other.spatial * 0.5).clamp(0.0, 1.0),
            valence: (self.valence + other.valence * 0.5).clamp(-1.0, 1.0),
            temporal: (self.temporal + other.temporal * 0.5).clamp(0.0, 1.0),
        }
    }

    /// Semantic distance between two vectors (Manhattan).
    pub fn distance(&self, other: &PrimitiveVector) -> f64 {
        (self.mass - other.mass).abs()
            + (self.velocity - other.velocity).abs()
            + (self.spatial - other.spatial).abs()
            + (self.valence - other.valence).abs()
            + (self.temporal - other.temporal).abs()
    }

    /// The dominant dimension (highest absolute value).
    pub fn dominant(&self) -> &'static str {
        let vals = [("mass", self.mass), ("velocity", self.velocity),
            ("spatial", self.spatial), ("valence", self.valence.abs()), ("temporal", self.temporal)];
        vals.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(n, _)| *n).unwrap_or("mass")
    }

    /// Derive the default InvariantContract for an edge connecting
    /// a source vector to a target vector. Based on dimensional overlap.
    pub fn contract_between(source: &PrimitiveVector, target: &PrimitiveVector) -> InvariantContract {
        let overlap = (source.mass * target.mass
            + source.velocity * target.velocity
            + source.spatial * target.spatial
            + source.valence.abs() * target.valence.abs()
            + source.temporal * target.temporal) / 5.0;
        if overlap > 0.6 {
            InvariantContract::DataFlow { output_type: DataType::Activation, input_type: DataType::Activation }
        } else if source.spatial > 0.0 && target.spatial > 0.0 {
            InvariantContract::Causal
        } else if (source.valence - target.valence).abs() < 0.3 {
            InvariantContract::Taxonomic
        } else {
            InvariantContract::Unspecified
        }
    }
}

/// Fundamental base primitives — unit vectors along each dimension.
pub mod base_primitives {
    use super::PrimitiveVector;

    pub const MATTER:    PrimitiveVector = PrimitiveVector::new(1.0, 0.0, 0.5, 0.0, 0.0);
    pub const MOTION:    PrimitiveVector = PrimitiveVector::new(0.0, 1.0, 0.0, 0.0, 0.5);
    pub const DIMENSION: PrimitiveVector = PrimitiveVector::new(0.0, 0.0, 1.0, 0.0, 0.0);
    pub const TIME:      PrimitiveVector = PrimitiveVector::new(0.0, 0.0, 0.0, 0.0, 1.0);
    pub const ENERGY:    PrimitiveVector = PrimitiveVector::new(0.3, 0.8, 0.0, 0.0, 0.3);
    pub const FORCE:     PrimitiveVector = PrimitiveVector::new(0.5, 0.5, 0.0, 0.0, 0.0);
    pub const MASS:      PrimitiveVector = PrimitiveVector::new(1.0, 0.0, 0.3, 0.0, 0.0);
    pub const LIGHT:     PrimitiveVector = PrimitiveVector::new(0.0, 1.0, 0.0, 0.2, 0.0);
    pub const SOUND:     PrimitiveVector = PrimitiveVector::new(0.0, 0.5, 0.3, 0.1, 0.3);
    pub const SOLID:     PrimitiveVector = PrimitiveVector::new(0.9, 0.0, 0.8, 0.0, 0.0);
    pub const LIQUID:    PrimitiveVector = PrimitiveVector::new(0.6, 0.3, 0.6, 0.0, 0.0);
    pub const GAS:       PrimitiveVector = PrimitiveVector::new(0.1, 0.7, 0.9, 0.0, 0.0);
    pub const HOT:       PrimitiveVector = PrimitiveVector::new(0.2, 0.6, 0.2, -0.1, 0.2);
    pub const COLD:      PrimitiveVector = PrimitiveVector::new(0.4, 0.1, 0.2, -0.2, 0.0);
    pub const COLOR:     PrimitiveVector = PrimitiveVector::new(0.0, 0.0, 0.0, 0.5, 0.0);
    pub const SHAPE:     PrimitiveVector = PrimitiveVector::new(0.2, 0.0, 0.9, 0.1, 0.0);
    pub const BODY:      PrimitiveVector = PrimitiveVector::new(0.7, 0.2, 0.6, 0.1, 0.0);
    pub const SURFACE:   PrimitiveVector = PrimitiveVector::new(0.3, 0.0, 0.8, 0.0, 0.0);
    pub const EDGE:      PrimitiveVector = PrimitiveVector::new(0.1, 0.0, 0.7, 0.0, 0.0);
    pub const CORNER:    PrimitiveVector = PrimitiveVector::new(0.1, 0.0, 0.6, 0.0, 0.0);
    pub const UP:        PrimitiveVector = PrimitiveVector::new(0.0, 0.3, 0.5, 0.1, 0.0);
    pub const DOWN:      PrimitiveVector = PrimitiveVector::new(0.3, 0.3, 0.5, -0.1, 0.0);
    pub const BIG:       PrimitiveVector = PrimitiveVector::new(0.5, 0.0, 1.0, 0.2, 0.0);
    pub const SMALL:     PrimitiveVector = PrimitiveVector::new(0.2, 0.0, 0.2, 0.0, 0.0);
    pub const FAST:      PrimitiveVector = PrimitiveVector::new(0.0, 1.0, 0.0, 0.3, 0.5);
    pub const SLOW:      PrimitiveVector = PrimitiveVector::new(0.2, 0.2, 0.0, 0.0, 0.3);

    /// Derived vectors via combination. Speed+Mass = Momentum.
    pub fn momentum() -> PrimitiveVector { MOTION.combine(&MASS) }
    pub fn temperature() -> PrimitiveVector { HOT.combine(&COLD) }
    pub fn texture() -> PrimitiveVector { SURFACE.combine(&MASS) }

    /// KineticEnergy = Motion + Mass — drives skeletal rotation scaling.
    /// Activation from accelerometer deltas → modulates rot0..rot5.
    pub fn kinetic_energy() -> PrimitiveVector { MOTION.combine(&MASS) }

    /// SpatialBound = Dimension + Matter — drives color intensity.
    /// Activation from proximity sensor → modulates RGB channels.
    pub fn spatial_bound() -> PrimitiveVector { DIMENSION.combine(&MATTER) }

    /// ColorIntensity = Light + Color — drives palette interpolation.
    /// Activation from light sensor → modulates palette coefficients.
    pub fn color_intensity() -> PrimitiveVector { LIGHT.combine(&COLOR) }
}

/// Look up a primitive vector by label. Returns Some if the label
/// matches a known primitive or derived concept.
pub fn primitive_for(label: &str) -> Option<PrimitiveVector> {
    match label {
        "matter"   => Some(base_primitives::MATTER),
        "motion"   => Some(base_primitives::MOTION),
        "dimension" => Some(base_primitives::DIMENSION),
        "time"     => Some(base_primitives::TIME),
        "energy"   => Some(base_primitives::ENERGY),
        "force"    => Some(base_primitives::FORCE),
        "mass"     => Some(base_primitives::MASS),
        "light"    => Some(base_primitives::LIGHT),
        "sound"    => Some(base_primitives::SOUND),
        "solid"    => Some(base_primitives::SOLID),
        "liquid"   => Some(base_primitives::LIQUID),
        "gas"      => Some(base_primitives::GAS),
        "hot"      => Some(base_primitives::HOT),
        "cold"     => Some(base_primitives::COLD),
        "color"    => Some(base_primitives::COLOR),
        "shape"    => Some(base_primitives::SHAPE),
        "body"     => Some(base_primitives::BODY),
        "surface"  => Some(base_primitives::SURFACE),
        "edge"     => Some(base_primitives::EDGE),
        "corner"   => Some(base_primitives::CORNER),
        "up"       => Some(base_primitives::UP),
        "down"     => Some(base_primitives::DOWN),
        "big"      => Some(base_primitives::BIG),
        "small"    => Some(base_primitives::SMALL),
        "fast"     => Some(base_primitives::FAST),
        "slow"     => Some(base_primitives::SLOW),
        "momentum"  => Some(base_primitives::momentum()),
        "temperature" => Some(base_primitives::temperature()),
        "texture"   => Some(base_primitives::texture()),
        "kinetic_energy" => Some(base_primitives::kinetic_energy()),
        "spatial_bound" => Some(base_primitives::spatial_bound()),
        "color_intensity" => Some(base_primitives::color_intensity()),
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────
//  VisualEffectorState — packed lock-free render state
//
//  A 22-element [f32] array stored as 11 AtomicU64 words (2 f32 per word).
//  Layout:
//    [0..5]   rot0..rot5        — skeletal rotation angles (radians)
//    [6..9]   color0_r/g/b/a    — first palette color (RGBA)
//    [10..13] color1_r/g/b/a    — second palette color (RGBA)
//    [14..16] scale_x/y/z       — skeletal scale factors
//    [17]     blend             — blend weight [0,1]
//    [18..19] pal_coeff_0/1     — palette interpolation coefficients
//    [20]     bone_twist        — bone twist angle modifier
//    [21]     wireframe_flag    — 0.0 = normal, 1.0 = wireframe
// ────────────────────────────────────────────────────────────

/// Number of f32 values in a VisualEffectorState.
pub const EFFECTOR_STATE_FLOATS: usize = 22;
/// Number of AtomicU64 words (2 f32 per word).
pub const EFFECTOR_STATE_WORDS: usize = 11;

/// Lock-free visual effector state — 22 f32 values packed as 11 AtomicU64.
///
/// Access is entirely lock-free: readers sample the readable generation,
/// read both AtomicU64 buffers, then verify no torn read by checking
/// the generation again. Writers write to the back buffer and flip.
#[derive(Debug)]
pub struct VisualEffectorBuffer {
    buffer_a: [AtomicU64; EFFECTOR_STATE_WORDS],
    buffer_b: [AtomicU64; EFFECTOR_STATE_WORDS],
    /// Which buffer is currently readable (0 = a, 1 = b).
    readable: AtomicU8,
}

unsafe impl Send for VisualEffectorBuffer {}
unsafe impl Sync for VisualEffectorBuffer {}

impl VisualEffectorBuffer {
    /// Create a new zero-initialized buffer.
    pub fn new() -> Self {
        VisualEffectorBuffer {
            buffer_a: Default::default(),
            buffer_b: Default::default(),
            readable: AtomicU8::new(0),
        }
    }

    /// Write a full effector state into the write buffer and atomically
    /// flip the readable generation. Called from the cognitive engine thread.
    #[inline]
    pub fn write(&self, state: &[f32; EFFECTOR_STATE_FLOATS]) {
        let write_idx = 1 - self.readable.load(Ordering::Relaxed);
        let buffer = if write_idx == 0 { &self.buffer_a } else { &self.buffer_b };
        for i in 0..EFFECTOR_STATE_WORDS {
            let lo = state[i * 2].to_bits() as u64;
            let hi = (state[i * 2 + 1].to_bits() as u64) << 32;
            buffer[i].store(lo | hi, Ordering::Relaxed);
        }
        // Ensure all buffer writes are visible before readable flip
        std::sync::atomic::fence(Ordering::Release);
        self.readable.store(write_idx, Ordering::Release);
    }

    /// Read the latest committed effector state. Returns `false` if
    /// a torn read was detected (reader should retry).
    #[inline]
    pub fn read(&self, out: &mut [f32; EFFECTOR_STATE_FLOATS]) -> bool {
        let idx = self.readable.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::Acquire);
        let buffer = if idx == 0 { &self.buffer_a } else { &self.buffer_b };
        for i in 0..EFFECTOR_STATE_WORDS {
            let word = buffer[i].load(Ordering::Relaxed);
            let lo = word as u32;
            let hi = (word >> 32) as u32;
            out[i * 2] = f32::from_bits(lo);
            out[i * 2 + 1] = f32::from_bits(hi);
        }
        // Verify no torn read: generation must match
        self.readable.load(Ordering::Acquire) == idx
    }
}

impl Default for VisualEffectorBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pack/unpack helpers for operating on [f32; 22] render state arrays.
pub mod effector_state {
    use super::*;

    /// Set a skeletal rotation angle (radians).
    #[inline]
    pub fn set_rotation(state: &mut [f32; EFFECTOR_STATE_FLOATS], joint: usize, radians: f32) {
        if joint < 6 {
            state[joint] = radians;
        }
    }

    /// Set palette color 0 (RGBA).
    #[inline]
    pub fn set_color0(state: &mut [f32; EFFECTOR_STATE_FLOATS], r: f32, g: f32, b: f32, a: f32) {
        state[6] = r;
        state[7] = g;
        state[8] = b;
        state[9] = a;
    }

    /// Set palette color 1 (RGBA).
    #[inline]
    pub fn set_color1(state: &mut [f32; EFFECTOR_STATE_FLOATS], r: f32, g: f32, b: f32, a: f32) {
        state[10] = r;
        state[11] = g;
        state[12] = b;
        state[13] = a;
    }

    /// Set skeletal scale.
    #[inline]
    pub fn set_scale(state: &mut [f32; EFFECTOR_STATE_FLOATS], x: f32, y: f32, z: f32) {
        state[14] = x;
        state[15] = y;
        state[16] = z;
    }

    /// Set blend weight [0,1].
    #[inline]
    pub fn set_blend(state: &mut [f32; EFFECTOR_STATE_FLOATS], blend: f32) {
        state[17] = blend;
    }

    /// Set palette interpolation coefficients.
    #[inline]
    pub fn set_palette_coeffs(state: &mut [f32; EFFECTOR_STATE_FLOATS], c0: f32, c1: f32) {
        state[18] = c0;
        state[19] = c1;
    }

    /// Set bone twist angle modifier.
    #[inline]
    pub fn set_bone_twist(state: &mut [f32; EFFECTOR_STATE_FLOATS], twist: f32) {
        state[20] = twist;
    }

    /// Set wireframe flag (0.0 = normal, 1.0 = wireframe).
    #[inline]
    pub fn set_wireframe(state: &mut [f32; EFFECTOR_STATE_FLOATS], wf: f32) {
        state[21] = wf;
    }
}

// ────────────────────────────────────────────────────────────
//  FrameSchema — compositional semantic framing
//
//  A FrameSchema defines mandatory relational slots that a
//  concept must fill to be "understood." For example,
//  CommercialTransfer requires Buyer, Seller, Goods, Money.
//
//  Meaning is evaluated by how well input satisfies frame
//  slot requirements.
// ────────────────────────────────────────────────────────────

/// Required role slot in a frame schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotBinding {
    /// Slot name (e.g., "buyer", "seller", "goods")
    pub name: String,
    /// Expected NodeType for the filler
    pub expected_type: NodeType,
    /// Expected Relation linking the frame node to the filler
    pub relation: Relation,
    /// Whether this slot is mandatory or optional
    pub required: bool,
}

/// A frame schema template — defines the structure of a concept
/// in terms of its mandatory and optional relational slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameSchema {
    /// Schema name (e.g., "CommercialTransfer")
    pub name: String,
    /// The base CD action this schema represents
    pub action: CDAction,
    /// Required and optional slot definitions
    pub slots: Vec<SlotBinding>,
}

/// An instantiated frame — a FrameSchema with bound slot fillers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameInstance {
    /// The schema this instance was created from
    pub schema_name: String,
    /// The node this frame is attached to
    pub frame_node: NodeId,
    /// Bound slots: slot_name → node_id
    pub bindings: Vec<(String, NodeId)>,
    /// Satisfaction score (0.0 = none, 1.0 = complete)
    pub satisfaction: f64,
}

impl FrameSchema {
    pub fn new(name: &str, action: CDAction) -> Self {
        FrameSchema { name: name.to_string(), action, slots: Vec::new() }
    }

    pub fn with_slot(mut self, name: &str, expected_type: NodeType, relation: Relation, required: bool) -> Self {
        self.slots.push(SlotBinding {
            name: name.to_string(),
            expected_type,
            relation,
            required,
        });
        self
    }

    /// Evaluate how well a set of edges satisfies this schema.
    /// Returns a satisfaction score [0.0, 1.0] and the set of
    /// bound slots.
    pub fn evaluate(&self, edges: &[Edge], graph: &GraphArena) -> (f64, Vec<(String, NodeId)>) {
        let mut bindings: Vec<(String, NodeId)> = Vec::new();
        let mut satisfied = 0usize;
        let total = self.slots.len();

        for slot in &self.slots {
            // Find an edge matching this slot's relation
            let mut found = false;
            for edge in edges {
                if edge.relation == slot.relation {
                    // Check if the target node matches the expected type
                    if let Some(node) = graph.get(edge.target) {
                        let n = node.read();
                        if n.node_type == slot.expected_type || slot.expected_type == NodeType::Entity {
                            bindings.push((slot.name.clone(), edge.target));
                            found = true;
                            break;
                        }
                    }
                }
            }
            if found {
                satisfied += 1;
            }
        }

        let satisfaction = if total == 0 { 1.0 } else { satisfied as f64 / total as f64 };
        (satisfaction, bindings)
    }
}

/// Built-in frame schemas.
pub mod frame_schemas {
    use super::*;

    pub fn commercial_transfer() -> FrameSchema {
        FrameSchema::new("CommercialTransfer", CDAction::Atrans)
            .with_slot("buyer", NodeType::Entity, Relation::Requires, true)
            .with_slot("seller", NodeType::Entity, Relation::CausedBy, true)
            .with_slot("goods", NodeType::Entity, Relation::HasProperty, true)
            .with_slot("money", NodeType::Concept, Relation::AssociatedWith, true)
    }

    pub fn motion_event() -> FrameSchema {
        FrameSchema::new("MotionEvent", CDAction::Ptrans)
            .with_slot("actor", NodeType::Entity, Relation::Requires, true)
            .with_slot("source", NodeType::Concept, Relation::CausedBy, false)
            .with_slot("goal", NodeType::Concept, Relation::Implies, true)
            .with_slot("instrument", NodeType::Concept, Relation::AssociatedWith, false)
    }

    pub fn sensory_event() -> FrameSchema {
        FrameSchema::new("SensoryEvent", CDAction::Attend)
            .with_slot("observer", NodeType::Entity, Relation::Requires, true)
            .with_slot("stimulus", NodeType::Concept, Relation::HasProperty, true)
            .with_slot("channel", NodeType::Sensor, Relation::GroundedIn, true)
    }

    pub fn ownership_change() -> FrameSchema {
        FrameSchema::new("OwnershipChange", CDAction::Grasp)
            .with_slot("owner", NodeType::Entity, Relation::Requires, true)
            .with_slot("possession", NodeType::Entity, Relation::HasProperty, true)
            .with_slot("previous_owner", NodeType::Entity, Relation::CausedBy, false)
    }

    /// Look up a schema by name.
    pub fn by_name(name: &str) -> Option<FrameSchema> {
        match name {
            "CommercialTransfer" => Some(commercial_transfer()),
            "MotionEvent" => Some(motion_event()),
            "SensoryEvent" => Some(sensory_event()),
            "OwnershipChange" => Some(ownership_change()),
            _ => None,
        }
    }
}

// Direct re-exports for external crates that need specific types.
pub use MotorCommandType;
pub use VisualEffectorBuffer;

// ────────────────────────────────────────────────────────────
//  Prelude
// ────────────────────────────────────────────────────────────

pub mod prelude {
    pub use super::{
        GroundedNode, NodeId, NodeType, Edge, Relation, Grounding, SensorNorm,
        ActivationBuffer, GraphArena, SemanticContext,
        ConceptualFrame, ConceptualFrame as CD, CDType, CDAction,
        SemanticRelation,
        PrimitiveVector, primitive_for, base_primitives,
        FrameSchema, SlotBinding, FrameInstance, frame_schemas,
    };
    pub use super::{
        Neuromodulator, FiringHistory, PredictionError,
        MotorCommandType,
        LTP_WINDOW, ELIGIBILITY_DECAY, LTP_RATE, DRIFT_RATE,
        PRUNE_THRESHOLD, PREDICTION_ERROR_THRESHOLD,
    };
    pub use super::{
        InvariantContract, DataType, StructuralError,
    };
    pub use super::{
        VisualEffectorBuffer, effector_state,
        EFFECTOR_STATE_FLOATS, EFFECTOR_STATE_WORDS,
    };
}
