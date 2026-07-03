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
    Abstract,
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
}

impl Edge {
    pub fn new(relation: Relation, target: NodeId) -> Self {
        let base = relation.spread_weight();
        Edge { relation, target, weight_override: None, dynamic_weight: base, eligibility: 0.0 }
    }

    pub fn with_weight(relation: Relation, target: NodeId, override_weight: f64) -> Self {
        Edge { relation, target, weight_override: Some(override_weight), dynamic_weight: override_weight, eligibility: 0.0 }
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
//  Prelude
// ────────────────────────────────────────────────────────────

pub mod prelude {
    pub use super::{
        GroundedNode, NodeId, NodeType, Edge, Relation, Grounding, SensorNorm,
        ActivationBuffer, GraphArena, SemanticContext,
        ConceptualFrame, ConceptualFrame as CD, CDType, CDAction,
        SemanticRelation,
    };
    pub use super::{
        Neuromodulator, FiringHistory, PredictionError,
        LTP_WINDOW, ELIGIBILITY_DECAY, LTP_RATE, DRIFT_RATE,
        PRUNE_THRESHOLD, PREDICTION_ERROR_THRESHOLD,
    };
}
