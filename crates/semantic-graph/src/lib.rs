use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

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
                if *raw as f64 > *threshold {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Grounded concept mapping
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Grounding {
    Sensor {
        sensor_type: String,
        channel: u8,
        norm: SensorNorm,
    },
    Action {
        intent_template: String,
    },
    Stored {
        keyspace: String,
        key: String,
    },
    HardwareQuery {
        /// For query-based grounding we store the query type as a string;
        /// the actual query function is registered in the daemon at init time.
        query_type: String,
    },
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Entity,
    Concept,
    Action,
    Sensor,
    State,
    Frame,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub relation: Relation,
    pub target: NodeId,
    pub weight_override: Option<f64>,
}

impl Edge {
    pub fn effective_weight(&self) -> f64 {
        self.weight_override
            .unwrap_or_else(|| self.relation.spread_weight())
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
        self.label_index
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, id)| *id)
    }

    pub fn by_type(&self, ty: NodeType) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.read().node_type == ty)
            .map(|n| n.read().id)
            .collect()
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
    /// Returns true if the target existed.
    pub fn link_to_self(&mut self, relation: Relation, target: NodeId) -> bool {
        if target.0 as usize >= self.nodes.len() || target == NodeId::ZERO || target == NodeId::SELF {
            return false;
        }
        let self_idx = NodeId::SELF.0 as usize;
        if self_idx >= self.nodes.len() {
            return false;
        }
        self.nodes[self_idx].write().edges.push(Edge {
            relation,
            target,
            weight_override: None,
        });
        true
    }

    /// Return all nodes directly connected to the self node.
    pub fn introspect(&self) -> Vec<(NodeId, String, Relation)> {
        let self_idx = NodeId::SELF.0 as usize;
        if self_idx >= self.nodes.len() {
            return Vec::new();
        }
        let self_node = self.nodes[self_idx].read();
        self_node
            .edges
            .iter()
            .filter_map(|edge| {
                let idx = edge.target.0 as usize;
                if idx >= self.nodes.len() {
                    return None;
                }
                let node = self.nodes[idx].read();
                Some((edge.target, node.label.clone(), edge.relation))
            })
            .collect()
    }
}

// ────────────────────────────────────────────────────────────
//  Lock-free activation double buffer
// ────────────────────────────────────────────────────────────

use std::sync::atomic::AtomicU8;

pub struct ActivationBuffer {
    buffers: [Box<[f64]>; 2],
    active: AtomicU8,
    len: usize,
}

impl ActivationBuffer {
    pub fn new(len: usize) -> Self {
        let zero = vec![0.0; len].into_boxed_slice();
        ActivationBuffer {
            buffers: [zero.clone(), zero],
            active: AtomicU8::new(0),
            len,
        }
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

    /// Return everything the self node is connected to.
    pub fn introspect(&self) -> Vec<(NodeId, String, Relation)> {
        self.graph.read().introspect()
    }

    /// Link a node to self with a given relation.
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
    Action,
    StateChange,
    MentalEvent,
    PhysicalTransfer,
    OwnershipTransfer,
    SensorEvent,
    SystemCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CDAction {
    Atrans,
    Ptrans,
    Mtrans,
    Mbuild,
    Propel,
    Ingest,
    Expel,
    Move,
    Grasp,
    Speak,
    Attend,
    SystemAction,
    SensorReading,
}

impl ConceptualFrame {
    pub fn new(action: CDAction) -> Self {
        ConceptualFrame {
            frame_type: CDType::Action,
            actor: None,
            action,
            object: None,
            recipient: None,
            instrument: None,
            source: None,
            goal: None,
            time: None,
        }
    }

    pub fn injection_targets(&self, base_energy: f64) -> Vec<(NodeId, f64)> {
        let mut targets = Vec::with_capacity(6);
        if let Some(a) = self.actor {
            targets.push((a, base_energy * 0.8));
        }
        if let Some(o) = self.object {
            targets.push((o, base_energy * 1.0));
        }
        if let Some(r) = self.recipient {
            targets.push((r, base_energy * 0.6));
        }
        if let Some(inst) = self.instrument {
            targets.push((inst, base_energy * 0.4));
        }
        if let Some(s) = self.source {
            targets.push((s, base_energy * 0.5));
        }
        if let Some(g) = self.goal {
            targets.push((g, base_energy * 0.7));
        }
        targets
    }
}

pub mod prelude {
    pub use super::{
        GroundedNode, NodeId, NodeType, Edge, Relation, Grounding, SensorNorm,
        ActivationBuffer, GraphArena, SemanticContext,
        ConceptualFrame, ConceptualFrame as CD, CDType, CDAction,
        SemanticRelation,
    };
}
