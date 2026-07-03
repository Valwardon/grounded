// ────────────────────────────────────────────────────────────
//  Engine Layering Topology
//
//  Three explicit layers enforced via trait + visibility boundaries:
//
//    Layer 0 (Physics - Immutable):
//      Primitives, Matrix Algebra, STDP, Energy Conservation,
//      Memory Allocator. Completely closed to runtime modification.
//      Lives in `semantic_graph` — no metacognition trait exposed.
//
//    Layer 1 (Cognitive Modules - Self-Healing & Mutable):
//      Gap Detector, CCG Parser, Frame Matcher, Curiosity Scheduler.
//      These implement rigid, statically typed traits defined below.
//      The Self-Healing Pipeline can hot-swap their implementations
//      at runtime via the double-buffer swap mechanism.
//
//    Layer 2 (Strategies - Highly Mutable):
//      Exploration Policies, Inference Order, Goal Planning.
//      Fully replaceable — no trait bound enforcement beyond Send+Sync.
// ────────────────────────────────────────────────────────────

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use semantic_graph::prelude::*;
use semantic_parser::relational::RelationalParser;
use crate::capability::CapabilityMetrics;

// ── Module Identity ───────────────────────────────────────

/// Unique identifier for an installed cognitive module.
/// Maps 1:1 to a slot in the hot-swap double-buffer table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub u64);

impl ModuleId {
    pub const PARSER_MODULE: Self = ModuleId(1);
    pub const FRAME_MATCHER: Self = ModuleId(2);
    pub const CURIOSITY_SCHEDULER: Self = ModuleId(3);
    pub const GAP_DETECTOR: Self = ModuleId(4);
    pub const EXPLORATION_POLICY: Self = ModuleId(5);

    pub const fn from_raw(raw: u64) -> Self {
        ModuleId(raw)
    }
}

// ── Layer 0 Marker ────────────────────────────────────────

/// Marker trait for Layer 0 (Physics) components.
/// Implementations are permanently closed to modification.
/// No runtime swap is ever permitted — enforced by the absence
/// of any swap slot in the ModuleRegistry.
///
/// # Safety
/// This trait is NOT `dyn`-safe for swapping. It exists only to
/// document the layer boundary at the type level.
pub trait Layer0Immutable: Send + Sync {}

// ── Layer 1 Traits (Self-Healing) ─────────────────────────

/// A self-healing cognitive parser.
/// The engine may replace the active implementation at runtime
/// when latency, accuracy, or memory constraints are violated.
pub trait CognitiveParser: Send + Sync {
    /// Parse a sequence of tokens into conceptual frames.
    fn parse(&self, tokens: &[String]) -> Result<Vec<ConceptualFrame>, ParseError>;

    /// Unique module identifier for swap-table lookup.
    fn module_id(&self) -> ModuleId;

    /// Clone into a new boxed instance (for candidate staging).
    fn box_clone(&self) -> Box<dyn CognitiveParser>;

    /// Current capability metrics (latency, success rate, etc.).
    fn metrics(&self) -> CapabilityMetrics;
}

/// A self-healing frame matcher.
/// Evaluates how well a FrameInstance's slot bindings satisfy a schema.
pub trait FrameMatcher: Send + Sync {
    fn match_frame(&self, instance: &FrameInstance, schema: &FrameSchema, graph: &GraphArena) -> f64;
    fn module_id(&self) -> ModuleId;
    fn box_clone(&self) -> Box<dyn FrameMatcher>;
    fn metrics(&self) -> CapabilityMetrics;
}

/// A self-healing curiosity scheduler.
/// Determines which knowledge gap to explore next, subject to budget.
pub trait CuriosityScheduler: Send + Sync {
    fn schedule_next(&self, gaps: &[KnowledgeGap], budget: &CuriosityBudget) -> Option<usize>;
    fn module_id(&self) -> ModuleId;
    fn box_clone(&self) -> Box<dyn CuriosityScheduler>;
    fn metrics(&self) -> CapabilityMetrics;
}

/// A self-healing gap detector.
/// Scans token streams for ungrounded concepts and produces KnowledgeGaps.
pub trait GapDetectorModule: Send + Sync {
    fn detect_gaps(&self, tokens: &[String], source: GapSource, ctx: &SemanticContext) -> Vec<KnowledgeGap>;
    fn module_id(&self) -> ModuleId;
    fn box_clone(&self) -> Box<dyn GapDetectorModule>;
    fn metrics(&self) -> CapabilityMetrics;
}

// ── Layer 2 Traits (Highly Mutable) ───────────────────────

/// An exploration policy — which gap to resolve first.
/// Layer 2: fully replaceable, no trait bound beyond Send+Sync.
pub trait ExplorationPolicy: Send + Sync {
    fn rank_gaps(&self, gaps: &[KnowledgeGap]) -> Vec<usize>;
    fn box_clone(&self) -> Box<dyn ExplorationPolicy>;
}

/// Inference ordering strategy — determines the order in which
/// conceptual frames are evaluated during spreading activation.
pub trait InferenceOrder: Send + Sync {
    fn order_frames(&self, frames: &[ConceptualFrame]) -> Vec<usize>;
    fn box_clone(&self) -> Box<dyn InferenceOrder>;
}

// ── Error Types ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ParseError {
    UnknownToken { token: String },
    AmbiguousReduction { tokens: Vec<String> },
    FrameUnsatisfiable { reason: String },
    InternalError { module: ModuleId, detail: String },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownToken { token } => write!(f, "unknown token: {}", token),
            ParseError::AmbiguousReduction { tokens } => {
                write!(f, "ambiguous reduction for tokens: {:?}", tokens)
            }
            ParseError::FrameUnsatisfiable { reason } => {
                write!(f, "frame unsatisfiable: {}", reason)
            }
            ParseError::InternalError { module, detail } => {
                write!(f, "internal error in module {:?}: {}", module, detail)
            }
        }
    }
}

// ── Module Registry ────────────────────────────────────────

/// Registered set of active Layer 1 and Layer 2 modules.
/// The Self-Healing Pipeline swaps entries in this registry.
pub struct ModuleRegistry {
    /// Active cognitive parser (Layer 1, self-healing).
    pub parser: Box<dyn CognitiveParser>,
    /// Active frame matcher (Layer 1, self-healing).
    pub frame_matcher: Box<dyn FrameMatcher>,
    /// Active curiosity scheduler (Layer 1, self-healing).
    pub curiosity_scheduler: Box<dyn CuriosityScheduler>,
    /// Active gap detector (Layer 1, self-healing).
    pub gap_detector: Box<dyn GapDetectorModule>,
    /// Active exploration policy (Layer 2, highly mutable).
    pub exploration_policy: Box<dyn ExplorationPolicy>,
    /// Active inference order strategy (Layer 2, highly mutable).
    pub inference_order: Box<dyn InferenceOrder>,
}

impl ModuleRegistry {
    /// Create a new registry populated with the default (stock) modules.
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        ModuleRegistry {
            parser: Box::new(StockCognitiveParser::new(ctx.clone())),
            frame_matcher: Box::new(StockFrameMatcher),
            curiosity_scheduler: Box::new(StockCuriosityScheduler),
            gap_detector: Box::new(StockGapDetector::new(ctx)),
            exploration_policy: Box::new(StockExplorationPolicy),
            inference_order: Box::new(StockInferenceOrder),
        }
    }

    /// Look up a Layer 1 module by its ModuleId for hot-swap targeting.
    pub fn get_layer1_mut(&mut self, id: ModuleId) -> Option<&mut dyn Any> {
        match id {
            ModuleId::PARSER_MODULE => Some(&mut self.parser),
            ModuleId::FRAME_MATCHER => Some(&mut self.frame_matcher),
            ModuleId::CURIOSITY_SCHEDULER => Some(&mut self.curiosity_scheduler),
            ModuleId::GAP_DETECTOR => Some(&mut self.gap_detector),
            _ => None,
        }
    }
}

// ── Stock Module Implementations ──────────────────────────

pub struct StockCognitiveParser {
    ctx: Arc<SemanticContext>,
    metrics: CapabilityMetrics,
}

impl StockCognitiveParser {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        StockCognitiveParser {
            ctx,
            metrics: CapabilityMetrics::new("stock_ccg_parser"),
        }
    }
}

impl CognitiveParser for StockCognitiveParser {
    fn parse(&self, tokens: &[String]) -> Result<Vec<ConceptualFrame>, ParseError> {
        let start = std::time::Instant::now();
        let parser = RelationalParser::new();
        let frames = parser.parse(tokens)
            .map_err(|e| ParseError::AmbiguousReduction { tokens: tokens.to_vec() })?;
        let elapsed = start.elapsed();
        // Track metrics via AtomicU64 interior mutability (no &mut self needed)
        let success = if frames.is_empty() { 0.0 } else { 1.0 };
        self.metrics.record_sample(elapsed, success);
        Ok(frames)
    }

    fn module_id(&self) -> ModuleId { ModuleId::PARSER_MODULE }

    fn box_clone(&self) -> Box<dyn CognitiveParser> {
        Box::new(StockCognitiveParser {
            ctx: self.ctx.clone(),
            metrics: self.metrics.clone(),
        })
    }

    fn metrics(&self) -> CapabilityMetrics { self.metrics.clone() }
}

pub struct StockFrameMatcher;

impl FrameMatcher for StockFrameMatcher {
    fn match_frame(&self, instance: &FrameInstance, schema: &FrameSchema, graph: &GraphArena) -> f64 {
        schema.evaluate(&[], graph).0
    }

    fn module_id(&self) -> ModuleId { ModuleId::FRAME_MATCHER }
    fn box_clone(&self) -> Box<dyn FrameMatcher> { Box::new(StockFrameMatcher) }
    fn metrics(&self) -> CapabilityMetrics { CapabilityMetrics::new("stock_frame_matcher") }
}

pub struct StockCuriosityScheduler;

impl CuriosityScheduler for StockCuriosityScheduler {
    fn schedule_next(&self, gaps: &[KnowledgeGap], budget: &CuriosityBudget) -> Option<usize> {
        if gaps.is_empty() || budget.remaining < budget.halt_threshold(0.0) {
            return None;
        }
        // Default: pick the gap with the most remaining budget
        gaps.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.budget.remaining.partial_cmp(&b.budget.remaining).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    fn module_id(&self) -> ModuleId { ModuleId::CURIOSITY_SCHEDULER }
    fn box_clone(&self) -> Box<dyn CuriosityScheduler> { Box::new(StockCuriosityScheduler) }
    fn metrics(&self) -> CapabilityMetrics { CapabilityMetrics::new("stock_curiosity_scheduler") }
}

pub struct StockGapDetector {
    ctx: Arc<SemanticContext>,
}

impl StockGapDetector {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        StockGapDetector { ctx }
    }
}

impl GapDetectorModule for StockGapDetector {
    fn detect_gaps(&self, tokens: &[String], source: GapSource, ctx: &SemanticContext) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();
        for token in tokens {
            let graph = ctx.graph.read();
            if graph.find_by_label(token).is_none() {
                gaps.push(KnowledgeGap::new(token, source));
            }
        }
        gaps
    }

    fn module_id(&self) -> ModuleId { ModuleId::GAP_DETECTOR }
    fn box_clone(&self) -> Box<dyn GapDetectorModule> {
        Box::new(StockGapDetector { ctx: self.ctx.clone() })
    }
    fn metrics(&self) -> CapabilityMetrics { CapabilityMetrics::new("stock_gap_detector") }
}

pub struct StockExplorationPolicy;

impl ExplorationPolicy for StockExplorationPolicy {
    fn rank_gaps(&self, gaps: &[KnowledgeGap]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..gaps.len()).collect();
        indices.sort_by(|&a, &b| {
            gaps[b].budget.remaining.partial_cmp(&gaps[a].budget.remaining)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    fn box_clone(&self) -> Box<dyn ExplorationPolicy> { Box::new(StockExplorationPolicy) }
}

pub struct StockInferenceOrder;

impl InferenceOrder for StockInferenceOrder {
    fn order_frames(&self, frames: &[ConceptualFrame]) -> Vec<usize> {
        // Default: process in insertion order (no reordering)
        (0..frames.len()).collect()
    }

    fn box_clone(&self) -> Box<dyn InferenceOrder> { Box::new(StockInferenceOrder) }
}
