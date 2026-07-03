// ────────────────────────────────────────────────────────────
//  Lock-Free Contextual Hot-Swap
//
//  Provides a thread-safe double-buffer mechanism for swapping
//  Layer 1 cognitive module implementations at runtime.
//
//  Design: each ModuleId has a corresponding SwapSlot that holds
//  two copies of the trait object: the "active" (currently serving)
//  and the "staging" (candidate awaiting activation). A generation
//  counter (AtomicU64) tracks which buffer is active.
//
//  Reader lock-freedom:
//    - The hot path (cognitive tick) reads the active trait object
//      via a load-acquire on the generation counter.
//    - No mutex is held during module execution.
//    - The staging buffer is written offline, then atomically
//      activated by incrementing the generation counter.
//
//  This mirrors the `VisualEffectorBuffer` double-buffer pattern
//  already in Grounded.
// ────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::layers::*;
use crate::capability::CapabilityMetrics;

// ── Swap Slot ─────────────────────────────────────────────

/// A double-buffered slot for one Layer 1 module.
///
/// `active_gen` bit 0 selects the active buffer:
///   0 → buffer A is active, buffer B is staging
///   1 → buffer B is active, buffer A is staging
///
/// Readers acquire-load `active_gen`, then read the selected buffer.
/// Writers write to the staging buffer, then flip `active_gen`.
pub struct SwapSlot<T: ?Sized + Send + Sync> {
    /// Two buffers for double-buffering.
    buffers: [Box<T>; 2],

    /// Bit 0 selects active buffer. Incremented on every flip.
    /// Reader: load-acquire → mask bit 0 → read buffer.
    /// Writer: write staging → flip with store-release.
    active_gen: AtomicU64,
}

impl<T: ?Sized + Send + Sync> SwapSlot<T> {
    /// Create a new swap slot with the given initial module.
    pub fn new(initial: Box<T>, fallback: Box<T>) -> Self {
        SwapSlot {
            buffers: [initial, fallback],
            active_gen: AtomicU64::new(0),
        }
    }

    /// Read the currently active module.
    /// Returns a reference valid until the next swap.
    /// Callers must not hold the reference across trait method calls
    /// that might trigger a swap (re-entrancy guard).
    #[inline]
    pub fn read(&self) -> &T {
        let gen = self.active_gen.load(Ordering::Acquire);
        let idx = (gen & 1) as usize;
        &*self.buffers[idx]
    }

    /// Stage a new module for activation (writes to staging buffer).
    /// Does NOT activate it — call `flip()` to atomically switch.
    pub fn stage(&mut self, new_module: Box<T>) {
        let gen = self.active_gen.load(Ordering::Acquire);
        let staging_idx = ((gen & 1) ^ 1) as usize;
        self.buffers[staging_idx] = new_module;
    }

    /// Atomically flip the active buffer.
    /// After this returns, all subsequent `read()` calls see the staged module.
    pub fn flip(&self) {
        self.active_gen.fetch_add(1, Ordering::Release);
    }

    /// Get the current generation (for debugging).
    pub fn generation(&self) -> u64 {
        self.active_gen.load(Ordering::Acquire)
    }

    /// Get references to both buffers (for benchmarking comparisons).
    pub fn both(&self) -> (&T, &T) {
        let gen = self.active_gen.load(Ordering::Acquire);
        match (gen & 1) == 0 {
            true => (&*self.buffers[0], &*self.buffers[1]),
            false => (&*self.buffers[1], &*self.buffers[0]),
        }
    }
}

// ── Module Swap Table ─────────────────────────────────────

/// The top-level registry of all hot-swappable modules.
/// Thread-safe: reads take `&self`, writes via interior `parking_lot::Mutex`.
pub struct ModuleSwapTable {
    /// Double-buffered cognitive parser.
    pub parser: SwapSlot<dyn CognitiveParser>,
    /// Double-buffered frame matcher.
    pub frame_matcher: SwapSlot<dyn FrameMatcher>,
    /// Double-buffered curiosity scheduler.
    pub curiosity_scheduler: SwapSlot<dyn CuriosityScheduler>,
    /// Double-buffered gap detector.
    pub gap_detector: SwapSlot<dyn GapDetectorModule>,
    /// Single-buffered exploration policy (Layer 2 — no swap needed).
    pub exploration_policy: Arc<parking_lot::Mutex<Box<dyn ExplorationPolicy>>>,
    /// Single-buffered inference order (Layer 2).
    pub inference_order: Arc<parking_lot::Mutex<Box<dyn InferenceOrder>>>,
}

impl ModuleSwapTable {
    /// Build the swap table from a ModuleRegistry, cloning each module
    /// into both buffers so the initial state has identical active + fallback.
    pub fn from_registry(registry: &ModuleRegistry) -> Self {
        ModuleSwapTable {
            parser: SwapSlot::new(
                registry.parser.box_clone(),
                registry.parser.box_clone(),
            ),
            frame_matcher: SwapSlot::new(
                registry.frame_matcher.box_clone(),
                registry.frame_matcher.box_clone(),
            ),
            curiosity_scheduler: SwapSlot::new(
                registry.curiosity_scheduler.box_clone(),
                registry.curiosity_scheduler.box_clone(),
            ),
            gap_detector: SwapSlot::new(
                registry.gap_detector.box_clone(),
                registry.gap_detector.box_clone(),
            ),
            exploration_policy: Arc::new(parking_lot::Mutex::new(
                registry.exploration_policy.box_clone(),
            )),
            inference_order: Arc::new(parking_lot::Mutex::new(
                registry.inference_order.box_clone(),
            )),
        }
    }

    /// Hot-swap a Layer 1 module by ModuleId with a new implementation.
    ///
    /// This is the primary API for the Self-Healing Pipeline.
    ///
    /// # Safety
    /// - The new module must satisfy the same trait bound (enforced by type).
    /// - The swap is atomic: readers see either the old or new, never a mix.
    pub fn hot_swap(
        &mut self,
        module_id: ModuleId,
        new_impl: Box<dyn CognitiveParser>,
    ) -> Result<(), &'static str> {
        match module_id {
            ModuleId::PARSER_MODULE => {
                self.parser.stage(new_impl);
                self.parser.flip();
                Ok(())
            }
            _ => Err("module ID does not match CognitiveParser"),
        }
    }

    /// Hot-swap the frame matcher.
    pub fn hot_swap_frame_matcher(
        &mut self,
        new_impl: Box<dyn FrameMatcher>,
    ) {
        self.frame_matcher.stage(new_impl);
        self.frame_matcher.flip();
    }

    /// Hot-swap the curiosity scheduler.
    pub fn hot_swap_curiosity_scheduler(
        &mut self,
        new_impl: Box<dyn CuriosityScheduler>,
    ) {
        self.curiosity_scheduler.stage(new_impl);
        self.curiosity_scheduler.flip();
    }

    /// Hot-swap the gap detector.
    pub fn hot_swap_gap_detector(
        &mut self,
        new_impl: Box<dyn GapDetectorModule>,
    ) {
        self.gap_detector.stage(new_impl);
        self.gap_detector.flip();
    }

    /// Read the current metrics from all active modules.
    pub fn collect_metrics(&self) -> Vec<CapabilityMetrics> {
        vec![
            self.parser.read().metrics(),
            self.frame_matcher.read().metrics(),
            self.curiosity_scheduler.read().metrics(),
            self.gap_detector.read().metrics(),
        ]
    }
}

// ── Ecological Benchmarking Context ───────────────────────

/// Holds references to both the active (stock) and staging (candidate)
/// modules for A/B comparison during Phase 4 (Ecological Benchmarking).
///
/// The benchmark runs both modules on the same input and compares
/// latency, output equivalence, and memory footprint.
pub struct BenchmarkContext<'a> {
    /// Active module reference (stock, currently serving).
    pub active_parser: &'a dyn CognitiveParser,
    /// Staging module reference (candidate, not yet active).
    pub candidate_parser: &'a dyn CognitiveParser,
}

impl<'a> BenchmarkContext<'a> {
    /// Create a benchmark context from a SwapSlot.
    /// `active` = whichever buffer is currently serving.
    /// `candidate` = the staging buffer (will become active after flip).
    pub fn from_slot(slot: &'a SwapSlot<dyn CognitiveParser>) -> Self {
        let (active, candidate) = slot.both();
        BenchmarkContext {
            active_parser: active,
            candidate_parser: candidate,
        }
    }
}
