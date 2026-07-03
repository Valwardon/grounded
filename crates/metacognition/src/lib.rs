// ────────────────────────────────────────────────────────────
//  Metacognition Crate
//
//  Self-healing evolutionary pipeline for Layer 1 cognitive
//  modules. Provides the DSL, bytecode interpreter, double-buffer
//  hot-swap mechanism, deficiency scanner, and the 5-phase
//  SelfHealingPipeline that runs during idle consolidation.
//
//  Crate-level re-exports for external consumers (hw-daemon, etc.)
// ────────────────────────────────────────────────────────────

pub mod capability;
pub mod dsl;
pub mod hotswap;
pub mod layers;
pub mod metacuriosity;
pub mod pipeline;

pub use capability::*;
pub use dsl::*;
pub use hotswap::*;
pub use layers::*;
pub use metacuriosity::*;
pub use pipeline::*;
