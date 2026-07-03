// ────────────────────────────────────────────────────────────
//  Episodic Memory Crate
//
//  A continuous, indexable timeline of lived experience.
//  Records events during the cognitive tick via lock-free SPSC
//  ring buffer, promotes important episodes into the semantic
//  graph during idle consolidation, and provides query APIs
//  for reflection and pattern detection.
//
//  Integrates with CognitiveDaemon via cognitive_core::EpisodicRecorder.
// ────────────────────────────────────────────────────────────

pub mod consolidation;
pub mod history;
pub mod query;
pub mod record;
pub mod ring;

pub use history::EpisodicHistory;
pub use query::*;
