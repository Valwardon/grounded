// ────────────────────────────────────────────────────────────
//  Episodic Consolidation — promote ring buffer events into
//  the semantic graph as Episode nodes linked to SELF.
//
//  Runs during idle cycles (alongside garbage collection and
//  the self-healing pipeline). Groups nearby events into
//  episode clusters, computes importance, and creates graph
//  nodes for the most significant experiences.
// ────────────────────────────────────────────────────────────

use std::sync::Arc;

use cognitive_core::EpisodicEvent;
use semantic_graph::prelude::*;

use crate::record::*;

/// Minimum importance threshold for an episode to be promoted
/// into the semantic graph. Below this, raw records are discarded.
pub const EPISODE_IMPORTANCE_THRESHOLD: f64 = 0.15;

/// Maximum number of episode nodes to create per consolidation cycle.
pub const MAX_EPISODES_PER_CYCLE: usize = 16;

/// Maximum tick gap between two events to be grouped into the same episode.
pub const EPISODE_TICK_WINDOW: u64 = 5;

/// Consolidate raw ring buffer records into semantic graph episode nodes.
///
/// Returns a human-readable summary of what was promoted.
pub fn consolidate_episodes(
    ctx: &SemanticContext,
    buffer: &EpisodicRingBuffer,
) -> String {
    let records = buffer.drain_all();
    if records.is_empty() {
        return String::new();
    }

    // Group temporally adjacent records into episode clusters
    let clusters = group_by_proximity(&records);

    let mut promoted = 0usize;
    let mut total_importance = 0.0f64;
    let mut graph = ctx.graph.write();

    for cluster in clusters.iter().take(MAX_EPISODES_PER_CYCLE) {
        let importance = compute_importance(cluster);

        if importance < EPISODE_IMPORTANCE_THRESHOLD {
            continue;
        }

        let summary = summarize_cluster(cluster);
        let tick = cluster.first().map(|r| r.tick).unwrap_or(0);
        let ts = cluster.first().map(|r| r.timestamp_ms).unwrap_or(0);

        let episode_node = GroundedNode {
            id: NodeId::ZERO,
            label: summary,
            node_type: NodeType::Episode,
            grounding: Grounding::Episode { tick, timestamp_ms: ts, importance },
            decay: 0.95,
            threshold: f64::MAX,
            base_activation: importance,
            edges: Vec::new(),
            epistemic_status: EpistemicStatus::CoreConcept,
            valence: compute_valence_shift(cluster),
            mean_error: 0.0,
            variance: 0.0,
        };

        let episode_id = graph.insert(episode_node);

        // Link SELF → Episode with Relation::Experienced
        graph.link_to_self(Relation::Experienced, episode_id);

        // Link involved nodes as AssociatedWith
        let mut involved = Vec::new();
        for rec in cluster {
            if rec.node_id_a != 0 && !involved.contains(&rec.node_id_a) {
                involved.push(rec.node_id_a);
            }
            if rec.node_id_b != 0 && !involved.contains(&rec.node_id_b) {
                involved.push(rec.node_id_b);
            }
        }
        for node_id in &involved {
            if let Some(node) = graph.get(NodeId::from_raw(*node_id)) {
                node.write().edges.push(Edge::new(
                    Relation::AssociatedWith,
                    episode_id,
                ));
            }
        }

        promoted += 1;
        total_importance += importance;
    }

    // Link consecutive episodes with Relation::Precedes
    if promoted > 1 {
        let episode_ids: Vec<NodeId> = graph.nodes()
            .iter()
            .filter_map(|n| {
                let node = n.read();
                if matches!(node.node_type, NodeType::Episode) {
                    Some(node.id)
                } else {
                    None
                }
            })
            .collect();

        // Sort by tick
        let mut with_ticks: Vec<(NodeId, u64)> = episode_ids.iter()
            .filter_map(|id| {
                let node = graph.get(*id)?;
                let n = node.read();
                match &n.grounding {
                    Grounding::Episode { tick, .. } => Some((n.id, *tick)),
                    _ => None,
                }
            })
            .collect();
        with_ticks.sort_by_key(|(_, t)| *t);

        // Link consecutive episodes
        for i in 1..with_ticks.len() {
            let prev = with_ticks[i - 1].0;
            let next = with_ticks[i].0;
            if let Some(node) = graph.get(prev) {
                node.write().edges.push(Edge::new(Relation::Precedes, next));
            }
        }
    }

    if promoted > 0 {
        format!(
            "Episodic: promoted {} episodes (avg importance {:.2})",
            promoted,
            total_importance / promoted as f64,
        )
    } else {
        String::new()
    }
}

/// Group temporally adjacent records into episode clusters.
fn group_by_proximity(records: &[RawEpisodicRecord]) -> Vec<Vec<RawEpisodicRecord>> {
    let mut clusters: Vec<Vec<RawEpisodicRecord>> = Vec::new();
    let mut current: Vec<RawEpisodicRecord> = Vec::new();

    for rec in records {
        if current.is_empty() {
            current.push(*rec);
        } else {
            let last_tick = current.last().unwrap().tick;
            if rec.tick.wrapping_sub(last_tick) <= EPISODE_TICK_WINDOW {
                current.push(*rec);
            } else {
                clusters.push(std::mem::take(&mut current));
                current.push(*rec);
            }
        }
    }
    if !current.is_empty() {
        clusters.push(current);
    }
    clusters
}

/// Compute the importance of an episode cluster (0.0 – 1.0).
///
/// Factors:
///   - Prediction errors → high importance (surprise)
///   - Structural faults → high importance (failures to learn from)
///   - High novelty/arousal → medium importance (emotionally salient)
///   - High reward → medium importance (successful predictions)
///   - More events → higher importance (richer experiences)
fn compute_importance(cluster: &[RawEpisodicRecord]) -> f64 {
    let mut has_error = false;
    let mut has_fault = false;
    let mut max_novelty = 0.0f64;
    let mut max_arousal = 0.0f64;
    let mut max_reward = 0.0f64;

    for rec in cluster {
        match rec.event_type() {
            1 => has_error = true,   // PredictionError
            2 => has_fault = true,   // StructuralFault
            _ => {}
        }
        max_novelty = max_novelty.max(rec.novelty() as f64);
        max_arousal = max_arousal.max(rec.arousal() as f64);
        max_reward = max_reward.max(rec.reward() as f64);
    }

    let error_factor = if has_error { 0.4 } else { 0.0 };
    let fault_factor = if has_fault { 0.5 } else { 0.0 };
    let novelty_factor = max_novelty * 0.3;
    let arousal_factor = max_arousal * 0.2;
    let reward_factor = max_reward * 0.15;
    let density_factor = (cluster.len() as f64).min(10.0) / 10.0 * 0.15;

    (error_factor + fault_factor + novelty_factor + arousal_factor + reward_factor + density_factor)
        .clamp(0.0, 1.0)
}

/// Compute the net valence shift for an episode cluster (-1.0 to +1.0).
fn compute_valence_shift(cluster: &[RawEpisodicRecord]) -> f64 {
    let mut shift = 0.0f64;
    for rec in cluster {
        match rec.event_type() {
            1 => shift -= rec.payload as f64 * 0.01, // PredictionError → negative
            2 => shift -= 0.1,                         // StructuralFault → negative
            _ => shift += rec.reward() as f64 * 0.05, // Other → small positive if rewarding
        }
    }
    shift.clamp(-1.0, 1.0)
}

/// Generate a human-readable summary label for an episode cluster.
fn summarize_cluster(cluster: &[RawEpisodicRecord]) -> String {
    let mut event_types: Vec<&str> = Vec::with_capacity(cluster.len());
    for rec in cluster {
        let label = match rec.event_type() {
            0 => "fired",
            1 => "error",
            2 => "fault",
            3 => "sensor",
            4 => "intent",
            _ => "unknown",
        };
        if !event_types.contains(&label) {
            event_types.push(label);
        }
    }
    format!(
        "episode:{}:tick={}:{}",
        cluster.first().map(|r| r.node_id_a).unwrap_or(0) % 10000,
        cluster.first().map(|r| r.tick).unwrap_or(0),
        event_types.join("+"),
    )
}
