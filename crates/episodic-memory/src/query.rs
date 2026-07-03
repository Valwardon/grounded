// ────────────────────────────────────────────────────────────
//  Episodic Query — retrieve and search the episodic timeline
//  stored in the semantic graph.
//
//  Queries scan Episode nodes linked to SELF via Relation::Experienced,
//  using the tick/timestamp metadata stored in Grounding::Episode.
// ────────────────────────────────────────────────────────────

use semantic_graph::prelude::*;

/// A promoted episode node with its metadata extracted from the graph.
#[derive(Debug, Clone)]
pub struct EpisodeSummary {
    pub node_id: NodeId,
    pub label: String,
    pub tick: u64,
    pub timestamp_ms: u64,
    pub importance: f64,
    pub valence: f64,
    pub involved_nodes: Vec<NodeId>,
    pub next_episode: Option<NodeId>,
}

/// Query the graph for episodes linked to SELF.
pub fn query_all_episodes(graph: &GraphArena) -> Vec<EpisodeSummary> {
    let self_node = match graph.get(NodeId::SELF) {
        Some(n) => n,
        None => return Vec::new(),
    };

    // Find all Episode nodes connected to SELF
    let mut episodes: Vec<EpisodeSummary> = Vec::new();
    for edge in &self_node.read().edges {
        if edge.relation == Relation::Experienced {
            if let Some(episode) = extract_episode(graph, edge.target) {
                episodes.push(episode);
            }
        }
    }

    episodes.sort_by(|a, b| b.tick.cmp(&a.tick)); // newest first
    episodes
}

/// Query the most recent N episodes.
pub fn query_recent(graph: &GraphArena, count: usize) -> Vec<EpisodeSummary> {
    let mut all = query_all_episodes(graph);
    all.truncate(count);
    all
}

/// Query episodes within a tick range [start, end].
pub fn query_tick_range(graph: &GraphArena, start: u64, end: u64) -> Vec<EpisodeSummary> {
    let mut episodes = Vec::new();
    let self_node = match graph.get(NodeId::SELF) {
        Some(n) => n,
        None => return episodes,
    };

    for edge in &self_node.read().edges {
        if edge.relation == Relation::Experienced {
            if let Some(episode) = extract_episode(graph, edge.target) {
                if episode.tick >= start && episode.tick <= end {
                    episodes.push(episode);
                }
            }
        }
    }

    episodes.sort_by(|a, b| b.tick.cmp(&a.tick));
    episodes
}

/// Query episodes involving a specific node (identified by label).
pub fn query_by_node_label(graph: &GraphArena, label: &str) -> Vec<EpisodeSummary> {
    let target = graph.find_by_label(label);
    let target_id = match target {
        Some(id) => id,
        None => return Vec::new(),
    };

    let mut episodes = Vec::new();
    let self_node = match graph.get(NodeId::SELF) {
        Some(n) => n,
        None => return episodes,
    };

    for edge in &self_node.read().edges {
        if edge.relation == Relation::Experienced {
            if let Some(episode) = extract_episode(graph, edge.target) {
                if episode.involved_nodes.contains(&target_id) {
                    episodes.push(episode);
                }
            }
        }
    }

    episodes.sort_by(|a, b| b.tick.cmp(&a.tick));
    episodes
}

/// Extract episode metadata from a graph node.
fn extract_episode(graph: &GraphArena, node_id: NodeId) -> Option<EpisodeSummary> {
    let node = graph.get(node_id)?;
    let guard = node.read();

    if !matches!(guard.node_type, NodeType::Episode) {
        return None;
    }

    let (tick, timestamp_ms, importance) = match &guard.grounding {
        Grounding::Episode { tick, timestamp_ms, importance } => {
            (*tick, *timestamp_ms, *importance)
        }
        _ => return None,
    };

    let involved_nodes: Vec<NodeId> = guard.edges.iter()
        .filter(|e| e.relation == Relation::AssociatedWith)
        .map(|e| e.target)
        .collect();

    let next_episode = guard.edges.iter()
        .find(|e| e.relation == Relation::Precedes)
        .map(|e| e.target);

    Some(EpisodeSummary {
        node_id: guard.id,
        label: guard.label.clone(),
        tick,
        timestamp_ms,
        importance,
        valence: guard.valence,
        involved_nodes,
        next_episode,
    })
}
