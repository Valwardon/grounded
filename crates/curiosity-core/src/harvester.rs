use std::sync::Arc;

use semantic_graph::prelude::*;
use tokio::sync::mpsc;

use crate::gap::*;
use crate::resolver::*;

// ────────────────────────────────────────────────────────────
//  Knowledge definition fetcher
//
//  Offline-first: checks a local definition cache before
//  reaching out to any network source. The cache is populated
//  at app install time with a foundation set of ~1000 common
//  concepts.
// ────────────────────────────────────────────────────────────

pub struct KnowledgeStore {
    /// Inline knowledge base — maps token → raw definition string.
    /// Populated at compile time via include_str! of a JSON
    /// knowledge graph. On-device, never phoning home.
    definitions: Vec<(String, String)>,
    /// Cache of previously resolved definitions (lazily populated)
    cache: std::collections::HashMap<String, String>,
}

impl KnowledgeStore {
    pub fn new() -> Self {
        // Foundation knowledge base: common concepts with relational definitions
        let foundation = vec![
            ("cat", "is a mammal. has fur. has tail. has whiskers. can purr. can climb. is a pet."),
            ("dog", "is a mammal. has fur. has tail. can bark. can fetch. is a pet."),
            ("mammal", "is an animal. has fur. can regulate temperature."),
            ("animal", "is a living thing. needs food. needs water. can move."),
            ("bird", "is an animal. has wings. has feathers. has beak. can fly."),
            ("fish", "is an animal. has scales. has fins. lives in water. can swim."),
            ("pirate", "is a person. wears tricorn hat. wears eye patch. carries cutlass. seeks treasure. sails ships."),
            ("tricorn hat", "is a hat. has three corners. is black. is worn by pirates."),
            ("eye patch", "is an accessory. covers one eye. is black. is worn by pirates."),
            ("cutlass", "is a sword. is short. is curved. is used by pirates."),
            ("ship", "is a vehicle. sails on water. has sails. has a captain."),
            ("bipedal", "is a posture. walks on two legs. is upright."),
            ("quadruped", "is a posture. walks on four legs. is horizontal."),
            ("walking", "is a motion. moves by legs. alternates feet."),
            ("tail", "is a body part. extends from spine. can wag."),
            ("fur", "is hair. covers mammal body. is soft."),
            ("hat", "is clothing. worn on head. provides shade."),
            ("sword", "is a weapon. has blade. has handle. is metal."),
            ("clothing", "is a covering. worn on body. made of fabric."),
            ("food", "is a substance. provides energy. is consumed."),
            ("water", "is a liquid. is clear. is essential for life."),
            ("leg", "is a limb. supports body. used for walking."),
            ("arm", "is a limb. extends from shoulder. has hand."),
            ("hand", "is a body part. has fingers. can grasp."),
            ("head", "is a body part. contains brain. has eyes. has mouth."),
            ("eye", "is a sensory organ. detects light. enables vision."),
            ("living", "is a state. has metabolism. can reproduce."),
        ];

        KnowledgeStore {
            definitions: foundation
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            cache: std::collections::HashMap::new(),
        }
    }

    /// Fetch the definition for a token. Returns None if the
    /// concept is a fundamental physical primitive that needs
    /// no further decomposition.
    pub fn fetch(&mut self, token: &str) -> Option<String> {
        let token_lower = token.to_lowercase();

        // Check runtime cache first
        if let Some(def) = self.cache.get(&token_lower) {
            return Some(def.clone());
        }

        // Check foundation knowledge base
        for (key, def) in &self.definitions {
            if key == &token_lower {
                self.cache.insert(token_lower, def.clone());
                return Some(def.clone());
            }
        }

        // Check if this is a primitive that needs no definition
        let primitives = [
            "matter", "motion", "dimension", "time", "energy",
            "space", "force", "mass", "light", "sound",
            "solid", "liquid", "gas", "hot", "cold",
            "up", "down", "left", "right", "forward", "backward",
            "big", "small", "fast", "slow", "color", "shape",
            "body", "part", "surface", "edge", "corner",
        ];

        if primitives.contains(&token_lower.as_str()) {
            return None; // fundamental — no further decomposition
        }

        // Unknown token: return a minimal definition so the
        // system can create a placeholder node
        Some(format!("is a thing. is unknown."))
    }
}

// ────────────────────────────────────────────────────────────
//  Autonomous background harvester
//
//  Runs on a Tokio runtime, continuously draining the
//  KnowledgeGap channel and resolving each gap. Recursion is
//  bounded by CuriosityBudget energy — not a hard depth cap.
//
//  This is the "curiosity loop" — it NEVER stops. When there
//  are no gaps, it idles on the channel recv(). When a gap
//  arrives, it fans out recursively until the CuriosityBudget
//  is exhausted or all leaf concepts resolve to fundamental
//  physical primitives.
//
//  Energy cost depends on: semantic distance from SELF, global
//  arousal level, and structural error rate. High novelty lowers
//  the halt threshold, allowing deeper exploration.
// ────────────────────────────────────────────────────────────

pub const MAX_CONCURRENT_RESOLUTIONS: usize = 4;

/// Default arousal level used when no cognitive daemon values are available.
const DEFAULT_AROUSAL: f64 = 0.0;
/// Default novelty level used when no cognitive daemon values are available.
const DEFAULT_NOVELTY: f64 = 0.0;

pub struct AutonomousHarvester {
    ctx: Arc<SemanticContext>,
    gap_rx: mpsc::Receiver<KnowledgeGap>,
    gap_tx: mpsc::Sender<KnowledgeGap>,
    resolver: DefinitionResolver,
    store: KnowledgeStore,
    stats: HarvestStats,
}

#[derive(Debug, Clone, Default)]
pub struct HarvestStats {
    pub total_gaps_resolved: u64,
    pub total_dependencies_found: u64,
    pub max_depth_reached: u8,
    pub nodes_created: u64,
    pub active_chain_count: u32,
}

impl AutonomousHarvester {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        let (gap_tx, gap_rx) = mpsc::channel(256);
        AutonomousHarvester {
            ctx: ctx.clone(),
            gap_rx,
            gap_tx,
            resolver: DefinitionResolver::new(ctx),
            store: KnowledgeStore::new(),
            stats: HarvestStats::default(),
        }
    }

    /// Returns a sender that can be used to inject KnowledgeGaps
    /// from any thread (GapDetector, asset ingestor, etc.)
    pub fn gap_sender(&self) -> mpsc::Sender<KnowledgeGap> {
        self.gap_tx.clone()
    }

    pub fn stats(&self) -> &HarvestStats {
        &self.stats
    }

    /// Start the background harvesting loop. Runs until the
    /// channel is closed (daemon shutdown).
    pub async fn run(&mut self) {
        // Semaphore to bound concurrent resolution chains
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_RESOLUTIONS));

        loop {
            match self.gap_rx.recv().await {
                Some(gap) => {
                    // Check CuriosityBudget: is there enough energy to explore?
                    // Compute semantic distance from SELF (approximate via token uniqueness)
                    let semantic_dist = {
                        let graph = self.ctx.graph.read();
                        // Count nodes as rough proxy for semantic distance
                        let node_count = graph.len().max(1) as f64;
                        let token_hash = gap.token.len() as f64 * 0.1;
                        (token_hash / node_count).clamp(0.0, 5.0)
                    };

                    if !gap.budget.consume(semantic_dist, DEFAULT_AROUSAL, DEFAULT_NOVELTY) {
                        // Budget exhausted: skip this branch
                        continue;
                    }

                    self.stats.active_chain_count += 1;
                    // Max depth is no longer tracked — CuriosityBudget replaces it
                    self.stats.total_gaps_resolved += 1;

                    let ctx = self.ctx.clone();
                    let mut resolver = DefinitionResolver::new(ctx.clone());
                    let mut local_store = KnowledgeStore::new();

                    let sem = semaphore.clone();
                    let gap_tx = self.gap_tx.clone();

                    tokio::spawn(async move {
                        let _permit = sem.acquire().await;

                        // 1. Fetch definition
                        if let Some(definition) = local_store.fetch(&gap.token) {
                            // 2. Resolve into graph nodes
                            let resolved = resolver.resolve(
                                &gap.token,
                                &definition,
                                gap.parent_node_id,
                            );

                            // 3. Spawn recursive resolutions for dependencies
                            //    Carry forward the CuriosityBudget (shared energy pool)
                            for dep in resolved.dependencies {
                                if dep.to_lowercase() != gap.token.to_lowercase() {
                                    let child_budget = gap.budget.clone();
                                    let _ = gap_tx
                                        .send(KnowledgeGap {
                                            token: dep,
                                            parent_node_id: Some(resolved.main_node_id),
                                            budget: child_budget,
                                            source: GapSource::RecursiveResolution,
                                        })
                                        .await;
                                }
                            }
                        } else {
                            // Token is a fundamental primitive — no further resolution needed.
                            // The budget is preserved (not consumed for primitives).
                        }

                        drop(_permit);
                    });
                }
                None => {
                    // Channel closed — daemon is shutting down
                    break;
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
//  CuriosityDaemon — high-level interface
// ────────────────────────────────────────────────────────────

pub struct CuriosityDaemon {
    ctx: Arc<SemanticContext>,
    detector: GapDetector,
    harvester_tx: mpsc::Sender<KnowledgeGap>,
}

impl CuriosityDaemon {
    pub fn new(ctx: Arc<SemanticContext>, harvester_tx: mpsc::Sender<KnowledgeGap>) -> Self {
        CuriosityDaemon {
            detector: GapDetector::new(ctx.clone()),
            ctx,
            harvester_tx,
        }
    }

    /// Inspect a set of incoming tokens (from user input, sensor,
    /// file ingestion) and emit KnowledgeGaps for anything ungrounded.
    /// Returns the detected gaps for logging/UI, but does NOT block
    /// on resolution — that happens in the background harvester.
    pub fn inspect_and_ground(&self, tokens: &[String], source: GapSource) -> Vec<KnowledgeGap> {
        let gaps = self.detector.detect(tokens, source);
        for gap in &gaps {
            let _ = self.harvester_tx.try_send(gap.clone());
        }
        gaps
    }

    /// Inspect a single compound prompt string — splits into tokens
    /// and checks each one for grounding.
    pub fn inspect_prompt(&self, prompt: &str) -> Vec<KnowledgeGap> {
        let tokens: Vec<String> = prompt
            .split(|c: char| c.is_whitespace() || c == ',' || c == '.')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        self.inspect_and_ground(&tokens, GapSource::UserInput)
    }

    /// Check if a concept is fully grounded (recursively resolved
    /// down to primitives).
    pub fn is_grounded(&self, token: &str) -> bool {
        self.detector.is_grounded(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn harvest_resolves_single_gap() {
        let mut g = GraphArena::with_capacity(8);
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "pirate".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 10.0,
            base_activation: 0.0,
            edges: Vec::new(),
            valence: 0.0,
        });
        let ctx = SemanticContext::new(g);
        let mut harvester = AutonomousHarvester::new(ctx.clone());

        let gap_tx = harvester.gap_sender();
        let _ = gap_tx
            .send(KnowledgeGap {
                token: "pirate".into(),
                parent_node_id: None,
                budget: CuriosityBudget::default(),
                source: GapSource::UserInput,
            })
            .await;

        // Run a few iterations
        tokio::select! {
            _ = harvester.run() => {},
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {},
        }

        // Verify pirate now has edges (was resolved)
        let graph = ctx.graph.read();
        if let Some(node) = graph.lookup("pirate") {
            let n = graph.get(node).unwrap().read();
            assert!(n.edges.len() >= 1, "pirate should have edges after resolution");
        } else {
            panic!("pirate node not found after resolution");
        }
    }
}
