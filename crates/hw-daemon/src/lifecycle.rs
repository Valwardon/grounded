use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cognitive_core::CognitiveDaemon;
use semantic_graph::prelude::*;
use crate::render_bridge::{RenderBridge, NullRenderBackend};

// ────────────────────────────────────────────────────────────
//  Foreground Service Lifecycle Manager
//
//  This maps the Android ForegroundService lifecycle onto the
//  Rust cognitive daemon. The Kotlin side calls:
//
//    onCreate()  →  CognitiveLifecycle::on_create(data_dir)
//    onStart()   →  lifecycle.start()
//    onDestroy() →  lifecycle.stop()
//
//  Graph state is persisted to disk on pause/destroy and
//  reloaded on create.
// ────────────────────────────────────────────────────────────

pub struct CognitiveLifecycle {
    daemon: Option<Arc<CognitiveDaemon>>,
    ctx: Option<Arc<SemanticContext>>,
    graph_path: Option<String>,
    render_bridge: Option<parking_lot::Mutex<RenderBridge>>,
    running: AtomicBool,
    /// Last tick timestamp (ms since epoch) — for keepalive heartbeat
    last_tick_ms: AtomicU64,
    /// Number of keepalive checks that detected a dead runtime
    missed_heartbeats: AtomicU64,
}

impl CognitiveLifecycle {
    pub fn new() -> Self {
        CognitiveLifecycle {
            daemon: None,
            ctx: None,
            graph_path: None,
            render_bridge: None,
            running: AtomicBool::new(false),
            last_tick_ms: AtomicU64::new(0),
            missed_heartbeats: AtomicU64::new(0),
        }
    }

    /// Called once from Kotlin ForegroundService.onCreate().
    /// Initializes the graph, optionally restoring from disk.
    /// Creates the RenderBridge + shared VisualEffectorBuffer.
    pub fn on_create(&mut self, data_dir: &str) {
        let graph_path = format!("{}/semantic_graph.bin", data_dir);
        self.graph_path = Some(graph_path.clone());

        // Attempt to restore persisted graph
        let graph = match std::fs::read(&graph_path) {
            Ok(data) => GraphArena::deserialize(&data),
            Err(_) => Self::build_default_graph(),
        };

        let ctx = SemanticContext::new(graph);
        self.ctx = Some(ctx.clone());

        // Create visual primitive ring buffer (SPSC, shared)
        let visual_ring = Arc::new(VisualPrimitiveRingBuffer::new());

        // Create render bridge with shared effector buffer + visual ring
        let (mut bridge, effector_buffer) = RenderBridge::new(Some(visual_ring.clone()));

        // Set penalize callback: validate_ast() errors → ContractMismatch → -0.05 valence
        let ctx_for_penalty = ctx.clone();
        bridge.set_penalize_fn(Box::new(move |_node_id: &str, _error: &str| {
            // Penalize all nodes with ContractMismatch → -0.05 valence deduction
            let graph = ctx_for_penalty.graph.write();
            for i in 1..graph.len() {
                let id = NodeId::from_raw(i as u64);
                graph.update_valence(id, -0.05, 0.1);
            }
        }));
        bridge.set_backend(Box::new(NullRenderBackend));
        self.render_bridge = Some(parking_lot::Mutex::new(bridge));

        // Pass the shared buffers to CognitiveDaemon
        self.daemon = Some(Arc::new(CognitiveDaemon::new(
            ctx, Some(effector_buffer), Some(visual_ring),
        )));
    }

    /// Start the cognitive background loop + render bridge.
    /// Maps to Kotlin's onStartCommand() with START_STICKY.
    pub fn start(&self) {
        if let Some(ref daemon) = self.daemon {
            daemon.start();
            self.running.store(true, Ordering::Release);
        }
        if let Some(ref bridge) = self.render_bridge {
            bridge.lock().start();
        }
    }

    /// Stop the cognitive loop + render bridge and persist graph state.
    /// Maps to Kotlin's onDestroy().
    pub fn stop(&self) {
        if let Some(ref daemon) = self.daemon {
            daemon.stop();
            self.running.store(false, Ordering::Release);
        }
        if let Some(ref bridge) = self.render_bridge {
            bridge.lock().stop();
        }
        self.persist_graph();
    }

    /// Called when the OS signals memory pressure.
    /// Persists graph to disk but keeps running.
    pub fn on_trim_memory(&self) {
        self.persist_graph();
    }

    /// Keepalive heartbeat. Kotlin calls this every 5 seconds.
    /// Returns true if the daemon thread is alive.
    ///
    /// Checks two signals:
    ///   1. The running flag is true
    ///   2. The context tick counter has advanced since last check
    pub fn keepalive(&self) -> bool {
        if !self.running.load(Ordering::Acquire) {
            self.missed_heartbeats.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let tick_now = self
            .ctx
            .as_ref()
            .map(|ctx| ctx.tick.load(Ordering::Acquire))
            .unwrap_or(0);

        let last = self.last_tick_ms.load(Ordering::Relaxed);

        if last == 0 || tick_now > last {
            // Tick has advanced since last check — daemon is alive
            self.last_tick_ms.store(tick_now, Ordering::Release);
            self.missed_heartbeats.store(0, Ordering::Relaxed);
            true
        } else {
            // Tick has NOT advanced — daemon thread may be stuck or dead
            let missed = self.missed_heartbeats.fetch_add(1, Ordering::Relaxed) + 1;
            if missed >= 3 {
                // 3 consecutive missed heartbeats ≈ 15 seconds dead
                false
            } else {
                true // still within tolerance
            }
        }
    }

    /// Read current neuromodulator levels (novelty, arousal, reward).
    pub fn read_modulators(&self) -> (f64, f64, f64) {
        self.daemon
            .as_ref()
            .map(|d| d.read_modulators())
            .unwrap_or((0.0, 0.0, 0.0))
    }

    /// Generate an opinion about a topic based on accumulated experience.
    pub fn get_opinion(&self, topic: &str) -> String {
        self.daemon
            .as_ref()
            .map(|d| d.get_opinion(topic))
            .unwrap_or_else(|| format!("I don't know what '{}' is.", topic))
    }

    /// Return the system's current interests (high-valence concepts).
    pub fn get_interests(&self, count: usize) -> Vec<String> {
        self.daemon
            .as_ref()
            .map(|d| d.get_interests(count))
            .unwrap_or_default()
    }

    /// Get a short description of the system's current mood.
    pub fn get_mood(&self) -> String {
        self.daemon
            .as_ref()
            .map(|d| d.get_mood())
            .unwrap_or_else(|| "Not running.".to_string())
    }

    /// Number of consecutive missed heartbeats.
    pub fn missed_heartbeats(&self) -> u64 {
        self.missed_heartbeats.load(Ordering::Relaxed)
    }

    /// Introspect — return everything the self node is connected to.
    pub fn introspect(&self) -> Vec<(NodeId, String, Relation)> {
        self.ctx
            .as_ref()
            .map(|ctx| ctx.introspect())
            .unwrap_or_default()
    }

    /// Get the current tick counter from the semantic context.
    pub fn tick_count(&self) -> u64 {
        self.ctx
            .as_ref()
            .map(|ctx| ctx.tick.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    /// Inspect a compound prompt for knowledge gaps.
    /// Feeds any detected gaps into the curiosity harvester.
    pub fn inspect_prompt(&self, prompt: &str) -> Vec<String> {
        // Gap detection is routed through the curiosity daemon
        // via the event channel. For now, return a placeholder.
        if let Some(ref chan) = self.event_channel() {
            chan.send(cognitive_core::CognitiveEvent::Intent {
                source: "prompt".to_string(),
                json: format!(r#"{{"action":"analyze","prompt":"{}"}}"#, prompt),
                timestamp_ms: self.now_ms(),
            });
        }
        // Tokenize and return as gap candidates
        prompt
            .split(|c: char| c.is_whitespace() || c == ',' || c == '.')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Access the event channel for injecting sensor data and intents.
    pub fn event_channel(&self) -> Option<Arc<cognitive_core::EventChannel>> {
        self.daemon.as_ref().map(|d| d.event_channel())
    }

    /// Access the output channel for draining realized actions.
    pub fn output_channel(
        &self,
    ) -> Option<Arc<parking_lot::RwLock<Vec<cognitive_core::CognitiveOutput>>>> {
        self.daemon.as_ref().map(|d| d.output_channel())
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Send a sensor reading into the cognitive loop.
    pub fn feed_sensor(&self, sensor: &str, channel: u8, value: f32) {
        if let Some(ref chan) = self.event_channel() {
            chan.send(cognitive_core::CognitiveEvent::SensorReading {
                sensor: sensor.to_string(),
                channel,
                value,
                timestamp_ms: self.now_ms(),
            });
        }
    }

    /// Send a JSON intent into the cognitive loop.
    pub fn feed_intent(&self, json: &str) {
        if let Some(ref chan) = self.event_channel() {
            chan.send(cognitive_core::CognitiveEvent::Intent {
                source: "android".to_string(),
                json: json.to_string(),
                timestamp_ms: self.now_ms(),
            });
        }
    }

    /// Drain all accumulated output actions and return them as JSON.
    pub fn drain_outputs(&self) -> Vec<String> {
        let mut json_outputs = Vec::new();
        if let Some(ref output) = self.output_channel() {
            let mut outputs = output.write();
            for o in outputs.drain(..) {
                    match o {
                        cognitive_core::CognitiveOutput::AndroidIntent { json } => {
                            json_outputs.push(json);
                        }
                        cognitive_core::CognitiveOutput::UpdateUi { json } => {
                            json_outputs.push(json);
                        }
                        cognitive_core::CognitiveOutput::LogMessage { level, text } => {
                            let prefix = match level {
                                0 => "INFO",
                                1 => "WARN",
                                _ => "ERROR",
                            };
                            eprintln!("[COG][{}] {}", prefix, text);
                        }
                    }
            }
        }
        json_outputs
    }

    /// Persist the graph to disk for service restart survival.
    fn persist_graph(&self) {
        if let (Some(ref ctx), Some(ref path)) = (self.ctx, &self.graph_path) {
            let data = ctx.graph.read().serialize();
            let _ = std::fs::write(path, &data);
        }
    }

    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Build the default semantic graph with grounded nodes for common
    /// Android intents, sensors, and state concepts.
    fn build_default_graph() -> GraphArena {
        let mut g = GraphArena::with_capacity(256);

        // ── Sensor nodes ──
        fn sensor_node(label: &str, sensor_type: &str, channel: u8, norm: SensorNorm, decay: f64, threshold: f64, target: u64) -> GroundedNode {
            GroundedNode {
                id: NodeId::ZERO,
                label: label.into(),
                node_type: NodeType::Sensor,
                grounding: Grounding::Sensor { sensor_type: sensor_type.into(), channel, norm },
                decay, threshold, base_activation: 0.0, edges: vec![Edge::new(Relation::Activates, NodeId::from_raw(target))],
                valence: 0.0,
            }
        }

        fn concept_node(label: &str, decay: f64, threshold: f64, edges: Vec<Edge>) -> GroundedNode {
            GroundedNode {
                id: NodeId::ZERO, label: label.into(), node_type: NodeType::Concept,
                grounding: Grounding::Abstract, decay, threshold, base_activation: 0.0, edges,
                valence: 0.0,
            }
        }

        fn action_node(label: &str, intent_template: &str, decay: f64, threshold: f64) -> GroundedNode {
            GroundedNode {
                id: NodeId::ZERO, label: label.into(), node_type: NodeType::Action,
                grounding: Grounding::Action { intent_template: intent_template.into() },
                decay, threshold, base_activation: 0.0, edges: Vec::new(),
                valence: 0.0,
            }
        }

        fn visual_primitive_node(label: &str, primitive_type: VisualPrimitiveType, threshold: f64, edges: Vec<Edge>) -> GroundedNode {
            GroundedNode {
                id: NodeId::ZERO, label: label.into(), node_type: NodeType::VisualPrimitive,
                grounding: Grounding::VisualPrimitive { primitive_type },
                decay: 0.95, threshold, base_activation: 0.0, edges,
                valence: 0.0,
            }
        }

        fn state_node(label: &str, keyspace: &str, key: &str, decay: f64, edges: Vec<Edge>) -> GroundedNode {
            GroundedNode {
                id: NodeId::ZERO, label: label.into(), node_type: NodeType::State,
                grounding: Grounding::Stored { keyspace: keyspace.into(), key: key.into() },
                decay, threshold: f64::MAX, base_activation: 0.0, edges,
                valence: 0.0,
            }
        }

        // ── Standard sensor/concept/action/state nodes ──
        g.insert(sensor_node("sensor_accelerometer", "accelerometer", 0,
            SensorNorm::Clamp { min: 0.0, max: 1.0 }, 0.85, 2.5, 3));
        g.insert(sensor_node("sensor_proximity", "proximity", 0,
            SensorNorm::Linear { scale: -0.1, offset: 1.0 }, 0.9, 1.8, 4));
        g.insert(sensor_node("sensor_light", "light", 0,
            SensorNorm::Linear { scale: 0.001, offset: 0.0 }, 0.95, 0.8, 5));
        g.insert(concept_node("concept_movement", 0.9, 1.2, vec![
            Edge::new(Relation::Implies, NodeId::from_raw(6)),
        ]));
        g.insert(concept_node("concept_proximity", 0.9, 1.5, vec![
            Edge::with_weight(Relation::Implies, NodeId::from_raw(6), 0.5),
        ]));
        g.insert(concept_node("concept_darkness", 0.92, 1.0, vec![
            Edge::new(Relation::Activates, NodeId::from_raw(7)),
            Edge::new(Relation::Activates, NodeId::from_raw(8)),
        ]));
        g.insert(action_node("lock_screen", r#"{"action":"lockScreen","params":{}}"#, 0.5, 1.0));
        g.insert(action_node("toggle_flashlight", r#"{"action":"toggleFlashlight","params":{"on":true}}"#, 0.3, 1.0));
        g.insert(state_node("state_night_mode", "system", "night_mode", 0.99, vec![
            Edge::with_weight(Relation::Inhibits, NodeId::from_raw(8), -0.3),
        ]));

        // ── Visual primitive nodes (fixed canonical indices) ──
        // These are inserted at their well-known positions so that
        // SensorMapper injection targets always resolve correctly.
        g.insert_at(VISUAL_SPATIAL_SCALE, visual_primitive_node(
            "visual_spatial_scale", VisualPrimitiveType::SpatialScale, 0.5, Vec::new()));
        g.insert_at(VISUAL_ROTATION_X, visual_primitive_node(
            "visual_rotation_x", VisualPrimitiveType::RotationX, 0.5, Vec::new()));
        g.insert_at(VISUAL_ROTATION_Y, visual_primitive_node(
            "visual_rotation_y", VisualPrimitiveType::RotationY, 0.5, Vec::new()));
        g.insert_at(VISUAL_ROTATION_Z, visual_primitive_node(
            "visual_rotation_z", VisualPrimitiveType::RotationZ, 0.5, Vec::new()));
        g.insert_at(VISUAL_COLOR_CHROMA, visual_primitive_node(
            "visual_color_chroma", VisualPrimitiveType::ColorChroma, 0.5, Vec::new()));
        g.insert_at(VISUAL_TOPOLOGY_WIREFRAME, visual_primitive_node(
            "visual_topology_wireframe", VisualPrimitiveType::TopologyWireframe, 0.5, Vec::new()));

        g
    }
}
