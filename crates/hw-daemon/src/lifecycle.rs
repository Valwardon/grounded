use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cognitive_core::CognitiveDaemon;
use semantic_graph::prelude::*;

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
            running: AtomicBool::new(false),
            last_tick_ms: AtomicU64::new(0),
            missed_heartbeats: AtomicU64::new(0),
        }
    }

    /// Called once from Kotlin ForegroundService.onCreate().
    /// Initializes the graph, optionally restoring from disk.
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
        self.daemon = Some(Arc::new(CognitiveDaemon::new(ctx)));
    }

    /// Start the cognitive background loop.
    /// Maps to Kotlin's onStartCommand() with START_STICKY.
    pub fn start(&self) {
        if let Some(ref daemon) = self.daemon {
            daemon.start();
            self.running.store(true, Ordering::Release);
        }
    }

    /// Stop the cognitive loop and persist graph state.
    /// Maps to Kotlin's onDestroy().
    pub fn stop(&self) {
        if let Some(ref daemon) = self.daemon {
            daemon.stop();
            self.running.store(false, Ordering::Release);
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
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "sensor_accelerometer".into(),
            node_type: NodeType::Sensor,
            grounding: Grounding::Sensor {
                sensor_type: "accelerometer".into(),
                channel: 0,
                norm: SensorNorm::Clamp { min: 0.0, max: 1.0 },
            },
            decay: 0.85,
            threshold: 2.5,
            base_activation: 0.0,
            edges: vec![Edge {
                relation: Relation::Activates,
                target: NodeId::from_raw(3),
                weight_override: None,
            }],
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "sensor_proximity".into(),
            node_type: NodeType::Sensor,
            grounding: Grounding::Sensor {
                sensor_type: "proximity".into(),
                channel: 0,
                norm: SensorNorm::Linear { scale: -0.1, offset: 1.0 },
            },
            decay: 0.9,
            threshold: 1.8,
            base_activation: 0.0,
            edges: vec![Edge {
                relation: Relation::Activates,
                target: NodeId::from_raw(4),
                weight_override: None,
            }],
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "sensor_light".into(),
            node_type: NodeType::Sensor,
            grounding: Grounding::Sensor {
                sensor_type: "light".into(),
                channel: 0,
                norm: SensorNorm::Linear { scale: 0.001, offset: 0.0 },
            },
            decay: 0.95,
            threshold: 0.8,
            base_activation: 0.1,
            edges: vec![Edge {
                relation: Relation::Activates,
                target: NodeId::from_raw(5),
                weight_override: None,
            }],
        });

        // ── Abstract concepts ──
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "concept_movement".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 1.2,
            base_activation: 0.0,
            edges: vec![Edge {
                relation: Relation::Implies,
                target: NodeId::from_raw(6),
                weight_override: None,
            }],
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "concept_proximity".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 1.5,
            base_activation: 0.0,
            edges: vec![Edge {
                relation: Relation::Implies,
                target: NodeId::from_raw(6),
                weight_override: Some(0.5),
            }],
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "concept_darkness".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.92,
            threshold: 1.0,
            base_activation: 0.0,
            edges: vec![
                Edge {
                    relation: Relation::Activates,
                    target: NodeId::from_raw(7),
                    weight_override: None,
                },
                Edge {
                    relation: Relation::Activates,
                    target: NodeId::from_raw(8),
                    weight_override: None,
                },
            ],
        });

        // ── Action nodes (grounded to Android intents) ──
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "lock_screen".into(),
            node_type: NodeType::Action,
            grounding: Grounding::Action {
                intent_template: r#"{"action":"lockScreen","params":{}}"#.into(),
            },
            decay: 0.5,
            threshold: 1.0,
            base_activation: 0.0,
            edges: Vec::new(),
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "toggle_flashlight".into(),
            node_type: NodeType::Action,
            grounding: Grounding::Action {
                intent_template: r#"{"action":"toggleFlashlight","params":{"on":true}}"#.into(),
            },
            decay: 0.3,
            threshold: 1.0,
            base_activation: 0.0,
            edges: Vec::new(),
        });

        // ── State nodes ──
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "state_night_mode".into(),
            node_type: NodeType::State,
            grounding: Grounding::Stored {
                keyspace: "system".into(),
                key: "night_mode".into(),
            },
            decay: 0.99,
            threshold: f64::MAX,
            base_activation: 0.0,
            edges: vec![
                Edge {
                    relation: Relation::Inhibits,
                    target: NodeId::from_raw(8),
                    weight_override: Some(-0.3),
                },
            ],
        });

        g
    }
}
