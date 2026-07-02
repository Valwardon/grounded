use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use semantic_graph::prelude::*;
use semantic_parser::{parse_intent, parse_sensor_event};

use crate::activation::*;

// ────────────────────────────────────────────────────────────
//  Background cognitive daemon loop
//
//  Runs on its own OS thread (not on the Tokio runtime).
//  Tick rate: 16ms (≈60Hz), matching the UI frame budget.
//
//  Architecture:
//
//    loop {
//        let start = Instant::now();
//
//        1. Drain external event queue (sensor frames, intents, timers)
//           → inject activation into graph nodes
//
//        2. Run one tick of spreading activation
//           → collect FiredActions
//
//        3. For each FiredAction:
//           a) If Grounding::Action: realize CD frame → JSON → push to output channel
//           b) If Grounding::Sensor: update graph state node
//           c) If Grounding::Stored: persist value
//
//        4. Check for lifecycle signals (shutdown, pause, config reload)
//
//        5. Sleep remaining of 16ms slot
//    }
//
//  No Tokio, no async. Pure deterministic sync loop.
// ────────────────────────────────────────────────────────────

/// Event sources that feed into the cognitive loop.
/// These are pushed from Kotlin via the lock-free channel.
#[derive(Debug, Clone)]
pub enum CognitiveEvent {
    Intent { source: String, json: String, timestamp_ms: u64 },
    SensorReading { sensor: String, channel: u8, value: f32, timestamp_ms: u64 },
    TimerElapsed { timer_id: u64 },
    GraphCommand { json: String },
    Shutdown,
    Pause,
    Resume,
}

/// Output actions produced by the cognitive loop.
/// These flow back to Kotlin for execution (intents, notifications, UI updates).
#[derive(Debug, Clone)]
pub enum CognitiveOutput {
    AndroidIntent { json: String },
    UpdateUi { json: String },
    LogMessage { level: u8, text: String },
}

/// Lock-free SPSC channel for sending events into the cognitive loop.
pub struct EventChannel {
    buffer: [RwLock<Option<CognitiveEvent>>; 128],
    write_seq: std::sync::atomic::AtomicU64,
    read_seq: std::sync::atomic::AtomicU64,
}

impl EventChannel {
    pub fn new() -> Self {
        const INIT: RwLock<Option<CognitiveEvent>> = RwLock::new(None);
        EventChannel {
            buffer: [INIT; 128],
            write_seq: std::sync::atomic::AtomicU64::new(0),
            read_seq: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn send(&self, event: CognitiveEvent) -> bool {
        let seq = self.write_seq.load(Ordering::Relaxed);
        let idx = seq as usize % 128;
        let mut slot = self.buffer[idx].write();
        if slot.is_some() {
            return false;
        }
        *slot = Some(event);
        self.write_seq.store(seq.wrapping_add(1), Ordering::Release);
        true
    }

    pub fn recv(&self) -> Option<CognitiveEvent> {
        let seq = self.read_seq.load(Ordering::Relaxed);
        let idx = seq as usize % 128;
        let mut slot = self.buffer[idx].write();
        let event = slot.take();
        if event.is_some() {
            self.read_seq.store(seq.wrapping_add(1), Ordering::Release);
        }
        event
    }
}

pub struct CognitiveDaemon {
    ctx: Arc<SemanticContext>,
    engine: parking_lot::Mutex<ActivationEngine>,
    event_channel: Arc<EventChannel>,
    output_channel: Arc<RwLock<Vec<CognitiveOutput>>>,
    running: AtomicBool,
    paused: AtomicBool,
    tick_interval: Duration,
}

impl CognitiveDaemon {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        let event_channel = Arc::new(EventChannel::new());
        let output_channel = Arc::new(RwLock::new(Vec::with_capacity(64)));

        CognitiveDaemon {
            engine: parking_lot::Mutex::new(ActivationEngine::new(ctx.clone())),
            ctx,
            event_channel,
            output_channel,
            running: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            tick_interval: Duration::from_millis(16),
        }
    }

    pub fn event_channel(&self) -> Arc<EventChannel> {
        self.event_channel.clone()
    }

    pub fn output_channel(&self) -> Arc<RwLock<Vec<CognitiveOutput>>> {
        self.output_channel.clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn start(self: &Arc<Self>) {
        self.running.store(true, Ordering::Release);
        let daemon = self.clone();
        std::thread::Builder::new()
            .name("cognitive-daemon".into())
            .stack_size(4 * 1024 * 1024)
            .spawn(move || daemon.run())
            .expect("cognitive daemon thread");
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    fn run(self: Arc<Self>) {
        while self.running.load(Ordering::Acquire) {
            let loop_start = Instant::now();

            // ── 1. Drain event channel ──
            while let Some(event) = self.event_channel.recv() {
                match event {
                    CognitiveEvent::Shutdown => {
                        self.running.store(false, Ordering::Release);
                        return;
                    }
                    CognitiveEvent::Pause => {
                        self.paused.store(true, Ordering::Release);
                    }
                    CognitiveEvent::Resume => {
                        self.paused.store(false, Ordering::Release);
                    }
                    CognitiveEvent::Intent { json, .. } => {
                        let parsed = parse_intent(&json);
                        let mut engine = self.engine.lock();
                        for frame in &parsed.frames {
                            engine.inject_frame(frame, BASE_INJECT * parsed.confidence);
                        }
                    }
                    CognitiveEvent::SensorReading { sensor, channel, value, .. } => {
                        let parsed = parse_sensor_event(&sensor, channel, value);
                        let mut engine = self.engine.lock();
                        for frame in &parsed.frames {
                            engine.inject_frame(frame, BASE_INJECT);
                        }
                    }
                    CognitiveEvent::TimerElapsed { timer_id } => {
                        let node_id = NodeId::from_raw(timer_id);
                        self.engine.lock().inject(node_id, BASE_INJECT * 0.5);
                    }
                    CognitiveEvent::GraphCommand { json } => {
                        if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&json) {
                            self.handle_graph_command(&cmd);
                        }
                    }
                }
            }

            // ── 2. Run activation tick ──
            if !self.paused.load(Ordering::Acquire) {
                let fired = {
                    let mut engine = self.engine.lock();
                    engine.tick().to_vec()
                };

                // ── 3. Dispatch fired actions ──
                let mut outputs = self.output_channel.write();
                for action in &fired {
                    match &action.grounding {
                        Grounding::Action { intent_template } => {
                            let intent = serde_json::json!({
                                "action": action.node_label,
                                "activation": action.activation_level,
                                "template": intent_template,
                            });
                            outputs.push(CognitiveOutput::AndroidIntent {
                                json: serde_json::to_string(&intent)
                                    .unwrap_or_else(|_| intent_template.to_string()),
                            });
                        }
                        Grounding::Sensor { sensor_type, channel, .. } => {
                            outputs.push(CognitiveOutput::LogMessage {
                                level: 1,
                                text: format!(
                                    "Sensor threshold: {}[{}] @ {:.3}",
                                    sensor_type, channel, action.activation_level
                                ),
                            });
                        }
                        _ => {
                            outputs.push(CognitiveOutput::LogMessage {
                                level: 1,
                                text: format!(
                                    "Fired: {} ({:.3})",
                                    action.node_label, action.activation_level
                                ),
                            });
                        }
                    }
                }
            }

            // ── 4. Sleep remaining of tick interval ──
            let elapsed = loop_start.elapsed();
            if elapsed < self.tick_interval {
                std::thread::sleep(self.tick_interval - elapsed);
            }
        }
    }

    fn handle_graph_command(&self, cmd: &serde_json::Value) {
        let op = cmd.get("op").and_then(|v| v.as_str()).unwrap_or("");
        let mut graph = self.ctx.graph.write();
        match op {
            "add_node" => {
                if let Some(label) = cmd.get("label").and_then(|v| v.as_str()) {
                    let ntype = match cmd.get("type").and_then(|v| v.as_str()) {
                        Some("entity") => NodeType::Entity,
                        Some("action") => NodeType::Action,
                        Some("sensor") => NodeType::Sensor,
                        Some("state") => NodeType::State,
                        _ => NodeType::Concept,
                    };
                    let node = GroundedNode {
                        id: NodeId::ZERO,
                        label: label.to_string(),
                        node_type: ntype,
                        grounding: Grounding::Abstract,
                        decay: cmd.get("decay").and_then(|v| v.as_f64()).unwrap_or(0.9),
                        threshold: cmd.get("threshold").and_then(|v| v.as_f64()).unwrap_or(f64::MAX),
                        base_activation: cmd.get("base").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        edges: Vec::new(),
                    };
                    graph.insert(node);
                }
            }
            "add_edge" => {
                let src = cmd.get("source").and_then(|v| v.as_u64()).unwrap_or(0);
                let dst = cmd.get("target").and_then(|v| v.as_u64()).unwrap_or(0);
                let rel = match cmd.get("relation").and_then(|v| v.as_str()) {
                    Some("is_a") => Relation::IsA,
                    Some("activates") => Relation::Activates,
                    Some("inhibits") => Relation::Inhibits,
                    Some("requires") => Relation::Requires,
                    Some("implies") => Relation::Implies,
                    Some("grounded_in") => Relation::GroundedIn,
                    _ => Relation::AssociatedWith,
                };
                if let Some(node) = graph.get(NodeId::from_raw(src)) {
                    node.write().edges.push(Edge {
                        relation: rel,
                        target: NodeId::from_raw(dst),
                        weight_override: cmd.get("weight").and_then(|v| v.as_f64()),
                    });
                }
            }
            _ => {}
        }
    }
}
