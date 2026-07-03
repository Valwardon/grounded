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
//  Architecture (extended with neuromodulation + STDP + prediction):
//
//    loop {
//        1. Drain external events → inject activation + handle modulation
//        2. Run one 4-phase tick (decay/inject/spread/fire + STDP)
//        3. Process prediction errors → spike novelty
//        4. Dispatch fired actions → output channel
//        5. Optional: trigger consolidation if idle
//        6. Sleep remaining of 16ms slot
//    }
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum CognitiveEvent {
    Intent { source: String, json: String, timestamp_ms: u64 },
    SensorReading { sensor: String, channel: u8, value: f32, timestamp_ms: u64 },
    TimerElapsed { timer_id: u64 },
    GraphCommand { json: String },
    /// Spike a neuromodulator channel from external source (e.g., curiosity harvester)
    Modulate { channel: String, amount: f64 },
    /// Trigger offline consolidation pass
    Consolidate,
    Shutdown,
    Pause,
    Resume,
}

#[derive(Debug, Clone)]
pub enum CognitiveOutput {
    AndroidIntent { json: String },
    UpdateUi { json: String },
    LogMessage { level: u8, text: String },
}

/// Lock-free SPSC channel for events.
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

    /// Ticks since last consolidation pass
    consolidation_counter: u64,

    /// Previous sensor readings for delta detection (arousal spike)
    prev_sensor_values: parking_lot::Mutex<Vec<(String, u8, f32)>>,
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
            consolidation_counter: 0,
            prev_sensor_values: parking_lot::Mutex::new(Vec::with_capacity(8)),
        }
    }

    /// Read current neuromodulator levels.
    pub fn read_modulators(&self) -> (f64, f64, f64) {
        self.engine.lock().read_modulators()
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
                self.handle_event(event);
            }

            // ── 2. Run 4-phase activation tick ──
            let mut prediction_errors: Vec<PredictionError> = Vec::new();
            if !self.paused.load(Ordering::Acquire) {
                let mut engine = self.engine.lock();

                let fired = engine.tick().to_vec();
                prediction_errors = engine.prediction_errors.clone();

                // ── 3. Process prediction errors → novelty spikes ──
                //     Low prediction error over several consecutive ticks → reward signal
                for err in &prediction_errors {
                    let mut outputs = self.output_channel.write();
                    outputs.push(CognitiveOutput::LogMessage {
                        level: 2,
                        text: format!(
                            "Prediction error: {} (expected {:.3}, got {:.3}, mag={:.3})",
                            err.node_id.0, err.expected, err.actual, err.error_magnitude
                        ),
                    });
                }

                // Reward: spike if no prediction errors for 5+ consecutive ticks
                if prediction_errors.is_empty() {
                    let tick = self.ctx.tick.load(Ordering::Relaxed);
                    if tick > 0 && tick % 100 == 0 {
                        engine.spike_reward(0.05); // small tonic reward for stable predictions
                    }
                }

                // ── 4. Anchor fired actions to self ──
                for action in &fired {
                    self.ctx.link_to_self(Relation::CausedBy, action.node_id);
                }

                // ── Dispatch fired actions ──
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

            // ── 5. Consolidation check (every ~1000 ticks = ~16s) ──
            self.consolidation_counter += 1;
            if self.consolidation_counter >= 1000 {
                self.consolidation_counter = 0;
                let mut engine = self.engine.lock();
                let (n, a, r) = engine.read_modulators();
                if n < 0.1 && a < 0.1 {
                    // Low neuromodulator activity = "idle" → safe to consolidate
                    let pruned = self.run_consolidation_pass();
                    if pruned > 0 {
                        let mut outputs = self.output_channel.write();
                        outputs.push(CognitiveOutput::LogMessage {
                            level: 1,
                            text: format!("Consolidation: pruned {} edges", pruned),
                        });
                    }
                }
            }

            // ── 6. Sleep remaining of tick interval ──
            let elapsed = loop_start.elapsed();
            if elapsed < self.tick_interval {
                std::thread::sleep(self.tick_interval - elapsed);
            }
        }
    }

    // ── Event handling ─────────────────────────────────

    fn handle_event(&self, event: CognitiveEvent) {
        match event {
            CognitiveEvent::Shutdown => {
                self.running.store(false, Ordering::Release);
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
                if let Some(obj) = parsed.frames.first().and_then(|f| f.object) {
                    self.ctx.link_to_self(Relation::HasProperty, obj);
                }
            }
            CognitiveEvent::SensorReading { sensor, channel, value, .. } => {
                // ── Arousal spike on rapid sensor delta ──
                let prev_value = {
                    let sensors = self.prev_sensor_values.lock();
                    sensors.iter()
                        .find(|(s, c, _)| s == &sensor && *c == channel)
                        .map(|(_, _, v)| *v)
                };
                if let Some(prev) = prev_value {
                    let delta = (value - prev).abs();
                    if delta > 0.5 {
                        self.engine.lock().spike_arousal((delta * 0.1).clamp(0.0, 0.5));
                    }
                }
                // Update stored sensor value (retain all except this one, then push new)
                let mut sensors = self.prev_sensor_values.lock();
                sensors.retain(|(s, c, _)| !(s == &sensor && *c == channel));
                sensors.push((sensor.clone(), channel, value));

                let parsed = parse_sensor_event(&sensor, channel, value);
                let mut engine = self.engine.lock();
                for frame in &parsed.frames {
                    engine.inject_frame(frame, BASE_INJECT);
                }
                if let Some(inst) = parsed.frames.first().and_then(|f| f.instrument) {
                    self.ctx.link_to_self(Relation::GroundedIn, inst);
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
            CognitiveEvent::Modulate { channel, amount } => {
                let mut engine = self.engine.lock();
                match channel.as_str() {
                    "novelty" => engine.spike_novelty(amount),
                    "arousal" => engine.spike_arousal(amount),
                    "reward" => engine.spike_reward(amount),
                    _ => {}
                }
            }
            CognitiveEvent::Consolidate => {
                let pruned = self.run_consolidation_pass();
                if pruned > 0 {
                    let mut outputs = self.output_channel.write();
                    outputs.push(CognitiveOutput::LogMessage {
                        level: 1,
                        text: format!("Consolidation: pruned {} edges", pruned),
                    });
                }
            }
        }
    }

    // ── Consolidation pass ─────────────────────────────

    /// Run one pass of offline consolidation (pruning dead edges).
    /// Returns number of edges removed.
    fn run_consolidation_pass(&self) -> usize {
        let mut graph = self.ctx.graph.write();
        let before = graph.len();
        graph.garbage_collect_edges();
        before - graph.len()
        // Future: linear chain compression, dead-end node removal
    }

    // ── Graph commands ─────────────────────────────────

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
                    let override_w = cmd.get("weight").and_then(|v| v.as_f64());
                    let mut edge = Edge::new(rel, NodeId::from_raw(dst));
                    if let Some(w) = override_w {
                        edge.weight_override = Some(w);
                        edge.dynamic_weight = w;
                    }
                    node.write().edges.push(edge);
                }
            }
            _ => {}
        }
    }
}
