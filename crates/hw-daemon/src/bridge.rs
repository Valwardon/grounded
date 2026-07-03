use std::sync::OnceLock;

use crate::lifecycle::CognitiveLifecycle;

// ────────────────────────────────────────────────────────────
//  UniFFI-exported bridge surface
//
//  These are the flat C-ABI functions that UniFFI generates
//  Kotlin bindings for. The Kotlin ForegroundService calls:
//
//    CognitiveBridge.init(dataDir)
//    CognitiveBridge.start()
//    CognitiveBridge.feedSensor("accelerometer", 0, 9.81)
//    CognitiveBridge.feedIntent("{\"action\":\"take_photo\"}")
//    CognitiveBridge.drainOutputs()  // called on UI tick
//    CognitiveBridge.stop()
//
//  The bridge is a global singleton — safe because it's guarded
//  by the Kotlin lifecycle (only one ForegroundService instance).
// ────────────────────────────────────────────────────────────

static LIFECYCLE: OnceLock<parking_lot::RwLock<Option<CognitiveLifecycle>>> = OnceLock::new();

fn lifecycle() -> &'static parking_lot::RwLock<Option<CognitiveLifecycle>> {
    LIFECYCLE.get_or_init(|| parking_lot::RwLock::new(None))
}

/// Initialize the cognitive bridge with the app's data directory.
/// Called from ForegroundService.onCreate().
pub fn init(data_dir: &str) {
    let mut guard = lifecycle().write();
    let mut lc = CognitiveLifecycle::new();
    lc.on_create(data_dir);
    *guard = Some(lc);
}

/// Start the cognitive background daemon.
/// Called from ForegroundService.onStartCommand().
pub fn start() {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        lc.start();
    }
}

/// Stop the daemon and persist state.
/// Called from ForegroundService.onDestroy().
pub fn stop() {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        lc.stop();
    }
}

/// Feed a sensor reading into the cognitive loop.
/// Called from Kotlin SensorEventListener.onSensorChanged().
pub fn feed_sensor(sensor_type: &str, channel: u8, value: f32) {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        lc.feed_sensor(sensor_type, channel, value);
    }
}

/// Feed a JSON intent (from notification action, voice command, or UI).
pub fn feed_intent(json: &str) {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        lc.feed_intent(json);
    }
}

/// Drain all accumulated output actions.
/// Called periodically from Kotlin (e.g., on UI tick or via a timer).
/// Returns JSON strings that should be dispatched as Android intents.
pub fn drain_outputs() -> Vec<String> {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        lc.drain_outputs()
    } else {
        Vec::new()
    }
}

/// Check if the cognitive daemon is running.
pub fn is_running() -> bool {
    let guard = lifecycle().read();
    guard
        .as_ref()
        .map(|lc| lc.is_running())
        .unwrap_or(false)
}

/// Add or modify a node in the semantic graph at runtime.
pub fn graph_command(json: &str) {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        if let Some(ref chan) = lc.event_channel() {
            chan.send(cognitive_core::CognitiveEvent::GraphCommand {
                json: json.to_string(),
            });
        }
    }
}

/// Spike a neuromodulator channel (novelty, arousal, or reward) from external source.
/// Amount is clamped to [0.0, 1.0].
pub fn modulate(channel: &str, amount: f64) {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        if let Some(ref chan) = lc.event_channel() {
            chan.send(cognitive_core::CognitiveEvent::Modulate {
                channel: channel.to_string(),
                amount: amount.clamp(0.0, 1.0),
            });
        }
    }
}

/// Request an offline consolidation pass (pruning + chunking).
/// The engine will consolidate when neuromodulator levels are low.
pub fn trigger_consolidation() {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        if let Some(ref chan) = lc.event_channel() {
            chan.send(cognitive_core::CognitiveEvent::Consolidate);
        }
    }
}

/// Return current neuromodulator levels as a JSON string.
pub fn read_modulators() -> String {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        let (n, a, r) = lc.read_modulators();
        return serde_json::json!({ "novelty": n, "arousal": a, "reward": r }).to_string();
    }
    "{}".to_string()
}

/// Called on memory pressure to persist graph state.
pub fn trim_memory() {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        lc.on_trim_memory();
    }
}

/// Keepalive heartbeat — Kotlin calls this every 5s.
/// Returns true if the daemon thread is alive.
pub fn keepalive() -> bool {
    let guard = lifecycle().read();
    guard
        .as_ref()
        .map(|lc| lc.keepalive())
        .unwrap_or(false)
}

/// Number of consecutive missed heartbeats.
pub fn missed_heartbeats() -> u64 {
    let guard = lifecycle().read();
    guard
        .as_ref()
        .map(|lc| lc.missed_heartbeats())
        .unwrap_or(0)
}

/// Inspect a compound prompt for knowledge gaps.
/// Returns the list of tokens that need grounding.
pub fn inspect_prompt(prompt: &str) -> Vec<String> {
    let guard = lifecycle().read();
    guard
        .as_ref()
        .map(|lc| lc.inspect_prompt(prompt))
        .unwrap_or_default()
}

/// Get the current daemon tick count.
pub fn tick_count() -> u64 {
    let guard = lifecycle().read();
    guard
        .as_ref()
        .map(|lc| lc.tick_count())
        .unwrap_or(0)
}

/// Introspect — return everything the self node is connected to.
/// Returns vec of (node_id, label, relation).
pub fn introspect() -> Vec<(u64, String, String)> {
    let guard = lifecycle().read();
    guard
        .as_ref()
        .map(|lc| {
            lc.introspect()
                .into_iter()
                .map(|(id, label, rel)| (id.0, label, format!("{:?}", rel)))
                .collect()
        })
        .unwrap_or_default()
}

/// Link a specific node to the self node for grounding.
pub fn link_to_self(node_id: u64, relation: &str) -> bool {
    let guard = lifecycle().read();
    if let Some(ref lc) = *guard {
        if let Some(ref ctx) = lc.ctx {
            let rel = match relation {
                "is_a" => Relation::IsA,
                "has" => Relation::HasProperty,
                "requires" => Relation::Requires,
                "causes" => Relation::CausedBy,
                "implies" => Relation::Implies,
                "grounded_in" => Relation::GroundedIn,
                "activates" => Relation::Activates,
                "inhibits" => Relation::Inhibits,
                _ => Relation::AssociatedWith,
            };
            return ctx.link_to_self(rel, NodeId::from_raw(node_id));
        }
    }
    false
}
