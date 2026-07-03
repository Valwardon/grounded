// ────────────────────────────────────────────────────────────
//  UniFFI-exported surface
//
//  This is the only crate that UniFFI sees. All functions are
//  flat C-ABI wrappers around the hw-daemon bridge.
//
//  Generated Kotlin will have:
//
//    object SemanticEngine {
//        fun init(dataDir: String)
//        fun start()
//        fun stop()
//        fun feedSensor(sensorType: String, channel: Byte, value: Float)
//        fun feedIntent(json: String)
//        fun drainOutputs(): List<String>
//        fun isRunning(): Boolean
//        fun graphCommand(json: String)
//        fun trimMemory()
//    }
// ────────────────────────────────────────────────────────────

uniffi::setup_scaffolding!();

#[uniffi::export]
pub fn semantic_engine_init(data_dir: String) {
    hw_daemon::init(&data_dir);
}

#[uniffi::export]
pub fn semantic_engine_start() {
    hw_daemon::start();
}

#[uniffi::export]
pub fn semantic_engine_stop() {
    hw_daemon::stop();
}

#[uniffi::export]
pub fn semantic_engine_feed_sensor(sensor_type: String, channel: u8, value: f32) {
    hw_daemon::feed_sensor(&sensor_type, channel, value);
}

#[uniffi::export]
pub fn semantic_engine_feed_intent(json: String) {
    hw_daemon::feed_intent(&json);
}

#[uniffi::export]
pub fn semantic_engine_drain_outputs() -> Vec<String> {
    hw_daemon::drain_outputs()
}

#[uniffi::export]
pub fn semantic_engine_is_running() -> bool {
    hw_daemon::is_running()
}

#[uniffi::export]
pub fn semantic_engine_graph_command(json: String) {
    hw_daemon::graph_command(&json);
}

#[uniffi::export]
pub fn semantic_engine_trim_memory() {
    hw_daemon::trim_memory();
}

#[uniffi::export]
pub fn semantic_engine_keepalive() -> bool {
    hw_daemon::keepalive()
}

#[uniffi::export]
pub fn semantic_engine_missed_heartbeats() -> u64 {
    hw_daemon::missed_heartbeats()
}

#[uniffi::export]
pub fn semantic_engine_inspect_prompt(prompt: String) -> Vec<String> {
    hw_daemon::inspect_prompt(&prompt)
}

#[uniffi::export]
pub fn semantic_engine_tick_count() -> u64 {
    hw_daemon::tick_count()
}

#[uniffi::export]
pub fn semantic_engine_modulate(channel: String, amount: f64) {
    hw_daemon::modulate(&channel, amount);
}

#[uniffi::export]
pub fn semantic_engine_trigger_consolidation() {
    hw_daemon::trigger_consolidation();
}

#[uniffi::export]
pub fn semantic_engine_read_modulators() -> String {
    hw_daemon::read_modulators()
}

#[uniffi::export]
pub fn semantic_engine_get_opinion(topic: String) -> String {
    hw_daemon::get_opinion(&topic)
}

#[uniffi::export]
pub fn semantic_engine_get_interests(count: usize) -> Vec<String> {
    hw_daemon::get_interests(count)
}

#[uniffi::export]
pub fn semantic_engine_get_mood() -> String {
    hw_daemon::get_mood()
}
