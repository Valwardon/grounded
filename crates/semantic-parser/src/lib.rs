use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Pattern → CD frame matchers
//
//  No LLM, no regex ambiguity — purely structural matching.
//  Input is a structured Intent JSON or a sensor event tuple.
//  Output is a deterministic ConceptualFrame.
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParsedInput {
    pub frames: Vec<ConceptualFrame>,
    pub confidence: f64,
}

/// Determines the CD action from a string verb using a hardcoded
/// verb → primitive lookup table. No stemming, no embedding.
fn verb_to_cd_action(verb: &str) -> Option<CDAction> {
    match verb.to_lowercase().as_str() {
        "give" | "transfer" | "send" | "pay" => Some(CDAction::Atrans),
        "take" | "steal" | "receive" | "buy" => Some(CDAction::Atrans),
        "go" | "move" | "walk" | "run" | "come" | "bring" => Some(CDAction::Ptrans),
        "tell" | "show" | "notify" | "inform" => Some(CDAction::Mtrans),
        "see" | "watch" | "read" | "scan" => Some(CDAction::Mtrans),
        "decide" | "conclude" | "plan" | "choose" => Some(CDAction::Mbuild),
        "think" | "consider" | "remember" => Some(CDAction::Mbuild),
        "push" | "pull" | "throw" | "hit" | "propel" => Some(CDAction::Propel),
        "eat" | "drink" | "consume" | "ingest" => Some(CDAction::Ingest),
        "exhale" | "spit" | "exhaust" => Some(CDAction::Expel),
        "wave" | "point" | "lift" | "lower" => Some(CDAction::Move),
        "hold" | "grasp" | "grip" | "grab" | "catch" => Some(CDAction::Grasp),
        "say" | "ask" | "speak" | "announce" | "reply" => Some(CDAction::Speak),
        "listen" | "hear" | "attend" | "focus" | "observe" => Some(CDAction::Attend),
        "open" | "close" | "start" | "stop" | "enable" | "disable" | "launch" | "lock" => {
            Some(CDAction::SystemAction)
        }
        _ => None,
    }
}

/// Parse a JSON intent (from Android intent, voice command, or notification action)
/// into a list of CD frames. Uses exact field matching — no guesswork.
pub fn parse_intent(json: &str) -> ParsedInput {
    let fallback = ParsedInput {
        frames: vec![ConceptualFrame::new(CDAction::Attend)],
        confidence: 0.1,
    };

    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return fallback,
    };

    let action_str = value
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let verb = action_str.split('_').next().unwrap_or(action_str);
    let cd_action = verb_to_cd_action(verb).unwrap_or(CDAction::SystemAction);

    let mut frame = ConceptualFrame::new(cd_action);

    if let Some(obj) = value.get("object").and_then(|v| v.as_str()) {
        // Look up the object in the graph — if not found, create a temp NodeId
        frame.object = Some(NodeId::from_raw(obj.len() as u64));
    }

    // Map Android-specific intents directly
    frame.frame_type = match action_str {
        "open_camera" | "take_photo" | "record_video" => CDType::SystemCommand,
        "send_message" | "make_call" | "open_app" => CDType::SystemCommand,
        "set_volume" | "set_brightness" | "toggle_wifi" | "toggle_bluetooth" => {
            CDType::StateChange
        }
        "get_location" | "get_sensor" | "read_battery" => CDType::SensorEvent,
        _ => CDType::Action,
    };

    ParsedInput {
        frames: vec![frame],
        confidence: if cd_action != CDAction::Attend { 0.9 } else { 0.3 },
    }
}

/// Parse a structured sensor event into a SensorReading CD frame.
pub fn parse_sensor_event(sensor: &str, channel: u8, value: f32) -> ParsedInput {
    let mut frame = ConceptualFrame::new(CDAction::SensorReading);
    frame.frame_type = CDType::SensorEvent;
    frame.instrument = Some(NodeId::from_raw(sensor.len() as u64));
    frame.goal = Some(NodeId::from_raw((channel as u64) + 1));
    frame.time = Some(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    );

    ParsedInput {
        frames: vec![frame],
        confidence: 1.0, // sensor readings are deterministic ground truth
    }
}

// ────────────────────────────────────────────────────────────
//  Realizer: CD frame → structured output
// ────────────────────────────────────────────────────────────

/// Template-based realizer. Maps CD frames to action JSON for Android intents,
/// or to human-readable strings via simple templates.
pub struct Realizer;

impl Realizer {
    /// Realize a CD frame into an Android intent JSON command.
    pub fn realize_to_intent(frame: &ConceptualFrame) -> String {
        let action = match frame.action {
            CDAction::Atrans => "transfer_ownership",
            CDAction::Ptrans => "move_physical",
            CDAction::SystemAction => {
                // Look up the template from the grounded node
                "system_action"
            }
            CDAction::Grasp => "acquire_lock",
            CDAction::Speak => "tts_speak",
            CDAction::Attend => "poll_sensor",
            _ => "noop",
        };

        let mut cmd = serde_json::json!({
            "action": action,
            "confidence": frame.injection_targets(1.0).len(),
        });

        if let Some(o) = frame.object {
            cmd["targetNode"] = serde_json::json!(o.0);
        }
        if let Some(r) = frame.recipient {
            cmd["recipientNode"] = serde_json::json!(r.0);
        }

        serde_json::to_string(&cmd).unwrap_or_else(|_| r#"{"action":"noop"}"#.into())
    }

    /// Realize a CD frame into a display string (for UI or logging).
    pub fn realize_to_text(frame: &ConceptualFrame) -> String {
        let action_str = match frame.action {
            CDAction::Atrans => "transferred",
            CDAction::Ptrans => "moved",
            CDAction::Mtrans => "informed",
            CDAction::Mbuild => "decided",
            CDAction::Propel => "propelled",
            CDAction::Ingest => "ingested",
            CDAction::Expel => "expelled",
            CDAction::Move => "moved",
            CDAction::Grasp => "grasped",
            CDAction::Speak => "spoke",
            CDAction::Attend => "attended",
            CDAction::SystemAction => "executed_system",
            CDAction::SensorReading => "sensed",
        };

        format!(
            "{} [actor={:?} object={:?} recipient={:?}]",
            action_str, frame.actor, frame.object, frame.recipient
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_camera_intent() {
        let result = parse_intent(r#"{"action":"open_camera","object":"camera"}"#);
        assert_eq!(result.frames[0].frame_type, CDType::SystemCommand);
        assert_eq!(result.frames[0].action, CDAction::SystemAction);
        assert!((result.confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn sensor_event_roundtrip() {
        let result = parse_sensor_event("accelerometer", 0, 9.81);
        assert_eq!(result.frames[0].action, CDAction::SensorReading);
        assert!((result.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn verb_table_exhaustive() {
        assert_eq!(verb_to_cd_action("give"), Some(CDAction::Atrans));
        assert_eq!(verb_to_cd_action("open"), Some(CDAction::SystemAction));
        assert_eq!(verb_to_cd_action("listen"), Some(CDAction::Attend));
        assert_eq!(verb_to_cd_action("bogus"), None);
    }
}
