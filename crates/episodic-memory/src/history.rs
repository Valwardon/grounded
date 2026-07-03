// ────────────────────────────────────────────────────────────
//  EpisodicHistory — the main recorder implementing
//  cognitive_core::EpisodicRecorder.
//
//  Records events into a lock-free ring buffer during the tick
//  loop (hot path) and consolidates them into the semantic graph
//  during idle cycles.
// ────────────────────────────────────────────────────────────

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use cognitive_core::{EpisodicEvent, EpisodicRecorder};
use semantic_graph::prelude::*;

use crate::consolidation::consolidate_episodes;
use crate::record::*;
use crate::ring::EpisodicRingBuffer;

pub struct EpisodicHistory {
    buffer: EpisodicRingBuffer,
    ctx: Arc<SemanticContext>,
    last_summary: parking_lot::Mutex<String>,
}

impl EpisodicHistory {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        EpisodicHistory {
            buffer: EpisodicRingBuffer::new(),
            ctx,
            last_summary: parking_lot::Mutex::new(String::new()),
        }
    }

    /// Access the ring buffer for debugging/inspection.
    pub fn buffer(&self) -> &EpisodicRingBuffer {
        &self.buffer
    }

    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn pack_record(
        &self,
        tick: u64,
        node_a: u64,
        node_b: u64,
        payload: u64,
        event_type: u8,
        novelty: f32,
        arousal: f32,
        reward: f32,
    ) -> RawEpisodicRecord {
        RawEpisodicRecord {
            tick,
            timestamp_ms: self.now_ms(),
            node_id_a: node_a,
            node_id_b: node_b,
            payload,
            meta: pack_meta(event_type, novelty, arousal, reward, 0),
            reserved_a: 0,
            reserved_b: 0,
        }
    }
}

impl EpisodicRecorder for EpisodicHistory {
    fn record(&self, event: EpisodicEvent) {
        let tick = self.ctx.tick.load(std::sync::atomic::Ordering::Relaxed);
        let rec = match event {
            EpisodicEvent::NodeFired { node_id, activation, novelty, arousal, reward } => {
                self.pack_record(
                    tick, node_id, 0,
                    f64::to_bits(activation as f64),
                    EventType::NodeFired as u8,
                    novelty, arousal, reward,
                )
            }
            EpisodicEvent::PredictionError { node_id, error_magnitude, novelty, arousal, reward } => {
                self.pack_record(
                    tick, node_id, 0,
                    f64::to_bits(error_magnitude as f64),
                    EventType::PredictionError as u8,
                    novelty, arousal, reward,
                )
            }
            EpisodicEvent::StructuralFault { fault_type, novelty, arousal, reward } => {
                self.pack_record(
                    tick, 0, 0,
                    fault_type as u64,
                    EventType::StructuralFault as u8,
                    novelty, arousal, reward,
                )
            }
            EpisodicEvent::SensorReading { sensor_hash, value, novelty, arousal, reward } => {
                self.pack_record(
                    tick, sensor_hash, 0,
                    f64::to_bits(value as f64),
                    EventType::SensorReading as u8,
                    novelty, arousal, reward,
                )
            }
            EpisodicEvent::IntentProcessed { intent_hash, novelty, arousal, reward } => {
                self.pack_record(
                    tick, intent_hash, 0, 0,
                    EventType::IntentProcessed as u8,
                    novelty, arousal, reward,
                )
            }
        };

        self.buffer.push(&rec);
    }

    fn consolidate(&self) {
        let summary = consolidate_episodes(&self.ctx, &self.buffer);
        if !summary.is_empty() {
            *self.last_summary.lock() = summary;
        }
    }

    fn last_summary(&self) -> String {
        self.last_summary.lock().clone()
    }
}
