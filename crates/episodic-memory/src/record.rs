// ────────────────────────────────────────────────────────────
//  Episodic Record — a single lived event in the timeline
//
//  Each record is 64 bytes (one cache line), stored in a
//  lock-free SPSC ring buffer. Written during the cognitive
//  tick (Phase 6 / event handling), read during idle consolidation.
// ────────────────────────────────────────────────────────────

use core::sync::atomic::AtomicU64;

/// Number of records in the ring buffer.
pub const EPISODIC_RING_SIZE: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct RawEpisodicRecord {
    pub tick: u64,
    pub timestamp_ms: u64,
    pub node_id_a: u64,
    pub node_id_b: u64,
    pub payload: u64,
    /// Bit-packed: [novelty:16][arousal:16][reward:16][event_type:8][involved_count:8]
    pub meta: u64,
    pub reserved_a: u64,
    pub reserved_b: u64,
}

impl RawEpisodicRecord {
    pub const ZERO: RawEpisodicRecord = RawEpisodicRecord {
        tick: 0, timestamp_ms: 0, node_id_a: 0, node_id_b: 0,
        payload: 0, meta: 0, reserved_a: 0, reserved_b: 0,
    };

    pub fn event_type(&self) -> u8 {
        ((self.meta >> 24) & 0xff) as u8
    }

    pub fn involved_count(&self) -> u8 {
        (self.meta & 0xff) as u8
    }

    pub fn novelty(&self) -> f32 {
        f32::from_bits(((self.meta >> 48) as u32) << 16)
    }

    pub fn arousal(&self) -> f32 {
        f32::from_bits(((self.meta >> 32) as u32) << 16)
    }

    pub fn reward(&self) -> f32 {
        f32::from_bits((self.meta >> 48) as u32)
    }
}

/// Atomic version of RawEpisodicRecord — 8 AtomicU64 = 64 bytes, one cache line.
#[repr(C)]
pub struct AtomicRecord {
    pub tick: AtomicU64,
    pub timestamp_ms: AtomicU64,
    pub node_id_a: AtomicU64,
    pub node_id_b: AtomicU64,
    pub payload: AtomicU64,
    pub meta: AtomicU64,
    pub reserved_a: AtomicU64,
    pub reserved_b: AtomicU64,
}

impl AtomicRecord {
    pub fn store(&self, rec: &RawEpisodicRecord, order: core::sync::atomic::Ordering) {
        use core::sync::atomic::Ordering::*;
        self.tick.store(rec.tick, Relaxed);
        self.timestamp_ms.store(rec.timestamp_ms, Relaxed);
        self.node_id_a.store(rec.node_id_a, Relaxed);
        self.node_id_b.store(rec.node_id_b, Relaxed);
        self.payload.store(rec.payload, Relaxed);
        self.meta.store(rec.meta, order);
        self.reserved_a.store(rec.reserved_a, Relaxed);
        self.reserved_b.store(rec.reserved_b, Relaxed);
    }

    pub fn load(&self, order: core::sync::atomic::Ordering) -> RawEpisodicRecord {
        use core::sync::atomic::Ordering::*;
        RawEpisodicRecord {
            tick: self.tick.load(Relaxed),
            timestamp_ms: self.timestamp_ms.load(Relaxed),
            node_id_a: self.node_id_a.load(Relaxed),
            node_id_b: self.node_id_b.load(Relaxed),
            payload: self.payload.load(Relaxed),
            meta: self.meta.load(order),
            reserved_a: self.reserved_a.load(Relaxed),
            reserved_b: self.reserved_b.load(Relaxed),
        }
    }
}

/// Event type discriminants stored in the meta byte.
#[repr(u8)]
pub enum EventType {
    NodeFired = 0,
    PredictionError = 1,
    StructuralFault = 2,
    SensorReading = 3,
    IntentProcessed = 4,
}

pub fn pack_meta(event_type: u8, novelty: f32, arousal: f32, reward: f32, count: u8) -> u64 {
    let n = (novelty.clamp(0.0, 1.0) * 65535.0) as u64;
    let a = (arousal.clamp(0.0, 1.0) * 65535.0) as u64;
    let r = (reward.clamp(0.0, 1.0) * 65535.0) as u64;
    (n << 48) | (a << 32) | (r << 16) | ((event_type as u64) << 8) | (count as u64)
}
