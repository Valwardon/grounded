// ────────────────────────────────────────────────────────────
//  Lock-Free SPSC Episodic Ring Buffer
//
//  Single producer (cognitive daemon thread), single consumer
//  (idle consolidation). 1024 records × 64 bytes = 64 KB total.
//
//  Writer protocol:
//    1. Load write_seq, compute slot index (wrapping)
//    2. Check if slot is empty (tick == 0) — spin if full (shouldn't
//       happen at 1024 slots with 50ms burst before consolidate)
//    3. Store all fields with Relaxed ordering
//    4. Store meta with Release ordering (commits the write)
//    5. Increment write_seq with Release
//
//  Reader protocol:
//    1. Load read_seq, count available = write_seq - read_seq
//    2. For each available slot: load meta with Acquire,
//       then load remaining fields with Relaxed
//    3. After processing, set slot tick to 0 with Release
//    4. Increment read_seq with Release
// ────────────────────────────────────────────────────────────

use core::sync::atomic::{AtomicU64, Ordering};

use crate::record::*;

pub struct EpisodicRingBuffer {
    records: Box<[AtomicRecord; EPISODIC_RING_SIZE]>,
    write_seq: AtomicU64,
    read_seq: AtomicU64,
}

impl EpisodicRingBuffer {
    pub fn new() -> Self {
        // Initialize all 1024 atomic records to zero
        let records = Box::new(Default::default());
        // We need to zero-init because AtomicU64 doesn't implement Default on all targets
        // Use array initialization via a helper
        EpisodicRingBuffer {
            records: Self::zeroed_records(),
            write_seq: AtomicU64::new(0),
            read_seq: AtomicU64::new(0),
        }
    }

    fn zeroed_records() -> Box<[AtomicRecord; EPISODIC_RING_SIZE]> {
        // SAFETY: AtomicU64 can be zero-initialized (all-zero bit pattern is valid)
        unsafe {
            let layout = std::alloc::Layout::new::<[AtomicRecord; EPISODIC_RING_SIZE]>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut [AtomicRecord; EPISODIC_RING_SIZE];
            if ptr.is_null() { panic!("OOM in episodic ring buffer"); }
            Box::from_raw(ptr)
        }
    }

    /// Writer: push one record. Returns false if the ring is full.
    pub fn push(&self, rec: &RawEpisodicRecord) -> bool {
        let seq = self.write_seq.load(Ordering::Relaxed);
        let idx = (seq as usize) & (EPISODIC_RING_SIZE - 1);

        // Quick check: slot should be empty (tick == 0 means consumed)
        if self.records[idx].tick.load(Ordering::Relaxed) != 0 {
            return false; // Ring full — event dropped (rare at 1024 slots)
        }

        // Store all fields; meta with Release commits the write
        self.records[idx].store(rec, Ordering::Release);

        // Publish the slot
        self.write_seq.store(seq.wrapping_add(1), Ordering::Release);
        true
    }

    /// Reader: pop one record. Returns None if empty.
    pub fn pop(&self) -> Option<RawEpisodicRecord> {
        let read = self.read_seq.load(Ordering::Relaxed);
        let write = self.write_seq.load(Ordering::Acquire);
        if read == write {
            return None;
        }

        let idx = (read as usize) & (EPISODIC_RING_SIZE - 1);
        let rec = self.records[idx].load(Ordering::Acquire);

        // Mark slot as consumed
        self.records[idx].tick.store(0, Ordering::Release);
        self.read_seq.store(read.wrapping_add(1), Ordering::Release);

        Some(rec)
    }

    /// Reader: drain all available records into a Vec.
    pub fn drain_all(&self) -> Vec<RawEpisodicRecord> {
        let mut out = Vec::with_capacity(128);
        while let Some(rec) = self.pop() {
            out.push(rec);
        }
        out
    }

    /// Writer: number of records written (for debugging).
    pub fn write_count(&self) -> u64 {
        self.write_seq.load(Ordering::Relaxed)
    }

    /// Reader: number of records read (for debugging).
    pub fn read_count(&self) -> u64 {
        self.read_seq.load(Ordering::Relaxed)
    }

    /// Available records for reading.
    pub fn available(&self) -> u64 {
        self.write_seq.load(Ordering::Acquire)
            .wrapping_sub(self.read_seq.load(Ordering::Relaxed))
    }
}

/// AtomicRecord doesn't derive Default; we do it manually.
impl Default for AtomicRecord {
    fn default() -> Self {
        AtomicRecord {
            tick: AtomicU64::new(0),
            timestamp_ms: AtomicU64::new(0),
            node_id_a: AtomicU64::new(0),
            node_id_b: AtomicU64::new(0),
            payload: AtomicU64::new(0),
            meta: AtomicU64::new(0),
            reserved_a: AtomicU64::new(0),
            reserved_b: AtomicU64::new(0),
        }
    }
}
