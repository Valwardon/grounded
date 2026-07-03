use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  RenderBridge — wgpu Effector Bridge
//
//  Runs on its own OS thread ("render-bridge"). Polls the
//  lock-free VisualEffectorBuffer for new render state frames,
//  and feeds them into a wgpu state machine via the RenderBackend
//  trait.
//
//  Architecture:
//    - Cognitive daemon thread (writer) → VisualEffectorBuffer
//    - RenderBridge thread (reader)     → VisualEffectorBuffer
//    - RenderBridge                     → RenderBackend trait
//    - AST validation errors            → GraphArena::penalize_node
//
//  All rendering happens through the RenderBackend trait, so the
//  bridge itself has no direct wgpu dependency. The actual wgpu
//  renderer is injected at construction time.
// ────────────────────────────────────────────────────────────

/// Abstract rendering backend that the bridge feeds state to.
///
/// The actual wgpu renderer implements this trait. The bridge
/// polls the effector buffer and calls `render()` with each
/// new frame.
pub trait RenderBackend: Send + 'static {
    /// Render one frame from the current effector state.
    /// Returns Ok(render_hash) on success.
    fn render(&mut self, state: &[f32; EFFECTOR_STATE_FLOATS]) -> Result<u64, String>;

    /// Handle a validate_ast() error. bridge calls this when the
    /// renderer detects a structural error.
    fn handle_validation_error(&mut self, node_id: &str, error: &str);
}

/// Callback type for AST validation error → GraphArena penalization.
pub type PenalizeFn = Box<dyn Fn(&str, &str) + Send + Sync>;

/// The render bridge — manages its own thread and polls the effector buffer
/// and visual primitive ring buffer.
pub struct RenderBridge {
    buffer: Arc<VisualEffectorBuffer>,
    visual_ring: Option<Arc<VisualPrimitiveRingBuffer>>,
    running: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    backend: Option<Box<dyn RenderBackend>>,
    penalize: Option<PenalizeFn>,
}

impl RenderBridge {
    /// Create a new RenderBridge with the shared effector buffer
    /// and optional visual primitive ring buffer.
    ///
    /// Returns (bridge, buffer_handle). Pass the buffer_handle and
    /// visual_ring to CognitiveDaemon::new() so both sides share
    /// the same buffers.
    pub fn new(visual_ring: Option<Arc<VisualPrimitiveRingBuffer>>) -> (Self, Arc<VisualEffectorBuffer>) {
        let buffer = Arc::new(VisualEffectorBuffer::new());
        let bridge = RenderBridge {
            buffer: buffer.clone(),
            visual_ring,
            running: Arc::new(AtomicBool::new(false)),
            thread: None,
            backend: None,
            penalize: None,
        };
        (bridge, buffer)
    }

    /// Attach a render backend (wgpu or null).
    pub fn set_backend(&mut self, backend: Box<dyn RenderBackend>) {
        self.backend = Some(backend);
    }

    /// Set the AST validation error callback.
    ///
    /// This callback is called when the renderer detects a
    /// structural error (e.g., contract mismatch). The bridge
    /// maps the error back to a GraphArena node index and
    /// penalizes the node.
    ///
    /// The standard callback is:
    ///   |node_id, error| {
    ///       ctx.graph.write().penalize_node(node_id, error);
    ///   }
    pub fn set_penalize_fn(&mut self, f: PenalizeFn) {
        self.penalize = Some(f);
    }

    /// Start the render bridge thread.
    pub fn start(&mut self) {
        if self.running.load(Ordering::Acquire) {
            return;
        }
        self.running.store(true, Ordering::Release);
        let running = self.running.clone();
        let buffer = self.buffer.clone();
        let visual_ring = self.visual_ring.clone();
        let penalize = self.penalize.take();

        // Send backend to the thread; replace with None in self
        let mut backend: Option<Box<dyn RenderBackend>> = self.backend.take();

        self.thread = Some(
            std::thread::Builder::new()
                .name("render-bridge".into())
                .stack_size(2 * 1024 * 1024)
                .spawn(move || {
                    let mut state = [0.0f32; EFFECTOR_STATE_FLOATS];
                    let mut visual_payload = FixedVisualPayload::new();

                    while running.load(Ordering::Acquire) {
                        // Poll the effector buffer for new state
                        if buffer.read(&mut state) {
                            // New state available — send to backend
                            if let Some(ref mut b) = backend {
                                match b.render(&state) {
                                    Ok(hash) => {
                                        let _ = hash;
                                    }
                                    Err(e) => {
                                        if let Some(ref penalize) = penalize {
                                            penalize("validate_ast", &e);
                                        }
                                    }
                                }
                            }
                        }

                        // Poll the visual primitive ring buffer
                        if let Some(ref ring) = visual_ring {
                            if ring.pop(&mut visual_payload) {
                                // Payload received — map to render call
                                let mut vis = [0.0f32; EFFECTOR_STATE_FLOATS];
                                let arr = visual_payload.as_array();
                                for (i, v) in arr.iter().enumerate().take(EFFECTOR_STATE_FLOATS.min(6)) {
                                    vis[i] = *v as f32;
                                }
                                if let Some(ref mut b) = backend {
                                    match b.render(&vis) {
                                        Ok(_hash) => {}
                                        Err(e) => {
                                            if let Some(ref penalize) = penalize {
                                                penalize("visual_primitive", &e);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Sleep for 8ms (≈120Hz render polling)
                        std::thread::sleep(std::time::Duration::from_millis(8));
                    }
                })
                .expect("render bridge thread"),
        );
    }

    /// Stop the bridge thread and join.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }

    /// Check if the bridge thread is alive.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get a reference to the shared effector buffer.
    pub fn buffer(&self) -> Arc<VisualEffectorBuffer> {
        self.buffer.clone()
    }
}

// ────────────────────────────────────────────────────────────
//  Null render backend — logs state but doesn't render
// ────────────────────────────────────────────────────────────

/// A no-op render backend that just logs the effector state.
pub struct NullRenderBackend;

impl RenderBackend for NullRenderBackend {
    fn render(&mut self, state: &[f32; EFFECTOR_STATE_FLOATS]) -> Result<u64, String> {
        // Compute a simple hash from the state for feedback loop
        let mut hash: u64 = 0;
        for v in state.iter() {
            hash = hash.wrapping_mul(31).wrapping_add(v.to_bits() as u64);
        }
        Ok(hash)
    }

    fn handle_validation_error(&mut self, node_id: &str, error: &str) {
        eprintln!("[NullRender] validation error on {}: {}", node_id, error);
    }
}

// ────────────────────────────────────────────────────────────
//  WGPU render backend stub
//
//  The actual wgpu renderer is constructed at runtime with
//  wgpu::Device, wgpu::Queue, and a surface. This stub
//  shows the expected interface.
// ────────────────────────────────────────────────────────────

/// WGPU-specific render backend. Created at runtime when a wgpu
/// surface is available (Android NativeActivity or desktop window).
///
/// To construct, pass a wgpu::Device + Queue + Surface and a
/// RenderAst pipeline. The backend interprets the effector state
/// as skeletal bone transformations and color palette updates.
///
/// ```ignore
/// let backend = WgpuRenderBackend::new(device, queue, surface, config, ast_pipeline);
/// bridge.set_backend(Box::new(backend));
/// ```
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    use super::*;

    /// WGPU render backend — interprets VisualEffectorState as
    /// skeletal animation + color palette commands for the wgpu
    /// render pipeline.
    pub struct WgpuRenderBackend {
        // device: wgpu::Device,
        // queue: wgpu::Queue,
        // surface: wgpu::Surface,
        // pipeline: RenderAstPipeline,
        last_hash: u64,
    }

    impl WgpuRenderBackend {
        pub fn new() -> Self {
            WgpuRenderBackend { last_hash: 0 }
        }
    }

    impl RenderBackend for WgpuRenderBackend {
        fn render(&mut self, state: &[f32; EFFECTOR_STATE_FLOATS]) -> Result<u64, String> {
            // Compute hash for feedback loop
            let mut hash: u64 = 0;
            for v in state.iter() {
                hash = hash.wrapping_mul(31).wrapping_add(v.to_bits() as u64);
            }
            self.last_hash = hash;

            // In production:
            // 1. Read skeletal rotation angles (state[0..6])
            // 2. Update bone transform uniform buffers
            // 3. Read palette colors (state[6..14])
            // 4. Update color palette textures/buffers
            // 5. Read scale (state[14..17])
            // 6. Update skeletal scale uniforms
            // 7. Read blend weight (state[17])
            // 8. Update blend weight uniforms
            // 9. Read palette coefficients (state[18..20])
            // 10. Interpolate palette in shader
            // 11. Read wireframe flag (state[21])
            // 12. Toggle wireframe rendering mode
            // 13. Encode and submit render pass
            // 14. Present

            Ok(hash)
        }

        fn handle_validation_error(&mut self, node_id: &str, error: &str) {
            eprintln!("[WgpuRender] validation error on {}: {}", node_id, error);
        }
    }
}
