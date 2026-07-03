// ────────────────────────────────────────────────────────────
//  Constrained Intermediate Module DSL
//
//  A strictly restricted, domain-specific intermediate language
//  for describing safe code rewrites of Layer 1 cognitive modules.
//
//  The `CandidateModule` AST describes:
//    - Inputs & Outputs (statically typed arrays of grounded primitives)
//    - State Scoping (isolated state variables, no raw pointers, no heap)
//    - A `tick_logic` placeholder that is compiled into a native
//      function satisfying a Layer 1 trait bound.
//
//  Safety guarantees (enforced at the DSL level):
//    - NO raw pointer access
//    - NO heap allocation in the execution path
//    - NO unsafe blocks
//    - All state is explicitly scoped and fixed-size
//    - All types must implement the `DslType` trait
// ────────────────────────────────────────────────────────────

use std::time::Duration;
use semantic_graph::prelude::*;
use crate::layers::ModuleId;

// ── DSL Type System ───────────────────────────────────────

/// Types valid in the DSL.
/// Only primitive Grounded types are permitted — no raw pointers,
/// no external references, no heap-allocated dynamic structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DslType {
    /// Scalar f64 activation value.
    Activation,
    /// A NodeId referencing a node in the GraphArena.
    NodeId,
    /// A fixed-length vector of f64 values (maximum 32 elements).
    FixedVec { max_len: usize },
    /// A conceptual frame (frame_spec reference by schema name).
    FrameRef,
    /// A boolean flag.
    Flag,
    /// An enum variant selector (range 0..N).
    Selector { variants: u8 },
}

impl DslType {
    /// Size of this type in the fixed stack frame (in bytes).
    pub const fn stack_size(&self) -> usize {
        match self {
            DslType::Activation => 8,  // f64
            DslType::NodeId => 8,      // u64
            DslType::FixedVec { max_len } => 8 + max_len * 8, // len + data
            DslType::FrameRef => 8,    // index
            DslType::Flag => 1,        // bool
            DslType::Selector { .. } => 1,
        }
    }
}

// ── DSL State Variable ────────────────────────────────────

/// A state variable in the DSL module.
///
/// Constraints:
///   - Must have a fixed size known at compile time.
///   - Stack-allocated in the module's execution context.
///   - Cannot be a pointer, reference, or heap-allocated type.
#[derive(Debug, Clone)]
pub struct StateVar {
    /// Name of the state variable.
    pub name: String,
    /// Type of the variable.
    pub var_type: DslType,
    /// Default value as raw bytes (max 256 bytes).
    pub default: [u8; 256],
    /// Valid range for validation (min, max).
    pub range: Option<(f64, f64)>,
}

// ── I/O Port Definition ───────────────────────────────────

/// An input or output port of a DSL module.
#[derive(Debug, Clone)]
pub struct Port {
    pub name: String,
    pub port_type: DslType,
    pub description: String,
}

impl Port {
    pub fn new(name: &str, port_type: DslType) -> Self {
        Port {
            name: name.to_string(),
            port_type,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }
}

// ── Tick Logic Opcodes ────────────────────────────────────

/// A restricted set of opcodes for the tick logic.
/// This is NOT a general-purpose instruction set — it only supports
/// the operations needed by cognitive modules (pattern matching,
/// frame slot filling, threshold gating, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    /// Load a value from an input port onto the evaluation stack.
    LoadInput,
    /// Load a state variable onto the evaluation stack.
    LoadState,
    /// Store the top of stack into a state variable.
    StoreState,
    /// Push a constant f64 onto the stack.
    PushConst,
    /// Add top two stack values.
    Add,
    /// Subtract top two stack values.
    Sub,
    /// Multiply top two stack values.
    Mul,
    /// Divide top two stack values.
    Div,
    /// Compare: push 1.0 if a < b, else 0.0
    LessThan,
    /// Compare: push 1.0 if a > b, else 0.0
    GreaterThan,
    /// Logical AND (both non-zero).
    And,
    /// Logical OR (either non-zero).
    Or,
    /// Clamp top of stack to [min, max] (min and max are immediates).
    Clamp,
    /// Select between two values based on a flag.
    Select,
    /// Check if a NodeId matches a label prefix.
    MatchLabel,
    /// Compute Jaccard overlap between two frame slot sets.
    FrameOverlap,
    /// Emit a frame with a given score (top of stack = score).
    EmitFrame,
    /// Halt execution (module finished).
    Halt,
}

/// A single instruction in the DSL tick logic.
#[derive(Debug, Clone)]
pub struct Instruction {
    pub opcode: Opcode,
    /// Immediate operands (opcode-specific).
    pub immediates: [u64; 4],
}

impl Instruction {
    pub fn new(opcode: Opcode) -> Self {
        Instruction {
            opcode,
            immediates: [0; 4],
        }
    }

    pub fn with_imm(mut self, idx: usize, val: u64) -> Self {
        if idx < 4 {
            self.immediates[idx] = val;
        }
        self
    }
}

// ── Compiled Logic ────────────────────────────────────────

/// A validated, compiled tick logic function.
///
/// After the DSL bytecode is validated and compiled, it produces
/// a native function pointer of this type. The function operates
/// on a fixed-size stack frame with no heap allocation.
///
/// Type signature:
///   inputs: fixed array of f64 values representing input ports
///   state: mutable reference to the module's state byte buffer
///   outputs: mutable reference to the output port byte buffer
///   graph: reference to the GraphArena for lookup operations
///
/// Returns the number of frames emitted, or u64::MAX on error.
pub type CompiledTickFn = fn(
    inputs: &[f64],
    state: &mut [u8],
    outputs: &mut [u8],
    graph: &GraphArena,
) -> u64;

/// Placeholder compiler — maps a CandidateModule hash to a pre-compiled
/// native function. In production, this would invoke a real DSL→Rust
/// code generator (e.g., using proc_macro2 + syn at build time, or
/// a lightweight bytecode interpreter for the opcodes above).
pub struct CompiledLogic {
    /// The compiled function, if available.
    pub tick_fn: Option<CompiledTickFn>,
    /// Fallback interpreter for un-compiled bytecode.
    pub bytecode: Vec<Instruction>,
    /// Whether this logic was verified safe.
    pub verified: bool,
    /// Verification hash of the source module definition.
    pub source_hash: u64,
}

impl CompiledLogic {
    pub fn new() -> Self {
        CompiledLogic {
            tick_fn: None,
            bytecode: Vec::new(),
            verified: false,
            source_hash: 0,
        }
    }

    /// Execute this logic with the given inputs and state.
    /// Falls back to the bytecode interpreter if no native function is compiled.
    pub fn execute(
        &self,
        inputs: &[f64],
        state: &mut [u8],
        outputs: &mut [u8],
        graph: &GraphArena,
    ) -> u64 {
        if let Some(f) = self.tick_fn {
            f(inputs, state, outputs, graph)
        } else {
            self.interpret(inputs, state, outputs)
        }
    }

    /// Lightweight bytecode interpreter for fallback execution.
    /// Processes opcodes against a small fixed evaluation stack (max 16 entries).
    fn interpret(&self, inputs: &[f64], state: &mut [u8], outputs: &mut [u8]) -> u64 {
        let mut stack: [f64; 16] = [0.0; 16];
        let mut sp: usize = 0; // stack pointer
        let mut frame_count: u64 = 0;
        let mut pc: usize = 0; // program counter

        while pc < self.bytecode.len() {
            let inst = &self.bytecode[pc];
            match inst.opcode {
                Opcode::Halt => break,
                Opcode::LoadInput => {
                    let idx = inst.immediates[0] as usize;
                    if idx < inputs.len() && sp < 16 {
                        stack[sp] = inputs[idx];
                        sp += 1;
                    }
                }
                Opcode::LoadState => {
                    let offset = inst.immediates[0] as usize;
                    if offset + 8 <= state.len() && sp < 16 {
                        let bytes: [u8; 8] = state[offset..offset + 8].try_into().unwrap_or([0; 8]);
                        stack[sp] = f64::from_le_bytes(bytes);
                        sp += 1;
                    }
                }
                Opcode::StoreState => {
                    if sp > 0 {
                        let offset = inst.immediates[0] as usize;
                        let val = stack[sp - 1];
                        if offset + 8 <= state.len() {
                            state[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
                        }
                        sp -= 1;
                    }
                }
                Opcode::PushConst => {
                    if sp < 16 {
                        let val = f64::from_bits(inst.immediates[0]);
                        stack[sp] = val;
                        sp += 1;
                    }
                }
                Opcode::Add => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = a + b; sp -= 1; }
                }
                Opcode::Sub => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = a - b; sp -= 1; }
                }
                Opcode::Mul => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = a * b; sp -= 1; }
                }
                Opcode::Div => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = if b != 0.0 { a / b } else { 0.0 }; sp -= 1; }
                }
                Opcode::LessThan => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = if a < b { 1.0 } else { 0.0 }; sp -= 1; }
                }
                Opcode::GreaterThan => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = if a > b { 1.0 } else { 0.0 }; sp -= 1; }
                }
                Opcode::And => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }; sp -= 1; }
                }
                Opcode::Or => {
                    if sp >= 2 { let b = stack[sp - 1]; let a = stack[sp - 2]; stack[sp - 2] = if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }; sp -= 1; }
                }
                Opcode::Clamp => {
                    if sp > 0 {
                        let min = f64::from_bits(inst.immediates[0]);
                        let max = f64::from_bits(inst.immediates[1]);
                        stack[sp - 1] = stack[sp - 1].clamp(min, max);
                    }
                }
                Opcode::EmitFrame => {
                    if sp > 0 {
                        let _score = stack[sp - 1];
                        frame_count += 1;
                        sp -= 1;
                    }
                }
                _ => {}
            }
            pc += 1;
        }
        frame_count
    }
}

// ── Candidate Module (The DSL Root) ───────────────────────

/// A candidate replacement for a Layer 1 cognitive module.
///
/// This is the root of the DSL AST. It describes the module's
/// interface, state, and tick logic in a safe, constrained form.
/// The Self-Healing Pipeline generates these, verifies them, and
/// if they pass all checks, hot-swaps them into the ModuleRegistry.
///
/// # Safety
/// - All state is stack-allocated (no heap in the hot path)
/// - All types are DslType (no raw pointers)
/// - The tick logic is verified before compilation
/// - unsafe is prohibited
#[derive(Debug, Clone)]
pub struct CandidateModule {
    /// Unique identifier for this candidate.
    pub id: ModuleId,

    /// Human-readable name describing the module's role.
    pub name: String,

    /// The Layer 1 trait this module satisfies.
    pub trait_bound: ModuleTrait,

    /// Input ports (all must be DslType, max 8 ports).
    pub inputs: Vec<Port>,

    /// Output ports (max 8 ports).
    pub outputs: Vec<Port>,

    /// State variables (all fixed-size, max 256 bytes total).
    pub state_vars: Vec<StateVar>,

    /// Maximum stack depth for the tick logic (max 16).
    pub max_stack_depth: u8,

    /// The tick logic as compiled bytecode.
    pub logic: CompiledLogic,

    /// Verification hash — derived from inputs + outputs + state + bytecode.
    /// Used to detect changes between candidate and stock module.
    pub verification_hash: u64,

    /// Whether this module is currently active (hot-swapped in).
    pub active: bool,
}

impl CandidateModule {
    /// Create a new candidate module with the given identity.
    pub fn new(id: ModuleId, name: &str, trait_bound: ModuleTrait) -> Self {
        CandidateModule {
            id,
            name: name.to_string(),
            trait_bound,
            inputs: Vec::with_capacity(8),
            outputs: Vec::with_capacity(8),
            state_vars: Vec::with_capacity(4),
            max_stack_depth: 8,
            logic: CompiledLogic::new(),
            verification_hash: 0,
            active: false,
        }
    }

    /// Add an input port.
    pub fn with_input(mut self, port: Port) -> Self {
        if self.inputs.len() < 8 {
            self.inputs.push(port);
        }
        self
    }

    /// Add an output port.
    pub fn with_output(mut self, port: Port) -> Self {
        if self.outputs.len() < 8 {
            self.outputs.push(port);
        }
        self
    }

    /// Add a state variable.
    pub fn with_state(mut self, var: StateVar) -> Self {
        self.state_vars.push(var);
        self
    }

    /// Compute the verification hash from the module's structure.
    pub fn compute_hash(&self) -> u64 {
        let mut h: u64 = self.id.0;
        h = h.wrapping_mul(31).wrapping_add(self.name.len() as u64);
        h = h.wrapping_mul(31).wrapping_add(self.trait_bound as u64);
        for p in &self.inputs {
            h = h.wrapping_mul(31).wrapping_add(p.name.len() as u64);
        }
        for p in &self.outputs {
            h = h.wrapping_mul(31).wrapping_add(p.name.len() as u64);
        }
        for inst in &self.logic.bytecode {
            h = h.wrapping_mul(31).wrapping_add(inst.opcode as u64);
            for imm in &inst.immediates {
                h = h.wrapping_mul(7).wrapping_add(*imm);
            }
        }
        h
    }

    /// Total state size in bytes.
    pub fn state_size(&self) -> usize {
        self.state_vars.iter().map(|v| v.var_type.stack_size()).sum()
    }

    /// Validate the module structure (checks all constraints).
    /// Returns Ok(()) if valid, Err with description of first violation.
    pub fn validate(&self) -> Result<(), String> {
        // Check input count
        if self.inputs.len() > 8 {
            return Err("too many input ports (max 8)".into());
        }
        // Check output count
        if self.outputs.len() > 8 {
            return Err("too many output ports (max 8)".into());
        }
        // Check total state size ≤ 256 bytes (stack allocation limit)
        if self.state_size() > 256 {
            return Err(format!(
                "state size {} exceeds max 256 bytes",
                self.state_size()
            ));
        }
        // Check max stack depth
        if self.max_stack_depth > 16 {
            return Err("max stack depth exceeds 16".into());
        }
        // Check that logic has been verified
        if !self.logic.verified {
            return Err("logic has not been verified".into());
        }
        // Check all state variables have valid types
        for var in &self.state_vars {
            match var.var_type {
                DslType::FixedVec { max_len } => {
                    if max_len > 32 {
                        return Err("FixedVec max_len > 32 not allowed".into());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

// ── Module Trait Enum ─────────────────────────────────────

/// The Layer 1 trait that a CandidateModule satisfies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleTrait {
    CognitiveParser = 1,
    FrameMatcher = 2,
    CuriosityScheduler = 3,
    GapDetectorModule = 4,
}

// ── DSL Compiler Placeholder ──────────────────────────────

/// Placeholder DSL compiler.
///
/// In production, this would:
///   1. Take a CandidateModule AST
///   2. Generate Rust source via proc_macro2 + syn at build time
///      OR compile bytecode to a native function at cold-start
///   3. Return a CompiledLogic with a valid tick_fn pointer
///
/// Currently, it validates the bytecode and marks it as verified.
pub struct DslCompiler;

impl DslCompiler {
    /// "Compile" a candidate module — validates bytecode and computes hash.
    /// In production, this would generate native code.
    pub fn compile(candidate: &mut CandidateModule) -> Result<(), String> {
        // 1. Validate structure
        if candidate.inputs.len() > 8 {
            return Err("too many inputs".into());
        }
        if candidate.outputs.len() > 8 {
            return Err("too many outputs".into());
        }
        if candidate.state_size() > 256 {
            return Err("state too large".into());
        }

        // 2. Verify bytecode terminates (last instruction must be Halt)
        let has_halt = candidate.logic.bytecode.last()
            .map(|i| i.opcode == Opcode::Halt)
            .unwrap_or(true);
        if !has_halt {
            return Err("bytecode must end with Halt".into());
        }

        // 3. Verify no invalid opcode sequences
        for instr in &candidate.logic.bytecode {
            match instr.opcode {
                Opcode::LoadInput | Opcode::LoadState | Opcode::StoreState
                | Opcode::PushConst | Opcode::Add | Opcode::Sub | Opcode::Mul
                | Opcode::Div | Opcode::LessThan | Opcode::GreaterThan
                | Opcode::And | Opcode::Or | Opcode::Clamp | Opcode::Select
                | Opcode::MatchLabel | Opcode::FrameOverlap | Opcode::EmitFrame
                | Opcode::Halt => {} // all valid
            }
        }

        // 4. Mark as verified
        candidate.logic.verified = true;
        candidate.verification_hash = candidate.compute_hash();

        Ok(())
    }

    /// Compute a signature hash for the module's interface.
    /// Used to check equality of type signatures between stock and candidate.
    pub fn interface_hash(candidate: &CandidateModule) -> u64 {
        let mut h: u64 = candidate.trait_bound as u64;
        for p in &candidate.inputs {
            h = h.wrapping_mul(31).wrapping_add(p.port_type as u64);
        }
        for p in &candidate.outputs {
            h = h.wrapping_mul(31).wrapping_add(p.port_type as u64);
        }
        h
    }
}
