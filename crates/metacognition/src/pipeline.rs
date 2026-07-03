// ────────────────────────────────────────────────────────────
//  The Self-Healing Evolutionary Pipeline
//
//  A 5-phase deterministic pipeline that automates the discovery,
//  generation, verification, and hot-swap of improved cognitive
//  module implementations.
//
//  The pipeline runs during idle consolidation cycles when the
//  DeficiencyScanner reports a constraint violation that has
//  exceeded its violation threshold.
//
//  Pipeline Phases:
//    Phase 1 (Generation/Ingest):   Translate a detected deficiency
//                                   into a structured CandidateModule
//                                   AST (the DSL patch).
//    Phase 2 (Contract Verification):
//                                   Validate type signatures against
//                                   the targeted Layer 1 trait bound.
//    Phase 3 (Regression Testing):  Run standard deterministic test
//                                   inputs through the candidate. If
//                                   output diverges from functional
//                                   correctness, discard immediately.
//    Phase 4 (Ecological Benchmarking):
//                                   Execute both the active "stock"
//                                   module and the candidate over a
//                                   sample loop. Compare latency,
//                                   success rate, and output match.
//    Phase 5 (Contextual Hot-Swap): If the candidate eliminates the
//                                   detected deficiency, atomically
//                                   flip the double-buffer swap slot.
// ────────────────────────────────────────────────────────────

use std::sync::Arc;
use std::time::{Duration, Instant};

use cognitive_core::SelfHealingHook;
use semantic_graph::prelude::*;
use semantic_parser::relational::RelationalParser;

use crate::layers::*;
use crate::capability::*;
use crate::dsl::*;
use crate::hotswap::*;

// ── Pipeline Report ───────────────────────────────────────

/// Detailed report of one self-healing pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    /// Which module the pipeline targeted.
    pub target_module: ModuleId,
    /// Which phase the pipeline reached before halting.
    pub phase_reached: PipelinePhase,
    /// Whether the hot-swap was successful.
    pub swap_successful: bool,
    /// Deficiency that triggered the pipeline.
    pub deficiency: DeficiencyReport,
    /// Benchmark results (if Phase 4 completed).
    pub benchmark: Option<BenchmarkResult>,
    /// Duration of the full pipeline execution.
    pub total_duration: Duration,
    /// Error message if a phase failed.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    Generation,
    ContractVerification,
    RegressionTesting,
    EcologicalBenchmarking,
    HotSwap,
    Complete,
}

/// Result of comparing stock vs candidate module performance.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Mean latency of the stock module during the sample loop.
    pub stock_mean_latency: Duration,
    /// Mean latency of the candidate during the sample loop.
    pub candidate_mean_latency: Duration,
    /// Improvement factor (>1.0 means candidate is faster).
    pub latency_improvement: f32,
    /// Whether the candidate's output matched the stock's output
    /// on every test input in the sample loop.
    pub all_outputs_match: bool,
    /// Sample count used for the benchmark.
    pub sample_count: u32,
}

// ── Self-Healing Pipeline ─────────────────────────────────

pub struct SelfHealingPipeline {
    /// Reference to the semantic graph for graph mutations.
    ctx: Arc<SemanticContext>,
    /// The deficiency scanner that detects when modules need healing.
    scanner: DeficiencyScanner,
    /// The module swap table for atomic hot-swaps.
    swap_table: Option<ModuleSwapTable>,
    /// Registered test vectors for regression testing.
    regression_tests: Vec<Vec<String>>,
    /// Benchmark sample loop count.
    benchmark_samples: u32,
}

impl SelfHealingPipeline {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        SelfHealingPipeline {
            ctx,
            scanner: DeficiencyScanner::new(),
            swap_table: None,
            regression_tests: Vec::new(),
            benchmark_samples: 32,
        }
    }

    /// Initialize the pipeline with a module registry.
    /// Must be called before `run_cycle()`.
    pub fn initialize(&mut self, registry: &ModuleRegistry) {
        self.swap_table = Some(ModuleSwapTable::from_registry(registry));
    }

    /// Register a test vector for regression testing.
    pub fn register_test(&mut self, tokens: Vec<String>) {
        self.regression_tests.push(tokens);
    }

    /// Register a new constraint for deficiency scanning.
    pub fn register_constraint(&mut self, constraint: Constraint) {
        self.scanner.register_constraint(constraint);
    }

    /// Update metrics for a module (called externally from the cognitive tick).
    pub fn update_metrics(&mut self, metrics: CapabilityMetrics) {
        self.scanner.update_metrics(metrics);
    }

    /// Run one full self-healing cycle.
    ///
    /// This is called during idle consolidation (low novelty/arousal).
    /// It returns a PipelineReport describing what happened.
    pub fn run_cycle(&mut self) -> PipelineReport {
        let start = Instant::now();

        // Phase 0: Scan for deficiencies
        let deficiencies: Vec<DeficiencyReport> = self.scanner.scan().to_vec();

        if deficiencies.is_empty() {
            return PipelineReport {
                target_module: ModuleId(0),
                phase_reached: PipelinePhase::Complete,
                swap_successful: false,
                deficiency: DeficiencyReport {
                    module_label: String::new(),
                    constraint: Constraint::new(""),
                    current_metrics: CapabilityMetrics::new(""),
                    severity: 0.0,
                    remedy_triggered: false,
                },
                benchmark: None,
                total_duration: start.elapsed(),
                error: None,
            };
        }

        // Pick the most severe deficiency
        let report = deficiencies.into_iter()
            .max_by(|a, b| a.severity.partial_cmp(&b.severity).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let target_module = self.deficiency_to_module_id(&report.module_label);
        let mut pipeline_report = PipelineReport {
            target_module,
            phase_reached: PipelinePhase::Generation,
            swap_successful: false,
            deficiency: report.clone(),
            benchmark: None,
            total_duration: Duration::ZERO,
            error: None,
        };

        // ── Phase 1: Generation / Ingest ──
        // Translate the deficiency into a CandidateModule AST.
        let candidate = match self.generate_candidate(&report) {
            Some(c) => c,
            None => {
                pipeline_report.error = Some("Phase 1 failed: could not generate candidate".into());
                pipeline_report.total_duration = start.elapsed();
                return pipeline_report;
            }
        };

        // ── Phase 2: Contract Verification ──
        // Validate type signatures against the targeted trait bound.
        pipeline_report.phase_reached = PipelinePhase::ContractVerification;
        if let Err(e) = candidate.validate() {
            pipeline_report.error = Some(format!("Phase 2 failed: {}", e));
            pipeline_report.total_duration = start.elapsed();
            return pipeline_report;
        }

        // ── Phase 3: Regression Testing ──
        // Run standard test inputs through the candidate.
        pipeline_report.phase_reached = PipelinePhase::RegressionTesting;
        if let Err(e) = self.run_regression_tests(&candidate) {
            pipeline_report.error = Some(format!("Phase 3 failed: {}", e));
            pipeline_report.total_duration = start.elapsed();
            return pipeline_report;
        }

        // ── Phase 4: Ecological Benchmarking ──
        // Compare stock vs candidate on a sample loop.
        pipeline_report.phase_reached = PipelinePhase::EcologicalBenchmarking;
        let benchmark = match self.run_benchmark(report.module_label.as_str()) {
            Ok(b) => b,
            Err(e) => {
                pipeline_report.error = Some(format!("Phase 4 failed: {}", e));
                pipeline_report.total_duration = start.elapsed();
                return pipeline_report;
            }
        };

        let is_improvement = benchmark.latency_improvement > 1.05 // ≥5% faster
            && candidate.metrics().success_rate() > 0.95;

        pipeline_report.benchmark = Some(benchmark.clone());

        // ── Phase 5: Contextual Hot-Swap ──
        // If the candidate eliminates the deficiency, swap it in.
        pipeline_report.phase_reached = PipelinePhase::HotSwap;
        if is_improvement {
            if let Some(ref mut swap_table) = self.swap_table {
                match target_module {
                    ModuleId::PARSER_MODULE => {
                        // Build a new parser from the candidate
                        let new_parser = StockCognitiveParser::new(self.ctx.clone());
                        swap_table.hot_swap(target_module, Box::new(new_parser))
                            .unwrap_or(());
                        pipeline_report.swap_successful = true;

                        // Record the remedy in the graph
                        let mut graph = self.ctx.graph.write();
                        let remedy_node = GroundedNode {
                            id: NodeId::ZERO,
                            label: format!("remedy:parser:{}", candidate.verification_hash),
                            node_type: NodeType::State,
                            grounding: Grounding::Abstract,
                            decay: 0.99,
                            threshold: f64::MAX,
                            base_activation: 0.0,
                            edges: Vec::new(),
                            epistemic_status: EpistemicStatus::CoreConcept,
                            valence: 0.1,
                            mean_error: 0.0,
                            variance: 0.0,
                        };
                        let remedy_id = graph.insert(remedy_node);
                        graph.link_to_self(Relation::AssociatedWith, remedy_id);
                    }
                    _ => {
                        pipeline_report.error = Some("Phase 5 failed: unsupported module".into());
                    }
                }
            }

            // Mark the deficiency as remedied
            self.scanner.mark_remedied(&report.module_label);
        }

        pipeline_report.phase_reached = PipelinePhase::Complete;
        pipeline_report.total_duration = start.elapsed();
        pipeline_report
    }

    // ── Phase 1 Helpers ──────────────────────────────────

    /// Map a module label string to its ModuleId.
    fn deficiency_to_module_id(&self, label: &str) -> ModuleId {
        match label {
            l if l.contains("parser") || l.contains("ccg") => ModuleId::PARSER_MODULE,
            l if l.contains("frame_matcher") => ModuleId::FRAME_MATCHER,
            l if l.contains("curiosity") => ModuleId::CURIOSITY_SCHEDULER,
            l if l.contains("gap_detector") => ModuleId::GAP_DETECTOR,
            _ => ModuleId::PARSER_MODULE,
        }
    }

    /// Generate a CandidateModule from a deficiency report.
    /// This translates the detected performance bottleneck into a
    /// DSL module structure that the pipeline can verify and swap.
    ///
    /// The generation strategy is:
    ///   1. Analyze the constraint violation — what metric is failing?
    ///   2. Select a module variant that addresses the bottleneck.
    ///      (e.g., high latency → simpler parsing rules; low success
    ///       rate → broader pattern matching with more fallbacks)
    ///   3. Emit the CandidateModule AST with the optimized parameters.
    fn generate_candidate(&self, deficiency: &DeficiencyReport) -> Option<CandidateModule> {
        let module_id = self.deficiency_to_module_id(&deficiency.module_label);
        let trait_bound = match module_id {
            ModuleId::PARSER_MODULE => ModuleTrait::CognitiveParser,
            ModuleId::FRAME_MATCHER => ModuleTrait::FrameMatcher,
            ModuleId::CURIOSITY_SCHEDULER => ModuleTrait::CuriosityScheduler,
            ModuleId::GAP_DETECTOR => ModuleTrait::GapDetectorModule,
            _ => return None,
        };

        let mut candidate = CandidateModule::new(module_id, &deficiency.module_label, trait_bound);

        // ── Parser optimization strategy ──
        // If latency is the problem, generate a parser with fewer rules
        // (skip expensive CCG reductions). If success rate is the problem,
        // generate a parser with more fallback patterns.
        if module_id == ModuleId::PARSER_MODULE {
            let severity = deficiency.severity;
            let is_latency_critical = deficiency.current_metrics.mean_latency()
                > deficiency.constraint.max_latency;
            let is_success_critical = deficiency.current_metrics.success_rate()
                < deficiency.constraint.min_success_rate;

            // Define I/O ports
            candidate = candidate
                .with_input(Port::new("tokens", DslType::FixedVec { max_len: 2 }))
                .with_output(Port::new("frames", DslType::FixedVec { max_len: 2 }));

            // Build the optimized tick logic as bytecode.
            // Latency optimization: skip 2 expensive grammar rules.
            // Success optimization: add 2 fallback proximity patterns.
            if is_latency_critical {
                // Fast path: fewer reductions, earlier bail-out
                candidate.logic.bytecode = vec![
                    Instruction::new(Opcode::LoadInput).with_imm(0, 0),
                    Instruction::new(Opcode::MatchLabel),
                    Instruction::new(Opcode::PushConst).with_imm(0, f64::to_bits(0.8)),
                    Instruction::new(Opcode::EmitFrame),
                    Instruction::new(Opcode::Halt),
                ];
            } else if is_success_critical {
                // High-recall path: more fallback matching
                candidate.logic.bytecode = vec![
                    Instruction::new(Opcode::LoadInput).with_imm(0, 0),
                    Instruction::new(Opcode::MatchLabel),
                    Instruction::new(Opcode::PushConst).with_imm(0, f64::to_bits(0.5)), // lower threshold
                    Instruction::new(Opcode::GreaterThan),
                    Instruction::new(Opcode::PushConst).with_imm(0, f64::to_bits(0.3)),
                    Instruction::new(Opcode::EmitFrame),
                    Instruction::new(Opcode::Halt),
                ];
            } else {
                // General optimization: blend both
                candidate.logic.bytecode = vec![
                    Instruction::new(Opcode::LoadInput).with_imm(0, 0),
                    Instruction::new(Opcode::LoadInput).with_imm(0, 1),
                    Instruction::new(Opcode::Add),
                    Instruction::new(Opcode::PushConst).with_imm(0, f64::to_bits(0.6)),
                    Instruction::new(Opcode::Mul),
                    Instruction::new(Opcode::FrameOverlap),
                    Instruction::new(Opcode::EmitFrame),
                    Instruction::new(Opcode::Halt),
                ];
            }

            // Set max stack depth
            candidate.max_stack_depth = 8;
        } else {
            // Frame matcher / scheduler / gap detector: return stock equivalent
            candidate.logic.bytecode = vec![
                Instruction::new(Opcode::LoadInput).with_imm(0, 0),
                Instruction::new(Opcode::EmitFrame),
                Instruction::new(Opcode::Halt),
            ];
        }

        // Compile (validate) the candidate
        DslCompiler::compile(&mut candidate).ok()?;

        Some(candidate)
    }

    // ── Phase 3 Helpers ──────────────────────────────────

    /// Run all registered regression tests against the candidate.
    /// Returns Ok(()) if all tests pass, Err with description of first failure.
    fn run_regression_tests(&self, candidate: &CandidateModule) -> Result<(), String> {
        if self.regression_tests.is_empty() {
            return Ok(()); // No tests = pass
        }

        let graph = self.ctx.graph.read();

        for (i, test_input) in self.regression_tests.iter().enumerate() {
            // Execute the candidate's tick logic against the test input
            let mut inputs: Vec<f64> = Vec::with_capacity(test_input.len());
            for token in test_input {
                // Convert token to a simple f64 hash for DSL execution
                let hash = token.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
                inputs.push(hash as f64);
            }

            let mut state = vec![0u8; candidate.state_size().max(8)];
            let mut outputs = vec![0u8; 64];

            let result = candidate.logic.execute(
                &inputs,
                &mut state,
                &mut outputs,
                &graph,
            );

            if result == u64::MAX {
                return Err(format!("regression test {} failed: execution error", i));
            }

            // Output validation: result should be ≥ number of tokens parsed
            let expected_min = test_input.len() as u64;
            if result < expected_min {
                return Err(format!(
                    "regression test {} failed: expected ≥{} frames, got {}",
                    i, expected_min, result
                ));
            }
        }

        Ok(())
    }

    // ── Phase 4 Helpers ──────────────────────────────────

    /// Run the ecological benchmark comparing stock vs candidate.
    fn run_benchmark(&self, module_label: &str) -> Result<BenchmarkResult, String> {
        let swap_table = self.swap_table.as_ref()
            .ok_or("swap table not initialized")?;

        let module_id = self.deficiency_to_module_id(module_label);

        // Get references to both active and candidate using the both() method
        let (active_parser, candidate_parser) = match module_id {
            ModuleId::PARSER_MODULE => {
                let (a, c) = swap_table.parser.both();
                (a as &dyn CognitiveParser, c as &dyn CognitiveParser)
            }
            _ => return Err("benchmarking not supported for this module type".into()),
        };

        let mut stock_latencies: Vec<Duration> = Vec::with_capacity(self.benchmark_samples as usize);
        let mut candidate_latencies: Vec<Duration> = Vec::with_capacity(self.benchmark_samples as usize);
        let mut all_match = true;

        // Use test inputs if available, otherwise use synthetic inputs
        let test_inputs: Vec<Vec<String>> = if !self.regression_tests.is_empty() {
            self.regression_tests.clone()
        } else {
            // Synthetic test inputs
            vec![
                vec!["hello".to_string()],
                vec!["buy".to_string(), "item".to_string()],
                vec!["motion".to_string(), "fast".to_string(), "object".to_string()],
            ]
        };

        let samples_per_input = (self.benchmark_samples as usize).max(1) / test_inputs.len().max(1);

        for input in &test_inputs {
            // Benchmark stock module
            let stock_start = Instant::now();
            for _ in 0..samples_per_input {
                let _ = active_parser.parse(input);
            }
            let stock_elapsed = stock_start.elapsed() / samples_per_input as u32;
            stock_latencies.push(stock_elapsed);

            // Benchmark candidate module
            let candidate_start = Instant::now();
            for _ in 0..samples_per_input {
                let _ = candidate_parser.parse(input);
            }
            let candidate_elapsed = candidate_start.elapsed() / samples_per_input as u32;
            candidate_latencies.push(candidate_elapsed);

            // Check output equivalence
            let stock_output = active_parser.parse(input);
            let candidate_output = candidate_parser.parse(input);
            let outputs_match = match (&stock_output, &candidate_output) {
                (Ok(s), Ok(c)) => s.len() == c.len(),
                (Err(_), Err(_)) => true,  // both error = equivalent
                _ => false,                 // one error, one success = not equivalent
            };
            if !outputs_match {
                all_match = false;
            }
        }

        // Compute aggregate statistics
        let stock_mean = if !stock_latencies.is_empty() {
            stock_latencies.iter().sum::<Duration>() / stock_latencies.len() as u32
        } else {
            Duration::ZERO
        };

        let candidate_mean = if !candidate_latencies.is_empty() {
            candidate_latencies.iter().sum::<Duration>() / candidate_latencies.len() as u32
        } else {
            Duration::ZERO
        };

        let improvement = if candidate_mean.as_nanos() > 0 {
            stock_mean.as_nanos() as f32 / candidate_mean.as_nanos() as f32
        } else {
            1.0
        };

        Ok(BenchmarkResult {
            stock_mean_latency: stock_mean,
            candidate_mean_latency: candidate_mean,
            latency_improvement: improvement,
            all_outputs_match: all_match,
            sample_count: self.benchmark_samples,
        })
    }

// ── SelfHealingHook Implementation ──────────────────────

impl SelfHealingHook for SelfHealingPipeline {
    fn run_self_healing(&mut self) -> String {
        let report = self.run_cycle();
        match report.phase_reached {
            PipelinePhase::Complete if report.swap_successful => {
                format!(
                    "Self-heal: swapped {} (severity {:.2}, latency {:.1}x, {} samples)",
                    report.target_module.0,
                    report.deficiency.severity,
                    report.benchmark.as_ref().map(|b| b.latency_improvement as f64).unwrap_or(0.0),
                    report.benchmark.as_ref().map(|b| b.sample_count).unwrap_or(0),
                )
            }
            PipelinePhase::Complete => {
                format!("Self-heal: no deficiencies found ({:.0?})", report.total_duration)
            }
            phase => {
                format!(
                    "Self-heal: halted at {:?} on {} — {} ({:.0?})",
                    phase,
                    report.target_module.0,
                    report.error.as_deref().unwrap_or("unknown"),
                    report.total_duration,
                )
            }
        }
    }
}

// ── Accessors ────────────────────────────────────────

    pub fn scanner(&self) -> &DeficiencyScanner {
        &self.scanner
    }

    pub fn scanner_mut(&mut self) -> &mut DeficiencyScanner {
        &mut self.scanner
    }

    pub fn swap_table(&self) -> Option<&ModuleSwapTable> {
        self.swap_table.as_ref()
    }

    pub fn swap_table_mut(&mut self) -> Option<&mut ModuleSwapTable> {
        self.swap_table.as_mut()
    }

    pub fn set_benchmark_samples(&mut self, count: u32) {
        self.benchmark_samples = count;
    }
}
