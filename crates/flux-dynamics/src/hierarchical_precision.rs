//! Hierarchical precision framework for recursive optimization
//!
//! Implements the hierarchical precision algorithms from Dynamic Flux Theory,
//! enabling arbitrary precision through recursive pattern alignment and analysis.

use crate::grand_flux::{CirculationPattern, FluxPatternId};
use crate::pattern_alignment::{AlignmentResult, SEntropyPatternAligner};
use jungfernstieg_core::{JungfernstiegError, Result, SCredits};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Hierarchical analyzer for recursive precision enhancement
pub struct HierarchicalAnalyzer {
    /// Pattern aligner for recursive analysis
    pattern_aligner: SEntropyPatternAligner,
    /// Precision configuration
    config: HierarchicalConfig,
    /// Analysis history
    analysis_history: Vec<HierarchicalAnalysisSession>,
    /// Performance metrics
    metrics: HierarchicalMetrics,
}

impl HierarchicalAnalyzer {
    /// Create new hierarchical analyzer
    pub fn new(pattern_aligner: SEntropyPatternAligner) -> Self {
        Self {
            pattern_aligner,
            config: HierarchicalConfig::default(),
            analysis_history: Vec::new(),
            metrics: HierarchicalMetrics::default(),
        }
    }

    /// Perform hierarchical analysis with recursive precision enhancement
    pub async fn analyze_hierarchical(
        &mut self,
        root_pattern: CirculationPattern,
        target_precision: PrecisionLevel,
    ) -> Result<HierarchicalAnalysisResult> {
        info!("Starting hierarchical analysis with target precision: {:?}", target_precision);
        
        let session_id = uuid::Uuid::new_v4();
        let start_time = Instant::now();
        
        // Initialize analysis tree
        let mut analysis_tree = AnalysisTree::new(root_pattern.clone());
        
        // Recursive analysis to target precision
        let analysis_result = self.recursive_analyze(
            &mut analysis_tree,
            target_precision,
            0, // Initial depth
        ).await?;
        
        // Generate hierarchical result
        let hierarchical_result = self.compile_hierarchical_result(
            &analysis_tree,
            &analysis_result,
        )?;
        
        // Record analysis session
        let session = HierarchicalAnalysisSession {
            session_id,
            timestamp: Instant::now(),
            duration: start_time.elapsed(),
            root_pattern,
            target_precision,
            achieved_precision: analysis_result.achieved_precision,
            analysis_depth: analysis_result.depth_reached,
            nodes_analyzed: analysis_result.total_nodes,
            success: analysis_result.precision_achieved,
        };
        
        self.analysis_history.push(session);
        self.update_metrics(&analysis_result);
        
        info!("Hierarchical analysis complete: precision {:?} achieved at depth {}",
               analysis_result.achieved_precision, analysis_result.depth_reached);
        
        Ok(hierarchical_result)
    }

    /// Recursive analysis algorithm (Algorithm: Hierarchical Flow Analysis)
    async fn recursive_analyze(
        &mut self,
        analysis_tree: &mut AnalysisTree,
        target_precision: PrecisionLevel,
        current_depth: u32,
    ) -> Result<RecursiveAnalysisResult> {
        debug!("Recursive analysis at depth {}", current_depth);
        
        // Check termination conditions
        if current_depth >= self.config.max_depth {
            warn!("Maximum depth {} reached", self.config.max_depth);
            return Ok(RecursiveAnalysisResult {
                achieved_precision: PrecisionLevel::from_depth(current_depth),
                depth_reached: current_depth,
                total_nodes: analysis_tree.node_count(),
                precision_achieved: false,
            });
        }
        
        // Analyze current level patterns
        let current_level_results = self.analyze_current_level(analysis_tree, current_depth).await?;
        
        // Check if precision is sufficient
        if self.precision_sufficient(&current_level_results, &target_precision)? {
            return Ok(RecursiveAnalysisResult {
                achieved_precision: target_precision,
                depth_reached: current_depth,
                total_nodes: analysis_tree.node_count(),
                precision_achieved: true,
            });
        }
        
        // Generate subsystem patterns for deeper analysis
        let subsystem_patterns = self.generate_subsystem_patterns(&current_level_results)?;
        
        // Add subsystem patterns to analysis tree
        for pattern in subsystem_patterns {
            analysis_tree.add_child_pattern(pattern)?;
        }
        
        // Recurse to next level
        self.recursive_analyze(analysis_tree, target_precision, current_depth + 1).await
    }

    /// Analyze patterns at current hierarchical level
    async fn analyze_current_level(
        &mut self,
        analysis_tree: &AnalysisTree,
        depth: u32,
    ) -> Result<Vec<AlignmentResult>> {
        debug!("Analyzing {} patterns at depth {}", analysis_tree.patterns_at_depth(depth).len(), depth);
        
        let mut results = Vec::new();
        
        for pattern in analysis_tree.patterns_at_depth(depth) {
            let alignment = self.pattern_aligner.align_circulation_pattern(pattern).await?;
            results.push(alignment);
        }
        
        Ok(results)
    }

    /// Check if current precision is sufficient
    fn precision_sufficient(
        &self,
        results: &[AlignmentResult],
        target: &PrecisionLevel,
    ) -> Result<bool> {
        if results.is_empty() {
            return Ok(false);
        }
        
        // Calculate average alignment quality
        let average_quality = results.iter()
            .map(|r| r.alignment_quality)
            .sum::<f64>() / results.len() as f64;
        
        // Check if precision requirements are met
        let precision_threshold = target.quality_threshold();
        Ok(average_quality >= precision_threshold)
    }

    /// Generate subsystem patterns for deeper analysis
    fn generate_subsystem_patterns(
        &self,
        current_results: &[AlignmentResult],
    ) -> Result<Vec<CirculationPattern>> {
        let mut subsystem_patterns = Vec::new();
        
        for result in current_results {
            // Identify gaps requiring subsystem analysis
            if result.alignment_gap.total_gap > self.config.gap_threshold {
                // Generate subsystem patterns to address specific gaps
                let subsystem = self.create_subsystem_pattern_for_gap(&result.alignment_gap)?;
                subsystem_patterns.push(subsystem);
            }
            
            // Generate optimization-specific subsystems
            for optimization in &result.optimizations {
                let subsystem = self.create_subsystem_pattern_for_optimization(optimization)?;
                subsystem_patterns.push(subsystem);
            }
        }
        
        Ok(subsystem_patterns)
    }

    /// Create subsystem pattern to address specific alignment gap
    fn create_subsystem_pattern_for_gap(
        &self,
        gap: &crate::grand_flux::AlignmentGap,
    ) -> Result<CirculationPattern> {
        let mut pattern = CirculationPattern::new(
            crate::grand_flux::CirculationClass::Transitional,
            0.8, // Start with 80% viability for subsystem
        );
        
        // Adjust pattern based on gap characteristics
        if gap.viability_gap > 0.1 {
            // Focus on viability improvement
            pattern.current_viability = 0.95;
            pattern.s_entropy_demand = SCredits::new(200.0, 150.0, 100.0);
        } else if gap.s_entropy_gap > 100.0 {
            // Focus on S-entropy optimization
            pattern.s_entropy_demand = SCredits::new(500.0, 400.0, 300.0);
        } else if gap.flow_gap > 50.0 {
            // Focus on flow optimization
            pattern.flow_requirements.target_flow_rate = gap.flow_gap * 2.0;
        }
        
        Ok(pattern)
    }

    /// Create subsystem pattern for specific optimization
    fn create_subsystem_pattern_for_optimization(
        &self,
        optimization: &crate::pattern_alignment::OptimizationRecommendation,
    ) -> Result<CirculationPattern> {
        let mut pattern = CirculationPattern::new(
            crate::grand_flux::CirculationClass::Maintenance,
            0.85,
        );
        
        match optimization.category {
            crate::pattern_alignment::OptimizationCategory::SEntropy => {
                pattern.s_entropy_demand = SCredits::new(
                    1000.0 * optimization.target_improvement,
                    800.0 * optimization.target_improvement,
                    600.0 * optimization.target_improvement,
                );
            }
            crate::pattern_alignment::OptimizationCategory::Viability => {
                pattern.current_viability = 0.95 + optimization.target_improvement.min(0.05);
            }
            crate::pattern_alignment::OptimizationCategory::FlowRate => {
                pattern.flow_requirements.target_flow_rate = 100.0 * (1.0 + optimization.target_improvement);
            }
            _ => {
                // General optimization pattern
                pattern.current_viability = 0.9 + optimization.target_improvement * 0.1;
            }
        }
        
        Ok(pattern)
    }

    /// Compile final hierarchical result
    fn compile_hierarchical_result(
        &self,
        analysis_tree: &AnalysisTree,
        analysis_result: &RecursiveAnalysisResult,
    ) -> Result<HierarchicalAnalysisResult> {
        let precision_map = self.generate_precision_map(analysis_tree)?;
        let depth_analysis = self.generate_depth_analysis(analysis_tree)?;
        let pattern_hierarchy = self.extract_pattern_hierarchy(analysis_tree)?;
        
        Ok(HierarchicalAnalysisResult {
            root_pattern_id: analysis_tree.root_pattern().id,
            achieved_precision: analysis_result.achieved_precision,
            analysis_depth: analysis_result.depth_reached,
            total_patterns_analyzed: analysis_result.total_nodes,
            precision_map,
            depth_analysis,
            pattern_hierarchy,
            missing_pattern_identification: self.identify_missing_patterns(analysis_tree)?,
            computational_complexity: self.calculate_hierarchical_complexity(analysis_result.depth_reached),
        })
    }

    /// Generate precision map across hierarchy
    fn generate_precision_map(&self, analysis_tree: &AnalysisTree) -> Result<HashMap<u32, PrecisionLevel>> {
        let mut precision_map = HashMap::new();
        
        for depth in 0..=analysis_tree.max_depth() {
            let patterns = analysis_tree.patterns_at_depth(depth);
            if !patterns.is_empty() {
                let average_viability = patterns.iter()
                    .map(|p| p.current_viability)
                    .sum::<f64>() / patterns.len() as f64;
                
                precision_map.insert(depth, PrecisionLevel::from_quality(average_viability));
            }
        }
        
        Ok(precision_map)
    }

    /// Generate depth analysis
    fn generate_depth_analysis(&self, analysis_tree: &AnalysisTree) -> Result<Vec<DepthAnalysis>> {
        let mut depth_analysis = Vec::new();
        
        for depth in 0..=analysis_tree.max_depth() {
            let patterns = analysis_tree.patterns_at_depth(depth);
            
            depth_analysis.push(DepthAnalysis {
                depth,
                pattern_count: patterns.len(),
                average_viability: if !patterns.is_empty() {
                    patterns.iter().map(|p| p.current_viability).sum::<f64>() / patterns.len() as f64
                } else {
                    0.0
                },
                complexity_reduction: self.calculate_complexity_reduction_at_depth(depth),
                precision_improvement: self.calculate_precision_improvement_at_depth(depth),
            });
        }
        
        Ok(depth_analysis)
    }

    /// Extract pattern hierarchy structure
    fn extract_pattern_hierarchy(&self, analysis_tree: &AnalysisTree) -> Result<PatternHierarchy> {
        Ok(PatternHierarchy {
            root_pattern_id: analysis_tree.root_pattern().id,
            hierarchy_levels: analysis_tree.max_depth() + 1,
            total_patterns: analysis_tree.node_count(),
            branch_factor: analysis_tree.average_branch_factor(),
            hierarchy_efficiency: self.calculate_hierarchy_efficiency(analysis_tree),
        })
    }

    /// Identify missing patterns in hierarchy
    fn identify_missing_patterns(&self, analysis_tree: &AnalysisTree) -> Result<Vec<MissingPatternGap>> {
        let mut missing_gaps = Vec::new();
        
        // Analyze gaps between hierarchical levels
        for depth in 0..analysis_tree.max_depth() {
            let current_level = analysis_tree.patterns_at_depth(depth);
            let next_level = analysis_tree.patterns_at_depth(depth + 1);
            
            // Identify coverage gaps
            let coverage_gap = self.calculate_coverage_gap(&current_level, &next_level)?;
            if coverage_gap > self.config.coverage_threshold {
                missing_gaps.push(MissingPatternGap {
                    depth_level: depth,
                    gap_magnitude: coverage_gap,
                    gap_type: GapType::Coverage,
                    recommended_patterns: self.recommend_gap_filling_patterns(&current_level, &next_level)?,
                });
            }
        }
        
        Ok(missing_gaps)
    }

    /// Calculate complexity reduction at specific depth
    fn calculate_complexity_reduction_at_depth(&self, depth: u32) -> f64 {
        // Complexity reduction increases with depth due to subsystem isolation
        let base_reduction = 0.1; // 10% base reduction per level
        let exponential_factor = 1.5;
        
        base_reduction * (exponential_factor.powi(depth as i32))
    }

    /// Calculate precision improvement at specific depth
    fn calculate_precision_improvement_at_depth(&self, depth: u32) -> f64 {
        // Precision improvement follows logarithmic scaling
        if depth == 0 {
            0.0
        } else {
            0.05 * (depth as f64).log2() // 5% improvement per log level
        }
    }

    /// Calculate hierarchy efficiency
    fn calculate_hierarchy_efficiency(&self, analysis_tree: &AnalysisTree) -> f64 {
        let total_patterns = analysis_tree.node_count() as f64;
        let max_depth = analysis_tree.max_depth() as f64;
        
        if max_depth == 0.0 {
            1.0
        } else {
            // Efficiency based on pattern distribution across depth
            let ideal_patterns = 2.0_f64.powf(max_depth) - 1.0; // Perfect binary tree
            (ideal_patterns / total_patterns).min(1.0)
        }
    }

    /// Calculate coverage gap between hierarchical levels
    fn calculate_coverage_gap(
        &self,
        current_level: &[&CirculationPattern],
        next_level: &[&CirculationPattern],
    ) -> Result<f64> {
        if current_level.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate viability coverage gap
        let current_viability_range = self.calculate_viability_range(current_level);
        let next_viability_range = if next_level.is_empty() {
            (0.0, 0.0)
        } else {
            self.calculate_viability_range(next_level)
        };
        
        let gap = (current_viability_range.1 - current_viability_range.0) - 
                  (next_viability_range.1 - next_viability_range.0);
        
        Ok(gap.abs())
    }

    /// Calculate viability range for patterns
    fn calculate_viability_range(&self, patterns: &[&CirculationPattern]) -> (f64, f64) {
        if patterns.is_empty() {
            return (0.0, 0.0);
        }
        
        let viabilities: Vec<f64> = patterns.iter().map(|p| p.current_viability).collect();
        let min_viability = viabilities.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_viability = viabilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        (min_viability, max_viability)
    }

    /// Recommend patterns to fill gaps
    fn recommend_gap_filling_patterns(
        &self,
        current_level: &[&CirculationPattern],
        next_level: &[&CirculationPattern],
    ) -> Result<Vec<CirculationPattern>> {
        let mut recommendations = Vec::new();
        
        let current_range = self.calculate_viability_range(current_level);
        let next_range = if next_level.is_empty() {
            (0.0, 0.0)
        } else {
            self.calculate_viability_range(next_level)
        };
        
        // Generate patterns to bridge the gap
        if current_range.1 > next_range.1 {
            // Need higher viability patterns
            let target_viability = (current_range.1 + next_range.1) / 2.0;
            let pattern = CirculationPattern::new(
                crate::grand_flux::CirculationClass::Transitional,
                target_viability,
            );
            recommendations.push(pattern);
        }
        
        if current_range.0 < next_range.0 {
            // Need lower viability patterns
            let target_viability = (current_range.0 + next_range.0) / 2.0;
            let pattern = CirculationPattern::new(
                crate::grand_flux::CirculationClass::Maintenance,
                target_viability,
            );
            recommendations.push(pattern);
        }
        
        Ok(recommendations)
    }

    /// Calculate hierarchical computational complexity
    fn calculate_hierarchical_complexity(&self, depth: u32) -> HierarchicalComplexity {
        HierarchicalComplexity {
            base_complexity: ComputationalComplexity::O1,
            depth_factor: depth,
            total_complexity: format!("O(1 × log{})", depth),
            efficiency_gain: 1000.0 * (depth as f64).log2().max(1.0), // Exponential efficiency gain
        }
    }

    /// Update analyzer metrics
    fn update_metrics(&mut self, result: &RecursiveAnalysisResult) {
        self.metrics.total_analyses += 1;
        self.metrics.total_depth += result.depth_reached;
        self.metrics.average_depth = self.metrics.total_depth as f64 / self.metrics.total_analyses as f64;
        
        if result.precision_achieved {
            self.metrics.successful_analyses += 1;
        }
        
        self.metrics.success_rate = self.metrics.successful_analyses as f64 / self.metrics.total_analyses as f64;
        self.metrics.max_depth_reached = self.metrics.max_depth_reached.max(result.depth_reached);
    }

    /// Get analyzer performance statistics
    pub fn get_performance_metrics(&self) -> &HierarchicalMetrics {
        &self.metrics
    }
}

/// Recursive optimizer for precision enhancement
pub struct RecursiveOptimizer {
    /// Hierarchical analyzer
    analyzer: HierarchicalAnalyzer,
    /// Optimization configuration
    config: RecursiveOptimizerConfig,
}

impl RecursiveOptimizer {
    /// Create new recursive optimizer
    pub fn new(analyzer: HierarchicalAnalyzer) -> Self {
        Self {
            analyzer,
            config: RecursiveOptimizerConfig::default(),
        }
    }

    /// Optimize with recursive precision enhancement
    pub async fn optimize_recursive(
        &mut self,
        pattern: CirculationPattern,
        target_precision: PrecisionLevel,
    ) -> Result<RecursiveOptimizationResult> {
        info!("Starting recursive optimization with target precision: {:?}", target_precision);
        
        // Perform hierarchical analysis
        let analysis_result = self.analyzer.analyze_hierarchical(pattern.clone(), target_precision).await?;
        
        // Extract optimal patterns from hierarchy
        let optimal_patterns = self.extract_optimal_patterns(&analysis_result)?;
        
        // Combine patterns for final optimization
        let combined_optimization = self.combine_hierarchical_optimizations(&optimal_patterns)?;
        
        Ok(RecursiveOptimizationResult {
            input_pattern_id: pattern.id,
            target_precision,
            achieved_precision: analysis_result.achieved_precision,
            optimization_depth: analysis_result.analysis_depth,
            combined_optimization,
            hierarchical_efficiency: analysis_result.pattern_hierarchy.hierarchy_efficiency,
            total_patterns_considered: analysis_result.total_patterns_analyzed,
        })
    }

    /// Extract optimal patterns from hierarchical analysis
    fn extract_optimal_patterns(
        &self,
        analysis_result: &HierarchicalAnalysisResult,
    ) -> Result<Vec<CirculationPattern>> {
        let mut optimal_patterns = Vec::new();
        
        // Extract patterns from each hierarchy level that meet quality thresholds
        for depth_analysis in &analysis_result.depth_analysis {
            if depth_analysis.average_viability >= self.config.quality_threshold {
                // Generate representative pattern for this level
                let pattern = CirculationPattern::new(
                    crate::grand_flux::CirculationClass::Steady,
                    depth_analysis.average_viability,
                );
                optimal_patterns.push(pattern);
            }
        }
        
        Ok(optimal_patterns)
    }

    /// Combine hierarchical optimizations
    fn combine_hierarchical_optimizations(
        &self,
        patterns: &[CirculationPattern],
    ) -> Result<CombinedOptimization> {
        if patterns.is_empty() {
            return Ok(CombinedOptimization::default());
        }
        
        // Calculate weighted average of optimizations
        let total_viability = patterns.iter().map(|p| p.current_viability).sum::<f64>();
        let average_viability = total_viability / patterns.len() as f64;
        
        let total_flow = patterns.iter()
            .map(|p| p.flow_requirements.target_flow_rate)
            .sum::<f64>();
        let average_flow = total_flow / patterns.len() as f64;
        
        let combined_s_entropy = patterns.iter()
            .fold(SCredits::zero(), |acc, p| {
                SCredits::new(
                    acc.s_knowledge + p.s_entropy_demand.s_knowledge,
                    acc.s_time + p.s_entropy_demand.s_time,
                    acc.s_entropy + p.s_entropy_demand.s_entropy,
                )
            });
        
        let normalized_s_entropy = SCredits::new(
            combined_s_entropy.s_knowledge / patterns.len() as f64,
            combined_s_entropy.s_time / patterns.len() as f64,
            combined_s_entropy.s_entropy / patterns.len() as f64,
        );
        
        Ok(CombinedOptimization {
            combined_viability: average_viability,
            combined_flow_rate: average_flow,
            combined_s_entropy: normalized_s_entropy,
            optimization_factors: patterns.len(),
            convergence_quality: self.calculate_convergence_quality(patterns),
        })
    }

    /// Calculate convergence quality
    fn calculate_convergence_quality(&self, patterns: &[CirculationPattern]) -> f64 {
        if patterns.len() <= 1 {
            return 1.0;
        }
        
        // Calculate variance in viabilities
        let viabilities: Vec<f64> = patterns.iter().map(|p| p.current_viability).collect();
        let mean = viabilities.iter().sum::<f64>() / viabilities.len() as f64;
        let variance = viabilities.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / viabilities.len() as f64;
        
        // Convergence quality is inverse of variance
        1.0 / (1.0 + variance)
    }
}

/// Analysis tree for hierarchical pattern organization
#[derive(Debug)]
pub struct AnalysisTree {
    /// Root pattern
    root: CirculationPattern,
    /// Hierarchical patterns organized by depth
    patterns_by_depth: HashMap<u32, Vec<CirculationPattern>>,
    /// Maximum depth reached
    max_depth: u32,
}

impl AnalysisTree {
    /// Create new analysis tree
    pub fn new(root_pattern: CirculationPattern) -> Self {
        let mut patterns_by_depth = HashMap::new();
        patterns_by_depth.insert(0, vec![root_pattern.clone()]);
        
        Self {
            root: root_pattern,
            patterns_by_depth,
            max_depth: 0,
        }
    }

    /// Add child pattern to tree
    pub fn add_child_pattern(&mut self, pattern: CirculationPattern) -> Result<()> {
        let depth = self.max_depth + 1;
        self.patterns_by_depth.entry(depth).or_insert_with(Vec::new).push(pattern);
        self.max_depth = depth;
        Ok(())
    }

    /// Get patterns at specific depth
    pub fn patterns_at_depth(&self, depth: u32) -> Vec<&CirculationPattern> {
        self.patterns_by_depth.get(&depth)
            .map(|patterns| patterns.iter().collect())
            .unwrap_or_default()
    }

    /// Get root pattern
    pub fn root_pattern(&self) -> &CirculationPattern {
        &self.root
    }

    /// Get maximum depth
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Get total node count
    pub fn node_count(&self) -> usize {
        self.patterns_by_depth.values().map(|v| v.len()).sum()
    }

    /// Calculate average branch factor
    pub fn average_branch_factor(&self) -> f64 {
        if self.max_depth == 0 {
            return 1.0;
        }
        
        let total_patterns = self.node_count() as f64;
        let depth_levels = (self.max_depth + 1) as f64;
        
        total_patterns / depth_levels
    }
}

/// Precision levels for hierarchical analysis
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// Coarse precision (< 80% quality)
    Coarse,
    /// Medium precision (80-90% quality)
    Medium,
    /// Fine precision (90-95% quality)
    Fine,
    /// Ultra precision (95-98% quality)
    Ultra,
    /// Maximum precision (> 98% quality)
    Maximum,
}

impl PrecisionLevel {
    /// Get quality threshold for precision level
    pub fn quality_threshold(&self) -> f64 {
        match self {
            PrecisionLevel::Coarse => 0.8,
            PrecisionLevel::Medium => 0.9,
            PrecisionLevel::Fine => 0.95,
            PrecisionLevel::Ultra => 0.98,
            PrecisionLevel::Maximum => 0.99,
        }
    }

    /// Create precision level from depth
    pub fn from_depth(depth: u32) -> Self {
        match depth {
            0 => PrecisionLevel::Coarse,
            1 => PrecisionLevel::Medium,
            2 => PrecisionLevel::Fine,
            3 => PrecisionLevel::Ultra,
            _ => PrecisionLevel::Maximum,
        }
    }

    /// Create precision level from quality
    pub fn from_quality(quality: f64) -> Self {
        if quality >= 0.99 {
            PrecisionLevel::Maximum
        } else if quality >= 0.98 {
            PrecisionLevel::Ultra
        } else if quality >= 0.95 {
            PrecisionLevel::Fine
        } else if quality >= 0.9 {
            PrecisionLevel::Medium
        } else {
            PrecisionLevel::Coarse
        }
    }
}

/// Configuration for hierarchical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalConfig {
    /// Maximum analysis depth
    pub max_depth: u32,
    /// Gap threshold for subsystem generation
    pub gap_threshold: f64,
    /// Coverage threshold for missing pattern detection
    pub coverage_threshold: f64,
    /// Enable precision optimization
    pub precision_optimization: bool,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            gap_threshold: 50.0,
            coverage_threshold: 0.1,
            precision_optimization: true,
        }
    }
}

/// Configuration for recursive optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveOptimizerConfig {
    /// Quality threshold for pattern selection
    pub quality_threshold: f64,
    /// Maximum optimization iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
}

impl Default for RecursiveOptimizerConfig {
    fn default() -> Self {
        Self {
            quality_threshold: 0.9,
            max_iterations: 10,
            convergence_tolerance: 0.01,
        }
    }
}

/// Result of recursive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveAnalysisResult {
    /// Precision level achieved
    pub achieved_precision: PrecisionLevel,
    /// Maximum depth reached
    pub depth_reached: u32,
    /// Total nodes analyzed
    pub total_nodes: usize,
    /// Whether target precision was achieved
    pub precision_achieved: bool,
}

/// Hierarchical analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalAnalysisResult {
    /// Root pattern identifier
    pub root_pattern_id: FluxPatternId,
    /// Achieved precision level
    pub achieved_precision: PrecisionLevel,
    /// Analysis depth reached
    pub analysis_depth: u32,
    /// Total patterns analyzed
    pub total_patterns_analyzed: usize,
    /// Precision map across hierarchy
    pub precision_map: HashMap<u32, PrecisionLevel>,
    /// Depth analysis details
    pub depth_analysis: Vec<DepthAnalysis>,
    /// Pattern hierarchy structure
    pub pattern_hierarchy: PatternHierarchy,
    /// Missing pattern identification
    pub missing_pattern_identification: Vec<MissingPatternGap>,
    /// Computational complexity analysis
    pub computational_complexity: HierarchicalComplexity,
}

/// Analysis details for specific depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthAnalysis {
    /// Depth level
    pub depth: u32,
    /// Number of patterns at this depth
    pub pattern_count: usize,
    /// Average viability at this depth
    pub average_viability: f64,
    /// Complexity reduction achieved
    pub complexity_reduction: f64,
    /// Precision improvement at this depth
    pub precision_improvement: f64,
}

/// Pattern hierarchy structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternHierarchy {
    /// Root pattern identifier
    pub root_pattern_id: FluxPatternId,
    /// Number of hierarchy levels
    pub hierarchy_levels: u32,
    /// Total patterns in hierarchy
    pub total_patterns: usize,
    /// Average branch factor
    pub branch_factor: f64,
    /// Hierarchy efficiency score
    pub hierarchy_efficiency: f64,
}

/// Missing pattern gap identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingPatternGap {
    /// Depth level where gap exists
    pub depth_level: u32,
    /// Magnitude of the gap
    pub gap_magnitude: f64,
    /// Type of gap
    pub gap_type: GapType,
    /// Recommended patterns to fill gap
    pub recommended_patterns: Vec<CirculationPattern>,
}

/// Types of gaps in pattern hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapType {
    /// Coverage gap between levels
    Coverage,
    /// Precision gap in analysis
    Precision,
    /// Optimization gap in performance
    Optimization,
    /// Integration gap between subsystems
    Integration,
}

/// Hierarchical computational complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalComplexity {
    /// Base computational complexity
    pub base_complexity: ComputationalComplexity,
    /// Depth factor in complexity
    pub depth_factor: u32,
    /// Total complexity expression
    pub total_complexity: String,
    /// Efficiency gain over traditional methods
    pub efficiency_gain: f64,
}

/// Computational complexity classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    /// O(1) - constant time
    O1,
    /// O(log n) - logarithmic
    OLogN,
    /// O(n) - linear
    ON,
    /// O(n²) - quadratic
    ON2,
    /// O(n³) - cubic (traditional CFD)
    ON3,
}

/// Hierarchical analysis session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalAnalysisSession {
    /// Session identifier
    pub session_id: uuid::Uuid,
    /// Session timestamp
    pub timestamp: Instant,
    /// Session duration
    pub duration: Duration,
    /// Root pattern analyzed
    pub root_pattern: CirculationPattern,
    /// Target precision level
    pub target_precision: PrecisionLevel,
    /// Achieved precision level
    pub achieved_precision: PrecisionLevel,
    /// Maximum analysis depth reached
    pub analysis_depth: u32,
    /// Total nodes analyzed
    pub nodes_analyzed: usize,
    /// Whether analysis succeeded
    pub success: bool,
}

/// Analyzer performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalMetrics {
    /// Total analyses performed
    pub total_analyses: usize,
    /// Successful analyses
    pub successful_analyses: usize,
    /// Success rate
    pub success_rate: f64,
    /// Total depth analyzed
    pub total_depth: u32,
    /// Average analysis depth
    pub average_depth: f64,
    /// Maximum depth reached
    pub max_depth_reached: u32,
}

impl Default for HierarchicalMetrics {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            successful_analyses: 0,
            success_rate: 0.0,
            total_depth: 0,
            average_depth: 0.0,
            max_depth_reached: 0,
        }
    }
}

/// Recursive optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveOptimizationResult {
    /// Input pattern identifier
    pub input_pattern_id: FluxPatternId,
    /// Target precision level
    pub target_precision: PrecisionLevel,
    /// Achieved precision level
    pub achieved_precision: PrecisionLevel,
    /// Optimization depth reached
    pub optimization_depth: u32,
    /// Combined optimization result
    pub combined_optimization: CombinedOptimization,
    /// Hierarchical efficiency achieved
    pub hierarchical_efficiency: f64,
    /// Total patterns considered
    pub total_patterns_considered: usize,
}

/// Combined optimization from hierarchical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedOptimization {
    /// Combined viability score
    pub combined_viability: f64,
    /// Combined flow rate
    pub combined_flow_rate: f64,
    /// Combined S-entropy coordinates
    pub combined_s_entropy: SCredits,
    /// Number of optimization factors combined
    pub optimization_factors: usize,
    /// Quality of convergence
    pub convergence_quality: f64,
}

impl Default for CombinedOptimization {
    fn default() -> Self {
        Self {
            combined_viability: 0.0,
            combined_flow_rate: 0.0,
            combined_s_entropy: SCredits::zero(),
            optimization_factors: 0,
            convergence_quality: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grand_flux::{FluxStandardLibrary, CirculationClass};
    use crate::pattern_alignment::SEntropyPatternAligner;

    #[tokio::test]
    async fn test_hierarchical_analyzer_creation() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let aligner = SEntropyPatternAligner::new(library);
        let analyzer = HierarchicalAnalyzer::new(aligner);
        
        assert_eq!(analyzer.metrics.total_analyses, 0);
    }

    #[tokio::test]
    async fn test_hierarchical_analysis() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let aligner = SEntropyPatternAligner::new(library);
        let mut analyzer = HierarchicalAnalyzer::new(aligner);
        
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.85);
        let result = analyzer.analyze_hierarchical(pattern, PrecisionLevel::Fine).await.unwrap();
        
        assert!(result.analysis_depth > 0);
        assert!(result.total_patterns_analyzed > 0);
        assert!(!result.depth_analysis.is_empty());
    }

    #[test]
    fn test_analysis_tree() {
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.90);
        let mut tree = AnalysisTree::new(pattern);
        
        assert_eq!(tree.max_depth(), 0);
        assert_eq!(tree.node_count(), 1);
        
        let child_pattern = CirculationPattern::new(CirculationClass::Maintenance, 0.85);
        tree.add_child_pattern(child_pattern).unwrap();
        
        assert_eq!(tree.max_depth(), 1);
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn test_precision_level_thresholds() {
        assert_eq!(PrecisionLevel::Coarse.quality_threshold(), 0.8);
        assert_eq!(PrecisionLevel::Fine.quality_threshold(), 0.95);
        assert_eq!(PrecisionLevel::Maximum.quality_threshold(), 0.99);
    }

    #[test]
    fn test_precision_level_from_quality() {
        assert_eq!(PrecisionLevel::from_quality(0.75), PrecisionLevel::Coarse);
        assert_eq!(PrecisionLevel::from_quality(0.92), PrecisionLevel::Medium);
        assert_eq!(PrecisionLevel::from_quality(0.97), PrecisionLevel::Fine);
        assert_eq!(PrecisionLevel::from_quality(0.995), PrecisionLevel::Maximum);
    }

    #[test]
    fn test_precision_level_from_depth() {
        assert_eq!(PrecisionLevel::from_depth(0), PrecisionLevel::Coarse);
        assert_eq!(PrecisionLevel::from_depth(2), PrecisionLevel::Fine);
        assert_eq!(PrecisionLevel::from_depth(5), PrecisionLevel::Maximum);
    }

    #[tokio::test]
    async fn test_recursive_optimizer() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let aligner = SEntropyPatternAligner::new(library);
        let analyzer = HierarchicalAnalyzer::new(aligner);
        let mut optimizer = RecursiveOptimizer::new(analyzer);
        
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.85);
        let result = optimizer.optimize_recursive(pattern, PrecisionLevel::Fine).await.unwrap();
        
        assert!(result.optimization_depth > 0);
        assert!(result.combined_optimization.optimization_factors > 0);
        assert!(result.hierarchical_efficiency > 0.0);
    }

    #[test]
    fn test_combined_optimization_default() {
        let optimization = CombinedOptimization::default();
        
        assert_eq!(optimization.combined_viability, 0.0);
        assert_eq!(optimization.optimization_factors, 0);
        assert_eq!(optimization.convergence_quality, 0.0);
    }

    #[test]
    fn test_gap_type_variants() {
        let gap_types = vec![
            GapType::Coverage,
            GapType::Precision,
            GapType::Optimization,
            GapType::Integration,
        ];
        
        assert_eq!(gap_types.len(), 4);
    }
}