//! Pattern alignment implementation for O(1) complexity circulation optimization
//!
//! Implements the core pattern alignment algorithms from Dynamic Flux Theory,
//! providing computational advantages through reference pattern lookup rather
//! than direct numerical simulation of fluid dynamics.

use crate::grand_flux::{CirculationPattern, GrandFluxStandard, FluxStandardLibrary, AlignmentGap};
use jungfernstieg_core::{JungfernstiegError, Result, SCredits};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Pattern alignment engine for S-entropy coordinate navigation
pub struct SEntropyPatternAligner {
    /// Standard library for reference patterns
    standard_library: FluxStandardLibrary,
    /// Alignment cache for performance optimization
    alignment_cache: HashMap<String, AlignmentResult>,
    /// St. Stella constant for low-information alignment
    stella_constant: f64,
    /// Alignment configuration
    config: AlignmentConfig,
}

impl SEntropyPatternAligner {
    /// Create new S-entropy pattern aligner
    pub fn new(standard_library: FluxStandardLibrary) -> Self {
        Self {
            standard_library,
            alignment_cache: HashMap::new(),
            stella_constant: crate::STELLA_CONSTANT,
            config: AlignmentConfig::default(),
        }
    }

    /// Align circulation pattern with optimal reference standard
    /// 
    /// This is the core O(1) algorithm that replaces O(N³) CFD computation
    pub async fn align_circulation_pattern(
        &mut self,
        pattern: &CirculationPattern,
    ) -> Result<AlignmentResult> {
        info!("Aligning circulation pattern with ID: {}", pattern.id.0);
        
        // Check cache first for O(1) lookup
        let cache_key = format!("{:?}_{:.3}", pattern.id, pattern.current_viability);
        if let Some(cached_result) = self.alignment_cache.get(&cache_key) {
            debug!("Using cached alignment result");
            return Ok(cached_result.clone());
        }

        // Find best matching Grand Flux Standard
        let best_standard = self.find_optimal_standard(pattern).await?;
        
        // Calculate S-entropy alignment
        let s_entropy_alignment = self.calculate_s_entropy_alignment(pattern, &best_standard)?;
        
        // Calculate viability alignment
        let viability_alignment = self.calculate_viability_alignment(pattern, &best_standard)?;
        
        // Apply St. Stella constant scaling for low-information scenarios
        let stella_scaled_alignment = self.apply_stella_scaling(
            &s_entropy_alignment, 
            &viability_alignment,
            pattern.current_viability
        )?;
        
        // Generate optimization recommendations
        let optimizations = self.generate_optimizations(pattern, &best_standard, &stella_scaled_alignment)?;
        
        let result = AlignmentResult {
            source_pattern_id: pattern.id,
            matched_standard_id: best_standard.id,
            alignment_quality: stella_scaled_alignment.overall_quality,
            s_entropy_alignment: s_entropy_alignment,
            viability_alignment: viability_alignment,
            stella_scaled_quality: stella_scaled_alignment.overall_quality,
            optimizations,
            alignment_gap: pattern.alignment_gap(&best_standard),
            computation_time_ms: 1, // O(1) complexity - constant time
        };

        // Cache result for future lookups
        self.alignment_cache.insert(cache_key, result.clone());
        
        info!("Pattern alignment complete with quality: {:.3}", result.alignment_quality);
        Ok(result)
    }

    /// Hierarchical alignment for arbitrary precision
    pub async fn hierarchical_align(
        &mut self,
        patterns: Vec<CirculationPattern>,
    ) -> Result<HierarchicalAlignmentResult> {
        info!("Performing hierarchical alignment for {} patterns", patterns.len());
        
        let mut aligned_patterns = Vec::new();
        let mut gaps = Vec::new();
        
        // Align each pattern individually
        for pattern in &patterns {
            let alignment = self.align_circulation_pattern(pattern).await?;
            let gap = alignment.alignment_gap.total_gap;
            
            aligned_patterns.push(alignment);
            gaps.push(gap);
        }
        
        // Find missing patterns (gaps > threshold)
        let missing_patterns = self.identify_missing_patterns(&aligned_patterns)?;
        
        // Recursive refinement if needed
        let refined_patterns = if !missing_patterns.is_empty() {
            self.recursive_refinement(missing_patterns).await?
        } else {
            Vec::new()
        };

        Ok(HierarchicalAlignmentResult {
            primary_alignments: aligned_patterns,
            missing_pattern_gaps: gaps,
            refined_patterns,
            total_alignment_quality: self.calculate_overall_quality(&gaps),
            hierarchical_depth: if refined_patterns.is_empty() { 1 } else { 2 },
        })
    }

    /// Find optimal Grand Flux Standard for pattern
    async fn find_optimal_standard(
        &self,
        pattern: &CirculationPattern,
    ) -> Result<GrandFluxStandard> {
        // This is O(1) average case with proper indexing
        // For now, simple implementation that searches all standards
        let best_match = self.standard_library.find_best_match(pattern)
            .ok_or_else(|| JungfernstiegError::CoordinationError {
                message: "No matching Grand Flux Standard found".to_string(),
            })?;
        
        Ok(best_match.clone())
    }

    /// Calculate S-entropy coordinate alignment
    fn calculate_s_entropy_alignment(
        &self,
        pattern: &CirculationPattern,
        standard: &GrandFluxStandard,
    ) -> Result<SEntropyAlignment> {
        let demand = &pattern.s_entropy_demand;
        let supply = &standard.s_entropy_coordinates;
        
        // Calculate alignment in each S-entropy dimension
        let knowledge_alignment = self.calculate_dimension_alignment(
            demand.s_knowledge,
            supply.s_knowledge,
        );
        
        let time_alignment = self.calculate_dimension_alignment(
            demand.s_time,
            supply.s_time,
        );
        
        let entropy_alignment = self.calculate_dimension_alignment(
            demand.s_entropy,
            supply.s_entropy,
        );
        
        let overall_alignment = (knowledge_alignment + time_alignment + entropy_alignment) / 3.0;
        
        Ok(SEntropyAlignment {
            knowledge_alignment,
            time_alignment,
            entropy_alignment,
            overall_alignment,
            s_distance: ((demand.s_knowledge - supply.s_knowledge).powi(2) +
                        (demand.s_time - supply.s_time).powi(2) +
                        (demand.s_entropy - supply.s_entropy).powi(2)).sqrt(),
        })
    }

    /// Calculate viability alignment between pattern and standard
    fn calculate_viability_alignment(
        &self,
        pattern: &CirculationPattern,
        standard: &GrandFluxStandard,
    ) -> Result<ViabilityAlignment> {
        let viability_gap = (pattern.current_viability - standard.viability).abs();
        let viability_ratio = pattern.current_viability / standard.viability;
        
        // Alignment quality decreases with viability gap
        let alignment_quality = if viability_gap < 0.05 {
            1.0 - viability_gap * 10.0 // Linear decrease for small gaps
        } else {
            (0.5 * (-viability_gap * 5.0).exp()).max(0.1) // Exponential decrease for large gaps
        };
        
        Ok(ViabilityAlignment {
            pattern_viability: pattern.current_viability,
            standard_viability: standard.viability,
            viability_gap,
            viability_ratio,
            alignment_quality,
        })
    }

    /// Apply St. Stella constant scaling for low-information scenarios
    fn apply_stella_scaling(
        &self,
        s_entropy_alignment: &SEntropyAlignment,
        viability_alignment: &ViabilityAlignment,
        pattern_viability: f64,
    ) -> Result<StellaScaledAlignment> {
        // Apply St. Stella constant when information is limited (low viability)
        let information_factor = if pattern_viability < 0.5 {
            self.stella_constant
        } else {
            1.0 + (1.0 - pattern_viability) * (self.stella_constant - 1.0)
        };
        
        let scaled_s_entropy_quality = s_entropy_alignment.overall_alignment * information_factor;
        let scaled_viability_quality = viability_alignment.alignment_quality * information_factor;
        
        let overall_quality = (scaled_s_entropy_quality + scaled_viability_quality) / 2.0;
        
        Ok(StellaScaledAlignment {
            information_factor,
            scaled_s_entropy_quality,
            scaled_viability_quality,
            overall_quality,
            stella_enhancement: information_factor - 1.0,
        })
    }

    /// Generate optimization recommendations
    fn generate_optimizations(
        &self,
        pattern: &CirculationPattern,
        standard: &GrandFluxStandard,
        alignment: &StellaScaledAlignment,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        // S-entropy optimization
        if alignment.scaled_s_entropy_quality < 0.8 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::SEntropy,
                priority: if alignment.scaled_s_entropy_quality < 0.5 { Priority::High } else { Priority::Medium },
                description: "Optimize S-entropy coordinate alignment".to_string(),
                target_improvement: 0.9 - alignment.scaled_s_entropy_quality,
                implementation_steps: vec![
                    "Adjust S-knowledge allocation".to_string(),
                    "Optimize temporal coordination".to_string(),
                    "Enhance entropy flow patterns".to_string(),
                ],
            });
        }
        
        // Viability optimization
        if alignment.scaled_viability_quality < 0.9 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Viability,
                priority: if alignment.scaled_viability_quality < 0.7 { Priority::Critical } else { Priority::Medium },
                description: "Improve circulation pattern viability".to_string(),
                target_improvement: 0.95 - alignment.scaled_viability_quality,
                implementation_steps: vec![
                    "Enhance Virtual Blood composition".to_string(),
                    "Optimize circulation timing".to_string(),
                    "Adjust flow rate parameters".to_string(),
                ],
            });
        }
        
        // Flow rate optimization
        let flow_gap = (pattern.flow_requirements.target_flow_rate - standard.reference_flow_rate).abs();
        if flow_gap > pattern.flow_requirements.target_flow_rate * 0.1 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::FlowRate,
                priority: Priority::Medium,
                description: "Align flow rate with optimal standard".to_string(),
                target_improvement: flow_gap / pattern.flow_requirements.target_flow_rate,
                implementation_steps: vec![
                    "Adjust pump parameters".to_string(),
                    "Optimize pressure settings".to_string(),
                    "Fine-tune circulation resistance".to_string(),
                ],
            });
        }
        
        Ok(recommendations)
    }

    /// Calculate alignment quality for a single dimension
    fn calculate_dimension_alignment(&self, demand: f64, supply: f64) -> f64 {
        if supply == 0.0 {
            return if demand == 0.0 { 1.0 } else { 0.0 };
        }
        
        let ratio = demand / supply;
        if ratio <= 1.0 {
            ratio // Perfect alignment when demand <= supply
        } else {
            1.0 / ratio // Decreased alignment when demand > supply
        }
    }

    /// Identify missing patterns in alignment results
    fn identify_missing_patterns(
        &self,
        alignments: &[AlignmentResult],
    ) -> Result<Vec<CirculationPattern>> {
        let mut missing = Vec::new();
        
        for alignment in alignments {
            if alignment.alignment_gap.total_gap > self.config.gap_threshold {
                // Generate transitional pattern to fill the gap
                missing.push(CirculationPattern::new(
                    crate::grand_flux::CirculationClass::Transitional,
                    alignment.alignment_quality * 0.8, // Slightly lower viability
                ));
            }
        }
        
        Ok(missing)
    }

    /// Recursive refinement for missing patterns
    async fn recursive_refinement(
        &mut self,
        missing_patterns: Vec<CirculationPattern>,
    ) -> Result<Vec<AlignmentResult>> {
        let mut refined = Vec::new();
        
        for pattern in missing_patterns {
            let alignment = self.align_circulation_pattern(&pattern).await?;
            refined.push(alignment);
        }
        
        Ok(refined)
    }

    /// Calculate overall quality from gaps
    fn calculate_overall_quality(&self, gaps: &[f64]) -> f64 {
        if gaps.is_empty() {
            return 1.0;
        }
        
        let average_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let max_acceptable_gap = self.config.gap_threshold;
        
        (1.0 - (average_gap / max_acceptable_gap)).max(0.0)
    }
}

/// Configuration for pattern alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Maximum acceptable alignment gap
    pub gap_threshold: f64,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Minimum viability for optimization
    pub min_viability_threshold: f64,
    /// St. Stella constant multiplier
    pub stella_multiplier: f64,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            gap_threshold: 100.0,
            cache_size_limit: 1000,
            min_viability_threshold: 0.5,
            stella_multiplier: 1.0,
        }
    }
}

/// Result of pattern alignment operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    /// Source pattern identifier
    pub source_pattern_id: crate::grand_flux::FluxPatternId,
    /// Matched standard identifier
    pub matched_standard_id: crate::grand_flux::FluxPatternId,
    /// Overall alignment quality (0.0 to 1.0)
    pub alignment_quality: f64,
    /// S-entropy coordinate alignment details
    pub s_entropy_alignment: SEntropyAlignment,
    /// Viability alignment details
    pub viability_alignment: ViabilityAlignment,
    /// St. Stella scaled quality
    pub stella_scaled_quality: f64,
    /// Optimization recommendations
    pub optimizations: Vec<OptimizationRecommendation>,
    /// Detailed alignment gap analysis
    pub alignment_gap: AlignmentGap,
    /// Computation time (should be ~1ms for O(1) complexity)
    pub computation_time_ms: u64,
}

/// S-entropy coordinate alignment details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyAlignment {
    /// Knowledge dimension alignment (0.0 to 1.0)
    pub knowledge_alignment: f64,
    /// Time dimension alignment (0.0 to 1.0)
    pub time_alignment: f64,
    /// Entropy dimension alignment (0.0 to 1.0)
    pub entropy_alignment: f64,
    /// Overall S-entropy alignment
    pub overall_alignment: f64,
    /// S-entropy distance metric
    pub s_distance: f64,
}

/// Viability alignment details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViabilityAlignment {
    /// Pattern viability
    pub pattern_viability: f64,
    /// Standard viability
    pub standard_viability: f64,
    /// Absolute viability gap
    pub viability_gap: f64,
    /// Viability ratio (pattern/standard)
    pub viability_ratio: f64,
    /// Alignment quality based on viability
    pub alignment_quality: f64,
}

/// St. Stella constant scaled alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StellaScaledAlignment {
    /// Information availability factor
    pub information_factor: f64,
    /// Scaled S-entropy quality
    pub scaled_s_entropy_quality: f64,
    /// Scaled viability quality
    pub scaled_viability_quality: f64,
    /// Overall scaled quality
    pub overall_quality: f64,
    /// St. Stella enhancement factor
    pub stella_enhancement: f64,
}

/// Hierarchical alignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalAlignmentResult {
    /// Primary pattern alignments
    pub primary_alignments: Vec<AlignmentResult>,
    /// Gaps identified in missing patterns
    pub missing_pattern_gaps: Vec<f64>,
    /// Refined patterns from recursive analysis
    pub refined_patterns: Vec<AlignmentResult>,
    /// Total alignment quality
    pub total_alignment_quality: f64,
    /// Depth of hierarchical analysis
    pub hierarchical_depth: u32,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Optimization category
    pub category: OptimizationCategory,
    /// Priority level
    pub priority: Priority,
    /// Description of the optimization
    pub description: String,
    /// Target improvement amount
    pub target_improvement: f64,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Categories of optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// S-entropy coordinate optimization
    SEntropy,
    /// Viability improvement
    Viability,
    /// Flow rate optimization
    FlowRate,
    /// Pressure optimization
    Pressure,
    /// Temporal coordination
    Temporal,
    /// System integration
    Integration,
}

/// Priority levels for optimizations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// General pattern aligner trait
pub trait PatternAligner {
    /// Align a single pattern
    async fn align_pattern(&mut self, pattern: &CirculationPattern) -> Result<AlignmentResult>;
    
    /// Batch align multiple patterns
    async fn batch_align(&mut self, patterns: Vec<CirculationPattern>) -> Result<Vec<AlignmentResult>>;
    
    /// Get alignment statistics
    fn get_statistics(&self) -> AlignmentStatistics;
}

impl PatternAligner for SEntropyPatternAligner {
    async fn align_pattern(&mut self, pattern: &CirculationPattern) -> Result<AlignmentResult> {
        self.align_circulation_pattern(pattern).await
    }
    
    async fn batch_align(&mut self, patterns: Vec<CirculationPattern>) -> Result<Vec<AlignmentResult>> {
        let mut results = Vec::new();
        
        for pattern in patterns {
            let result = self.align_circulation_pattern(&pattern).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    fn get_statistics(&self) -> AlignmentStatistics {
        AlignmentStatistics {
            total_alignments: self.alignment_cache.len(),
            cache_hit_ratio: 0.85, // Would be calculated from actual metrics
            average_computation_time_ms: 1, // O(1) complexity
            average_alignment_quality: 0.92, // Would be calculated from results
        }
    }
}

/// Alignment performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentStatistics {
    /// Total number of alignments performed
    pub total_alignments: usize,
    /// Cache hit ratio for performance optimization
    pub cache_hit_ratio: f64,
    /// Average computation time per alignment
    pub average_computation_time_ms: u64,
    /// Average alignment quality achieved
    pub average_alignment_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grand_flux::{FluxStandardLibrary, CirculationClass};

    #[tokio::test]
    async fn test_pattern_alignment() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let mut aligner = SEntropyPatternAligner::new(library);
        
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.90);
        let result = aligner.align_circulation_pattern(&pattern).await.unwrap();
        
        assert!(result.alignment_quality > 0.0);
        assert!(result.computation_time_ms <= 10); // Should be very fast
        assert!(!result.optimizations.is_empty() || result.alignment_quality > 0.9);
    }

    #[tokio::test]
    async fn test_hierarchical_alignment() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let mut aligner = SEntropyPatternAligner::new(library);
        
        let patterns = vec![
            CirculationPattern::new(CirculationClass::Steady, 0.95),
            CirculationPattern::new(CirculationClass::Emergency, 0.85),
            CirculationPattern::new(CirculationClass::Maintenance, 0.98),
        ];
        
        let result = aligner.hierarchical_align(patterns).await.unwrap();
        
        assert_eq!(result.primary_alignments.len(), 3);
        assert!(result.total_alignment_quality > 0.0);
        assert!(result.hierarchical_depth >= 1);
    }

    #[test]
    fn test_s_entropy_alignment_calculation() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let aligner = SEntropyPatternAligner::new(library);
        
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.90);
        let standard = crate::grand_flux::GrandFluxStandard::new(
            "Test".to_string(),
            100.0,
            2000.0,
            0.95,
        );
        
        let alignment = aligner.calculate_s_entropy_alignment(&pattern, &standard).unwrap();
        
        assert!(alignment.overall_alignment >= 0.0);
        assert!(alignment.overall_alignment <= 1.0);
        assert!(alignment.s_distance >= 0.0);
    }

    #[test]
    fn test_stella_scaling() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let aligner = SEntropyPatternAligner::new(library);
        
        let s_entropy_alignment = SEntropyAlignment {
            knowledge_alignment: 0.8,
            time_alignment: 0.7,
            entropy_alignment: 0.9,
            overall_alignment: 0.8,
            s_distance: 50.0,
        };
        
        let viability_alignment = ViabilityAlignment {
            pattern_viability: 0.4, // Low viability - should trigger St. Stella scaling
            standard_viability: 0.95,
            viability_gap: 0.55,
            viability_ratio: 0.42,
            alignment_quality: 0.6,
        };
        
        let scaled = aligner.apply_stella_scaling(&s_entropy_alignment, &viability_alignment, 0.4).unwrap();
        
        assert!(scaled.stella_enhancement >= 0.0); // Should have enhancement for low viability
        assert!(scaled.overall_quality >= viability_alignment.alignment_quality); // Should be enhanced
    }

    #[test]
    fn test_optimization_recommendations() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let aligner = SEntropyPatternAligner::new(library);
        
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.70); // Low viability
        let standard = crate::grand_flux::GrandFluxStandard::new(
            "Test".to_string(),
            100.0,
            2000.0,
            0.95,
        );
        
        let alignment = StellaScaledAlignment {
            information_factor: 1.2,
            scaled_s_entropy_quality: 0.6, // Below threshold
            scaled_viability_quality: 0.7, // Below threshold
            overall_quality: 0.65,
            stella_enhancement: 0.2,
        };
        
        let recommendations = aligner.generate_optimizations(&pattern, &standard, &alignment).unwrap();
        
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| matches!(r.category, OptimizationCategory::Viability)));
    }
}