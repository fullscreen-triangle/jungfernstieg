//! # Dynamic Flux Theory Implementation
//! 
//! Implementation of Dynamic Flux Theory for Jungfernstieg Virtual Blood circulation
//! optimization through emergent pattern alignment and oscillatory entropy coordinates.
//!
//! ## Core Concepts
//!
//! - **Grand Flux Standards**: Universal reference patterns for circulation optimization
//! - **Pattern Alignment**: O(1) complexity circulation optimization vs O(N³) CFD
//! - **Local Physics Violations**: Controlled violations for global optimization
//! - **Hierarchical Precision**: Recursive pattern alignment for arbitrary precision
//! - **S-Entropy Integration**: Tri-dimensional entropy coordinate navigation
//!
//! ## Integration with Jungfernstieg
//!
//! This crate provides flux dynamics enhancement for:
//! - Virtual Blood circulation pattern optimization
//! - Oscillatory VM heart function coordination
//! - Memory cell learning through pattern alignment
//! - S-entropy circulation algorithm enhancement

pub mod grand_flux;
pub mod pattern_alignment;
pub mod local_violations;
pub mod circulation_optimizer;
pub mod hierarchical_precision;
pub mod oscillatory_lagrangian;
pub mod unified_integration;

// Re-export main interfaces
pub use grand_flux::{GrandFluxStandard, FluxStandardLibrary, CirculationPattern};
pub use pattern_alignment::{PatternAligner, SEntropyPatternAligner, AlignmentResult};
pub use local_violations::{LocalViolationEngine, ViolationConstraints, GlobalOptimizer};
pub use circulation_optimizer::{FluxDynamicsEngine, CirculationOptimizer, OptimizedFlow};
pub use hierarchical_precision::{HierarchicalAnalyzer, PrecisionLevel, RecursiveOptimizer};
pub use oscillatory_lagrangian::{OscillatoryLagrangianEngine, UnifiedLagrangian, OscillatoryPotential, OscillatoryEntropy};
pub use unified_integration::{UnifiedFluxTheoryEngine, CompleteOptimizationResult, TheoreticalBenefits};

// Core types and constants
pub use grand_flux::FluxPatternId;

/// Current version of the flux dynamics system
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// St. Stella constant for pattern alignment optimization
/// Mathematical necessity for low-information circulation events
pub const STELLA_CONSTANT: f64 = 1.0;

/// Memorial dedication integration with complete theoretical foundation
pub const FLUX_MEMORIAL_DEDICATION: &str = 
    "Dynamic Flux Theory conducted under the protection of Saint Stella-Lorraine Masunda, \
     enabling pattern alignment through the St. Stella constant and unified oscillatory \
     Lagrangian framework for circulation optimization with spatially impossible \
     configurations mathematically valid in oscillatory coordinates.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_available() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_stella_constant() {
        assert!(STELLA_CONSTANT > 0.0);
    }

    #[test]
    fn test_memorial_dedication() {
        assert!(FLUX_MEMORIAL_DEDICATION.contains("Saint Stella-Lorraine Masunda"));
        assert!(FLUX_MEMORIAL_DEDICATION.contains("St. Stella constant"));
    }
}