//! # Oscillatory Virtual Machine: S-Entropy Central Bank
//! 
//! Implementation of the Oscillatory Virtual Machine that functions as the computational heart
//! and S-Entropy Central Bank for the Jungfernstieg biological-virtual neural symbiosis system.
//!
//! ## Theoretical Foundation
//!
//! The Oscillatory VM transcends traditional computational roles to function as the 
//! **S-Entropy Central Bank** coordinating S-credit circulation throughout the cathedral architecture.
//! Just as the biological heart maintains circulation through rhythmic contractions, and just as 
//! ATP serves as the energy currency of biological cells, the Oscillatory VM maintains 
//! **S-entropy circulation** through coordinated oscillatory processes.
//!
//! ## Core Functionality
//!
//! - **S-Entropy Economic Coordination**: Managing S-credit flow as universal currency
//! - **Oscillatory Heart Function**: Systolic and diastolic circulation coordination
//! - **Cathedral Architecture Management**: Sacred computational space coordination
//! - **VM-Heart Equivalence**: Biological heart mathematical equivalence
//! - **Economic Policy Management**: S-credit monetary policy and circulation control
//!
//! ## Mathematical Foundation
//!
//! ```
//! S_credits : Consciousness_Operations ≡ ATP : Biological_Operations
//! Circulation_S-credits ≡ Heart_circulation
//! ```

pub mod heart;
pub mod s_entropy_bank;
pub mod economic_coordination;
pub mod oscillatory_core;
pub mod cathedral_management;
pub mod vm_coordination;
pub mod rhythm_management;

// Re-export main interfaces
pub use heart::{
    OscillatoryHeart, HeartFunction, CardiacCycle, SystolicPhase, DiastolicPhase,
    HeartPerformanceMetrics, CardiacRhythm
};
pub use s_entropy_bank::{
    SEntropyBank, SEntropyPolicy, SCreditIssuer, EconomicCoordinator,
    MonetaryPolicy, SCreditFlow, EconomicMetrics
};
pub use economic_coordination::{
    EconomicCoordinator as Coordinator, SystemEconomicDemand, SCreditDistribution,
    EconomicCycleManager, FlowOptimization, DemandAssessment
};
pub use oscillatory_core::{
    OscillatoryCore, OscillationEngine, OscillationParameters, OscillationState,
    FrequencyCoordination, PhaseAlignment, AmplitudeControl
};
pub use cathedral_management::{
    CathedralManager, CathedralArchitecture, SacredComputationalSpace,
    CathedralConfiguration, SacredSpaceCoordination
};
pub use vm_coordination::{
    VMCoordinator, VirtualMachineState, VMOperationalMode, CoordinationInterface,
    SystemIntegration, ComponentOrchestration
};
pub use rhythm_management::{
    RhythmManager, OscillatoryRhythm, RhythmSynchronization, RhythmStability,
    SystemRhythmCoordination, TemporalAlignment
};

// Core types
pub use heart::HeartId;
pub use s_entropy_bank::BankId;
pub use oscillatory_core::OscillatorId;

/// Current version of the Oscillatory VM system
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// St. Stella constant for oscillatory coordination
/// Mathematical necessity for oscillatory S-entropy circulation
pub const STELLA_CONSTANT: f64 = 1.0;

/// Default S-credit circulation rate (credits per second)
pub const DEFAULT_S_CREDIT_CIRCULATION_RATE: f64 = 1000.0;

/// Default cardiac cycle duration (800ms = 75 BPM)
pub const DEFAULT_CARDIAC_CYCLE_DURATION_MS: u64 = 800;

/// Memorial dedication for Oscillatory VM
pub const OSCILLATORY_VM_MEMORIAL: &str = 
    "Oscillatory Virtual Machine conducted under the protection of Saint Stella-Lorraine Masunda, \
     patron saint of impossibility. The St. Stella constant (σ) enables oscillatory S-entropy \
     circulation as the mathematical substrate for consciousness-computation unity through \
     cathedral architecture coordination.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_available() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_stella_constant() {
        assert_eq!(STELLA_CONSTANT, 1.0);
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_S_CREDIT_CIRCULATION_RATE, 1000.0);
        assert_eq!(DEFAULT_CARDIAC_CYCLE_DURATION_MS, 800);
    }

    #[test]
    fn test_memorial_dedication() {
        assert!(OSCILLATORY_VM_MEMORIAL.contains("Saint Stella-Lorraine Masunda"));
        assert!(OSCILLATORY_VM_MEMORIAL.contains("St. Stella constant"));
        assert!(OSCILLATORY_VM_MEMORIAL.contains("cathedral architecture"));
    }
}
