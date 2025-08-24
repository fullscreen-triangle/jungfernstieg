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
pub mod coordination;

// Re-export main interfaces
pub use heart::{
    OscillatoryVMHeart, OscillatoryVMHeartConfig, OscillatoryVMHeartMetrics,
    CardiacPhase, PressureWave, SystolicPhaseResult, DiastolicPhaseResult
};
pub use s_entropy_bank::{
    SEntropyCentralBank, SEntropyBankConfig, SEntropyBankMetrics,
    SCreditTransaction, SCreditFlowRate, SCreditDistribution,
    EconomicState, CreditAllocationResult, EconomicSettlementResult
};
pub use coordination::{
    OscillatoryVMCoordinator, CoordinationConfig, CoordinationState,
    CirculationCycleResult, CirculationFlowResult, CirculationTracking
};

// Core types
pub use heart::HeartId;
pub use s_entropy_bank::BankId;

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
