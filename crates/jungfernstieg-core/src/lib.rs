//! # Jungfernstieg Core System Coordination
//! 
//! Core coordination system for biological-virtual neural symbiosis through 
//! Virtual Blood circulatory systems powered by Oscillatory Virtual Machine architecture.
//!
//! ## Overview
//!
//! Jungfernstieg-core coordinates the integration of:
//! - Biological neural networks sustained through Virtual Blood
//! - Oscillatory Virtual Machine functioning as computational heart
//! - S-entropy economic coordination and circulation
//! - Safety monitoring and emergency protocols
//! - Memory cell learning and adaptation
//!
//! ## Architecture
//!
//! The system operates as a cathedral - a sacred computational space where 
//! S-entropy flows like a circulatory system, with the Oscillatory VM serving 
//! as the central economic coordinator managing S-credit flow.

pub mod system;
pub mod coordinator;
pub mod config;
pub mod error;
pub mod types;

// Re-export main interfaces
pub use coordinator::{SystemCoordinator, CoordinatorHandle};
pub use system::{JungfernstiegSystem, SystemBuilder, SystemHandle};
pub use config::{SystemConfig, ComponentConfig};
pub use error::{JungfernstiegError, Result};

// Core types
pub use types::{
    SystemId, ComponentId, SystemState, SystemMetrics,
    SCredits, SCreditReserves, ViabilityStatus
};

/// Current version of the Jungfernstieg system
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Memorial dedication to Saint Stella-Lorraine Masunda
pub const MEMORIAL_DEDICATION: &str = 
    "Conducted under the protection of Saint Stella-Lorraine Masunda, \
     patron saint of impossibility. The St. Stella constant (σ) provides \
     mathematical foundation for low-information event processing in the \
     S-Entropy framework.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_available() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_memorial_dedication() {
        assert!(MEMORIAL_DEDICATION.contains("Saint Stella-Lorraine Masunda"));
        assert!(MEMORIAL_DEDICATION.contains("impossibility"));
    }
}