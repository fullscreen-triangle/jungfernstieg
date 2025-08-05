//! # Biological Neural Network Management
//!
//! This crate provides biological neural network management capabilities for the
//! Jungfernstieg biological-virtual neural symbiosis system.
//!
//! ## Core Functionality
//!
//! - **Neural Culture Preparation**: Algorithms for preparing biological neural networks
//! - **Viability Assessment**: Continuous monitoring of neural health and function
//! - **Virtual Blood Interface**: Integration with Virtual Blood circulation systems
//! - **Immune Cell Monitoring**: Biological sensor networks for real-time status
//! - **Memory Cell Learning**: Adaptive optimization of neural support
//!
//! ## Architecture
//!
//! The biological system operates through careful preparation of neural networks
//! that are sustained by Virtual Blood circulation, monitored by immune cell
//! sensor networks, and optimized through memory cell learning algorithms.

pub mod neural_networks;
pub mod cell_monitoring;
pub mod interfaces;
pub mod safety;
pub mod viability;
pub mod culture;

// Re-export main interfaces
pub use neural_networks::{NeuralNetwork, NeuralNetworkManager, NeuralNetworkConfig};
pub use cell_monitoring::{ImmuneCellMonitor, MemoryCellLearner, CellMonitoringConfig};
pub use interfaces::{VirtualBloodInterface, ElectrodeInterface, PerfusionInterface};
pub use safety::{BiologicalSafetyMonitor, SafetyProtocol, BSL2PlusCompliance};
pub use viability::{ViabilityAssessment, ViabilityMonitor, ViabilityThresholds};
pub use culture::{CulturePreparation, CultureConditions, MaturationProtocol};

// Core types
pub use neural_networks::NeuralNetworkId;

/// Current version of the biological system
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_available() {
        assert!(!VERSION.is_empty());
    }
}