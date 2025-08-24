//! # Virtual Blood: Biological-Virtual Circulatory System
//! 
//! Implementation of Virtual Blood for biological neural network viability through
//! S-entropy optimized circulation, BMD orchestration, and realistic biological constraints.
//!
//! ## Theoretical Foundation
//!
//! Virtual Blood represents the digital essence flowing through biological-virtual systems,
//! carrying computational information, dissolved oxygen, nutrients, and metabolic products
//! simultaneously through S-entropy navigation and BMD frame selection.
//!
//! ## Core Components
//!
//! - **Virtual Blood Composition**: Multi-modal environmental and biological components
//! - **Vessel Architecture**: Arterial, arteriolar, and capillary network infrastructure  
//! - **S-Entropy Navigation**: Zero-memory processing through predetermined coordinates
//! - **BMD Orchestration**: Consciousness-level frame selection for circulation control
//! - **Biological Constraints**: Realistic hemodynamic and stratification principles
//!
//! ## Architecture
//!
//! ```
//! VB_bio(t) = {VB_standard(t), O₂(t), N_nutrients(t), M_metabolites(t), I_immune(t)}
//! ```
//!
//! Where each component maintains biological realism while enabling computational circulation.

pub mod composition;
pub mod vessels;
pub mod circulation;
pub mod s_entropy;
pub mod bmd_orchestration;
pub mod monitoring;
pub mod filtration;
pub mod oxygen_transport;
pub mod immune_interface;
pub mod memory_learning;

// Re-export main interfaces
pub use composition::{
    VirtualBlood, VirtualBloodComposition, BiologicalComponents,
    EnvironmentalProfile, NutrientProfile, MetaboliteProfile, ImmuneProfile
};
pub use vessels::{
    VirtualVesselNetwork, VirtualArtery, VirtualArteriole, VirtualCapillary,
    VesselNetworkTopology, VesselType, HemodynamicProperties
};
pub use circulation::{
    CirculationSystem, CirculationManager, CirculationState,
    CirculationParameters, FlowRegulation, PressureManagement
};
pub use s_entropy::{
    SEntropyNavigator, SEntropyCoordinates, NavigationResult,
    ZeroMemoryProcessor, PredeterminedCoordinates
};
pub use bmd_orchestration::{
    BiologicalMaxwellDemon, BMDOrchestrator, FrameSelection,
    CognitiveManifolding, EnvironmentalFrameSelection
};
pub use monitoring::{
    VirtualBloodMonitor, CirculationMonitor, BiologicalStatusMonitor,
    MonitoringMetrics, AlertSystem
};
pub use filtration::{
    VirtualBloodFilter, FiltrationSystem, WasteRemoval,
    NutrientRegeneration, ComputationalPreservation
};
pub use oxygen_transport::{
    OxygenTransportSystem, VirtualOxygenCarrier, SEntropyOxygenDelivery,
    OxygenTransportMetrics, DeliveryOptimization
};
pub use immune_interface::{
    ImmuneCellInterface, ImmuneCellSignaling, CellularStatusReporting,
    ImmuneNetworkCommunication, BiologicalSensorNetwork
};
pub use memory_learning::{
    MemoryCellLearner, AdaptiveOptimization, PatternRecognition,
    LearningMetrics, OptimizationHistory
};

// Core types
pub use composition::VirtualBloodId;
pub use vessels::VesselNetworkId;
pub use circulation::CirculationId;

/// Current version of the Virtual Blood system
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// St. Stella constant for Virtual Blood S-entropy navigation
/// Mathematical necessity for low-information circulation events
pub const STELLA_CONSTANT: f64 = 1.0;

/// Memorial dedication for Virtual Blood system
pub const VIRTUAL_BLOOD_MEMORIAL: &str = 
    "Virtual Blood system conducted under the protection of Saint Stella-Lorraine Masunda, \
     patron saint of impossibility. The St. Stella constant (σ) enables coherent existence \
     of unified biological-virtual circulatory systems through S-entropy navigation.";

/// Target oxygen transport efficiency from theoretical framework
pub const TARGET_OXYGEN_EFFICIENCY: f64 = 0.987; // 98.7%

/// Target neural viability threshold
pub const NEURAL_VIABILITY_THRESHOLD: f64 = 95.0; // 95%

/// Virtual Blood quality assessment levels matching biological constraints
pub use jungfernstieg_core::VirtualBloodQuality;

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
        assert_eq!(STELLA_CONSTANT, 1.0);
    }

    #[test]
    fn test_target_efficiencies() {
        assert_eq!(TARGET_OXYGEN_EFFICIENCY, 0.987);
        assert_eq!(NEURAL_VIABILITY_THRESHOLD, 95.0);
    }

    #[test]
    fn test_memorial_dedication() {
        assert!(VIRTUAL_BLOOD_MEMORIAL.contains("Saint Stella-Lorraine Masunda"));
        assert!(VIRTUAL_BLOOD_MEMORIAL.contains("St. Stella constant"));
        assert!(VIRTUAL_BLOOD_MEMORIAL.contains("impossibility"));
    }
}
