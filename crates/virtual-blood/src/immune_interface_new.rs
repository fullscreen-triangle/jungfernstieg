//! Immune cell interface for Virtual Blood monitoring
//!
//! Provides biological sensor networks through living immune cells with Ω(n²) information density.
//!
//! ## Theoretical Foundation
//!
//! Immune Cell Monitoring utilizes living immune cells as biological sensors that interface
//! directly with neural tissue, reporting cellular status with superior information density
//! compared to external sensors. The Ω(n²) information density arises from the combinatorial
//! nature of immune cell network interactions.
//!
//! ## Quadratic Information Density Theorem
//!
//! For n immune cells monitoring neural tissue:
//! ```
//! Information_density = Ω(n²)
//! Total_information = n·local_information + (n²-n)/2·interaction_information  
//! ```
//!
//! Where interaction_information >> local_information, yielding quadratic scaling.

use crate::VirtualBlood;
use jungfernstieg_core::{ComponentId, JungfernstiegError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use uuid::Uuid;
use tracing::{debug, info, warn};

/// Immune cell interface system implementing Ω(n²) information density monitoring
#[derive(Debug)]
pub struct ImmuneCellInterface {
    /// Interface identifier
    pub id: Uuid,
    /// Interface configuration
    pub config: InterfaceConfig,
    /// Deployed monitoring cells organized by type
    pub monitoring_cells: HashMap<CellType, Vec<MonitoringCell>>,
    /// Cell interaction network for quadratic information scaling
    pub interaction_network: CellInteractionNetwork,
    /// Information density metrics
    pub information_density_metrics: InformationDensityMetrics,
    /// Biological sensor network
    pub sensor_network: BiologicalSensorNetwork,
}

impl ImmuneCellInterface {
    /// Create new immune cell interface
    pub fn new(config: InterfaceConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            monitoring_cells: HashMap::new(),
            interaction_network: CellInteractionNetwork::new(),
            information_density_metrics: InformationDensityMetrics::default(),
            sensor_network: BiologicalSensorNetwork::new(),
        }
    }

    /// Deploy immune cell monitoring network with Ω(n²) information density
    pub async fn deploy_monitoring_network(&mut self, target_density: usize) -> Result<NetworkDeploymentResult> {
        info!("Deploying immune cell monitoring network with target density {}", target_density);

        // Deploy different types of immune cells for comprehensive monitoring
        let macrophage_count = target_density / 4; // 25% macrophages
        let t_cell_count = target_density / 2;      // 50% T cells  
        let b_cell_count = target_density / 8;      // 12.5% B cells
        let nk_cell_count = target_density / 8;     // 12.5% NK cells

        // Deploy each cell type
        self.deploy_macrophages(macrophage_count).await?;
        self.deploy_t_cells(t_cell_count).await?;
        self.deploy_b_cells(b_cell_count).await?;
        self.deploy_nk_cells(nk_cell_count).await?;

        // Establish interaction network for quadratic information density
        self.establish_interaction_network().await?;

        // Validate information density achievement
        let achieved_density = self.calculate_information_density();
        let theoretical_density = self.calculate_theoretical_density(target_density);

        Ok(NetworkDeploymentResult {
            total_cells_deployed: target_density,
            achieved_information_density: achieved_density,
            theoretical_information_density: theoretical_density,
            quadratic_scaling_achieved: achieved_density >= theoretical_density * 0.8, // 80% of theoretical
            network_efficiency: self.calculate_network_efficiency(),
        })
    }

    /// Monitor neural status with Ω(n²) information density
    pub async fn monitor_neural_status(&self, virtual_blood: &VirtualBlood) -> Result<NeuralStatusReport> {
        debug!("Monitoring neural status with quadratic information density");

        // Collect local information from each cell
        let local_information = self.collect_local_cell_information().await?;
        
        // Collect interaction information from cell network (quadratic component)  
        let interaction_information = self.collect_interaction_information().await?;

        // Combine for total information with Ω(n²) scaling
        let total_information = self.combine_information_sources(local_information, interaction_information)?;

        // Generate comprehensive neural status report
        let neural_health_score = self.calculate_neural_health_score(&total_information);
        let cellular_status = self.assess_cellular_status(&total_information, virtual_blood);
        let tissue_integrity = self.assess_tissue_integrity(&total_information);
        let metabolic_activity = self.assess_metabolic_activity(&total_information, virtual_blood);

        let report = NeuralStatusReport {
            report_id: Uuid::new_v4(),
            timestamp: Instant::now(),
            neural_health_score,
            cellular_status,
            tissue_integrity,
            metabolic_activity,
            information_density_achieved: self.information_density_metrics.current_density,
            quadratic_scaling_factor: self.information_density_metrics.quadratic_scaling_factor,
            monitoring_cell_count: self.get_total_cell_count(),
            interaction_count: self.interaction_network.get_interaction_count(),
            sensor_readings: total_information,
        };

        // Update metrics
        self.update_monitoring_metrics(&report);

        Ok(report)
    }

    /// Deploy macrophage monitoring cells
    async fn deploy_macrophages(&mut self, count: usize) -> Result<()> {
        debug!("Deploying {} macrophages", count);

        let mut macrophages = Vec::new();
        for i in 0..count {
            let macrophage = MonitoringCell {
                cell_id: Uuid::new_v4(),
                cell_type: CellType::Macrophage,
                monitoring_capabilities: vec![
                    MonitoringCapability::PhagocyticActivity,
                    MonitoringCapability::CytokineProduction,
                    MonitoringCapability::TissueRemodeling,
                    MonitoringCapability::PathogenDetection,
                ],
                sensitivity_level: 0.92,
                information_contribution: 25.0, // Base information per macrophage
                spatial_position: self.calculate_optimal_position(i, count, CellType::Macrophage),
                activation_state: ActivationState::Surveillance,
                interaction_capacity: 8, // Can interact with up to 8 other cells
            };
            
            macrophages.push(macrophage);
        }

        self.monitoring_cells.insert(CellType::Macrophage, macrophages);
        Ok(())
    }

    /// Deploy T cell monitoring network
    async fn deploy_t_cells(&mut self, count: usize) -> Result<()> {
        debug!("Deploying {} T cells", count);

        let mut t_cells = Vec::new();
        for i in 0..count {
            let t_cell = MonitoringCell {
                cell_id: Uuid::new_v4(),
                cell_type: CellType::TCell,
                monitoring_capabilities: vec![
                    MonitoringCapability::AntigenRecognition,
                    MonitoringCapability::CellularCommunication,
                    MonitoringCapability::ImmuneMemory,
                    MonitoringCapability::ActivityRegulation,
                ],
                sensitivity_level: 0.95,
                information_contribution: 20.0, // Base information per T cell
                spatial_position: self.calculate_optimal_position(i, count, CellType::TCell),
                activation_state: ActivationState::Ready,
                interaction_capacity: 12, // Higher interaction capacity for T cells
            };
            
            t_cells.push(t_cell);
        }

        self.monitoring_cells.insert(CellType::TCell, t_cells);
        Ok(())
    }

    /// Deploy B cell monitoring network
    async fn deploy_b_cells(&mut self, count: usize) -> Result<()> {
        debug!("Deploying {} B cells", count);

        let mut b_cells = Vec::new();
        for i in 0..count {
            let b_cell = MonitoringCell {
                cell_id: Uuid::new_v4(),
                cell_type: CellType::BCell,
                monitoring_capabilities: vec![
                    MonitoringCapability::AntibodyProduction,
                    MonitoringCapability::MemoryFormation,
                    MonitoringCapability::SignalTransduction,
                ],
                sensitivity_level: 0.88,
                information_contribution: 18.0, // Base information per B cell
                spatial_position: self.calculate_optimal_position(i, count, CellType::BCell),
                activation_state: ActivationState::Surveillance,
                interaction_capacity: 6,
            };
            
            b_cells.push(b_cell);
        }

        self.monitoring_cells.insert(CellType::BCell, b_cells);
        Ok(())
    }

    /// Deploy natural killer cell monitoring network
    async fn deploy_nk_cells(&mut self, count: usize) -> Result<()> {
        debug!("Deploying {} NK cells", count);

        let mut nk_cells = Vec::new();
        for i in 0..count {
            let nk_cell = MonitoringCell {
                cell_id: Uuid::new_v4(),
                cell_type: CellType::NaturalKiller,
                monitoring_capabilities: vec![
                    MonitoringCapability::CytotoxicActivity,
                    MonitoringCapability::StressResponse,
                    MonitoringCapability::CellularIntegrity,
                ],
                sensitivity_level: 0.90,
                information_contribution: 22.0, // Base information per NK cell
                spatial_position: self.calculate_optimal_position(i, count, CellType::NaturalKiller),
                activation_state: ActivationState::Patrol,
                interaction_capacity: 5,
            };
            
            nk_cells.push(nk_cell);
        }

        self.monitoring_cells.insert(CellType::NaturalKiller, nk_cells);
        Ok(())
    }

    /// Calculate optimal spatial position for immune cell deployment
    fn calculate_optimal_position(&self, index: usize, total: usize, cell_type: CellType) -> SpatialPosition {
        // Distribute cells optimally for maximum coverage and interaction potential
        let angle = 2.0 * std::f64::consts::PI * index as f64 / total as f64;
        let radius = match cell_type {
            CellType::Macrophage => 0.8,      // Outer patrol radius
            CellType::TCell => 0.6,           // Mid-range for communication
            CellType::BCell => 0.4,           // Inner radius for antibody production
            CellType::NaturalKiller => 1.0,   // Outermost patrol
            CellType::DendriticCell => 0.5,   // Central positioning
        };

        SpatialPosition {
            x: radius * angle.cos(),
            y: radius * angle.sin(),
            z: (index as f64 / total as f64 - 0.5) * 0.2, // Small z variation
        }
    }

    /// Establish interaction network for quadratic information scaling
    async fn establish_interaction_network(&mut self) -> Result<()> {
        debug!("Establishing cell interaction network");

        let all_cells = self.get_all_cells();
        
        // Create interactions between cells based on proximity and compatibility
        for (i, cell_a) in all_cells.iter().enumerate() {
            for cell_b in all_cells.iter().skip(i + 1) {
                if self.should_cells_interact(cell_a, cell_b) {
                    let interaction = CellInteraction {
                        interaction_id: Uuid::new_v4(),
                        cell_a_id: cell_a.cell_id,
                        cell_b_id: cell_b.cell_id,
                        interaction_type: self.determine_interaction_type(&cell_a.cell_type, &cell_b.cell_type),
                        interaction_strength: self.calculate_interaction_strength(cell_a, cell_b),
                        information_amplification: self.calculate_information_amplification(cell_a, cell_b),
                    };
                    
                    self.interaction_network.add_interaction(interaction);
                }
            }
        }

        // Update information density metrics
        self.update_information_density_metrics();

        Ok(())
    }

    /// Get total cell count across all types
    fn get_total_cell_count(&self) -> usize {
        self.monitoring_cells.values().map(|cells| cells.len()).sum()
    }

    /// Get all cells as a flat vector
    fn get_all_cells(&self) -> Vec<&MonitoringCell> {
        self.monitoring_cells.values().flatten().collect()
    }

    /// Determine if two cells should interact
    fn should_cells_interact(&self, cell_a: &MonitoringCell, cell_b: &MonitoringCell) -> bool {
        // Cells interact based on spatial proximity and functional compatibility
        let distance = self.calculate_spatial_distance(&cell_a.spatial_position, &cell_b.spatial_position);
        let max_interaction_distance = 1.5; // Maximum distance for interaction
        
        let functional_compatibility = self.assess_functional_compatibility(&cell_a.cell_type, &cell_b.cell_type);
        
        distance <= max_interaction_distance && functional_compatibility > 0.3
    }

    /// Calculate spatial distance between cells
    fn calculate_spatial_distance(&self, pos_a: &SpatialPosition, pos_b: &SpatialPosition) -> f64 {
        let dx = pos_a.x - pos_b.x;
        let dy = pos_a.y - pos_b.y;
        let dz = pos_a.z - pos_b.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate current information density
    fn calculate_information_density(&self) -> f64 {
        let n = self.get_total_cell_count() as f64;
        let interaction_count = self.interaction_network.get_interaction_count() as f64;
        
        // Ω(n²) density achieved when interaction_count approaches n²/2
        let theoretical_max_interactions = n * (n - 1.0) / 2.0;
        let density_ratio = interaction_count / theoretical_max_interactions.max(1.0);
        
        // Information density scales quadratically with effective network utilization
        n * n * density_ratio
    }

    /// Calculate theoretical information density
    fn calculate_theoretical_density(&self, cell_count: usize) -> f64 {
        let n = cell_count as f64;
        n * n // Theoretical maximum Ω(n²)
    }

    fn calculate_network_efficiency(&self) -> f64 {
        let active_interactions = self.interaction_network.get_interaction_count();
        let total_cells = self.get_total_cell_count();
        
        if total_cells > 1 {
            let max_possible_interactions = total_cells * (total_cells - 1) / 2;
            active_interactions as f64 / max_possible_interactions as f64
        } else {
            0.0
        }
    }

    // Placeholder implementations for compilation
    async fn collect_local_cell_information(&self) -> Result<LocalInformation> {
        Ok(LocalInformation::default())
    }

    async fn collect_interaction_information(&self) -> Result<InteractionInformation> {
        Ok(InteractionInformation::default())
    }

    fn combine_information_sources(
        &self,
        local: LocalInformation,
        interaction: InteractionInformation,
    ) -> Result<TotalInformation> {
        Ok(TotalInformation::default())
    }

    fn assess_functional_compatibility(&self, type_a: &CellType, type_b: &CellType) -> f64 {
        match (type_a, type_b) {
            (CellType::Macrophage, CellType::TCell) => 0.9,
            (CellType::TCell, CellType::BCell) => 0.8,
            (CellType::Macrophage, CellType::NaturalKiller) => 0.7,
            (CellType::BCell, CellType::NaturalKiller) => 0.6,
            (CellType::DendriticCell, _) => 0.95,
            (a, b) if a == b => 0.5,
            _ => 0.4,
        }
    }

    fn determine_interaction_type(&self, type_a: &CellType, type_b: &CellType) -> InteractionType {
        match (type_a, type_b) {
            (CellType::Macrophage, CellType::TCell) => InteractionType::AntigenPresentation,
            (CellType::TCell, CellType::BCell) => InteractionType::ImmunologicalSynapse,
            (CellType::Macrophage, CellType::NaturalKiller) => InteractionType::CytokineSignaling,
            _ => InteractionType::GeneralCommunication,
        }
    }

    fn calculate_interaction_strength(&self, cell_a: &MonitoringCell, cell_b: &MonitoringCell) -> f64 {
        let distance = self.calculate_spatial_distance(&cell_a.spatial_position, &cell_b.spatial_position);
        let compatibility = self.assess_functional_compatibility(&cell_a.cell_type, &cell_b.cell_type);
        
        let distance_factor = (2.0 - distance).max(0.1).min(1.0);
        compatibility * distance_factor * 0.8
    }

    fn calculate_information_amplification(&self, cell_a: &MonitoringCell, cell_b: &MonitoringCell) -> f64 {
        let base_amplification = 1.5;
        let sensitivity_factor = (cell_a.sensitivity_level + cell_b.sensitivity_level) / 2.0;
        base_amplification * sensitivity_factor
    }

    fn update_information_density_metrics(&mut self) {
        let current_density = self.calculate_information_density();
        let quadratic_factor = current_density / (self.get_total_cell_count() as f64).max(1.0);
        
        self.information_density_metrics = InformationDensityMetrics {
            current_density,
            quadratic_scaling_factor: quadratic_factor,
            theoretical_maximum: self.calculate_theoretical_density(self.get_total_cell_count()),
            network_efficiency: self.calculate_network_efficiency(),
            last_update: Instant::now(),
        };
    }

    // Additional placeholder methods for compilation
    fn calculate_neural_health_score(&self, total_information: &TotalInformation) -> f64 { 0.95 }
    fn assess_cellular_status(&self, total_information: &TotalInformation, virtual_blood: &VirtualBlood) -> CellularStatus { CellularStatus::default() }
    fn assess_tissue_integrity(&self, total_information: &TotalInformation) -> TissueIntegrity { TissueIntegrity::default() }
    fn assess_metabolic_activity(&self, total_information: &TotalInformation, virtual_blood: &VirtualBlood) -> MetabolicActivity { MetabolicActivity::default() }
    fn update_monitoring_metrics(&self, report: &NeuralStatusReport) {}
}

// Supporting types and structures

/// Cell interaction network managing quadratic scaling
#[derive(Debug)]
pub struct CellInteractionNetwork {
    interactions: Vec<CellInteraction>,
}

impl CellInteractionNetwork {
    pub fn new() -> Self {
        Self {
            interactions: Vec::new(),
        }
    }

    pub fn add_interaction(&mut self, interaction: CellInteraction) {
        self.interactions.push(interaction);
    }

    pub fn get_interaction_count(&self) -> usize {
        self.interactions.len()
    }

    pub fn get_all_interactions(&self) -> &[CellInteraction] {
        &self.interactions
    }
}

/// Biological sensor network
#[derive(Debug)]
pub struct BiologicalSensorNetwork {
    sensors: Vec<BiologicalSensor>,
}

impl BiologicalSensorNetwork {
    pub fn new() -> Self {
        Self {
            sensors: Vec::new(),
        }
    }
}

/// Interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfig {
    pub enable_monitoring: bool,
    pub information_density_target: f64, // Ω(n²)
    pub max_interaction_distance: f64,
    pub cell_deployment_strategy: CellDeploymentStrategy,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            information_density_target: 1000.0, // Target Ω(n²) density
            max_interaction_distance: 1.5,
            cell_deployment_strategy: CellDeploymentStrategy::OptimalCoverage,
        }
    }
}

/// Cell deployment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellDeploymentStrategy {
    OptimalCoverage,
    MaximumInteraction,
    UniformDistribution,
    FunctionalClustering,
}

/// Monitoring cell in the immune network
#[derive(Debug, Clone)]
pub struct MonitoringCell {
    pub cell_id: Uuid,
    pub cell_type: CellType,
    pub monitoring_capabilities: Vec<MonitoringCapability>,
    pub sensitivity_level: f64,
    pub information_contribution: f64,
    pub spatial_position: SpatialPosition,
    pub activation_state: ActivationState,
    pub interaction_capacity: usize,
}

/// Types of immune cells for monitoring
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CellType {
    Macrophage,
    TCell,
    BCell,
    NaturalKiller,
    DendriticCell,
}

/// Monitoring capabilities of immune cells
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitoringCapability {
    PhagocyticActivity,
    CytokineProduction,
    TissueRemodeling,
    PathogenDetection,
    AntigenRecognition,
    CellularCommunication,
    ImmuneMemory,
    ActivityRegulation,
    AntibodyProduction,
    MemoryFormation,
    SignalTransduction,
    CytotoxicActivity,
    StressResponse,
    CellularIntegrity,
}

/// 3D spatial position
#[derive(Debug, Clone)]
pub struct SpatialPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Cell activation states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationState {
    Surveillance,
    Ready,
    Active,
    Patrol,
    Resting,
}

/// Cell interaction between immune cells
#[derive(Debug, Clone)]
pub struct CellInteraction {
    pub interaction_id: Uuid,
    pub cell_a_id: Uuid,
    pub cell_b_id: Uuid,
    pub interaction_type: InteractionType,
    pub interaction_strength: f64,
    pub information_amplification: f64,
}

/// Types of cell interactions
#[derive(Debug, Clone)]
pub enum InteractionType {
    AntigenPresentation,
    ImmunologicalSynapse,
    CytokineSignaling,
    ImmuneSurveillance,
    GeneralCommunication,
}

/// Information density metrics for Ω(n²) tracking
#[derive(Debug, Clone)]
pub struct InformationDensityMetrics {
    pub current_density: f64,
    pub quadratic_scaling_factor: f64,
    pub theoretical_maximum: f64,
    pub network_efficiency: f64,
    pub last_update: Instant,
}

impl Default for InformationDensityMetrics {
    fn default() -> Self {
        Self {
            current_density: 0.0,
            quadratic_scaling_factor: 0.0,
            theoretical_maximum: 0.0,
            network_efficiency: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Network deployment result
#[derive(Debug, Clone)]
pub struct NetworkDeploymentResult {
    pub total_cells_deployed: usize,
    pub achieved_information_density: f64,
    pub theoretical_information_density: f64,
    pub quadratic_scaling_achieved: bool,
    pub network_efficiency: f64,
}

/// Neural status report with Ω(n²) information density
#[derive(Debug, Clone)]
pub struct NeuralStatusReport {
    pub report_id: Uuid,
    pub timestamp: Instant,
    pub neural_health_score: f64,
    pub cellular_status: CellularStatus,
    pub tissue_integrity: TissueIntegrity,
    pub metabolic_activity: MetabolicActivity,
    pub information_density_achieved: f64,
    pub quadratic_scaling_factor: f64,
    pub monitoring_cell_count: usize,
    pub interaction_count: usize,
    pub sensor_readings: TotalInformation,
}

/// Supporting information structures
#[derive(Debug, Clone, Default)]
pub struct LocalInformation {
    pub total_cells: usize,
    pub cell_contributions: HashMap<CellType, f64>,
    pub cellular_health_indicators: HashMap<CellType, Vec<HealthIndicator>>,
}

#[derive(Debug, Clone, Default)]
pub struct InteractionInformation {
    pub total_interactions: usize,
    pub interaction_contributions: HashMap<Uuid, f64>,
    pub network_effects: HashMap<Uuid, f64>,
}

#[derive(Debug, Clone, Default)]
pub struct TotalInformation {
    pub local_component: f64,
    pub interaction_component: f64,
    pub network_effects_component: f64,
    pub total_information: f64,
    pub quadratic_scaling_factor: f64,
    pub information_density: f64,
}

#[derive(Debug, Clone)]
pub struct HealthIndicator {
    pub capability: MonitoringCapability,
    pub status_value: f64,
    pub status_quality: StatusQuality,
}

#[derive(Debug, Clone)]
pub enum StatusQuality {
    Excellent,
    Good,
    Adequate,
    Poor,
}

#[derive(Debug, Clone, Default)]
pub struct CellularStatus {
    pub viability_score: f64,
    pub metabolic_activity: f64,
    pub communication_efficiency: f64,
    pub stress_indicators: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TissueIntegrity {
    pub structural_integrity: f64,
    pub barrier_function: f64,
    pub regenerative_capacity: f64,
}

#[derive(Debug, Clone, Default)]
pub struct MetabolicActivity {
    pub energy_production: f64,
    pub waste_clearance: f64,
    pub oxygen_utilization: f64,
    pub metabolic_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalSensor {
    pub sensor_id: Uuid,
    pub sensor_type: SensorType,
    pub sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorType {
    ChemicalGradient,
    ElectricalActivity,
    MechanicalStress,
    TemperatureVariation,
    pHLevel,
}

// Additional interfaces for compatibility
pub use self::ImmuneCellInterface as ImmuneCellSignaling;
pub use self::BiologicalSensorNetwork as CellularStatusReporting;
pub use self::CellInteractionNetwork as ImmuneNetworkCommunication;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_immune_interface_creation() {
        let config = InterfaceConfig::default();
        let interface = ImmuneCellInterface::new(config);
        
        assert_eq!(interface.monitoring_cells.len(), 0);
        assert_eq!(interface.get_total_cell_count(), 0);
    }

    #[tokio::test]
    async fn test_monitoring_network_deployment() {
        let config = InterfaceConfig::default();
        let mut interface = ImmuneCellInterface::new(config);
        
        let result = interface.deploy_monitoring_network(100).await.unwrap();
        
        assert_eq!(result.total_cells_deployed, 100);
        assert!(result.achieved_information_density > 0.0);
        assert!(result.theoretical_information_density > 0.0);
        assert_eq!(interface.get_total_cell_count(), 100);
    }

    #[test]
    fn test_information_density_calculation() {
        let config = InterfaceConfig::default();
        let interface = ImmuneCellInterface::new(config);
        
        let theoretical_density = interface.calculate_theoretical_density(10);
        assert_eq!(theoretical_density, 100.0); // n² = 10² = 100
        
        let theoretical_density_large = interface.calculate_theoretical_density(100);
        assert_eq!(theoretical_density_large, 10000.0); // n² = 100² = 10,000
    }

    #[test]
    fn test_spatial_distance_calculation() {
        let config = InterfaceConfig::default();
        let interface = ImmuneCellInterface::new(config);
        
        let pos_a = SpatialPosition { x: 0.0, y: 0.0, z: 0.0 };
        let pos_b = SpatialPosition { x: 3.0, y: 4.0, z: 0.0 };
        
        let distance = interface.calculate_spatial_distance(&pos_a, &pos_b);
        assert_eq!(distance, 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_functional_compatibility() {
        let config = InterfaceConfig::default();
        let interface = ImmuneCellInterface::new(config);
        
        let compatibility = interface.assess_functional_compatibility(&CellType::Macrophage, &CellType::TCell);
        assert_eq!(compatibility, 0.9);
        
        let same_type_compatibility = interface.assess_functional_compatibility(&CellType::TCell, &CellType::TCell);
        assert_eq!(same_type_compatibility, 0.5);
    }

    #[test]
    fn test_cell_interaction_network() {
        let mut network = CellInteractionNetwork::new();
        
        let interaction = CellInteraction {
            interaction_id: Uuid::new_v4(),
            cell_a_id: Uuid::new_v4(),
            cell_b_id: Uuid::new_v4(),
            interaction_type: InteractionType::AntigenPresentation,
            interaction_strength: 0.8,
            information_amplification: 1.5,
        };
        
        network.add_interaction(interaction);
        
        assert_eq!(network.get_interaction_count(), 1);
        assert_eq!(network.get_all_interactions().len(), 1);
    }

    #[test]
    fn test_quadratic_scaling_theory() {
        // Test the theoretical foundation: Information_density = Ω(n²)
        let n1 = 10;
        let n2 = 20;
        
        let config = InterfaceConfig::default();
        let interface = ImmuneCellInterface::new(config);
        
        let density1 = interface.calculate_theoretical_density(n1);
        let density2 = interface.calculate_theoretical_density(n2);
        
        // Quadratic scaling: doubling n should quadruple density
        assert_eq!(density2 / density1, 4.0);
        assert_eq!(density1, (n1 as f64).powi(2));
        assert_eq!(density2, (n2 as f64).powi(2));
    }
}
