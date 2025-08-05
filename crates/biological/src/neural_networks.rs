//! Neural network management and coordination
//!
//! Implements biological neural network preparation, maintenance, and integration
//! with Virtual Blood circulation systems as described in the theoretical framework.

use jungfernstieg_core::{ComponentId, JungfernstiegError, Result, ViabilityStatus, VirtualBloodQuality};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Unique identifier for neural networks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuralNetworkId(pub Uuid);

impl NeuralNetworkId {
    /// Generate a new neural network ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for NeuralNetworkId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<NeuralNetworkId> for ComponentId {
    fn from(id: NeuralNetworkId) -> Self {
        ComponentId(id.0)
    }
}

/// Configuration for neural network setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    /// Network identifier
    pub network_id: NeuralNetworkId,
    /// Number of neurons in the network
    pub neuron_count: usize,
    /// Network topology
    pub topology: NetworkTopology,
    /// Culture conditions
    pub culture_conditions: CultureConditions,
    /// Maturation settings
    pub maturation: MaturationConfig,
    /// Virtual Blood interface settings
    pub virtual_blood_interface: VirtualBloodInterfaceConfig,
}

/// Neural network topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Random connections
    Random { connection_probability: f64 },
    /// Small-world network
    SmallWorld { neighbors: usize, rewiring_probability: f64 },
    /// Scale-free network
    ScaleFree { attachment_preference: f64 },
    /// Custom topology from file
    Custom { topology_file: String },
}

/// Culture conditions for neural network preparation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CultureConditions {
    /// Growth factors concentration
    pub growth_factors_concentration: f64,
    /// Culture medium type
    pub culture_medium: String,
    /// Substrate material
    pub substrate_material: String,
    /// Temperature (Celsius)
    pub temperature_celsius: f64,
    /// CO2 concentration (%)
    pub co2_concentration: f64,
    /// pH level
    pub ph_level: f64,
    /// Osmolarity (mOsm/L)
    pub osmolarity: f64,
}

impl Default for CultureConditions {
    fn default() -> Self {
        Self {
            growth_factors_concentration: 1.0,
            culture_medium: "Neurobasal-A".to_string(),
            substrate_material: "PDL-Laminin".to_string(),
            temperature_celsius: 37.0,
            co2_concentration: 5.0,
            ph_level: 7.4,
            osmolarity: 320.0,
        }
    }
}

/// Maturation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturationConfig {
    /// Maturation duration in days
    pub duration_days: u32,
    /// Assessment frequency during maturation
    pub assessment_frequency_hours: u32,
    /// Minimum viability threshold for completion
    pub min_viability_threshold: f64,
}

impl Default for MaturationConfig {
    fn default() -> Self {
        Self {
            duration_days: 14,
            assessment_frequency_hours: 24,
            min_viability_threshold: 95.0,
        }
    }
}

/// Virtual Blood interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBloodInterfaceConfig {
    /// Perfusion flow rate (μL/min)
    pub flow_rate: f64,
    /// Exchange efficiency target
    pub exchange_efficiency_target: f64,
    /// Interface material
    pub interface_material: String,
    /// Membrane permeability
    pub membrane_permeability: f64,
}

impl Default for VirtualBloodInterfaceConfig {
    fn default() -> Self {
        Self {
            flow_rate: 100.0,
            exchange_efficiency_target: 0.95,
            interface_material: "PDMS".to_string(),
            membrane_permeability: 0.8,
        }
    }
}

/// Neural network state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuralNetworkState {
    /// Network is being prepared/cultured
    Culturing,
    /// Network is maturing
    Maturing,
    /// Network is ready for Virtual Blood integration
    Ready,
    /// Network is integrated with Virtual Blood
    Integrated,
    /// Network is operating normally
    Operational,
    /// Network has viability warnings
    Warning,
    /// Network is in critical condition
    Critical,
    /// Network has failed
    Failed,
}

/// Individual neural network
#[derive(Debug)]
pub struct NeuralNetwork {
    /// Network identifier
    pub id: NeuralNetworkId,
    /// Network configuration
    config: NeuralNetworkConfig,
    /// Current state
    state: NeuralNetworkState,
    /// Creation timestamp
    created_at: Instant,
    /// Current viability status
    viability_status: Option<ViabilityStatus>,
    /// Virtual Blood integration status
    virtual_blood_integrated: bool,
    /// Activity metrics
    activity_metrics: ActivityMetrics,
    /// Network statistics
    statistics: NetworkStatistics,
}

impl NeuralNetwork {
    /// Create new neural network
    pub fn new(config: NeuralNetworkConfig) -> Self {
        Self {
            id: config.network_id,
            config,
            state: NeuralNetworkState::Culturing,
            created_at: Instant::now(),
            viability_status: None,
            virtual_blood_integrated: false,
            activity_metrics: ActivityMetrics::default(),
            statistics: NetworkStatistics::default(),
        }
    }

    /// Get network identifier
    pub fn id(&self) -> NeuralNetworkId {
        self.id
    }

    /// Get current state
    pub fn state(&self) -> &NeuralNetworkState {
        &self.state
    }

    /// Get network age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get current viability status
    pub fn viability_status(&self) -> Option<&ViabilityStatus> {
        self.viability_status.as_ref()
    }

    /// Check if integrated with Virtual Blood
    pub fn is_virtual_blood_integrated(&self) -> bool {
        self.virtual_blood_integrated
    }

    /// Get activity metrics
    pub fn activity_metrics(&self) -> &ActivityMetrics {
        &self.activity_metrics
    }

    /// Get network statistics
    pub fn statistics(&self) -> &NetworkStatistics {
        &self.statistics
    }

    /// Initialize network culture according to Algorithm: Biological Neural Network Preparation
    pub async fn initialize_culture(&mut self) -> Result<()> {
        info!("Initializing neural culture for network {}", self.id.0);
        
        // Extract primary neurons
        self.extract_primary_neurons().await?;
        
        // Culture neural network
        self.culture_neural_network().await?;
        
        // Allow network maturation
        self.allow_network_maturation().await?;
        
        // Assess network viability
        let viability = self.assess_network_viability().await?;
        
        if viability.viability_percent > self.config.maturation.min_viability_threshold {
            self.prepare_for_vb_integration().await?;
            self.state = NeuralNetworkState::Ready;
            info!("Neural network {} culture initialization complete", self.id.0);
        } else {
            warn!("Neural network {} failed viability threshold: {:.1}%", 
                  self.id.0, viability.viability_percent);
            self.state = NeuralNetworkState::Failed;
            return Err(JungfernstiegError::biological(
                self.id.into(),
                format!("Network viability {:.1}% below threshold {:.1}%",
                       viability.viability_percent,
                       self.config.maturation.min_viability_threshold)
            ));
        }
        
        Ok(())
    }

    /// Extract primary neurons from tissue
    async fn extract_primary_neurons(&mut self) -> Result<()> {
        debug!("Extracting primary neurons for network {}", self.id.0);
        
        // Simulate neural extraction process
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        self.statistics.neuron_count = self.config.neuron_count;
        self.statistics.extraction_efficiency = 0.85; // 85% extraction efficiency
        
        Ok(())
    }

    /// Culture neural network with growth factors
    async fn culture_neural_network(&mut self) -> Result<()> {
        debug!("Culturing neural network {} with growth factors", self.id.0);
        
        // Validate culture conditions
        self.validate_culture_conditions()?;
        
        // Simulate culture process
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        self.statistics.culture_success_rate = 0.92; // 92% culture success
        self.state = NeuralNetworkState::Maturing;
        
        Ok(())
    }

    /// Allow network maturation for specified duration
    async fn allow_network_maturation(&mut self) -> Result<()> {
        info!("Starting maturation for network {} ({} days)", 
              self.id.0, self.config.maturation.duration_days);
        
        // In production, this would be actual days, but for testing we simulate
        let maturation_simulation_duration = Duration::from_millis(500);
        
        let start_time = Instant::now();
        while start_time.elapsed() < maturation_simulation_duration {
            // Assess maturation progress
            self.assess_maturation_progress().await?;
            
            // Simulate maturation time step
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        
        self.statistics.maturation_completion = 1.0; // 100% maturation
        info!("Network {} maturation complete", self.id.0);
        
        Ok(())
    }

    /// Assess current network viability
    pub async fn assess_network_viability(&mut self) -> Result<ViabilityStatus> {
        debug!("Assessing viability for network {}", self.id.0);
        
        // Calculate viability based on network health indicators
        let viability_percent = self.calculate_viability_percentage().await?;
        let metabolic_activity = self.calculate_metabolic_activity().await?;
        let synaptic_function = self.calculate_synaptic_function().await?;
        let vb_quality = self.assess_virtual_blood_quality().await?;
        
        let status = ViabilityStatus::new(
            viability_percent,
            metabolic_activity,
            synaptic_function,
            vb_quality,
        );
        
        // Update state based on viability
        self.update_state_from_viability(&status);
        self.viability_status = Some(status.clone());
        
        Ok(status)
    }

    /// Prepare network for Virtual Blood integration
    async fn prepare_for_vb_integration(&mut self) -> Result<()> {
        debug!("Preparing network {} for Virtual Blood integration", self.id.0);
        
        // Validate interface configuration
        self.validate_vb_interface_config()?;
        
        // Prepare interface materials
        self.prepare_interface_materials().await?;
        
        // Establish connection points
        self.establish_vb_connection_points().await?;
        
        self.statistics.vb_integration_readiness = 1.0;
        
        Ok(())
    }

    /// Integrate with Virtual Blood circulation system
    pub async fn integrate_virtual_blood(&mut self) -> Result<()> {
        if self.state != NeuralNetworkState::Ready {
            return Err(JungfernstiegError::biological(
                self.id.into(),
                format!("Network not ready for VB integration, current state: {:?}", self.state)
            ));
        }
        
        info!("Integrating network {} with Virtual Blood circulation", self.id.0);
        
        // Establish Virtual Blood interface
        self.establish_vb_interface().await?;
        
        // Begin circulation monitoring
        self.begin_circulation_monitoring().await?;
        
        self.virtual_blood_integrated = true;
        self.state = NeuralNetworkState::Integrated;
        
        info!("Network {} successfully integrated with Virtual Blood", self.id.0);
        Ok(())
    }

    /// Start operational mode
    pub async fn start_operation(&mut self) -> Result<()> {
        if !self.virtual_blood_integrated {
            return Err(JungfernstiegError::biological(
                self.id.into(),
                "Cannot start operation without Virtual Blood integration".to_string()
            ));
        }
        
        info!("Starting operational mode for network {}", self.id.0);
        
        // Begin neural activity monitoring
        self.begin_activity_monitoring().await?;
        
        // Start adaptive optimization
        self.start_adaptive_optimization().await?;
        
        self.state = NeuralNetworkState::Operational;
        
        info!("Network {} is now operational", self.id.0);
        Ok(())
    }

    // Private helper methods

    async fn assess_maturation_progress(&mut self) -> Result<()> {
        // Simulate maturation assessment
        self.statistics.maturation_completion += 0.1;
        self.statistics.maturation_completion = self.statistics.maturation_completion.min(1.0);
        Ok(())
    }

    async fn calculate_viability_percentage(&self) -> Result<f64> {
        // Calculate based on network health indicators
        let base_viability = 95.0;
        let age_factor = (1.0 - (self.age().as_secs() as f64 / (30.0 * 24.0 * 3600.0))).max(0.8);
        let culture_factor = self.statistics.culture_success_rate;
        let maturation_factor = self.statistics.maturation_completion;
        
        Ok(base_viability * age_factor * culture_factor * maturation_factor)
    }

    async fn calculate_metabolic_activity(&self) -> Result<f64> {
        // Simulate metabolic activity calculation
        Ok(95.0 + (self.activity_metrics.firing_rate / 100.0) * 5.0)
    }

    async fn calculate_synaptic_function(&self) -> Result<f64> {
        // Simulate synaptic function assessment
        Ok(96.0 + (self.activity_metrics.synaptic_strength - 0.5) * 8.0)
    }

    async fn assess_virtual_blood_quality(&self) -> Result<VirtualBloodQuality> {
        if self.virtual_blood_integrated {
            // Assess based on circulation metrics
            if self.statistics.vb_circulation_efficiency > 0.98 {
                Ok(VirtualBloodQuality::Optimal)
            } else if self.statistics.vb_circulation_efficiency > 0.95 {
                Ok(VirtualBloodQuality::Excellent)
            } else if self.statistics.vb_circulation_efficiency > 0.90 {
                Ok(VirtualBloodQuality::Good)
            } else {
                Ok(VirtualBloodQuality::Warning)
            }
        } else {
            Ok(VirtualBloodQuality::Stable)
        }
    }

    fn update_state_from_viability(&mut self, status: &ViabilityStatus) {
        if status.is_critical() {
            self.state = NeuralNetworkState::Critical;
        } else if status.is_warning() {
            self.state = NeuralNetworkState::Warning;
        }
        // Keep current state if viability is good
    }

    fn validate_culture_conditions(&self) -> Result<()> {
        let conditions = &self.config.culture_conditions;
        
        if conditions.temperature_celsius < 35.0 || conditions.temperature_celsius > 39.0 {
            return Err(JungfernstiegError::ValidationError {
                field: "temperature_celsius".to_string(),
                message: "Temperature must be between 35-39°C".to_string(),
            });
        }
        
        if conditions.ph_level < 7.2 || conditions.ph_level > 7.6 {
            return Err(JungfernstiegError::ValidationError {
                field: "ph_level".to_string(),
                message: "pH must be between 7.2-7.6".to_string(),
            });
        }
        
        Ok(())
    }

    fn validate_vb_interface_config(&self) -> Result<()> {
        let interface = &self.config.virtual_blood_interface;
        
        if interface.flow_rate <= 0.0 {
            return Err(JungfernstiegError::ValidationError {
                field: "flow_rate".to_string(),
                message: "Flow rate must be positive".to_string(),
            });
        }
        
        Ok(())
    }

    async fn prepare_interface_materials(&mut self) -> Result<()> {
        debug!("Preparing interface materials for network {}", self.id.0);
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    async fn establish_vb_connection_points(&mut self) -> Result<()> {
        debug!("Establishing VB connection points for network {}", self.id.0);
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    async fn establish_vb_interface(&mut self) -> Result<()> {
        debug!("Establishing Virtual Blood interface for network {}", self.id.0);
        tokio::time::sleep(Duration::from_millis(100)).await;
        self.statistics.vb_circulation_efficiency = 0.98;
        Ok(())
    }

    async fn begin_circulation_monitoring(&mut self) -> Result<()> {
        debug!("Beginning circulation monitoring for network {}", self.id.0);
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    async fn begin_activity_monitoring(&mut self) -> Result<()> {
        debug!("Beginning activity monitoring for network {}", self.id.0);
        self.activity_metrics.monitoring_active = true;
        Ok(())
    }

    async fn start_adaptive_optimization(&mut self) -> Result<()> {
        debug!("Starting adaptive optimization for network {}", self.id.0);
        self.activity_metrics.adaptive_optimization = true;
        Ok(())
    }
}

/// Neural network activity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityMetrics {
    /// Average firing rate (Hz)
    pub firing_rate: f64,
    /// Synaptic strength measure
    pub synaptic_strength: f64,
    /// Network synchronization level
    pub synchronization: f64,
    /// Activity monitoring enabled
    pub monitoring_active: bool,
    /// Adaptive optimization enabled
    pub adaptive_optimization: bool,
}

impl Default for ActivityMetrics {
    fn default() -> Self {
        Self {
            firing_rate: 10.0,
            synaptic_strength: 0.5,
            synchronization: 0.3,
            monitoring_active: false,
            adaptive_optimization: false,
        }
    }
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatistics {
    /// Number of neurons
    pub neuron_count: usize,
    /// Extraction efficiency
    pub extraction_efficiency: f64,
    /// Culture success rate
    pub culture_success_rate: f64,
    /// Maturation completion (0.0-1.0)
    pub maturation_completion: f64,
    /// VB integration readiness
    pub vb_integration_readiness: f64,
    /// VB circulation efficiency
    pub vb_circulation_efficiency: f64,
}

impl Default for NetworkStatistics {
    fn default() -> Self {
        Self {
            neuron_count: 0,
            extraction_efficiency: 0.0,
            culture_success_rate: 0.0,
            maturation_completion: 0.0,
            vb_integration_readiness: 0.0,
            vb_circulation_efficiency: 0.0,
        }
    }
}

/// Neural network manager for coordinating multiple networks
pub struct NeuralNetworkManager {
    /// Managed networks
    networks: HashMap<NeuralNetworkId, NeuralNetwork>,
    /// Configuration
    config: NeuralNetworkManagerConfig,
    /// Statistics
    manager_stats: ManagerStatistics,
}

/// Configuration for neural network manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkManagerConfig {
    /// Maximum number of networks
    pub max_networks: usize,
    /// Default network configuration
    pub default_network_config: NeuralNetworkConfig,
    /// Assessment frequency (seconds)
    pub assessment_frequency_secs: u64,
}

/// Manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerStatistics {
    /// Total networks managed
    pub total_networks: usize,
    /// Operational networks
    pub operational_networks: usize,
    /// Networks in warning state
    pub warning_networks: usize,
    /// Critical networks
    pub critical_networks: usize,
    /// Failed networks
    pub failed_networks: usize,
    /// Average viability across all networks
    pub average_viability: f64,
}

impl Default for ManagerStatistics {
    fn default() -> Self {
        Self {
            total_networks: 0,
            operational_networks: 0,
            warning_networks: 0,
            critical_networks: 0,
            failed_networks: 0,
            average_viability: 0.0,
        }
    }
}

impl NeuralNetworkManager {
    /// Create new neural network manager
    pub fn new(config: NeuralNetworkManagerConfig) -> Self {
        Self {
            networks: HashMap::new(),
            config,
            manager_stats: ManagerStatistics::default(),
        }
    }

    /// Add new neural network
    pub async fn add_network(&mut self, config: NeuralNetworkConfig) -> Result<NeuralNetworkId> {
        if self.networks.len() >= self.config.max_networks {
            return Err(JungfernstiegError::ResourceError {
                message: format!("Maximum networks limit reached: {}", self.config.max_networks),
            });
        }

        let network_id = config.network_id;
        let mut network = NeuralNetwork::new(config);
        
        // Initialize the network culture
        network.initialize_culture().await?;
        
        self.networks.insert(network_id, network);
        self.update_statistics();
        
        info!("Added neural network {} to manager", network_id.0);
        Ok(network_id)
    }

    /// Get network by ID
    pub fn get_network(&self, id: &NeuralNetworkId) -> Option<&NeuralNetwork> {
        self.networks.get(id)
    }

    /// Get mutable network by ID
    pub fn get_network_mut(&mut self, id: &NeuralNetworkId) -> Option<&mut NeuralNetwork> {
        self.networks.get_mut(id)
    }

    /// List all network IDs
    pub fn list_networks(&self) -> Vec<NeuralNetworkId> {
        self.networks.keys().cloned().collect()
    }

    /// Get manager statistics
    pub fn statistics(&self) -> &ManagerStatistics {
        &self.manager_stats
    }

    /// Assess all networks
    pub async fn assess_all_networks(&mut self) -> Result<HashMap<NeuralNetworkId, ViabilityStatus>> {
        let mut results = HashMap::new();
        
        for (id, network) in self.networks.iter_mut() {
            match network.assess_network_viability().await {
                Ok(status) => {
                    results.insert(*id, status);
                }
                Err(e) => {
                    warn!("Failed to assess network {}: {}", id.0, e);
                }
            }
        }
        
        self.update_statistics();
        Ok(results)
    }

    /// Update manager statistics
    fn update_statistics(&mut self) {
        self.manager_stats.total_networks = self.networks.len();
        self.manager_stats.operational_networks = self.networks.values()
            .filter(|n| matches!(n.state, NeuralNetworkState::Operational))
            .count();
        self.manager_stats.warning_networks = self.networks.values()
            .filter(|n| matches!(n.state, NeuralNetworkState::Warning))
            .count();
        self.manager_stats.critical_networks = self.networks.values()
            .filter(|n| matches!(n.state, NeuralNetworkState::Critical))
            .count();
        self.manager_stats.failed_networks = self.networks.values()
            .filter(|n| matches!(n.state, NeuralNetworkState::Failed))
            .count();

        // Calculate average viability
        let total_viability: f64 = self.networks.values()
            .filter_map(|n| n.viability_status.as_ref())
            .map(|status| status.viability_percent)
            .sum();
        
        let viable_networks = self.networks.values()
            .filter(|n| n.viability_status.is_some())
            .count();
            
        self.manager_stats.average_viability = if viable_networks > 0 {
            total_viability / viable_networks as f64
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_neural_network_creation() {
        let config = NeuralNetworkConfig {
            network_id: NeuralNetworkId::new(),
            neuron_count: 1000,
            topology: NetworkTopology::Random { connection_probability: 0.1 },
            culture_conditions: CultureConditions::default(),
            maturation: MaturationConfig::default(),
            virtual_blood_interface: VirtualBloodInterfaceConfig::default(),
        };

        let network = NeuralNetwork::new(config);
        assert_eq!(network.state(), &NeuralNetworkState::Culturing);
        assert!(!network.is_virtual_blood_integrated());
    }

    #[tokio::test]
    async fn test_neural_network_culture_initialization() {
        let config = NeuralNetworkConfig {
            network_id: NeuralNetworkId::new(),
            neuron_count: 500,
            topology: NetworkTopology::Random { connection_probability: 0.1 },
            culture_conditions: CultureConditions::default(),
            maturation: MaturationConfig {
                duration_days: 14,
                assessment_frequency_hours: 24,
                min_viability_threshold: 90.0, // Lower threshold for testing
            },
            virtual_blood_interface: VirtualBloodInterfaceConfig::default(),
        };

        let mut network = NeuralNetwork::new(config);
        let result = network.initialize_culture().await;
        
        // Should succeed with default conditions
        assert!(result.is_ok());
        assert_eq!(network.state(), &NeuralNetworkState::Ready);
    }

    #[tokio::test]
    async fn test_neural_network_manager() {
        let manager_config = NeuralNetworkManagerConfig {
            max_networks: 5,
            default_network_config: NeuralNetworkConfig {
                network_id: NeuralNetworkId::new(),
                neuron_count: 1000,
                topology: NetworkTopology::Random { connection_probability: 0.1 },
                culture_conditions: CultureConditions::default(),
                maturation: MaturationConfig {
                    duration_days: 14,
                    assessment_frequency_hours: 24,
                    min_viability_threshold: 85.0,
                },
                virtual_blood_interface: VirtualBloodInterfaceConfig::default(),
            },
            assessment_frequency_secs: 60,
        };

        let mut manager = NeuralNetworkManager::new(manager_config);
        
        let network_config = NeuralNetworkConfig {
            network_id: NeuralNetworkId::new(),
            neuron_count: 800,
            topology: NetworkTopology::Random { connection_probability: 0.1 },
            culture_conditions: CultureConditions::default(),
            maturation: MaturationConfig {
                duration_days: 14,
                assessment_frequency_hours: 24,
                min_viability_threshold: 85.0,
            },
            virtual_blood_interface: VirtualBloodInterfaceConfig::default(),
        };

        let network_id = manager.add_network(network_config).await.unwrap();
        assert!(manager.get_network(&network_id).is_some());
        assert_eq!(manager.statistics().total_networks, 1);
    }

    #[test]
    fn test_culture_conditions_validation() {
        let config = NeuralNetworkConfig {
            network_id: NeuralNetworkId::new(),
            neuron_count: 1000,
            topology: NetworkTopology::Random { connection_probability: 0.1 },
            culture_conditions: CultureConditions {
                temperature_celsius: 40.0, // Invalid temperature
                ..CultureConditions::default()
            },
            maturation: MaturationConfig::default(),
            virtual_blood_interface: VirtualBloodInterfaceConfig::default(),
        };

        let network = NeuralNetwork::new(config);
        let result = network.validate_culture_conditions();
        assert!(result.is_err());
    }
}