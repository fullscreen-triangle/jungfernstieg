//! Configuration management for Jungfernstieg biological-virtual neural symbiosis system
//!
//! Provides configuration structures and validation for all system components,
//! ensuring compliance with BSL-2+ safety requirements and biological constraints.

use crate::error::{JungfernstiegError, Result};
use crate::types::{SCredits, VirtualBloodQuality};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Complete system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System identification and metadata
    pub system: SystemIdentityConfig,
    
    /// Biological neural network configuration
    pub biological: BiologicalConfig,
    
    /// Virtual Blood circulation configuration
    pub virtual_blood: VirtualBloodConfig,
    
    /// Oscillatory VM configuration
    pub oscillatory_vm: OscillatoryVMConfig,
    
    /// S-entropy system configuration
    pub s_entropy: SEntropyConfig,
    
    /// Safety and monitoring configuration
    pub safety: SafetyConfig,
    
    /// Hardware interface configuration
    pub hardware: HardwareConfig,
    
    /// Monitoring and metrics configuration
    pub monitoring: MonitoringConfig,
}

impl SystemConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)
            .map_err(|e| JungfernstiegError::ConfigurationError {
                message: format!("Failed to parse configuration: {}", e),
            })?;
        
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .map_err(|e| JungfernstiegError::ConfigurationError {
                message: format!("Failed to serialize configuration: {}", e),
            })?;
        
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate complete configuration
    pub fn validate(&self) -> Result<()> {
        self.system.validate()?;
        self.biological.validate()?;
        self.virtual_blood.validate()?;
        self.oscillatory_vm.validate()?;
        self.s_entropy.validate()?;
        self.safety.validate()?;
        self.hardware.validate()?;
        self.monitoring.validate()?;
        Ok(())
    }

    /// Create default configuration for development
    pub fn default_development() -> Self {
        Self {
            system: SystemIdentityConfig::default_development(),
            biological: BiologicalConfig::default_development(),
            virtual_blood: VirtualBloodConfig::default(),
            oscillatory_vm: OscillatoryVMConfig::default(),
            s_entropy: SEntropyConfig::default(),
            safety: SafetyConfig::default_development(),
            hardware: HardwareConfig::default_simulation(),
            monitoring: MonitoringConfig::default(),
        }
    }

    /// Create production configuration with enhanced safety
    pub fn default_production() -> Self {
        Self {
            system: SystemIdentityConfig::default_production(),
            biological: BiologicalConfig::default_production(),
            virtual_blood: VirtualBloodConfig::default(),
            oscillatory_vm: OscillatoryVMConfig::default(),
            s_entropy: SEntropyConfig::default(),
            safety: SafetyConfig::default_production(),
            hardware: HardwareConfig::default_hardware(),
            monitoring: MonitoringConfig::default_production(),
        }
    }
}

/// System identity and metadata configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemIdentityConfig {
    /// System name
    pub name: String,
    /// System version
    pub version: String,
    /// Environment (development, testing, production)
    pub environment: String,
    /// Memorial dedication enabled
    pub memorial_dedication: bool,
    /// St. Stella constant enabled
    pub stella_constant_enabled: bool,
}

impl SystemIdentityConfig {
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(JungfernstiegError::ValidationError {
                field: "system.name".to_string(),
                message: "System name cannot be empty".to_string(),
            });
        }
        Ok(())
    }

    fn default_development() -> Self {
        Self {
            name: "Jungfernstieg-Dev".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "development".to_string(),
            memorial_dedication: true,
            stella_constant_enabled: true,
        }
    }

    fn default_production() -> Self {
        Self {
            name: "Jungfernstieg".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "production".to_string(),
            memorial_dedication: true,
            stella_constant_enabled: true,
        }
    }
}

/// Biological neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    /// Number of neural networks to maintain
    pub neural_network_count: usize,
    /// Minimum neural viability threshold (%)
    pub min_viability_threshold: f64,
    /// Warning viability threshold (%)
    pub warning_viability_threshold: f64,
    /// Neural culture preparation parameters
    pub culture_preparation: CulturePreparationConfig,
    /// Neural maturation duration (days)
    pub maturation_duration_days: u32,
    /// Assessment frequency (seconds)
    pub assessment_frequency_secs: u64,
}

impl BiologicalConfig {
    fn validate(&self) -> Result<()> {
        if self.min_viability_threshold < 85.0 || self.min_viability_threshold > 100.0 {
            return Err(JungfernstiegError::ValidationError {
                field: "biological.min_viability_threshold".to_string(),
                message: "Minimum viability threshold must be between 85-100%".to_string(),
            });
        }
        
        if self.warning_viability_threshold <= self.min_viability_threshold {
            return Err(JungfernstiegError::ValidationError {
                field: "biological.warning_viability_threshold".to_string(),
                message: "Warning threshold must be higher than minimum threshold".to_string(),
            });
        }
        
        Ok(())
    }

    fn default_development() -> Self {
        Self {
            neural_network_count: 3,
            min_viability_threshold: 90.0,
            warning_viability_threshold: 95.0,
            culture_preparation: CulturePreparationConfig::default(),
            maturation_duration_days: 14,
            assessment_frequency_secs: 30,
        }
    }

    fn default_production() -> Self {
        Self {
            neural_network_count: 10,
            min_viability_threshold: 95.0,
            warning_viability_threshold: 97.0,
            culture_preparation: CulturePreparationConfig::default(),
            maturation_duration_days: 14,
            assessment_frequency_secs: 10,
        }
    }
}

/// Neural culture preparation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturePreparationConfig {
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
}

impl Default for CulturePreparationConfig {
    fn default() -> Self {
        Self {
            growth_factors_concentration: 1.0,
            culture_medium: "Neurobasal-A".to_string(),
            substrate_material: "PDL-Laminin".to_string(),
            temperature_celsius: 37.0,
            co2_concentration: 5.0,
        }
    }
}

/// Virtual Blood circulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBloodConfig {
    /// Circulation frequency (Hz)
    pub circulation_frequency: f64,
    /// Target oxygen transport efficiency (%)
    pub target_oxygen_efficiency: f64,
    /// Virtual Blood composition optimization
    pub composition_optimization: CompositionConfig,
    /// Filtration parameters
    pub filtration: FiltrationConfig,
}

impl VirtualBloodConfig {
    fn validate(&self) -> Result<()> {
        if self.circulation_frequency <= 0.0 || self.circulation_frequency > 100.0 {
            return Err(JungfernstiegError::ValidationError {
                field: "virtual_blood.circulation_frequency".to_string(),
                message: "Circulation frequency must be between 0-100 Hz".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for VirtualBloodConfig {
    fn default() -> Self {
        Self {
            circulation_frequency: 1.0, // 1 Hz cardiac cycle
            target_oxygen_efficiency: 98.7,
            composition_optimization: CompositionConfig::default(),
            filtration: FiltrationConfig::default(),
        }
    }
}

/// Virtual Blood composition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionConfig {
    /// Environmental profile weight
    pub environmental_weight: f64,
    /// Oxygen transport weight
    pub oxygen_weight: f64,
    /// Nutrient delivery weight
    pub nutrient_weight: f64,
    /// Metabolite removal weight
    pub metabolite_weight: f64,
    /// Immune factor weight
    pub immune_weight: f64,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            environmental_weight: 0.2,
            oxygen_weight: 0.3,
            nutrient_weight: 0.25,
            metabolite_weight: 0.15,
            immune_weight: 0.1,
        }
    }
}

/// Virtual Blood filtration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiltrationConfig {
    /// Filtration efficiency target (%)
    pub efficiency_target: f64,
    /// Waste removal rate
    pub waste_removal_rate: f64,
    /// Nutrient regeneration rate
    pub nutrient_regeneration_rate: f64,
}

impl Default for FiltrationConfig {
    fn default() -> Self {
        Self {
            efficiency_target: 95.0,
            waste_removal_rate: 0.9,
            nutrient_regeneration_rate: 0.8,
        }
    }
}

/// Oscillatory Virtual Machine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryVMConfig {
    /// Heart function parameters
    pub heart_function: HeartFunctionConfig,
    /// S-credit reserves configuration
    pub s_credit_reserves: SCreditReservesConfig,
    /// Economic coordination parameters
    pub economic_coordination: EconomicCoordinationConfig,
}

impl OscillatoryVMConfig {
    fn validate(&self) -> Result<()> {
        self.heart_function.validate()?;
        self.s_credit_reserves.validate()?;
        Ok(())
    }
}

impl Default for OscillatoryVMConfig {
    fn default() -> Self {
        Self {
            heart_function: HeartFunctionConfig::default(),
            s_credit_reserves: SCreditReservesConfig::default(),
            economic_coordination: EconomicCoordinationConfig::default(),
        }
    }
}

/// Heart function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartFunctionConfig {
    /// Cardiac cycle duration (ms)
    pub cardiac_cycle_duration_ms: u64,
    /// Systolic duration ratio (0.0-1.0)
    pub systolic_duration_ratio: f64,
    /// Diastolic duration ratio (0.0-1.0)
    pub diastolic_duration_ratio: f64,
}

impl HeartFunctionConfig {
    fn validate(&self) -> Result<()> {
        let total_ratio = self.systolic_duration_ratio + self.diastolic_duration_ratio;
        if (total_ratio - 1.0).abs() > 0.01 {
            return Err(JungfernstiegError::ValidationError {
                field: "oscillatory_vm.heart_function".to_string(),
                message: "Systolic and diastolic ratios must sum to 1.0".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for HeartFunctionConfig {
    fn default() -> Self {
        Self {
            cardiac_cycle_duration_ms: 1000, // 1 second cycle
            systolic_duration_ratio: 0.35,   // 35% systolic
            diastolic_duration_ratio: 0.65,  // 65% diastolic
        }
    }
}

/// S-credit reserves configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCreditReservesConfig {
    /// Initial reserves capacity
    pub initial_capacity: SCredits,
    /// Circulation rate (credits/second)
    pub circulation_rate: f64,
    /// Reserve warning threshold (0.0-1.0)
    pub warning_threshold: f64,
    /// Reserve critical threshold (0.0-1.0)
    pub critical_threshold: f64,
}

impl SCreditReservesConfig {
    fn validate(&self) -> Result<()> {
        if self.warning_threshold <= self.critical_threshold {
            return Err(JungfernstiegError::ValidationError {
                field: "oscillatory_vm.s_credit_reserves.thresholds".to_string(),
                message: "Warning threshold must be higher than critical threshold".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for SCreditReservesConfig {
    fn default() -> Self {
        Self {
            initial_capacity: SCredits::new(10000.0, 10000.0, 10000.0),
            circulation_rate: 1000.0,
            warning_threshold: 0.3,
            critical_threshold: 0.1,
        }
    }
}

/// Economic coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicCoordinationConfig {
    /// Economic cycle duration (ms)
    pub economic_cycle_duration_ms: u64,
    /// Demand assessment frequency (Hz)
    pub demand_assessment_frequency: f64,
    /// Supply optimization enabled
    pub supply_optimization_enabled: bool,
}

impl Default for EconomicCoordinationConfig {
    fn default() -> Self {
        Self {
            economic_cycle_duration_ms: 100, // 100ms economic cycles
            demand_assessment_frequency: 10.0, // 10 Hz
            supply_optimization_enabled: true,
        }
    }
}

/// S-entropy system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyConfig {
    /// St. Stella constant value
    pub stella_constant: f64,
    /// Navigation precision
    pub navigation_precision: f64,
    /// Coordinate caching enabled
    pub coordinate_caching_enabled: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
}

impl SEntropyConfig {
    fn validate(&self) -> Result<()> {
        if self.stella_constant <= 0.0 {
            return Err(JungfernstiegError::ValidationError {
                field: "s_entropy.stella_constant".to_string(),
                message: "St. Stella constant must be positive".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for SEntropyConfig {
    fn default() -> Self {
        Self {
            stella_constant: 1.0, // Default St. Stella constant
            navigation_precision: 1e-12,
            coordinate_caching_enabled: true,
            max_cache_size: 10000,
        }
    }
}

/// Safety and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// BSL-2+ compliance enabled
    pub bsl2_plus_compliance: bool,
    /// Safety check frequency (Hz)
    pub safety_check_frequency: f64,
    /// Emergency shutdown timeout (ms)
    pub emergency_shutdown_timeout_ms: u64,
    /// Sterile environment monitoring
    pub sterile_environment_monitoring: bool,
    /// Redundant systems enabled
    pub redundant_systems_enabled: bool,
}

impl SafetyConfig {
    fn validate(&self) -> Result<()> {
        if !self.bsl2_plus_compliance {
            return Err(JungfernstiegError::SafetyProtocolViolation {
                protocol: "BSL-2+".to_string(),
                message: "BSL-2+ compliance is mandatory for biological neural systems".to_string(),
            });
        }
        Ok(())
    }

    fn default_development() -> Self {
        Self {
            bsl2_plus_compliance: true,
            safety_check_frequency: 2.0, // 2 Hz for development
            emergency_shutdown_timeout_ms: 500,
            sterile_environment_monitoring: true,
            redundant_systems_enabled: false, // Disabled for development
        }
    }

    fn default_production() -> Self {
        Self {
            bsl2_plus_compliance: true,
            safety_check_frequency: 10.0, // 10 Hz for production
            emergency_shutdown_timeout_ms: 100,
            sterile_environment_monitoring: true,
            redundant_systems_enabled: true,
        }
    }
}

/// Hardware interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Simulation mode (no real hardware)
    pub simulation_mode: bool,
    /// Neural interface configuration
    pub neural_interfaces: NeuralInterfaceConfig,
    /// Circulation hardware configuration
    pub circulation_hardware: CirculationHardwareConfig,
    /// Environmental sensors configuration
    pub environmental_sensors: EnvironmentalSensorsConfig,
}

impl HardwareConfig {
    fn validate(&self) -> Result<()> {
        // Hardware configuration is flexible for development
        Ok(())
    }

    fn default_simulation() -> Self {
        Self {
            simulation_mode: true,
            neural_interfaces: NeuralInterfaceConfig::default_simulation(),
            circulation_hardware: CirculationHardwareConfig::default_simulation(),
            environmental_sensors: EnvironmentalSensorsConfig::default_simulation(),
        }
    }

    fn default_hardware() -> Self {
        Self {
            simulation_mode: false,
            neural_interfaces: NeuralInterfaceConfig::default_hardware(),
            circulation_hardware: CirculationHardwareConfig::default_hardware(),
            environmental_sensors: EnvironmentalSensorsConfig::default_hardware(),
        }
    }
}

/// Neural interface hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralInterfaceConfig {
    /// Electrode array configuration
    pub electrode_arrays: Vec<ElectrodeArrayConfig>,
    /// Stimulation parameters
    pub stimulation_enabled: bool,
    /// Recording sampling rate (Hz)
    pub recording_sample_rate: f64,
}

impl NeuralInterfaceConfig {
    fn default_simulation() -> Self {
        Self {
            electrode_arrays: vec![ElectrodeArrayConfig::default_simulation()],
            stimulation_enabled: false,
            recording_sample_rate: 1000.0,
        }
    }

    fn default_hardware() -> Self {
        Self {
            electrode_arrays: vec![ElectrodeArrayConfig::default_hardware()],
            stimulation_enabled: true,
            recording_sample_rate: 20000.0,
        }
    }
}

/// Electrode array configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectrodeArrayConfig {
    /// Array name/identifier
    pub name: String,
    /// Number of electrodes
    pub electrode_count: usize,
    /// Inter-electrode spacing (μm)
    pub spacing_micrometers: f64,
}

impl ElectrodeArrayConfig {
    fn default_simulation() -> Self {
        Self {
            name: "Simulation Array".to_string(),
            electrode_count: 64,
            spacing_micrometers: 200.0,
        }
    }

    fn default_hardware() -> Self {
        Self {
            name: "MEA 8x8".to_string(),
            electrode_count: 64,
            spacing_micrometers: 200.0,
        }
    }
}

/// Circulation hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CirculationHardwareConfig {
    /// Pump configuration
    pub pumps: Vec<PumpConfig>,
    /// Valve configuration
    pub valves: Vec<ValveConfig>,
    /// Sensor configuration
    pub sensors: Vec<SensorConfig>,
}

impl CirculationHardwareConfig {
    fn default_simulation() -> Self {
        Self {
            pumps: vec![PumpConfig::default_simulation()],
            valves: vec![ValveConfig::default_simulation()],
            sensors: vec![SensorConfig::default_simulation()],
        }
    }

    fn default_hardware() -> Self {
        Self {
            pumps: vec![PumpConfig::default_hardware()],
            valves: vec![ValveConfig::default_hardware()],
            sensors: vec![SensorConfig::default_hardware()],
        }
    }
}

/// Pump configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpConfig {
    /// Pump identifier
    pub id: String,
    /// Maximum flow rate (μL/min)
    pub max_flow_rate: f64,
    /// Pump type
    pub pump_type: String,
}

impl PumpConfig {
    fn default_simulation() -> Self {
        Self {
            id: "VirtualPump1".to_string(),
            max_flow_rate: 1000.0,
            pump_type: "Peristaltic-Sim".to_string(),
        }
    }

    fn default_hardware() -> Self {
        Self {
            id: "Pump1".to_string(),
            max_flow_rate: 1000.0,
            pump_type: "Peristaltic".to_string(),
        }
    }
}

/// Valve configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValveConfig {
    /// Valve identifier
    pub id: String,
    /// Valve type
    pub valve_type: String,
    /// Response time (ms)
    pub response_time_ms: u64,
}

impl ValveConfig {
    fn default_simulation() -> Self {
        Self {
            id: "VirtualValve1".to_string(),
            valve_type: "Solenoid-Sim".to_string(),
            response_time_ms: 1,
        }
    }

    fn default_hardware() -> Self {
        Self {
            id: "Valve1".to_string(),
            valve_type: "Solenoid".to_string(),
            response_time_ms: 10,
        }
    }
}

/// Sensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    /// Sensor identifier
    pub id: String,
    /// Sensor type
    pub sensor_type: String,
    /// Sampling rate (Hz)
    pub sampling_rate: f64,
}

impl SensorConfig {
    fn default_simulation() -> Self {
        Self {
            id: "VirtualSensor1".to_string(),
            sensor_type: "Pressure-Sim".to_string(),
            sampling_rate: 100.0,
        }
    }

    fn default_hardware() -> Self {
        Self {
            id: "PressureSensor1".to_string(),
            sensor_type: "Pressure".to_string(),
            sampling_rate: 1000.0,
        }
    }
}

/// Environmental sensors configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalSensorsConfig {
    /// Temperature monitoring enabled
    pub temperature_monitoring: bool,
    /// Humidity monitoring enabled
    pub humidity_monitoring: bool,
    /// CO2 monitoring enabled
    pub co2_monitoring: bool,
    /// Air quality monitoring enabled
    pub air_quality_monitoring: bool,
}

impl EnvironmentalSensorsConfig {
    fn default_simulation() -> Self {
        Self {
            temperature_monitoring: true,
            humidity_monitoring: true,
            co2_monitoring: true,
            air_quality_monitoring: false,
        }
    }

    fn default_hardware() -> Self {
        Self {
            temperature_monitoring: true,
            humidity_monitoring: true,
            co2_monitoring: true,
            air_quality_monitoring: true,
        }
    }
}

/// Monitoring and metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics collection enabled
    pub metrics_enabled: bool,
    /// Metrics collection frequency (Hz)
    pub metrics_frequency: f64,
    /// Historical data retention (hours)
    pub data_retention_hours: u64,
    /// Real-time dashboard enabled
    pub dashboard_enabled: bool,
    /// Alert thresholds configuration
    pub alert_thresholds: AlertThresholdsConfig,
}

impl MonitoringConfig {
    fn validate(&self) -> Result<()> {
        if self.metrics_frequency <= 0.0 {
            return Err(JungfernstiegError::ValidationError {
                field: "monitoring.metrics_frequency".to_string(),
                message: "Metrics frequency must be positive".to_string(),
            });
        }
        Ok(())
    }

    fn default_production() -> Self {
        Self {
            metrics_enabled: true,
            metrics_frequency: 1.0,
            data_retention_hours: 168, // 1 week
            dashboard_enabled: true,
            alert_thresholds: AlertThresholdsConfig::default_production(),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            metrics_frequency: 0.1, // 0.1 Hz for development
            data_retention_hours: 24,
            dashboard_enabled: false,
            alert_thresholds: AlertThresholdsConfig::default(),
        }
    }
}

/// Alert thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholdsConfig {
    /// Neural viability warning threshold (%)
    pub neural_viability_warning: f64,
    /// Neural viability critical threshold (%)
    pub neural_viability_critical: f64,
    /// S-credit reserves warning threshold (%)
    pub s_credit_warning: f64,
    /// S-credit reserves critical threshold (%)
    pub s_credit_critical: f64,
    /// Circulation efficiency warning threshold (%)
    pub circulation_warning: f64,
}

impl AlertThresholdsConfig {
    fn default_production() -> Self {
        Self {
            neural_viability_warning: 97.0,
            neural_viability_critical: 95.0,
            s_credit_warning: 30.0,
            s_credit_critical: 10.0,
            circulation_warning: 95.0,
        }
    }
}

impl Default for AlertThresholdsConfig {
    fn default() -> Self {
        Self {
            neural_viability_warning: 95.0,
            neural_viability_critical: 90.0,
            s_credit_warning: 30.0,
            s_credit_critical: 10.0,
            circulation_warning: 90.0,
        }
    }
}

/// Component-specific configuration trait
pub trait ComponentConfig {
    /// Validate component configuration
    fn validate(&self) -> Result<()>;
    
    /// Get component configuration name
    fn component_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_config_validation() {
        let config = SystemConfig::default_development();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_biological_config_validation() {
        let mut config = BiologicalConfig::default_development();
        
        // Test invalid viability threshold
        config.min_viability_threshold = 105.0;
        assert!(config.validate().is_err());
        
        // Test invalid threshold relationship
        config.min_viability_threshold = 95.0;
        config.warning_viability_threshold = 90.0;
        assert!(config.validate().is_err());
        
        // Test valid configuration
        config.warning_viability_threshold = 97.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_heart_function_validation() {
        let mut config = HeartFunctionConfig::default();
        
        // Test invalid ratio sum
        config.systolic_duration_ratio = 0.5;
        config.diastolic_duration_ratio = 0.6; // Sum = 1.1
        assert!(config.validate().is_err());
        
        // Test valid ratio sum
        config.diastolic_duration_ratio = 0.5; // Sum = 1.0
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_safety_config_validation() {
        let mut config = SafetyConfig::default_production();
        
        // Test BSL-2+ compliance requirement
        config.bsl2_plus_compliance = false;
        assert!(config.validate().is_err());
        
        // Test valid configuration
        config.bsl2_plus_compliance = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_file_operations() {
        let config = SystemConfig::default_development();
        let temp_file = std::env::temp_dir().join("test_config.toml");
        
        // Test save
        assert!(config.save_to_file(&temp_file).is_ok());
        
        // Test load
        let loaded_config = SystemConfig::from_file(&temp_file);
        assert!(loaded_config.is_ok());
        
        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}