//! Error types for Jungfernstieg biological-virtual neural symbiosis system

use thiserror::Error;
use crate::types::{ComponentId, SystemId};

/// Result type for Jungfernstieg operations
pub type Result<T> = std::result::Result<T, JungfernstiegError>;

/// Comprehensive error types for the Jungfernstieg system
#[derive(Error, Debug)]
pub enum JungfernstiegError {
    /// System initialization errors
    #[error("System initialization failed: {message}")]
    InitializationError { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Biological neural network errors
    #[error("Biological neural network error in component {component_id}: {message}")]
    BiologicalError {
        component_id: ComponentId,
        message: String,
    },

    /// Virtual Blood circulation errors
    #[error("Virtual Blood circulation error: {message}")]
    CirculationError { message: String },

    /// Oscillatory VM errors
    #[error("Oscillatory VM error: {message}")]
    OscillatoryVMError { message: String },

    /// S-entropy system errors
    #[error("S-entropy system error: {message}")]
    SEntropyError { message: String },

    /// S-credit circulation errors
    #[error("S-credit circulation error: {message}")]
    SCreditError { message: String },

    /// Neural viability errors
    #[error("Neural viability critical in component {component_id}: viability {viability_percent}% below threshold")]
    ViabilityError {
        component_id: ComponentId,
        viability_percent: f64,
    },

    /// Safety system errors
    #[error("Safety system error: {message}")]
    SafetyError { message: String },

    /// Emergency shutdown errors
    #[error("Emergency shutdown initiated: {reason}")]
    EmergencyShutdown { reason: String },

    /// Immune cell monitoring errors
    #[error("Immune cell monitoring error in component {component_id}: {message}")]
    ImmuneMonitoringError {
        component_id: ComponentId,
        message: String,
    },

    /// Memory cell learning errors
    #[error("Memory cell learning error: {message}")]
    MemoryCellError { message: String },

    /// System coordination errors
    #[error("System coordination error: {message}")]
    CoordinationError { message: String },

    /// Component communication errors
    #[error("Component communication error between {from} and {to}: {message}")]
    CommunicationError {
        from: ComponentId,
        to: ComponentId,
        message: String,
    },

    /// Resource allocation errors
    #[error("Resource allocation error: {message}")]
    ResourceError { message: String },

    /// Monitoring system errors
    #[error("Monitoring system error: {message}")]
    MonitoringError { message: String },

    /// Hardware interface errors
    #[error("Hardware interface error: {message}")]
    HardwareError { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Configuration parsing errors
    #[error("Configuration parsing error: {0}")]
    ConfigError(#[from] config::ConfigError),

    /// Generic errors with context
    #[error("System error: {0}")]
    GenericError(#[from] anyhow::Error),

    /// Mathematical computation errors
    #[error("Mathematical computation error: {message}")]
    MathError { message: String },

    /// Timeout errors
    #[error("Operation timeout: {operation} exceeded {timeout_ms}ms")]
    TimeoutError {
        operation: String,
        timeout_ms: u64,
    },

    /// Validation errors
    #[error("Validation error: {field} failed validation: {message}")]
    ValidationError {
        field: String,
        message: String,
    },

    /// System state errors
    #[error("Invalid system state transition from {from:?} to {to:?}")]
    StateTransitionError {
        from: crate::types::SystemState,
        to: crate::types::SystemState,
    },

    /// BSL-2+ safety protocol violations
    #[error("BSL-2+ safety protocol violation: {protocol} - {message}")]
    SafetyProtocolViolation {
        protocol: String,
        message: String,
    },

    /// Sterile environment maintenance errors
    #[error("Sterile environment compromise detected: {message}")]
    SterileEnvironmentError { message: String },

    /// Neural tissue protection errors
    #[error("Neural tissue protection failure: {message}")]
    NeuralProtectionError { message: String },
}

impl JungfernstiegError {
    /// Check if error is critical and requires emergency shutdown
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            JungfernstiegError::ViabilityError { viability_percent, .. } if *viability_percent < 90.0
                | JungfernstiegError::EmergencyShutdown { .. }
                | JungfernstiegError::SafetyProtocolViolation { .. }
                | JungfernstiegError::SterileEnvironmentError { .. }
                | JungfernstiegError::NeuralProtectionError { .. }
        )
    }

    /// Check if error requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(
            self,
            JungfernstiegError::ViabilityError { viability_percent, .. } if *viability_percent < 95.0
                | JungfernstiegError::SafetyError { .. }
                | JungfernstiegError::CirculationError { .. }
                | JungfernstiegError::OscillatoryVMError { .. }
        ) || self.is_critical()
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        if self.is_critical() {
            ErrorSeverity::Critical
        } else if self.requires_immediate_attention() {
            ErrorSeverity::High
        } else {
            match self {
                JungfernstiegError::ConfigurationError { .. }
                | JungfernstiegError::ValidationError { .. }
                | JungfernstiegError::SerializationError(..)
                | JungfernstiegError::ConfigError(..) => ErrorSeverity::Low,
                
                JungfernstiegError::MonitoringError { .. }
                | JungfernstiegError::CommunicationError { .. }
                | JungfernstiegError::HardwareError { .. } => ErrorSeverity::Medium,
                
                _ => ErrorSeverity::High,
            }
        }
    }

    /// Create a new biological error
    pub fn biological(component_id: ComponentId, message: impl Into<String>) -> Self {
        Self::BiologicalError {
            component_id,
            message: message.into(),
        }
    }

    /// Create a new circulation error
    pub fn circulation(message: impl Into<String>) -> Self {
        Self::CirculationError {
            message: message.into(),
        }
    }

    /// Create a new safety error
    pub fn safety(message: impl Into<String>) -> Self {
        Self::SafetyError {
            message: message.into(),
        }
    }

    /// Create a new viability error
    pub fn viability(component_id: ComponentId, viability_percent: f64) -> Self {
        Self::ViabilityError {
            component_id,
            viability_percent,
        }
    }

    /// Create an emergency shutdown error
    pub fn emergency_shutdown(reason: impl Into<String>) -> Self {
        Self::EmergencyShutdown {
            reason: reason.into(),
        }
    }
}

/// Error severity levels for prioritization and response
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - informational or minor issues
    Low,
    /// Medium severity - operational impact but not critical
    Medium,
    /// High severity - significant impact requiring prompt attention
    High,
    /// Critical severity - system-threatening requiring immediate emergency response
    Critical,
}

/// Error context for enhanced debugging and monitoring
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// System ID where error occurred
    pub system_id: Option<SystemId>,
    /// Component ID where error occurred
    pub component_id: Option<ComponentId>,
    /// Operation being performed when error occurred
    pub operation: Option<String>,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    /// Error timestamp
    pub timestamp: std::time::Instant,
}

impl ErrorContext {
    /// Create new error context
    pub fn new() -> Self {
        Self {
            system_id: None,
            component_id: None,
            operation: None,
            context: std::collections::HashMap::new(),
            timestamp: std::time::Instant::now(),
        }
    }

    /// Set system ID context
    pub fn with_system_id(mut self, system_id: SystemId) -> Self {
        self.system_id = Some(system_id);
        self
    }

    /// Set component ID context
    pub fn with_component_id(mut self, component_id: ComponentId) -> Self {
        self.component_id = Some(component_id);
        self
    }

    /// Set operation context
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    /// Add context information
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ComponentId;

    #[test]
    fn test_error_severity_classification() {
        let critical_error = JungfernstiegError::viability(ComponentId::new(), 85.0);
        assert_eq!(critical_error.severity(), ErrorSeverity::Critical);
        assert!(critical_error.is_critical());

        let config_error = JungfernstiegError::ConfigurationError {
            message: "test".to_string(),
        };
        assert_eq!(config_error.severity(), ErrorSeverity::Low);
        assert!(!config_error.is_critical());
    }

    #[test]
    fn test_error_context_building() {
        let system_id = crate::types::SystemId::new();
        let component_id = ComponentId::new();
        
        let context = ErrorContext::new()
            .with_system_id(system_id)
            .with_component_id(component_id)
            .with_operation("test_operation")
            .with_context("test_key", "test_value");

        assert_eq!(context.system_id, Some(system_id));
        assert_eq!(context.component_id, Some(component_id));
        assert_eq!(context.operation, Some("test_operation".to_string()));
        assert_eq!(context.context.get("test_key"), Some(&"test_value".to_string()));
    }

    #[test]
    fn test_viability_error_detection() {
        let low_viability = JungfernstiegError::viability(ComponentId::new(), 75.0);
        let medium_viability = JungfernstiegError::viability(ComponentId::new(), 92.0);
        let good_viability = JungfernstiegError::viability(ComponentId::new(), 98.0);

        assert!(low_viability.is_critical());
        assert!(medium_viability.requires_immediate_attention());
        assert!(!medium_viability.is_critical());
        assert!(good_viability.requires_immediate_attention());
    }
}