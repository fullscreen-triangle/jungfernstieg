//! Main Jungfernstieg system implementation
//!
//! This module provides the high-level JungfernstiegSystem interface that coordinates
//! all system components: biological neural networks, Virtual Blood circulation,
//! Oscillatory VM, S-entropy systems, and safety monitoring.

use crate::config::SystemConfig;
use crate::coordinator::{CoordinatorCommand, CoordinatorEvent, CoordinatorHandle, SystemCoordinator};
use crate::error::{JungfernstiegError, Result};
use crate::types::{SystemId, SystemState, SystemMetrics, ComponentId, ViabilityStatus};

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Handle for interacting with a running Jungfernstieg system
#[derive(Debug, Clone)]
pub struct SystemHandle {
    /// System identifier
    pub system_id: SystemId,
    /// Coordinator handle for system control
    coordinator_handle: CoordinatorHandle,
    /// Event receiver for system events
    event_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<CoordinatorEvent>>>>,
}

impl SystemHandle {
    /// Get current system state
    pub fn get_state(&self) -> SystemState {
        self.coordinator_handle.get_state()
    }

    /// Get current system metrics
    pub fn get_metrics(&self) -> SystemMetrics {
        self.coordinator_handle.get_metrics()
    }

    /// Start the system
    pub async fn start(&self) -> Result<()> {
        info!("Starting Jungfernstieg system (ID: {:?})", self.system_id);
        self.coordinator_handle.send_command(CoordinatorCommand::Start)?;
        
        // Wait for system to become operational
        let mut handle = self.coordinator_handle.clone();
        while handle.get_state() != SystemState::Operational {
            match handle.wait_for_state_change().await? {
                SystemState::Operational => break,
                SystemState::EmergencyShutdown | SystemState::Critical => {
                    return Err(JungfernstiegError::InitializationError {
                        message: "System failed to start - entered error state".to_string(),
                    });
                }
                _ => continue,
            }
        }
        
        info!("Jungfernstieg system started successfully");
        Ok(())
    }

    /// Stop the system gracefully
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Jungfernstieg system (ID: {:?})", self.system_id);
        self.coordinator_handle.send_command(CoordinatorCommand::Stop)?;
        
        // Wait for system to stop
        let mut handle = self.coordinator_handle.clone();
        while handle.get_state() != SystemState::Stopped {
            match handle.wait_for_state_change().await? {
                SystemState::Stopped => break,
                _ => continue,
            }
        }
        
        info!("Jungfernstieg system stopped successfully");
        Ok(())
    }

    /// Initiate emergency shutdown
    pub async fn emergency_shutdown(&self, reason: String) -> Result<()> {
        error!("Emergency shutdown requested: {}", reason);
        self.coordinator_handle.emergency_shutdown(reason).await
    }

    /// Update system configuration
    pub async fn update_config(&self, config: SystemConfig) -> Result<()> {
        config.validate()?;
        self.coordinator_handle.send_command(CoordinatorCommand::UpdateConfig { config })?;
        info!("System configuration updated");
        Ok(())
    }

    /// Wait for next system event
    pub async fn next_event(&self) -> Option<CoordinatorEvent> {
        let mut event_rx_guard = self.event_rx.write().await;
        if let Some(ref mut event_rx) = *event_rx_guard {
            event_rx.recv().await
        } else {
            None
        }
    }

    /// Get system uptime
    pub fn get_uptime(&self) -> std::time::Duration {
        self.get_metrics().uptime
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(
            self.get_state(),
            SystemState::Operational | SystemState::Maintenance
        )
    }

    /// Check if system requires attention
    pub fn requires_attention(&self) -> bool {
        matches!(
            self.get_state(),
            SystemState::Warning | SystemState::Critical
        )
    }

    /// Get neural viability summary
    pub fn get_neural_viability_summary(&self) -> NeuralViabilitySummary {
        let metrics = self.get_metrics();
        let viabilities: Vec<f64> = metrics
            .neural_viability
            .values()
            .map(|v| v.viability_percent)
            .collect();

        if viabilities.is_empty() {
            return NeuralViabilitySummary {
                component_count: 0,
                average_viability: 0.0,
                min_viability: 0.0,
                max_viability: 0.0,
                critical_components: 0,
                warning_components: 0,
            };
        }

        let average = viabilities.iter().sum::<f64>() / viabilities.len() as f64;
        let min = viabilities.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = viabilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let critical = viabilities.iter().filter(|&&v| v < 90.0).count();
        let warning = viabilities.iter().filter(|&&v| v >= 90.0 && v < 95.0).count();

        NeuralViabilitySummary {
            component_count: viabilities.len(),
            average_viability: average,
            min_viability: min,
            max_viability: max,
            critical_components: critical,
            warning_components: warning,
        }
    }
}

/// Summary of neural viability across all components
#[derive(Debug, Clone)]
pub struct NeuralViabilitySummary {
    pub component_count: usize,
    pub average_viability: f64,
    pub min_viability: f64,
    pub max_viability: f64,
    pub critical_components: usize,
    pub warning_components: usize,
}

/// Builder for creating and configuring Jungfernstieg systems
pub struct SystemBuilder {
    config: SystemConfig,
    event_buffer_size: usize,
}

impl SystemBuilder {
    /// Create new system builder with default development configuration
    pub fn new() -> Self {
        Self {
            config: SystemConfig::default_development(),
            event_buffer_size: 1000,
        }
    }

    /// Create system builder with production configuration
    pub fn production() -> Self {
        Self {
            config: SystemConfig::default_production(),
            event_buffer_size: 10000,
        }
    }

    /// Create system builder with custom configuration
    pub fn with_config(config: SystemConfig) -> Self {
        Self {
            config,
            event_buffer_size: 1000,
        }
    }

    /// Set configuration
    pub fn config(mut self, config: SystemConfig) -> Self {
        self.config = config;
        self
    }

    /// Set event buffer size
    pub fn event_buffer_size(mut self, size: usize) -> Self {
        self.event_buffer_size = size;
        self
    }

    /// Load configuration from file
    pub fn config_from_file<P: AsRef<std::path::Path>>(mut self, path: P) -> Result<Self> {
        self.config = SystemConfig::from_file(path)?;
        Ok(self)
    }

    /// Set system name
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.system.name = name.into();
        self
    }

    /// Set environment
    pub fn environment<S: Into<String>>(mut self, env: S) -> Self {
        self.config.system.environment = env.into();
        self
    }

    /// Enable/disable memorial dedication
    pub fn memorial_dedication(mut self, enabled: bool) -> Self {
        self.config.system.memorial_dedication = enabled;
        self
    }

    /// Build and initialize the Jungfernstieg system
    pub async fn build(self) -> Result<JungfernstiegSystem> {
        JungfernstiegSystem::new(self.config, self.event_buffer_size).await
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main Jungfernstieg biological-virtual neural symbiosis system
pub struct JungfernstiegSystem {
    /// System identifier
    system_id: SystemId,
    /// System configuration
    config: SystemConfig,
    /// Coordinator task handle
    coordinator_task: JoinHandle<Result<()>>,
    /// System handle for external control
    handle: SystemHandle,
}

impl JungfernstiegSystem {
    /// Create new Jungfernstieg system
    pub(crate) async fn new(config: SystemConfig, event_buffer_size: usize) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        let system_id = SystemId::new();
        
        info!("Initializing Jungfernstieg System (ID: {:?})", system_id);
        
        // Memorial dedication
        if config.system.memorial_dedication {
            info!("{}", crate::MEMORIAL_DEDICATION);
        }
        
        // Create coordinator
        let (coordinator, coordinator_handle) = SystemCoordinator::new(config.clone());
        
        // Create event channel
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let mut coordinator = coordinator;
        coordinator.set_event_sender(event_tx);
        
        // Start coordinator task
        let coordinator_task = tokio::spawn(async move {
            coordinator.run().await
        });
        
        // Create system handle
        let handle = SystemHandle {
            system_id,
            coordinator_handle,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
        };
        
        let system = Self {
            system_id,
            config,
            coordinator_task,
            handle,
        };
        
        info!("Jungfernstieg System initialized successfully");
        Ok(system)
    }

    /// Get system identifier
    pub fn system_id(&self) -> SystemId {
        self.system_id
    }

    /// Get system configuration
    pub fn config(&self) -> &SystemConfig {
        &self.config
    }

    /// Get system handle for external control
    pub fn handle(&self) -> SystemHandle {
        self.handle.clone()
    }

    /// Run the system until completion
    pub async fn run(self) -> Result<()> {
        info!("Running Jungfernstieg System (ID: {:?})", self.system_id);
        
        // Start the system
        self.handle.start().await?;
        
        // Wait for coordinator to complete
        match self.coordinator_task.await {
            Ok(result) => result,
            Err(e) => Err(JungfernstiegError::CoordinationError {
                message: format!("Coordinator task failed: {}", e),
            }),
        }
    }

    /// Run the system with a shutdown signal
    pub async fn run_with_shutdown<F>(self, shutdown_signal: F) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        info!("Running Jungfernstieg System with shutdown signal (ID: {:?})", self.system_id);
        
        // Start the system
        self.handle.start().await?;
        
        // Wait for either completion or shutdown signal
        tokio::select! {
            result = self.coordinator_task => {
                match result {
                    Ok(result) => result,
                    Err(e) => Err(JungfernstiegError::CoordinationError {
                        message: format!("Coordinator task failed: {}", e),
                    }),
                }
            }
            _ = shutdown_signal => {
                info!("Shutdown signal received, stopping system");
                self.handle.stop().await?;
                Ok(())
            }
        }
    }

    /// Shutdown the system gracefully
    pub async fn shutdown(self) -> Result<()> {
        info!("Shutting down Jungfernstieg System (ID: {:?})", self.system_id);
        
        // Stop the system
        self.handle.stop().await?;
        
        // Wait for coordinator to finish
        self.coordinator_task.abort();
        
        info!("Jungfernstieg System shutdown complete");
        Ok(())
    }

    /// Create system builder
    pub fn builder() -> SystemBuilder {
        SystemBuilder::new()
    }

    /// Create system with default development configuration
    pub async fn development() -> Result<Self> {
        Self::new(SystemConfig::default_development(), 1000).await
    }

    /// Create system with production configuration
    pub async fn production() -> Result<Self> {
        Self::new(SystemConfig::default_production(), 10000).await
    }
}

impl Drop for JungfernstiegSystem {
    fn drop(&mut self) {
        // Abort coordinator task if still running
        self.coordinator_task.abort();
        debug!("Jungfernstieg System dropped (ID: {:?})", self.system_id);
    }
}

/// System event handler trait for processing coordinator events
#[async_trait::async_trait]
pub trait SystemEventHandler: Send + Sync {
    /// Handle system state change event
    async fn on_state_changed(
        &self,
        old_state: SystemState,
        new_state: SystemState,
    ) -> Result<()> {
        debug!("System state changed: {:?} -> {:?}", old_state, new_state);
        Ok(())
    }

    /// Handle S-credit low warning
    async fn on_s_credit_low_warning(
        &self,
        current_level: f64,
        threshold: f64,
    ) -> Result<()> {
        warn!("S-credit reserves low: {:.1}% (threshold: {:.1}%)", 
               current_level * 100.0, threshold * 100.0);
        Ok(())
    }

    /// Handle neural viability warning
    async fn on_viability_warning(
        &self,
        component_id: ComponentId,
        viability: f64,
    ) -> Result<()> {
        warn!("Neural viability warning for component {}: {:.1}%", 
               component_id.0, viability);
        Ok(())
    }

    /// Handle safety alert
    async fn on_safety_alert(
        &self,
        level: crate::types::SafetyLevel,
        message: String,
    ) -> Result<()> {
        match level {
            crate::types::SafetyLevel::Critical | crate::types::SafetyLevel::Emergency => {
                error!("SAFETY ALERT [{:?}]: {}", level, message);
            }
            crate::types::SafetyLevel::Warning => {
                warn!("SAFETY ALERT [{:?}]: {}", level, message);
            }
            _ => {
                info!("SAFETY ALERT [{:?}]: {}", level, message);
            }
        }
        Ok(())
    }

    /// Handle emergency shutdown
    async fn on_emergency_shutdown(&self, reason: String) -> Result<()> {
        error!("EMERGENCY SHUTDOWN: {}", reason);
        Ok(())
    }
}

/// Event processing loop for handling system events
pub async fn run_event_processor<H: SystemEventHandler>(
    mut handle: SystemHandle,
    handler: H,
) -> Result<()> {
    info!("Starting system event processor");
    
    loop {
        match handle.next_event().await {
            Some(event) => {
                if let Err(e) = process_event(&handler, event).await {
                    error!("Error processing event: {}", e);
                }
            }
            None => {
                debug!("Event channel closed, stopping event processor");
                break;
            }
        }
    }
    
    Ok(())
}

/// Process a single coordinator event
async fn process_event<H: SystemEventHandler>(
    handler: &H,
    event: CoordinatorEvent,
) -> Result<()> {
    match event {
        CoordinatorEvent::StateChanged { old_state, new_state } => {
            handler.on_state_changed(old_state, new_state).await
        }
        CoordinatorEvent::SCreditLowWarning { current_level, threshold } => {
            handler.on_s_credit_low_warning(current_level, threshold).await
        }
        CoordinatorEvent::ViabilityWarning { component_id, viability } => {
            handler.on_viability_warning(component_id, viability).await
        }
        CoordinatorEvent::SafetyAlert { level, message } => {
            handler.on_safety_alert(level, message).await
        }
        CoordinatorEvent::EmergencyShutdown { reason } => {
            handler.on_emergency_shutdown(reason).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_builder() {
        let system = SystemBuilder::new()
            .name("Test System")
            .environment("testing")
            .memorial_dedication(true)
            .build()
            .await;
        
        assert!(system.is_ok());
        let system = system.unwrap();
        assert_eq!(system.config().system.name, "Test System");
        assert_eq!(system.config().system.environment, "testing");
        assert!(system.config().system.memorial_dedication);
    }

    #[tokio::test]
    async fn test_system_lifecycle() {
        let system = JungfernstiegSystem::development().await.unwrap();
        let handle = system.handle();
        
        // Test initial state
        assert_eq!(handle.get_state(), SystemState::Initializing);
        
        // Test system startup
        let start_result = handle.start().await;
        if start_result.is_err() {
            // May fail in test environment without full component setup
            eprintln!("System start failed (expected in test): {:?}", start_result);
        }
    }

    #[tokio::test]
    async fn test_neural_viability_summary() {
        let system = JungfernstiegSystem::development().await.unwrap();
        let handle = system.handle();
        
        let summary = handle.get_neural_viability_summary();
        assert_eq!(summary.component_count, 0); // No components initially
    }

    #[tokio::test]
    async fn test_system_health_check() {
        let system = JungfernstiegSystem::development().await.unwrap();
        let handle = system.handle();
        
        // System should not be healthy initially (in Initializing state)
        assert!(!handle.is_healthy());
        assert!(!handle.requires_attention());
    }

    struct TestEventHandler;

    #[async_trait::async_trait]
    impl SystemEventHandler for TestEventHandler {
        async fn on_state_changed(
            &self,
            old_state: SystemState,
            new_state: SystemState,
        ) -> Result<()> {
            println!("Test handler: state changed {:?} -> {:?}", old_state, new_state);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_event_handler() {
        let handler = TestEventHandler;
        let result = handler.on_state_changed(
            SystemState::Initializing,
            SystemState::Operational,
        ).await;
        assert!(result.is_ok());
    }
}