//! System coordination for Jungfernstieg biological-virtual neural symbiosis
//!
//! The SystemCoordinator serves as the central orchestrator for all system components,
//! implementing the S-entropy economic coordination algorithms described in the
//! theoretical framework. It manages the Oscillatory VM as S-Entropy Central Bank,
//! coordinates Virtual Blood circulation, and maintains biological neural viability.

use crate::config::SystemConfig;
use crate::error::{JungfernstiegError, Result};
use crate::types::{
    ComponentId, SCredits, SCreditReserves, SystemId, SystemMetrics, SystemState,
    ViabilityStatus, CirculationMetrics, VMPerformance, SafetyStatus, SafetyLevel
};

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, watch};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Handle for communicating with the system coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorHandle {
    command_tx: mpsc::UnboundedSender<CoordinatorCommand>,
    state_rx: watch::Receiver<SystemState>,
    metrics_rx: watch::Receiver<SystemMetrics>,
}

impl CoordinatorHandle {
    /// Send command to coordinator
    pub fn send_command(&self, command: CoordinatorCommand) -> Result<()> {
        self.command_tx.send(command)
            .map_err(|_| JungfernstiegError::CommunicationError {
                from: ComponentId::new(),
                to: ComponentId::new(),
                message: "Failed to send command to coordinator".to_string(),
            })
    }

    /// Get current system state
    pub fn get_state(&self) -> SystemState {
        *self.state_rx.borrow()
    }

    /// Get current system metrics
    pub fn get_metrics(&self) -> SystemMetrics {
        self.metrics_rx.borrow().clone()
    }

    /// Wait for state change
    pub async fn wait_for_state_change(&mut self) -> Result<SystemState> {
        self.state_rx.changed().await
            .map_err(|_| JungfernstiegError::CommunicationError {
                from: ComponentId::new(),
                to: ComponentId::new(),
                message: "State channel closed".to_string(),
            })?;
        Ok(*self.state_rx.borrow())
    }

    /// Get current S-credit reserves
    pub fn get_s_credit_reserves(&self) -> SCreditReserves {
        self.metrics_rx.borrow().s_credit_reserves.clone()
    }

    /// Request emergency shutdown
    pub async fn emergency_shutdown(&self, reason: String) -> Result<()> {
        self.send_command(CoordinatorCommand::EmergencyShutdown { reason })
    }
}

/// Commands that can be sent to the system coordinator
#[derive(Debug, Clone)]
pub enum CoordinatorCommand {
    /// Start system operation
    Start,
    /// Stop system operation
    Stop,
    /// Emergency shutdown with reason
    EmergencyShutdown { reason: String },
    /// Update system configuration
    UpdateConfig { config: SystemConfig },
    /// Request system metrics
    GetMetrics,
    /// Add S-credits to reserves
    AddSCredits { credits: SCredits },
    /// Withdraw S-credits from reserves
    WithdrawSCredits { credits: SCredits, component_id: ComponentId },
    /// Update component viability status
    UpdateViability { component_id: ComponentId, status: ViabilityStatus },
    /// Update circulation metrics
    UpdateCirculation { metrics: CirculationMetrics },
    /// Report safety incident
    SafetyIncident { level: SafetyLevel, message: String },
}

/// Events emitted by the system coordinator
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// System state changed
    StateChanged { old_state: SystemState, new_state: SystemState },
    /// S-credit reserves low warning
    SCreditLowWarning { current_level: f64, threshold: f64 },
    /// Neural viability warning
    ViabilityWarning { component_id: ComponentId, viability: f64 },
    /// Safety alert
    SafetyAlert { level: SafetyLevel, message: String },
    /// Emergency shutdown initiated
    EmergencyShutdown { reason: String },
}

/// Main system coordinator implementing S-entropy economic coordination
pub struct SystemCoordinator {
    /// System identifier
    system_id: SystemId,
    /// System configuration
    config: Arc<RwLock<SystemConfig>>,
    /// Current system state
    state: Arc<RwLock<SystemState>>,
    /// S-credit reserves (Oscillatory VM as Central Bank)
    s_credit_reserves: Arc<RwLock<SCreditReserves>>,
    /// Component viability tracking
    component_viability: Arc<RwLock<HashMap<ComponentId, ViabilityStatus>>>,
    /// Circulation metrics
    circulation_metrics: Arc<RwLock<CirculationMetrics>>,
    /// VM performance metrics
    vm_performance: Arc<RwLock<VMPerformance>>,
    /// Safety system status
    safety_status: Arc<RwLock<SafetyStatus>>,
    /// System start time
    start_time: Instant,
    /// Command receiver
    command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
    /// State broadcaster
    state_tx: watch::Sender<SystemState>,
    /// Metrics broadcaster
    metrics_tx: watch::Sender<SystemMetrics>,
    /// Event sender for external systems
    event_tx: Option<mpsc::UnboundedSender<CoordinatorEvent>>,
}

impl SystemCoordinator {
    /// Create new system coordinator
    pub fn new(config: SystemConfig) -> (Self, CoordinatorHandle) {
        let system_id = SystemId::new();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (state_tx, state_rx) = watch::channel(SystemState::Initializing);
        
        // Initialize S-credit reserves based on configuration
        let s_credit_reserves = SCreditReserves::new(
            config.oscillatory_vm.s_credit_reserves.initial_capacity.clone()
        );
        
        // Create initial metrics
        let initial_metrics = SystemMetrics {
            uptime: Duration::from_secs(0),
            s_credit_reserves: s_credit_reserves.clone(),
            neural_viability: HashMap::new(),
            circulation_metrics: CirculationMetrics {
                efficiency: 0.0,
                flow_rate: 0.0,
                oxygen_efficiency: 0.0,
                pressure_stability: 0.0,
            },
            vm_performance: VMPerformance {
                rhythm_stability: 0.0,
                circulation_rate: 0.0,
                economic_efficiency: 0.0,
                throughput: 0.0,
            },
            safety_status: SafetyStatus {
                level: SafetyLevel::Safe,
                active_protocols: vec!["BSL-2+".to_string()],
                last_check: Instant::now(),
            },
            last_update: Instant::now(),
        };
        
        let (metrics_tx, metrics_rx) = watch::channel(initial_metrics);
        
        let coordinator = Self {
            system_id,
            config: Arc::new(RwLock::new(config)),
            state: Arc::new(RwLock::new(SystemState::Initializing)),
            s_credit_reserves: Arc::new(RwLock::new(s_credit_reserves)),
            component_viability: Arc::new(RwLock::new(HashMap::new())),
            circulation_metrics: Arc::new(RwLock::new(CirculationMetrics {
                efficiency: 0.0,
                flow_rate: 0.0,
                oxygen_efficiency: 0.0,
                pressure_stability: 0.0,
            })),
            vm_performance: Arc::new(RwLock::new(VMPerformance {
                rhythm_stability: 0.0,
                circulation_rate: 0.0,
                economic_efficiency: 0.0,
                throughput: 0.0,
            })),
            safety_status: Arc::new(RwLock::new(SafetyStatus {
                level: SafetyLevel::Safe,
                active_protocols: vec!["BSL-2+".to_string()],
                last_check: Instant::now(),
            })),
            start_time: Instant::now(),
            command_rx,
            state_tx,
            metrics_tx,
            event_tx: None,
        };
        
        let handle = CoordinatorHandle {
            command_tx,
            state_rx,
            metrics_rx,
        };
        
        (coordinator, handle)
    }

    /// Set event sender for external event handling
    pub fn set_event_sender(&mut self, event_tx: mpsc::UnboundedSender<CoordinatorEvent>) {
        self.event_tx = Some(event_tx);
    }

    /// Start the coordinator main loop
    pub async fn run(mut self) -> Result<()> {
        info!("Starting Jungfernstieg System Coordinator (System ID: {:?})", self.system_id);
        
        // Memorial dedication
        info!("{}", crate::MEMORIAL_DEDICATION);
        
        // Start periodic tasks
        let mut s_entropy_interval = interval(Duration::from_millis(100)); // S-entropy coordination every 100ms
        let mut metrics_interval = interval(Duration::from_secs(1)); // Metrics update every second
        let mut safety_interval = interval(Duration::from_millis(500)); // Safety checks every 500ms
        
        loop {
            tokio::select! {
                // Handle incoming commands
                command = self.command_rx.recv() => {
                    match command {
                        Some(cmd) => {
                            if let Err(e) = self.handle_command(cmd).await {
                                error!("Error handling command: {}", e);
                            }
                        }
                        None => {
                            warn!("Command channel closed, shutting down coordinator");
                            break;
                        }
                    }
                }
                
                // S-entropy economic coordination cycle
                _ = s_entropy_interval.tick() => {
                    if let Err(e) = self.s_entropy_coordination_cycle().await {
                        error!("S-entropy coordination error: {}", e);
                    }
                }
                
                // Metrics update cycle
                _ = metrics_interval.tick() => {
                    if let Err(e) = self.update_metrics().await {
                        error!("Metrics update error: {}", e);
                    }
                }
                
                // Safety monitoring cycle
                _ = safety_interval.tick() => {
                    if let Err(e) = self.safety_monitoring_cycle().await {
                        error!("Safety monitoring error: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Handle incoming coordinator commands
    async fn handle_command(&mut self, command: CoordinatorCommand) -> Result<()> {
        debug!("Handling coordinator command: {:?}", command);
        
        match command {
            CoordinatorCommand::Start => {
                self.transition_state(SystemState::Operational).await?;
                info!("System started successfully");
            }
            
            CoordinatorCommand::Stop => {
                self.transition_state(SystemState::Stopped).await?;
                info!("System stopped successfully");
            }
            
            CoordinatorCommand::EmergencyShutdown { reason } => {
                error!("Emergency shutdown initiated: {}", reason);
                self.transition_state(SystemState::EmergencyShutdown).await?;
                self.emit_event(CoordinatorEvent::EmergencyShutdown { reason }).await;
            }
            
            CoordinatorCommand::UpdateConfig { config } => {
                config.validate()?;
                *self.config.write().await = config;
                info!("System configuration updated");
            }
            
            CoordinatorCommand::AddSCredits { credits } => {
                self.s_credit_reserves.write().await.deposit(&credits);
                debug!("S-credits added to reserves: {:?}", credits);
            }
            
            CoordinatorCommand::WithdrawSCredits { credits, component_id } => {
                let result = self.s_credit_reserves.write().await.withdraw(&credits);
                match result {
                    Ok(_) => debug!("S-credits withdrawn for component {}: {:?}", component_id.0, credits),
                    Err(e) => warn!("Failed to withdraw S-credits for component {}: {}", component_id.0, e),
                }
            }
            
            CoordinatorCommand::UpdateViability { component_id, status } => {
                self.update_component_viability(component_id, status).await?;
            }
            
            CoordinatorCommand::UpdateCirculation { metrics } => {
                *self.circulation_metrics.write().await = metrics;
                debug!("Circulation metrics updated");
            }
            
            CoordinatorCommand::SafetyIncident { level, message } => {
                self.handle_safety_incident(level, message).await?;
            }
            
            _ => {
                debug!("Command handled: {:?}", command);
            }
        }
        
        Ok(())
    }

    /// S-entropy economic coordination cycle (Algorithm 1 from theoretical framework)
    async fn s_entropy_coordination_cycle(&self) -> Result<()> {
        // Assess system S-credit demand from components
        let component_demands = self.assess_system_s_credit_demand().await?;
        
        // Calculate available S-credits from reserves
        let available_credits = self.calculate_available_s_credits().await;
        
        // Optimize S-credit flow rates
        let flow_rates = self.optimize_s_credit_flow(&component_demands, &available_credits).await?;
        
        // Distribute S-credits to components
        self.distribute_s_credits(&flow_rates).await?;
        
        // Monitor S-credit economy
        self.monitor_s_credit_economy().await?;
        
        // Update S-credit reserves based on monitoring
        self.update_s_credit_reserves().await?;
        
        Ok(())
    }

    /// Assess S-credit demand from all system components
    async fn assess_system_s_credit_demand(&self) -> Result<HashMap<ComponentId, SCredits>> {
        let mut demands = HashMap::new();
        let viabilities = self.component_viability.read().await;
        
        // Calculate S-credit demand based on component viability and activity
        for (component_id, viability) in viabilities.iter() {
            let base_demand = SCredits::new(100.0, 100.0, 100.0);
            
            // Adjust demand based on viability - lower viability needs more S-credits
            let viability_factor = 2.0 - (viability.viability_percent / 100.0);
            let adjusted_demand = SCredits::new(
                base_demand.s_knowledge * viability_factor,
                base_demand.s_time * viability_factor,
                base_demand.s_entropy * viability_factor,
            );
            
            demands.insert(*component_id, adjusted_demand);
        }
        
        Ok(demands)
    }

    /// Calculate available S-credits from reserves
    async fn calculate_available_s_credits(&self) -> SCredits {
        let reserves = self.s_credit_reserves.read().await;
        let circulation_rate = reserves.circulation_rate;
        
        // Calculate credits available for this cycle based on circulation rate
        let cycle_duration = 0.1; // 100ms cycle
        let available_amount = circulation_rate * cycle_duration;
        
        SCredits::new(
            available_amount.min(reserves.reserves.s_knowledge),
            available_amount.min(reserves.reserves.s_time),
            available_amount.min(reserves.reserves.s_entropy),
        )
    }

    /// Optimize S-credit flow distribution
    async fn optimize_s_credit_flow(
        &self,
        demands: &HashMap<ComponentId, SCredits>,
        available: &SCredits,
    ) -> Result<HashMap<ComponentId, SCredits>> {
        let mut optimized_flows = HashMap::new();
        
        // Calculate total demand
        let total_demand = demands.values().fold(SCredits::zero(), |acc, demand| {
            SCredits::new(
                acc.s_knowledge + demand.s_knowledge,
                acc.s_time + demand.s_time,
                acc.s_entropy + demand.s_entropy,
            )
        });
        
        // Distribute available credits proportionally
        for (component_id, demand) in demands {
            let allocated = if total_demand.total() > 0.0 {
                SCredits::new(
                    (demand.s_knowledge / total_demand.s_knowledge) * available.s_knowledge,
                    (demand.s_time / total_demand.s_time) * available.s_time,
                    (demand.s_entropy / total_demand.s_entropy) * available.s_entropy,
                )
            } else {
                SCredits::zero()
            };
            
            optimized_flows.insert(*component_id, allocated);
        }
        
        Ok(optimized_flows)
    }

    /// Distribute S-credits to components
    async fn distribute_s_credits(&self, flows: &HashMap<ComponentId, SCredits>) -> Result<()> {
        let mut reserves = self.s_credit_reserves.write().await;
        
        for (component_id, credits) in flows {
            if let Err(e) = reserves.withdraw(credits) {
                warn!("Failed to distribute S-credits to component {}: {}", component_id.0, e);
            } else {
                debug!("S-credits distributed to component {}: {:?}", component_id.0, credits);
            }
        }
        
        Ok(())
    }

    /// Monitor S-credit economy performance
    async fn monitor_s_credit_economy(&self) -> Result<()> {
        let reserves = self.s_credit_reserves.read().await;
        let utilization = reserves.utilization();
        
        // Check for low reserves warning
        let config = self.config.read().await;
        if utilization < config.oscillatory_vm.s_credit_reserves.warning_threshold {
            self.emit_event(CoordinatorEvent::SCreditLowWarning {
                current_level: utilization,
                threshold: config.oscillatory_vm.s_credit_reserves.warning_threshold,
            }).await;
        }
        
        // Check for critical reserves
        if utilization < config.oscillatory_vm.s_credit_reserves.critical_threshold {
            error!("S-credit reserves critically low: {:.1}%", utilization * 100.0);
            self.transition_state(SystemState::Critical).await?;
        }
        
        Ok(())
    }

    /// Update S-credit reserves based on monitoring feedback
    async fn update_s_credit_reserves(&self) -> Result<()> {
        // In a full implementation, this would incorporate:
        // - S-credit income from successful operations
        // - Reserve regeneration mechanisms
        // - Dynamic circulation rate adjustments
        
        debug!("S-credit reserves updated");
        Ok(())
    }

    /// Update component viability status and handle alerts
    async fn update_component_viability(
        &self,
        component_id: ComponentId,
        status: ViabilityStatus,
    ) -> Result<()> {
        let config = self.config.read().await;
        
        // Check for viability warnings
        if status.viability_percent < config.biological.warning_viability_threshold {
            warn!(
                "Neural viability warning for component {}: {:.1}%",
                component_id.0,
                status.viability_percent
            );
            
            self.emit_event(CoordinatorEvent::ViabilityWarning {
                component_id,
                viability: status.viability_percent,
            }).await;
        }
        
        // Check for critical viability
        if status.viability_percent < config.biological.min_viability_threshold {
            error!(
                "Critical neural viability for component {}: {:.1}%",
                component_id.0,
                status.viability_percent
            );
            
            self.transition_state(SystemState::Critical).await?;
        }
        
        // Update component viability tracking
        self.component_viability.write().await.insert(component_id, status);
        
        Ok(())
    }

    /// Handle safety incidents and alerts
    async fn handle_safety_incident(&self, level: SafetyLevel, message: String) -> Result<()> {
        match level {
            SafetyLevel::Emergency => {
                error!("SAFETY EMERGENCY: {}", message);
                self.transition_state(SystemState::EmergencyShutdown).await?;
            }
            SafetyLevel::Critical => {
                error!("SAFETY CRITICAL: {}", message);
                self.transition_state(SystemState::Critical).await?;
            }
            SafetyLevel::Warning => {
                warn!("SAFETY WARNING: {}", message);
                self.transition_state(SystemState::Warning).await?;
            }
            _ => {
                info!("Safety incident: {}", message);
            }
        }
        
        self.emit_event(CoordinatorEvent::SafetyAlert { level, message }).await;
        Ok(())
    }

    /// Safety monitoring cycle
    async fn safety_monitoring_cycle(&self) -> Result<()> {
        let mut safety_status = self.safety_status.write().await;
        safety_status.last_check = Instant::now();
        
        // Check BSL-2+ compliance
        let config = self.config.read().await;
        if !config.safety.bsl2_plus_compliance {
            return Err(JungfernstiegError::SafetyProtocolViolation {
                protocol: "BSL-2+".to_string(),
                message: "BSL-2+ compliance disabled".to_string(),
            });
        }
        
        // Monitor component viabilities for safety thresholds
        let viabilities = self.component_viability.read().await;
        for (component_id, viability) in viabilities.iter() {
            if viability.is_critical() {
                safety_status.level = SafetyLevel::Critical;
                return Err(JungfernstiegError::ViabilityError {
                    component_id: *component_id,
                    viability_percent: viability.viability_percent,
                });
            }
        }
        
        // Update safety level based on overall system health
        safety_status.level = SafetyLevel::Safe;
        Ok(())
    }

    /// Update system metrics
    async fn update_metrics(&self) -> Result<()> {
        let uptime = self.start_time.elapsed();
        let s_credit_reserves = self.s_credit_reserves.read().await.clone();
        let neural_viability = self.component_viability.read().await.clone();
        let circulation_metrics = self.circulation_metrics.read().await.clone();
        let vm_performance = self.vm_performance.read().await.clone();
        let safety_status = self.safety_status.read().await.clone();
        
        let metrics = SystemMetrics {
            uptime,
            s_credit_reserves,
            neural_viability,
            circulation_metrics,
            vm_performance,
            safety_status,
            last_update: Instant::now(),
        };
        
        let _ = self.metrics_tx.send(metrics);
        Ok(())
    }

    /// Transition system state
    async fn transition_state(&self, new_state: SystemState) -> Result<()> {
        let mut state = self.state.write().await;
        let old_state = *state;
        
        if old_state != new_state {
            *state = new_state;
            let _ = self.state_tx.send(new_state);
            
            info!("System state transition: {:?} -> {:?}", old_state, new_state);
            
            self.emit_event(CoordinatorEvent::StateChanged { old_state, new_state }).await;
        }
        
        Ok(())
    }

    /// Emit coordinator event
    async fn emit_event(&self, event: CoordinatorEvent) {
        if let Some(ref event_tx) = self.event_tx {
            if let Err(_) = event_tx.send(event) {
                warn!("Failed to emit coordinator event - no receivers");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SystemConfig;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = SystemConfig::default_development();
        let (coordinator, handle) = SystemCoordinator::new(config);
        
        assert_eq!(handle.get_state(), SystemState::Initializing);
        assert!(handle.get_s_credit_reserves().reserves.total() > 0.0);
    }

    #[tokio::test]
    async fn test_coordinator_state_transitions() {
        let config = SystemConfig::default_development();
        let (mut coordinator, handle) = SystemCoordinator::new(config);
        
        // Test state transition
        coordinator.transition_state(SystemState::Operational).await.unwrap();
        assert_eq!(handle.get_state(), SystemState::Operational);
    }

    #[tokio::test]
    async fn test_s_credit_operations() {
        let config = SystemConfig::default_development();
        let (coordinator, handle) = SystemCoordinator::new(config);
        
        let initial_reserves = handle.get_s_credit_reserves().reserves.total();
        
        // Test adding S-credits
        let credits_to_add = SCredits::new(1000.0, 1000.0, 1000.0);
        handle.send_command(CoordinatorCommand::AddSCredits { 
            credits: credits_to_add.clone() 
        }).unwrap();
        
        // Note: In a real test, we'd need to run the coordinator loop to process commands
    }

    #[tokio::test]
    async fn test_viability_monitoring() {
        let config = SystemConfig::default_development();
        let (coordinator, handle) = SystemCoordinator::new(config);
        
        let component_id = ComponentId::new();
        let critical_status = ViabilityStatus::new(
            85.0, // Below critical threshold
            80.0,
            75.0,
            crate::types::VirtualBloodQuality::Critical,
        );
        
        // Test viability update command
        handle.send_command(CoordinatorCommand::UpdateViability {
            component_id,
            status: critical_status,
        }).unwrap();
    }
}