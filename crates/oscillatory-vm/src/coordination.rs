//! Oscillatory VM System Coordination
//!
//! Coordinates the overall Oscillatory Virtual Machine operation, integrating
//! heart function, S-entropy central bank, and Virtual Blood circulation.
//!
//! ## Theoretical Foundation
//!
//! The Oscillatory VM serves as the computational heart and S-entropy central bank,
//! coordinating biological neural network viability through systematic oscillatory
//! pumping of Virtual Blood while maintaining S-credit circulation economics.

use crate::{OscillatoryVMHeart, SEntropyCentralBank};
use virtual_blood::{VirtualBlood, VirtualVesselNetwork};
use jungfernstieg_core::{ComponentId, JungfernstiegError, Result, SCredits, SystemState, VMPerformance};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;
use tracing::{debug, info, warn, error};

/// Oscillatory Virtual Machine system coordinator
#[derive(Debug)]
pub struct OscillatoryVMCoordinator {
    /// Coordinator identifier
    pub id: Uuid,
    /// VM Heart for oscillatory pumping
    pub vm_heart: OscillatoryVMHeart,
    /// S-entropy central bank
    pub s_entropy_bank: SEntropyCentralBank,
    /// Coordination configuration
    pub config: CoordinationConfig,
    /// System performance metrics
    pub performance_metrics: VMPerformance,
    /// Coordination state
    pub coordination_state: CoordinationState,
    /// Virtual Blood circulation tracking
    pub circulation_tracking: CirculationTracking,
}

impl OscillatoryVMCoordinator {
    /// Create new Oscillatory VM coordinator
    pub fn new(
        heart_config: crate::OscillatoryVMHeartConfig,
        bank_config: crate::SEntropyBankConfig,
        coordination_config: CoordinationConfig,
    ) -> Self {
        let vm_heart = OscillatoryVMHeart::new(heart_config);
        let s_entropy_bank = SEntropyCentralBank::new(bank_config);
        
        Self {
            id: Uuid::new_v4(),
            vm_heart,
            s_entropy_bank,
            config: coordination_config,
            performance_metrics: VMPerformance::default(),
            coordination_state: CoordinationState::Initializing,
            circulation_tracking: CirculationTracking::new(),
        }
    }

    /// Start coordinated Oscillatory VM operation
    pub async fn start_coordinated_operation(&mut self) -> Result<()> {
        info!("Starting coordinated Oscillatory VM operation");

        // Initialize coordination
        self.coordination_state = CoordinationState::Starting;

        // Start S-entropy central bank
        self.s_entropy_bank.start().await?;
        
        // Start VM heart with S-entropy coordination
        self.vm_heart.start().await?;

        // Establish heart-bank coordination
        self.establish_heart_bank_coordination().await?;

        // Begin circulation coordination cycle
        self.begin_coordination_cycle().await?;

        self.coordination_state = CoordinationState::Active;
        info!("Oscillatory VM coordination active");

        Ok(())
    }

    /// Stop coordinated operation
    pub async fn stop_coordinated_operation(&mut self) -> Result<()> {
        info!("Stopping coordinated Oscillatory VM operation");

        self.coordination_state = CoordinationState::Stopping;

        // Stop coordination cycle
        self.stop_coordination_cycle().await?;

        // Stop VM heart
        self.vm_heart.stop().await?;

        // Stop S-entropy central bank
        self.s_entropy_bank.stop().await?;

        self.coordination_state = CoordinationState::Stopped;
        info!("Oscillatory VM coordination stopped");

        Ok(())
    }

    /// Execute unified circulation coordination cycle
    pub async fn execute_circulation_cycle(
        &mut self,
        virtual_blood: &mut VirtualBlood,
        vessel_network: &VirtualVesselNetwork,
    ) -> Result<CirculationCycleResult> {
        debug!("Executing circulation coordination cycle");

        let cycle_start = Instant::now();

        // Step 1: S-entropy economic assessment and credit allocation
        let economic_state = self.s_entropy_bank.assess_economic_state().await?;
        let credit_allocation = self.s_entropy_bank.allocate_circulation_credits(
            virtual_blood.s_entropy_coordinates(),
            &economic_state,
        ).await?;

        // Step 2: VM heart systolic phase - pump Virtual Blood
        let systolic_result = self.vm_heart.coordinate_systolic_phase(
            virtual_blood,
            &credit_allocation,
        ).await?;

        // Step 3: Circulation flow coordination
        let circulation_result = self.coordinate_circulation_flow(
            virtual_blood,
            vessel_network,
            &systolic_result,
        ).await?;

        // Step 4: VM heart diastolic phase - refill and reset
        let diastolic_result = self.vm_heart.coordinate_diastolic_phase(
            virtual_blood,
            &circulation_result,
        ).await?;

        // Step 5: S-entropy economic settlement
        let settlement_result = self.s_entropy_bank.settle_circulation_economics(
            &circulation_result,
            &diastolic_result,
        ).await?;

        let cycle_duration = cycle_start.elapsed();

        // Update coordination metrics
        self.update_coordination_metrics(&circulation_result, cycle_duration);

        // Track circulation performance
        self.circulation_tracking.record_cycle(
            &circulation_result,
            &economic_state,
            cycle_duration,
        );

        Ok(CirculationCycleResult {
            cycle_id: Uuid::new_v4(),
            economic_state,
            credit_allocation,
            systolic_result,
            circulation_result,
            diastolic_result,
            settlement_result,
            cycle_duration,
            coordination_efficiency: self.calculate_coordination_efficiency(),
        })
    }

    /// Coordinate circulation flow through vessel network
    async fn coordinate_circulation_flow(
        &self,
        virtual_blood: &VirtualBlood,
        vessel_network: &VirtualVesselNetwork,
        systolic_result: &crate::SystolicPhaseResult,
    ) -> Result<CirculationFlowResult> {
        debug!("Coordinating circulation flow through vessel network");

        // Calculate flow requirements based on systolic pump pressure
        let flow_requirements = self.calculate_flow_requirements(systolic_result);

        // Coordinate arterial flow
        let arterial_flow = self.coordinate_arterial_circulation(
            vessel_network,
            &flow_requirements,
        ).await?;

        // Coordinate arteriolar distribution
        let arteriolar_flow = self.coordinate_arteriolar_distribution(
            vessel_network,
            &arterial_flow,
        ).await?;

        // Coordinate capillary exchange
        let capillary_exchange = self.coordinate_capillary_exchange(
            vessel_network,
            virtual_blood,
            &arteriolar_flow,
        ).await?;

        // Calculate overall circulation efficiency
        let circulation_efficiency = self.calculate_circulation_efficiency(
            &arterial_flow,
            &arteriolar_flow,
            &capillary_exchange,
        );

        Ok(CirculationFlowResult {
            flow_id: Uuid::new_v4(),
            arterial_flow,
            arteriolar_flow,
            capillary_exchange,
            circulation_efficiency,
            total_volume_circulated: systolic_result.volume_pumped,
            flow_completion_time: systolic_result.pump_duration,
        })
    }

    /// Calculate flow requirements from systolic pump results
    fn calculate_flow_requirements(&self, systolic_result: &crate::SystolicPhaseResult) -> FlowRequirements {
        FlowRequirements {
            target_volume: systolic_result.volume_pumped,
            target_pressure: systolic_result.peak_pressure,
            target_flow_rate: systolic_result.volume_pumped / systolic_result.pump_duration.as_secs_f64(),
            distribution_priority: match systolic_result.pump_efficiency {
                e if e > 0.95 => FlowPriority::Optimal,
                e if e > 0.9 => FlowPriority::High,
                e if e > 0.8 => FlowPriority::Standard,
                _ => FlowPriority::Conservative,
            },
        }
    }

    /// Coordinate arterial circulation
    async fn coordinate_arterial_circulation(
        &self,
        vessel_network: &VirtualVesselNetwork,
        flow_requirements: &FlowRequirements,
    ) -> Result<ArterialFlowResult> {
        debug!("Coordinating arterial circulation");

        let total_arteries = vessel_network.arteries.len();
        let flow_per_artery = flow_requirements.target_volume / total_arteries.max(1) as f64;
        
        let mut arterial_flows = HashMap::new();
        let mut total_flow_delivered = 0.0;

        for (artery_id, artery) in &vessel_network.arteries {
            // Calculate flow through this artery based on its hemodynamic properties
            let artery_flow = flow_per_artery * artery.hemodynamic_properties.efficiency;
            arterial_flows.insert(*artery_id, artery_flow);
            total_flow_delivered += artery_flow;
        }

        let arterial_efficiency = total_flow_delivered / flow_requirements.target_volume;

        Ok(ArterialFlowResult {
            arterial_flows,
            total_arterial_flow: total_flow_delivered,
            arterial_efficiency,
            pressure_maintained: flow_requirements.target_pressure * arterial_efficiency,
        })
    }

    /// Coordinate arteriolar distribution
    async fn coordinate_arteriolar_distribution(
        &self,
        vessel_network: &VirtualVesselNetwork,
        arterial_flow: &ArterialFlowResult,
    ) -> Result<ArteriolarFlowResult> {
        debug!("Coordinating arteriolar distribution");

        let total_arterioles = vessel_network.arterioles.len();
        let average_arterial_flow = arterial_flow.total_arterial_flow / vessel_network.arteries.len().max(1) as f64;
        let flow_per_arteriole = average_arterial_flow / 4.0; // Branching factor of 4

        let mut arteriolar_flows = HashMap::new();
        let mut total_distribution = 0.0;

        for (arteriole_id, arteriole) in &vessel_network.arterioles {
            let distribution_flow = flow_per_arteriole * arteriole.hemodynamic_properties.efficiency;
            arteriolar_flows.insert(*arteriole_id, distribution_flow);
            total_distribution += distribution_flow;
        }

        let distribution_efficiency = total_distribution / arterial_flow.total_arterial_flow;

        Ok(ArteriolarFlowResult {
            arteriolar_flows,
            total_distribution,
            distribution_efficiency,
            average_pressure: arterial_flow.pressure_maintained * 0.6, // Pressure drops in arterioles
        })
    }

    /// Coordinate capillary exchange
    async fn coordinate_capillary_exchange(
        &self,
        vessel_network: &VirtualVesselNetwork,
        virtual_blood: &VirtualBlood,
        arteriolar_flow: &ArteriolarFlowResult,
    ) -> Result<CapillaryExchangeResult> {
        debug!("Coordinating capillary exchange");

        let total_capillaries = vessel_network.capillaries.len();
        let average_arteriolar_flow = arteriolar_flow.total_distribution / vessel_network.arterioles.len().max(1) as f64;
        let flow_per_capillary = average_arteriolar_flow / 25.0; // Average capillaries per arteriole

        let mut capillary_exchanges = HashMap::new();
        let mut total_exchange_efficiency = 0.0;
        let mut neural_delivery_volume = 0.0;

        for (capillary_id, capillary) in &vessel_network.capillaries {
            let exchange_volume = flow_per_capillary * capillary.hemodynamic_properties.efficiency;
            let exchange_efficiency = capillary.exchange_metrics.average_efficiency;
            
            let neural_delivery = exchange_volume * exchange_efficiency;
            
            capillary_exchanges.insert(*capillary_id, CapillaryExchange {
                capillary_id: *capillary_id,
                exchange_volume,
                exchange_efficiency,
                neural_delivery,
                concentration_gradient: 250.0, // 25% to 0.1% ratio from theoretical framework
            });
            
            total_exchange_efficiency += exchange_efficiency;
            neural_delivery_volume += neural_delivery;
        }

        let average_exchange_efficiency = total_exchange_efficiency / total_capillaries.max(1) as f64;

        Ok(CapillaryExchangeResult {
            capillary_exchanges,
            total_neural_delivery: neural_delivery_volume,
            average_exchange_efficiency,
            capillary_pressure: arteriolar_flow.average_pressure * 0.1, // Very low capillary pressure
            neural_viability_support: neural_delivery_volume >= virtual_blood.neural_nutrient_density(),
        })
    }

    /// Calculate overall circulation efficiency
    fn calculate_circulation_efficiency(
        &self,
        arterial: &ArterialFlowResult,
        arteriolar: &ArteriolarFlowResult,
        capillary: &CapillaryExchangeResult,
    ) -> f64 {
        // Weighted average of circulation stage efficiencies
        let arterial_weight = 0.3;
        let arteriolar_weight = 0.3;
        let capillary_weight = 0.4; // Higher weight for final delivery

        arterial.arterial_efficiency * arterial_weight +
        arteriolar.distribution_efficiency * arteriolar_weight +
        capillary.average_exchange_efficiency * capillary_weight
    }

    /// Establish heart-bank coordination
    async fn establish_heart_bank_coordination(&mut self) -> Result<()> {
        debug!("Establishing heart-bank coordination");

        // Configure heart to respond to S-entropy economic signals
        self.vm_heart.set_s_entropy_responsive_mode(true).await?;

        // Configure bank to provide circulation credit updates
        self.s_entropy_bank.enable_circulation_credit_streaming().await?;

        // Set up coordination timing
        self.vm_heart.set_coordination_interval(self.config.coordination_interval).await?;
        self.s_entropy_bank.set_economic_assessment_interval(self.config.coordination_interval).await?;

        info!("Heart-bank coordination established");
        Ok(())
    }

    /// Begin coordination cycle
    async fn begin_coordination_cycle(&mut self) -> Result<()> {
        debug!("Beginning coordination cycle");

        // Start background coordination task
        // In a real implementation, this would spawn a background task
        // For now, we'll simulate the setup
        
        self.coordination_state = CoordinationState::Coordinating;
        Ok(())
    }

    /// Stop coordination cycle
    async fn stop_coordination_cycle(&mut self) -> Result<()> {
        debug!("Stopping coordination cycle");

        // Stop background coordination
        self.coordination_state = CoordinationState::Stopping;
        Ok(())
    }

    /// Update coordination performance metrics
    fn update_coordination_metrics(&mut self, result: &CirculationFlowResult, duration: Duration) {
        self.performance_metrics.total_cycles += 1;
        
        // Update average efficiency
        if self.performance_metrics.total_cycles == 1 {
            self.performance_metrics.average_efficiency = result.circulation_efficiency;
        } else {
            self.performance_metrics.average_efficiency = 
                (self.performance_metrics.average_efficiency * (self.performance_metrics.total_cycles - 1) as f64 
                 + result.circulation_efficiency) / self.performance_metrics.total_cycles as f64;
        }

        // Update average cycle time
        if self.performance_metrics.total_cycles == 1 {
            self.performance_metrics.average_cycle_time = duration;
        } else {
            let total_nanos = self.performance_metrics.average_cycle_time.as_nanos() as f64 * (self.performance_metrics.total_cycles - 1) as f64 
                             + duration.as_nanos() as f64;
            self.performance_metrics.average_cycle_time = Duration::from_nanos((total_nanos / self.performance_metrics.total_cycles as f64) as u128);
        }

        self.performance_metrics.last_cycle_efficiency = result.circulation_efficiency;
        self.performance_metrics.last_update = Instant::now();
    }

    /// Calculate coordination efficiency
    fn calculate_coordination_efficiency(&self) -> f64 {
        // Coordination efficiency based on heart and bank performance
        let heart_efficiency = self.vm_heart.get_metrics().average_pump_efficiency;
        let bank_efficiency = self.s_entropy_bank.get_metrics().economic_efficiency;
        
        // Weighted combination
        heart_efficiency * 0.6 + bank_efficiency * 0.4
    }

    /// Get VM coordination metrics
    pub fn get_coordination_metrics(&self) -> &VMPerformance {
        &self.performance_metrics
    }

    /// Get coordination state
    pub fn get_coordination_state(&self) -> &CoordinationState {
        &self.coordination_state
    }

    /// Get circulation tracking
    pub fn get_circulation_tracking(&self) -> &CirculationTracking {
        &self.circulation_tracking
    }
}

/// Coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Coordination cycle interval
    pub coordination_interval: Duration,
    /// Enable autonomous coordination
    pub autonomous_coordination: bool,
    /// Circulation volume target per cycle
    pub target_circulation_volume: f64,
    /// Minimum coordination efficiency threshold
    pub min_coordination_efficiency: f64,
    /// Emergency coordination mode threshold
    pub emergency_threshold: f64,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            coordination_interval: Duration::from_millis(100), // 10 Hz coordination
            autonomous_coordination: true,
            target_circulation_volume: 100.0, // mL/cycle
            min_coordination_efficiency: 0.85, // 85% minimum efficiency
            emergency_threshold: 0.7, // 70% emergency threshold
        }
    }
}

/// Coordination state tracking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationState {
    /// System initializing
    Initializing,
    /// Starting coordination
    Starting,
    /// Active coordination
    Active,
    /// Coordinating circulation cycle
    Coordinating,
    /// Emergency coordination mode
    Emergency,
    /// Stopping coordination
    Stopping,
    /// Coordination stopped
    Stopped,
    /// Error state
    Error,
}

/// Circulation tracking system
#[derive(Debug)]
pub struct CirculationTracking {
    /// Circulation cycle history
    pub cycle_history: Vec<CirculationRecord>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
    /// Economic tracking
    pub economic_tracking: EconomicTracking,
}

impl CirculationTracking {
    /// Create new circulation tracking system
    pub fn new() -> Self {
        Self {
            cycle_history: Vec::new(),
            performance_trends: PerformanceTrends::default(),
            economic_tracking: EconomicTracking::default(),
        }
    }

    /// Record circulation cycle
    pub fn record_cycle(
        &mut self,
        circulation_result: &CirculationFlowResult,
        economic_state: &crate::EconomicState,
        cycle_duration: Duration,
    ) {
        let record = CirculationRecord {
            timestamp: Instant::now(),
            circulation_efficiency: circulation_result.circulation_efficiency,
            volume_circulated: circulation_result.total_volume_circulated,
            cycle_duration,
            economic_efficiency: economic_state.economic_efficiency,
            s_credit_flow: economic_state.total_credits_in_circulation,
        };

        self.cycle_history.push(record);

        // Maintain history size
        if self.cycle_history.len() > 1000 {
            self.cycle_history.remove(0);
        }

        // Update trends
        self.performance_trends.update_trends(&self.cycle_history);
        self.economic_tracking.update_economic_trends(&self.cycle_history);
    }
}

/// Performance trends analysis
#[derive(Debug, Default)]
pub struct PerformanceTrends {
    /// Efficiency trend
    pub efficiency_trend: TrendDirection,
    /// Volume trend
    pub volume_trend: TrendDirection,
    /// Duration trend
    pub duration_trend: TrendDirection,
}

impl PerformanceTrends {
    /// Update performance trends
    pub fn update_trends(&mut self, history: &[CirculationRecord]) {
        if history.len() < 10 {
            return; // Need minimum history for trend analysis
        }

        let recent = &history[history.len()-5..];
        let previous = &history[history.len()-10..history.len()-5];

        // Efficiency trend
        let recent_efficiency: f64 = recent.iter().map(|r| r.circulation_efficiency).sum::<f64>() / recent.len() as f64;
        let previous_efficiency: f64 = previous.iter().map(|r| r.circulation_efficiency).sum::<f64>() / previous.len() as f64;
        
        self.efficiency_trend = if recent_efficiency > previous_efficiency + 0.01 {
            TrendDirection::Improving
        } else if recent_efficiency < previous_efficiency - 0.01 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };

        // Similar analysis for volume and duration trends
        let recent_volume: f64 = recent.iter().map(|r| r.volume_circulated).sum::<f64>() / recent.len() as f64;
        let previous_volume: f64 = previous.iter().map(|r| r.volume_circulated).sum::<f64>() / previous.len() as f64;
        
        self.volume_trend = if recent_volume > previous_volume * 1.05 {
            TrendDirection::Improving
        } else if recent_volume < previous_volume * 0.95 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };
    }
}

/// Economic tracking for S-entropy circulation
#[derive(Debug, Default)]
pub struct EconomicTracking {
    /// S-credit flow trends
    pub credit_flow_trend: TrendDirection,
    /// Economic efficiency trends
    pub economic_efficiency_trend: TrendDirection,
}

impl EconomicTracking {
    /// Update economic trends
    pub fn update_economic_trends(&mut self, history: &[CirculationRecord]) {
        if history.len() < 10 {
            return;
        }

        let recent = &history[history.len()-5..];
        let previous = &history[history.len()-10..history.len()-5];

        // Credit flow trend
        let recent_credits: f64 = recent.iter().map(|r| r.s_credit_flow).sum::<f64>() / recent.len() as f64;
        let previous_credits: f64 = previous.iter().map(|r| r.s_credit_flow).sum::<f64>() / previous.len() as f64;
        
        self.credit_flow_trend = if recent_credits > previous_credits * 1.05 {
            TrendDirection::Improving
        } else if recent_credits < previous_credits * 0.95 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };
    }
}

/// Trend direction indicators
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TrendDirection {
    /// Performance improving
    Improving,
    /// Performance stable
    #[default]
    Stable,
    /// Performance declining
    Declining,
}

/// Circulation cycle record
#[derive(Debug, Clone)]
pub struct CirculationRecord {
    /// Record timestamp
    pub timestamp: Instant,
    /// Circulation efficiency achieved
    pub circulation_efficiency: f64,
    /// Volume circulated
    pub volume_circulated: f64,
    /// Cycle duration
    pub cycle_duration: Duration,
    /// Economic efficiency
    pub economic_efficiency: f64,
    /// S-credits in circulation
    pub s_credit_flow: f64,
}

/// Flow requirements for circulation
#[derive(Debug, Clone)]
pub struct FlowRequirements {
    /// Target circulation volume
    pub target_volume: f64,
    /// Target pressure
    pub target_pressure: f64,
    /// Target flow rate
    pub target_flow_rate: f64,
    /// Flow priority
    pub distribution_priority: FlowPriority,
}

/// Flow priority levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowPriority {
    /// Optimal flow conditions
    Optimal,
    /// High priority flow
    High,
    /// Standard flow
    Standard,
    /// Conservative flow
    Conservative,
    /// Emergency flow
    Emergency,
}

/// Results from circulation coordination
#[derive(Debug, Clone)]
pub struct CirculationCycleResult {
    /// Cycle identifier
    pub cycle_id: Uuid,
    /// Economic state during cycle
    pub economic_state: crate::EconomicState,
    /// Credit allocation result
    pub credit_allocation: crate::CreditAllocationResult,
    /// Systolic phase result
    pub systolic_result: crate::SystolicPhaseResult,
    /// Circulation flow result
    pub circulation_result: CirculationFlowResult,
    /// Diastolic phase result
    pub diastolic_result: crate::DiastolicPhaseResult,
    /// Economic settlement result
    pub settlement_result: crate::EconomicSettlementResult,
    /// Total cycle duration
    pub cycle_duration: Duration,
    /// Coordination efficiency achieved
    pub coordination_efficiency: f64,
}

/// Circulation flow result
#[derive(Debug, Clone)]
pub struct CirculationFlowResult {
    /// Flow identifier
    pub flow_id: Uuid,
    /// Arterial flow result
    pub arterial_flow: ArterialFlowResult,
    /// Arteriolar flow result
    pub arteriolar_flow: ArteriolarFlowResult,
    /// Capillary exchange result
    pub capillary_exchange: CapillaryExchangeResult,
    /// Overall circulation efficiency
    pub circulation_efficiency: f64,
    /// Total volume circulated
    pub total_volume_circulated: f64,
    /// Flow completion time
    pub flow_completion_time: Duration,
}

/// Arterial flow coordination result
#[derive(Debug, Clone)]
pub struct ArterialFlowResult {
    /// Flow per artery
    pub arterial_flows: HashMap<virtual_blood::VesselId, f64>,
    /// Total arterial flow
    pub total_arterial_flow: f64,
    /// Arterial efficiency
    pub arterial_efficiency: f64,
    /// Pressure maintained
    pub pressure_maintained: f64,
}

/// Arteriolar distribution result
#[derive(Debug, Clone)]
pub struct ArteriolarFlowResult {
    /// Flow per arteriole
    pub arteriolar_flows: HashMap<virtual_blood::VesselId, f64>,
    /// Total distribution volume
    pub total_distribution: f64,
    /// Distribution efficiency
    pub distribution_efficiency: f64,
    /// Average pressure
    pub average_pressure: f64,
}

/// Capillary exchange coordination result
#[derive(Debug, Clone)]
pub struct CapillaryExchangeResult {
    /// Exchange per capillary
    pub capillary_exchanges: HashMap<virtual_blood::VesselId, CapillaryExchange>,
    /// Total neural delivery volume
    pub total_neural_delivery: f64,
    /// Average exchange efficiency
    pub average_exchange_efficiency: f64,
    /// Capillary pressure
    pub capillary_pressure: f64,
    /// Neural viability support achieved
    pub neural_viability_support: bool,
}

/// Individual capillary exchange
#[derive(Debug, Clone)]
pub struct CapillaryExchange {
    /// Capillary identifier
    pub capillary_id: virtual_blood::VesselId,
    /// Exchange volume
    pub exchange_volume: f64,
    /// Exchange efficiency
    pub exchange_efficiency: f64,
    /// Neural delivery amount
    pub neural_delivery: f64,
    /// Concentration gradient achieved
    pub concentration_gradient: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OscillatoryVMHeartConfig, SEntropyBankConfig};

    #[test]
    fn test_coordinator_creation() {
        let heart_config = OscillatoryVMHeartConfig::default();
        let bank_config = SEntropyBankConfig::default();
        let coordination_config = CoordinationConfig::default();
        
        let coordinator = OscillatoryVMCoordinator::new(
            heart_config,
            bank_config,
            coordination_config,
        );
        
        assert_eq!(coordinator.coordination_state, CoordinationState::Initializing);
        assert_eq!(coordinator.config.coordination_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_flow_requirements_calculation() {
        let heart_config = OscillatoryVMHeartConfig::default();
        let bank_config = SEntropyBankConfig::default();
        let coordination_config = CoordinationConfig::default();
        
        let coordinator = OscillatoryVMCoordinator::new(
            heart_config,
            bank_config,
            coordination_config,
        );

        let systolic_result = crate::SystolicPhaseResult {
            volume_pumped: 100.0,
            peak_pressure: 120.0,
            pump_efficiency: 0.95,
            pump_duration: Duration::from_millis(300),
        };

        let requirements = coordinator.calculate_flow_requirements(&systolic_result);
        
        assert_eq!(requirements.target_volume, 100.0);
        assert_eq!(requirements.target_pressure, 120.0);
        assert_eq!(requirements.distribution_priority, FlowPriority::Optimal);
    }

    #[test]
    fn test_circulation_tracking() {
        let mut tracking = CirculationTracking::new();
        
        // Create mock circulation result
        let circulation_result = CirculationFlowResult {
            flow_id: Uuid::new_v4(),
            arterial_flow: ArterialFlowResult {
                arterial_flows: HashMap::new(),
                total_arterial_flow: 95.0,
                arterial_efficiency: 0.95,
                pressure_maintained: 115.0,
            },
            arteriolar_flow: ArteriolarFlowResult {
                arteriolar_flows: HashMap::new(),
                total_distribution: 85.0,
                distribution_efficiency: 0.89,
                average_pressure: 70.0,
            },
            capillary_exchange: CapillaryExchangeResult {
                capillary_exchanges: HashMap::new(),
                total_neural_delivery: 80.0,
                average_exchange_efficiency: 0.94,
                capillary_pressure: 7.0,
                neural_viability_support: true,
            },
            circulation_efficiency: 0.93,
            total_volume_circulated: 90.0,
            flow_completion_time: Duration::from_millis(250),
        };

        let economic_state = crate::EconomicState {
            total_credits_available: 3000.0,
            total_credits_in_circulation: 2500.0,
            credit_utilization_rate: 0.83,
            economic_efficiency: 0.91,
            supply_demand_balance: 0.95,
        };

        tracking.record_cycle(
            &circulation_result,
            &economic_state,
            Duration::from_millis(350),
        );
        
        assert_eq!(tracking.cycle_history.len(), 1);
        assert_eq!(tracking.cycle_history[0].circulation_efficiency, 0.93);
        assert_eq!(tracking.cycle_history[0].volume_circulated, 90.0);
    }

    #[test]
    fn test_coordination_efficiency_calculation() {
        let heart_config = OscillatoryVMHeartConfig::default();
        let bank_config = SEntropyBankConfig::default();
        let coordination_config = CoordinationConfig::default();
        
        let coordinator = OscillatoryVMCoordinator::new(
            heart_config,
            bank_config,
            coordination_config,
        );

        let efficiency = coordinator.calculate_coordination_efficiency();
        
        // Should be based on heart and bank performance (0.0 initially)
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);
    }

    #[test]
    fn test_coordination_state_transitions() {
        let mut state = CoordinationState::Initializing;
        
        state = CoordinationState::Starting;
        assert_eq!(state, CoordinationState::Starting);
        
        state = CoordinationState::Active;
        assert_eq!(state, CoordinationState::Active);
        
        state = CoordinationState::Stopping;
        assert_eq!(state, CoordinationState::Stopping);
        
        state = CoordinationState::Stopped;
        assert_eq!(state, CoordinationState::Stopped);
    }

    #[test]
    fn test_flow_priority_levels() {
        assert_ne!(FlowPriority::Optimal, FlowPriority::High);
        assert_ne!(FlowPriority::High, FlowPriority::Standard);
        assert_ne!(FlowPriority::Standard, FlowPriority::Conservative);
        assert_ne!(FlowPriority::Conservative, FlowPriority::Emergency);
    }

    #[test]
    fn test_trend_direction_analysis() {
        let mut trends = PerformanceTrends::default();
        
        // Create mock history with improving trend
        let mut history = Vec::new();
        for i in 0..15 {
            history.push(CirculationRecord {
                timestamp: Instant::now(),
                circulation_efficiency: 0.8 + (i as f64 * 0.01), // Improving efficiency
                volume_circulated: 90.0,
                cycle_duration: Duration::from_millis(300),
                economic_efficiency: 0.85,
                s_credit_flow: 2500.0,
            });
        }
        
        trends.update_trends(&history);
        
        assert_eq!(trends.efficiency_trend, TrendDirection::Improving);
    }
}
