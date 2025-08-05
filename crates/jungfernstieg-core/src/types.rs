//! Core types for Jungfernstieg biological-virtual neural symbiosis system
//!
//! This module defines the fundamental types used throughout the system,
//! including S-entropy credits, neural viability measures, and system states.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Unique identifier for Jungfernstieg system instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SystemId(pub Uuid);

impl SystemId {
    /// Generate a new random system ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SystemId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for system components (neural networks, VM, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(pub Uuid);

impl ComponentId {
    /// Generate a new random component ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ComponentId {
    fn default() -> Self {
        Self::new()
    }
}

/// S-entropy credits as universal currency for consciousness-computation operations
///
/// S_credits = {S_knowledge, S_time, S_entropy} ≡ Universal_Currency
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SCredits {
    /// Information processing currency
    pub s_knowledge: f64,
    /// Temporal coordination currency  
    pub s_time: f64,
    /// System optimization currency
    pub s_entropy: f64,
}

impl SCredits {
    /// Create new S-credits with specified amounts
    pub fn new(s_knowledge: f64, s_time: f64, s_entropy: f64) -> Self {
        Self {
            s_knowledge,
            s_time,
            s_entropy,
        }
    }

    /// Create zero S-credits (empty currency)
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Total S-credit value across all dimensions
    pub fn total(&self) -> f64 {
        self.s_knowledge + self.s_time + self.s_entropy
    }

    /// Check if S-credits are sufficient for operation
    pub fn is_sufficient(&self, required: &SCredits) -> bool {
        self.s_knowledge >= required.s_knowledge
            && self.s_time >= required.s_time
            && self.s_entropy >= required.s_entropy
    }

    /// Consume S-credits for operation
    pub fn consume(&mut self, required: &SCredits) -> Result<(), &'static str> {
        if !self.is_sufficient(required) {
            return Err("Insufficient S-credits");
        }
        
        self.s_knowledge -= required.s_knowledge;
        self.s_time -= required.s_time;
        self.s_entropy -= required.s_entropy;
        Ok(())
    }

    /// Add S-credits (for circulation income)
    pub fn add(&mut self, credits: &SCredits) {
        self.s_knowledge += credits.s_knowledge;
        self.s_time += credits.s_time;
        self.s_entropy += credits.s_entropy;
    }
}

/// S-credit reserves managed by the Oscillatory VM as S-Entropy Central Bank
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCreditReserves {
    /// Current reserves
    pub reserves: SCredits,
    /// Maximum capacity
    pub capacity: SCredits,
    /// Circulation rate (credits per second)
    pub circulation_rate: f64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl SCreditReserves {
    /// Create new S-credit reserves with specified capacity
    pub fn new(capacity: SCredits) -> Self {
        Self {
            reserves: capacity.clone(),
            capacity,
            circulation_rate: 1000.0, // Default 1000 credits/second
            last_update: Instant::now(),
        }
    }

    /// Get current reserves utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.capacity.total() == 0.0 {
            return 0.0;
        }
        self.reserves.total() / self.capacity.total()
    }

    /// Check if reserves can support operation
    pub fn can_support(&self, required: &SCredits) -> bool {
        self.reserves.is_sufficient(required)
    }

    /// Withdraw S-credits from reserves
    pub fn withdraw(&mut self, amount: &SCredits) -> Result<SCredits, &'static str> {
        if !self.can_support(amount) {
            return Err("Insufficient reserves");
        }
        
        self.reserves.consume(amount)?;
        self.last_update = Instant::now();
        Ok(amount.clone())
    }

    /// Deposit S-credits to reserves
    pub fn deposit(&mut self, amount: &SCredits) {
        self.reserves.add(amount);
        self.last_update = Instant::now();
    }
}

/// Neural viability status for biological neural networks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViabilityStatus {
    /// Overall viability percentage (0.0 to 100.0)
    pub viability_percent: f64,
    /// Metabolic activity level (0.0 to 100.0)
    pub metabolic_activity: f64,
    /// Synaptic function quality (0.0 to 100.0)
    pub synaptic_function: f64,
    /// Virtual Blood quality assessment
    pub vb_quality: VirtualBloodQuality,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl ViabilityStatus {
    /// Create new viability status
    pub fn new(
        viability_percent: f64,
        metabolic_activity: f64,
        synaptic_function: f64,
        vb_quality: VirtualBloodQuality,
    ) -> Self {
        Self {
            viability_percent,
            metabolic_activity,
            synaptic_function,
            vb_quality,
            timestamp: Instant::now(),
        }
    }

    /// Check if viability is above critical threshold (95%)
    pub fn is_viable(&self) -> bool {
        self.viability_percent >= 95.0
    }

    /// Check if viability is in warning range (90-95%)
    pub fn is_warning(&self) -> bool {
        self.viability_percent >= 90.0 && self.viability_percent < 95.0
    }

    /// Check if viability is critical (below 90%)
    pub fn is_critical(&self) -> bool {
        self.viability_percent < 90.0
    }
}

/// Virtual Blood quality assessment levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VirtualBloodQuality {
    /// Optimal Virtual Blood composition and circulation
    Optimal,
    /// Excellent quality with minor variations
    Excellent,
    /// Very good quality with acceptable parameters
    VeryGood,
    /// Good quality within operational range
    Good,
    /// Stable but suboptimal quality
    Stable,
    /// Warning level quality requiring attention
    Warning,
    /// Critical quality requiring immediate intervention
    Critical,
}

/// Overall system state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemState {
    /// System is initializing
    Initializing,
    /// System is operational and functioning normally
    Operational,
    /// System is in maintenance mode
    Maintenance,
    /// System is in warning state
    Warning,
    /// System is in critical state requiring intervention
    Critical,
    /// System is performing emergency shutdown
    EmergencyShutdown,
    /// System is stopped
    Stopped,
}

/// Comprehensive system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// System uptime
    pub uptime: Duration,
    /// Current S-credit reserves
    pub s_credit_reserves: SCreditReserves,
    /// Neural viability statistics
    pub neural_viability: HashMap<ComponentId, ViabilityStatus>,
    /// Virtual Blood circulation metrics
    pub circulation_metrics: CirculationMetrics,
    /// Oscillatory VM performance
    pub vm_performance: VMPerformance,
    /// Safety system status
    pub safety_status: SafetyStatus,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Virtual Blood circulation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CirculationMetrics {
    /// Circulation efficiency (0.0 to 1.0)
    pub efficiency: f64,
    /// Flow rate (units per second)
    pub flow_rate: f64,
    /// Oxygen transport efficiency (target: 98.7%)
    pub oxygen_efficiency: f64,
    /// Pressure stability
    pub pressure_stability: f64,
}

/// Oscillatory VM performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMPerformance {
    /// Heart function rhythm stability
    pub rhythm_stability: f64,
    /// S-credit circulation rate
    pub circulation_rate: f64,
    /// Economic coordination efficiency
    pub economic_efficiency: f64,
    /// Processing throughput
    pub throughput: f64,
}

/// Safety system status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyStatus {
    /// Overall safety level
    pub level: SafetyLevel,
    /// Active safety protocols
    pub active_protocols: Vec<String>,
    /// Last safety check timestamp
    pub last_check: Instant,
}

/// Safety alert levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyLevel {
    /// All systems operating safely
    Safe,
    /// Minor safety concerns
    Caution,
    /// Significant safety concerns requiring attention
    Warning,
    /// Critical safety situation requiring immediate action
    Critical,
    /// Emergency shutdown in progress
    Emergency,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_credits_operations() {
        let mut credits = SCredits::new(100.0, 50.0, 75.0);
        let required = SCredits::new(30.0, 20.0, 25.0);
        
        assert!(credits.is_sufficient(&required));
        assert!(credits.consume(&required).is_ok());
        assert_eq!(credits.s_knowledge, 70.0);
        assert_eq!(credits.s_time, 30.0);
        assert_eq!(credits.s_entropy, 50.0);
    }

    #[test]
    fn test_viability_status_thresholds() {
        let optimal = ViabilityStatus::new(99.0, 98.0, 97.0, VirtualBloodQuality::Optimal);
        let warning = ViabilityStatus::new(92.0, 90.0, 88.0, VirtualBloodQuality::Warning);
        let critical = ViabilityStatus::new(85.0, 80.0, 75.0, VirtualBloodQuality::Critical);

        assert!(optimal.is_viable());
        assert!(!optimal.is_warning());
        assert!(!optimal.is_critical());

        assert!(!warning.is_viable());
        assert!(warning.is_warning());
        assert!(!warning.is_critical());

        assert!(!critical.is_viable());
        assert!(!critical.is_warning());
        assert!(critical.is_critical());
    }

    #[test]
    fn test_s_credit_reserves() {
        let capacity = SCredits::new(1000.0, 1000.0, 1000.0);
        let mut reserves = SCreditReserves::new(capacity);
        
        let withdrawal = SCredits::new(100.0, 50.0, 75.0);
        assert!(reserves.can_support(&withdrawal));
        
        let result = reserves.withdraw(&withdrawal);
        assert!(result.is_ok());
        assert!(reserves.utilization() < 1.0);
    }
}