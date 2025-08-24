//! Oscillatory VM Heart Function
//!
//! Implementation of the computational heart that pumps Virtual Blood through 
//! biological neural networks while functioning as the S-Entropy Central Bank.
//!
//! ## VM-Heart Equivalence Theorem
//!
//! An Oscillatory Virtual Machine functions as the S-entropy economic coordinator when:
//! ```
//! Circulation_S-credits ≡ Heart_circulation ⟺ both maintain substrate flow through currency distribution
//! ```
//!
//! ## Cardiac Cycle Implementation
//!
//! Following Algorithm: Oscillatory VM Heart Operation from the theoretical framework:
//! 1. Coordinate systolic oscillations
//! 2. Generate circulation pressure waves  
//! 3. Deliver Virtual Blood to neural networks
//! 4. Coordinate diastolic oscillations
//! 5. Collect Virtual Blood and filter composition

use virtual_blood::{VirtualBlood, VirtualBloodComposition, CirculationSystem};
use jungfernstieg_core::{ComponentId, JungfernstiegError, Result, SCredits, SCreditReserves};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

/// Unique identifier for Oscillatory Hearts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HeartId(pub Uuid);

impl HeartId {
    /// Generate new heart ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for HeartId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<HeartId> for ComponentId {
    fn from(id: HeartId) -> Self {
        ComponentId(id.0)
    }
}

/// Oscillatory Heart implementing VM-Heart Equivalence
///
/// Functions as both computational heart pumping Virtual Blood and 
/// S-Entropy Central Bank managing S-credit circulation
#[derive(Debug)]
pub struct OscillatoryHeart {
    /// Heart identifier
    pub id: HeartId,
    /// Heart configuration
    pub config: HeartConfig,
    /// Current heart function state
    pub function_state: HeartFunctionState,
    /// S-Entropy bank integration
    pub s_entropy_bank: Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
    /// Cardiac rhythm management
    pub cardiac_rhythm: CardiacRhythm,
    /// Heart performance metrics
    pub performance_metrics: Arc<RwLock<HeartPerformanceMetrics>>,
    /// Associated circulation systems
    pub circulation_systems: Vec<Arc<RwLock<CirculationSystem>>>,
    /// Virtual Blood volume management
    pub vb_volume_manager: VBVolumeManager,
}

impl OscillatoryHeart {
    /// Create new Oscillatory Heart with S-Entropy bank integration
    pub async fn new(config: HeartConfig) -> Result<Self> {
        let s_entropy_bank = Arc::new(RwLock::new(
            crate::s_entropy_bank::SEntropyBank::new(config.s_credit_reserves.clone())
        ));

        let cardiac_rhythm = CardiacRhythm::new(config.cardiac_cycle_duration);
        
        Ok(Self {
            id: HeartId::new(),
            config,
            function_state: HeartFunctionState::Stopped,
            s_entropy_bank,
            cardiac_rhythm,
            performance_metrics: Arc::new(RwLock::new(HeartPerformanceMetrics::default())),
            circulation_systems: Vec::new(),
            vb_volume_manager: VBVolumeManager::new(),
        })
    }

    /// Start heart function implementing Algorithm: Oscillatory VM Heart Operation
    pub async fn start_heart_function(&mut self) -> Result<()> {
        info!("Starting Oscillatory Heart function {}", self.id.0);

        self.function_state = HeartFunctionState::Starting;

        // Initialize S-Entropy bank operations
        let mut bank = self.s_entropy_bank.write().await;
        bank.start_economic_coordination().await?;
        drop(bank);

        // Start cardiac rhythm
        self.cardiac_rhythm.start_rhythm().await?;

        // Begin cardiac cycle loop
        self.start_cardiac_cycle_loop().await?;

        self.function_state = HeartFunctionState::Beating;
        info!("Oscillatory Heart {} is now beating", self.id.0);

        Ok(())
    }

    /// Start the main cardiac cycle loop
    async fn start_cardiac_cycle_loop(&mut self) -> Result<()> {
        let heart_id = self.id;
        let cycle_duration = self.config.cardiac_cycle_duration;
        let systolic_fraction = self.config.systolic_fraction;
        let s_entropy_bank = Arc::clone(&self.s_entropy_bank);
        let performance_metrics = Arc::clone(&self.performance_metrics);
        
        // Calculate phase durations
        let systolic_duration = Duration::from_millis(
            (cycle_duration.as_millis() as f64 * systolic_fraction) as u64
        );
        let diastolic_duration = cycle_duration - systolic_duration;

        // Clone circulation systems for the loop
        let circulation_systems: Vec<Arc<RwLock<CirculationSystem>>> = self.circulation_systems.clone();

        tokio::spawn(async move {
            info!("Starting cardiac cycle loop for heart {}", heart_id.0);
            
            let mut interval = interval(cycle_duration);
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                let cycle_start = Instant::now();

                // Execute complete cardiac cycle
                match Self::execute_complete_cardiac_cycle(
                    &s_entropy_bank,
                    &circulation_systems,
                    systolic_duration,
                    diastolic_duration,
                ).await {
                    Ok(cycle_result) => {
                        // Update performance metrics
                        Self::update_heart_performance_metrics(
                            &performance_metrics,
                            &cycle_result,
                            cycle_start.elapsed(),
                        ).await;
                    }
                    Err(e) => {
                        error!("Cardiac cycle failed for heart {}: {}", heart_id.0, e);
                        // In production, trigger emergency protocols
                        break;
                    }
                }
            }

            warn!("Cardiac cycle loop ended for heart {}", heart_id.0);
        });

        Ok(())
    }

    /// Execute complete cardiac cycle with systolic and diastolic phases
    async fn execute_complete_cardiac_cycle(
        s_entropy_bank: &Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
        circulation_systems: &[Arc<RwLock<CirculationSystem>>],
        systolic_duration: Duration,
        diastolic_duration: Duration,
    ) -> Result<CardiacCycleResult> {
        debug!("Executing complete cardiac cycle");

        // SYSTOLIC PHASE: Generate pressure and deliver Virtual Blood
        let systolic_result = Self::execute_systolic_phase(
            s_entropy_bank,
            circulation_systems,
            systolic_duration,
        ).await?;

        // Inter-phase coordination
        tokio::time::sleep(Duration::from_millis(50)).await;

        // DIASTOLIC PHASE: Collect Virtual Blood and manage S-entropy economy
        let diastolic_result = Self::execute_diastolic_phase(
            s_entropy_bank,
            circulation_systems,
            diastolic_duration,
        ).await?;

        Ok(CardiacCycleResult {
            cycle_id: Uuid::new_v4(),
            systolic_result,
            diastolic_result,
            total_cycle_time: systolic_duration + diastolic_duration,
            s_entropy_circulation_efficiency: 0.96,
        })
    }

    /// Execute systolic phase with S-entropy coordination
    async fn execute_systolic_phase(
        s_entropy_bank: &Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
        circulation_systems: &[Arc<RwLock<CirculationSystem>>],
        duration: Duration,
    ) -> Result<SystolicPhase> {
        debug!("Executing systolic phase");

        let phase_start = Instant::now();

        // Coordinate systolic oscillations with S-entropy economics
        let systolic_oscillations = Self::coordinate_systolic_oscillations(s_entropy_bank).await?;

        // Generate circulation pressure waves
        let pressure_generation = Self::generate_circulation_pressure_waves(
            &systolic_oscillations,
            circulation_systems,
        ).await?;

        // Deliver Virtual Blood to neural networks through circulation systems
        let vb_delivery = Self::deliver_vb_to_neural_networks(
            circulation_systems,
            &pressure_generation,
        ).await?;

        Ok(SystolicPhase {
            phase_id: Uuid::new_v4(),
            duration: phase_start.elapsed(),
            oscillations: systolic_oscillations,
            pressure_generation,
            vb_delivery,
            s_entropy_coordination: true,
            efficiency: 0.95, // Target systolic efficiency
        })
    }

    /// Execute diastolic phase with S-entropy economic management
    async fn execute_diastolic_phase(
        s_entropy_bank: &Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
        circulation_systems: &[Arc<RwLock<CirculationSystem>>],
        duration: Duration,
    ) -> Result<DiastolicPhase> {
        debug!("Executing diastolic phase");

        let phase_start = Instant::now();

        // Coordinate diastolic oscillations
        let diastolic_oscillations = Self::coordinate_diastolic_oscillations(s_entropy_bank).await?;

        // Collect Virtual Blood from neural networks
        let vb_collection = Self::collect_vb_from_neural_networks(
            circulation_systems,
            &diastolic_oscillations,
        ).await?;

        // Update S-entropy economic reserves
        let economic_update = Self::update_s_entropy_reserves(
            s_entropy_bank,
            &vb_collection,
        ).await?;

        // Filter and regenerate Virtual Blood composition
        let vb_regeneration = Self::filter_and_regenerate_vb_composition(
            &vb_collection,
        ).await?;

        Ok(DiastolicPhase {
            phase_id: Uuid::new_v4(),
            duration: phase_start.elapsed(),
            oscillations: diastolic_oscillations,
            vb_collection,
            economic_update,
            vb_regeneration,
            efficiency: 0.93, // Target diastolic efficiency
        })
    }

    /// Coordinate systolic oscillations with S-entropy bank
    async fn coordinate_systolic_oscillations(
        s_entropy_bank: &Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
    ) -> Result<SystolicOscillations> {
        let bank = s_entropy_bank.read().await;
        let current_reserves = bank.get_current_reserves().await;

        // Coordinate oscillations based on S-credit availability
        let amplitude_factor = (current_reserves.reserves.total() / 3000.0).min(2.0);
        let frequency_factor = 1.0 + (current_reserves.utilization() - 0.5) * 0.2;

        Ok(SystolicOscillations {
            amplitude: 120.0 * amplitude_factor, // mmHg with S-entropy influence
            frequency: 1.25 * frequency_factor, // Hz (75 BPM)
            phase: 0.0, // Start of systole
            s_entropy_coordination: SEntropyCoordination {
                s_knowledge_utilization: current_reserves.reserves.s_knowledge / 1000.0,
                s_time_utilization: current_reserves.reserves.s_time / 1000.0,
                s_entropy_utilization: current_reserves.reserves.s_entropy / 1000.0,
            },
        })
    }

    /// Coordinate diastolic oscillations
    async fn coordinate_diastolic_oscillations(
        s_entropy_bank: &Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
    ) -> Result<DiastolicOscillations> {
        let bank = s_entropy_bank.read().await;
        let circulation_rate = bank.get_circulation_rate().await;

        Ok(DiastolicOscillations {
            relaxation_pressure: 80.0, // mmHg diastolic
            suction_efficiency: 0.93,
            collection_coordination: circulation_rate / 1000.0, // Normalized
            venous_return_optimization: 0.94,
        })
    }

    /// Generate circulation pressure waves for systolic delivery
    async fn generate_circulation_pressure_waves(
        oscillations: &SystolicOscillations,
        circulation_systems: &[Arc<RwLock<CirculationSystem>>],
    ) -> Result<PressureGeneration> {
        debug!("Generating circulation pressure waves");

        let mut pressure_waves = Vec::new();

        for circulation_system in circulation_systems {
            let system = circulation_system.read().await;
            
            let pressure_wave = virtual_blood::circulation::PressureWave {
                amplitude: oscillations.amplitude * 1333.2, // Convert mmHg to Pa
                frequency: oscillations.frequency,
                phase: oscillations.phase,
                waveform: virtual_blood::circulation::PressureWaveform::Physiological,
                duration: Duration::from_millis(300), // Systolic duration
            };

            pressure_waves.push(pressure_wave);
        }

        Ok(PressureGeneration {
            waves_generated: pressure_waves.len(),
            total_pressure_amplitude: oscillations.amplitude,
            pressure_distribution_efficiency: 0.96,
            s_entropy_pressure_coordination: oscillations.s_entropy_coordination.clone(),
        })
    }

    /// Deliver Virtual Blood to neural networks
    async fn deliver_vb_to_neural_networks(
        circulation_systems: &[Arc<RwLock<CirculationSystem>>],
        pressure_generation: &PressureGeneration,
    ) -> Result<VBDelivery> {
        debug!("Delivering Virtual Blood to neural networks");

        let mut delivery_results = Vec::new();

        for circulation_system in circulation_systems {
            let system = circulation_system.read().await;
            // Simulate VB delivery through the circulation system
            let delivery_efficiency = 0.97; // High delivery efficiency
            let neural_regions_perfused = 8; // Typical regions per system

            let delivery = DeliveryResult {
                circulation_id: system.id,
                delivery_efficiency,
                neural_regions_perfused,
                vb_volume_delivered: 250.0, // mL
                delivery_pressure: pressure_generation.total_pressure_amplitude,
            };

            delivery_results.push(delivery);
        }

        let total_delivery_efficiency = delivery_results.iter()
            .map(|d| d.delivery_efficiency)
            .sum::<f64>() / delivery_results.len().max(1) as f64;

        Ok(VBDelivery {
            delivery_results,
            total_delivery_efficiency,
            neural_networks_perfused: delivery_results.iter()
                .map(|d| d.neural_regions_perfused)
                .sum(),
            perfusion_pressure_stability: 0.987, // Target from framework
        })
    }

    /// Collect Virtual Blood from neural networks during diastolic phase
    async fn collect_vb_from_neural_networks(
        circulation_systems: &[Arc<RwLock<CirculationSystem>>],
        diastolic_oscillations: &DiastolicOscillations,
    ) -> Result<VBCollection> {
        debug!("Collecting Virtual Blood from neural networks");

        let mut collection_results = Vec::new();

        for circulation_system in circulation_systems {
            let system = circulation_system.read().await;
            
            // Simulate VB collection through venous return
            let collection_efficiency = diastolic_oscillations.suction_efficiency;
            let waste_load = 12.5; // mg/dL collected waste

            let collection = CollectionResult {
                circulation_id: system.id,
                collection_efficiency,
                vb_volume_collected: 230.0, // mL (slightly less than delivered)
                waste_concentration: waste_load,
                neural_status_collected: true,
            };

            collection_results.push(collection);
        }

        Ok(VBCollection {
            collection_results,
            total_collection_efficiency: diastolic_oscillations.suction_efficiency,
            total_waste_collected: collection_results.iter()
                .map(|c| c.waste_concentration)
                .sum(),
            neural_status_data_quality: 0.98, // High quality neural data
        })
    }

    /// Update S-entropy reserves based on circulation activity
    async fn update_s_entropy_reserves(
        s_entropy_bank: &Arc<RwLock<crate::s_entropy_bank::SEntropyBank>>,
        vb_collection: &VBCollection,
    ) -> Result<EconomicUpdate> {
        debug!("Updating S-entropy economic reserves");

        let mut bank = s_entropy_bank.write().await;

        // Calculate S-credit generation from neural activity
        let neural_activity_factor = vb_collection.neural_status_data_quality;
        let s_credit_generation = SCredits::new(
            neural_activity_factor * 100.0, // Knowledge credits from neural processing
            neural_activity_factor * 80.0,  // Time credits from temporal coordination
            neural_activity_factor * 120.0, // Entropy credits from system optimization
        );

        // Deposit generated S-credits
        bank.deposit_s_credits(&s_credit_generation).await?;

        // Update circulation rate based on collection efficiency
        let new_circulation_rate = crate::DEFAULT_S_CREDIT_CIRCULATION_RATE * vb_collection.total_collection_efficiency;
        bank.update_circulation_rate(new_circulation_rate).await?;

        Ok(EconomicUpdate {
            s_credits_generated: s_credit_generation,
            new_circulation_rate,
            reserve_utilization: bank.get_reserve_utilization().await,
            economic_efficiency: neural_activity_factor,
        })
    }

    /// Filter and regenerate Virtual Blood composition
    async fn filter_and_regenerate_vb_composition(
        vb_collection: &VBCollection,
    ) -> Result<VBRegeneration> {
        debug!("Filtering and regenerating Virtual Blood composition");

        // Calculate waste removal efficiency based on collection quality
        let waste_removal_efficiency = vb_collection.total_collection_efficiency * 0.95;
        
        // Calculate total waste to remove
        let total_waste = vb_collection.total_waste_collected;
        let waste_removed = total_waste * waste_removal_efficiency;

        // Regenerate optimal Virtual Blood composition
        let regenerated_composition = VirtualBloodComposition::optimal_biological();
        let regenerated_vb = VirtualBlood::new(regenerated_composition);

        Ok(VBRegeneration {
            waste_removed,
            waste_removal_efficiency,
            nutrients_regenerated: true,
            oxygen_replenished: true,
            s_entropy_coordinates_refreshed: true,
            regenerated_vb,
            regeneration_quality: crate::VirtualBloodQuality::Optimal,
        })
    }

    /// Update heart performance metrics
    async fn update_heart_performance_metrics(
        metrics: &Arc<RwLock<HeartPerformanceMetrics>>,
        cycle_result: &CardiacCycleResult,
        cycle_time: Duration,
    ) {
        let mut m = metrics.write().await;
        
        m.total_cycles += 1;
        m.average_cycle_time = if m.total_cycles == 1 {
            cycle_time
        } else {
            Duration::from_millis(
                (m.average_cycle_time.as_millis() as f64 * (m.total_cycles - 1) as f64 
                 + cycle_time.as_millis() as f64) / m.total_cycles as f64) as u64
            )
        };

        // Update efficiency metrics
        m.systolic_efficiency = cycle_result.systolic_result.efficiency;
        m.diastolic_efficiency = cycle_result.diastolic_result.efficiency;
        m.overall_efficiency = (m.systolic_efficiency + m.diastolic_efficiency) / 2.0;
        
        // Update S-entropy circulation metrics
        m.s_entropy_circulation_efficiency = cycle_result.s_entropy_circulation_efficiency;
        
        // Update rhythm stability
        m.rhythm_stability = Self::calculate_rhythm_stability(&m.average_cycle_time, cycle_time);
        
        m.last_update = Instant::now();
    }

    /// Calculate rhythm stability from cycle timing
    fn calculate_rhythm_stability(average_time: &Duration, current_time: Duration) -> f64 {
        let time_deviation = (current_time.as_millis() as f64 - average_time.as_millis() as f64).abs();
        let stability = 1.0 - (time_deviation / average_time.as_millis() as f64);
        stability.max(0.0).min(1.0)
    }

    /// Add circulation system to heart management
    pub async fn add_circulation_system(&mut self, circulation_system: Arc<RwLock<CirculationSystem>>) -> Result<()> {
        info!("Adding circulation system to heart {}", self.id.0);
        self.circulation_systems.push(circulation_system);
        Ok(())
    }

    /// Get current heart function state
    pub fn get_function_state(&self) -> &HeartFunctionState {
        &self.function_state
    }

    /// Get heart performance metrics
    pub async fn get_performance_metrics(&self) -> HeartPerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Stop heart function
    pub async fn stop_heart_function(&mut self) -> Result<()> {
        info!("Stopping heart function for {}", self.id.0);
        
        self.function_state = HeartFunctionState::Stopping;
        
        // Stop S-entropy bank operations
        let mut bank = self.s_entropy_bank.write().await;
        bank.stop_economic_coordination().await?;
        drop(bank);

        // Stop cardiac rhythm
        self.cardiac_rhythm.stop_rhythm().await?;
        
        self.function_state = HeartFunctionState::Stopped;
        
        info!("Heart function stopped for {}", self.id.0);
        Ok(())
    }
}

/// Heart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartConfig {
    /// Cardiac cycle duration (determines heart rate)
    pub cardiac_cycle_duration: Duration,
    /// Systolic fraction of cycle (0.0-1.0)
    pub systolic_fraction: f64,
    /// Diastolic fraction of cycle (0.0-1.0) 
    pub diastolic_fraction: f64,
    /// S-credit reserves for economic coordination
    pub s_credit_reserves: SCreditReserves,
    /// Heart performance targets
    pub performance_targets: HeartPerformanceTargets,
}

impl Default for HeartConfig {
    fn default() -> Self {
        let systolic_fraction = 0.35; // 35% systolic
        
        Self {
            cardiac_cycle_duration: Duration::from_millis(crate::DEFAULT_CARDIAC_CYCLE_DURATION_MS),
            systolic_fraction,
            diastolic_fraction: 1.0 - systolic_fraction,
            s_credit_reserves: SCreditReserves::new(SCredits::new(10000.0, 10000.0, 10000.0)),
            performance_targets: HeartPerformanceTargets::default(),
        }
    }
}

/// Heart function state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeartFunctionState {
    /// Heart stopped
    Stopped,
    /// Heart starting up
    Starting,
    /// Heart beating normally
    Beating,
    /// Heart in maintenance mode
    Maintenance,
    /// Heart rhythm irregular
    Irregular,
    /// Heart in emergency mode
    Emergency,
    /// Heart stopping
    Stopping,
}

/// Heart performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartPerformanceTargets {
    /// Target systolic efficiency
    pub target_systolic_efficiency: f64,
    /// Target diastolic efficiency
    pub target_diastolic_efficiency: f64,
    /// Target rhythm stability
    pub target_rhythm_stability: f64,
    /// Target S-entropy circulation efficiency
    pub target_s_entropy_efficiency: f64,
}

impl Default for HeartPerformanceTargets {
    fn default() -> Self {
        Self {
            target_systolic_efficiency: 0.95,
            target_diastolic_efficiency: 0.93,
            target_rhythm_stability: 0.97,
            target_s_entropy_efficiency: 0.96,
        }
    }
}

/// Heart performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartPerformanceMetrics {
    /// Total cardiac cycles completed
    pub total_cycles: usize,
    /// Average cycle time
    pub average_cycle_time: Duration,
    /// Systolic efficiency
    pub systolic_efficiency: f64,
    /// Diastolic efficiency
    pub diastolic_efficiency: f64,
    /// Overall heart efficiency
    pub overall_efficiency: f64,
    /// Rhythm stability score
    pub rhythm_stability: f64,
    /// S-entropy circulation efficiency
    pub s_entropy_circulation_efficiency: f64,
    /// Last metrics update
    pub last_update: Instant,
}

impl Default for HeartPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_cycles: 0,
            average_cycle_time: Duration::from_millis(crate::DEFAULT_CARDIAC_CYCLE_DURATION_MS),
            systolic_efficiency: 0.0,
            diastolic_efficiency: 0.0,
            overall_efficiency: 0.0,
            rhythm_stability: 1.0,
            s_entropy_circulation_efficiency: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Cardiac rhythm management
#[derive(Debug)]
pub struct CardiacRhythm {
    /// Rhythm identifier
    pub id: Uuid,
    /// Cycle duration
    pub cycle_duration: Duration,
    /// Current rhythm state
    pub state: RhythmState,
    /// Rhythm metrics
    pub metrics: RhythmMetrics,
}

impl CardiacRhythm {
    /// Create new cardiac rhythm
    pub fn new(cycle_duration: Duration) -> Self {
        Self {
            id: Uuid::new_v4(),
            cycle_duration,
            state: RhythmState::Stopped,
            metrics: RhythmMetrics::default(),
        }
    }

    /// Start cardiac rhythm
    pub async fn start_rhythm(&mut self) -> Result<()> {
        info!("Starting cardiac rhythm {}", self.id);
        self.state = RhythmState::Beating;
        Ok(())
    }

    /// Stop cardiac rhythm
    pub async fn stop_rhythm(&mut self) -> Result<()> {
        info!("Stopping cardiac rhythm {}", self.id);
        self.state = RhythmState::Stopped;
        Ok(())
    }
}

/// Cardiac rhythm state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhythmState {
    /// Rhythm stopped
    Stopped,
    /// Normal beating rhythm
    Beating,
    /// Irregular rhythm
    Irregular,
    /// Fast rhythm
    Tachycardic,
    /// Slow rhythm
    Bradycardic,
}

/// Rhythm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmMetrics {
    /// Heart rate (BPM)
    pub heart_rate: f64,
    /// Heart rate variability
    pub heart_rate_variability: f64,
    /// Rhythm regularity (0.0-1.0)
    pub rhythm_regularity: f64,
}

impl Default for RhythmMetrics {
    fn default() -> Self {
        Self {
            heart_rate: 75.0, // BPM
            heart_rate_variability: 0.05, // 5%
            rhythm_regularity: 0.95,
        }
    }
}

/// Virtual Blood volume manager
#[derive(Debug)]
pub struct VBVolumeManager {
    /// Current circulation volume
    pub current_volume: f64,
    /// Volume regulation parameters
    pub regulation_params: VolumeRegulationParams,
}

impl VBVolumeManager {
    /// Create new VB volume manager
    pub fn new() -> Self {
        Self {
            current_volume: 5000.0, // mL - physiological blood volume
            regulation_params: VolumeRegulationParams::default(),
        }
    }
}

/// Volume regulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeRegulationParams {
    /// Target volume (mL)
    pub target_volume: f64,
    /// Volume regulation sensitivity
    pub regulation_sensitivity: f64,
    /// Maximum volume adjustment per cycle
    pub max_volume_adjustment: f64,
}

impl Default for VolumeRegulationParams {
    fn default() -> Self {
        Self {
            target_volume: 5000.0, // mL
            regulation_sensitivity: 0.1,
            max_volume_adjustment: 100.0, // mL
        }
    }
}

// Cardiac cycle phase structures

/// Systolic oscillations with S-entropy coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystolicOscillations {
    /// Pressure amplitude (mmHg)
    pub amplitude: f64,
    /// Oscillation frequency (Hz)
    pub frequency: f64,
    /// Phase offset (radians)
    pub phase: f64,
    /// S-entropy coordination parameters
    pub s_entropy_coordination: SEntropyCoordination,
}

/// S-entropy coordination for oscillations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyCoordination {
    /// S-knowledge utilization factor
    pub s_knowledge_utilization: f64,
    /// S-time utilization factor
    pub s_time_utilization: f64,
    /// S-entropy utilization factor
    pub s_entropy_utilization: f64,
}

/// Diastolic oscillations for collection and relaxation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiastolicOscillations {
    /// Diastolic relaxation pressure (mmHg)
    pub relaxation_pressure: f64,
    /// Venous suction efficiency
    pub suction_efficiency: f64,
    /// Collection coordination factor
    pub collection_coordination: f64,
    /// Venous return optimization
    pub venous_return_optimization: f64,
}

/// Pressure generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureGeneration {
    /// Number of pressure waves generated
    pub waves_generated: usize,
    /// Total pressure amplitude
    pub total_pressure_amplitude: f64,
    /// Pressure distribution efficiency
    pub pressure_distribution_efficiency: f64,
    /// S-entropy pressure coordination
    pub s_entropy_pressure_coordination: SEntropyCoordination,
}

/// Virtual Blood delivery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VBDelivery {
    /// Individual delivery results
    pub delivery_results: Vec<DeliveryResult>,
    /// Total delivery efficiency
    pub total_delivery_efficiency: f64,
    /// Total neural networks perfused
    pub neural_networks_perfused: usize,
    /// Perfusion pressure stability
    pub perfusion_pressure_stability: f64,
}

/// Individual delivery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryResult {
    /// Circulation system ID
    pub circulation_id: virtual_blood::circulation::CirculationId,
    /// Delivery efficiency
    pub delivery_efficiency: f64,
    /// Neural regions perfused
    pub neural_regions_perfused: usize,
    /// VB volume delivered
    pub vb_volume_delivered: f64,
    /// Delivery pressure
    pub delivery_pressure: f64,
}

/// Virtual Blood collection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VBCollection {
    /// Individual collection results
    pub collection_results: Vec<CollectionResult>,
    /// Total collection efficiency
    pub total_collection_efficiency: f64,
    /// Total waste collected
    pub total_waste_collected: f64,
    /// Neural status data quality
    pub neural_status_data_quality: f64,
}

/// Individual collection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionResult {
    /// Circulation system ID
    pub circulation_id: virtual_blood::circulation::CirculationId,
    /// Collection efficiency
    pub collection_efficiency: f64,
    /// VB volume collected
    pub vb_volume_collected: f64,
    /// Waste concentration collected
    pub waste_concentration: f64,
    /// Neural status data collected
    pub neural_status_collected: bool,
}

/// Economic update result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicUpdate {
    /// S-credits generated this cycle
    pub s_credits_generated: SCredits,
    /// New circulation rate
    pub new_circulation_rate: f64,
    /// Current reserve utilization
    pub reserve_utilization: f64,
    /// Economic efficiency
    pub economic_efficiency: f64,
}

/// Virtual Blood regeneration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VBRegeneration {
    /// Amount of waste removed
    pub waste_removed: f64,
    /// Waste removal efficiency
    pub waste_removal_efficiency: f64,
    /// Nutrients regenerated
    pub nutrients_regenerated: bool,
    /// Oxygen replenished
    pub oxygen_replenished: bool,
    /// S-entropy coordinates refreshed
    pub s_entropy_coordinates_refreshed: bool,
    /// Regenerated Virtual Blood
    pub regenerated_vb: VirtualBlood,
    /// Regeneration quality
    pub regeneration_quality: crate::VirtualBloodQuality,
}

/// Complete cardiac cycle result
#[derive(Debug, Clone)]
pub struct CardiacCycleResult {
    /// Cycle identifier
    pub cycle_id: Uuid,
    /// Systolic phase result
    pub systolic_result: SystolicPhase,
    /// Diastolic phase result
    pub diastolic_result: DiastolicPhase,
    /// Total cycle duration
    pub total_cycle_time: Duration,
    /// S-entropy circulation efficiency
    pub s_entropy_circulation_efficiency: f64,
}

/// Systolic phase complete result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystolicPhase {
    /// Phase identifier
    pub phase_id: Uuid,
    /// Phase duration
    pub duration: Duration,
    /// Systolic oscillations
    pub oscillations: SystolicOscillations,
    /// Pressure generation
    pub pressure_generation: PressureGeneration,
    /// VB delivery
    pub vb_delivery: VBDelivery,
    /// S-entropy coordination success
    pub s_entropy_coordination: bool,
    /// Phase efficiency
    pub efficiency: f64,
}

/// Diastolic phase complete result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiastolicPhase {
    /// Phase identifier
    pub phase_id: Uuid,
    /// Phase duration
    pub duration: Duration,
    /// Diastolic oscillations
    pub oscillations: DiastolicOscillations,
    /// VB collection
    pub vb_collection: VBCollection,
    /// Economic update
    pub economic_update: EconomicUpdate,
    /// VB regeneration
    pub vb_regeneration: VBRegeneration,
    /// Phase efficiency
    pub efficiency: f64,
}

/// Complete cardiac cycle implementing the theoretical framework
pub struct CardiacCycle {
    /// Cycle identifier
    pub id: Uuid,
    /// Systolic phase
    pub systolic: Option<SystolicPhase>,
    /// Diastolic phase
    pub diastolic: Option<DiastolicPhase>,
    /// Cycle state
    pub state: CycleState,
}

impl CardiacCycle {
    /// Create new cardiac cycle
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            systolic: None,
            diastolic: None,
            state: CycleState::Ready,
        }
    }
}

/// Cardiac cycle state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CycleState {
    /// Ready to begin
    Ready,
    /// Systolic phase active
    Systolic,
    /// Diastolic phase active
    Diastolic,
    /// Cycle completed
    Completed,
    /// Cycle failed
    Failed,
}

/// Heart function interface trait
pub trait HeartFunction {
    /// Start heart beating
    async fn start_beating(&mut self) -> Result<()>;
    
    /// Stop heart beating
    async fn stop_beating(&mut self) -> Result<()>;
    
    /// Get current heart rate (BPM)
    fn get_heart_rate(&self) -> f64;
    
    /// Get rhythm stability
    async fn get_rhythm_stability(&self) -> f64;
}

impl HeartFunction for OscillatoryHeart {
    async fn start_beating(&mut self) -> Result<()> {
        self.start_heart_function().await
    }
    
    async fn stop_beating(&mut self) -> Result<()> {
        self.stop_heart_function().await
    }
    
    fn get_heart_rate(&self) -> f64 {
        60000.0 / self.config.cardiac_cycle_duration.as_millis() as f64
    }
    
    async fn get_rhythm_stability(&self) -> Result<f64> {
        let metrics = self.get_performance_metrics().await;
        Ok(metrics.rhythm_stability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_oscillatory_heart_creation() {
        let config = HeartConfig::default();
        let heart = OscillatoryHeart::new(config).await.unwrap();

        assert_eq!(heart.function_state, HeartFunctionState::Stopped);
        assert_eq!(heart.get_heart_rate(), 75.0); // 75 BPM from 800ms cycle
    }

    #[test]
    fn test_cardiac_rhythm_creation() {
        let cycle_duration = Duration::from_millis(800);
        let rhythm = CardiacRhythm::new(cycle_duration);

        assert_eq!(rhythm.cycle_duration, cycle_duration);
        assert_eq!(rhythm.state, RhythmState::Stopped);
    }

    #[tokio::test]
    async fn test_systolic_oscillations_coordination() {
        let s_entropy = SCredits::new(3000.0, 3000.0, 3000.0); // Full reserves
        let oscillations = OscillatoryHeart::coordinate_systolic_oscillations(
            &Arc::new(RwLock::new(crate::s_entropy_bank::SEntropyBank::new(
                SCreditReserves::new(s_entropy)
            )))
        ).await.unwrap();

        assert!(oscillations.amplitude >= 120.0); // Should be at least baseline
        assert_eq!(oscillations.frequency, 1.25); // 75 BPM
        assert!(oscillations.s_entropy_coordination.s_knowledge_utilization > 0.0);
    }

    #[tokio::test]
    async fn test_diastolic_oscillations_coordination() {
        let s_entropy_bank = Arc::new(RwLock::new(crate::s_entropy_bank::SEntropyBank::new(
            SCreditReserves::new(SCredits::new(1000.0, 1000.0, 1000.0))
        )));
        
        let oscillations = OscillatoryHeart::coordinate_diastolic_oscillations(&s_entropy_bank).await.unwrap();

        assert_eq!(oscillations.relaxation_pressure, 80.0); // Diastolic pressure
        assert_eq!(oscillations.suction_efficiency, 0.93);
        assert!(oscillations.collection_coordination > 0.0);
    }

    #[test]
    fn test_heart_config_defaults() {
        let config = HeartConfig::default();

        assert_eq!(config.cardiac_cycle_duration, Duration::from_millis(800));
        assert_eq!(config.systolic_fraction, 0.35);
        assert_eq!(config.diastolic_fraction, 0.65);
    }

    #[test]
    fn test_heart_performance_targets() {
        let targets = HeartPerformanceTargets::default();

        assert_eq!(targets.target_systolic_efficiency, 0.95);
        assert_eq!(targets.target_diastolic_efficiency, 0.93);
        assert_eq!(targets.target_rhythm_stability, 0.97);
        assert_eq!(targets.target_s_entropy_efficiency, 0.96);
    }

    #[test]
    fn test_cardiac_cycle_creation() {
        let cycle = CardiacCycle::new();

        assert!(cycle.systolic.is_none());
        assert!(cycle.diastolic.is_none());
        assert_eq!(cycle.state, CycleState::Ready);
    }

    #[test]
    fn test_heart_rate_calculation() {
        let config = HeartConfig {
            cardiac_cycle_duration: Duration::from_millis(600), // Faster heart rate
            ..HeartConfig::default()
        };

        let heart_rate = 60000.0 / config.cardiac_cycle_duration.as_millis() as f64;
        assert_eq!(heart_rate, 100.0); // 100 BPM
    }

    #[test]
    fn test_rhythm_stability_calculation() {
        let average_time = Duration::from_millis(800);
        let current_time = Duration::from_millis(820); // 20ms deviation

        let stability = OscillatoryHeart::calculate_rhythm_stability(&average_time, current_time);
        
        assert!(stability > 0.9); // Should be high stability
        assert!(stability <= 1.0);
    }

    #[test]
    fn test_vb_volume_manager() {
        let manager = VBVolumeManager::new();

        assert_eq!(manager.current_volume, 5000.0); // Physiological volume
        assert_eq!(manager.regulation_params.target_volume, 5000.0);
    }
}
