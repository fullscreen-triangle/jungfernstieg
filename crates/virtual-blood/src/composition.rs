//! Virtual Blood composition implementation
//!
//! Implements the complete Virtual Blood composition as defined in the theoretical framework:
//! VB_bio(t) = {VB_standard(t), O₂(t), N_nutrients(t), M_metabolites(t), I_immune(t)}
//!
//! This module provides the fundamental Virtual Blood composition that enables 
//! simultaneous biological sustenance and computational processing through 
//! S-entropy navigation and BMD orchestration.

use jungfernstieg_core::{SCredits, ComponentId, JungfernstiegError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

/// Unique identifier for Virtual Blood instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VirtualBloodId(pub Uuid);

impl VirtualBloodId {
    /// Generate a new Virtual Blood ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VirtualBloodId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<VirtualBloodId> for ComponentId {
    fn from(id: VirtualBloodId) -> Self {
        ComponentId(id.0)
    }
}

/// Complete Virtual Blood composition implementing the theoretical framework
///
/// VB_bio(t) = {VB_standard(t), O₂(t), N_nutrients(t), M_metabolites(t), I_immune(t)}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBlood {
    /// Virtual Blood identifier
    pub id: VirtualBloodId,
    /// Standard Virtual Blood environmental profile
    pub environmental_profile: EnvironmentalProfile,
    /// Dissolved oxygen concentration and transport dynamics
    pub oxygen_concentration: f64, // mg/L
    /// Nutrient profile (glucose, amino acids, lipids)
    pub nutrient_profile: NutrientProfile,
    /// Metabolite profile (waste products, signaling molecules)
    pub metabolite_profile: MetaboliteProfile,
    /// Immune cell populations and inflammatory factors
    pub immune_profile: ImmuneProfile,
    /// S-entropy coordinates for navigation
    pub s_entropy_coordinates: SCredits,
    /// Composition timestamp
    pub timestamp: Instant,
    /// Quality assessment
    pub quality: crate::VirtualBloodQuality,
}

impl VirtualBlood {
    /// Create new Virtual Blood with specified composition
    pub fn new(composition: VirtualBloodComposition) -> Self {
        Self {
            id: VirtualBloodId::new(),
            environmental_profile: composition.environmental_profile,
            oxygen_concentration: composition.oxygen_concentration,
            nutrient_profile: composition.nutrient_profile,
            metabolite_profile: composition.metabolite_profile,
            immune_profile: composition.immune_profile,
            s_entropy_coordinates: composition.s_entropy_coordinates,
            timestamp: Instant::now(),
            quality: crate::VirtualBloodQuality::Optimal,
        }
    }

    /// Create Virtual Blood with optimal biological parameters
    pub fn optimal_biological() -> Self {
        Self::new(VirtualBloodComposition::optimal_biological())
    }

    /// Calculate total dissolved oxygen content
    pub fn total_oxygen_content(&self) -> f64 {
        // Standard dissolved oxygen calculation for biological systems
        self.oxygen_concentration * 0.00314 // mL O₂/100mL blood at normal conditions
    }

    /// Calculate nutrient density for neural sustenance
    pub fn neural_nutrient_density(&self) -> f64 {
        self.nutrient_profile.glucose_concentration * 0.4
            + self.nutrient_profile.amino_acid_concentration * 0.3
            + self.nutrient_profile.lipid_concentration * 0.3
    }

    /// Assess metabolic waste load
    pub fn metabolic_waste_load(&self) -> f64 {
        self.metabolite_profile.lactate_concentration * 0.5
            + self.metabolite_profile.urea_concentration * 0.3
            + self.metabolite_profile.co2_concentration * 0.2
    }

    /// Check if Virtual Blood can support neural viability
    pub fn can_support_neural_viability(&self) -> bool {
        self.oxygen_concentration >= 6.5 // mg/L minimum for neural function
            && self.nutrient_profile.glucose_concentration >= 4.0 // mmol/L minimum
            && self.metabolic_waste_load() <= 15.0 // Maximum acceptable waste load
            && self.immune_profile.inflammatory_index <= 2.0 // Low inflammation
    }

    /// Update Virtual Blood composition with new components
    pub fn update_composition(&mut self, new_composition: VirtualBloodComposition) -> Result<()> {
        // Validate composition before update
        if !self.validate_biological_constraints(&new_composition)? {
            return Err(JungfernstiegError::ValidationError {
                field: "composition".to_string(),
                message: "New composition violates biological constraints".to_string(),
            });
        }

        self.environmental_profile = new_composition.environmental_profile;
        self.oxygen_concentration = new_composition.oxygen_concentration;
        self.nutrient_profile = new_composition.nutrient_profile;
        self.metabolite_profile = new_composition.metabolite_profile;
        self.immune_profile = new_composition.immune_profile;
        self.s_entropy_coordinates = new_composition.s_entropy_coordinates;
        self.timestamp = Instant::now();

        // Reassess quality after update
        self.quality = self.assess_quality();

        Ok(())
    }

    /// Validate biological constraints for Virtual Blood composition
    fn validate_biological_constraints(&self, composition: &VirtualBloodComposition) -> Result<bool> {
        // Oxygen concentration constraints
        if composition.oxygen_concentration < 0.0 || composition.oxygen_concentration > 15.0 {
            return Ok(false);
        }

        // Glucose concentration constraints (normal: 4.0-7.8 mmol/L)
        if composition.nutrient_profile.glucose_concentration < 3.0 
            || composition.nutrient_profile.glucose_concentration > 12.0 {
            return Ok(false);
        }

        // pH constraints (normal: 7.35-7.45)
        if composition.environmental_profile.ph_level < 7.2 
            || composition.environmental_profile.ph_level > 7.6 {
            return Ok(false);
        }

        // Temperature constraints (normal: 36.5-37.5°C)
        if composition.environmental_profile.temperature_celsius < 35.0 
            || composition.environmental_profile.temperature_celsius > 39.0 {
            return Ok(false);
        }

        Ok(true)
    }

    /// Assess current Virtual Blood quality
    fn assess_quality(&self) -> crate::VirtualBloodQuality {
        let oxygen_quality = if self.oxygen_concentration >= 8.0 { 1.0 }
        else if self.oxygen_concentration >= 6.5 { 0.8 }
        else { 0.4 };

        let nutrient_quality = if self.nutrient_profile.glucose_concentration >= 5.0 { 1.0 }
        else if self.nutrient_profile.glucose_concentration >= 4.0 { 0.8 }
        else { 0.4 };

        let waste_quality = if self.metabolic_waste_load() <= 10.0 { 1.0 }
        else if self.metabolic_waste_load() <= 15.0 { 0.8 }
        else { 0.4 };

        let immune_quality = if self.immune_profile.inflammatory_index <= 1.5 { 1.0 }
        else if self.immune_profile.inflammatory_index <= 2.0 { 0.8 }
        else { 0.4 };

        let overall_quality = (oxygen_quality + nutrient_quality + waste_quality + immune_quality) / 4.0;

        if overall_quality >= 0.95 {
            crate::VirtualBloodQuality::Optimal
        } else if overall_quality >= 0.90 {
            crate::VirtualBloodQuality::Excellent
        } else if overall_quality >= 0.85 {
            crate::VirtualBloodQuality::VeryGood
        } else if overall_quality >= 0.75 {
            crate::VirtualBloodQuality::Good
        } else if overall_quality >= 0.65 {
            crate::VirtualBloodQuality::Stable
        } else if overall_quality >= 0.50 {
            crate::VirtualBloodQuality::Warning
        } else {
            crate::VirtualBloodQuality::Critical
        }
    }

    /// Get S-entropy coordinates for navigation
    pub fn s_entropy_coordinates(&self) -> &SCredits {
        &self.s_entropy_coordinates
    }

    /// Get current quality assessment
    pub fn quality(&self) -> &crate::VirtualBloodQuality {
        &self.quality
    }

    /// Get composition age
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

/// Virtual Blood composition specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBloodComposition {
    /// Environmental sensing profile
    pub environmental_profile: EnvironmentalProfile,
    /// Oxygen concentration (mg/L)
    pub oxygen_concentration: f64,
    /// Nutrient concentrations
    pub nutrient_profile: NutrientProfile,
    /// Metabolite concentrations
    pub metabolite_profile: MetaboliteProfile,
    /// Immune system components
    pub immune_profile: ImmuneProfile,
    /// S-entropy navigation coordinates
    pub s_entropy_coordinates: SCredits,
}

impl VirtualBloodComposition {
    /// Create optimal biological Virtual Blood composition
    pub fn optimal_biological() -> Self {
        Self {
            environmental_profile: EnvironmentalProfile::optimal(),
            oxygen_concentration: 8.5, // mg/L - optimal for neural function
            nutrient_profile: NutrientProfile::optimal_neural(),
            metabolite_profile: MetaboliteProfile::minimal_waste(),
            immune_profile: ImmuneProfile::healthy_baseline(),
            s_entropy_coordinates: SCredits::new(1000.0, 1000.0, 1000.0), // High S-credit reserves
        }
    }

    /// Create Virtual Blood composition for emergency situations
    pub fn emergency_composition() -> Self {
        Self {
            environmental_profile: EnvironmentalProfile::stable(),
            oxygen_concentration: 12.0, // Higher oxygen for emergency support
            nutrient_profile: NutrientProfile::high_energy(),
            metabolite_profile: MetaboliteProfile::minimal_waste(),
            immune_profile: ImmuneProfile::alert_state(),
            s_entropy_coordinates: SCredits::new(1500.0, 800.0, 1200.0), // Emergency S-credit allocation
        }
    }
}

/// Environmental profile component of Virtual Blood
/// 
/// Corresponds to VB_standard(t) in the theoretical framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalProfile {
    /// Temperature (°C)
    pub temperature_celsius: f64,
    /// pH level
    pub ph_level: f64,
    /// Osmolarity (mOsm/L)
    pub osmolarity: f64,
    /// Ionic strength
    pub ionic_strength: f64,
    /// Atmospheric pressure component
    pub atmospheric_pressure: f64,
    /// Environmental noise level (for consciousness integration)
    pub noise_level: f64,
}

impl EnvironmentalProfile {
    /// Optimal environmental profile for neural function
    pub fn optimal() -> Self {
        Self {
            temperature_celsius: 37.0,
            ph_level: 7.4,
            osmolarity: 320.0,
            ionic_strength: 0.15,
            atmospheric_pressure: 101325.0, // Pa
            noise_level: 0.1, // Low noise for optimal function
        }
    }

    /// Stable environmental profile
    pub fn stable() -> Self {
        Self {
            temperature_celsius: 37.2,
            ph_level: 7.38,
            osmolarity: 315.0,
            ionic_strength: 0.14,
            atmospheric_pressure: 101200.0, // Slightly varied
            noise_level: 0.15,
        }
    }
}

/// Nutrient profile component of Virtual Blood
/// 
/// Corresponds to N_nutrients(t) in the theoretical framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientProfile {
    /// Glucose concentration (mmol/L)
    pub glucose_concentration: f64,
    /// Amino acid concentration (mg/dL)
    pub amino_acid_concentration: f64,
    /// Lipid concentration (mg/dL)
    pub lipid_concentration: f64,
    /// Essential fatty acids (mg/dL)
    pub fatty_acid_concentration: f64,
    /// Vitamins and cofactors (μg/dL)
    pub vitamin_concentration: f64,
    /// Mineral content (mg/dL)
    pub mineral_concentration: f64,
}

impl NutrientProfile {
    /// Optimal nutrient profile for neural networks
    pub fn optimal_neural() -> Self {
        Self {
            glucose_concentration: 5.5, // mmol/L - optimal for neural metabolism
            amino_acid_concentration: 35.0, // mg/dL
            lipid_concentration: 150.0, // mg/dL
            fatty_acid_concentration: 25.0, // mg/dL
            vitamin_concentration: 50.0, // μg/dL
            mineral_concentration: 100.0, // mg/dL
        }
    }

    /// High energy nutrient profile for demanding operations
    pub fn high_energy() -> Self {
        Self {
            glucose_concentration: 7.0, // Higher glucose for energy demands
            amino_acid_concentration: 45.0,
            lipid_concentration: 180.0,
            fatty_acid_concentration: 35.0,
            vitamin_concentration: 65.0,
            mineral_concentration: 120.0,
        }
    }

    /// Calculate total nutrient density
    pub fn total_nutrient_density(&self) -> f64 {
        self.glucose_concentration * 0.4
            + (self.amino_acid_concentration / 10.0) * 0.3
            + (self.lipid_concentration / 100.0) * 0.2
            + (self.fatty_acid_concentration / 10.0) * 0.1
    }
}

/// Metabolite profile component of Virtual Blood
/// 
/// Corresponds to M_metabolites(t) in the theoretical framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaboliteProfile {
    /// Lactate concentration (mmol/L)
    pub lactate_concentration: f64,
    /// Urea concentration (mg/dL)
    pub urea_concentration: f64,
    /// CO₂ concentration (mmol/L)
    pub co2_concentration: f64,
    /// Creatinine concentration (mg/dL)
    pub creatinine_concentration: f64,
    /// Signaling molecules (arbitrary units)
    pub signaling_molecules: HashMap<String, f64>,
    /// Metabolic byproducts (mg/dL)
    pub metabolic_byproducts: f64,
}

impl MetaboliteProfile {
    /// Minimal waste metabolite profile
    pub fn minimal_waste() -> Self {
        let mut signaling = HashMap::new();
        signaling.insert("ATP".to_string(), 2.5);
        signaling.insert("cAMP".to_string(), 0.1);
        signaling.insert("calcium".to_string(), 1.2);

        Self {
            lactate_concentration: 1.0, // mmol/L - low lactate
            urea_concentration: 15.0, // mg/dL - minimal urea
            co2_concentration: 24.0, // mmol/L - normal CO₂
            creatinine_concentration: 0.8, // mg/dL - normal creatinine
            signaling_molecules: signaling,
            metabolic_byproducts: 5.0, // mg/dL - minimal byproducts
        }
    }

    /// Calculate total waste burden
    pub fn total_waste_burden(&self) -> f64 {
        self.lactate_concentration * 0.4
            + (self.urea_concentration / 10.0) * 0.3
            + self.co2_concentration * 0.2
            + (self.metabolic_byproducts / 10.0) * 0.1
    }
}

/// Immune profile component of Virtual Blood
/// 
/// Corresponds to I_immune(t) in the theoretical framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmuneProfile {
    /// White blood cell count (cells/μL)
    pub wbc_count: f64,
    /// Neutrophil percentage
    pub neutrophil_percentage: f64,
    /// Lymphocyte percentage
    pub lymphocyte_percentage: f64,
    /// Monocyte percentage
    pub monocyte_percentage: f64,
    /// Inflammatory markers (mg/L)
    pub inflammatory_markers: HashMap<String, f64>,
    /// Inflammatory index (0-10 scale)
    pub inflammatory_index: f64,
    /// Immune system activation level (0-1 scale)
    pub activation_level: f64,
}

impl ImmuneProfile {
    /// Healthy baseline immune profile
    pub fn healthy_baseline() -> Self {
        let mut markers = HashMap::new();
        markers.insert("CRP".to_string(), 1.0); // mg/L - low C-reactive protein
        markers.insert("ESR".to_string(), 10.0); // mm/hr - normal ESR
        markers.insert("IL-6".to_string(), 2.0); // pg/mL - low IL-6

        Self {
            wbc_count: 6500.0, // cells/μL - normal range
            neutrophil_percentage: 60.0,
            lymphocyte_percentage: 30.0,
            monocyte_percentage: 8.0,
            inflammatory_markers: markers,
            inflammatory_index: 1.2, // Low inflammation
            activation_level: 0.3, // Baseline activation
        }
    }

    /// Alert state immune profile for threat response
    pub fn alert_state() -> Self {
        let mut markers = HashMap::new();
        markers.insert("CRP".to_string(), 3.0); // Elevated but controlled
        markers.insert("ESR".to_string(), 15.0);
        markers.insert("IL-6".to_string(), 5.0);

        Self {
            wbc_count: 8000.0, // Elevated but within range
            neutrophil_percentage: 65.0,
            lymphocyte_percentage: 25.0,
            monocyte_percentage: 8.0,
            inflammatory_markers: markers,
            inflammatory_index: 1.8, // Moderate inflammation
            activation_level: 0.6, // Elevated activation
        }
    }

    /// Calculate immune system activation level
    pub fn immune_activation(&self) -> f64 {
        let wbc_factor = (self.wbc_count / 6500.0).min(2.0); // Normalized to baseline
        let inflammatory_factor = self.inflammatory_index / 10.0;
        let activation_factor = self.activation_level;

        (wbc_factor + inflammatory_factor + activation_factor) / 3.0
    }
}

/// Biological components container for grouped operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalComponents {
    /// Oxygen transport components
    pub oxygen_components: OxygenComponents,
    /// Nutrient delivery components
    pub nutrient_components: NutrientComponents,
    /// Waste management components
    pub waste_components: WasteComponents,
    /// Immune monitoring components
    pub immune_components: ImmuneComponents,
}

/// Oxygen transport specific components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxygenComponents {
    /// Virtual oxygen carriers (hemoglobin equivalent)
    pub virtual_carriers: Vec<VirtualOxygenCarrier>,
    /// Oxygen binding affinity
    pub binding_affinity: f64,
    /// Transport efficiency
    pub transport_efficiency: f64,
    /// Release kinetics
    pub release_kinetics: f64,
}

/// Virtual oxygen carrier structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualOxygenCarrier {
    /// Carrier identifier
    pub id: Uuid,
    /// Oxygen binding capacity
    pub binding_capacity: f64,
    /// Current oxygen saturation (0.0-1.0)
    pub saturation: f64,
    /// S-entropy optimization state
    pub s_entropy_optimized: bool,
    /// Transport efficiency factor
    pub efficiency_factor: f64,
}

impl VirtualOxygenCarrier {
    /// Create new virtual oxygen carrier with optimal parameters
    pub fn new_optimal() -> Self {
        Self {
            id: Uuid::new_v4(),
            binding_capacity: 1.34, // mL O₂/g Hb equivalent
            saturation: 0.98, // 98% saturation
            s_entropy_optimized: true,
            efficiency_factor: 0.987, // Target efficiency from framework
        }
    }

    /// Calculate oxygen content for this carrier
    pub fn oxygen_content(&self) -> f64 {
        self.binding_capacity * self.saturation * self.efficiency_factor
    }
}

/// Nutrient delivery components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientComponents {
    /// Glucose transport mechanisms
    pub glucose_transporters: usize,
    /// Amino acid delivery efficiency
    pub amino_acid_efficiency: f64,
    /// Lipid transport capacity
    pub lipid_transport_capacity: f64,
    /// Vitamin delivery systems
    pub vitamin_delivery_systems: usize,
}

/// Waste management components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasteComponents {
    /// Waste collection efficiency
    pub collection_efficiency: f64,
    /// Metabolite filtering capacity
    pub filtering_capacity: f64,
    /// CO₂ transport efficiency
    pub co2_transport_efficiency: f64,
    /// Toxin neutralization capacity
    pub toxin_neutralization: f64,
}

/// Immune monitoring components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmuneComponents {
    /// Monitoring cell populations
    pub monitoring_cells: HashMap<String, usize>,
    /// Communication efficiency
    pub communication_efficiency: f64,
    /// Alert response time (ms)
    pub alert_response_time: f64,
    /// Threat detection sensitivity
    pub threat_detection_sensitivity: f64,
}

impl Default for BiologicalComponents {
    fn default() -> Self {
        let mut monitoring_cells = HashMap::new();
        monitoring_cells.insert("macrophages".to_string(), 1000);
        monitoring_cells.insert("t_cells".to_string(), 500);
        monitoring_cells.insert("b_cells".to_string(), 300);
        monitoring_cells.insert("neutrophils".to_string(), 800);

        Self {
            oxygen_components: OxygenComponents {
                virtual_carriers: vec![VirtualOxygenCarrier::new_optimal(); 4],
                binding_affinity: 0.98,
                transport_efficiency: 0.987,
                release_kinetics: 0.95,
            },
            nutrient_components: NutrientComponents {
                glucose_transporters: 1000,
                amino_acid_efficiency: 0.92,
                lipid_transport_capacity: 150.0,
                vitamin_delivery_systems: 500,
            },
            waste_components: WasteComponents {
                collection_efficiency: 0.94,
                filtering_capacity: 200.0,
                co2_transport_efficiency: 0.96,
                toxin_neutralization: 0.88,
            },
            immune_components: ImmuneComponents {
                monitoring_cells,
                communication_efficiency: 0.95,
                alert_response_time: 150.0, // ms
                threat_detection_sensitivity: 0.92,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_blood_creation() {
        let composition = VirtualBloodComposition::optimal_biological();
        let vb = VirtualBlood::new(composition);

        assert!(matches!(vb.quality, crate::VirtualBloodQuality::Optimal));
        assert!(vb.can_support_neural_viability());
        assert!(vb.total_oxygen_content() > 0.0);
    }

    #[test]
    fn test_biological_constraints_validation() {
        let mut composition = VirtualBloodComposition::optimal_biological();
        let vb = VirtualBlood::new(composition.clone());

        // Test valid composition
        assert!(vb.validate_biological_constraints(&composition).unwrap());

        // Test invalid oxygen concentration
        composition.oxygen_concentration = 20.0; // Too high
        assert!(!vb.validate_biological_constraints(&composition).unwrap());

        // Test invalid pH
        composition.oxygen_concentration = 8.0; // Fix oxygen
        composition.environmental_profile.ph_level = 6.0; // Too low
        assert!(!vb.validate_biological_constraints(&composition).unwrap());
    }

    #[test]
    fn test_nutrient_density_calculation() {
        let profile = NutrientProfile::optimal_neural();
        let density = profile.total_nutrient_density();
        
        assert!(density > 0.0);
        assert!(density < 10.0); // Reasonable range
    }

    #[test]
    fn test_metabolic_waste_assessment() {
        let profile = MetaboliteProfile::minimal_waste();
        let waste_load = profile.total_waste_burden();
        
        assert!(waste_load > 0.0);
        assert!(waste_load < 20.0); // Should be manageable
    }

    #[test]
    fn test_immune_activation_calculation() {
        let healthy = ImmuneProfile::healthy_baseline();
        let alert = ImmuneProfile::alert_state();

        let healthy_activation = healthy.immune_activation();
        let alert_activation = alert.immune_activation();

        assert!(healthy_activation < alert_activation);
        assert!(healthy_activation > 0.0 && healthy_activation < 1.0);
        assert!(alert_activation > 0.0 && alert_activation < 2.0);
    }

    #[test]
    fn test_virtual_oxygen_carrier() {
        let carrier = VirtualOxygenCarrier::new_optimal();
        
        assert_eq!(carrier.saturation, 0.98);
        assert_eq!(carrier.efficiency_factor, crate::TARGET_OXYGEN_EFFICIENCY);
        assert!(carrier.s_entropy_optimized);
        assert!(carrier.oxygen_content() > 0.0);
    }

    #[test]
    fn test_emergency_composition() {
        let emergency = VirtualBloodComposition::emergency_composition();
        let vb = VirtualBlood::new(emergency);

        assert!(vb.oxygen_concentration > 10.0); // Higher oxygen for emergency
        assert!(vb.can_support_neural_viability());
        assert!(vb.s_entropy_coordinates.total() > 3000.0); // High S-credit reserves
    }

    #[test]
    fn test_virtual_blood_aging() {
        let composition = VirtualBloodComposition::optimal_biological();
        let vb = VirtualBlood::new(composition);

        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(vb.age().as_millis() >= 10);
    }

    #[test]
    fn test_biological_components_default() {
        let components = BiologicalComponents::default();

        assert!(!components.oxygen_components.virtual_carriers.is_empty());
        assert!(components.nutrient_components.glucose_transporters > 0);
        assert!(components.waste_components.collection_efficiency > 0.0);
        assert!(!components.immune_components.monitoring_cells.is_empty());
    }
}
