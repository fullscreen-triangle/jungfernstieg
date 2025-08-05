//! Grand Flux Standards implementation
//!
//! Provides universal reference patterns for Virtual Blood circulation,
//! analogous to circuit equivalent theory for complex flow systems.

use jungfernstieg_core::{JungfernstiegError, Result, SCredits};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Unique identifier for flux patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FluxPatternId(pub Uuid);

impl FluxPatternId {
    /// Generate a new flux pattern ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for FluxPatternId {
    fn default() -> Self {
        Self::new()
    }
}

/// Grand Flux Standard - universal reference pattern for circulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrandFluxStandard {
    /// Pattern identifier
    pub id: FluxPatternId,
    /// Pattern name and description
    pub name: String,
    /// Reference flow rate (ideal conditions)
    pub reference_flow_rate: f64,
    /// Reference pressure (Pa)
    pub reference_pressure: f64,
    /// Reference temperature (Celsius)
    pub reference_temperature: f64,
    /// Reference fluid properties
    pub reference_fluid: ReferenceFluid,
    /// Geometric constraints
    pub geometry: ReferenceGeometry,
    /// Pattern viability (0.0 to 1.0)
    pub viability: f64,
    /// S-entropy coordinates for this pattern
    pub s_entropy_coordinates: SCredits,
    /// Correction factors library
    pub correction_factors: HashMap<String, CorrectionFactor>,
}

impl GrandFluxStandard {
    /// Create new Grand Flux Standard
    pub fn new(
        name: String,
        reference_flow_rate: f64,
        reference_pressure: f64,
        viability: f64,
    ) -> Self {
        Self {
            id: FluxPatternId::new(),
            name,
            reference_flow_rate,
            reference_pressure,
            reference_temperature: 37.0, // Biological temperature
            reference_fluid: ReferenceFluid::virtual_blood(),
            geometry: ReferenceGeometry::default(),
            viability,
            s_entropy_coordinates: SCredits::new(1000.0, 1000.0, 1000.0),
            correction_factors: HashMap::new(),
        }
    }

    /// Calculate real flow rate with correction factors
    pub fn calculate_real_flow(&self, conditions: &FlowConditions) -> Result<f64> {
        let mut corrected_flow = self.reference_flow_rate;

        // Apply temperature correction
        if let Some(temp_factor) = self.correction_factors.get("temperature") {
            corrected_flow *= temp_factor.apply(conditions.temperature, self.reference_temperature)?;
        }

        // Apply pressure correction
        if let Some(pressure_factor) = self.correction_factors.get("pressure") {
            corrected_flow *= pressure_factor.apply(conditions.pressure, self.reference_pressure)?;
        }

        // Apply geometry correction
        if let Some(geom_factor) = self.correction_factors.get("geometry") {
            corrected_flow *= geom_factor.apply_geometry_correction(&conditions.geometry, &self.geometry)?;
        }

        // Apply St. Stella constant scaling for low-information conditions
        if conditions.information_availability < 0.5 {
            corrected_flow *= crate::STELLA_CONSTANT;
        }

        Ok(corrected_flow)
    }

    /// Add correction factor
    pub fn add_correction_factor(&mut self, name: String, factor: CorrectionFactor) {
        self.correction_factors.insert(name, factor);
    }

    /// Get pattern alignment cost with another pattern
    pub fn alignment_cost(&self, other: &CirculationPattern) -> f64 {
        // Calculate S-entropy distance
        let s_distance = (
            (self.s_entropy_coordinates.s_knowledge - other.s_entropy_demand.s_knowledge).powi(2) +
            (self.s_entropy_coordinates.s_time - other.s_entropy_demand.s_time).powi(2) +
            (self.s_entropy_coordinates.s_entropy - other.s_entropy_demand.s_entropy).powi(2)
        ).sqrt();

        // Include viability difference
        let viability_distance = (self.viability - other.current_viability).abs();

        s_distance + viability_distance * 100.0
    }
}

/// Reference fluid properties for Grand Flux Standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceFluid {
    /// Fluid name
    pub name: String,
    /// Density (kg/m³)
    pub density: f64,
    /// Dynamic viscosity (Pa·s)
    pub viscosity: f64,
    /// Thermal conductivity (W/m·K)
    pub thermal_conductivity: f64,
    /// Specific heat capacity (J/kg·K)
    pub specific_heat: f64,
}

impl ReferenceFluid {
    /// Create Virtual Blood reference fluid
    pub fn virtual_blood() -> Self {
        Self {
            name: "Virtual Blood".to_string(),
            density: 1060.0,     // Similar to blood density
            viscosity: 0.0035,   // Blood viscosity at 37°C
            thermal_conductivity: 0.52,
            specific_heat: 3617.0,
        }
    }

    /// Create standard water reference
    pub fn water() -> Self {
        Self {
            name: "Water".to_string(),
            density: 1000.0,
            viscosity: 0.001,
            thermal_conductivity: 0.6,
            specific_heat: 4184.0,
        }
    }
}

/// Reference geometry for flux standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceGeometry {
    /// Geometric type
    pub geometry_type: GeometryType,
    /// Characteristic length (m)
    pub characteristic_length: f64,
    /// Characteristic area (m²)
    pub characteristic_area: f64,
    /// Surface roughness (m)
    pub surface_roughness: f64,
}

impl Default for ReferenceGeometry {
    fn default() -> Self {
        Self {
            geometry_type: GeometryType::CircularPipe,
            characteristic_length: 0.001, // 1mm diameter
            characteristic_area: std::f64::consts::PI * (0.0005_f64).powi(2),
            surface_roughness: 1e-6, // Very smooth
        }
    }
}

/// Geometry types for flux standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometryType {
    /// Circular pipe
    CircularPipe,
    /// Rectangular channel
    RectangularChannel,
    /// Complex network
    ComplexNetwork,
    /// Biological capillary
    BiologicalCapillary,
    /// Custom geometry
    Custom(String),
}

/// Correction factor for adjusting reference patterns to real conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionFactor {
    /// Factor name
    pub name: String,
    /// Factor type
    pub factor_type: CorrectionType,
    /// Scaling parameters
    pub parameters: HashMap<String, f64>,
}

impl CorrectionFactor {
    /// Apply correction factor
    pub fn apply(&self, actual_value: f64, reference_value: f64) -> Result<f64> {
        match self.factor_type {
            CorrectionType::Linear => {
                let slope = self.parameters.get("slope").unwrap_or(&1.0);
                Ok(1.0 + slope * (actual_value - reference_value) / reference_value)
            }
            CorrectionType::Exponential => {
                let exponent = self.parameters.get("exponent").unwrap_or(&1.0);
                Ok((actual_value / reference_value).powf(*exponent))
            }
            CorrectionType::Logarithmic => {
                let base = self.parameters.get("base").unwrap_or(&std::f64::consts::E);
                Ok(base.ln() / (actual_value / reference_value).ln())
            }
            CorrectionType::Custom => {
                // Custom correction logic would be implemented here
                Ok(1.0)
            }
        }
    }

    /// Apply geometry-specific correction
    pub fn apply_geometry_correction(
        &self,
        actual_geometry: &ReferenceGeometry,
        reference_geometry: &ReferenceGeometry,
    ) -> Result<f64> {
        match (&actual_geometry.geometry_type, &reference_geometry.geometry_type) {
            (GeometryType::CircularPipe, GeometryType::CircularPipe) => {
                // Apply diameter correction
                let diameter_ratio = actual_geometry.characteristic_length / reference_geometry.characteristic_length;
                Ok(diameter_ratio.powi(4)) // Flow scales as D⁴ for pipes
            }
            (GeometryType::BiologicalCapillary, GeometryType::CircularPipe) => {
                // Special correction for biological systems
                let bio_factor = self.parameters.get("biological_factor").unwrap_or(&0.8);
                let diameter_ratio = actual_geometry.characteristic_length / reference_geometry.characteristic_length;
                Ok(bio_factor * diameter_ratio.powi(4))
            }
            _ => {
                // Generic correction
                Ok(1.0)
            }
        }
    }
}

/// Types of correction factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionType {
    /// Linear correction
    Linear,
    /// Exponential correction
    Exponential,
    /// Logarithmic correction
    Logarithmic,
    /// Custom correction algorithm
    Custom,
}

/// Current flow conditions for correction factor application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConditions {
    /// Current temperature (Celsius)
    pub temperature: f64,
    /// Current pressure (Pa)
    pub pressure: f64,
    /// Current geometry
    pub geometry: ReferenceGeometry,
    /// Information availability (0.0 to 1.0)
    pub information_availability: f64,
    /// System viability
    pub system_viability: f64,
}

/// Circulation pattern for Virtual Blood systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CirculationPattern {
    /// Pattern identifier
    pub id: FluxPatternId,
    /// Pattern classification
    pub pattern_class: CirculationClass,
    /// Current viability (0.0 to 1.0)
    pub current_viability: f64,
    /// S-entropy demand for this pattern
    pub s_entropy_demand: SCredits,
    /// Flow requirements
    pub flow_requirements: FlowRequirements,
    /// Temporal characteristics
    pub temporal_profile: TemporalProfile,
}

impl CirculationPattern {
    /// Create new circulation pattern
    pub fn new(pattern_class: CirculationClass, viability: f64) -> Self {
        Self {
            id: FluxPatternId::new(),
            pattern_class,
            current_viability: viability,
            s_entropy_demand: SCredits::new(100.0, 100.0, 100.0),
            flow_requirements: FlowRequirements::default(),
            temporal_profile: TemporalProfile::default(),
        }
    }

    /// Calculate alignment gap with a Grand Flux Standard
    pub fn alignment_gap(&self, standard: &GrandFluxStandard) -> AlignmentGap {
        let viability_gap = (standard.viability - self.current_viability).abs();
        let s_entropy_gap = standard.alignment_cost(self);
        let flow_gap = (standard.reference_flow_rate - self.flow_requirements.target_flow_rate).abs();

        AlignmentGap {
            viability_gap,
            s_entropy_gap,
            flow_gap,
            total_gap: viability_gap + s_entropy_gap + flow_gap,
        }
    }
}

/// Classification of circulation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CirculationClass {
    /// Normal steady circulation
    Steady,
    /// Pulsatile circulation (heart-like)
    Pulsatile,
    /// Emergency high-flow circulation
    Emergency,
    /// Low-flow maintenance circulation
    Maintenance,
    /// Complex multi-phase circulation
    MultiPhase,
    /// Transitional circulation pattern
    Transitional,
}

/// Flow requirements for circulation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowRequirements {
    /// Target flow rate (mL/min)
    pub target_flow_rate: f64,
    /// Minimum acceptable flow rate
    pub min_flow_rate: f64,
    /// Maximum safe flow rate
    pub max_flow_rate: f64,
    /// Pressure requirements (Pa)
    pub pressure_range: (f64, f64),
}

impl Default for FlowRequirements {
    fn default() -> Self {
        Self {
            target_flow_rate: 100.0,
            min_flow_rate: 50.0,
            max_flow_rate: 200.0,
            pressure_range: (1000.0, 5000.0),
        }
    }
}

/// Temporal profile for circulation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalProfile {
    /// Cycle duration
    pub cycle_duration: Duration,
    /// Systolic fraction (0.0 to 1.0)
    pub systolic_fraction: f64,
    /// Flow rate variation profile
    pub flow_profile: Vec<f64>,
}

impl Default for TemporalProfile {
    fn default() -> Self {
        Self {
            cycle_duration: Duration::from_millis(1000), // 1 second cycle
            systolic_fraction: 0.35, // 35% systolic
            flow_profile: vec![1.0, 0.8, 0.6, 0.8], // Simple profile
        }
    }
}

/// Gap analysis between circulation patterns and standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentGap {
    /// Viability gap
    pub viability_gap: f64,
    /// S-entropy coordinate gap
    pub s_entropy_gap: f64,
    /// Flow rate gap
    pub flow_gap: f64,
    /// Total alignment gap
    pub total_gap: f64,
}

impl AlignmentGap {
    /// Check if gap is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.total_gap < 100.0 // Configurable threshold
    }

    /// Get gap priority level
    pub fn priority_level(&self) -> GapPriority {
        if self.total_gap < 50.0 {
            GapPriority::Low
        } else if self.total_gap < 150.0 {
            GapPriority::Medium
        } else if self.total_gap < 300.0 {
            GapPriority::High
        } else {
            GapPriority::Critical
        }
    }
}

/// Priority levels for alignment gaps
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GapPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Library of Grand Flux Standards
pub struct FluxStandardLibrary {
    /// Collection of standards
    standards: HashMap<FluxPatternId, GrandFluxStandard>,
    /// Standards organized by circulation class
    class_index: HashMap<CirculationClass, Vec<FluxPatternId>>,
    /// Standards organized by viability range
    viability_index: HashMap<ViabilityRange, Vec<FluxPatternId>>,
}

impl FluxStandardLibrary {
    /// Create new flux standard library
    pub fn new() -> Self {
        Self {
            standards: HashMap::new(),
            class_index: HashMap::new(),
            viability_index: HashMap::new(),
        }
    }

    /// Create library with default Virtual Blood standards
    pub fn default_virtual_blood_library() -> Self {
        let mut library = Self::new();
        library.load_default_standards();
        library
    }

    /// Add standard to library
    pub fn add_standard(&mut self, standard: GrandFluxStandard) {
        let id = standard.id;
        
        // Add to main collection
        self.standards.insert(id, standard.clone());
        
        // Update indices would be implemented here
        // For now, simple implementation
    }

    /// Find best matching standard for circulation pattern
    pub fn find_best_match(&self, pattern: &CirculationPattern) -> Option<&GrandFluxStandard> {
        let mut best_match = None;
        let mut best_gap = f64::INFINITY;

        for standard in self.standards.values() {
            let gap = pattern.alignment_gap(standard);
            if gap.total_gap < best_gap {
                best_gap = gap.total_gap;
                best_match = Some(standard);
            }
        }

        best_match
    }

    /// Load default standards for Virtual Blood circulation
    fn load_default_standards(&mut self) {
        // Standard steady circulation
        let mut steady_standard = GrandFluxStandard::new(
            "Steady Virtual Blood Circulation".to_string(),
            100.0, // 100 mL/min
            2000.0, // 2000 Pa
            0.95, // 95% viability
        );
        
        // Add correction factors
        steady_standard.add_correction_factor(
            "temperature".to_string(),
            CorrectionFactor {
                name: "Temperature Correction".to_string(),
                factor_type: CorrectionType::Linear,
                parameters: [("slope".to_string(), 0.02)].iter().cloned().collect(),
            }
        );

        self.add_standard(steady_standard);

        // Emergency high-flow circulation
        let emergency_standard = GrandFluxStandard::new(
            "Emergency High-Flow Circulation".to_string(),
            300.0, // 300 mL/min
            5000.0, // 5000 Pa
            0.85, // 85% viability acceptable in emergency
        );

        self.add_standard(emergency_standard);

        // Maintenance low-flow circulation
        let maintenance_standard = GrandFluxStandard::new(
            "Maintenance Low-Flow Circulation".to_string(),
            50.0, // 50 mL/min
            1000.0, // 1000 Pa
            0.98, // 98% viability for maintenance
        );

        self.add_standard(maintenance_standard);
    }

    /// Get all standards
    pub fn get_all_standards(&self) -> Vec<&GrandFluxStandard> {
        self.standards.values().collect()
    }

    /// Get standards by viability range
    pub fn get_standards_by_viability(&self, min_viability: f64, max_viability: f64) -> Vec<&GrandFluxStandard> {
        self.standards.values()
            .filter(|s| s.viability >= min_viability && s.viability <= max_viability)
            .collect()
    }
}

impl Default for FluxStandardLibrary {
    fn default() -> Self {
        Self::default_virtual_blood_library()
    }
}

/// Viability ranges for indexing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViabilityRange {
    Critical,    // 0-90%
    Warning,     // 90-95%
    Good,        // 95-98%
    Excellent,   // 98-100%
}

impl ViabilityRange {
    pub fn from_viability(viability: f64) -> Self {
        if viability < 0.90 {
            Self::Critical
        } else if viability < 0.95 {
            Self::Warning
        } else if viability < 0.98 {
            Self::Good
        } else {
            Self::Excellent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grand_flux_standard_creation() {
        let standard = GrandFluxStandard::new(
            "Test Standard".to_string(),
            100.0,
            2000.0,
            0.95,
        );

        assert_eq!(standard.name, "Test Standard");
        assert_eq!(standard.reference_flow_rate, 100.0);
        assert_eq!(standard.viability, 0.95);
    }

    #[test]
    fn test_circulation_pattern_alignment() {
        let pattern = CirculationPattern::new(CirculationClass::Steady, 0.90);
        let standard = GrandFluxStandard::new(
            "Test".to_string(),
            100.0,
            2000.0,
            0.95,
        );

        let gap = pattern.alignment_gap(&standard);
        assert!(gap.viability_gap > 0.0);
        assert!(!gap.is_acceptable() || gap.priority_level() != GapPriority::Critical);
    }

    #[test]
    fn test_flux_standard_library() {
        let library = FluxStandardLibrary::default_virtual_blood_library();
        let standards = library.get_all_standards();
        
        assert!(!standards.is_empty());
        assert!(standards.len() >= 3); // Should have at least steady, emergency, maintenance
    }

    #[test]
    fn test_correction_factor_application() {
        let mut factor = CorrectionFactor {
            name: "Test Factor".to_string(),
            factor_type: CorrectionType::Linear,
            parameters: [("slope".to_string(), 0.1)].iter().cloned().collect(),
        };

        let correction = factor.apply(40.0, 37.0).unwrap(); // 40°C vs 37°C reference
        assert!(correction > 1.0); // Should increase with higher temperature
    }

    #[test]
    fn test_viability_range_classification() {
        assert_eq!(ViabilityRange::from_viability(0.85), ViabilityRange::Critical);
        assert_eq!(ViabilityRange::from_viability(0.92), ViabilityRange::Warning);
        assert_eq!(ViabilityRange::from_viability(0.96), ViabilityRange::Good);
        assert_eq!(ViabilityRange::from_viability(0.99), ViabilityRange::Excellent);
    }
}