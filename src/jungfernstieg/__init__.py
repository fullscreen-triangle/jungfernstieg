"""
Jungfernstieg: Biological-Virtual Neural Symbiosis System

This module provides the core framework for sustaining living biological neural networks
through Virtual Blood circulatory systems powered by Oscillatory Virtual Machine architecture.

CRITICAL SAFETY WARNING:
This system manages living biological neural tissue. All safety protocols must be 
initialized and validated before any biological operations commence.

Usage:
    >>> import jungfernstieg as jf
    >>> # MANDATORY: Initialize safety systems first
    >>> jf.safety.initialize_all_protocols()
    >>> jf.safety.validate_system_safety()
    >>> # Only after safety validation
    >>> system = jf.JungfernstiegSystem()

Modules:
    - biological: Living neural network management and monitoring
    - virtual_blood: Virtual Blood circulation and composition systems  
    - oscillatory_vm: Oscillatory Virtual Machine core and S-entropy navigation
    - consciousness: Human-AI consciousness integration layer
    - safety: Critical safety protocols and emergency systems
    - sensors: Environmental sensing framework integration
    - monitoring: Real-time system monitoring and diagnostics
"""

import logging
import sys
import warnings
from typing import Optional

# Configure logging for biological safety compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jungfernstieg_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Version information
__version__ = "0.1.0"
__author__ = "Kundai Farai Sachikonye"
__email__ = "kundai.sachikonye@wzw.tum.de"
__license__ = "MIT with Biological Safety Requirements"

# Critical system status
_SAFETY_INITIALIZED = False
_SAFETY_VALIDATED = False
_NEURAL_SYSTEMS_ACTIVE = False

def get_version() -> str:
    """Get the current version of Jungfernstieg."""
    return __version__

def get_system_status() -> dict:
    """Get current system safety and operational status."""
    return {
        "version": __version__,
        "safety_initialized": _SAFETY_INITIALIZED,
        "safety_validated": _SAFETY_VALIDATED,
        "neural_systems_active": _NEURAL_SYSTEMS_ACTIVE,
        "biological_safety_level": "BSL-2+" if _SAFETY_VALIDATED else "UNDEFINED"
    }

def check_safety_prerequisites() -> bool:
    """Check if safety prerequisites are met for biological operations."""
    if not _SAFETY_INITIALIZED:
        warnings.warn(
            "CRITICAL SAFETY WARNING: Safety systems not initialized. "
            "Call jungfernstieg.safety.initialize_all_protocols() first.",
            UserWarning,
            stacklevel=2
        )
        return False
    
    if not _SAFETY_VALIDATED:
        warnings.warn(
            "CRITICAL SAFETY WARNING: Safety systems not validated. "
            "Call jungfernstieg.safety.validate_system_safety() first.",
            UserWarning,
            stacklevel=2
        )
        return False
    
    return True

# Import order is critical for safety compliance
try:
    # Import safety systems first - MANDATORY
    from . import safety
    
    # Core computational systems
    from . import oscillatory_vm
    from . import virtual_blood
    
    # Environmental sensing systems
    from . import sensors
    
    # Monitoring and diagnostics
    from . import monitoring
    
    # Biological systems - ONLY import after safety validation
    # Note: biological module has internal safety checks
    from . import biological
    
    # Consciousness integration layer
    from . import consciousness
    
    logger.info("Jungfernstieg modules imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import Jungfernstieg modules: {e}")
    raise

# Main system class
class JungfernstiegSystem:
    """
    Main Jungfernstieg system controller for biological-virtual neural symbiosis.
    
    This class coordinates all system components including biological neural networks,
    Virtual Blood circulation, Oscillatory VM operation, and consciousness integration.
    
    CRITICAL: Safety protocols must be initialized and validated before instantiation.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Jungfernstieg system.
        
        Args:
            config: Optional system configuration dictionary
            
        Raises:
            RuntimeError: If safety prerequisites are not met
            ValueError: If configuration is invalid
        """
        if not check_safety_prerequisites():
            raise RuntimeError(
                "CRITICAL SAFETY ERROR: Cannot initialize Jungfernstieg system "
                "without safety validation. Initialize safety protocols first."
            )
        
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.JungfernstiegSystem")
        
        # Initialize core components
        self._safety_manager = None
        self._biological_manager = None  
        self._virtual_blood_system = None
        self._oscillatory_vm = None
        self._consciousness_layer = None
        self._monitoring_system = None
        
        self.logger.info("Jungfernstieg system initialized with safety validation")
    
    def initialize_all_systems(self) -> bool:
        """
        Initialize all Jungfernstieg system components in safe order.
        
        Returns:
            bool: True if all systems initialized successfully
        """
        try:
            self.logger.info("Initializing Jungfernstieg system components...")
            
            # 1. Safety manager (already validated)
            self._safety_manager = safety.SafetyManager()
            
            # 2. Monitoring system
            self._monitoring_system = monitoring.SystemMonitor()
            
            # 3. Oscillatory Virtual Machine (computational heart)
            self._oscillatory_vm = oscillatory_vm.OscillatoryVM()
            
            # 4. Virtual Blood circulation system
            self._virtual_blood_system = virtual_blood.VirtualBloodSystem(
                vm_heart=self._oscillatory_vm
            )
            
            # 5. Biological systems (with neural tissue)
            self._biological_manager = biological.BiologicalManager(
                virtual_blood=self._virtual_blood_system,
                safety_manager=self._safety_manager
            )
            
            # 6. Consciousness integration layer
            self._consciousness_layer = consciousness.ConsciousnessLayer(
                biological_manager=self._biological_manager,
                virtual_blood=self._virtual_blood_system
            )
            
            self.logger.info("All Jungfernstieg systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize systems: {e}")
            self._emergency_shutdown()
            return False
    
    def start_neural_operations(self) -> bool:
        """
        Start neural operations with full safety monitoring.
        
        Returns:
            bool: True if neural operations started successfully
        """
        if not self._biological_manager:
            raise RuntimeError("Biological manager not initialized")
        
        return self._biological_manager.start_neural_operations()
    
    def get_neural_viability(self) -> float:
        """
        Get current neural viability percentage.
        
        Returns:
            float: Neural viability percentage (0.0 to 100.0)
        """
        if not self._biological_manager:
            return 0.0
        
        return self._biological_manager.get_neural_viability()
    
    def _emergency_shutdown(self) -> None:
        """Execute emergency shutdown of all systems."""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        if self._biological_manager:
            self._biological_manager.emergency_shutdown()
        
        if self._virtual_blood_system:
            self._virtual_blood_system.emergency_shutdown()
        
        if self._safety_manager:
            self._safety_manager.log_emergency_event()

# Export main classes and functions
__all__ = [
    "JungfernstiegSystem",
    "get_version", 
    "get_system_status",
    "check_safety_prerequisites",
    "safety",
    "biological", 
    "virtual_blood",
    "oscillatory_vm",
    "consciousness",
    "sensors",
    "monitoring"
]

# Safety reminder on import
logger.warning(
    "JUNGFERNSTIEG SAFETY REMINDER: This system manages living biological neural tissue. "
    "Initialize safety protocols before any biological operations."
)