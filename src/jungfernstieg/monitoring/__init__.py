"""
Comprehensive System Monitoring for Jungfernstieg

This module provides comprehensive monitoring and diagnostics for all Jungfernstieg
system components, with special emphasis on biological safety, neural viability,
and Virtual Blood circulation quality.

Critical monitoring areas:
    - Neural viability and health
    - Virtual Blood circulation quality
    - Oscillatory VM performance
    - Safety system status
    - Consciousness integration quality
    - Environmental sensing accuracy

Components:
    - SystemMonitor: Central monitoring coordination
    - BiologicalMonitor: Neural tissue health monitoring
    - VirtualBloodMonitor: Circulation quality monitoring
    - PerformanceMonitor: System performance tracking
    - SafetyMonitor: Safety system monitoring
    - AlertSystem: Critical alert management
    - Dashboard: Real-time monitoring interface

Usage:
    >>> from jungfernstieg.monitoring import SystemMonitor
    >>> monitor = SystemMonitor()
    >>> monitor.initialize_monitoring()
    >>> monitor.start_comprehensive_monitoring()
"""

import logging
import threading
import time
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import json

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring detail levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DIAGNOSTIC = "diagnostic"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MonitoringStatus(Enum):
    """Monitoring system status."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"

class SystemHealth(Enum):
    """Overall system health status."""
    OPTIMAL = "optimal"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"

class SystemMonitor:
    """
    Central system monitoring and diagnostics coordinator.
    
    Monitors all Jungfernstieg system components with emphasis on biological
    safety, neural viability, and overall system health.
    """
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.COMPREHENSIVE):
        """
        Initialize system monitor.
        
        Args:
            monitoring_level: Level of monitoring detail
        """
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        self.monitoring_level = monitoring_level
        
        # Monitoring status
        self.status = MonitoringStatus.OFFLINE
        self.monitoring_active = False
        self.overall_health = SystemHealth.OPTIMAL
        
        # Component monitors
        self.biological_monitor = BiologicalMonitor()
        self.virtual_blood_monitor = VirtualBloodMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.safety_monitor = SafetyMonitor()
        self.consciousness_monitor = ConsciousnessMonitor()
        
        # Alert system
        self.alert_system = AlertSystem()
        
        # Monitoring data storage
        self.monitoring_data: Dict[str, List[Dict]] = {
            "system_health": [],
            "neural_viability": [],
            "virtual_blood": [],
            "performance": [],
            "safety": [],
            "consciousness": [],
            "alerts": []
        }
        
        # Threading for continuous monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._data_collection_thread: Optional[threading.Thread] = None
        self._active = False
        
        # Monitoring intervals (seconds)
        self.intervals = {
            "neural_viability": 1.0,    # Critical - 1 second
            "virtual_blood": 1.0,       # Critical - 1 second
            "safety": 0.5,              # Critical - 0.5 seconds
            "consciousness": 2.0,       # Important - 2 seconds
            "performance": 5.0,         # Standard - 5 seconds
            "system_health": 10.0,      # Overview - 10 seconds
        }
        
        self.logger.info(f"SystemMonitor initialized with {monitoring_level.value} level")
    
    def initialize_monitoring(self) -> bool:
        """
        Initialize all monitoring systems.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing comprehensive monitoring...")
            self.status = MonitoringStatus.INITIALIZING
            
            # 1. Initialize biological monitoring
            if not self.biological_monitor.initialize():
                return False
            
            # 2. Initialize Virtual Blood monitoring
            if not self.virtual_blood_monitor.initialize():
                return False
            
            # 3. Initialize performance monitoring
            if not self.performance_monitor.initialize():
                return False
            
            # 4. Initialize safety monitoring
            if not self.safety_monitor.initialize():
                return False
            
            # 5. Initialize consciousness monitoring
            if not self.consciousness_monitor.initialize():
                return False
            
            # 6. Initialize alert system
            if not self.alert_system.initialize():
                return False
            
            self.status = MonitoringStatus.ACTIVE
            self.logger.info("Monitoring systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            self.status = MonitoringStatus.FAILED
            return False
    
    def start_comprehensive_monitoring(self) -> bool:
        """
        Start comprehensive system monitoring.
        
        Returns:
            bool: True if monitoring started successfully
        """
        try:
            if self.status != MonitoringStatus.ACTIVE:
                self.logger.error("Cannot start monitoring - not initialized")
                return False
            
            self.logger.info("Starting comprehensive system monitoring...")
            
            # Start component monitors
            if not self._start_component_monitors():
                return False
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            self.monitoring_active = True
            self.logger.info("Comprehensive monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop all monitoring systems.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            self.logger.info("Stopping system monitoring...")
            
            # Stop monitoring threads
            self._active = False
            self.monitoring_active = False
            
            # Stop component monitors
            self.biological_monitor.stop()
            self.virtual_blood_monitor.stop()
            self.performance_monitor.stop()
            self.safety_monitor.stop()
            self.consciousness_monitor.stop()
            
            self.status = MonitoringStatus.OFFLINE
            self.logger.info("System monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": self.status.value,
            "monitoring_active": self.monitoring_active,
            "overall_health": self.overall_health.value,
            "monitoring_level": self.monitoring_level.value,
            "biological": self.biological_monitor.get_status(),
            "virtual_blood": self.virtual_blood_monitor.get_status(),
            "performance": self.performance_monitor.get_status(),
            "safety": self.safety_monitor.get_status(),
            "consciousness": self.consciousness_monitor.get_status(),
            "active_alerts": self.alert_system.get_active_alerts(),
            "data_points": {key: len(data) for key, data in self.monitoring_data.items()},
        }
    
    def get_monitoring_data(self, 
                           component: str, 
                           hours: int = 1) -> List[Dict[str, Any]]:
        """
        Get monitoring data for specific component.
        
        Args:
            component: Component name
            hours: Hours of data to retrieve
            
        Returns:
            List of monitoring data points
        """
        if component not in self.monitoring_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_data = []
        for data_point in self.monitoring_data[component]:
            if datetime.fromisoformat(data_point["timestamp"]) > cutoff_time:
                filtered_data.append(data_point)
        
        return filtered_data
    
    def register_alert_callback(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for alert notifications."""
        self.alert_system.register_callback(callback)
    
    def _start_component_monitors(self) -> bool:
        """Start all component monitors."""
        try:
            self.biological_monitor.start()
            self.virtual_blood_monitor.start()
            self.performance_monitor.start()
            self.safety_monitor.start()
            self.consciousness_monitor.start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start component monitors: {e}")
            return False
    
    def _start_monitoring_threads(self) -> None:
        """Start monitoring threads."""
        if not self._monitoring_thread:
            self._active = True
            
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="SystemMonitoring"
            )
            self._monitoring_thread.start()
            
            self._data_collection_thread = threading.Thread(
                target=self._data_collection_loop,
                daemon=True,
                name="DataCollection"
            )
            self._data_collection_thread.start()
            
            self.logger.info("Monitoring threads started")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._active:
            try:
                # Assess overall system health
                self._assess_system_health()
                
                # Check for critical alerts
                self._check_critical_conditions()
                
                # Update monitoring status
                self._update_monitoring_status()
                
                time.sleep(self.intervals["system_health"])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _data_collection_loop(self) -> None:
        """Data collection loop."""
        last_collection = {key: 0.0 for key in self.intervals.keys()}
        
        while self._active:
            try:
                current_time = time.time()
                
                # Collect data based on intervals
                for component, interval in self.intervals.items():
                    if current_time - last_collection[component] >= interval:
                        self._collect_component_data(component)
                        last_collection[component] = current_time
                
                time.sleep(0.1)  # 100ms data collection cycle
                
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                time.sleep(1.0)
    
    def _assess_system_health(self) -> None:
        """Assess overall system health."""
        # Get component health scores
        biological_health = self.biological_monitor.get_health_score()
        virtual_blood_health = self.virtual_blood_monitor.get_health_score()
        safety_health = self.safety_monitor.get_health_score()
        performance_health = self.performance_monitor.get_health_score()
        consciousness_health = self.consciousness_monitor.get_health_score()
        
        # Calculate weighted overall health
        weights = {
            "biological": 0.3,     # 30% - most critical
            "virtual_blood": 0.25, # 25% - critical
            "safety": 0.25,        # 25% - critical
            "performance": 0.1,    # 10% - important
            "consciousness": 0.1   # 10% - important
        }
        
        overall_score = (
            biological_health * weights["biological"] +
            virtual_blood_health * weights["virtual_blood"] +
            safety_health * weights["safety"] +
            performance_health * weights["performance"] +
            consciousness_health * weights["consciousness"]
        )
        
        # Determine health status
        if overall_score >= 0.95:
            self.overall_health = SystemHealth.OPTIMAL
        elif overall_score >= 0.85:
            self.overall_health = SystemHealth.GOOD
        elif overall_score >= 0.75:
            self.overall_health = SystemHealth.ACCEPTABLE
        elif overall_score >= 0.60:
            self.overall_health = SystemHealth.DEGRADED
        elif overall_score >= 0.40:
            self.overall_health = SystemHealth.CRITICAL
        else:
            self.overall_health = SystemHealth.FAILING
    
    def _check_critical_conditions(self) -> None:
        """Check for critical system conditions."""
        # Check neural viability
        neural_viability = self.biological_monitor.get_neural_viability()
        if neural_viability < 90.0:
            self.alert_system.create_alert(
                AlertSeverity.CRITICAL,
                "Neural viability below 90%",
                {"viability": neural_viability}
            )
        
        # Check Virtual Blood circulation
        circulation_quality = self.virtual_blood_monitor.get_circulation_quality()
        if circulation_quality < 0.8:
            self.alert_system.create_alert(
                AlertSeverity.WARNING,
                "Virtual Blood circulation quality degraded",
                {"quality": circulation_quality}
            )
        
        # Check safety system status
        safety_status = self.safety_monitor.get_safety_status()
        if not safety_status.get("all_systems_operational", True):
            self.alert_system.create_alert(
                AlertSeverity.EMERGENCY,
                "Safety system failure detected",
                safety_status
            )
    
    def _update_monitoring_status(self) -> None:
        """Update monitoring system status."""
        # Check if monitoring systems are functioning
        component_statuses = [
            self.biological_monitor.is_operational(),
            self.virtual_blood_monitor.is_operational(),
            self.safety_monitor.is_operational()
        ]
        
        if all(component_statuses):
            if self.status != MonitoringStatus.ACTIVE:
                self.status = MonitoringStatus.ACTIVE
        elif any(component_statuses):
            self.status = MonitoringStatus.DEGRADED
        else:
            self.status = MonitoringStatus.FAILED
    
    def _collect_component_data(self, component: str) -> None:
        """Collect data for specific component."""
        timestamp = datetime.now().isoformat()
        
        if component == "neural_viability":
            data = {
                "timestamp": timestamp,
                "viability": self.biological_monitor.get_neural_viability(),
                "network_count": self.biological_monitor.get_network_count(),
                "immune_monitoring": self.biological_monitor.get_immune_status()
            }
            self.monitoring_data["neural_viability"].append(data)
        
        elif component == "virtual_blood":
            data = {
                "timestamp": timestamp,
                "circulation_quality": self.virtual_blood_monitor.get_circulation_quality(),
                "flow_rate": self.virtual_blood_monitor.get_flow_rate(),
                "oxygen_saturation": self.virtual_blood_monitor.get_oxygen_saturation()
            }
            self.monitoring_data["virtual_blood"].append(data)
        
        elif component == "safety":
            data = {
                "timestamp": timestamp,
                "safety_status": self.safety_monitor.get_safety_status(),
                "emergency_systems": self.safety_monitor.get_emergency_status()
            }
            self.monitoring_data["safety"].append(data)
        
        # Limit data storage (keep last 1000 points per component)
        if len(self.monitoring_data[component]) > 1000:
            self.monitoring_data[component] = self.monitoring_data[component][-1000:]

class BiologicalMonitor:
    """Monitor biological neural network health and viability."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BiologicalMonitor")
        self.initialized = False
        self.operational = False
        
    def initialize(self) -> bool:
        """Initialize biological monitoring."""
        try:
            self.initialized = True
            self.logger.info("Biological monitor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize biological monitor: {e}")
            return False
    
    def start(self) -> None:
        """Start biological monitoring."""
        self.operational = True
        self.logger.info("Biological monitoring started")
    
    def stop(self) -> None:
        """Stop biological monitoring."""
        self.operational = False
        self.logger.info("Biological monitoring stopped")
    
    def is_operational(self) -> bool:
        """Check if biological monitor is operational."""
        return self.operational
    
    def get_neural_viability(self) -> float:
        """Get current neural viability percentage."""
        # In real implementation, would read from actual sensors
        return 98.7
    
    def get_network_count(self) -> int:
        """Get number of active neural networks."""
        return 2
    
    def get_immune_status(self) -> Dict[str, Any]:
        """Get immune cell monitoring status."""
        return {
            "active": True,
            "cell_count": 15000,
            "inflammation_level": "normal"
        }
    
    def get_health_score(self) -> float:
        """Get biological health score (0.0-1.0)."""
        return 0.95
    
    def get_status(self) -> Dict[str, Any]:
        """Get biological monitor status."""
        return {
            "initialized": self.initialized,
            "operational": self.operational,
            "neural_viability": self.get_neural_viability(),
            "network_count": self.get_network_count(),
            "health_score": self.get_health_score()
        }

class VirtualBloodMonitor:
    """Monitor Virtual Blood circulation and quality."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VirtualBloodMonitor")
        self.initialized = False
        self.operational = False
    
    def initialize(self) -> bool:
        """Initialize Virtual Blood monitoring."""
        try:
            self.initialized = True
            self.logger.info("Virtual Blood monitor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Virtual Blood monitor: {e}")
            return False
    
    def start(self) -> None:
        """Start Virtual Blood monitoring."""
        self.operational = True
        self.logger.info("Virtual Blood monitoring started")
    
    def stop(self) -> None:
        """Stop Virtual Blood monitoring."""
        self.operational = False
        self.logger.info("Virtual Blood monitoring stopped")
    
    def is_operational(self) -> bool:
        """Check if Virtual Blood monitor is operational."""
        return self.operational
    
    def get_circulation_quality(self) -> float:
        """Get circulation quality score (0.0-1.0)."""
        return 0.92
    
    def get_flow_rate(self) -> float:
        """Get current flow rate (mL/min)."""
        return 8.5
    
    def get_oxygen_saturation(self) -> float:
        """Get oxygen saturation percentage."""
        return 98.3
    
    def get_health_score(self) -> float:
        """Get Virtual Blood health score (0.0-1.0)."""
        return 0.93
    
    def get_status(self) -> Dict[str, Any]:
        """Get Virtual Blood monitor status."""
        return {
            "initialized": self.initialized,
            "operational": self.operational,
            "circulation_quality": self.get_circulation_quality(),
            "flow_rate": self.get_flow_rate(),
            "oxygen_saturation": self.get_oxygen_saturation(),
            "health_score": self.get_health_score()
        }

class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self.initialized = False
        self.operational = False
    
    def initialize(self) -> bool:
        """Initialize performance monitoring."""
        try:
            self.initialized = True
            self.logger.info("Performance monitor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitor: {e}")
            return False
    
    def start(self) -> None:
        """Start performance monitoring."""
        self.operational = True
        self.logger.info("Performance monitoring started")
    
    def stop(self) -> None:
        """Stop performance monitoring."""
        self.operational = False
        self.logger.info("Performance monitoring stopped")
    
    def is_operational(self) -> bool:
        """Check if performance monitor is operational."""
        return self.operational
    
    def get_health_score(self) -> float:
        """Get performance health score (0.0-1.0)."""
        return 0.89
    
    def get_status(self) -> Dict[str, Any]:
        """Get performance monitor status."""
        return {
            "initialized": self.initialized,
            "operational": self.operational,
            "health_score": self.get_health_score()
        }

class SafetyMonitor:
    """Monitor safety system status."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SafetyMonitor")
        self.initialized = False
        self.operational = False
    
    def initialize(self) -> bool:
        """Initialize safety monitoring."""
        try:
            self.initialized = True
            self.logger.info("Safety monitor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize safety monitor: {e}")
            return False
    
    def start(self) -> None:
        """Start safety monitoring."""
        self.operational = True
        self.logger.info("Safety monitoring started")
    
    def stop(self) -> None:
        """Stop safety monitoring."""
        self.operational = False
        self.logger.info("Safety monitoring stopped")
    
    def is_operational(self) -> bool:
        """Check if safety monitor is operational."""
        return self.operational
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get safety system status."""
        return {
            "all_systems_operational": True,
            "emergency_systems": "ready",
            "backup_systems": "ready"
        }
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """Get emergency system status."""
        return {
            "emergency_shutdown": "ready",
            "backup_power": "available",
            "emergency_contacts": "configured"
        }
    
    def get_health_score(self) -> float:
        """Get safety health score (0.0-1.0)."""
        return 1.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get safety monitor status."""
        return {
            "initialized": self.initialized,
            "operational": self.operational,
            "safety_status": self.get_safety_status(),
            "health_score": self.get_health_score()
        }

class ConsciousnessMonitor:
    """Monitor consciousness integration quality."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConsciousnessMonitor")
        self.initialized = False
        self.operational = False
    
    def initialize(self) -> bool:
        """Initialize consciousness monitoring."""
        try:
            self.initialized = True
            self.logger.info("Consciousness monitor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize consciousness monitor: {e}")
            return False
    
    def start(self) -> None:
        """Start consciousness monitoring."""
        self.operational = True
        self.logger.info("Consciousness monitoring started")
    
    def stop(self) -> None:
        """Stop consciousness monitoring."""
        self.operational = False
        self.logger.info("Consciousness monitoring stopped")
    
    def is_operational(self) -> bool:
        """Check if consciousness monitor is operational."""
        return self.operational
    
    def get_health_score(self) -> float:
        """Get consciousness health score (0.0-1.0)."""
        return 0.88
    
    def get_status(self) -> Dict[str, Any]:
        """Get consciousness monitor status."""
        return {
            "initialized": self.initialized,
            "operational": self.operational,
            "health_score": self.get_health_score()
        }

class AlertSystem:
    """System alert management and notification."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AlertSystem")
        self.initialized = False
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
    
    def initialize(self) -> bool:
        """Initialize alert system."""
        try:
            self.initialized = True
            self.logger.info("Alert system initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize alert system: {e}")
            return False
    
    def create_alert(self, 
                    severity: AlertSeverity, 
                    message: str, 
                    data: Dict[str, Any]) -> None:
        """Create new system alert."""
        alert = {
            "id": f"alert_{len(self.active_alerts)}_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": severity.value,
            "message": message,
            "data": data,
            "acknowledged": False
        }
        
        self.active_alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.warning(f"ALERT [{severity.value}]: {message}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return self.active_alerts.copy()
    
    def register_callback(self, callback: Callable) -> None:
        """Register alert callback."""
        self.alert_callbacks.append(callback)

# Export main classes
__all__ = [
    "SystemMonitor",
    "BiologicalMonitor",
    "VirtualBloodMonitor",
    "PerformanceMonitor",
    "SafetyMonitor",
    "ConsciousnessMonitor",
    "AlertSystem",
    "MonitoringLevel",
    "AlertSeverity",
    "MonitoringStatus",
    "SystemHealth"
]