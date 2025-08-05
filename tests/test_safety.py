"""
Test suite for Jungfernstieg safety systems.

These tests validate the critical safety protocols that protect living biological
neural tissue. All safety tests must pass before any biological operations.

CRITICAL SAFETY TESTING:
- Safety protocol initialization
- Emergency shutdown procedures
- Neural viability monitoring
- Safety validation processes
- Emergency response systems
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from jungfernstieg.safety import (
    SafetyManager,
    SafetyLevel,
    SystemStatus,
    NeuralViabilityStatus,
    initialize_all_protocols,
    validate_system_safety,
    is_safe_for_biological_operations
)


class TestSafetyManager:
    """Test suite for SafetyManager class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.safety_manager = SafetyManager()
    
    def test_safety_manager_initialization(self):
        """Test SafetyManager initialization."""
        assert self.safety_manager.status == SystemStatus.OFFLINE
        assert self.safety_manager.safety_level == SafetyLevel.UNDEFINED
        assert self.safety_manager.neural_viability == 0.0
        assert self.safety_manager.neural_status == NeuralViabilityStatus.UNKNOWN
        assert not self.safety_manager._monitoring_active
    
    def test_safety_protocol_initialization(self):
        """Test safety protocol initialization."""
        # This should initialize all safety protocols
        result = self.safety_manager.initialize_safety_protocols()
        
        # Should return True for successful initialization
        assert result is True
        assert self.safety_manager.status == SystemStatus.SAFE
        assert self.safety_manager.safety_level == SafetyLevel.BSL_2_PLUS
    
    def test_safety_validation(self):
        """Test safety system validation."""
        # Initialize protocols first
        self.safety_manager.initialize_safety_protocols()
        
        # Validate safety systems
        result = self.safety_manager.validate_system_safety()
        
        assert result is True
    
    def test_neural_viability_monitoring(self):
        """Test neural viability monitoring and thresholds."""
        # Initialize safety systems
        self.safety_manager.initialize_safety_protocols()
        
        # Test normal viability
        self.safety_manager.update_neural_viability(98.5)
        assert self.safety_manager.neural_status == NeuralViabilityStatus.EXCELLENT
        
        # Test good viability
        self.safety_manager.update_neural_viability(96.2)
        assert self.safety_manager.neural_status == NeuralViabilityStatus.GOOD
        
        # Test acceptable viability
        self.safety_manager.update_neural_viability(92.1)
        assert self.safety_manager.neural_status == NeuralViabilityStatus.ACCEPTABLE
        
        # Test warning viability
        self.safety_manager.update_neural_viability(87.3)
        assert self.safety_manager.neural_status == NeuralViabilityStatus.WARNING
        
        # Test critical viability
        self.safety_manager.update_neural_viability(82.8)
        assert self.safety_manager.neural_status == NeuralViabilityStatus.CRITICAL
        
        # Test failing viability
        self.safety_manager.update_neural_viability(75.2)
        assert self.safety_manager.neural_status == NeuralViabilityStatus.FAILING
    
    def test_emergency_shutdown_trigger(self):
        """Test emergency shutdown triggering."""
        # Initialize safety systems
        self.safety_manager.initialize_safety_protocols()
        
        # Register emergency callback to verify it's called
        callback_called = threading.Event()
        
        def emergency_callback():
            callback_called.set()
        
        self.safety_manager.register_emergency_callback(emergency_callback)
        
        # Trigger emergency shutdown
        self.safety_manager.trigger_emergency_shutdown("Test emergency")
        
        # Verify emergency state
        assert self.safety_manager.status == SystemStatus.EMERGENCY_SHUTDOWN
        
        # Verify callback was called
        assert callback_called.wait(timeout=1.0)
    
    def test_critical_viability_threshold(self):
        """Test critical neural viability threshold handling."""
        # Initialize safety systems
        self.safety_manager.initialize_safety_protocols()
        
        # Test viability above threshold (should be OK)
        self.safety_manager.update_neural_viability(96.0)
        assert self.safety_manager.status != SystemStatus.EMERGENCY_SHUTDOWN
        
        # Test viability at threshold (should be OK)
        self.safety_manager.update_neural_viability(95.0)
        assert self.safety_manager.status != SystemStatus.EMERGENCY_SHUTDOWN
        
        # Test viability below threshold (should trigger emergency for very low values)
        self.safety_manager.update_neural_viability(75.0)
        assert self.safety_manager.status == SystemStatus.EMERGENCY_SHUTDOWN
    
    def test_safety_event_logging(self):
        """Test safety event logging."""
        # Initialize safety systems
        self.safety_manager.initialize_safety_protocols()
        
        # Get initial log count
        initial_log_count = len(self.safety_manager._safety_log)
        
        # Update neural viability to trigger logging
        self.safety_manager.update_neural_viability(98.0)
        self.safety_manager.update_neural_viability(95.0)  # Significant change
        
        # Verify logging occurred
        assert len(self.safety_manager._safety_log) > initial_log_count
    
    def test_safety_status_report(self):
        """Test comprehensive safety status reporting."""
        # Initialize safety systems
        self.safety_manager.initialize_safety_protocols()
        
        # Update some status
        self.safety_manager.update_neural_viability(97.5)
        
        # Get safety status
        status = self.safety_manager.get_safety_status()
        
        # Verify status structure
        assert "timestamp" in status
        assert "system_status" in status
        assert "safety_level" in status
        assert "neural_viability" in status
        assert "neural_status" in status
        assert "safety_initialized" in status
        assert "safety_validated" in status
        assert "component_status" in status
        
        # Verify values
        assert status["neural_viability"] == 97.5
        assert status["safety_initialized"] is True


class TestGlobalSafetyFunctions:
    """Test suite for global safety functions."""
    
    def test_initialize_all_protocols(self):
        """Test global protocol initialization."""
        result = initialize_all_protocols()
        assert result is True
    
    def test_validate_system_safety(self):
        """Test global safety validation."""
        # Initialize first
        initialize_all_protocols()
        
        # Then validate
        result = validate_system_safety()
        assert result is True
    
    def test_is_safe_for_biological_operations(self):
        """Test biological operations safety check."""
        # Should be False initially
        assert is_safe_for_biological_operations() is False
        
        # Initialize and validate
        initialize_all_protocols()
        validate_system_safety()
        
        # Should be True after initialization and validation
        assert is_safe_for_biological_operations() is True


class TestSafetyIntegration:
    """Integration tests for safety systems."""
    
    def test_complete_safety_workflow(self):
        """Test complete safety initialization and validation workflow."""
        # Step 1: Initialize protocols
        init_result = initialize_all_protocols()
        assert init_result is True
        
        # Step 2: Validate safety
        validation_result = validate_system_safety()
        assert validation_result is True
        
        # Step 3: Check biological operations safety
        bio_safe = is_safe_for_biological_operations()
        assert bio_safe is True
        
        # Step 4: Create safety manager and verify it works
        safety_manager = SafetyManager()
        status = safety_manager.get_safety_status()
        assert status["safety_initialized"] is True
        assert status["safety_validated"] is True
    
    def test_safety_system_resilience(self):
        """Test safety system resilience under various conditions."""
        # Initialize safety
        initialize_all_protocols()
        validate_system_safety()
        
        safety_manager = SafetyManager()
        safety_manager.initialize_safety_protocols()
        
        # Test multiple rapid viability updates
        viabilities = [98.5, 97.2, 96.8, 98.1, 97.5]
        for viability in viabilities:
            safety_manager.update_neural_viability(viability)
            assert safety_manager.neural_viability == viability
        
        # Verify system remains stable
        assert safety_manager.status != SystemStatus.EMERGENCY_SHUTDOWN
    
    def test_safety_monitoring_thread(self):
        """Test safety monitoring thread operation."""
        safety_manager = SafetyManager()
        safety_manager.initialize_safety_protocols()
        
        # Verify monitoring thread is started
        assert safety_manager._monitoring_active is True
        assert safety_manager._monitoring_thread is not None
        
        # Wait a short time to let monitoring run
        time.sleep(0.1)
        
        # Verify thread is running
        assert safety_manager._monitoring_thread.is_alive()


@pytest.mark.safety
class TestCriticalSafetyScenarios:
    """Critical safety scenario tests - these MUST pass."""
    
    def test_emergency_response_time(self):
        """Test emergency response time is adequate."""
        safety_manager = SafetyManager()
        safety_manager.initialize_safety_protocols()
        
        # Measure emergency response time
        start_time = time.time()
        safety_manager.trigger_emergency_shutdown("Response time test")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Emergency response must be under 100ms
        assert response_time < 0.1, f"Emergency response too slow: {response_time:.3f}s"
    
    def test_neural_viability_critical_threshold(self):
        """Test critical neural viability threshold handling."""
        safety_manager = SafetyManager()
        safety_manager.initialize_safety_protocols()
        
        # Test exactly at critical threshold
        safety_manager.update_neural_viability(80.0)
        # Should not trigger emergency at exactly 80%
        assert safety_manager.status != SystemStatus.EMERGENCY_SHUTDOWN
        
        # Test below critical threshold
        safety_manager.update_neural_viability(79.9)
        # Should trigger emergency below 80%
        assert safety_manager.status == SystemStatus.EMERGENCY_SHUTDOWN
    
    def test_multiple_emergency_callbacks(self):
        """Test multiple emergency callbacks are executed."""
        safety_manager = SafetyManager()
        safety_manager.initialize_safety_protocols()
        
        # Register multiple callbacks
        callback_results = []
        
        def callback1():
            callback_results.append("callback1")
        
        def callback2():
            callback_results.append("callback2")
        
        def callback3():
            callback_results.append("callback3")
        
        safety_manager.register_emergency_callback(callback1)
        safety_manager.register_emergency_callback(callback2)
        safety_manager.register_emergency_callback(callback3)
        
        # Trigger emergency
        safety_manager.trigger_emergency_shutdown("Multi-callback test")
        
        # Verify all callbacks were executed
        assert "callback1" in callback_results
        assert "callback2" in callback_results
        assert "callback3" in callback_results
        assert len(callback_results) == 3
    
    def test_safety_system_failure_detection(self):
        """Test detection of safety system failures."""
        safety_manager = SafetyManager()
        
        # Test validation without initialization
        result = safety_manager.validate_system_safety()
        assert result is False
        
        # Test that biological operations are not safe
        assert not is_safe_for_biological_operations()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])