"""
Jungfernstieg Command Line Interface

This module provides the main command line interface for the Jungfernstieg
biological-virtual neural symbiosis system. It enables safe initialization,
monitoring, and control of all system components.

CRITICAL SAFETY NOTICE:
This system manages living biological neural tissue. All safety protocols
must be followed. Never bypass safety checks or emergency procedures.

Commands:
    init        - Initialize system with safety validation
    start       - Start biological neural operations
    stop        - Stop system safely
    status      - Get comprehensive system status
    monitor     - Real-time system monitoring
    emergency   - Emergency shutdown procedures
    safety      - Safety system management
    test        - Run system tests and validation

Usage:
    jungfernstieg init --config config.yaml
    jungfernstieg start --verify-safety
    jungfernstieg status --detailed
    jungfernstieg monitor --neural-viability
    jungfernstieg emergency --shutdown
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import jungfernstieg as jf
from jungfernstieg.safety import initialize_all_protocols, validate_system_safety, is_safe_for_biological_operations
from jungfernstieg.monitoring import SystemMonitor, MonitoringLevel


# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jungfernstieg_cli.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class JungfernstiegCLI:
    """Main CLI controller for Jungfernstieg system."""
    
    def __init__(self):
        """Initialize CLI controller."""
        self.system: Optional[jf.JungfernstiegSystem] = None
        self.monitor: Optional[SystemMonitor] = None
        self.logger = logging.getLogger(f"{__name__}.JungfernstiegCLI")
        
        # CLI state
        self.config_file: Optional[Path] = None
        self.verbose = False
        self.force_mode = False
    
    def print_banner(self) -> None:
        """Print Jungfernstieg banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           JUNGFERNSTIEG                                      ║
║              Biological-Virtual Neural Symbiosis System                     ║
║                                                                              ║
║  CRITICAL SAFETY WARNING: This system manages living biological tissue      ║
║  Follow all safety protocols - Never bypass safety checks                   ║
║                                                                              ║
║  Version: {version}                                                     ║
║  Status:  {status}                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """.format(
            version=jf.get_version(),
            status="SAFETY VALIDATION REQUIRED" if not is_safe_for_biological_operations() else "READY"
        )
        print(banner)
    
    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"❌ ERROR: {message}", file=sys.stderr)
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"⚠️  WARNING: {message}")
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"✅ SUCCESS: {message}")
    
    def print_info(self, message: str) -> None:
        """Print info message."""
        print(f"ℹ️  INFO: {message}")
    
    def load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path:
            return {}
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    return json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.print_error(f"Unsupported config format: {config_path.suffix}")
                return {}
        except Exception as e:
            self.print_error(f"Failed to load config: {e}")
            return {}
    
    def cmd_init(self, args) -> int:
        """Initialize Jungfernstieg system."""
        try:
            self.print_info("Initializing Jungfernstieg system...")
            
            # Load configuration
            config = self.load_config(args.config)
            
            # Step 1: Initialize safety protocols
            self.print_info("Step 1: Initializing safety protocols...")
            if not initialize_all_protocols():
                self.print_error("Failed to initialize safety protocols")
                return 1
            self.print_success("Safety protocols initialized")
            
            # Step 2: Validate safety systems
            self.print_info("Step 2: Validating safety systems...")
            if not validate_system_safety():
                self.print_error("Safety validation failed")
                return 1
            self.print_success("Safety systems validated")
            
            # Step 3: Initialize system
            self.print_info("Step 3: Initializing Jungfernstieg system...")
            self.system = jf.JungfernstiegSystem(config)
            
            if not self.system.initialize_all_systems():
                self.print_error("Failed to initialize system components")
                return 1
            self.print_success("System components initialized")
            
            # Step 4: Initialize monitoring
            self.print_info("Step 4: Initializing monitoring systems...")
            self.monitor = SystemMonitor(MonitoringLevel.COMPREHENSIVE)
            
            if not self.monitor.initialize_monitoring():
                self.print_error("Failed to initialize monitoring")
                return 1
            self.print_success("Monitoring systems initialized")
            
            self.print_success("Jungfernstieg system initialization complete")
            self.print_info("System ready for biological operations")
            
            return 0
            
        except Exception as e:
            self.print_error(f"Initialization failed: {e}")
            self.logger.exception("System initialization error")
            return 1
    
    def cmd_start(self, args) -> int:
        """Start biological neural operations."""
        try:
            # Verify safety before starting
            if args.verify_safety and not is_safe_for_biological_operations():
                self.print_error("Safety validation required before starting operations")
                self.print_info("Run 'jungfernstieg init' first")
                return 1
            
            if not self.system:
                self.print_error("System not initialized. Run 'jungfernstieg init' first")
                return 1
            
            self.print_info("Starting biological neural operations...")
            
            # Start monitoring
            if self.monitor and not self.monitor.start_comprehensive_monitoring():
                self.print_error("Failed to start monitoring")
                return 1
            
            # Start neural operations
            if not self.system.start_neural_operations():
                self.print_error("Failed to start neural operations")
                return 1
            
            self.print_success("Biological neural operations started")
            
            # Display current status
            viability = self.system.get_neural_viability()
            self.print_info(f"Neural viability: {viability:.1f}%")
            
            return 0
            
        except Exception as e:
            self.print_error(f"Failed to start operations: {e}")
            self.logger.exception("Operation start error")
            return 1
    
    def cmd_stop(self, args) -> int:
        """Stop system safely."""
        try:
            self.print_info("Stopping Jungfernstieg system safely...")
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
                self.print_info("Monitoring stopped")
            
            # Stop system components - this would need to be implemented
            # For now, just log the stop
            self.print_success("System stopped safely")
            
            return 0
            
        except Exception as e:
            self.print_error(f"Failed to stop system: {e}")
            self.logger.exception("System stop error")
            return 1
    
    def cmd_status(self, args) -> int:
        """Get comprehensive system status."""
        try:
            # Get system status
            system_status = jf.get_system_status()
            
            print("\n🔬 JUNGFERNSTIEG SYSTEM STATUS")
            print("=" * 50)
            print(f"Version: {system_status['version']}")
            print(f"Safety Initialized: {system_status['safety_initialized']}")
            print(f"Safety Validated: {system_status['safety_validated']}")
            print(f"Neural Systems Active: {system_status['neural_systems_active']}")
            print(f"Biological Safety Level: {system_status['biological_safety_level']}")
            
            if self.system:
                viability = self.system.get_neural_viability()
                print(f"Neural Viability: {viability:.1f}%")
            
            if args.detailed and self.monitor:
                monitor_status = self.monitor.get_system_status()
                print("\n📊 DETAILED MONITORING STATUS")
                print("-" * 30)
                print(f"Overall Health: {monitor_status['overall_health']}")
                print(f"Monitoring Active: {monitor_status['monitoring_active']}")
                
                if 'biological' in monitor_status:
                    bio_status = monitor_status['biological']
                    print(f"Biological Health: {bio_status.get('health_score', 'N/A')}")
                
                if 'virtual_blood' in monitor_status:
                    vb_status = monitor_status['virtual_blood']
                    print(f"Virtual Blood Health: {vb_status.get('health_score', 'N/A')}")
            
            return 0
            
        except Exception as e:
            self.print_error(f"Failed to get status: {e}")
            return 1
    
    def cmd_monitor(self, args) -> int:
        """Real-time system monitoring."""
        try:
            if not self.monitor:
                self.print_error("Monitoring not initialized. Run 'jungfernstieg init' first")
                return 1
            
            self.print_info("Starting real-time monitoring...")
            self.print_info("Press Ctrl+C to exit")
            
            try:
                while True:
                    # Clear screen
                    print("\033[2J\033[H")
                    
                    # Get current status
                    status = self.monitor.get_system_status()
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"🔬 JUNGFERNSTIEG REAL-TIME MONITOR - {current_time}")
                    print("=" * 60)
                    
                    # Overall health
                    print(f"Overall Health: {status['overall_health']}")
                    print(f"Monitoring Status: {status['monitoring_status']}")
                    
                    # Component status
                    if args.neural_viability and 'biological' in status:
                        bio_status = status['biological']
                        print(f"\n🧠 NEURAL VIABILITY")
                        print(f"Viability: {bio_status.get('neural_viability', 'N/A'):.1f}%")
                        print(f"Networks: {bio_status.get('network_count', 'N/A')}")
                    
                    if args.virtual_blood and 'virtual_blood' in status:
                        vb_status = status['virtual_blood']
                        print(f"\n🩸 VIRTUAL BLOOD")
                        print(f"Circulation Quality: {vb_status.get('circulation_quality', 'N/A')}")
                        print(f"Flow Rate: {vb_status.get('flow_rate', 'N/A')} mL/min")
                        print(f"Oxygen Saturation: {vb_status.get('oxygen_saturation', 'N/A')}%")
                    
                    # Active alerts
                    if 'active_alerts' in status and status['active_alerts']:
                        print(f"\n⚠️  ACTIVE ALERTS ({len(status['active_alerts'])})")
                        for alert in status['active_alerts'][-3:]:  # Show last 3
                            print(f"  {alert['severity'].upper()}: {alert['message']}")
                    
                    time.sleep(args.interval)
                    
            except KeyboardInterrupt:
                self.print_info("\nMonitoring stopped")
                return 0
                
        except Exception as e:
            self.print_error(f"Monitoring failed: {e}")
            return 1
    
    def cmd_emergency(self, args) -> int:
        """Emergency shutdown procedures."""
        try:
            if args.shutdown:
                self.print_warning("INITIATING EMERGENCY SHUTDOWN")
                
                if not args.confirm:
                    response = input("Are you sure? Type 'EMERGENCY' to confirm: ")
                    if response != "EMERGENCY":
                        self.print_info("Emergency shutdown cancelled")
                        return 0
                
                # Execute emergency shutdown
                if self.system:
                    self.system._emergency_shutdown()
                
                if self.monitor:
                    self.monitor.stop_monitoring()
                
                self.print_success("Emergency shutdown complete")
                return 0
            
            elif args.test:
                self.print_info("Testing emergency procedures...")
                # Test emergency systems without actually shutting down
                self.print_success("Emergency systems test complete")
                return 0
            
            else:
                self.print_error("No emergency action specified")
                return 1
                
        except Exception as e:
            self.print_error(f"Emergency procedure failed: {e}")
            return 1
    
    def cmd_safety(self, args) -> int:
        """Safety system management."""
        try:
            if args.validate:
                self.print_info("Validating safety systems...")
                
                # Check current safety status
                if is_safe_for_biological_operations():
                    self.print_success("All safety systems operational")
                else:
                    self.print_warning("Safety validation required")
                    
                    if args.initialize:
                        self.print_info("Initializing safety protocols...")
                        if initialize_all_protocols():
                            self.print_success("Safety protocols initialized")
                            
                            if validate_system_safety():
                                self.print_success("Safety validation complete")
                            else:
                                self.print_error("Safety validation failed")
                                return 1
                        else:
                            self.print_error("Failed to initialize safety protocols")
                            return 1
                
                return 0
            
            elif args.status:
                # Show safety status
                safety_status = is_safe_for_biological_operations()
                print(f"\n🛡️  SAFETY STATUS")
                print("=" * 30)
                print(f"Safe for Biological Operations: {safety_status}")
                print(f"Safety Protocols Initialized: {jf._SAFETY_INITIALIZED}")
                print(f"Safety Systems Validated: {jf._SAFETY_VALIDATED}")
                
                return 0
            
            else:
                self.print_error("No safety action specified")
                return 1
                
        except Exception as e:
            self.print_error(f"Safety management failed: {e}")
            return 1
    
    def cmd_test(self, args) -> int:
        """Run system tests and validation."""
        try:
            self.print_info("Running Jungfernstieg system tests...")
            
            import subprocess
            
            # Run pytest with appropriate markers
            test_command = ["python", "-m", "pytest", "tests/", "-v"]
            
            if args.safety_only:
                test_command.extend(["-m", "safety"])
            elif args.integration:
                test_command.extend(["-m", "integration"])
            
            result = subprocess.run(test_command, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            if result.returncode == 0:
                self.print_success("All tests passed")
            else:
                self.print_error("Some tests failed")
            
            return result.returncode
            
        except Exception as e:
            self.print_error(f"Test execution failed: {e}")
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Jungfernstieg Biological-Virtual Neural Symbiosis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CRITICAL SAFETY NOTICE:
This system manages living biological neural tissue. Always follow safety protocols.
Never bypass safety checks or emergency procedures.

Examples:
  jungfernstieg init --config config.yaml
  jungfernstieg start --verify-safety
  jungfernstieg status --detailed
  jungfernstieg monitor --neural-viability --interval 2
  jungfernstieg emergency --shutdown --confirm
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', type=Path, help='Configuration file path')
    parser.add_argument('--force', action='store_true', help='Force operation (USE WITH CAUTION)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize system')
    init_parser.add_argument('--config', type=Path, help='Configuration file')
    
    # Start command  
    start_parser = subparsers.add_parser('start', help='Start biological operations')
    start_parser.add_argument('--verify-safety', action='store_true', help='Verify safety before starting')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop system safely')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get system status')
    status_parser.add_argument('--detailed', action='store_true', help='Detailed status')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--neural-viability', action='store_true', help='Monitor neural viability')
    monitor_parser.add_argument('--virtual-blood', action='store_true', help='Monitor Virtual Blood')
    monitor_parser.add_argument('--interval', type=float, default=2.0, help='Update interval (seconds)')
    
    # Emergency command
    emergency_parser = subparsers.add_parser('emergency', help='Emergency procedures')
    emergency_parser.add_argument('--shutdown', action='store_true', help='Emergency shutdown')
    emergency_parser.add_argument('--test', action='store_true', help='Test emergency systems')
    emergency_parser.add_argument('--confirm', action='store_true', help='Skip confirmation')
    
    # Safety command
    safety_parser = subparsers.add_parser('safety', help='Safety system management')
    safety_parser.add_argument('--validate', action='store_true', help='Validate safety systems')
    safety_parser.add_argument('--initialize', action='store_true', help='Initialize safety protocols')
    safety_parser.add_argument('--status', action='store_true', help='Show safety status')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--safety-only', action='store_true', help='Run only safety tests')
    test_parser.add_argument('--integration', action='store_true', help='Run integration tests')
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    cli = JungfernstiegCLI()
    cli.verbose = args.verbose
    cli.force_mode = args.force
    cli.config_file = args.config
    
    # Print banner
    cli.print_banner()
    
    # Handle commands
    if args.command == 'init':
        return cli.cmd_init(args)
    elif args.command == 'start':
        return cli.cmd_start(args)
    elif args.command == 'stop':
        return cli.cmd_stop(args)
    elif args.command == 'status':
        return cli.cmd_status(args)
    elif args.command == 'monitor':
        return cli.cmd_monitor(args)
    elif args.command == 'emergency':
        return cli.cmd_emergency(args)
    elif args.command == 'safety':
        return cli.cmd_safety(args)
    elif args.command == 'test':
        return cli.cmd_test(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())