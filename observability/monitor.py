"""
Performance monitoring framework for the Row ID Generator system.

Provides comprehensive performance tracking with automatic logging, metrics collection,
system resource monitoring, and operation-specific analysis capabilities.
"""

import time
import threading
import psutil
from contextlib import contextmanager
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import os
import gc

from .logger import StructuredLogger, LogLevel
from .metrics import MetricsCollector, MetricType


@dataclass
class SystemSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_usage_percent: float
    process_memory_mb: float
    process_cpu_percent: float
    thread_count: int
    open_files: int


@dataclass
class OperationProfile:
    """Performance profile for a specific operation."""
    operation_name: str
    total_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_execution: Optional[float] = None
    memory_delta_mb: float = 0.0
    cpu_usage_avg: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Operation tracking with context managers
    - System resource monitoring
    - Memory usage analysis
    - CPU utilization tracking
    - Performance profiling and statistics
    - Integration with logging and metrics systems
    - Automatic threshold detection and alerting
    - Background monitoring capabilities
    """
    
    def __init__(
        self, 
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
        enable_system_monitoring: bool = True,
        monitoring_interval: float = 30.0,
        profile_recent_operations: int = 100
    ):
        """
        Initialize performance monitor.
        
        Args:
            metrics_collector: Optional metrics collector for recording performance data
            logger: Optional structured logger for logging performance events
            enable_system_monitoring: Whether to enable background system monitoring
            monitoring_interval: Interval in seconds for background monitoring
            profile_recent_operations: Number of recent operations to keep for profiling
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.logger = logger or StructuredLogger("performance_monitor")
        self.enable_system_monitoring = enable_system_monitoring
        self.monitoring_interval = monitoring_interval
        
        # Thread-safe storage
        self._lock = threading.RLock()
        
        # Operation tracking
        self._active_operations: Dict[str, List[float]] = defaultdict(list)
        self._operation_profiles: Dict[str, OperationProfile] = {}
        self._recent_operations = deque(maxlen=profile_recent_operations)
        
        # System monitoring
        self._system_snapshots = deque(maxlen=1000)  # Keep last 1000 snapshots
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance thresholds
        self._duration_thresholds: Dict[str, float] = {}
        self._memory_threshold_mb = 100.0  # Default memory increase threshold
        self._cpu_threshold_percent = 80.0  # Default CPU usage threshold
        
        # Initialize system monitoring if enabled
        if self.enable_system_monitoring:
            self._start_system_monitoring()
    
    def track_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for tracking operation performance.
        
        Args:
            operation_name: Name of the operation to track
            tags: Optional tags for categorization
            
        Returns:
            Context manager that automatically tracks performance
        """
        return OperationTracker(self, operation_name, tags)
    
    def start_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start tracking an operation manually.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags for categorization
            
        Returns:
            Unique operation ID for stopping the operation
        """
        operation_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"
        start_time = time.time()
        
        with self._lock:
            self._active_operations[operation_id] = [start_time, operation_name, tags or {}]
        
        # Log operation start
        self.logger.info(f"Started operation: {operation_name}", {
            'operation_type': 'performance_start',
            'operation_name': operation_name,
            'operation_id': operation_id,
            'tags': tags or {}
        })
        
        # Record operation start metric
        self.metrics.increment_counter(f"operations.{operation_name}.started", tags=tags)
        
        return operation_id
    
    def end_operation(
        self, 
        operation_id: str, 
        success: bool = True, 
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        End tracking an operation manually.
        
        Args:
            operation_id: Unique operation ID from start_operation
            success: Whether the operation was successful
            error: Optional exception if operation failed
            metadata: Optional additional metadata about the operation
            
        Returns:
            Performance data for the operation
        """
        end_time = time.time()
        
        with self._lock:
            if operation_id not in self._active_operations:
                self.logger.warning(f"Operation ID not found: {operation_id}")
                return None
            
            start_time, operation_name, tags = self._active_operations.pop(operation_id)
        
        # Calculate performance metrics
        duration = end_time - start_time
        
        # Get system resource usage
        current_snapshot = self._take_system_snapshot()
        
        # Create performance data
        performance_data = {
            'operation_name': operation_name,
            'operation_id': operation_id,
            'duration': duration,
            'success': success,
            'timestamp': end_time,
            'tags': tags,
            'metadata': metadata or {},
            'system_snapshot': current_snapshot.__dict__ if current_snapshot else {}
        }
        
        # Update operation profile
        self._update_operation_profile(operation_name, duration, success, current_snapshot)
        
        # Log performance data
        log_data = {
            'operation_type': 'performance_end',
            'operation_name': operation_name,
            'duration_seconds': duration,
            'success': success,
            'tags': tags,
            'metadata': metadata or {}
        }
        
        if error:
            log_data['error'] = {
                'type': type(error).__name__,
                'message': str(error)
            }
            self.logger.error(f"Operation failed: {operation_name}", log_data)
        else:
            self.logger.info(f"Operation completed: {operation_name}", log_data)
        
        # Record metrics
        self._record_operation_metrics(operation_name, duration, success, tags)
        
        # Check thresholds and alert if needed
        self._check_performance_thresholds(operation_name, duration, current_snapshot)
        
        # Add to recent operations
        self._recent_operations.append(performance_data)
        
        return performance_data
    
    def set_duration_threshold(self, operation_name: str, threshold_seconds: float):
        """
        Set a duration threshold for an operation.
        
        Args:
            operation_name: Name of the operation
            threshold_seconds: Threshold in seconds
        """
        with self._lock:
            self._duration_thresholds[operation_name] = threshold_seconds
        
        self.logger.info(f"Set duration threshold for {operation_name}: {threshold_seconds}s")
    
    def get_operation_profile(self, operation_name: str) -> Optional[OperationProfile]:
        """
        Get performance profile for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            OperationProfile or None if not found
        """
        with self._lock:
            return self._operation_profiles.get(operation_name)
    
    def get_all_operation_profiles(self) -> Dict[str, OperationProfile]:
        """Get performance profiles for all operations."""
        with self._lock:
            return dict(self._operation_profiles)
    
    def get_recent_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent operation performance data.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent operation performance data
        """
        with self._lock:
            recent = list(self._recent_operations)
            return recent[-limit:] if limit else recent
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health metrics.
        
        Returns:
            Dictionary with system health information
        """
        snapshot = self._take_system_snapshot()
        
        with self._lock:
            active_ops = len(self._active_operations)
            total_profiles = len(self._operation_profiles)
        
        return {
            'timestamp': time.time(),
            'system': snapshot.__dict__ if snapshot else {},
            'active_operations': active_ops,
            'total_operation_types': total_profiles,
            'monitoring_enabled': self.enable_system_monitoring,
            'snapshots_collected': len(self._system_snapshots)
        }
    
    def export_performance_report(self, include_recent: bool = True) -> Dict[str, Any]:
        """
        Export comprehensive performance report.
        
        Args:
            include_recent: Whether to include recent operation data
            
        Returns:
            Comprehensive performance report
        """
        with self._lock:
            report = {
                'timestamp': time.time(),
                'system_health': self.get_system_health(),
                'operation_profiles': {
                    name: {
                        'operation_name': profile.operation_name,
                        'total_calls': profile.total_calls,
                        'total_duration': profile.total_duration,
                        'avg_duration': profile.avg_duration,
                        'min_duration': profile.min_duration,
                        'max_duration': profile.max_duration,
                        'success_rate': profile.success_count / max(profile.total_calls, 1),
                        'error_rate': profile.error_count / max(profile.total_calls, 1),
                        'last_execution': profile.last_execution,
                        'memory_delta_mb': profile.memory_delta_mb,
                        'cpu_usage_avg': profile.cpu_usage_avg
                    }
                    for name, profile in self._operation_profiles.items()
                },
                'thresholds': dict(self._duration_thresholds),
                'monitoring_config': {
                    'system_monitoring_enabled': self.enable_system_monitoring,
                    'monitoring_interval': self.monitoring_interval,
                    'memory_threshold_mb': self._memory_threshold_mb,
                    'cpu_threshold_percent': self._cpu_threshold_percent
                }
            }
            
            if include_recent:
                report['recent_operations'] = list(self._recent_operations)
        
        return report
    
    def reset_profiles(self, operation_name: Optional[str] = None):
        """
        Reset operation profiles.
        
        Args:
            operation_name: Optional specific operation to reset (resets all if None)
        """
        with self._lock:
            if operation_name:
                self._operation_profiles.pop(operation_name, None)
                self.logger.info(f"Reset profile for operation: {operation_name}")
            else:
                self._operation_profiles.clear()
                self._recent_operations.clear()
                self.logger.info("Reset all operation profiles")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Stopped background system monitoring")
    
    def _start_system_monitoring(self):
        """Start background system monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitoring_thread.start()
        self.logger.info("Started background system monitoring")
    
    def _system_monitoring_loop(self):
        """Background system monitoring loop."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                snapshot = self._take_system_snapshot()
                if snapshot:
                    with self._lock:
                        self._system_snapshots.append(snapshot)
                    
                    # Record system metrics
                    self.metrics.set_gauge("system.cpu_percent", snapshot.cpu_percent)
                    self.metrics.set_gauge("system.memory_percent", snapshot.memory_percent)
                    self.metrics.set_gauge("system.memory_usage_mb", snapshot.memory_usage_mb)
                    self.metrics.set_gauge("system.disk_usage_percent", snapshot.disk_usage_percent)
                    self.metrics.set_gauge("process.memory_mb", snapshot.process_memory_mb)
                    self.metrics.set_gauge("process.cpu_percent", snapshot.process_cpu_percent)
                    self.metrics.set_gauge("process.thread_count", snapshot.thread_count)
                    self.metrics.set_gauge("process.open_files", snapshot.open_files)
                    
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}", exc_info=True)
    
    def _take_system_snapshot(self) -> Optional[SystemSnapshot]:
        """Take a snapshot of current system resources."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(),
                memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                memory_percent=psutil.virtual_memory().percent,
                disk_usage_percent=psutil.disk_usage('/').percent,
                process_memory_mb=memory_info.rss / 1024 / 1024,
                process_cpu_percent=process.cpu_percent(),
                thread_count=process.num_threads(),
                open_files=len(process.open_files())
            )
        except Exception as e:
            self.logger.error(f"Failed to take system snapshot: {e}")
            return None
    
    def _update_operation_profile(
        self, 
        operation_name: str, 
        duration: float, 
        success: bool,
        snapshot: Optional[SystemSnapshot]
    ):
        """Update the performance profile for an operation."""
        with self._lock:
            if operation_name not in self._operation_profiles:
                self._operation_profiles[operation_name] = OperationProfile(operation_name)
            
            profile = self._operation_profiles[operation_name]
            profile.total_calls += 1
            profile.total_duration += duration
            profile.min_duration = min(profile.min_duration, duration)
            profile.max_duration = max(profile.max_duration, duration)
            profile.avg_duration = profile.total_duration / profile.total_calls
            profile.last_execution = time.time()
            profile.recent_durations.append(duration)
            
            if success:
                profile.success_count += 1
            else:
                profile.error_count += 1
            
            # Update resource usage if snapshot available
            if snapshot:
                profile.memory_delta_mb = snapshot.process_memory_mb
                profile.cpu_usage_avg = (
                    (profile.cpu_usage_avg * (profile.total_calls - 1) + snapshot.process_cpu_percent) /
                    profile.total_calls
                )
    
    def _record_operation_metrics(
        self, 
        operation_name: str, 
        duration: float, 
        success: bool,
        tags: Optional[Dict[str, str]]
    ):
        """Record operation metrics."""
        base_tags = {'operation': operation_name}
        if tags:
            base_tags.update(tags)
        
        # Record duration
        self.metrics.record_timer(f"operations.{operation_name}.duration", duration, base_tags)
        
        # Record success/failure
        if success:
            self.metrics.increment_counter(f"operations.{operation_name}.success", tags=base_tags)
        else:
            self.metrics.increment_counter(f"operations.{operation_name}.error", tags=base_tags)
        
        # Record completion
        self.metrics.increment_counter(f"operations.{operation_name}.completed", tags=base_tags)
    
    def _check_performance_thresholds(
        self, 
        operation_name: str, 
        duration: float,
        snapshot: Optional[SystemSnapshot]
    ):
        """Check performance thresholds and log warnings."""
        # Check duration threshold
        threshold = self._duration_thresholds.get(operation_name)
        if threshold and duration > threshold:
            self.logger.warning(
                f"Operation {operation_name} exceeded duration threshold",
                {
                    'operation_name': operation_name,
                    'duration': duration,
                    'threshold': threshold,
                    'excess': duration - threshold
                }
            )
        
        # Check system resource thresholds
        if snapshot:
            if snapshot.process_memory_mb > self._memory_threshold_mb:
                self.logger.warning(
                    f"High memory usage during {operation_name}",
                    {
                        'operation_name': operation_name,
                        'memory_mb': snapshot.process_memory_mb,
                        'threshold_mb': self._memory_threshold_mb
                    }
                )
            
            if snapshot.cpu_percent > self._cpu_threshold_percent:
                self.logger.warning(
                    f"High CPU usage during {operation_name}",
                    {
                        'operation_name': operation_name,
                        'cpu_percent': snapshot.cpu_percent,
                        'threshold_percent': self._cpu_threshold_percent
                    }
                )


class OperationTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(
        self, 
        monitor: PerformanceMonitor, 
        operation_name: str, 
        tags: Optional[Dict[str, str]] = None
    ):
        self.monitor = monitor
        self.operation_name = operation_name
        self.tags = tags
        self.operation_id: Optional[str] = None
        self.start_snapshot: Optional[SystemSnapshot] = None
    
    def __enter__(self):
        self.operation_id = self.monitor.start_operation(self.operation_name, self.tags)
        self.start_snapshot = self.monitor._take_system_snapshot()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            success = exc_type is None
            error = exc_val if exc_type else None
            
            # Calculate memory delta if we have snapshots
            metadata = {}
            if self.start_snapshot:
                end_snapshot = self.monitor._take_system_snapshot()
                if end_snapshot:
                    metadata['memory_delta_mb'] = (
                        end_snapshot.process_memory_mb - self.start_snapshot.process_memory_mb
                    )
                    metadata['cpu_delta_percent'] = (
                        end_snapshot.process_cpu_percent - self.start_snapshot.process_cpu_percent
                    )
            
            self.monitor.end_operation(self.operation_id, success, error, metadata)


# Convenience function for creating a performance monitor
def create_performance_monitor(
    metrics_collector: Optional[MetricsCollector] = None,
    logger: Optional[StructuredLogger] = None,
    enable_system_monitoring: bool = True
) -> PerformanceMonitor:
    """
    Create a performance monitor with default configuration.
    
    Args:
        metrics_collector: Optional metrics collector
        logger: Optional structured logger
        enable_system_monitoring: Whether to enable background monitoring
        
    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor(
        metrics_collector=metrics_collector,
        logger=logger,
        enable_system_monitoring=enable_system_monitoring
    ) 