"""
Core functionality for row ID generation.

This module contains the main functions for generating unique row IDs
from DataFrame data using SHA-256 hashing.
"""

import warnings
import uuid
import json
import traceback
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable, Iterator
from datetime import datetime
import pandas as pd
import numpy as np
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import gc
import psutil
import os
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from statistics import mean, stdev
import pickle
import cProfile
import pstats
import io
import resource
import linecache
import tracemalloc
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional development dependencies
try:
    from memory_profiler import profile as memory_profile
except ImportError:
    # Create a no-op decorator for production environments
    def memory_profile(func):
        return func

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    # Create placeholder for missing visualization libraries
    plt = None
    sns = None


class CollisionAlert:
    """Represents a collision detection alert with detailed information."""
    
    def __init__(self, hash_value: str, input1: str, input2: str, 
                 timestamp: datetime, alert_id: str):
        self.hash_value = hash_value
        self.input1 = input1
        self.input2 = input2
        self.timestamp = timestamp
        self.alert_id = alert_id
        self.severity = self._calculate_severity()
    
    def _calculate_severity(self) -> str:
        """Calculate alert severity based on input similarity."""
        # Calculate similarity between inputs
        similarity = self._calculate_similarity(self.input1, self.input2)
        
        if similarity > 0.8:
            return "CRITICAL"  # Very similar inputs with same hash = serious problem
        elif similarity > 0.5:
            return "HIGH"      # Somewhat similar inputs
        else:
            return "MEDIUM"    # Different inputs (expected rare collisions)
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate Jaccard similarity between two strings."""
        set1 = set(str1.lower().split('|'))
        set2 = set(str2.lower().split('|'))
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            'alert_id': self.alert_id,
            'hash_value': self.hash_value,
            'input1': self.input1,
            'input2': self.input2,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'similarity_score': self._calculate_similarity(self.input1, self.input2)
        }


class CollisionAlertManager:
    """Advanced collision alert management with multiple notification channels."""
    
    def __init__(self, alert_threshold: int = 1, enable_logging: bool = True):
        """
        Initialize collision alert manager.
        
        Args:
            alert_threshold: Number of collisions before triggering alerts
            enable_logging: Whether to enable detailed collision logging
        """
        self.alert_threshold = alert_threshold
        self.enable_logging = enable_logging
        self.alerts: List[CollisionAlert] = []
        self.alert_callbacks: List[Callable[[CollisionAlert], None]] = []
        self.collision_count = 0
        self.last_alert_time = None
        self.alert_cooldown = 60  # seconds between alerts for same hash
        
        # Setup logging
        if enable_logging:
            self.collision_logger = logging.getLogger('collision_detector')
            if not self.collision_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - COLLISION - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.collision_logger.addHandler(handler)
                self.collision_logger.setLevel(logging.WARNING)
    
    def add_alert_callback(self, callback: Callable[[CollisionAlert], None]) -> None:
        """Add a callback function for collision alerts."""
        self.alert_callbacks.append(callback)
    
    def process_collision(self, hash_value: str, input1: str, input2: str) -> None:
        """
        Process a detected collision and trigger alerts if necessary.
        
        Args:
            hash_value: The colliding hash value
            input1: First input that produced the hash
            input2: Second input that produced the hash
        """
        self.collision_count += 1
        
        # Create collision alert
        alert_id = f"COL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.collision_count}"
        alert = CollisionAlert(hash_value, input1, input2, datetime.now(), alert_id)
        self.alerts.append(alert)
        
        # Log collision
        if self.enable_logging:
            self.collision_logger.warning(
                f"HASH COLLISION DETECTED! ID: {alert_id}, "
                f"Hash: {hash_value[:16]}..., "
                f"Severity: {alert.severity}, "
                f"Input1: {input1[:50]}{'...' if len(input1) > 50 else ''}, "
                f"Input2: {input2[:50]}{'...' if len(input2) > 50 else ''}"
            )
        
        # Check if alert threshold is reached
        if self.collision_count >= self.alert_threshold:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: CollisionAlert) -> None:
        """Trigger alert through registered callbacks."""
        # Check cooldown
        current_time = time.time()
        if (self.last_alert_time and 
            current_time - self.last_alert_time < self.alert_cooldown):
            return
        
        self.last_alert_time = current_time
        
        # Trigger all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in collision alert callback: {e}")
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Get detailed collision statistics."""
        if not self.alerts:
            return {
                'total_collisions': 0,
                'collision_rate': 0.0,
                'severity_breakdown': {},
                'recent_collisions': []
            }
        
        # Severity breakdown
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity_counts[alert.severity] += 1
        
        # Recent collisions (last 10)
        recent_alerts = sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:10]
        
        return {
            'total_collisions': len(self.alerts),
            'collision_rate': len(self.alerts) / max(1, self.collision_count),
            'severity_breakdown': dict(severity_counts),
            'recent_collisions': [alert.to_dict() for alert in recent_alerts],
            'first_collision': self.alerts[0].timestamp.isoformat() if self.alerts else None,
            'last_collision': self.alerts[-1].timestamp.isoformat() if self.alerts else None
        }
    
    def export_collision_data(self, format: str = 'json') -> str:
        """
        Export collision data in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Serialized collision data
        """
        if format.lower() == 'json':
            data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_alerts': len(self.alerts),
                    'alert_threshold': self.alert_threshold
                },
                'alerts': [alert.to_dict() for alert in self.alerts],
                'statistics': self.get_collision_statistics()
            }
            return json.dumps(data, indent=2)
        
        elif format.lower() == 'csv':
            if not self.alerts:
                return "alert_id,hash_value,input1,input2,timestamp,severity,similarity_score\n"
            
            csv_lines = ["alert_id,hash_value,input1,input2,timestamp,severity,similarity_score"]
            for alert in self.alerts:
                alert_dict = alert.to_dict()
                csv_lines.append(
                    f"{alert_dict['alert_id']},{alert_dict['hash_value']},"
                    f"\"{alert_dict['input1']}\",\"{alert_dict['input2']}\","
                    f"{alert_dict['timestamp']},{alert_dict['severity']},"
                    f"{alert_dict['similarity_score']:.4f}"
                )
            return "\n".join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Standard alert callback functions
def email_alert_callback(alert: CollisionAlert) -> None:
    """Example email alert callback (placeholder implementation)."""
    logger.info(f"EMAIL ALERT: Collision detected - {alert.alert_id} (Severity: {alert.severity})")

def file_alert_callback(alert: CollisionAlert) -> None:
    """File-based alert callback that writes to collision log file."""
    try:
        with open('collision_alerts.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {alert.to_dict()}\n")
    except Exception as e:
        logger.error(f"Failed to write collision alert to file: {e}")

def webhook_alert_callback(alert: CollisionAlert) -> None:
    """Example webhook alert callback (placeholder implementation)."""
    logger.info(f"WEBHOOK ALERT: Collision {alert.alert_id} - Severity: {alert.severity}")


class PerformanceProfiler:
    """Advanced performance profiling for the hashing engine."""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        
    def start_profile(self, operation_name: str):
        """Start profiling an operation."""
        self.current_profile = {
            'operation': operation_name,
            'start_time': time.perf_counter(),
            'start_memory': self._get_memory_mb(),
            'cpu_start': psutil.cpu_percent()
        }
        
    def end_profile(self) -> Dict[str, Any]:
        """End profiling and return results."""
        if not self.current_profile:
            return {}
            
        end_time = time.perf_counter()
        end_memory = self._get_memory_mb()
        
        profile_result = {
            'operation': self.current_profile['operation'],
            'duration': end_time - self.current_profile['start_time'],
            'memory_delta': end_memory - self.current_profile['start_memory'],
            'peak_memory': end_memory,
            'cpu_usage': psutil.cpu_percent() - self.current_profile['cpu_start']
        }
        
        # Store in profiles
        op_name = self.current_profile['operation']
        if op_name not in self.profiles:
            self.profiles[op_name] = []
        self.profiles[op_name].append(profile_result)
        
        self.current_profile = None
        return profile_result
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        summary = {}
        for op_name, profiles in self.profiles.items():
            durations = [p['duration'] for p in profiles]
            memories = [p['memory_delta'] for p in profiles]
            
            summary[op_name] = {
                'call_count': len(profiles),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_delta': sum(memories) / len(memories),
                'total_duration': sum(durations)
            }
        
        return summary


def _hash_chunk_worker(chunk_data: List[str]) -> List[str]:
    """
    Worker function for parallel hash processing.
    
    Args:
        chunk_data: List of concatenated strings to hash
        
    Returns:
        List of hash values
    """
    return [hashlib.sha256(data.encode('utf-8')).hexdigest() for data in chunk_data]


class HashingEventType:
    """Event types for hashing operations observability."""
    HASH_GENERATION_START = "hash_generation_start"
    HASH_GENERATION_COMPLETE = "hash_generation_complete"
    VECTORIZED_PROCESSING_START = "vectorized_processing_start"
    VECTORIZED_PROCESSING_COMPLETE = "vectorized_processing_complete"
    COLLISION_DETECTED = "collision_detected"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"
    MEMORY_THRESHOLD_EXCEEDED = "memory_threshold_exceeded"
    BATCH_PROCESSING_START = "batch_processing_start"
    BATCH_PROCESSING_COMPLETE = "batch_processing_complete"
    PARALLEL_PROCESSING_START = "parallel_processing_start"
    PARALLEL_PROCESSING_COMPLETE = "parallel_processing_complete"
    ERROR_OCCURRED = "error_occurred"


class HashingEvent:
    """Structured event for hashing operations."""
    
    def __init__(self, event_type: str, event_id: str = None, **kwargs):
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = event_type
        self.timestamp = datetime.now()
        self.thread_id = threading.get_ident()
        self.process_id = os.getpid()
        self.attributes = kwargs
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for structured logging."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            **self.attributes
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class HashingMetrics:
    """Specialized metrics collection for hashing operations."""
    
    def __init__(self):
        self.operation_counts = defaultdict(int)
        self.operation_durations = defaultdict(list)
        self.memory_samples = []
        self.throughput_samples = []
        self.error_counts = defaultdict(int)
        self.collision_events = []
        self.lock = threading.Lock()
    
    def record_operation(self, operation_type: str, duration: float, memory_used: float = 0, rows_processed: int = 0):
        """Record an operation with metrics."""
        with self.lock:
            self.operation_counts[operation_type] += 1
            self.operation_durations[operation_type].append(duration)
            
            if memory_used > 0:
                self.memory_samples.append({
                    'timestamp': datetime.now(),
                    'memory_mb': memory_used,
                    'operation': operation_type
                })
            
            if rows_processed > 0 and duration > 0:
                throughput = rows_processed / duration
                self.throughput_samples.append({
                    'timestamp': datetime.now(),
                    'rows_per_second': throughput,
                    'operation': operation_type,
                    'rows_processed': rows_processed
                })
    
    def record_error(self, error_type: str, operation: str):
        """Record an error occurrence."""
        with self.lock:
            error_key = f"{operation}:{error_type}"
            self.error_counts[error_key] += 1
    
    def record_collision(self, hash_value: str, input1: str, input2: str):
        """Record a collision event."""
        with self.lock:
            self.collision_events.append({
                'timestamp': datetime.now(),
                'hash_value': hash_value,
                'input1_sample': input1[:100],  # First 100 chars
                'input2_sample': input2[:100],
                'similarity': self._calculate_similarity(input1, input2)
            })
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        set1 = set(str1.lower().split('|'))
        set2 = set(str2.lower().split('|'))
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self.lock:
            return {
                'operation_counts': dict(self.operation_counts),
                'operation_statistics': {
                    op: {
                        'count': len(durations),
                        'avg_duration': sum(durations) / len(durations) if durations else 0,
                        'min_duration': min(durations) if durations else 0,
                        'max_duration': max(durations) if durations else 0,
                        'total_duration': sum(durations)
                    }
                    for op, durations in self.operation_durations.items()
                },
                'memory_statistics': {
                    'sample_count': len(self.memory_samples),
                    'avg_memory_mb': sum(s['memory_mb'] for s in self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
                    'max_memory_mb': max((s['memory_mb'] for s in self.memory_samples), default=0),
                    'recent_samples': self.memory_samples[-10:]  # Last 10 samples
                },
                'throughput_statistics': {
                    'sample_count': len(self.throughput_samples),
                    'avg_throughput': sum(s['rows_per_second'] for s in self.throughput_samples) / len(self.throughput_samples) if self.throughput_samples else 0,
                    'max_throughput': max((s['rows_per_second'] for s in self.throughput_samples), default=0),
                    'total_rows_processed': sum(s['rows_processed'] for s in self.throughput_samples)
                },
                'error_statistics': dict(self.error_counts),
                'collision_statistics': {
                    'total_collisions': len(self.collision_events),
                    'recent_collisions': self.collision_events[-5:] if self.collision_events else []
                }
            }


class StructuredHashingLogger:
    """Structured logger for hashing operations with JSON formatting."""
    
    def __init__(self, name: str = "hashing_operations", log_file: str = "hashing_operations.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not create file handler: {e}")
    
    def log_event(self, event: HashingEvent, level: str = "INFO"):
        """Log a structured hashing event."""
        log_data = {
            'structured_event': True,
            'event_data': event.to_dict()
        }
        
        message = f"HASHING_EVENT: {json.dumps(log_data, default=str)}"
        
        if level.upper() == "DEBUG":
            self.logger.debug(message)
        elif level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        else:
            self.logger.info(message)
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics in structured format."""
        log_data = {
            'structured_metrics': True,
            'metrics_data': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        message = f"PERFORMANCE_METRICS: {json.dumps(log_data, default=str)}"
        self.logger.info(message)
    
    def log_operation_context(self, operation: str, context: Dict[str, Any]):
        """Log operation context information."""
        log_data = {
            'structured_context': True,
            'operation': operation,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        message = f"OPERATION_CONTEXT: {json.dumps(log_data, default=str)}"
        self.logger.info(message)


class HashingObserver:
    """Comprehensive observability framework for hashing operations."""
    
    def __init__(self, enable_detailed_logging: bool = True, metrics_retention_hours: int = 24):
        self.metrics = HashingMetrics()
        self.logger = StructuredHashingLogger()
        self.enable_detailed_logging = enable_detailed_logging
        self.metrics_retention_hours = metrics_retention_hours
        self.operation_contexts = {}
        self.active_operations = {}
        self.thresholds = {
            'max_memory_mb': 1000,
            'max_duration_seconds': 300,
            'min_throughput_rows_per_second': 1000
        }
    
    def set_thresholds(self, **kwargs):
        """Set performance thresholds for alerting."""
        self.thresholds.update(kwargs)
    
    def start_operation(self, operation_type: str, context: Dict[str, Any] = None) -> str:
        """Start observing an operation."""
        operation_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        self.active_operations[operation_id] = {
            'type': operation_type,
            'start_time': start_time,
            'context': context or {}
        }
        
        if context:
            self.operation_contexts[operation_id] = context
        
        # Log operation start
        event = HashingEvent(
            event_type=f"{operation_type}_start",
            operation_id=operation_id,
            context=context
        )
        
        if self.enable_detailed_logging:
            self.logger.log_event(event)
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, **result_data):
        """End observing an operation."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        end_time = time.perf_counter()
        duration = end_time - operation['start_time']
        
        # Calculate memory usage if available
        memory_used = result_data.get('memory_used', 0)
        rows_processed = result_data.get('rows_processed', 0)
        
        # Record metrics
        self.metrics.record_operation(
            operation['type'], 
            duration, 
            memory_used, 
            rows_processed
        )
        
        # Check thresholds
        alerts = self._check_thresholds(operation['type'], duration, memory_used, rows_processed)
        
        # Log operation completion
        event = HashingEvent(
            event_type=f"{operation['type']}_complete",
            operation_id=operation_id,
            duration_seconds=duration,
            success=success,
            memory_used_mb=memory_used,
            alerts=alerts,
            **result_data
        )
        
        if self.enable_detailed_logging:
            self.logger.log_event(event)
        
        # Log alerts if any
        for alert in alerts:
            self.logger.log_event(
                HashingEvent(
                    event_type=HashingEventType.PERFORMANCE_THRESHOLD_EXCEEDED,
                    operation_id=operation_id,
                    alert_type=alert['type'],
                    alert_message=alert['message'],
                    threshold_value=alert['threshold'],
                    actual_value=alert['actual']
                ),
                level="WARNING"
            )
        
        # Clean up
        del self.active_operations[operation_id]
        if operation_id in self.operation_contexts:
            del self.operation_contexts[operation_id]
    
    def log_collision(self, hash_value: str, input1: str, input2: str):
        """Log a collision event."""
        self.metrics.record_collision(hash_value, input1, input2)
        
        event = HashingEvent(
            event_type=HashingEventType.COLLISION_DETECTED,
            hash_value=hash_value[:16] + "...",
            input1_sample=input1[:50] + "..." if len(input1) > 50 else input1,
            input2_sample=input2[:50] + "..." if len(input2) > 50 else input2,
            similarity_score=self.metrics._calculate_similarity(input1, input2)
        )
        
        self.logger.log_event(event, level="WARNING")
    
    def log_error(self, operation: str, error: Exception, context: Dict[str, Any] = None):
        """Log an error event."""
        error_type = type(error).__name__
        self.metrics.record_error(error_type, operation)
        
        event = HashingEvent(
            event_type=HashingEventType.ERROR_OCCURRED,
            operation=operation,
            error_type=error_type,
            error_message=str(error),
            context=context or {}
        )
        
        self.logger.log_event(event, level="ERROR")
    
    def _check_thresholds(self, operation_type: str, duration: float, memory_used: float, rows_processed: int) -> List[Dict[str, Any]]:
        """Check if any performance thresholds were exceeded."""
        alerts = []
        
        if memory_used > self.thresholds['max_memory_mb']:
            alerts.append({
                'type': 'memory_threshold_exceeded',
                'message': f"Memory usage {memory_used:.2f}MB exceeds threshold {self.thresholds['max_memory_mb']}MB",
                'threshold': self.thresholds['max_memory_mb'],
                'actual': memory_used
            })
        
        if duration > self.thresholds['max_duration_seconds']:
            alerts.append({
                'type': 'duration_threshold_exceeded',
                'message': f"Operation duration {duration:.2f}s exceeds threshold {self.thresholds['max_duration_seconds']}s",
                'threshold': self.thresholds['max_duration_seconds'],
                'actual': duration
            })
        
        if rows_processed > 0 and duration > 0:
            throughput = rows_processed / duration
            if throughput < self.thresholds['min_throughput_rows_per_second']:
                alerts.append({
                    'type': 'throughput_threshold_exceeded',
                    'message': f"Throughput {throughput:.0f} rows/sec below threshold {self.thresholds['min_throughput_rows_per_second']} rows/sec",
                    'threshold': self.thresholds['min_throughput_rows_per_second'],
                    'actual': throughput
                })
        
        return alerts
    
    def get_observability_report(self) -> Dict[str, Any]:
        """Get comprehensive observability report."""
        return {
            'observer_info': {
                'detailed_logging_enabled': self.enable_detailed_logging,
                'metrics_retention_hours': self.metrics_retention_hours,
                'active_operations': len(self.active_operations),
                'thresholds': self.thresholds
            },
            'metrics_summary': self.metrics.get_summary(),
            'active_operations': {
                op_id: {
                    'type': op_data['type'],
                    'duration_so_far': time.perf_counter() - op_data['start_time'],
                    'context': op_data['context']
                }
                for op_id, op_data in self.active_operations.items()
            }
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'observer_config': {
                'detailed_logging_enabled': self.enable_detailed_logging,
                'metrics_retention_hours': self.metrics_retention_hours,
                'thresholds': self.thresholds
            },
            'metrics': self.metrics.get_summary()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.metrics = HashingMetrics()
        self.operation_contexts.clear()
        # Don't clear active operations as they might still be running


class HashingEngine:
    """
    High-performance SHA-256 hashing engine with advanced collision detection,
    performance monitoring, and parallel processing capabilities.
    """
    
    def __init__(self, enable_profiling: bool = False, max_workers: Optional[int] = None,
                 collision_alert_threshold: int = 1, enable_collision_alerts: bool = True,
                 enable_regression_detection: bool = False, baseline_name: str = "default",
                 enable_observability: bool = True, observability_config: Dict[str, Any] = None):
        """
        Initialize the hashing engine with enhanced collision detection, regression monitoring, and observability.
        
        Args:
            enable_profiling: Enable detailed performance profiling
            max_workers: Maximum number of worker processes for parallel processing
            collision_alert_threshold: Number of collisions before triggering alerts
            enable_collision_alerts: Whether to enable collision alerting system
            enable_regression_detection: Whether to enable performance regression detection
            baseline_name: Name of the performance baseline to use
            enable_observability: Whether to enable comprehensive observability framework
            observability_config: Configuration for observability system
        """
        self.collision_tracker = defaultdict(list)
        self.performance_metrics = {
            'total_rows_processed': 0,
            'processing_times': [],
            'memory_usage': [],
            'collision_count': 0
        }
        self.logger = logging.getLogger(__name__)
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Limit to prevent overhead
        
        # Enhanced collision detection system
        self.collision_alert_manager = CollisionAlertManager(
            alert_threshold=collision_alert_threshold,
            enable_logging=enable_collision_alerts
        ) if enable_collision_alerts else None
        
        # Add default alert callbacks if alerting is enabled
        if self.collision_alert_manager:
            self.collision_alert_manager.add_alert_callback(file_alert_callback)
            # Add email/webhook callbacks if needed
            # self.collision_alert_manager.add_alert_callback(email_alert_callback)
            # self.collision_alert_manager.add_alert_callback(webhook_alert_callback)
        
        # Performance regression detection system
        self.enable_regression_detection = enable_regression_detection
        self.baseline_name = baseline_name
        self.regression_detector = None
        
        if enable_regression_detection:
            self.regression_detector = PerformanceRegressionDetector()
            # Add default regression alert callbacks
            self.regression_detector.add_alert_callback(file_regression_alert_callback)
            self.regression_detector.add_alert_callback(console_regression_alert_callback)
        
        # Comprehensive observability framework
        self.enable_observability = enable_observability
        self.observer = None
        
        if enable_observability:
            config = observability_config or {}
            self.observer = HashingObserver(
                enable_detailed_logging=config.get('detailed_logging', True),
                metrics_retention_hours=config.get('metrics_retention_hours', 24)
            )
            
            # Set custom thresholds if provided
            if 'thresholds' in config:
                self.observer.set_thresholds(**config['thresholds'])
        
        self.logger.info(
            f"HashingEngine initialized with {self.max_workers} max workers, "
            f"collision alerting: {enable_collision_alerts}, "
            f"regression detection: {enable_regression_detection}, "
            f"observability: {enable_observability}"
        )

    def generate_row_hash(self, row_data: List[Any], separator: str = '|') -> str:
        """
        Generate SHA-256 hash for a single row of data with comprehensive observability.
        
        Args:
            row_data: List of values to be hashed
            separator: String to join values (default: '|')
            
        Returns:
            64-character hexadecimal SHA-256 hash
            
        Raises:
            ValueError: If row_data is empty or separator is invalid
            TypeError: If separator is not a string
        """
        # Start observability tracking
        operation_id = None
        if self.observer:
            operation_id = self.observer.start_operation(
                HashingEventType.HASH_GENERATION_START,
                context={
                    'data_length': len(row_data) if row_data else 0,
                    'separator': separator,
                    'row_sample': str(row_data)[:100] if row_data else "empty"
                }
            )
        
        try:
            # Input validation with observability
            if not row_data:
                error = ValueError("row_data cannot be empty")
                if self.observer:
                    self.observer.log_error("generate_row_hash", error, {'input': 'empty_row_data'})
                raise error
            
            if not isinstance(separator, str):
                error = TypeError("separator must be a string")
                if self.observer:
                    self.observer.log_error("generate_row_hash", error, {'separator_type': type(separator)})
                raise error
            
            if '|' in separator and separator != '|':
                error = ValueError("Custom separator cannot contain '|' character")
                if self.observer:
                    self.observer.log_error("generate_row_hash", error, {'invalid_separator': separator})
                raise error
            
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Convert all values to strings, handling None values
            str_values = []
            for value in row_data:
                if value is None:
                    str_values.append('NULL')
                else:
                    str_values.append(str(value))
            
            # Join values with separator
            concatenated = separator.join(str_values)
            
            # Generate hash
            hash_value = hashlib.sha256(concatenated.encode('utf-8')).hexdigest()
            
            # Track performance metrics
            end_time = time.perf_counter()
            duration = end_time - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            
            self.performance_metrics['total_rows_processed'] += 1
            self.performance_metrics['processing_times'].append(duration)
            self.performance_metrics['memory_usage'].append(memory_used)
            
            # Profile this operation if enabled
            if self.profiler:
                self.profiler.record_operation(
                    'single_hash_generation',
                    duration,
                    memory_used,
                    {'input_size': len(concatenated), 'separator': separator}
                )
            
            # Collision detection with observability
            collision_key = hash_value
            input_signature = concatenated
            
            if collision_key in self.collision_tracker:
                # Check if this is a real collision (different input, same hash)
                existing_inputs = self.collision_tracker[collision_key]
                for existing_input in existing_inputs:
                    if existing_input != input_signature:
                        # Real collision detected!
                        collision_info = {
                            'hash': hash_value,
                            'input1': existing_input,
                            'input2': input_signature,
                            'timestamp': datetime.now()
                        }
                        
                        self.performance_metrics['collision_count'] += 1
                        self.logger.warning(f"Hash collision detected: {collision_info}")
                        
                        # Trigger collision alerts
                        if self.collision_alert_manager:
                            alert = CollisionAlert(
                                hash_value=hash_value,
                                input1=existing_input,
                                input2=input_signature,
                                detection_method="single_row_generation"
                            )
                            self.collision_alert_manager.process_collision(hash_value, existing_input, input_signature)
                        
                        # Log with observability framework
                        if self.observer:
                            self.observer.log_collision(hash_value, existing_input, input_signature)
                        
                        break
                
                # Add this input to the tracker (even if collision)
                self.collision_tracker[collision_key].append(input_signature)
            else:
                # First time seeing this hash
                self.collision_tracker[collision_key] = [input_signature]
            
            # End observability tracking
            if self.observer:
                self.observer.end_operation(
                    operation_id,
                    success=True,
                    rows_processed=1,
                    memory_used=memory_used,
                    hash_value=hash_value[:16] + "...",
                    processing_duration=duration,
                    input_size=len(concatenated)
                )
            
            return hash_value
            
        except Exception as e:
            # Log error and end observability tracking
            if self.observer:
                self.observer.log_error("generate_row_hash", e, {'row_data': str(row_data)[:100]})
                if operation_id:
                    self.observer.end_operation(operation_id, success=False, error=str(e))
            raise

    def generate_row_ids_vectorized(self, processed_df: pd.DataFrame, separator: str = '|', 
                                   batch_size: int = 10000, progress_callback: Optional[Callable] = None) -> pd.Series:
        """
        Generate SHA-256 hashes for all rows in a DataFrame using vectorized operations with observability.
        
        Args:
            processed_df: DataFrame with prepared data
            separator: String separator for joining row values
            batch_size: Number of rows to process in each batch
            progress_callback: Optional callback function for progress updates
            
        Returns:
            pandas Series containing SHA-256 hashes
        """
        # Start observability tracking
        operation_id = None
        if self.observer:
            operation_id = self.observer.start_operation(
                HashingEventType.VECTORIZED_PROCESSING_START,
                context={
                    'total_rows': len(processed_df),
                    'batch_size': batch_size,
                    'separator': separator,
                    'columns': list(processed_df.columns)
                }
            )
        
        try:
            if processed_df.empty:
                error = ValueError("DataFrame cannot be empty")
                if self.observer:
                    self.observer.log_error("generate_row_ids_vectorized", error, {'dataframe_shape': processed_df.shape})
                raise error
            
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            all_hashes = []
            total_rows = len(processed_df)
            processed_rows = 0
            collision_count = 0
            
            # Log processing start
            self.logger.info(f"Starting vectorized hash generation for {total_rows} rows with batch size {batch_size}")
            
            # Process in batches for memory efficiency
            for i in range(0, total_rows, batch_size):
                batch_start_time = time.perf_counter()
                batch_end = min(i + batch_size, total_rows)
                batch_df = processed_df.iloc[i:batch_end]
                
                # Start batch observability tracking
                batch_operation_id = None
                if self.observer:
                    batch_operation_id = self.observer.start_operation(
                        HashingEventType.BATCH_PROCESSING_START,
                        context={
                            'batch_index': i // batch_size,
                            'batch_start': i,
                            'batch_end': batch_end,
                            'batch_size': len(batch_df)
                        }
                    )
                
                # Handle None values and convert to strings
                batch_df_str = batch_df.fillna('NULL').astype(str)
                
                # Vectorized concatenation with separator
                if self.profiler and hasattr(self.profiler, 'use_fast_concatenation') and self.profiler.use_fast_concatenation:
                    # Use optimized concatenation
                    concatenated_rows = self._optimized_vectorized_concat(batch_df_str, separator)
                else:
                    # Standard concatenation
                    concatenated_rows = batch_df_str.apply(lambda row: separator.join(row), axis=1)
                
                # Generate hashes for the batch
                batch_hashes = concatenated_rows.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
                
                # Batch collision detection with observability
                batch_collisions = 0
                for idx, (hash_val, input_str) in enumerate(zip(batch_hashes, concatenated_rows)):
                    collision_key = hash_val
                    input_signature = input_str
                    
                    if collision_key in self.collision_tracker:
                        existing_inputs = self.collision_tracker[collision_key]
                        for existing_input in existing_inputs:
                            if existing_input != input_signature:
                                # Real collision detected!
                                collision_count += 1
                                batch_collisions += 1
                                
                                collision_info = {
                                    'hash': hash_val,
                                    'input1': existing_input,
                                    'input2': input_signature,
                                    'batch_index': i // batch_size,
                                    'row_index': i + idx,
                                    'timestamp': datetime.now()
                                }
                                
                                self.logger.warning(f"Vectorized collision detected: {collision_info}")
                                
                                # Trigger collision alerts
                                if self.collision_alert_manager:
                                    alert = CollisionAlert(
                                        hash_value=hash_val,
                                        input1=existing_input,
                                        input2=input_signature,
                                        detection_method="vectorized_processing"
                                    )
                                    self.collision_alert_manager.process_collision(hash_val, existing_input, input_signature)
                                
                                # Log with observability framework
                                if self.observer:
                                    self.observer.log_collision(hash_val, existing_input, input_signature)
                                
                                break
                        
                        self.collision_tracker[collision_key].append(input_signature)
                    else:
                        self.collision_tracker[collision_key] = [input_signature]
                
                all_hashes.extend(batch_hashes.tolist())
                processed_rows += len(batch_df)
                
                # Track batch performance
                batch_end_time = time.perf_counter()
                batch_duration = batch_end_time - batch_start_time
                batch_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Log batch completion with observability
                if self.observer and batch_operation_id:
                    self.observer.end_operation(
                        batch_operation_id,
                        success=True,
                        rows_processed=len(batch_df),
                        memory_used=batch_memory - start_memory,
                        collisions_detected=batch_collisions,
                        processing_duration=batch_duration
                    )
                
                # Progress callback
                if progress_callback:
                    progress_callback(processed_rows, total_rows, batch_duration)
                
                # Profile batch if enabled
                if self.profiler:
                    self.profiler.record_operation(
                        'vectorized_batch_processing',
                        batch_duration,
                        batch_memory - start_memory,
                        {
                            'batch_size': len(batch_df),
                            'batch_index': i // batch_size,
                            'collisions': batch_collisions
                        }
                    )
                
                # Log batch progress
                if (i // batch_size) % 10 == 0:  # Log every 10 batches
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    rows_per_sec = processed_rows / (batch_end_time - start_time) if (batch_end_time - start_time) > 0 else 0
                    self.logger.info(
                        f"Processed batch {i // batch_size + 1}, "
                        f"{processed_rows}/{total_rows} rows ({processed_rows/total_rows*100:.1f}%), "
                        f"{rows_per_sec:.0f} rows/sec, "
                        f"Memory: {current_memory:.1f}MB, "
                        f"Collisions: {collision_count}"
                    )
            
            # Calculate final performance metrics
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            
            # Update performance metrics
            self.performance_metrics['total_rows_processed'] += total_rows
            self.performance_metrics['processing_times'].append(total_duration)
            self.performance_metrics['memory_usage'].append(memory_used)
            self.performance_metrics['collision_count'] += collision_count
            
            # Calculate throughput
            rows_per_second = total_rows / total_duration if total_duration > 0 else 0
            
            # Log completion
            self.logger.info(
                f"Vectorized processing completed: {total_rows} rows in {total_duration:.2f}s "
                f"({rows_per_second:.0f} rows/sec), Memory used: {memory_used:.2f}MB, "
                f"Collisions detected: {collision_count}"
            )
            
            # Profile overall operation if enabled
            if self.profiler:
                self.profiler.record_operation(
                    'vectorized_processing_complete',
                    total_duration,
                    memory_used,
                    {
                        'total_rows': total_rows,
                        'batch_count': (total_rows + batch_size - 1) // batch_size,
                        'rows_per_second': rows_per_second,
                        'total_collisions': collision_count
                    }
                )
            
            # End observability tracking
            if self.observer:
                self.observer.end_operation(
                    operation_id,
                    success=True,
                    rows_processed=total_rows,
                    memory_used=memory_used,
                    processing_duration=total_duration,
                    throughput_rows_per_sec=rows_per_second,
                    total_collisions=collision_count,
                    batch_count=(total_rows + batch_size - 1) // batch_size
                )
            
            return pd.Series(all_hashes, index=processed_df.index)
            
        except Exception as e:
            # Log error and end observability tracking
            if self.observer:
                self.observer.log_error("generate_row_ids_vectorized", e, {
                    'dataframe_shape': processed_df.shape,
                    'batch_size': batch_size,
                    'processed_rows': processed_rows if 'processed_rows' in locals() else 0
                })
                if operation_id:
                    self.observer.end_operation(operation_id, success=False, error=str(e))
            raise

    def get_performance_profile(self) -> Dict[str, Any]:
        """
        Get detailed performance profiling information.
        
        Returns:
            Dictionary containing profiling data
        """
        if not self.profiler:
            return {"profiling": "disabled"}
        
        return {
            "profiling_enabled": True,
            "profile_summary": self.profiler.get_profile_summary(),
            "performance_metrics": self.get_performance_metrics()
        }

    def benchmark_performance(self, test_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run performance benchmarks with different dataset sizes.
        
        Args:
            test_sizes: List of dataset sizes to test
            
        Returns:
            Benchmark results
        """
        if test_sizes is None:
            test_sizes = [100, 1000, 10000, 50000]
        
        benchmark_results = {}
        
        for size in test_sizes:
            # Generate test data
            test_df = pd.DataFrame({
                'id': range(size),
                'value': np.random.randint(1, 1000, size),
                'category': np.random.choice(['A', 'B', 'C', 'D'], size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H')
            })
            
            # Reset metrics for clean measurement
            self.reset_metrics()
            
            # Benchmark different approaches
            start_time = time.perf_counter()
            
            if size <= 10000:
                # Test single-threaded
                results_single = self.generate_row_ids_vectorized(test_df, parallel=False)
                single_time = time.perf_counter() - start_time
                
                # Test parallel (if size is large enough)
                if size >= 1000:
                    start_time = time.perf_counter()
                    results_parallel = self.generate_row_ids_vectorized(test_df, parallel=True)
                    parallel_time = time.perf_counter() - start_time
                else:
                    parallel_time = None
            else:
                # Only test parallel for very large datasets
                results_parallel = self.generate_row_ids_vectorized(test_df, parallel=True)
                parallel_time = time.perf_counter() - start_time
                single_time = None
            
            metrics = self.get_performance_metrics()
            
            benchmark_results[f"{size}_rows"] = {
                "dataset_size": size,
                "single_threaded_time": single_time,
                "parallel_time": parallel_time,
                "processing_rate": metrics.get('overall_processing_rate', 0),
                "memory_usage": metrics.get('peak_memory_usage', 0),
                "collision_count": metrics.get('collision_count', 0)
            }
        
        return benchmark_results

    def _track_hash(self, hash_value: str, original_data: str) -> None:
        """
        Enhanced hash collision tracking with advanced alerting.
        
        Args:
            hash_value: The generated hash
            original_data: The original concatenated string that produced the hash
        """
        if hash_value in self.collision_tracker:
            # Check if this is a true collision (different data, same hash)
            existing_data = self.collision_tracker[hash_value]
            if original_data not in existing_data:
                existing_data.append(original_data)
                self.performance_metrics['collision_count'] += 1
                
                # Enhanced collision logging
                self.logger.warning(
                    f"Hash collision detected for hash {hash_value[:8]}... "
                    f"between data: '{existing_data[0][:50]}...' and '{original_data[:50]}...'"
                )
                
                # Trigger advanced collision alerting system
                if self.collision_alert_manager:
                    self.collision_alert_manager.process_collision(
                        hash_value, existing_data[0], original_data
                    )
        else:
            self.collision_tracker[hash_value] = [original_data]
    
    def get_enhanced_collision_report(self) -> Dict[str, Any]:
        """
        Get comprehensive collision report with enhanced alerting data.
        
        Returns:
            Dictionary containing enhanced collision statistics and alert information
        """
        # Get basic collision report
        basic_report = self.get_collision_report()
        
        # Add enhanced collision detection data
        enhanced_report = basic_report.copy()
        
        if self.collision_alert_manager:
            alert_stats = self.collision_alert_manager.get_collision_statistics()
            enhanced_report.update({
                'alert_system_enabled': True,
                'alert_statistics': alert_stats,
                'alert_threshold': self.collision_alert_manager.alert_threshold,
                'total_alerts_generated': len(self.collision_alert_manager.alerts)
            })
        else:
            enhanced_report['alert_system_enabled'] = False
        
        return enhanced_report
    
    def export_collision_alerts(self, format: str = 'json') -> Optional[str]:
        """
        Export collision alert data in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Serialized collision alert data or None if alerting disabled
        """
        if not self.collision_alert_manager:
            return None
        
        return self.collision_alert_manager.export_collision_data(format)
    
    def add_collision_alert_callback(self, callback: Callable) -> bool:
        """
        Add a custom callback function for collision alerts.
        
        Args:
            callback: Function to call when collision alerts are triggered
            
        Returns:
            True if callback was added, False if alerting is disabled
        """
        if not self.collision_alert_manager:
            return False
        
        self.collision_alert_manager.add_alert_callback(callback)
        return True
    
    def simulate_collision_for_testing(self, hash_value: str = None) -> bool:
        """
        Simulate a hash collision for testing the alerting system.
        
        Args:
            hash_value: Optional specific hash value to use for simulation
            
        Returns:
            True if simulation was successful, False if alerting disabled
        """
        if not self.collision_alert_manager:
            return False
        
        test_hash = hash_value or "test_collision_hash_123456789abcdef"
        test_input1 = "simulation_input_1|test|data"
        test_input2 = "simulation_input_2|different|data"
        
        # Simulate collision detection
        self.collision_alert_manager.process_collision(test_hash, test_input1, test_input2)
        
        self.logger.info("Collision simulation completed for testing purposes")
        return True

    def get_collision_report(self) -> Dict[str, Any]:
        """
        Get a report of detected hash collisions.
        
        Returns:
            Dictionary containing collision statistics and details
        """
        total_collisions = self.performance_metrics['collision_count']
        collision_details = {}
        
        for hash_val, data_list in self.collision_tracker.items():
            if len(data_list) > 1:
                collision_details[hash_val] = {
                    'collision_count': len(data_list) - 1,
                    'data_samples': data_list[:3]  # Show first 3 samples
                }
        
        return {
            'total_collisions': total_collisions,
            'unique_collision_hashes': len(collision_details),
            'collision_details': collision_details,
            'collision_rate': (
                total_collisions / self.performance_metrics['total_rows_processed']
                if self.performance_metrics['total_rows_processed'] > 0 else 0
            )
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance statistics
        """
        processing_times = self.performance_metrics['processing_times']
        memory_usage = self.performance_metrics['memory_usage']
        
        return {
            'total_rows_processed': self.performance_metrics['total_rows_processed'],
            'collision_count': self.performance_metrics['collision_count'],
            'avg_processing_time': (
                sum(processing_times) / len(processing_times) 
                if processing_times else 0
            ),
            'total_processing_time': sum(processing_times),
            'peak_memory_usage': max(memory_usage) if memory_usage else 0,
            'avg_memory_usage': (
                sum(memory_usage) / len(memory_usage) 
                if memory_usage else 0
            ),
            'overall_processing_rate': (
                self.performance_metrics['total_rows_processed'] / sum(self.performance_metrics['processing_times'])
                if sum(self.performance_metrics['processing_times']) > 0 else 0
            )
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics and collision tracking."""
        self.collision_tracker.clear()
        self.performance_metrics = {
            'total_rows_processed': 0,
            'processing_times': [],
            'memory_usage': [],
            'collision_count': 0
        }
        self.logger.info("HashingEngine metrics reset")

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _update_metrics(self, rows_processed: int, processing_time: float, 
                       memory_used: float, rows_per_second: float) -> None:
        """
        Update performance metrics with processing statistics.
        
        Args:
            rows_processed: Number of rows processed
            processing_time: Time taken for processing
            memory_used: Memory consumed during processing
            rows_per_second: Processing rate
        """
        self.performance_metrics['total_rows_processed'] += rows_processed
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['memory_usage'].append(memory_used)
        
        self.logger.info(
            f"Metrics updated: +{rows_processed} rows, {processing_time:.2f}s, "
            f"{memory_used:.2f}MB, {rows_per_second:.0f} rows/sec"
        )
    
    def _process_in_batches(self, df: pd.DataFrame, separator: str, batch_size: int, 
                           show_progress: bool) -> pd.Series:
        """
        Process DataFrame in batches for memory-efficient handling of large datasets.
        
        Args:
            df: Input DataFrame
            separator: String separator for concatenation
            batch_size: Number of rows per batch
            show_progress: Whether to show progress
            
        Returns:
            Series of generated hash IDs
        """
        total_rows = len(df)
        batch_results = []
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            
            # Use optimized batch processing
            batch_hashes = self._process_vectorized_batch_optimized(batch_df, separator)
            batch_results.append(batch_hashes)
            
            if show_progress:
                progress = (batch_end / total_rows) * 100
                print(f"Progress: {batch_end}/{total_rows} rows ({progress:.1f}%)")
        
        # Combine all batch results
        return pd.concat(batch_results, ignore_index=True)
    
    def _generate_and_track_hash(self, concatenated_string: str) -> str:
        """
        Generate hash for a concatenated string and track for collisions.
        
        Args:
            concatenated_string: Pre-concatenated string data
            
        Returns:
            64-character hexadecimal hash
        """
        hash_value = hashlib.sha256(concatenated_string.encode('utf-8')).hexdigest()
        self._track_hash(hash_value, concatenated_string)
        return hash_value
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Dictionary containing detailed performance statistics
        """
        processing_times = self.performance_metrics['processing_times']
        memory_usage = self.performance_metrics['memory_usage']
        total_rows = self.performance_metrics['total_rows_processed']
        
        if not processing_times:
            return {
                'total_rows_processed': 0,
                'performance_summary': 'No processing completed yet'
            }
        
        total_time = sum(processing_times)
        overall_rate = total_rows / total_time if total_time > 0 else 0
        
        report = {
            'total_rows_processed': total_rows,
            'total_processing_time': total_time,
            'overall_processing_rate': overall_rate,
            'avg_batch_time': sum(processing_times) / len(processing_times),
            'fastest_batch_time': min(processing_times),
            'slowest_batch_time': max(processing_times),
            'peak_memory_usage': max(memory_usage) if memory_usage else 0,
            'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'collision_count': self.performance_metrics['collision_count'],
            'collision_rate': (
                self.performance_metrics['collision_count'] / total_rows
                if total_rows > 0 else 0
            ),
            'batches_processed': len(processing_times)
        }
        
        # Automatic regression detection if enabled
        if self.enable_regression_detection and self.regression_detector:
            regressions = self.check_for_regressions()
            if regressions:
                report['performance_regressions'] = [alert.to_dict() for alert in regressions]
                report['regression_detected'] = True
            else:
                report['regression_detected'] = False
        
        return report
    
    def establish_performance_baseline(self, baseline_name: Optional[str] = None) -> bool:
        """
        Establish a performance baseline from current metrics.
        
        Args:
            baseline_name: Name for the baseline (defaults to instance baseline_name)
            
        Returns:
            True if baseline was established successfully
        """
        if not self.enable_regression_detection or not self.regression_detector:
            self.logger.warning("Regression detection is not enabled")
            return False
        
        baseline_name = baseline_name or self.baseline_name
        current_metrics = self.get_performance_metrics()
        
        if current_metrics['total_rows_processed'] == 0:
            self.logger.warning("No performance data available to establish baseline")
            return False
        
        return self.regression_detector.establish_baseline(baseline_name, current_metrics)
    
    def check_for_regressions(self, baseline_name: Optional[str] = None) -> List['RegressionAlert']:
        """
        Check current performance against baseline for regressions.
        
        Args:
            baseline_name: Name of baseline to compare against (defaults to instance baseline_name)
            
        Returns:
            List of regression alerts
        """
        if not self.enable_regression_detection or not self.regression_detector:
            return []
        
        baseline_name = baseline_name or self.baseline_name
        current_metrics = self.get_performance_metrics()
        
        return self.regression_detector.detect_regressions(current_metrics, baseline_name)
    
    def get_regression_detection_report(self) -> Dict[str, Any]:
        """
        Get comprehensive regression detection report.
        
        Returns:
            Dictionary containing regression detection statistics and alerts
        """
        if not self.enable_regression_detection or not self.regression_detector:
            return {
                'regression_detection_enabled': False,
                'message': 'Regression detection is not enabled'
            }
        
        report = self.regression_detector.get_regression_report()
        report['regression_detection_enabled'] = True
        report['current_baseline'] = self.baseline_name
        
        return report
    
    def set_regression_thresholds(self, metric: str, critical: float, warning: float) -> bool:
        """
        Set custom regression detection thresholds.
        
        Args:
            metric: Performance metric name
            critical: Critical threshold percentage
            warning: Warning threshold percentage
            
        Returns:
            True if thresholds were set successfully
        """
        if not self.enable_regression_detection or not self.regression_detector:
            self.logger.warning("Regression detection is not enabled")
            return False
        
        self.regression_detector.set_thresholds(metric, critical, warning)
        return True
    
    def export_performance_baselines(self) -> Optional[str]:
        """
        Export all performance baselines to JSON format.
        
        Returns:
            JSON string with baseline data or None if regression detection disabled
        """
        if not self.enable_regression_detection or not self.regression_detector:
            return None
        
        return self.regression_detector.export_baselines()
    
    def add_regression_alert_callback(self, callback: Callable[['RegressionAlert'], None]) -> bool:
        """
        Add a custom callback for regression alerts.
        
        Args:
            callback: Function to call when regression alerts are triggered
            
        Returns:
            True if callback was added successfully
        """
        if not self.enable_regression_detection or not self.regression_detector:
            return False
        
        self.regression_detector.add_alert_callback(callback)
        return True


# Create global instance for backward compatibility
_hashing_engine = HashingEngine()


def generate_unique_row_ids(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    id_column_name: str = 'row_id',
    uniqueness_threshold: float = 0.95,
    separator: str = '|',
    enable_monitoring: bool = True,
    enable_quality_checks: bool = True,
    show_progress: bool = True,
    show_warnings: bool = True,
    enable_enhanced_lineage: bool = False,
    return_audit_trail: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Generate unique row IDs for DataFrame rows with comprehensive validation and monitoring.
    
    Args:
        df: Input DataFrame
        columns: Optional list of columns to use for ID generation
        id_column_name: Name for the generated ID column
        uniqueness_threshold: Minimum uniqueness ratio for column selection
        separator: String separator for concatenation
        enable_monitoring: Enable observability and monitoring
        enable_quality_checks: Enable data quality validation
        show_progress: Show progress indicators
        show_warnings: Display validation warnings
        enable_enhanced_lineage: Enable enhanced data lineage tracking
        return_audit_trail: Return comprehensive audit trail instead of DataFrame
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with row IDs or comprehensive audit trail dictionary
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If input validation fails
        RuntimeError: If processing encounters critical errors
    """
    # Create session and operation context
    session_id = str(uuid.uuid4())
    operation_context = create_error_context(
        'generate_unique_row_ids',
        session_id=session_id,
        df_shape=df.shape if hasattr(df, 'shape') else None,
        columns_specified=columns is not None,
        enable_monitoring=enable_monitoring,
        enable_quality_checks=enable_quality_checks
    )
    
    try:
        # ========================================
        # STAGE 1: COMPREHENSIVE INPUT VALIDATION
        # ========================================
        logger.info(f"Starting row ID generation (Session: {session_id})")
        
        # Validate all inputs using the new validation system
        validation_results = validate_all_inputs(
            df=df,
            columns=columns,
            id_column_name=id_column_name,
            uniqueness_threshold=uniqueness_threshold,
            separator=separator,
            show_progress=show_progress,
            enable_monitoring=enable_monitoring,
            enable_quality_checks=enable_quality_checks,
            show_warnings=show_warnings,
            **kwargs
        )
        
        # Handle validation warnings
        handle_validation_warnings(
            validation_results, 
            show_warnings=show_warnings
        )
        
        # Use validated parameters
        validated_params = validation_results['parameters_validated']
        id_column_name = validated_params['id_column_name']
        uniqueness_threshold = validated_params['uniqueness_threshold']
        separator = validated_params['separator']
        
        # ========================================
        # STAGE 2: INITIALIZE MONITORING AND LINEAGE
        # ========================================
        
        # Initialize monitoring if enabled
        observer = None
        if enable_monitoring:
            observer = HashingObserver()
            observer.start_session(session_id)
        
        # Initialize enhanced lineage tracking if enabled
        lineage_tracker = None
        if enable_enhanced_lineage:
            lineage_tracker = DataLineageTracker()
            lineage_tracker.start_session(session_id)
            lineage_tracker.add_event('validation_completed', {
                'validation_results': validation_results,
                'validated_parameters': validated_params
            })
        
        # ========================================
        # STAGE 3: COLUMN SELECTION
        # ========================================
        
        if columns is None:
            logger.info("Performing automatic column selection...")
            try:
                columns = select_columns_for_hashing(
                    df, 
                    uniqueness_threshold=uniqueness_threshold,
                    observer=observer
                )
                logger.info(f"Automatically selected {len(columns)} columns: {columns}")
                
                if lineage_tracker:
                    lineage_tracker.add_event('column_selection_completed', {
                        'selected_columns': columns,
                        'selection_method': 'automatic'
                    })
                    
            except Exception as e:
                error_context = create_error_context(
                    'column_selection', 
                    session_id=session_id,
                    threshold=uniqueness_threshold
                )
                handle_processing_error(e, error_context)
                raise RuntimeError(f"Column selection failed: {str(e)}") from e
        else:
            logger.info(f"Using manually specified columns: {columns}")
            
            # Perform additional uniqueness analysis for manual columns
            uniqueness_analysis = check_uniqueness_warning(
                df, columns, uniqueness_threshold
            )
            
            if lineage_tracker:
                lineage_tracker.add_event('column_selection_completed', {
                    'selected_columns': columns,
                    'selection_method': 'manual',
                    'uniqueness_analysis': uniqueness_analysis
                })
        
        # ========================================
        # STAGE 4: DATA PREPROCESSING
        # ========================================
        
        logger.info("Preprocessing data for hashing...")
        try:
            processed_df = preprocess_data_for_hashing(
                df, 
                columns, 
                separator=separator,
                observer=observer
            )
            
            if lineage_tracker:
                lineage_tracker.add_event('preprocessing_completed', {
                    'processed_columns': columns,
                    'separator': separator,
                    'processed_shape': processed_df.shape
                })
                
        except Exception as e:
            error_context = create_error_context(
                'data_preprocessing', 
                session_id=session_id,
                columns=columns,
                separator=separator
            )
            handle_processing_error(e, error_context)
            raise RuntimeError(f"Data preprocessing failed: {str(e)}") from e
        
        # ========================================
        # STAGE 5: HASH GENERATION
        # ========================================
        
        logger.info("Generating row IDs...")
        try:
            # Initialize progress tracking
            progress_tracker = None
            if show_progress:
                progress_tracker = ProgressTracker()
                progress_tracker.start_progress(len(processed_df), "Generating row IDs")
            
            # Generate hashes with monitoring
            if observer:
                observer.start_hashing()
            
            row_ids = generate_hashes_vectorized(
                processed_df, 
                observer=observer,
                progress_tracker=progress_tracker
            )
            
            if observer:
                observer.end_hashing()
            
            if progress_tracker:
                progress_tracker.finish_progress()
            
            # Add row IDs to DataFrame
            result_df = df.copy()
            result_df[id_column_name] = row_ids
            
            if lineage_tracker:
                lineage_tracker.add_event('hash_generation_completed', {
                    'hash_count': len(row_ids),
                    'id_column_name': id_column_name,
                    'hash_method': 'vectorized_sha256'
                })
                
        except Exception as e:
            error_context = create_error_context(
                'hash_generation', 
                session_id=session_id,
                data_shape=processed_df.shape if 'processed_df' in locals() else None
            )
            handle_processing_error(e, error_context)
            raise RuntimeError(f"Hash generation failed: {str(e)}") from e
        
        # ========================================
        # STAGE 6: QUALITY VALIDATION AND FINALIZATION
        # ========================================
        
        if enable_quality_checks:
            logger.info("Performing quality validation...")
            try:
                # Check for hash collisions
                hash_series = pd.Series(row_ids)
                duplicates = hash_series.duplicated().sum()
                
                if duplicates > 0:
                    collision_msg = (
                        f"CRITICAL: Found {duplicates} hash collisions! "
                        f"This indicates insufficient uniqueness in selected columns."
                    )
                    logger.error(collision_msg)
                    
                    if show_warnings:
                        warnings.warn(collision_msg, RuntimeWarning)
                    
                    if lineage_tracker:
                        lineage_tracker.add_event('quality_check_warning', {
                            'warning_type': 'hash_collisions',
                            'collision_count': duplicates,
                            'severity': 'critical'
                        })
                
                # Validate final data quality
                quality_metrics = {
                    'total_rows': len(result_df),
                    'unique_ids': hash_series.nunique(),
                    'collision_count': duplicates,
                    'uniqueness_ratio': hash_series.nunique() / len(hash_series) if len(hash_series) > 0 else 0,
                    'columns_used': columns,
                    'processing_successful': True
                }
                
                if lineage_tracker:
                    lineage_tracker.add_event('quality_validation_completed', quality_metrics)
                
            except Exception as e:
                error_context = create_error_context(
                    'quality_validation', 
                    session_id=session_id,
                    result_shape=result_df.shape if 'result_df' in locals() else None
                )
                handle_processing_error(e, error_context)
                logger.warning(f"Quality validation failed: {str(e)}")
                # Don't raise - quality checks are supplementary
        
        # ========================================
        # STAGE 7: FINALIZATION AND AUDIT TRAIL
        # ========================================
        
        # Finalize monitoring
        if observer:
            observer.end_session()
        
        # Generate comprehensive audit trail if requested
        if return_audit_trail or enable_enhanced_lineage:
            audit_trail = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'input_metadata': {
                    'dataframe_shape': df.shape,
                    'columns_available': list(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                'processing_metadata': {
                    'columns_selected': columns,
                    'id_column_name': id_column_name,
                    'separator': separator,
                    'uniqueness_threshold': uniqueness_threshold,
                    'processing_parameters': validated_params
                },
                'validation_results': validation_results,
                'output_metadata': {
                    'result_shape': result_df.shape,
                    'unique_ids_generated': len(set(row_ids)),
                    'processing_time_seconds': None  # Would need timing implementation
                },
                'quality_metrics': quality_metrics if enable_quality_checks else None,
                'monitoring_data': observer.get_session_report() if observer else None,
                'lineage_data': lineage_tracker.get_session_summary() if lineage_tracker else None
            }
            
            if return_audit_trail:
                audit_trail['result_dataframe'] = result_df
                logger.info(f"Row ID generation completed successfully (Session: {session_id})")
                return audit_trail
        
        logger.info(f"Row ID generation completed successfully (Session: {session_id})")
        return result_df
        
    except Exception as e:
        # Enhanced error handling
        handle_processing_error(e, operation_context)
        
        # Clean up resources
        if 'observer' in locals() and observer:
            try:
                observer.end_session()
            except:
                pass
        
        if 'lineage_tracker' in locals() and lineage_tracker:
            try:
                lineage_tracker.add_event('processing_error', {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'operation_context': operation_context
                })
            except:
                pass
        
        # Re-raise the exception
        raise


def generate_row_hash(row_data: List[Any], separator: str = '|') -> str:
    """
    Generate SHA-256 hash for a single row of data using the global hashing engine.
    
    Args:
        row_data: List of values from the row
        separator: String separator for concatenating values
        
    Returns:
        64-character hexadecimal hash string
    """
    return _hashing_engine.generate_row_hash(row_data, separator)


def generate_row_ids_vectorized(processed_df: pd.DataFrame, separator: str = '|', 
                               batch_size: Optional[int] = None, show_progress: bool = False,
                               parallel: bool = False, chunk_size: Optional[int] = None) -> pd.Series:
    """
    Generate row IDs for all rows in a DataFrame using optimized vectorized operations.
    
    This function uses the global HashingEngine instance to process multiple rows
    simultaneously with built-in performance monitoring, collision detection, and
    optional parallel processing for large datasets.
    
    Args:
        processed_df: DataFrame with preprocessed data
        separator: String separator for concatenating values
        batch_size: Optional batch size for processing large datasets
        show_progress: Whether to show progress information
        parallel: Enable parallel processing for large datasets (>10k rows)
        chunk_size: Chunk size for parallel processing (default: auto-calculated)
        
    Returns:
        Series of generated row IDs (64-character hexadecimal hashes)
    """
    return _hashing_engine.generate_row_ids_vectorized(
        processed_df, separator, batch_size, show_progress, parallel, chunk_size
    )


def prepare_for_snowflake(df_with_ids: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Prepare DataFrame with row IDs for Snowflake compatibility.
    
    Args:
        df_with_ids: DataFrame with generated row IDs
        table_name: Target Snowflake table name
        
    Returns:
        DataFrame prepared for Snowflake loading
    """
    # TODO: Implement in Task 8 - Snowflake Integration
    logger.debug("prepare_for_snowflake called - implementation pending")
    
    # Placeholder implementation
    return df_with_ids.copy()


def load_to_snowflake(
    df_with_ids: pd.DataFrame,
    connection_params: Dict[str, Any],
    table_name: str,
    if_exists: str = 'append'
) -> Tuple[bool, int]:
    """
    Load DataFrame with row IDs to Snowflake database.
    
    Args:
        df_with_ids: DataFrame with generated row IDs
        connection_params: Snowflake connection parameters
        table_name: Target table name
        if_exists: How to behave if table exists ('append', 'replace', 'fail')
        
    Returns:
        Tuple of (success: bool, rows_loaded: int)
    """
    # TODO: Implement in Task 8 - Snowflake Integration
    logger.debug("load_to_snowflake called - implementation pending")
    
    # Placeholder implementation
    return False, 0


class PerformanceBaseline:
    """Stores historical performance baselines for regression detection."""
    
    def __init__(self, name: str = "default"):
        """
        Initialize performance baseline.
        
        Args:
            name: Name identifier for this baseline
        """
        self.name = name
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.samples = []
        self.statistics = {}
        self.metadata = {}
    
    def add_sample(self, performance_data: Dict[str, Any]) -> None:
        """
        Add a performance sample to the baseline.
        
        Args:
            performance_data: Dictionary containing performance metrics
        """
        sample = {
            'timestamp': datetime.now(),
            'processing_rate': performance_data.get('overall_processing_rate', 0),
            'avg_memory_usage': performance_data.get('avg_memory_usage', 0),
            'peak_memory_usage': performance_data.get('peak_memory_usage', 0),
            'avg_batch_time': performance_data.get('avg_batch_time', 0),
            'collision_rate': performance_data.get('collision_rate', 0),
            'total_rows_processed': performance_data.get('total_rows_processed', 0)
        }
        
        self.samples.append(sample)
        self.updated_at = datetime.now()
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update baseline statistics from samples."""
        if not self.samples:
            return
        
        # Calculate statistics for each metric
        metrics = ['processing_rate', 'avg_memory_usage', 'peak_memory_usage', 
                  'avg_batch_time', 'collision_rate']
        
        for metric in metrics:
            values = [sample[metric] for sample in self.samples if sample[metric] > 0]
            if values:
                self.statistics[metric] = {
                    'mean': mean(values),
                    'std': stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'sample_count': len(values)
                }
    
    def get_baseline_stats(self) -> Dict[str, Any]:
        """Get baseline statistics."""
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'sample_count': len(self.samples),
            'statistics': self.statistics,
            'metadata': self.metadata
        }
    
    def is_ready(self, min_samples: int = 3) -> bool:
        """Check if baseline has enough samples for reliable comparison."""
        return len(self.samples) >= min_samples


class RegressionAlert:
    """Represents a performance regression alert."""
    
    def __init__(self, metric_name: str, current_value: float, baseline_value: float,
                 deviation_percent: float, severity: str, threshold: float):
        """
        Initialize regression alert.
        
        Args:
            metric_name: Name of the performance metric
            current_value: Current metric value
            baseline_value: Baseline metric value
            deviation_percent: Percentage deviation from baseline
            severity: Alert severity level
            threshold: Threshold that was exceeded
        """
        self.alert_id = f"REG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{metric_name}"
        self.metric_name = metric_name
        self.current_value = current_value
        self.baseline_value = baseline_value
        self.deviation_percent = deviation_percent
        self.severity = severity
        self.threshold = threshold
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            'alert_id': self.alert_id,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'deviation_percent': self.deviation_percent,
            'severity': self.severity,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceRegressionDetector:
    """Advanced performance regression detection with configurable thresholds and alerting."""
    
    def __init__(self, baseline_file: str = ".performance_baselines.pkl"):
        """
        Initialize regression detector.
        
        Args:
            baseline_file: File path for storing baselines
        """
        self.baseline_file = baseline_file
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.alerts: List[RegressionAlert] = []
        self.alert_callbacks: List[Callable[[RegressionAlert], None]] = []
        
        # Default regression thresholds (percentage degradation)
        self.thresholds = {
            'processing_rate': {'critical': -30.0, 'warning': -15.0},  # Slower is worse
            'avg_memory_usage': {'critical': 50.0, 'warning': 25.0},   # Higher is worse
            'peak_memory_usage': {'critical': 50.0, 'warning': 25.0},  # Higher is worse
            'avg_batch_time': {'critical': 50.0, 'warning': 25.0},     # Higher is worse
            'collision_rate': {'critical': 100.0, 'warning': 50.0}     # Higher is worse
        }
        
        self.logger = logging.getLogger('regression_detector')
        
        # Load existing baselines
        self._load_baselines()
    
    def set_thresholds(self, metric: str, critical: float, warning: float) -> None:
        """
        Set custom regression thresholds for a metric.
        
        Args:
            metric: Metric name
            critical: Critical threshold percentage
            warning: Warning threshold percentage
        """
        self.thresholds[metric] = {'critical': critical, 'warning': warning}
        self.logger.info(f"Updated thresholds for {metric}: critical={critical}%, warning={warning}%")
    
    def add_alert_callback(self, callback: Callable[[RegressionAlert], None]) -> None:
        """Add callback function for regression alerts."""
        self.alert_callbacks.append(callback)
    
    def establish_baseline(self, name: str, performance_data: Dict[str, Any]) -> bool:
        """
        Establish or update a performance baseline.
        
        Args:
            name: Baseline name
            performance_data: Performance metrics to add to baseline
            
        Returns:
            True if baseline was established/updated successfully
        """
        try:
            if name not in self.baselines:
                self.baselines[name] = PerformanceBaseline(name)
                self.logger.info(f"Created new baseline: {name}")
            
            self.baselines[name].add_sample(performance_data)
            self._save_baselines()
            
            self.logger.info(
                f"Updated baseline '{name}' with new sample "
                f"(total samples: {len(self.baselines[name].samples)})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline '{name}': {e}")
            return False
    
    def detect_regressions(self, current_performance: Dict[str, Any], 
                          baseline_name: str = "default") -> List[RegressionAlert]:
        """
        Detect performance regressions by comparing current metrics to baseline.
        
        Args:
            current_performance: Current performance metrics
            baseline_name: Name of baseline to compare against
            
        Returns:
            List of regression alerts
        """
        if baseline_name not in self.baselines:
            self.logger.warning(f"Baseline '{baseline_name}' not found")
            return []
        
        baseline = self.baselines[baseline_name]
        if not baseline.is_ready():
            self.logger.warning(f"Baseline '{baseline_name}' not ready (needs more samples)")
            return []
        
        regressions = []
        
        # Check each metric for regression
        for metric_name, thresholds in self.thresholds.items():
            current_value = current_performance.get(metric_name)
            if current_value is None:
                continue
            
            baseline_stats = baseline.statistics.get(metric_name)
            if not baseline_stats:
                continue
            
            baseline_mean = baseline_stats['mean']
            if baseline_mean == 0:
                continue
            
            # Calculate percentage deviation
            if metric_name == 'processing_rate':
                # For processing rate, negative deviation is bad
                deviation = ((current_value - baseline_mean) / baseline_mean) * 100
            else:
                # For other metrics, positive deviation is bad
                deviation = ((current_value - baseline_mean) / baseline_mean) * 100
            
            # Check if deviation exceeds thresholds
            severity = None
            threshold_exceeded = None
            
            if metric_name == 'processing_rate':
                # For processing rate, we only care about negative deviations (slower is worse)
                if deviation <= thresholds['critical']:
                    severity = 'CRITICAL'
                    threshold_exceeded = thresholds['critical']
                elif deviation <= thresholds['warning']:
                    severity = 'WARNING'
                    threshold_exceeded = thresholds['warning']
            else:
                # For memory usage, batch time, collision rate - positive deviations are bad
                if deviation >= abs(thresholds['critical']):
                    severity = 'CRITICAL'
                    threshold_exceeded = thresholds['critical']
                elif deviation >= abs(thresholds['warning']):
                    severity = 'WARNING'
                    threshold_exceeded = thresholds['warning']
            
            if severity:
                alert = RegressionAlert(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_mean,
                    deviation_percent=deviation,
                    severity=severity,
                    threshold=threshold_exceeded
                )
                
                regressions.append(alert)
                self.alerts.append(alert)
                
                self.logger.warning(
                    f"PERFORMANCE REGRESSION DETECTED: {alert.alert_id} - "
                    f"{metric_name} deviated {deviation:.1f}% from baseline "
                    f"({current_value:.3f} vs {baseline_mean:.3f}) - Severity: {severity}"
                )
        
        # Trigger alert callbacks
        for alert in regressions:
            self._trigger_alert(alert)
        
        return regressions
    
    def _trigger_alert(self, alert: RegressionAlert) -> None:
        """Trigger alert through registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in regression alert callback: {e}")
    
    def get_regression_report(self) -> Dict[str, Any]:
        """
        Get comprehensive regression detection report.
        
        Returns:
            Dictionary containing regression statistics and recent alerts
        """
        recent_alerts = sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:10]
        
        # Count alerts by severity
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity_counts[alert.severity] += 1
        
        # Count alerts by metric
        metric_counts = defaultdict(int)
        for alert in self.alerts:
            metric_counts[alert.metric_name] += 1
        
        return {
            'total_alerts': len(self.alerts),
            'recent_alerts': [alert.to_dict() for alert in recent_alerts],
            'alerts_by_severity': dict(severity_counts),
            'alerts_by_metric': dict(metric_counts),
            'available_baselines': list(self.baselines.keys()),
            'baseline_summary': {
                name: baseline.get_baseline_stats() 
                for name, baseline in self.baselines.items()
            },
            'current_thresholds': self.thresholds
        }
    
    def export_baselines(self) -> str:
        """Export all baselines to JSON format."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'baselines': {
                name: {
                    'metadata': baseline.get_baseline_stats(),
                    'samples': [
                        {**sample, 'timestamp': sample['timestamp'].isoformat()}
                        for sample in baseline.samples
                    ]
                }
                for name, baseline in self.baselines.items()
            },
            'thresholds': self.thresholds
        }
        return json.dumps(export_data, indent=2)
    
    def _save_baselines(self) -> None:
        """Save baselines to file."""
        try:
            with open(self.baseline_file, 'wb') as f:
                pickle.dump(self.baselines, f)
        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")
    
    def _load_baselines(self) -> None:
        """Load baselines from file."""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'rb') as f:
                    self.baselines = pickle.load(f)
                self.logger.info(f"Loaded {len(self.baselines)} baselines from {self.baseline_file}")
        except Exception as e:
            self.logger.error(f"Failed to load baselines: {e}")
            self.baselines = {}


# Regression alert callback functions
def file_regression_alert_callback(alert: RegressionAlert) -> None:
    """File-based regression alert callback."""
    try:
        with open('performance_regression_alerts.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {alert.to_dict()}\n")
    except Exception as e:
        logger.error(f"Failed to write regression alert to file: {e}")

def console_regression_alert_callback(alert: RegressionAlert) -> None:
    """Console-based regression alert callback."""
    print(f"🚨 PERFORMANCE REGRESSION ALERT: {alert.alert_id}")
    print(f"   Metric: {alert.metric_name}")
    print(f"   Current: {alert.current_value:.3f}")
    print(f"   Baseline: {alert.baseline_value:.3f}")
    print(f"   Deviation: {alert.deviation_percent:.1f}%")
    print(f"   Severity: {alert.severity}")


# ========================================
# INPUT VALIDATION FUNCTIONS - TASK 5.1
# ========================================

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    def __init__(self, message: str, parameter: str = None, value: Any = None):
        self.message = message
        self.parameter = parameter
        self.value = value
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            'error_type': 'ValidationError',
            'message': self.message,
            'parameter': self.parameter,
            'value': str(self.value) if self.value is not None else None,
            'timestamp': datetime.now().isoformat()
        }


def validate_dataframe_input(df: Any) -> None:
    """
    Comprehensive DataFrame validation.
    
    Args:
        df: Input to validate as DataFrame
        
    Raises:
        ValidationError: If input is invalid
    """
    # Check if input is actually a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"Input must be a pandas DataFrame, got {type(df).__name__}",
            parameter="df",
            value=type(df).__name__
        )
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValidationError(
            "Input DataFrame is empty (no rows)",
            parameter="df",
            value="empty"
        )
    
    # Check if DataFrame has columns
    if len(df.columns) == 0:
        raise ValidationError(
            "DataFrame has no columns",
            parameter="df",
            value="no_columns"
        )
    
    # Check for extremely large DataFrames (memory protection)
    if len(df) > 10_000_000:  # 10 million rows
        raise ValidationError(
            f"DataFrame too large for processing: {len(df):,} rows (max: 10,000,000)",
            parameter="df",
            value=len(df)
        )
    
    # Check for extremely wide DataFrames
    if len(df.columns) > 1000:
        raise ValidationError(
            f"DataFrame has too many columns: {len(df.columns):,} (max: 1,000)",
            parameter="df",
            value=len(df.columns)
        )
    
    # Check for duplicate column names
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValidationError(
            f"DataFrame contains duplicate column names: {duplicate_columns}",
            parameter="df.columns",
            value=duplicate_columns
        )


def validate_columns_parameter(columns: Any, df: pd.DataFrame) -> Optional[List[str]]:
    """
    Validate the columns parameter.
    
    Args:
        columns: Columns parameter to validate
        df: DataFrame to validate against
        
    Returns:
        Validated list of column names or None
        
    Raises:
        ValidationError: If columns parameter is invalid
    """
    if columns is None:
        return None
    
    # Convert single string to list
    if isinstance(columns, str):
        columns = [columns]
    
    # Check if it's a list-like object
    if not isinstance(columns, (list, tuple, set)):
        raise ValidationError(
            f"Columns parameter must be a list, tuple, set, or None, got {type(columns).__name__}",
            parameter="columns",
            value=type(columns).__name__
        )
    
    # Convert to list and validate each column
    columns_list = list(columns)
    
    if len(columns_list) == 0:
        raise ValidationError(
            "Columns list cannot be empty",
            parameter="columns",
            value="empty_list"
        )
    
    # Check for duplicate columns in the list
    if len(columns_list) != len(set(columns_list)):
        duplicates = [col for col in set(columns_list) if columns_list.count(col) > 1]
        raise ValidationError(
            f"Duplicate columns specified: {duplicates}",
            parameter="columns",
            value=duplicates
        )
    
    # Validate each column name
    invalid_columns = []
    for col in columns_list:
        if not isinstance(col, str):
            raise ValidationError(
                f"Column names must be strings, got {type(col).__name__} for column: {col}",
                parameter="columns",
                value=col
            )
        
        if col not in df.columns:
            invalid_columns.append(col)
    
    if invalid_columns:
        available_columns = list(df.columns)[:10]  # Show first 10 for brevity
        raise ValidationError(
            f"Columns not found in DataFrame: {invalid_columns}. "
            f"Available columns (first 10): {available_columns}",
            parameter="columns",
            value=invalid_columns
        )
    
    return columns_list


def validate_id_column_name(id_column_name: Any, df: pd.DataFrame) -> str:
    """
    Validate the ID column name parameter.
    
    Args:
        id_column_name: ID column name to validate
        df: DataFrame to check for conflicts
        
    Returns:
        Validated column name
        
    Raises:
        ValidationError: If ID column name is invalid
    """
    if not isinstance(id_column_name, str):
        raise ValidationError(
            f"ID column name must be a string, got {type(id_column_name).__name__}",
            parameter="id_column_name",
            value=id_column_name
        )
    
    if len(id_column_name.strip()) == 0:
        raise ValidationError(
            "ID column name cannot be empty or whitespace",
            parameter="id_column_name",
            value=id_column_name
        )
    
    # Check for invalid characters in column name
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        if char in id_column_name:
            raise ValidationError(
                f"ID column name contains invalid character '{char}': {id_column_name}",
                parameter="id_column_name",
                value=id_column_name
            )
    
    # Check if column already exists in DataFrame
    if id_column_name in df.columns:
        raise ValidationError(
            f"Column '{id_column_name}' already exists in DataFrame. Choose a different name.",
            parameter="id_column_name",
            value=id_column_name
        )
    
    return id_column_name.strip()


def validate_uniqueness_threshold(uniqueness_threshold: Any) -> float:
    """
    Validate the uniqueness threshold parameter.
    
    Args:
        uniqueness_threshold: Threshold value to validate
        
    Returns:
        Validated threshold value
        
    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(uniqueness_threshold, (int, float)):
        raise ValidationError(
            f"Uniqueness threshold must be a number, got {type(uniqueness_threshold).__name__}",
            parameter="uniqueness_threshold",
            value=uniqueness_threshold
        )
    
    threshold = float(uniqueness_threshold)
    
    if not (0.0 <= threshold <= 1.0):
        raise ValidationError(
            f"Uniqueness threshold must be between 0.0 and 1.0, got {threshold}",
            parameter="uniqueness_threshold",
            value=threshold
        )
    
    # Warn about extreme values
    if threshold < 0.1:
        logger = logging.getLogger(__name__)
        logger.warning(f"Very low uniqueness threshold ({threshold}). This may result in poor column selection.")
    elif threshold > 0.99:
        logger = logging.getLogger(__name__)
        logger.warning(f"Very high uniqueness threshold ({threshold}). This may require many columns.")
    
    return threshold


def validate_separator(separator: Any) -> str:
    """
    Validate the separator parameter.
    
    Args:
        separator: Separator string to validate
        
    Returns:
        Validated separator
        
    Raises:
        ValidationError: If separator is invalid
    """
    if not isinstance(separator, str):
        raise ValidationError(
            f"Separator must be a string, got {type(separator).__name__}",
            parameter="separator",
            value=separator
        )
    
    if len(separator) == 0:
        raise ValidationError(
            "Separator cannot be empty",
            parameter="separator",
            value=separator
        )
    
    # Check for problematic separators
    problematic_chars = ['\n', '\r', '\t']
    for char in problematic_chars:
        if char in separator:
            char_name = {'\\n': 'newline', '\\r': 'carriage return', '\\t': 'tab'}
            raise ValidationError(
                f"Separator contains problematic character: {char_name.get(repr(char), repr(char))}",
                parameter="separator",
                value=separator
            )
    
    # Warn about very long separators
    if len(separator) > 10:
        logger = logging.getLogger(__name__)
        logger.warning(f"Very long separator ({len(separator)} characters): '{separator}'")
    
    return separator


def validate_boolean_parameter(value: Any, parameter_name: str) -> bool:
    """
    Validate boolean parameters.
    
    Args:
        value: Value to validate as boolean
        parameter_name: Name of the parameter for error messages
        
    Returns:
        Validated boolean value
        
    Raises:
        ValidationError: If value is not a valid boolean
    """
    if not isinstance(value, bool):
        # Allow string representations of booleans
        if isinstance(value, str):
            if value.lower() in ('true', 'yes', '1', 'on'):
                return True
            elif value.lower() in ('false', 'no', '0', 'off'):
                return False
        
        # Allow numeric representations
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            elif value == 0:
                return False
        
        raise ValidationError(
            f"Parameter '{parameter_name}' must be a boolean, got {type(value).__name__}",
            parameter=parameter_name,
            value=value
        )
    
    return value


def validate_all_parameters(
    df: Any,
    columns: Any = None,
    id_column_name: Any = 'row_id',
    uniqueness_threshold: Any = 0.95,
    separator: Any = '|',
    show_progress: Any = True,
    enable_monitoring: Any = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of all input parameters.
    
    Args:
        df: DataFrame to validate
        columns: Columns parameter to validate
        id_column_name: ID column name to validate
        uniqueness_threshold: Uniqueness threshold to validate
        separator: Separator to validate
        show_progress: Progress display flag to validate
        enable_monitoring: Monitoring flag to validate
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    try:
        # Validate DataFrame first (required for other validations)
        validate_dataframe_input(df)
        
        # Validate all other parameters
        validated_params = {
            'df': df,
            'columns': validate_columns_parameter(columns, df),
            'id_column_name': validate_id_column_name(id_column_name, df),
            'uniqueness_threshold': validate_uniqueness_threshold(uniqueness_threshold),
            'separator': validate_separator(separator),
            'show_progress': validate_boolean_parameter(show_progress, 'show_progress'),
            'enable_monitoring': validate_boolean_parameter(enable_monitoring, 'enable_monitoring')
        }
        
        # Log successful validation
        logger = logging.getLogger(__name__)
        logger.info(
            f"Input validation successful for DataFrame with {len(df):,} rows, "
            f"{len(df.columns)} columns, threshold={validated_params['uniqueness_threshold']}"
        )
        
        return validated_params
        
    except ValidationError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in ValidationError
        raise ValidationError(
            f"Unexpected error during validation: {str(e)}",
            parameter="validation_process",
            value=str(e)
        )


def log_validation_summary(validated_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a validation summary for logging and audit purposes.
    
    Args:
        validated_params: Dictionary of validated parameters
        
    Returns:
        Validation summary dictionary
    """
    df = validated_params['df']
    
    summary = {
        'validation_timestamp': datetime.now().isoformat(),
        'dataframe_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'has_nulls': df.isnull().any().any(),
            'null_counts': df.isnull().sum().to_dict()
        },
        'parameters': {
            'specified_columns': validated_params['columns'],
            'id_column_name': validated_params['id_column_name'],
            'uniqueness_threshold': validated_params['uniqueness_threshold'],
            'separator': validated_params['separator'],
            'show_progress': validated_params['show_progress'],
            'enable_monitoring': validated_params['enable_monitoring']
        },
        'validation_status': 'SUCCESS'
    }
    
    return summary


# ========================================
# PROCESS MONITORING AND SESSION TRACKING - TASK 5.7
# ========================================

class ProcessStage:
    """Represents a processing stage with timing and resource tracking."""
    
    def __init__(self, name: str, stage_id: str = None):
        self.name = name
        self.stage_id = stage_id or f"STAGE_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.start_memory = None
        self.end_memory = None
        self.memory_delta = None
        self.status = "initialized"
        self.metadata = {}
        self.errors = []
        
    def start(self) -> None:
        """Start the processing stage."""
        self.start_time = datetime.now()
        self.start_memory = self._get_memory_usage()
        self.status = "running"
        
    def end(self, success: bool = True) -> None:
        """End the processing stage."""
        self.end_time = datetime.now()
        self.end_memory = self._get_memory_usage()
        
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
            
        if self.start_memory is not None and self.end_memory is not None:
            self.memory_delta = self.end_memory - self.start_memory
            
        self.status = "completed" if success else "failed"
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the stage."""
        self.metadata[key] = value
        
    def add_error(self, error: Exception) -> None:
        """Add an error to the stage."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        })
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            import gc
            import sys
            gc.collect()
            return sum(sys.getsizeof(obj) for obj in gc.get_objects()) / 1024 / 1024
        except:
            return 0.0
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert stage to dictionary format."""
        return {
            'stage_id': self.stage_id,
            'name': self.name,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': self.end_memory,
            'memory_delta_mb': self.memory_delta,
            'metadata': self.metadata,
            'errors': self.errors
        }


class SessionMetrics:
    """Tracks metrics for a processing session."""
    
    def __init__(self):
        self.total_rows_processed = 0
        self.total_processing_time = 0.0
        self.peak_memory_usage = 0.0
        self.total_memory_allocated = 0.0
        self.stage_count = 0
        self.error_count = 0
        self.warning_count = 0
        self.performance_alerts = []
        self.custom_metrics = {}
        
    def update_row_count(self, rows: int) -> None:
        """Update total rows processed."""
        self.total_rows_processed += rows
        
    def update_processing_time(self, time_seconds: float) -> None:
        """Update total processing time."""
        self.total_processing_time += time_seconds
        
    def update_memory_usage(self, memory_mb: float) -> None:
        """Update peak memory usage."""
        self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)
        
    def add_performance_alert(self, alert: str) -> None:
        """Add a performance alert."""
        self.performance_alerts.append({
            'timestamp': datetime.now().isoformat(),
            'alert': alert
        })
        
    def set_custom_metric(self, name: str, value: Any) -> None:
        """Set a custom metric."""
        self.custom_metrics[name] = value
        
    def get_processing_rate(self) -> float:
        """Calculate rows per second processing rate."""
        if self.total_processing_time > 0:
            return self.total_rows_processed / self.total_processing_time
        return 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'total_rows_processed': self.total_rows_processed,
            'total_processing_time_seconds': self.total_processing_time,
            'processing_rate_rows_per_second': self.get_processing_rate(),
            'peak_memory_usage_mb': self.peak_memory_usage,
            'total_memory_allocated_mb': self.total_memory_allocated,
            'stage_count': self.stage_count,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'performance_alerts': self.performance_alerts,
            'custom_metrics': self.custom_metrics
        }


def generate_session_id() -> str:
    """Generate a unique session ID."""
    import uuid
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"SESSION_{timestamp}_{unique_id}"


class ProcessMonitor:
    """Comprehensive process monitoring with session tracking."""
    
    def __init__(self, session_id: str = None, enable_monitoring: bool = True):
        """
        Initialize process monitor.
        
        Args:
            session_id: Unique session identifier
            enable_monitoring: Whether to enable monitoring features
        """
        self.session_id = session_id or generate_session_id()
        self.enable_monitoring = enable_monitoring
        self.session_start_time = datetime.now()
        self.session_end_time = None
        self.current_stage = None
        self.stages = {}
        self.stage_order = []
        self.metrics = SessionMetrics()
        self.logger = logging.getLogger(f'process_monitor.{self.session_id}')
        self.alerts = []
        
        # Resource monitoring thresholds
        self.memory_threshold_mb = 1000  # 1 GB
        self.processing_time_threshold = 300  # 5 minutes
        self.stage_time_threshold = 60  # 1 minute per stage
        
        if self.enable_monitoring:
            self.logger.info(f"Process monitoring session started: {self.session_id}")
            
        # Initialize session errors list
        self.session_errors = []
        
        # Create logger
        self.logger = logging.getLogger(f"process_monitor.{self.session_id}")
        
        # Initialize metrics
        self.metrics = SessionMetrics()
        
    def set_thresholds(self, memory_mb: float = None, processing_time: float = None, 
                      stage_time: float = None) -> None:
        """Set monitoring thresholds."""
        if memory_mb is not None:
            self.memory_threshold_mb = memory_mb
        if processing_time is not None:
            self.processing_time_threshold = processing_time
        if stage_time is not None:
            self.stage_time_threshold = stage_time
            
    def track_session(self):
        """Context manager for tracking the entire session."""
        return SessionTracker(self)
        
    def track_stage(self, stage_name: str):
        """Context manager for tracking a processing stage."""
        return StageTracker(self, stage_name)
        
    def start_stage(self, stage_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start tracking a processing stage.
        
        Args:
            stage_name: Name of the processing stage
            metadata: Optional metadata for the stage
            
        Returns:
            Stage ID
        """
        if not self.enable_monitoring:
            return "monitoring_disabled"
            
        stage = ProcessStage(stage_name)
        stage.start()
        
        if metadata:
            for key, value in metadata.items():
                stage.add_metadata(key, value)
                
        self.stages[stage.stage_id] = stage
        self.stage_order.append(stage.stage_id)
        self.current_stage = stage
        self.metrics.stage_count += 1
        
        self.logger.info(f"Started stage '{stage_name}' with ID: {stage.stage_id}")
        
        return stage.stage_id
        
    def end_stage(self, stage_id: str, success: bool = True, metadata: Dict[str, Any] = None) -> None:
        """
        End tracking a processing stage.
        
        Args:
            stage_id: Stage ID to end
            success: Whether the stage completed successfully
            metadata: Optional metadata to add
        """
        if not self.enable_monitoring or stage_id not in self.stages:
            return
            
        stage = self.stages[stage_id]
        stage.end(success)
        
        if metadata:
            for key, value in metadata.items():
                stage.add_metadata(key, value)
                
        # Update session metrics
        if stage.duration:
            self.metrics.update_processing_time(stage.duration)
            
        if stage.end_memory:
            self.metrics.update_memory_usage(stage.end_memory)
            
        if not success:
            self.metrics.error_count += 1
            
        # Check thresholds
        self._check_stage_thresholds(stage)
        
        self.logger.info(
            f"Ended stage '{stage.name}' (ID: {stage_id}) - "
            f"Duration: {stage.duration:.3f}s, Success: {success}"
        )
        
        if stage == self.current_stage:
            self.current_stage = None
            
    def add_stage_error(self, stage_id: str, error: Exception) -> None:
        """Add an error to a specific stage."""
        if not self.enable_monitoring or stage_id not in self.stages:
            return
            
        stage = self.stages[stage_id]
        stage.add_error(error)
        self.metrics.error_count += 1
        
        self.logger.error(f"Error in stage '{stage.name}': {str(error)}")
        
    def update_row_progress(self, rows_processed: int) -> None:
        """Update the number of rows processed."""
        if not self.enable_monitoring:
            return
            
        self.metrics.update_row_count(rows_processed)
        
        if self.current_stage:
            self.current_stage.add_metadata('rows_processed', rows_processed)
            
    def add_alert(self, alert_message: str, severity: str = "INFO") -> None:
        """Add a monitoring alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'message': alert_message,
            'severity': severity,
            'current_stage': self.current_stage.name if self.current_stage else None
        }
        
        self.alerts.append(alert)
        self.metrics.add_performance_alert(alert_message)
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"ALERT: {alert_message}")
        
    def end_session(self, success: bool = True) -> None:
        """End the monitoring session."""
        if not self.enable_monitoring:
            return
            
        self.session_end_time = datetime.now()
        
        # End any active stage
        if self.current_stage:
            self.end_stage(self.current_stage.stage_id, success=success)
            
        session_duration = (self.session_end_time - self.session_start_time).total_seconds()
        
        self.logger.info(
            f"Session ended: {self.session_id} - "
            f"Duration: {session_duration:.3f}s, "
            f"Stages: {len(self.stages)}, "
            f"Rows: {self.metrics.total_rows_processed:,}, "
            f"Success: {success}"
        )
        
    def _check_stage_thresholds(self, stage: ProcessStage) -> None:
        """Check if stage exceeded thresholds."""
        if stage.duration and stage.duration > self.stage_time_threshold:
            self.add_alert(
                f"Stage '{stage.name}' exceeded time threshold: "
                f"{stage.duration:.1f}s > {self.stage_time_threshold}s",
                severity="WARNING"
            )
            
        if stage.memory_delta and stage.memory_delta > self.memory_threshold_mb:
            self.add_alert(
                f"Stage '{stage.name}' exceeded memory threshold: "
                f"{stage.memory_delta:.1f}MB > {self.memory_threshold_mb}MB",
                severity="WARNING"
            )
            
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        session_duration = None
        if self.session_end_time:
            session_duration = (self.session_end_time - self.session_start_time).total_seconds()
        elif self.session_start_time:
            session_duration = (datetime.now() - self.session_start_time).total_seconds()
            
        return {
            'session_id': self.session_id,
            'session_start_time': self.session_start_time.isoformat(),
            'session_end_time': self.session_end_time.isoformat() if self.session_end_time else None,
            'session_duration_seconds': session_duration,
            'monitoring_enabled': self.enable_monitoring,
            'stages': {stage_id: stage.to_dict() for stage_id, stage in self.stages.items()},
            'stage_order': self.stage_order,
            'metrics': self.metrics.to_dict(),
            'alerts': self.alerts,
            'thresholds': {
                'memory_threshold_mb': self.memory_threshold_mb,
                'processing_time_threshold': self.processing_time_threshold,
                'stage_time_threshold': self.stage_time_threshold
            }
        }
        
    def export_session_data(self, format: str = 'json') -> str:
        """Export session data in specified format."""
        summary = self.get_session_summary()
        
        if format.lower() == 'json':
            return json.dumps(summary, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def add_session_error(self, error: Exception) -> None:
        """Add an error to the session."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'stage_id': None
        }
        self.session_errors.append(error_info)
        logger.error(f"Session error recorded: {error_info}")

    def end_session(self, success: bool = True) -> None:
        """End the monitoring session."""
        if not self.enable_monitoring:
            return
            
        self.session_end_time = datetime.now()
        
        # End any active stage
        if self.current_stage:
            self.end_stage(self.current_stage.stage_id, success=success)
            
        session_duration = (self.session_end_time - self.session_start_time).total_seconds()
        
        self.logger.info(
            f"Session ended: {self.session_id} - "
            f"Duration: {session_duration:.3f}s, "
            f"Stages: {len(self.stages)}, "
            f"Rows: {self.metrics.total_rows_processed:,}, "
            f"Success: {success}"
        )


class SessionTracker:
    """Context manager for session tracking."""
    
    def __init__(self, monitor: ProcessMonitor):
        self.monitor = monitor
        
    def __enter__(self) -> ProcessMonitor:
        return self.monitor
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        success = exc_type is None
        self.monitor.end_session(success=success)
        
        if exc_type:
            self.monitor.add_alert(
                f"Session ended with exception: {exc_type.__name__}: {exc_val}",
                severity="ERROR"
            )


class StageTracker:
    """Context manager for stage tracking."""
    
    def __init__(self, monitor: ProcessMonitor, stage_name: str):
        self.monitor = monitor
        self.stage_name = stage_name
        self.stage_id = None
        
    def __enter__(self) -> str:
        self.stage_id = self.monitor.start_stage(self.stage_name)
        return self.stage_id
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        success = exc_type is None
        
        if exc_type:
            error = exc_val if exc_val else Exception(f"{exc_type.__name__}")
            self.monitor.add_stage_error(self.stage_id, error)
            
        self.monitor.end_stage(self.stage_id, success=success)


# ========================================
# PROGRESS INDICATORS AND USER EXPERIENCE - TASK 5.8
# ========================================

class ProgressTracker:
    """Advanced progress tracking with ETA estimation and user feedback."""
    
    def __init__(self, total_work: int, description: str = "Processing", show_progress: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            total_work: Total units of work to complete
            description: Description of the work being done
            show_progress: Whether to display progress indicators
        """
        self.total_work = total_work
        self.description = description
        self.show_progress = show_progress
        self.current_progress = 0
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.progress_bar = None
        self.status_messages = []
        self.warnings = []
        
        if self.show_progress:
            try:
                from tqdm import tqdm
                self.progress_bar = tqdm(
                    total=total_work,
                    desc=description,
                    unit='rows',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    colour='green' if total_work > 1000 else 'blue'
                )
                self.has_tqdm = True
            except ImportError:
                self.progress_bar = None
                self.has_tqdm = False
                logger.warning("tqdm not available, progress will be shown as periodic updates")
                
    def update(self, increment: int = 1, message: str = None) -> None:
        """Update progress and optionally display a status message."""
        self.current_progress += increment
        self.last_update_time = datetime.now()
        
        if message:
            self.status_messages.append({
                'timestamp': self.last_update_time.isoformat(),
                'message': message,
                'progress': self.current_progress
            })
            
        if self.show_progress:
            if self.progress_bar:
                self.progress_bar.update(increment)
                if message:
                    self.progress_bar.set_postfix_str(message[:50])
            else:
                # Fallback progress display
                percentage = (self.current_progress / self.total_work) * 100
                eta = self.calculate_eta()
                eta_str = f"ETA: {eta}" if eta else ""
                print(f"{self.description}: {self.current_progress}/{self.total_work} ({percentage:.1f}%) {eta_str}")
                if message:
                    print(f"  Status: {message}")
                    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'warning': warning,
            'progress': self.current_progress
        }
        self.warnings.append(warning_entry)
        
        if self.show_progress:
            if self.progress_bar:
                self.progress_bar.write(f"⚠️  WARNING: {warning}")
            else:
                print(f"⚠️  WARNING: {warning}")
                
    def calculate_eta(self) -> Optional[str]:
        """Calculate estimated time to completion."""
        if self.current_progress == 0:
            return None
            
        elapsed = (self.last_update_time - self.start_time).total_seconds()
        if elapsed == 0:
            return None
            
        rate = self.current_progress / elapsed
        if rate == 0:
            return None
            
        remaining_work = self.total_work - self.current_progress
        eta_seconds = remaining_work / rate
        
        # Format ETA as human readable
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"
            
    def finish(self, success: bool = True, final_message: str = None) -> None:
        """Complete the progress tracking."""
        if self.progress_bar:
            if final_message:
                self.progress_bar.set_postfix_str(final_message)
            self.progress_bar.close()
        elif self.show_progress:
            status = "✅ COMPLETED" if success else "❌ FAILED"
            total_time = (datetime.now() - self.start_time).total_seconds()
            print(f"{status}: {self.description} - {total_time:.1f}s")
            if final_message:
                print(f"  {final_message}")
                
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        return {
            'description': self.description,
            'total_work': self.total_work,
            'completed_work': self.current_progress,
            'completion_percentage': (self.current_progress / self.total_work) * 100,
            'elapsed_time_seconds': total_time,
            'average_rate': self.current_progress / total_time if total_time > 0 else 0,
            'status_messages': self.status_messages,
            'warnings': self.warnings,
            'start_time': self.start_time.isoformat(),
            'last_update_time': self.last_update_time.isoformat()
        }


class DataQualityMonitor:
    """Monitor data quality issues during processing."""
    
    def __init__(self, enable_warnings: bool = True):
        self.enable_warnings = enable_warnings
        self.quality_issues = []
        self.warning_thresholds = {
            'high_null_ratio': 0.3,  # 30% or more nulls
            'low_uniqueness': 0.1,   # 10% or less unique values
            'suspicious_patterns': True
        }
        
    def check_column_quality(self, df: pd.DataFrame, column: str) -> List[str]:
        """Check quality issues for a specific column."""
        issues = []
        
        # Check null ratio
        null_ratio = df[column].isnull().sum() / len(df)
        if null_ratio > self.warning_thresholds['high_null_ratio']:
            issues.append(f"High null ratio ({null_ratio:.1%}) in column '{column}'")
            
        # Check uniqueness
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio < self.warning_thresholds['low_uniqueness']:
            issues.append(f"Low uniqueness ({unique_ratio:.1%}) in column '{column}'")
            
        # Check for suspicious patterns
        if self.warning_thresholds['suspicious_patterns']:
            if df[column].dtype == 'object':
                # Check for potential encoding issues
                sample_values = df[column].dropna().astype(str).head(100)
                encoding_issues = sum(1 for val in sample_values if any(ord(char) > 127 for char in val))
                if encoding_issues > 10:
                    issues.append(f"Potential encoding issues in column '{column}'")
                    
        self.quality_issues.extend(issues)
        return issues
        
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of quality issues."""
        return {
            'total_issues': len(self.quality_issues),
            'issues': self.quality_issues,
            'thresholds': self.warning_thresholds,
            'timestamp': datetime.now().isoformat()
        }


# ========================================
# CONFIGURATION LOGGING AND DATA FINGERPRINTING - TASK 5.9
# ========================================

def create_data_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive fingerprint of the input data for audit trails.
    
    Args:
        df: Input DataFrame to fingerprint
        
    Returns:
        Dictionary containing data fingerprint information
    """
    import hashlib
    
    # Basic DataFrame structure
    structure_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'index_type': str(type(df.index).__name__),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Data content summary
    content_summary = {}
    for col in df.columns:
        col_data = df[col]
        content_summary[col] = {
            'dtype': str(col_data.dtype),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique(),
            'sample_values': col_data.dropna().head(3).tolist() if not col_data.empty else []
        }
    
    # Create content hash (sample-based for large datasets)
    sample_size = min(1000, len(df))
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
        
    # Hash the sample data
    content_string = sample_df.to_string()
    content_hash = hashlib.sha256(content_string.encode()).hexdigest()[:16]
    
    # Structure hash
    structure_string = json.dumps(structure_info, sort_keys=True)
    structure_hash = hashlib.sha256(structure_string.encode()).hexdigest()[:16]
    
    return {
        'timestamp': datetime.now().isoformat(),
        'structure_fingerprint': structure_hash,
        'content_fingerprint': content_hash,
        'structure_info': structure_info,
        'content_summary': content_summary,
        'sample_size': sample_size,
        'total_rows': len(df)
    }


def log_configuration(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    uniqueness_threshold: float,
    separator: str,
    **additional_params
) -> Dict[str, Any]:
    """
    Log comprehensive configuration information for audit trails.
    
    Args:
        df: Input DataFrame
        columns: Selected columns
        uniqueness_threshold: Uniqueness threshold
        separator: Row separator
        **additional_params: Additional parameters to log
        
    Returns:
        Configuration dictionary
    """
    config = {
        'function_call': {
            'function_name': 'generate_unique_row_ids',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'columns': columns,
                'uniqueness_threshold': uniqueness_threshold,
                'separator': separator,
                **additional_params
            }
        },
        'input_data': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'has_nulls': df.isnull().any().any(),
            'total_null_count': df.isnull().sum().sum()
        },
        'execution_environment': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'pandas_version': pd.__version__,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        },
        'processing_metadata': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'selected_columns_count': len(columns) if columns else 0,
            'estimated_memory_impact': df.memory_usage(deep=True).sum() / 1024 / 1024 * 1.5  # Estimate
        }
    }
    
    return config


def create_audit_trail(
    session_id: str,
    config: Dict[str, Any],
    data_fingerprint: Dict[str, Any],
    processing_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create comprehensive audit trail for the row ID generation process.
    
    Args:
        session_id: Unique session identifier
        config: Configuration information
        data_fingerprint: Data fingerprint
        processing_summary: Processing summary from monitor
        
    Returns:
        Complete audit trail dictionary
    """
    audit_trail = {
        'audit_metadata': {
            'session_id': session_id,
            'audit_version': '1.0',
            'created_timestamp': datetime.now().isoformat(),
            'audit_id': f"AUDIT_{session_id}"
        },
        'configuration': config,
        'data_fingerprint': data_fingerprint,
        'processing_summary': processing_summary,
        'data_lineage': {
            'input_source': 'pandas_dataframe',
            'transformations_applied': [
                'input_validation',
                'column_selection',
                'data_preprocessing',
                'hash_generation'
            ],
            'output_format': 'pandas_dataframe_with_row_ids'
        },
        'quality_assurance': {
            'validation_passed': True,  # Will be updated based on actual validation
            'warnings_generated': [],
            'errors_encountered': [],
            'quality_score': None  # Will be calculated
        },
        'traceability': {
            'reproducible': True,
            'deterministic_output': True,
            'version_info': {
                'library_version': '1.0.0',  # Replace with actual version
                'algorithm_version': 'SHA256_v1'
            }
        }
    }
    
    return audit_trail


# ========================================
# COLUMN SELECTION INTEGRATION - TASK 5.2
# ========================================

def integrate_column_selection(
    df: pd.DataFrame,
    manual_columns: Optional[List[str]],
    uniqueness_threshold: float,
    monitor: Optional[ProcessMonitor] = None
) -> Dict[str, Any]:
    """
    Integrate column selection functionality with monitoring and validation.
    
    Args:
        df: Input DataFrame
        manual_columns: Manually specified columns
        uniqueness_threshold: Uniqueness threshold for automatic selection
        monitor: Optional process monitor for tracking
        
    Returns:
        Dictionary containing selected columns and selection metadata
    """
    from .utils import select_columns_for_hashing
    
    # Start monitoring if available
    stage_id = None
    if monitor:
        stage_id = monitor.start_stage("Column Selection", {
            'manual_columns': manual_columns,
            'uniqueness_threshold': uniqueness_threshold,
            'total_columns': len(df.columns)
        })
    
    try:
        # Perform column selection
        if manual_columns:
            # Validate manual columns exist
            missing_columns = set(manual_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Manual columns not found in DataFrame: {list(missing_columns)}")
            
            selected_columns = manual_columns
            selection_method = "manual"
            selection_metadata = {
                'method': 'manual',
                'columns_specified': len(manual_columns),
                'columns_validated': len(manual_columns),
                'validation_success': True
            }
        else:
            # Use automatic column selection
            selected_columns = select_columns_for_hashing(
                df=df,
                manual_columns=None,
                uniqueness_threshold=uniqueness_threshold,
                include_email=True
            )
            selection_method = "automatic"
            selection_metadata = {
                'method': 'automatic',
                'uniqueness_threshold': uniqueness_threshold,
                'total_columns_analyzed': len(df.columns),
                'columns_selected': len(selected_columns),
                'selection_ratio': len(selected_columns) / len(df.columns)
            }
        
        # Calculate selection quality metrics
        quality_metrics = {}
        for col in selected_columns:
            completeness = 1 - (df[col].isnull().sum() / len(df))
            uniqueness = df[col].nunique() / len(df)
            quality_metrics[col] = {
                'completeness': completeness,
                'uniqueness': uniqueness,
                'quality_score': (completeness + uniqueness) / 2
            }
        
        result = {
            'selected_columns': selected_columns,
            'selection_method': selection_method,
            'selection_metadata': selection_metadata,
            'quality_metrics': quality_metrics,
            'overall_quality_score': sum(m['quality_score'] for m in quality_metrics.values()) / len(quality_metrics),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update monitoring
        if monitor and stage_id:
            monitor.end_stage(stage_id, success=True, metadata={
                'columns_selected': len(selected_columns),
                'selection_method': selection_method,
                'quality_score': result['overall_quality_score']
            })
            
        return result
        
    except Exception as e:
        # Handle errors and update monitoring
        if monitor and stage_id:
            monitor.add_stage_error(stage_id, e)
            monitor.end_stage(stage_id, success=False)
        raise


def validate_column_selection_result(
    selection_result: Dict[str, Any],
    df: pd.DataFrame,
    minimum_quality_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Validate the column selection result and provide recommendations.
    
    Args:
        selection_result: Result from column selection
        df: Original DataFrame
        minimum_quality_threshold: Minimum acceptable quality score
        
    Returns:
        Validation result with recommendations
    """
    selected_columns = selection_result['selected_columns']
    quality_metrics = selection_result['quality_metrics']
    overall_quality = selection_result['overall_quality_score']
    
    validation_result = {
        'validation_passed': True,
        'quality_issues': [],
        'recommendations': [],
        'warnings': []
    }
    
    # Check overall quality
    if overall_quality < minimum_quality_threshold:
        validation_result['validation_passed'] = False
        validation_result['quality_issues'].append(
            f"Overall quality score ({overall_quality:.2f}) below threshold ({minimum_quality_threshold})"
        )
    
    # Check individual columns
    low_quality_columns = []
    for col, metrics in quality_metrics.items():
        if metrics['quality_score'] < minimum_quality_threshold:
            low_quality_columns.append(col)
            validation_result['quality_issues'].append(
                f"Column '{col}' has low quality score: {metrics['quality_score']:.2f}"
            )
    
    # Generate recommendations
    if low_quality_columns:
        validation_result['recommendations'].append(
            f"Consider excluding low-quality columns: {low_quality_columns}"
        )
    
    if len(selected_columns) < 2:
        validation_result['warnings'].append(
            "Only one column selected - consider adding more columns for better uniqueness"
        )
    
    if len(selected_columns) > 10:
        validation_result['warnings'].append(
            "Many columns selected - this may impact performance"
        )
    
    return validation_result


# ========================================
# DATA PREPROCESSING INTEGRATION - TASK 5.3
# ========================================

def integrate_data_preprocessing(
    df: pd.DataFrame,
    selected_columns: List[str],
    monitor: Optional[ProcessMonitor] = None,
    quality_monitor: Optional[DataQualityMonitor] = None
) -> Dict[str, Any]:
    """
    Integrate data preprocessing with monitoring and quality checks.
    
    Args:
        df: Input DataFrame
        selected_columns: Columns to preprocess
        monitor: Optional process monitor
        quality_monitor: Optional data quality monitor
        
    Returns:
        Dictionary containing preprocessed data and processing metadata
    """
    from .utils import prepare_data_for_hashing
    
    # Start monitoring
    stage_id = None
    if monitor:
        stage_id = monitor.start_stage("Data Preprocessing", {
            'columns_to_process': len(selected_columns),
            'total_rows': len(df)
        })
    
    try:
        # Check data quality before preprocessing
        quality_issues = []
        if quality_monitor:
            for col in selected_columns:
                issues = quality_monitor.check_column_quality(df, col)
                quality_issues.extend(issues)
        
        # Perform preprocessing
        preprocessed_df = prepare_data_for_hashing(df, selected_columns)
        
        # Calculate preprocessing metrics
        original_memory = df[selected_columns].memory_usage(deep=True).sum()
        processed_memory = preprocessed_df.memory_usage(deep=True).sum()
        memory_change = processed_memory - original_memory
        
        # Analyze preprocessing impact
        preprocessing_stats = {}
        for col in selected_columns:
            original_col = df[col]
            processed_col = preprocessed_df[col]
            
            preprocessing_stats[col] = {
                'original_dtype': str(original_col.dtype),
                'processed_dtype': str(processed_col.dtype),
                'original_nulls': original_col.isnull().sum(),
                'processed_nulls': processed_col.isnull().sum(),
                'original_unique': original_col.nunique(),
                'processed_unique': processed_col.nunique(),
                'data_changed': not original_col.equals(processed_col)
            }
        
        result = {
            'preprocessed_data': preprocessed_df,
            'preprocessing_stats': preprocessing_stats,
            'memory_impact': {
                'original_memory_mb': original_memory / 1024 / 1024,
                'processed_memory_mb': processed_memory / 1024 / 1024,
                'memory_change_mb': memory_change / 1024 / 1024,
                'memory_change_percent': (memory_change / original_memory * 100) if original_memory > 0 else 0
            },
            'quality_issues': quality_issues,
            'processing_summary': {
                'rows_processed': len(preprocessed_df),
                'columns_processed': len(selected_columns),
                'successful': True,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Update monitoring
        if monitor and stage_id:
            monitor.end_stage(stage_id, success=True, metadata={
                'rows_processed': len(preprocessed_df),
                'memory_change_mb': memory_change / 1024 / 1024,
                'quality_issues': len(quality_issues)
            })
            
        return result
        
    except Exception as e:
        if monitor and stage_id:
            monitor.add_stage_error(stage_id, e)
            monitor.end_stage(stage_id, success=False)
        raise


# ========================================
# ERROR HANDLING AND DATA QUALITY CHECKS - TASK 5.5
# ========================================

class RowIDGenerationError(Exception):
    """Base exception for row ID generation errors."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class DataValidationError(RowIDGenerationError):
    """Exception for data validation failures."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)


class ColumnSelectionError(RowIDGenerationError):
    """Exception for column selection failures."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="COLUMN_SELECTION_ERROR", **kwargs)


class PreprocessingError(RowIDGenerationError):
    """Exception for data preprocessing failures."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PREPROCESSING_ERROR", **kwargs)


class HashGenerationError(RowIDGenerationError):
    """Exception for hash generation failures."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="HASH_GENERATION_ERROR", **kwargs)


def comprehensive_error_handler(func):
    """Decorator for comprehensive error handling with context preservation."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RowIDGenerationError:
            # Re-raise our custom errors as-is
            raise
        except pd.errors.EmptyDataError as e:
            raise DataValidationError(f"Empty DataFrame provided: {str(e)}")
        except KeyError as e:
            raise ColumnSelectionError(f"Column not found: {str(e)}")
        except MemoryError as e:
            raise HashGenerationError(f"Insufficient memory for processing: {str(e)}")
        except Exception as e:
            # Wrap unexpected errors
            raise RowIDGenerationError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={'function': func.__name__, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )
    return wrapper


def validate_data_quality_requirements(
    df: pd.DataFrame,
    selected_columns: List[str],
    requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate data quality against specified requirements.
    
    Args:
        df: Input DataFrame
        selected_columns: Selected columns for processing
        requirements: Quality requirements dictionary
        
    Returns:
        Validation result with pass/fail status and detailed findings
    """
    default_requirements = {
        'max_null_ratio': 0.5,      # Maximum 50% nulls per column
        'min_uniqueness': 0.1,      # Minimum 10% unique values per column
        'min_row_count': 1,         # Minimum 1 row
        'max_memory_mb': 1024,      # Maximum 1GB memory usage
        'require_non_empty': True   # Require non-empty data
    }
    
    requirements = {**default_requirements, **(requirements or {})}
    
    validation_result = {
        'passed': True,
        'failures': [],
        'warnings': [],
        'metrics': {},
        'requirements': requirements
    }
    
    # Check basic DataFrame requirements
    if requirements['require_non_empty'] and df.empty:
        validation_result['passed'] = False
        validation_result['failures'].append("DataFrame is empty")
        return validation_result
    
    if len(df) < requirements['min_row_count']:
        validation_result['passed'] = False
        validation_result['failures'].append(f"Row count ({len(df)}) below minimum ({requirements['min_row_count']})")
    
    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    validation_result['metrics']['memory_usage_mb'] = memory_mb
    if memory_mb > requirements['max_memory_mb']:
        validation_result['passed'] = False
        validation_result['failures'].append(f"Memory usage ({memory_mb:.1f}MB) exceeds maximum ({requirements['max_memory_mb']}MB)")
    
    # Check column-specific requirements
    for col in selected_columns:
        col_metrics = {}
        
        # Null ratio check
        null_ratio = df[col].isnull().sum() / len(df)
        col_metrics['null_ratio'] = null_ratio
        if null_ratio > requirements['max_null_ratio']:
            validation_result['passed'] = False
            validation_result['failures'].append(f"Column '{col}' null ratio ({null_ratio:.2%}) exceeds maximum ({requirements['max_null_ratio']:.2%})")
        
        # Uniqueness check
        uniqueness = df[col].nunique() / len(df)
        col_metrics['uniqueness'] = uniqueness
        if uniqueness < requirements['min_uniqueness']:
            validation_result['warnings'].append(f"Column '{col}' uniqueness ({uniqueness:.2%}) below recommended minimum ({requirements['min_uniqueness']:.2%})")
        
        validation_result['metrics'][col] = col_metrics
    
    return validation_result


# ========================================
# MAIN FUNCTION ARCHITECTURE - TASK 5.4
# ========================================

@comprehensive_error_handler
def generate_unique_row_ids(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    id_column_name: str = 'row_id',
    uniqueness_threshold: float = 0.95,
    separator: str = '|',
    enable_monitoring: bool = True,
    enable_quality_checks: bool = True,
    show_progress: bool = True,
    show_warnings: bool = True,
    enable_enhanced_lineage: bool = False,
    return_audit_trail: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Generate unique row IDs for a pandas DataFrame using SHA-256 hashing.
    
    This is the main entry point for the row ID generation system, providing
    comprehensive validation, monitoring, and error handling with enhanced
    data lineage tracking and audit trail generation.
    
    Args:
        df: Input pandas DataFrame
        columns: Optional list of columns to use for hashing
        id_column_name: Name for the new ID column
        uniqueness_threshold: Minimum uniqueness ratio for automatic column selection
        separator: String to separate column values before hashing
        enable_monitoring: Enable process monitoring and resource tracking
        enable_quality_checks: Enable data quality validation
        show_warnings: Display warnings for data quality issues
        return_audit_trail: Return comprehensive audit trail with results
        enable_enhanced_lineage: Enable enhanced data lineage tracking (new feature)
        **kwargs: Additional parameters for customization
        
    Returns:
        DataFrame with added row ID column, or dictionary with results and audit trail
        
    Raises:
        DataValidationError: If input data validation fails
        ColumnSelectionError: If column selection fails
        PreprocessingError: If data preprocessing fails
        HashGenerationError: If hash generation fails
        RowIDGenerationError: For other unexpected errors
    """
    # Initialize session and monitoring
    session_id = generate_session_id() if enable_monitoring else None
    monitor = ProcessMonitor(session_id=session_id) if enable_monitoring else None
    progress_tracker = None
    quality_monitor = DataQualityMonitor(enable_warnings=show_warnings) if enable_quality_checks else None
    audit_trail_components = {}
    
    # Initialize enhanced lineage tracking
    lineage_tracker = None
    if enable_enhanced_lineage:
        lineage_tracker = DataLineageTracker(session_id=session_id)
        lineage_tracker.add_lineage_event(
            event_type="FUNCTION_START",
            description="Row ID generation process initiated",
            data_shape=df.shape,
            metadata={
                'parameters': {
                    'columns': columns,
                    'id_column_name': id_column_name,
                    'uniqueness_threshold': uniqueness_threshold,
                    'separator': separator,
                    'enable_monitoring': enable_monitoring,
                    'show_progress': show_progress,
                    'enable_quality_checks': enable_quality_checks
                }
            }
        )
    
    try:
        # Start session tracking
        if monitor:
            monitor.start_session({
                'function': 'generate_unique_row_ids',
                'input_shape': df.shape,
                'parameters': {
                    'columns': columns,
                    'id_column_name': id_column_name,
                    'uniqueness_threshold': uniqueness_threshold,
                    'separator': separator,
                    'enable_monitoring': enable_monitoring,
                    'show_progress': show_progress,
                    'enable_quality_checks': enable_quality_checks,
                    'enable_enhanced_lineage': enable_enhanced_lineage
                }
            })
        
        # Create progress tracker
        progress_tracker = None
        if show_progress:
            total_steps = 6  # Validation, Column Selection, Preprocessing, Hashing, Quality Check, Finalization
            progress_tracker = ProgressTracker(
                total_work=total_steps,
                description="Generating Row IDs",
                show_progress=show_progress
            )
        
        # Step 1: Input Validation
        if progress_tracker:
            progress_tracker.update(1, "Validating input data")
        
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="VALIDATION_START",
                description="Input validation phase started",
                data_shape=df.shape
            )
        
        validate_all_inputs(df, columns, id_column_name, uniqueness_threshold, separator, **kwargs)
        
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="VALIDATION_COMPLETE",
                description="Input validation completed successfully",
                data_shape=df.shape
            )
            # Create initial data checkpoint
            checkpoint_id = lineage_tracker.create_data_checkpoint(
                checkpoint_name="input_data",
                df=df,
                stage="validation",
                additional_metadata={'validation_status': 'passed'}
            )
        
        # Create data fingerprint for audit trail
        if return_audit_trail:
            audit_trail_components['data_fingerprint'] = create_data_fingerprint(df)
            audit_trail_components['config'] = log_configuration(df, columns, uniqueness_threshold, separator, **kwargs)
        
        # Step 2: Column Selection
        if progress_tracker:
            progress_tracker.update(1, "Selecting columns for hashing")
        
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="COLUMN_SELECTION_START",
                description="Column selection phase started",
                data_shape=df.shape
            )
        
        column_selection_result = integrate_column_selection(
            df=df,
            manual_columns=columns,
            uniqueness_threshold=uniqueness_threshold,
            monitor=monitor
        )
        selected_columns = column_selection_result['selected_columns']
        
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="COLUMN_SELECTION_COMPLETE",
                description=f"Selected {len(selected_columns)} columns for hashing",
                data_shape=df.shape,
                metadata={'selected_columns': selected_columns, 'selection_method': 'manual' if columns else 'automatic'}
            )
            # Add quality assessment for column selection
            lineage_tracker.add_quality_assessment(
                assessment_type="column_selection_quality",
                results=column_selection_result,
                stage="column_selection"
            )
        
        # Validate column selection quality
        if enable_quality_checks:
            column_validation = validate_column_selection_result(column_selection_result, df)
            if not column_validation['validation_passed'] and show_warnings:
                for issue in column_validation['quality_issues']:
                    if progress_tracker:
                        progress_tracker.add_warning(issue)
        
        # Step 3: Data Preprocessing
        if progress_tracker:
            progress_tracker.update(1, "Preprocessing data")
        
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="PREPROCESSING_START",
                description="Data preprocessing phase started",
                data_shape=df.shape
            )
        
        preprocessing_result = integrate_data_preprocessing(
            df=df,
            selected_columns=selected_columns,
            monitor=monitor,
            quality_monitor=quality_monitor
        )
        preprocessed_df = preprocessing_result['preprocessed_data']
        
        if lineage_tracker:
            # Track preprocessing transformation
            lineage_tracker.add_transformation(
                transformation_name="data_preprocessing",
                input_shape=df.shape,
                output_shape=preprocessed_df.shape,
                transformation_details=preprocessing_result['preprocessing_stats']
            )
            lineage_tracker.add_lineage_event(
                event_type="PREPROCESSING_COMPLETE",
                description="Data preprocessing completed",
                data_shape=preprocessed_df.shape
            )
            # Create preprocessing checkpoint
            checkpoint_id = lineage_tracker.create_data_checkpoint(
                checkpoint_name="preprocessed_data",
                df=preprocessed_df,
                stage="preprocessing",
                additional_metadata=preprocessing_result['memory_impact']
            )
            # Add performance metrics
            memory_change = preprocessing_result['memory_impact']['memory_change_mb']
            lineage_tracker.add_performance_metric(
                metric_name="memory_change",
                value=memory_change,
                unit="MB",
                stage="preprocessing"
            )
        
        # Step 4: Hash Generation
        if progress_tracker:
            progress_tracker.update(1, "Generating row hashes")
        
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="HASH_GENERATION_START",
                description="Hash generation phase started",
                data_shape=preprocessed_df.shape
            )
        
        # Create hashing engine with observability
        observer = HashingObserver() if enable_monitoring else None
        if observer and enable_monitoring:
            # Create basic engine first
            engine = HashingEngine()
            # Then integrate observability
            observer = integrate_observability_with_main_function(
                engine=engine,
                monitor=monitor,
                enable_alerts=True
            )
        else:
            engine = HashingEngine()
            observer = HashingObserver() if enable_monitoring else None
        
        # Generate row IDs using vectorized method
        result_df = engine.generate_row_ids_vectorized(
            df=preprocessed_df,
            columns=selected_columns,
            id_column_name=id_column_name,
            separator=separator
        )
        
        if lineage_tracker:
            # Track hash generation transformation
            lineage_tracker.add_transformation(
                transformation_name="hash_generation",
                input_shape=preprocessed_df.shape,
                output_shape=result_df.shape,
                transformation_details={
                    'hash_algorithm': 'SHA-256',
                    'columns_used': selected_columns,
                    'separator': separator,
                    'id_column_name': id_column_name
                }
            )
            lineage_tracker.add_lineage_event(
                event_type="HASH_GENERATION_COMPLETE",
                description=f"Generated {len(result_df)} row IDs",
                data_shape=result_df.shape
            )
            # Add performance metrics from observer
            if observer:
                observer_metrics = observer.export_metrics()
                if observer_metrics.get('average_rows_per_second'):
                    lineage_tracker.add_performance_metric(
                        metric_name="processing_rate",
                        value=observer_metrics['average_rows_per_second'],
                        unit="rows/second",
                        stage="hash_generation"
                    )
                if observer_metrics.get('peak_memory_mb'):
                    lineage_tracker.add_performance_metric(
                        metric_name="peak_memory",
                        value=observer_metrics['peak_memory_mb'],
                        unit="MB",
                        stage="hash_generation"
                    )
        
        # Step 5: Quality Checks
        if progress_tracker:
            progress_tracker.update(1, "Performing quality checks")
        
        if enable_quality_checks:
            if lineage_tracker:
                lineage_tracker.add_lineage_event(
                    event_type="QUALITY_CHECK_START",
                    description="Final quality validation started",
                    data_shape=result_df.shape
                )
            
            quality_requirements = kwargs.get('quality_requirements', {})
            quality_validation = validate_data_quality_requirements(
                df=result_df,
                selected_columns=[id_column_name],
                requirements=quality_requirements
            )
            
            if lineage_tracker:
                lineage_tracker.add_quality_assessment(
                    assessment_type="final_quality_validation",
                    results=quality_validation,
                    stage="quality_checks"
                )
                lineage_tracker.add_lineage_event(
                    event_type="QUALITY_CHECK_COMPLETE",
                    description=f"Quality validation {'passed' if quality_validation['passed'] else 'failed'}",
                    data_shape=result_df.shape
                )
            
            if not quality_validation['passed']:
                for failure in quality_validation['failures']:
                    if progress_tracker:
                        progress_tracker.add_warning(f"Quality check failed: {failure}")
        
        # Step 6: Finalization
        if progress_tracker:
            progress_tracker.update(1, "Finalizing results")
        
        if lineage_tracker:
            # Create final checkpoint
            checkpoint_id = lineage_tracker.create_data_checkpoint(
                checkpoint_name="final_result",
                df=result_df,
                stage="finalization",
                additional_metadata={'id_column_added': id_column_name}
            )
            lineage_tracker.add_lineage_event(
                event_type="FUNCTION_COMPLETE",
                description="Row ID generation process completed successfully",
                data_shape=result_df.shape
            )
        
        # Complete monitoring
        if monitor:
            processing_summary = monitor.end_session(success=True)
        else:
            processing_summary = {}
        
        # Finalize progress tracking
        if progress_tracker:
            progress_tracker.finish(success=True, final_message=f"Generated {len(result_df)} row IDs")
        
        # Prepare audit trail if requested
        if return_audit_trail:
            if enable_enhanced_lineage and lineage_tracker:
                # Use enhanced audit trail with lineage tracking
                audit_trail = integrate_enhanced_audit_trail(
                    session_id=session_id or "NO_SESSION",
                    config=audit_trail_components['config'],
                    data_fingerprint=audit_trail_components['data_fingerprint'],
                    processing_summary=processing_summary,
                    lineage_tracker=lineage_tracker,
                    observer=observer
                )
            else:
                # Use standard audit trail
                audit_trail = create_audit_trail(
                    session_id=session_id or "NO_SESSION",
                    config=audit_trail_components['config'],
                    data_fingerprint=audit_trail_components['data_fingerprint'],
                    processing_summary=processing_summary
                )
            
            return {
                'result_dataframe': result_df,
                'audit_trail': audit_trail,
                'column_selection': column_selection_result,
                'preprocessing': preprocessing_result,
                'session_id': session_id,
                'processing_summary': processing_summary,
                'lineage_tracker': lineage_tracker if enable_enhanced_lineage else None,
                'observer_metrics': observer.export_metrics() if observer else None
            }
        
        return result_df
        
    except Exception as e:
        # Handle errors comprehensively
        if lineage_tracker:
            lineage_tracker.add_lineage_event(
                event_type="ERROR_OCCURRED",
                description=f"Error during processing: {str(e)}",
                data_shape=df.shape if df is not None else None,
                metadata={'error_type': type(e).__name__, 'error_message': str(e)}
            )
        
        if progress_tracker:
            progress_tracker.finish(success=False, final_message=f"Failed: {str(e)}")
        
        if monitor:
            monitor.add_session_error(e)
            monitor.end_session(success=False)
        
        # Re-raise the error with additional context
        if isinstance(e, RowIDGenerationError):
            raise
        else:
            raise RowIDGenerationError(
                f"Row ID generation failed: {str(e)}",
                error_code="GENERATION_FAILED",
                context={
                    'session_id': session_id,
                    'input_shape': df.shape if df is not None else None,
                    'selected_columns': selected_columns if 'selected_columns' in locals() else None,
                    'lineage_available': lineage_tracker is not None
                }
            )


# ========================================
# OBSERVABILITY INTEGRATION - TASK 5.6
# ========================================

def integrate_observability_with_main_function(
    engine: HashingEngine,
    monitor: Optional[ProcessMonitor] = None,
    enable_alerts: bool = True,
    performance_thresholds: Optional[Dict[str, float]] = None
) -> HashingObserver:
    """
    Integrate comprehensive observability into the main hashing function.
    
    Args:
        engine: HashingEngine instance to enhance with observability
        monitor: Optional ProcessMonitor for session tracking
        enable_alerts: Enable performance threshold alerts
        performance_thresholds: Custom performance thresholds
        
    Returns:
        Configured HashingObserver instance
    """
    # Default performance thresholds
    default_thresholds = {
        'max_memory_mb': 500.0,      # 500MB memory limit
        'max_duration_seconds': 300.0,  # 5 minute processing limit
        'min_throughput_rows_per_second': 1000.0,  # Minimum processing rate
        'max_collision_rate': 0.001   # Maximum 0.1% collision rate
    }
    
    thresholds = {**default_thresholds, **(performance_thresholds or {})}
    
    # Create observer with alerting
    observer = HashingObserver(
        enable_alerts=enable_alerts,
        alert_thresholds=thresholds
    )
    
    # Set up alert callbacks for different alert types
    if enable_alerts:
        # Memory threshold alerts
        def memory_alert_callback(alert_data: Dict[str, Any]) -> None:
            message = f"Memory usage ({alert_data['current_memory_mb']:.1f}MB) exceeds threshold ({alert_data['threshold_mb']:.1f}MB)"
            if monitor:
                monitor.add_alert("MEMORY", "WARNING", message, alert_data)
            print(f"⚠️ MEMORY WARNING: {message}")
        
        # Performance threshold alerts
        def performance_alert_callback(alert_data: Dict[str, Any]) -> None:
            message = f"Processing rate ({alert_data['current_rate']:.0f} rows/sec) below threshold ({alert_data['threshold_rate']:.0f} rows/sec)"
            if monitor:
                monitor.add_alert("PERFORMANCE", "WARNING", message, alert_data)
            print(f"⚠️ PERFORMANCE WARNING: {message}")
        
        # Collision alerts
        def collision_alert_callback(alert_data: Dict[str, Any]) -> None:
            message = f"Hash collision detected: {alert_data['collision_details']}"
            if monitor:
                monitor.add_alert("COLLISION", "ERROR", message, alert_data)
            print(f"🔥 COLLISION ALERT: {message}")
        
        # Register alert callbacks
        observer.register_alert_callback("memory_threshold", memory_alert_callback)
        observer.register_alert_callback("performance_threshold", performance_alert_callback)
        observer.register_alert_callback("collision_detected", collision_alert_callback)
    
    # Integrate observer with engine
    engine.observer = observer
    
    return observer


def create_observability_report(
    observer: HashingObserver,
    monitor: Optional[ProcessMonitor] = None,
    include_detailed_metrics: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive observability report combining observer and monitor data.
    
    Args:
        observer: HashingObserver instance with collected metrics
        monitor: Optional ProcessMonitor with session data
        include_detailed_metrics: Include detailed performance metrics
        
    Returns:
        Comprehensive observability report
    """
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'comprehensive_observability',
            'version': '1.0'
        }
    }
    
    # Observer metrics
    if observer:
        observer_data = observer.export_metrics()
        report['hashing_metrics'] = {
            'operations_performed': observer_data.get('total_operations', 0),
            'total_rows_processed': observer_data.get('total_rows_processed', 0),
            'total_processing_time_seconds': observer_data.get('total_duration_seconds', 0),
            'average_processing_rate': observer_data.get('average_rows_per_second', 0),
            'memory_usage_mb': observer_data.get('peak_memory_mb', 0),
            'collisions_detected': observer_data.get('collision_count', 0),
            'error_count': observer_data.get('error_count', 0)
        }
        
        if include_detailed_metrics:
            report['detailed_hashing_metrics'] = observer_data
    
    # Monitor data
    if monitor:
        session_data = monitor.export_session_data()
        report['session_metrics'] = {
            'session_id': session_data.get('session_id'),
            'session_duration_seconds': session_data.get('session_duration_seconds', 0),
            'stages_completed': len(session_data.get('stages', [])),
            'alerts_generated': len(session_data.get('alerts', [])),
            'errors_encountered': len(session_data.get('errors', [])),
            'resource_usage': session_data.get('resource_usage', {})
        }
        
        if include_detailed_metrics:
            report['detailed_session_metrics'] = session_data
    
    # Performance analysis
    if observer and monitor:
        report['performance_analysis'] = {
            'overall_efficiency_score': _calculate_efficiency_score(observer_data, session_data),
            'bottleneck_analysis': _identify_bottlenecks(observer_data, session_data),
            'recommendations': _generate_performance_recommendations(observer_data, session_data)
        }
    
    return report


def _calculate_efficiency_score(observer_data: Dict[str, Any], session_data: Dict[str, Any]) -> float:
    """Calculate overall efficiency score (0-100)."""
    score = 100.0
    
    # Deduct points for errors
    error_count = observer_data.get('error_count', 0) + len(session_data.get('errors', []))
    score -= min(error_count * 10, 50)  # Max 50 points deduction for errors
    
    # Deduct points for collisions
    collision_count = observer_data.get('collision_count', 0)
    score -= min(collision_count * 5, 25)  # Max 25 points deduction for collisions
    
    # Deduct points for slow processing
    processing_rate = observer_data.get('average_rows_per_second', 0)
    if processing_rate < 1000:  # Below 1000 rows/sec
        score -= min((1000 - processing_rate) / 100, 20)  # Max 20 points deduction
    
    # Deduct points for high memory usage
    memory_mb = observer_data.get('peak_memory_mb', 0)
    if memory_mb > 500:  # Above 500MB
        score -= min((memory_mb - 500) / 100, 15)  # Max 15 points deduction
    
    return max(score, 0.0)


def _identify_bottlenecks(observer_data: Dict[str, Any], session_data: Dict[str, Any]) -> List[str]:
    """Identify performance bottlenecks."""
    bottlenecks = []
    
    # Check processing rate
    processing_rate = observer_data.get('average_rows_per_second', 0)
    if processing_rate < 500:
        bottlenecks.append(f"Low processing rate: {processing_rate:.0f} rows/sec")
    
    # Check memory usage
    memory_mb = observer_data.get('peak_memory_mb', 0)
    if memory_mb > 750:
        bottlenecks.append(f"High memory usage: {memory_mb:.1f}MB")
    
    # Check stage durations
    stages = session_data.get('stages', [])
    for stage in stages:
        if stage.get('duration_seconds', 0) > 60:  # More than 1 minute
            bottlenecks.append(f"Slow stage: {stage.get('name')} took {stage.get('duration_seconds'):.1f}s")
    
    return bottlenecks


def _generate_performance_recommendations(observer_data: Dict[str, Any], session_data: Dict[str, Any]) -> List[str]:
    """Generate performance optimization recommendations."""
    recommendations = []
    
    # Processing rate recommendations
    processing_rate = observer_data.get('average_rows_per_second', 0)
    if processing_rate < 1000:
        recommendations.append("Consider using vectorized operations or parallel processing to improve throughput")
    
    # Memory recommendations
    memory_mb = observer_data.get('peak_memory_mb', 0)
    if memory_mb > 500:
        recommendations.append("Consider processing data in smaller batches to reduce memory usage")
    
    # Error handling recommendations
    error_count = observer_data.get('error_count', 0)
    if error_count > 0:
        recommendations.append("Review error logs and implement additional data validation")
    
    # Collision recommendations
    collision_count = observer_data.get('collision_count', 0)
    if collision_count > 0:
        recommendations.append("Consider adding more columns to reduce hash collision probability")
    
    return recommendations


# ========================================
# CONVENIENCE FUNCTIONS AND ALIASES
# ========================================

# Create convenient aliases for the main function with different configurations
def generate_row_ids_simple(df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Simple row ID generation with minimal output and monitoring."""
    return generate_unique_row_ids(
        df=df,
        columns=columns,
        enable_monitoring=False,
        enable_progress=False,
        return_audit_trail=False,
        **kwargs
    )


def generate_row_ids_with_audit(df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Row ID generation with comprehensive audit trail."""
    return generate_unique_row_ids(
        df=df,
        columns=columns,
        enable_monitoring=True,
        enable_progress=True,
        return_audit_trail=True,
        **kwargs
    )


def generate_row_ids_fast(df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """High-performance row ID generation with minimal overhead."""
    return generate_unique_row_ids(
        df=df,
        columns=columns,
        enable_monitoring=False,
        enable_progress=False,
        enable_quality_checks=False,
        show_warnings=False,
        return_audit_trail=False,
        **kwargs
    )


# ========================================
# ENHANCED DATA LINEAGE AND AUDIT TRAIL - TASK 5.10
# ========================================

class DataLineageTracker:
    """Advanced data lineage tracking for comprehensive audit trails."""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or generate_session_id()
        self.lineage_events = []
        self.transformation_history = []
        self.data_checkpoints = {}
        self.performance_tracking = {}
        self.quality_tracking = {}
    
    def add_lineage_event(
        self,
        event_type: str,
        description: str,
        data_shape: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a lineage event to the tracking history."""
        event = {
            'event_id': f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'data_shape': data_shape,
            'metadata': metadata or {}
        }
        self.lineage_events.append(event)
    
    def add_transformation(
        self,
        transformation_name: str,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        transformation_details: Dict[str, Any]
    ) -> None:
        """Track a data transformation operation."""
        transformation = {
            'transformation_id': f"TRANS_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'timestamp': datetime.now().isoformat(),
            'name': transformation_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'rows_changed': output_shape[0] - input_shape[0],
            'columns_changed': output_shape[1] - input_shape[1],
            'details': transformation_details
        }
        self.transformation_history.append(transformation)
    
    def create_data_checkpoint(
        self,
        checkpoint_name: str,
        df: pd.DataFrame,
        stage: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a data checkpoint for lineage tracking."""
        checkpoint_id = f"CHKPT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create comprehensive checkpoint data
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'name': checkpoint_name,
            'stage': stage,
            'data_fingerprint': create_data_fingerprint(df),
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'metadata': additional_metadata or {}
        }
        
        self.data_checkpoints[checkpoint_id] = checkpoint_data
        return checkpoint_id
    
    def add_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        stage: str = None
    ) -> None:
        """Add performance metrics to lineage tracking."""
        if stage not in self.performance_tracking:
            self.performance_tracking[stage] = []
        
        metric = {
            'metric_id': f"PERF_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'timestamp': datetime.now().isoformat(),
            'name': metric_name,
            'value': value,
            'unit': unit,
            'stage': stage
        }
        self.performance_tracking[stage].append(metric)
    
    def add_quality_assessment(
        self,
        assessment_type: str,
        results: Dict[str, Any],
        stage: str = None
    ) -> None:
        """Add quality assessment results to lineage tracking."""
        if stage not in self.quality_tracking:
            self.quality_tracking[stage] = []
        
        assessment = {
            'assessment_id': f"QUAL_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'timestamp': datetime.now().isoformat(),
            'type': assessment_type,
            'results': results,
            'stage': stage
        }
        self.quality_tracking[stage].append(assessment)
    
    def generate_comprehensive_lineage_report(self) -> Dict[str, Any]:
        """Generate comprehensive data lineage report."""
        return {
            'lineage_metadata': {
                'session_id': self.session_id,
                'generated_at': datetime.now().isoformat(),
                'total_events': len(self.lineage_events),
                'total_transformations': len(self.transformation_history),
                'total_checkpoints': len(self.data_checkpoints),
                'lineage_version': '2.0'
            },
            'lineage_timeline': {
                'events': self.lineage_events,
                'transformations': self.transformation_history
            },
            'data_checkpoints': self.data_checkpoints,
            'performance_tracking': self.performance_tracking,
            'quality_tracking': self.quality_tracking,
            'lineage_summary': self._generate_lineage_summary()
        }
    
    def _generate_lineage_summary(self) -> Dict[str, Any]:
        """Generate summary of lineage tracking."""
        # Calculate data flow metrics
        input_shapes = [t['input_shape'] for t in self.transformation_history]
        output_shapes = [t['output_shape'] for t in self.transformation_history]
        
        if input_shapes and output_shapes:
            initial_shape = input_shapes[0]
            final_shape = output_shapes[-1]
            total_row_change = final_shape[0] - initial_shape[0]
            total_col_change = final_shape[1] - initial_shape[1]
        else:
            initial_shape = final_shape = (0, 0)
            total_row_change = total_col_change = 0
        
        # Performance summary
        all_performance = []
        for stage_metrics in self.performance_tracking.values():
            all_performance.extend(stage_metrics)
        
        # Quality summary
        all_quality = []
        for stage_assessments in self.quality_tracking.values():
            all_quality.extend(stage_assessments)
        
        return {
            'data_flow': {
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'total_row_change': total_row_change,
                'total_column_change': total_col_change,
                'transformation_count': len(self.transformation_history)
            },
            'performance_summary': {
                'total_metrics_collected': len(all_performance),
                'stages_monitored': len(self.performance_tracking),
                'performance_events': all_performance
            },
            'quality_summary': {
                'total_assessments': len(all_quality),
                'quality_stages': len(self.quality_tracking),
                'quality_events': all_quality
            },
            'processing_timeline': {
                'start_time': self.lineage_events[0]['timestamp'] if self.lineage_events else None,
                'end_time': self.lineage_events[-1]['timestamp'] if self.lineage_events else None,
                'total_duration_events': len(self.lineage_events)
            }
        }


def create_enhanced_audit_trail(
    session_id: str,
    config: Dict[str, Any],
    data_fingerprint: Dict[str, Any],
    processing_summary: Dict[str, Any],
    lineage_tracker: Optional[DataLineageTracker] = None,
    observer_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create enhanced audit trail with comprehensive lineage tracking.
    
    Args:
        session_id: Unique session identifier
        config: Configuration logging data
        data_fingerprint: Input data fingerprint
        processing_summary: Processing summary from monitor
        lineage_tracker: Optional lineage tracker instance
        observer_metrics: Optional observer metrics data
        
    Returns:
        Comprehensive audit trail with enhanced lineage tracking
    """
    audit_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Base audit trail
    audit_trail = {
        'audit_metadata': {
            'audit_id': audit_id,
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'version': '2.0',
            'audit_type': 'comprehensive_enhanced'
        },
        'configuration': config,
        'input_data_fingerprint': data_fingerprint,
        'processing_summary': processing_summary
    }
    
    # Add lineage tracking if available
    if lineage_tracker:
        lineage_report = lineage_tracker.generate_comprehensive_lineage_report()
        audit_trail['data_lineage'] = lineage_report
        
        # Add lineage summary to top level
        audit_trail['lineage_summary'] = {
            'total_transformations': len(lineage_tracker.transformation_history),
            'data_checkpoints': len(lineage_tracker.data_checkpoints),
            'performance_metrics': sum(len(metrics) for metrics in lineage_tracker.performance_tracking.values()),
            'quality_assessments': sum(len(assessments) for assessments in lineage_tracker.quality_tracking.values())
        }
    
    # Add observer metrics if available
    if observer_metrics:
        audit_trail['observability_metrics'] = observer_metrics
        
        # Add observer summary
        audit_trail['observability_summary'] = {
            'total_operations': observer_metrics.get('total_operations', 0),
            'total_rows_processed': observer_metrics.get('total_rows_processed', 0),
            'average_processing_rate': observer_metrics.get('average_rows_per_second', 0),
            'peak_memory_mb': observer_metrics.get('peak_memory_mb', 0),
            'error_count': observer_metrics.get('error_count', 0),
            'collision_count': observer_metrics.get('collision_count', 0)
        }
    
    # Generate processing insights
    audit_trail['processing_insights'] = _generate_processing_insights(
        config, processing_summary, lineage_tracker, observer_metrics
    )
    
    return audit_trail


def _generate_processing_insights(
    config: Dict[str, Any],
    processing_summary: Dict[str, Any],
    lineage_tracker: Optional[DataLineageTracker] = None,
    observer_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate insights from processing data."""
    insights = {
        'data_characteristics': {},
        'processing_efficiency': {},
        'quality_indicators': {},
        'recommendations': []
    }
    
    # Data characteristics insights
    input_data = config.get('input_data', {})
    if input_data:
        insights['data_characteristics'] = {
            'dataset_size': input_data.get('shape', [0, 0])[0],
            'column_count': input_data.get('shape', [0, 0])[1],
            'memory_footprint_mb': input_data.get('memory_usage_mb', 0),
            'data_complexity': 'high' if input_data.get('memory_usage_mb', 0) > 100 else 'medium' if input_data.get('memory_usage_mb', 0) > 10 else 'low'
        }
    
    # Processing efficiency insights
    if observer_metrics:
        processing_rate = observer_metrics.get('average_rows_per_second', 0)
        memory_usage = observer_metrics.get('peak_memory_mb', 0)
        
        insights['processing_efficiency'] = {
            'processing_rate_classification': 'high' if processing_rate > 10000 else 'medium' if processing_rate > 1000 else 'low',
            'memory_efficiency': 'efficient' if memory_usage < 100 else 'moderate' if memory_usage < 500 else 'memory_intensive',
            'overall_performance': 'excellent' if processing_rate > 10000 and memory_usage < 100 else 'good' if processing_rate > 1000 else 'needs_optimization'
        }
    
    # Quality indicators
    if lineage_tracker:
        quality_events = []
        for stage_assessments in lineage_tracker.quality_tracking.values():
            quality_events.extend(stage_assessments)
        
        insights['quality_indicators'] = {
            'quality_checks_performed': len(quality_events),
            'data_integrity': 'high' if len(quality_events) > 0 else 'unknown',
            'validation_coverage': 'comprehensive' if len(quality_events) > 2 else 'basic' if len(quality_events) > 0 else 'minimal'
        }
    
    # Generate recommendations
    recommendations = []
    
    if observer_metrics:
        processing_rate = observer_metrics.get('average_rows_per_second', 0)
        if processing_rate < 1000:
            recommendations.append("Consider using vectorized operations to improve processing speed")
        
        memory_usage = observer_metrics.get('peak_memory_mb', 0)
        if memory_usage > 500:
            recommendations.append("Consider processing data in smaller batches to reduce memory usage")
        
        error_count = observer_metrics.get('error_count', 0)
        if error_count > 0:
            recommendations.append("Review and address data quality issues to reduce processing errors")
    
    if not recommendations:
        recommendations.append("Processing completed successfully with optimal performance")
    
    insights['recommendations'] = recommendations
    
    return insights


# Enhanced integration function for main workflow
def integrate_enhanced_audit_trail(
    session_id: str,
    config: Dict[str, Any],
    data_fingerprint: Dict[str, Any],
    processing_summary: Dict[str, Any],
    lineage_tracker: DataLineageTracker,
    observer: Optional[HashingObserver] = None
) -> Dict[str, Any]:
    """
    Integrate enhanced audit trail generation into the main workflow.
    
    Args:
        session_id: Session identifier
        config: Configuration data
        data_fingerprint: Input data fingerprint
        processing_summary: Processing summary
        lineage_tracker: Data lineage tracker instance
        observer: Optional hashing observer
        
    Returns:
        Enhanced audit trail with complete lineage tracking
    """
    # Get observer metrics if available
    observer_metrics = None
    if observer:
        observer_metrics = observer.export_metrics()
    
    # Create enhanced audit trail
    audit_trail = create_enhanced_audit_trail(
        session_id=session_id,
        config=config,
        data_fingerprint=data_fingerprint,
        processing_summary=processing_summary,
        lineage_tracker=lineage_tracker,
        observer_metrics=observer_metrics
    )
    
    return audit_trail


# ========================================
# TASK 6: ERROR HANDLING AND VALIDATION SYSTEM
# ========================================

import warnings

def validate_input_dataframe(df: Any) -> None:
    """
    Comprehensive DataFrame input validation with detailed error messages.
    
    Args:
        df: Input to validate as DataFrame
        
    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame is empty or has no columns
        RuntimeError: If DataFrame has structural issues
    """
    # Check if input is actually a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(df).__name__}. "
            f"Please convert your data to a pandas DataFrame using pd.DataFrame(data)."
        )
    
    # Check if DataFrame is empty (no rows)
    if df.empty:
        raise ValueError(
            "Input DataFrame is empty (contains no rows). "
            "Please provide a DataFrame with at least one row of data."
        )
    
    # Check if DataFrame has no columns
    if len(df.columns) == 0:
        raise ValueError(
            "DataFrame has no columns. "
            "Please provide a DataFrame with at least one column."
        )
    
    # Check for extremely large DataFrames (memory protection)
    max_rows = 50_000_000  # 50 million rows
    if len(df) > max_rows:
        raise RuntimeError(
            f"DataFrame is too large for safe processing: {len(df):,} rows "
            f"(maximum supported: {max_rows:,}). "
            f"Consider processing in smaller chunks or using a more powerful machine."
        )
    
    # Check for extremely wide DataFrames
    max_columns = 10_000
    if len(df.columns) > max_columns:
        raise RuntimeError(
            f"DataFrame has too many columns: {len(df.columns):,} "
            f"(maximum supported: {max_columns:,}). "
            f"Consider selecting specific columns for processing."
        )
    
    # Check for duplicate column names
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            f"DataFrame contains duplicate column names: {duplicate_columns}. "
            f"Please rename duplicate columns to ensure unique column names."
        )
    
    # Check for problematic column names
    problematic_columns = []
    for col in df.columns:
        if not isinstance(col, (str, int, float)):
            problematic_columns.append(f"{col} (type: {type(col).__name__})")
        elif isinstance(col, str) and (not col.strip() or '\n' in col or '\r' in col):
            problematic_columns.append(f"'{col}' (empty or contains line breaks)")
    
    if problematic_columns:
        raise ValueError(
            f"DataFrame contains problematic column names: {problematic_columns}. "
            f"Column names should be strings without line breaks or empty values."
        )


def validate_manual_columns(df: pd.DataFrame, columns: Optional[List[str]]) -> None:
    """
    Validate manual column specifications with comprehensive error handling.
    
    Args:
        df: DataFrame to validate columns against
        columns: List of column names to validate
        
    Raises:
        TypeError: If columns parameter has invalid type
        ValueError: If columns are missing or invalid
    """
    if columns is None:
        return  # Nothing to validate
    
    # Check if columns is the correct type
    if not isinstance(columns, (list, tuple)):
        if isinstance(columns, str):
            raise TypeError(
                f"Expected a list of column names, but got a single string: '{columns}'. "
                f"If you want to specify a single column, use: ['{columns}']"
            )
        else:
            raise TypeError(
                f"Columns parameter must be a list or tuple of strings, "
                f"got {type(columns).__name__}: {columns}"
            )
    
    # Convert to list for consistency
    columns_list = list(columns)
    
    # Check for empty list
    if len(columns_list) == 0:
        raise ValueError(
            "Columns list cannot be empty. "
            "Either provide column names or use None for automatic selection."
        )
    
    # Validate each column name
    invalid_columns = []
    non_string_columns = []
    empty_columns = []
    
    for i, col in enumerate(columns_list):
        if not isinstance(col, str):
            non_string_columns.append(f"Position {i}: {col} (type: {type(col).__name__})")
        elif not col.strip():
            empty_columns.append(f"Position {i}: '{col}' (empty or whitespace)")
        elif col not in df.columns:
            invalid_columns.append(col)
    
    # Report validation errors
    error_messages = []
    
    if non_string_columns:
        error_messages.append(
            f"Non-string column names found: {non_string_columns}. "
            f"All column names must be strings."
        )
    
    if empty_columns:
        error_messages.append(
            f"Empty or whitespace column names found: {empty_columns}. "
            f"Column names cannot be empty."
        )
    
    if invalid_columns:
        available_columns = list(df.columns)[:10]  # Show first 10 for brevity
        more_cols_msg = f" (and {len(df.columns) - 10} more)" if len(df.columns) > 10 else ""
        error_messages.append(
            f"Columns not found in DataFrame: {invalid_columns}. "
            f"Available columns: {available_columns}{more_cols_msg}"
        )
    
    if error_messages:
        full_error = "Column validation failed:\n" + "\n".join(f"- {msg}" for msg in error_messages)
        raise ValueError(full_error)
    
    # Check for duplicate columns in the list
    seen_columns = set()
    duplicate_specs = []
    for col in columns_list:
        if col in seen_columns:
            duplicate_specs.append(col)
        seen_columns.add(col)
    
    if duplicate_specs:
        raise ValueError(
            f"Duplicate columns specified in list: {duplicate_specs}. "
            f"Each column should be specified only once."
        )


def check_uniqueness_warning(df: pd.DataFrame, selected_columns: List[str], 
                           threshold: float = 0.95, warn_duplicates: bool = True) -> Dict[str, Any]:
    """
    Check uniqueness of selected columns and generate warnings for potential issues.
    
    Args:
        df: Input DataFrame
        selected_columns: List of columns to check for uniqueness
        threshold: Minimum uniqueness ratio before warning (default: 0.95)
        warn_duplicates: Whether to warn about duplicate combinations
        
    Returns:
        Dictionary containing uniqueness analysis results
    """
    uniqueness_report = {
        'uniqueness_ratio': 0.0,
        'total_rows': len(df),
        'unique_combinations': 0,
        'duplicate_count': 0,
        'warnings_generated': [],
        'column_analysis': {}
    }
    
    try:
        # Analyze the selected columns subset
        subset_df = df[selected_columns].copy()
        
        # Handle missing values by filling with a sentinel value
        subset_df = subset_df.fillna('__NULL__')
        
        # Count unique combinations
        unique_combinations = subset_df.drop_duplicates()
        uniqueness_ratio = len(unique_combinations) / len(subset_df) if len(subset_df) > 0 else 0.0
        duplicate_count = len(subset_df) - len(unique_combinations)
        
        uniqueness_report.update({
            'uniqueness_ratio': uniqueness_ratio,
            'unique_combinations': len(unique_combinations),
            'duplicate_count': duplicate_count
        })
        
        # Generate warnings based on uniqueness ratio
        if uniqueness_ratio < threshold:
            warning_msg = (
                f"Selected columns may not provide sufficient uniqueness. "
                f"Uniqueness ratio: {uniqueness_ratio:.3f} (below threshold: {threshold:.3f}). "
                f"Found {duplicate_count:,} duplicate combinations out of {len(subset_df):,} rows. "
                f"Consider adding more columns or removing duplicate data."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
            uniqueness_report['warnings_generated'].append({
                'type': 'low_uniqueness',
                'message': warning_msg,
                'severity': 'medium'
            })
        
        # Warn about high duplicate count
        if warn_duplicates and duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(subset_df)) * 100
            if duplicate_percentage > 10:  # More than 10% duplicates
                warning_msg = (
                    f"High number of duplicate row combinations detected: "
                    f"{duplicate_count:,} duplicates ({duplicate_percentage:.1f}% of data). "
                    f"This may indicate data quality issues."
                )
                warnings.warn(warning_msg, UserWarning, stacklevel=3)
                uniqueness_report['warnings_generated'].append({
                    'type': 'high_duplicates',
                    'message': warning_msg,
                    'severity': 'high' if duplicate_percentage > 25 else 'medium'
                })
        
        # Analyze individual columns
        for col in selected_columns:
            col_series = df[col].fillna('__NULL__')
            col_unique_count = col_series.nunique()
            col_uniqueness = col_unique_count / len(col_series) if len(col_series) > 0 else 0.0
            
            uniqueness_report['column_analysis'][col] = {
                'unique_count': col_unique_count,
                'uniqueness_ratio': col_uniqueness,
                'null_count': df[col].isnull().sum(),
                'data_type': str(df[col].dtype)
            }
            
            # Warn about low-uniqueness individual columns
            if col_uniqueness < 0.1:  # Less than 10% unique values
                warning_msg = (
                    f"Column '{col}' has very low uniqueness: "
                    f"{col_uniqueness:.3f} ({col_unique_count:,} unique values in {len(col_series):,} rows). "
                    f"This column may not contribute effectively to row uniqueness."
                )
                warnings.warn(warning_msg, UserWarning, stacklevel=3)
                uniqueness_report['warnings_generated'].append({
                    'type': 'low_column_uniqueness',
                    'column': col,
                    'message': warning_msg,
                    'severity': 'low'
                })
        
        return uniqueness_report
        
    except Exception as e:
        error_msg = f"Error during uniqueness analysis: {str(e)}"
        warnings.warn(error_msg, RuntimeWarning, stacklevel=3)
        uniqueness_report['warnings_generated'].append({
            'type': 'analysis_error',
            'message': error_msg,
            'severity': 'high'
        })
        return uniqueness_report


def validate_processing_parameters(
    id_column_name: str = 'row_id',
    uniqueness_threshold: float = 0.95,
    separator: str = '|',
    **kwargs
) -> Dict[str, Any]:
    """
    Validate processing parameters with comprehensive error checking.
    
    Args:
        id_column_name: Name for the generated ID column
        uniqueness_threshold: Minimum uniqueness ratio
        separator: String separator for concatenation
        **kwargs: Additional parameters to validate
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
    """
    validated_params = {}
    
    # Validate ID column name
    if not isinstance(id_column_name, str):
        raise TypeError(
            f"ID column name must be a string, got {type(id_column_name).__name__}: {id_column_name}"
        )
    
    if not id_column_name.strip():
        raise ValueError(
            "ID column name cannot be empty or whitespace only."
        )
    
    # Check for problematic characters in column name
    problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']
    found_problematic = [char for char in problematic_chars if char in id_column_name]
    if found_problematic:
        raise ValueError(
            f"ID column name contains invalid characters: {found_problematic}. "
            f"Please use only alphanumeric characters, spaces, hyphens, and underscores."
        )
    
    validated_params['id_column_name'] = id_column_name.strip()
    
    # Validate uniqueness threshold
    if not isinstance(uniqueness_threshold, (int, float)):
        raise TypeError(
            f"Uniqueness threshold must be a number, got {type(uniqueness_threshold).__name__}: {uniqueness_threshold}"
        )
    
    if not (0.0 <= uniqueness_threshold <= 1.0):
        raise ValueError(
            f"Uniqueness threshold must be between 0.0 and 1.0, got {uniqueness_threshold}"
        )
    
    # Warn about extreme threshold values
    if uniqueness_threshold < 0.1:
        warnings.warn(
            f"Very low uniqueness threshold ({uniqueness_threshold}). "
            f"This may result in poor column selection.",
            UserWarning
        )
    elif uniqueness_threshold > 0.99:
        warnings.warn(
            f"Very high uniqueness threshold ({uniqueness_threshold}). "
            f"This may make it difficult to find suitable columns.",
            UserWarning
        )
    
    validated_params['uniqueness_threshold'] = float(uniqueness_threshold)
    
    # Validate separator
    if not isinstance(separator, str):
        raise TypeError(
            f"Separator must be a string, got {type(separator).__name__}: {separator}"
        )
    
    if len(separator) == 0:
        raise ValueError("Separator cannot be empty string.")
    
    # Check for problematic separators
    if '\n' in separator or '\r' in separator or '\t' in separator:
        raise ValueError(
            f"Separator contains line break or tab characters which may cause issues: '{repr(separator)}'"
        )
    
    if len(separator) > 10:
        warnings.warn(
            f"Very long separator ({len(separator)} characters): '{separator}'. "
            f"This may impact performance and readability.",
            UserWarning
        )
    
    validated_params['separator'] = separator
    
    # Validate additional boolean parameters
    boolean_params = ['show_progress', 'enable_monitoring', 'enable_quality_checks', 'show_warnings']
    for param in boolean_params:
        if param in kwargs:
            value = kwargs[param]
            if not isinstance(value, bool):
                # Try to convert string representations
                if isinstance(value, str):
                    if value.lower() in ('true', 'yes', '1'):
                        validated_params[param] = True
                    elif value.lower() in ('false', 'no', '0'):
                        validated_params[param] = False
                    else:
                        raise ValueError(
                            f"Parameter '{param}' must be boolean or boolean string, got: '{value}'"
                        )
                else:
                    raise TypeError(
                        f"Parameter '{param}' must be boolean, got {type(value).__name__}: {value}"
                    )
            else:
                validated_params[param] = value
    
    return validated_params


def validate_all_inputs(
    df: Any,
    columns: Optional[List[str]] = None,
    id_column_name: str = 'row_id',
    uniqueness_threshold: float = 0.95,
    separator: str = '|',
    **kwargs
) -> Dict[str, Any]:
    """
    Comprehensive validation of all inputs for the row ID generation process.
    
    Args:
        df: Input DataFrame to validate
        columns: Optional list of columns to validate
        id_column_name: ID column name to validate
        uniqueness_threshold: Uniqueness threshold to validate
        separator: Separator string to validate
        **kwargs: Additional parameters to validate
        
    Returns:
        Dictionary containing validation results and warnings
        
    Raises:
        Various exceptions for validation failures
    """
    validation_results = {
        'validation_passed': True,
        'warnings_generated': [],
        'parameters_validated': {},
        'uniqueness_analysis': {},
        'validation_timestamp': datetime.now().isoformat()
    }
    
    try:
        # Step 1: Validate DataFrame input
        logger.info("Validating DataFrame input...")
        validate_input_dataframe(df)
        validation_results['dataframe_validation'] = 'passed'
        
        # Step 2: Validate processing parameters
        logger.info("Validating processing parameters...")
        validated_params = validate_processing_parameters(
            id_column_name=id_column_name,
            uniqueness_threshold=uniqueness_threshold,
            separator=separator,
            **kwargs
        )
        validation_results['parameters_validated'] = validated_params
        
        # Step 3: Check if ID column already exists
        if validated_params['id_column_name'] in df.columns:
            raise ValueError(
                f"Column '{validated_params['id_column_name']}' already exists in DataFrame. "
                f"Please choose a different ID column name or remove the existing column."
            )
        
        # Step 4: Validate manual columns if provided
        if columns is not None:
            logger.info("Validating manual column specification...")
            validate_manual_columns(df, columns)
            validation_results['column_validation'] = 'passed'
            
            # Step 5: Check uniqueness for manual columns
            logger.info("Analyzing uniqueness of selected columns...")
            uniqueness_analysis = check_uniqueness_warning(
                df, columns, validated_params['uniqueness_threshold']
            )
            validation_results['uniqueness_analysis'] = uniqueness_analysis
            validation_results['warnings_generated'].extend(uniqueness_analysis['warnings_generated'])
        else:
            validation_results['column_validation'] = 'skipped_auto_selection'
        
        logger.info("Input validation completed successfully")
        return validation_results
        
    except Exception as e:
        validation_results['validation_passed'] = False
        validation_results['validation_error'] = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        logger.error(f"Input validation failed: {str(e)}")
        raise


def handle_validation_warnings(validation_results: Dict[str, Any], 
                             show_warnings: bool = True,
                             warning_callback: Optional[Callable] = None) -> None:
    """
    Handle and display validation warnings in a user-friendly manner.
    
    Args:
        validation_results: Results from validation process
        show_warnings: Whether to display warnings
        warning_callback: Optional callback function for custom warning handling
    """
    warnings_generated = validation_results.get('warnings_generated', [])
    
    if not warnings_generated:
        return
    
    if show_warnings:
        print("\n" + "="*60)
        print("VALIDATION WARNINGS")
        print("="*60)
        
        # Group warnings by severity
        warning_groups = {'high': [], 'medium': [], 'low': []}
        for warning in warnings_generated:
            severity = warning.get('severity', 'medium')
            warning_groups[severity].append(warning)
        
        # Display warnings by severity
        for severity in ['high', 'medium', 'low']:
            if warning_groups[severity]:
                severity_symbols = {'high': '🔥', 'medium': '⚠️', 'low': 'ℹ️'}
                print(f"\n{severity_symbols[severity]} {severity.upper()} WARNINGS:")
                for i, warning in enumerate(warning_groups[severity], 1):
                    print(f"  {i}. {warning['message']}")
        
        print("\n" + "="*60)
    
    # Call custom warning callback if provided
    if warning_callback:
        for warning in warnings_generated:
            warning_callback(warning)


# ========================================
# ENHANCED ERROR HANDLING FOR MAIN WORKFLOW INTEGRATION
# ========================================

def create_error_context(operation: str, **context_data) -> Dict[str, Any]:
    """Create error context for enhanced error reporting."""
    return {
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
        'context_data': context_data
    }


def handle_processing_error(error: Exception, context: Dict[str, Any]) -> None:
    """Enhanced error handling with context information."""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'operation': context.get('operation', 'unknown'),
        'timestamp': context.get('timestamp'),
        'context_data': context.get('context_data', {}),
        'traceback': traceback.format_exc()
    }
    
    logger.error(f"Processing error in {error_info['operation']}: {error_info}")
    
    # Log to file for debugging
    try:
        with open('error_log.json', 'a') as f:
            f.write(json.dumps(error_info) + '\n')
    except Exception:
        pass  # Don't fail if we can't log


# ========================================
# TASK 7: PERFORMANCE OPTIMIZATION SYSTEM
# ========================================

# ========================================
# SUBTASK 7.1: BOTTLENECK ANALYSIS SYSTEM
# ========================================

class PerformanceBottleneckAnalyzer:
    """
    Comprehensive performance bottleneck analyzer for row ID generation system.
    Identifies CPU, memory, and I/O bottlenecks with detailed profiling.
    """
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.profiling_results = {}
        self.memory_snapshots = []
        self.execution_times = defaultdict(list)
        self.memory_usage_patterns = defaultdict(list)
        
    def profile_function(self, func_name: str = None):
        """Decorator to profile function execution with detailed analysis."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start memory profiling if enabled
                if self.enable_memory_profiling:
                    tracemalloc.start()
                
                # CPU profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                # Track execution time
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = None
                    success = False
                    raise
                finally:
                    # Stop profiling
                    profiler.disable()
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Collect profiling data
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Store results
                    function_name = func_name or func.__name__
                    self.execution_times[function_name].append(execution_time)
                    self.memory_usage_patterns[function_name].append(memory_delta)
                    
                    # Store detailed profiling results
                    self.profiling_results[function_name] = {
                        'profiler': profiler,
                        'execution_time': execution_time,
                        'memory_delta_mb': memory_delta,
                        'success': success,
                        'timestamp': datetime.now()
                    }
                    
                    # Memory snapshot if enabled
                    if self.enable_memory_profiling and tracemalloc.is_tracing():
                        snapshot = tracemalloc.take_snapshot()
                        self.memory_snapshots.append({
                            'function': function_name,
                            'snapshot': snapshot,
                            'timestamp': datetime.now()
                        })
                        tracemalloc.stop()
                
                return result
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0
    
    def analyze_bottlenecks(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze profiling results to identify bottlenecks.
        
        Returns:
            Dict containing bottleneck analysis results
        """
        bottlenecks = {
            'cpu_bottlenecks': [],
            'memory_bottlenecks': [],
            'function_analysis': {},
            'recommendations': []
        }
        
        # Analyze each profiled function
        for func_name, data in self.profiling_results.items():
            # CPU Analysis
            profiler = data['profiler']
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(top_n)
            
            cpu_analysis = self._analyze_cpu_profile(stats, func_name)
            bottlenecks['cpu_bottlenecks'].extend(cpu_analysis['bottlenecks'])
            
            # Memory Analysis
            memory_analysis = self._analyze_memory_usage(func_name)
            bottlenecks['memory_bottlenecks'].extend(memory_analysis['bottlenecks'])
            
            # Function-specific analysis
            bottlenecks['function_analysis'][func_name] = {
                'cpu_analysis': cpu_analysis,
                'memory_analysis': memory_analysis,
                'execution_times': self.execution_times[func_name],
                'memory_deltas': self.memory_usage_patterns[func_name]
            }
        
        # Generate recommendations
        bottlenecks['recommendations'] = self._generate_optimization_recommendations(bottlenecks)
        
        return bottlenecks
    
    def _analyze_cpu_profile(self, stats: pstats.Stats, func_name: str) -> Dict[str, Any]:
        """Analyze CPU profiling results for bottlenecks."""
        # Extract top CPU consumers
        stats_data = []
        
        # Get stats from the pstats object correctly
        # stats.stats is a dictionary where key is (file, line, func) and value is (cc, nc, tt, ct, callers)
        for func_key, func_stats in stats.stats.items():
            filename, line, function = func_key
            cc, nc, tt, ct, callers = func_stats
            stats_data.append({
                'function': function,
                'filename': filename,
                'line': line,
                'cumulative_time': ct,
                'total_time': tt,
                'call_count': cc
            })
        
        # Sort by cumulative time
        stats_data.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        # Identify bottlenecks (functions taking >5% of total time)
        total_time = sum(s['cumulative_time'] for s in stats_data) if stats_data else 0
        bottlenecks = []
        
        for stat in stats_data[:10]:  # Top 10
            time_percentage = (stat['cumulative_time'] / total_time) * 100 if total_time > 0 else 0
            if time_percentage > 5:  # 5% threshold
                bottlenecks.append({
                    'function': stat['function'],
                    'cumulative_time': stat['cumulative_time'],
                    'time_percentage': time_percentage,
                    'call_count': stat['call_count'],
                    'severity': 'high' if time_percentage > 20 else 'medium'
                })
        
        return {
            'bottlenecks': bottlenecks,
            'total_time': total_time,
            'top_functions': stats_data[:10]
        }
    
    def _analyze_memory_usage(self, func_name: str) -> Dict[str, Any]:
        """Analyze memory usage patterns for bottlenecks."""
        memory_deltas = self.memory_usage_patterns[func_name]
        
        if not memory_deltas:
            return {'bottlenecks': [], 'statistics': {}}
        
        # Calculate statistics
        avg_memory = np.mean(memory_deltas)
        max_memory = max(memory_deltas)
        std_memory = np.std(memory_deltas)
        
        bottlenecks = []
        
        # High memory usage threshold (>100MB average)
        if avg_memory > 100:
            bottlenecks.append({
                'type': 'high_memory_usage',
                'function': func_name,
                'average_memory_mb': avg_memory,
                'max_memory_mb': max_memory,
                'severity': 'high' if avg_memory > 500 else 'medium'
            })
        
        # Memory growth pattern detection
        if len(memory_deltas) > 3:
            # Check if memory usage is consistently growing
            growth_trend = np.polyfit(range(len(memory_deltas)), memory_deltas, 1)[0]
            if growth_trend > 10:  # Growing by 10MB per call
                bottlenecks.append({
                    'type': 'memory_growth',
                    'function': func_name,
                    'growth_rate_mb_per_call': growth_trend,
                    'severity': 'high' if growth_trend > 50 else 'medium'
                })
        
        return {
            'bottlenecks': bottlenecks,
            'statistics': {
                'average_memory_mb': avg_memory,
                'max_memory_mb': max_memory,
                'std_memory_mb': std_memory,
                'total_calls': len(memory_deltas)
            }
        }
    
    def _generate_optimization_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on bottleneck analysis."""
        recommendations = []
        
        # CPU bottleneck recommendations
        cpu_bottlenecks = bottlenecks['cpu_bottlenecks']
        if cpu_bottlenecks:
            high_severity = [b for b in cpu_bottlenecks if b['severity'] == 'high']
            if high_severity:
                recommendations.extend([
                    "Implement vectorized operations for high-CPU functions",
                    "Consider parallel processing for CPU-intensive operations",
                    "Profile and optimize hot code paths identified in analysis"
                ])
        
        # Memory bottleneck recommendations
        memory_bottlenecks = bottlenecks['memory_bottlenecks']
        if memory_bottlenecks:
            for bottleneck in memory_bottlenecks:
                if bottleneck['type'] == 'high_memory_usage':
                    recommendations.append(
                        f"Implement chunked processing for {bottleneck['function']} to reduce memory usage"
                    )
                elif bottleneck['type'] == 'memory_growth':
                    recommendations.append(
                        f"Investigate memory leaks in {bottleneck['function']} - potential object accumulation"
                    )
        
        # General recommendations
        recommendations.extend([
            "Consider implementing lazy evaluation for large data operations",
            "Optimize string concatenation using efficient methods",
            "Implement caching for frequently accessed data",
            "Use generators instead of lists for large datasets"
        ])
        
        return recommendations
    
    def generate_profiling_report(self, save_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        bottlenecks = self.analyze_bottlenecks()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_functions_profiled': len(self.profiling_results),
                'cpu_bottlenecks_found': len(bottlenecks['cpu_bottlenecks']),
                'memory_bottlenecks_found': len(bottlenecks['memory_bottlenecks']),
                'total_recommendations': len(bottlenecks['recommendations'])
            },
            'bottleneck_analysis': bottlenecks,
            'execution_statistics': self._generate_execution_statistics(),
            'memory_statistics': self._generate_memory_statistics()
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_execution_statistics(self) -> Dict[str, Any]:
        """Generate execution time statistics."""
        stats = {}
        
        for func_name, times in self.execution_times.items():
            if times:
                stats[func_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times) if len(times) > 1 else 0
                }
        
        return stats
    
    def _generate_memory_statistics(self) -> Dict[str, Any]:
        """Generate memory usage statistics."""
        stats = {}
        
        for func_name, memory_deltas in self.memory_usage_patterns.items():
            if memory_deltas:
                stats[func_name] = {
                    'count': len(memory_deltas),
                    'total_memory_delta': sum(memory_deltas),
                    'average_memory_delta': np.mean(memory_deltas),
                    'min_memory_delta': min(memory_deltas),
                    'max_memory_delta': max(memory_deltas),
                    'std_memory_delta': np.std(memory_deltas) if len(memory_deltas) > 1 else 0
                }
        
        return stats
    
    def visualize_performance_data(self, save_path: str = None) -> None:
        """Create visualizations of performance data."""
        if not HAS_VISUALIZATION:
            logger.warning("matplotlib not available, skipping performance visualization")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Execution time distribution
            if self.execution_times:
                all_times = []
                labels = []
                for func_name, times in self.execution_times.items():
                    all_times.extend(times)
                    labels.extend([func_name] * len(times))
                
                if all_times:
                    ax1 = axes[0, 0]
                    ax1.hist(all_times, bins=20, alpha=0.7)
                    ax1.set_title('Execution Time Distribution')
                    ax1.set_xlabel('Time (seconds)')
                    ax1.set_ylabel('Frequency')
            
            # Memory usage patterns
            if self.memory_usage_patterns:
                ax2 = axes[0, 1]
                for func_name, memory_deltas in self.memory_usage_patterns.items():
                    if memory_deltas:
                        ax2.plot(memory_deltas, label=func_name, marker='o')
                ax2.set_title('Memory Usage Patterns')
                ax2.set_xlabel('Call Number')
                ax2.set_ylabel('Memory Delta (MB)')
                ax2.legend()
            
            # Function performance comparison
            if self.execution_times:
                ax3 = axes[1, 0]
                func_names = list(self.execution_times.keys())
                avg_times = [np.mean(times) for times in self.execution_times.values()]
                ax3.bar(func_names, avg_times)
                ax3.set_title('Average Execution Time by Function')
                ax3.set_ylabel('Time (seconds)')
                ax3.tick_params(axis='x', rotation=45)
            
            # Memory usage comparison
            if self.memory_usage_patterns:
                ax4 = axes[1, 1]
                func_names = list(self.memory_usage_patterns.keys())
                avg_memory = [np.mean(memory_deltas) for memory_deltas in self.memory_usage_patterns.values()]
                ax4.bar(func_names, avg_memory)
                ax4.set_title('Average Memory Usage by Function')
                ax4.set_ylabel('Memory Delta (MB)')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available, skipping performance visualization")
    
    def reset_profiling_data(self) -> None:
        """Reset all profiling data."""
        self.profiling_results.clear()
        self.memory_snapshots.clear()
        self.execution_times.clear()
        self.memory_usage_patterns.clear()


class SystemResourceMonitor:
    """Monitor system resource usage during row ID generation."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.resource_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'io_read_bytes': [],
            'io_write_bytes': []
        }
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in a separate thread."""
        self.monitoring = True
        self.resource_data = {key: [] for key in self.resource_data.keys()}
        
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        return self.get_monitoring_summary()
    
    def _monitor_resources(self) -> None:
        """Monitor system resources in background."""
        try:
            import psutil
            process = psutil.Process()
            
            while self.monitoring:
                timestamp = time.time()
                
                # CPU and memory usage
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # I/O statistics
                try:
                    io_counters = process.io_counters()
                    io_read = io_counters.read_bytes
                    io_write = io_counters.write_bytes
                except AttributeError:
                    io_read = 0
                    io_write = 0
                
                # Store data
                self.resource_data['timestamps'].append(timestamp)
                self.resource_data['cpu_percent'].append(cpu_percent)
                self.resource_data['memory_percent'].append(memory_percent)
                self.resource_data['memory_mb'].append(memory_info.rss / (1024 * 1024))
                self.resource_data['io_read_bytes'].append(io_read)
                self.resource_data['io_write_bytes'].append(io_write)
                
                time.sleep(self.sampling_interval)
                
        except ImportError:
            logger.warning("psutil not available, resource monitoring disabled")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitored resources."""
        if not self.resource_data['timestamps']:
            return {'error': 'No monitoring data collected'}
        
        summary = {
            'duration_seconds': self.resource_data['timestamps'][-1] - self.resource_data['timestamps'][0],
            'cpu_stats': {
                'max_cpu_percent': max(self.resource_data['cpu_percent']),
                'avg_cpu_percent': np.mean(self.resource_data['cpu_percent']),
                'min_cpu_percent': min(self.resource_data['cpu_percent'])
            },
            'memory_stats': {
                'max_memory_mb': max(self.resource_data['memory_mb']),
                'avg_memory_mb': np.mean(self.resource_data['memory_mb']),
                'min_memory_mb': min(self.resource_data['memory_mb']),
                'max_memory_percent': max(self.resource_data['memory_percent']),
                'avg_memory_percent': np.mean(self.resource_data['memory_percent'])
            },
            'io_stats': {
                'total_read_bytes': max(self.resource_data['io_read_bytes']) - min(self.resource_data['io_read_bytes']),
                'total_write_bytes': max(self.resource_data['io_write_bytes']) - min(self.resource_data['io_write_bytes'])
            },
            'sample_count': len(self.resource_data['timestamps'])
        }
        
        return summary


def comprehensive_performance_analysis(
    test_data_sizes: List[int] = [1000, 10000, 100000, 500000],
    enable_profiling: bool = True,
    enable_resource_monitoring: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive performance analysis of the row ID generation system.
    
    Args:
        test_data_sizes: List of dataset sizes to test
        enable_profiling: Enable detailed function profiling
        enable_resource_monitoring: Enable system resource monitoring
        save_results: Save analysis results to files
    
    Returns:
        Dict containing comprehensive performance analysis results
    """
    # Initialize analyzers
    bottleneck_analyzer = PerformanceBottleneckAnalyzer(enable_memory_profiling=True)
    resource_monitor = SystemResourceMonitor() if enable_resource_monitoring else None
    
    # Results storage
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'test_configuration': {
            'test_data_sizes': test_data_sizes,
            'enable_profiling': enable_profiling,
            'enable_resource_monitoring': enable_resource_monitoring
        },
        'size_performance_results': {},
        'bottleneck_analysis': {},
        'resource_monitoring_results': {},
        'recommendations': []
    }
    
    # Test different data sizes
    for size in test_data_sizes:
        print(f"\n🔍 Testing performance with {size:,} rows...")
        
        # Generate test data
        test_df = pd.DataFrame({
            'id': range(size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'value': np.random.randn(size),
            'timestamp': pd.date_range(start='2023-01-01', periods=size, freq='1H')
        })
        
        # Start resource monitoring
        if resource_monitor:
            resource_monitor.start_monitoring()
        
        # Profile the main function
        if enable_profiling:
            profiled_func = bottleneck_analyzer.profile_function(f'generate_row_ids_{size}')(
                generate_unique_row_ids
            )
        else:
            profiled_func = generate_unique_row_ids
        
        # Execute and time the function
        start_time = time.time()
        try:
            result_df = profiled_func(
                test_df,
                columns=['id', 'category', 'value'],
                enable_monitoring=False,  # Disable internal monitoring to avoid interference
                show_progress=False
            )
            success = True
            rows_processed = len(result_df)
        except Exception as e:
            success = False
            rows_processed = 0
            logger.error(f"Error processing {size} rows: {e}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Stop resource monitoring
        resource_summary = {}
        if resource_monitor:
            resource_summary = resource_monitor.stop_monitoring()
        
        # Calculate performance metrics
        rows_per_second = rows_processed / execution_time if execution_time > 0 else 0
        
        # Store results
        analysis_results['size_performance_results'][size] = {
            'success': success,
            'execution_time_seconds': execution_time,
            'rows_processed': rows_processed,
            'rows_per_second': rows_per_second,
            'resource_summary': resource_summary
        }
        
        print(f"✅ Completed {size:,} rows in {execution_time:.2f}s ({rows_per_second:,.0f} rows/sec)")
    
    # Generate bottleneck analysis
    if enable_profiling:
        bottleneck_report = bottleneck_analyzer.generate_profiling_report()
        analysis_results['bottleneck_analysis'] = bottleneck_report
        analysis_results['recommendations'].extend(bottleneck_report['bottleneck_analysis']['recommendations'])
    
    # Add size-based recommendations
    performance_results = analysis_results['size_performance_results']
    if performance_results:
        slowest_rate = min(r['rows_per_second'] for r in performance_results.values() if r['success'])
        if slowest_rate < 10000:  # Less than 10K rows/second
            analysis_results['recommendations'].extend([
                "Performance below 10K rows/second detected - implement chunked processing",
                "Consider parallel processing for large datasets",
                "Optimize memory usage to prevent swapping"
            ])
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis report
        report_path = f"performance_analysis_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save profiling visualizations
        if enable_profiling:
            viz_path = f"performance_visualizations_{timestamp}.png"
            bottleneck_analyzer.visualize_performance_data(viz_path)
        
        print(f"\n📊 Results saved to: {report_path}")
    
    return analysis_results


# ========================================
# SUBTASK 7.2: CHUNKED PROCESSING SYSTEM
# ========================================

class ChunkProcessor:
    """
    Advanced chunked processing system for handling large datasets efficiently.
    Automatically determines optimal chunk sizes and processes data in batches.
    """
    
    def __init__(self, 
                 max_memory_mb: float = 500,
                 min_chunk_size: int = 1000,
                 max_chunk_size: int = 100000,
                 enable_parallel: bool = True,
                 max_workers: Optional[int] = None):
        self.max_memory_mb = max_memory_mb
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(4, mp.cpu_count())
        
        # Processing statistics
        self.processing_stats = {
            'total_chunks_processed': 0,
            'total_rows_processed': 0,
            'total_processing_time': 0,
            'average_chunk_time': 0,
            'chunks_failed': 0,
            'memory_usage_per_chunk': []
        }
        
        self.chunk_results = []
        self.failed_chunks = []
        
    def calculate_optimal_chunk_size(self, 
                                   df: pd.DataFrame,
                                   sample_size: int = 1000) -> int:
        """
        Calculate optimal chunk size based on data characteristics and memory constraints.
        
        Args:
            df: DataFrame to analyze
            sample_size: Size of sample to use for memory estimation
            
        Returns:
            Optimal chunk size
        """
        if len(df) <= self.min_chunk_size:
            return len(df)
        
        # Sample data to estimate memory usage
        sample_df = df.head(min(sample_size, len(df)))
        sample_memory_mb = sample_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Estimate memory per row
        memory_per_row_mb = sample_memory_mb / len(sample_df)
        
        # Calculate chunk size based on memory constraint
        memory_based_chunk_size = int(self.max_memory_mb / memory_per_row_mb)
        
        # Apply min/max constraints
        optimal_chunk_size = max(
            self.min_chunk_size,
            min(memory_based_chunk_size, self.max_chunk_size)
        )
        
        # Adjust for CPU efficiency (prefer chunk sizes that divide evenly)
        total_rows = len(df)
        if total_rows > optimal_chunk_size:
            num_chunks = total_rows // optimal_chunk_size
            if total_rows % optimal_chunk_size > 0:
                num_chunks += 1
            optimal_chunk_size = total_rows // num_chunks
        
        logger.info(f"Calculated optimal chunk size: {optimal_chunk_size:,} rows "
                   f"(estimated {memory_per_row_mb:.4f} MB/row)")
        
        return optimal_chunk_size
    
    def process_dataframe_in_chunks(self,
                                  df: pd.DataFrame,
                                  processing_func: Callable,
                                  chunk_size: Optional[int] = None,
                                  progress_callback: Optional[Callable] = None,
                                  **processing_kwargs) -> pd.DataFrame:
        """
        Process large DataFrame in chunks with automatic optimization.
        
        Args:
            df: DataFrame to process
            processing_func: Function to apply to each chunk
            chunk_size: Override automatic chunk size calculation
            progress_callback: Optional callback for progress updates
            **processing_kwargs: Additional arguments for processing function
            
        Returns:
            Processed DataFrame
        """
        # Calculate optimal chunk size
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(df)
        
        # Initialize processing
        total_rows = len(df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
        
        self.processing_stats['total_rows_processed'] = total_rows
        self.chunk_results = []
        self.failed_chunks = []
        
        logger.info(f"Processing {total_rows:,} rows in {num_chunks} chunks of ~{chunk_size:,} rows")
        
        # Process chunks
        start_time = time.time()
        
        if self.enable_parallel and num_chunks > 1:
            results = self._process_chunks_parallel(
                df, processing_func, chunk_size, progress_callback, **processing_kwargs
            )
        else:
            results = self._process_chunks_sequential(
                df, processing_func, chunk_size, progress_callback, **processing_kwargs
            )
        
        # Combine results
        if results:
            try:
                combined_df = pd.concat(results, ignore_index=True)
                self.processing_stats['total_chunks_processed'] = len(results)
                
                # Calculate statistics
                total_time = time.time() - start_time
                self.processing_stats['total_processing_time'] = total_time
                self.processing_stats['average_chunk_time'] = total_time / len(results)
                
                logger.info(f"✅ Chunked processing completed: {len(results)} chunks, "
                           f"{total_time:.2f}s total, {total_time/num_chunks:.2f}s/chunk avg")
                
                return combined_df
                
            except Exception as e:
                logger.error(f"Error combining chunk results: {e}")
                raise ChunkProcessingError(f"Failed to combine chunk results: {e}")
        else:
            raise ChunkProcessingError("No chunks were processed successfully")
    
    def _process_chunks_sequential(self,
                                 df: pd.DataFrame,
                                 processing_func: Callable,
                                 chunk_size: int,
                                 progress_callback: Optional[Callable],
                                 **processing_kwargs) -> List[pd.DataFrame]:
        """Process chunks sequentially."""
        results = []
        total_rows = len(df)
        
        for i in range(0, total_rows, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            
            try:
                # Process chunk
                chunk_start_time = time.time()
                result = self._process_single_chunk(
                    chunk_df, processing_func, chunk_start, **processing_kwargs
                )
                chunk_time = time.time() - chunk_start_time
                
                if result is not None:
                    results.append(result)
                    self.processing_stats['memory_usage_per_chunk'].append(
                        chunk_df.memory_usage(deep=True).sum() / (1024 * 1024)
                    )
                
                # Progress callback
                if progress_callback:
                    progress_callback(chunk_end, total_rows, chunk_time)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                self.failed_chunks.append({
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'error': str(e)
                })
                self.processing_stats['chunks_failed'] += 1
        
        return results
    
    def _process_chunks_parallel(self,
                               df: pd.DataFrame,
                               processing_func: Callable,
                               chunk_size: int,
                               progress_callback: Optional[Callable],
                               **processing_kwargs) -> List[pd.DataFrame]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        results = []
        total_rows = len(df)
        
        # Create chunks
        chunks = []
        for i in range(0, total_rows, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            chunks.append((chunk_df, chunk_start, processing_kwargs))
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_single_chunk, chunk_df, processing_func, chunk_start, **kwargs): 
                (chunk_start, chunk_start + len(chunk_df))
                for chunk_df, chunk_start, kwargs in chunks
            }
            
            # Collect results
            completed_chunks = 0
            for future in as_completed(future_to_chunk):
                chunk_start, chunk_end = future_to_chunk[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"Error processing parallel chunk {chunk_start}-{chunk_end}: {e}")
                    self.failed_chunks.append({
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                        'error': str(e)
                    })
                    self.processing_stats['chunks_failed'] += 1
                
                completed_chunks += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed_chunks * chunk_size, total_rows, 0)
        
        return results
    
    def _process_single_chunk(self,
                            chunk_df: pd.DataFrame,
                            processing_func: Callable,
                            chunk_start: int,
                            **processing_kwargs) -> Optional[pd.DataFrame]:
        """Process a single chunk with error handling."""
        try:
            # Add chunk metadata to processing kwargs
            processing_kwargs['chunk_info'] = {
                'chunk_start': chunk_start,
                'chunk_size': len(chunk_df),
                'chunk_index': chunk_start // len(chunk_df) if len(chunk_df) > 0 else 0
            }
            
            # Process the chunk
            result = processing_func(chunk_df, **processing_kwargs)
            
            # Validate result
            if result is None or len(result) == 0:
                logger.warning(f"Chunk {chunk_start} returned empty result")
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"Error in chunk {chunk_start}: {e}")
            raise ChunkProcessingError(f"Chunk processing failed at position {chunk_start}: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        stats = self.processing_stats.copy()
        
        # Calculate additional metrics
        if stats['memory_usage_per_chunk']:
            stats['average_memory_per_chunk_mb'] = np.mean(stats['memory_usage_per_chunk'])
            stats['max_memory_per_chunk_mb'] = max(stats['memory_usage_per_chunk'])
            stats['total_memory_processed_mb'] = sum(stats['memory_usage_per_chunk'])
        
        # Processing rate
        if stats['total_processing_time'] > 0:
            stats['rows_per_second'] = stats['total_rows_processed'] / stats['total_processing_time']
        
        # Success rate
        total_chunks = stats['total_chunks_processed'] + stats['chunks_failed']
        if total_chunks > 0:
            stats['success_rate'] = stats['total_chunks_processed'] / total_chunks
        
        return stats
    
    def get_failed_chunks_summary(self) -> Dict[str, Any]:
        """Get summary of failed chunks for debugging."""
        return {
            'total_failed': len(self.failed_chunks),
            'failed_chunks': self.failed_chunks,
            'failed_row_ranges': [(chunk['chunk_start'], chunk['chunk_end']) 
                                for chunk in self.failed_chunks]
        }


class ChunkProcessingError(Exception):
    """Custom exception for chunk processing errors."""
    pass


class AdaptiveChunkProcessor(ChunkProcessor):
    """
    Advanced chunk processor that adapts chunk size based on performance feedback.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
        self.adaptive_enabled = True
        self.performance_threshold = 0.8  # Minimum acceptable performance ratio
        
    def process_dataframe_in_chunks(self,
                                  df: pd.DataFrame,
                                  processing_func: Callable,
                                  chunk_size: Optional[int] = None,
                                  progress_callback: Optional[Callable] = None,
                                  **processing_kwargs) -> pd.DataFrame:
        """Process DataFrame with adaptive chunk sizing."""
        
        # Start with calculated or provided chunk size
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(df)
        
        initial_chunk_size = chunk_size
        
        # If adaptive processing is enabled and we have history
        if self.adaptive_enabled and self.performance_history:
            chunk_size = self._adapt_chunk_size(df, chunk_size)
        
        # Process with current chunk size
        start_time = time.time()
        result = super().process_dataframe_in_chunks(
            df, processing_func, chunk_size, progress_callback, **processing_kwargs
        )
        processing_time = time.time() - start_time
        
        # Record performance for future adaptation
        self._record_performance(len(df), chunk_size, processing_time)
        
        logger.info(f"Adaptive processing: initial={initial_chunk_size:,}, "
                   f"used={chunk_size:,}, time={processing_time:.2f}s")
        
        return result
    
    def _adapt_chunk_size(self, df: pd.DataFrame, initial_chunk_size: int) -> int:
        """Adapt chunk size based on performance history."""
        if not self.performance_history:
            return initial_chunk_size
        
        # Analyze recent performance
        recent_performance = self.performance_history[-5:]  # Last 5 runs
        avg_performance = np.mean([p['rows_per_second'] for p in recent_performance])
        
        # Find best performing chunk size
        best_performance = max(recent_performance, key=lambda x: x['rows_per_second'])
        
        # Adapt chunk size based on performance
        if avg_performance < self.performance_threshold * best_performance['rows_per_second']:
            # Performance is below threshold, try best chunk size
            adapted_size = best_performance['chunk_size']
            logger.info(f"Adapting chunk size from {initial_chunk_size:,} to {adapted_size:,} "
                       f"(performance below threshold)")
        else:
            # Performance is acceptable, use calculated size
            adapted_size = initial_chunk_size
        
        # Apply constraints
        return max(self.min_chunk_size, min(adapted_size, self.max_chunk_size))
    
    def _record_performance(self, total_rows: int, chunk_size: int, processing_time: float):
        """Record performance metrics for adaptation."""
        rows_per_second = total_rows / processing_time if processing_time > 0 else 0
        
        performance_record = {
            'timestamp': datetime.now(),
            'total_rows': total_rows,
            'chunk_size': chunk_size,
            'processing_time': processing_time,
            'rows_per_second': rows_per_second,
            'success_rate': self.get_processing_statistics().get('success_rate', 0)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history (last 20 runs)
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive processing performance."""
        if not self.performance_history:
            return {'message': 'No performance history available'}
        
        recent_performance = self.performance_history[-10:]
        
        return {
            'total_runs': len(self.performance_history),
            'recent_runs': len(recent_performance),
            'recent_avg_rows_per_second': np.mean([p['rows_per_second'] for p in recent_performance]),
            'recent_avg_chunk_size': np.mean([p['chunk_size'] for p in recent_performance]),
            'best_performance': max(self.performance_history, key=lambda x: x['rows_per_second']),
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend over recent runs."""
        if len(self.performance_history) < 3:
            return 'insufficient_data'
        
        recent_performance = [p['rows_per_second'] for p in self.performance_history[-5:]]
        
        # Simple trend analysis
        if len(recent_performance) >= 2:
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            
            if trend > 1000:  # Improving by >1000 rows/sec
                return 'improving'
            elif trend < -1000:  # Degrading by >1000 rows/sec
                return 'degrading'
            else:
                return 'stable'
        
        return 'stable'


def create_chunked_row_id_processor(
    df: pd.DataFrame,
    max_memory_mb: float = 500,
    enable_adaptive: bool = True,
    **processor_kwargs
) -> Union[ChunkProcessor, AdaptiveChunkProcessor]:
    """
    Factory function to create appropriate chunk processor for row ID generation.
    
    Args:
        df: DataFrame to process
        max_memory_mb: Maximum memory per chunk
        enable_adaptive: Whether to use adaptive chunk sizing
        **processor_kwargs: Additional processor arguments
        
    Returns:
        Configured chunk processor
    """
    processor_class = AdaptiveChunkProcessor if enable_adaptive else ChunkProcessor
    
    # Configure processor based on data characteristics
    data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Adjust settings based on data size
    if data_size_mb > 1000:  # Large dataset (>1GB)
        processor_kwargs.setdefault('max_memory_mb', min(max_memory_mb, 200))
        processor_kwargs.setdefault('enable_parallel', True)
        processor_kwargs.setdefault('max_workers', min(8, mp.cpu_count()))
    elif data_size_mb > 100:  # Medium dataset (>100MB)
        processor_kwargs.setdefault('max_memory_mb', min(max_memory_mb, 300))
        processor_kwargs.setdefault('enable_parallel', True)
        processor_kwargs.setdefault('max_workers', min(4, mp.cpu_count()))
    else:  # Small dataset
        processor_kwargs.setdefault('max_memory_mb', max_memory_mb)
        processor_kwargs.setdefault('enable_parallel', False)
    
    return processor_class(**processor_kwargs)


# ========================================
# SUBTASK 7.3: MEMORY-EFFICIENT CONCATENATION SYSTEM
# ========================================

class MemoryEfficientConcatenator:
    """
    Memory-efficient concatenation system for large data operations.
    Uses streaming, lazy evaluation, and optimized buffer management.
    """
    
    def __init__(self, 
                 buffer_size_mb: float = 100,
                 enable_streaming: bool = True,
                 compression_enabled: bool = False):
        self.buffer_size_mb = buffer_size_mb
        self.enable_streaming = enable_streaming
        self.compression_enabled = compression_enabled
        
        # Statistics
        self.concatenation_stats = {
            'total_concatenations': 0,
            'total_memory_saved_mb': 0,
            'peak_memory_usage_mb': 0,
            'streaming_operations': 0,
            'buffer_flushes': 0
        }
        
    def efficient_string_concatenation(self,
                                     df: pd.DataFrame,
                                     separator: str = '|',
                                     chunk_size: Optional[int] = None) -> pd.Series:
        """
        Memory-efficient string concatenation using numpy and streaming operations.
        
        Args:
            df: DataFrame to concatenate
            separator: Separator character
            chunk_size: Optional chunk size for processing
            
        Returns:
            Series of concatenated strings
        """
        if len(df) == 0:
            return pd.Series([], dtype=str)
        
        start_memory = self._get_memory_usage()
        
        # Use chunked processing for large datasets
        if chunk_size or len(df) > 50000:
            return self._chunked_string_concatenation(df, separator, chunk_size)
        
        # Use vectorized numpy operations for smaller datasets
        result = self._vectorized_string_concatenation(df, separator)
        
        # Update statistics
        end_memory = self._get_memory_usage()
        memory_used = max(0, end_memory - start_memory)
        self.concatenation_stats['total_concatenations'] += 1
        self.concatenation_stats['peak_memory_usage_mb'] = max(
            self.concatenation_stats['peak_memory_usage_mb'], memory_used
        )
        
        return result
    
    def _vectorized_string_concatenation(self, df: pd.DataFrame, separator: str) -> pd.Series:
        """Fast vectorized string concatenation using numpy."""
        # Convert to string array for efficient processing
        str_values = df.values.astype(str)
        
        # Use numpy's vectorized string join
        concatenated = np.apply_along_axis(
            lambda row: separator.join(row), axis=1, arr=str_values
        )
        
        return pd.Series(concatenated, index=df.index)
    
    def _chunked_string_concatenation(self,
                                    df: pd.DataFrame,
                                    separator: str,
                                    chunk_size: Optional[int] = None) -> pd.Series:
        """Memory-efficient chunked string concatenation."""
        if chunk_size is None:
            # Calculate chunk size based on memory constraints
            row_memory_estimate = df.memory_usage(deep=True).sum() / len(df)
            chunk_size = max(1000, int(self.buffer_size_mb * 1024 * 1024 / row_memory_estimate))
        
        results = []
        total_rows = len(df)
        
        for i in range(0, total_rows, chunk_size):
            chunk_end = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[i:chunk_end]
            
            # Process chunk
            chunk_result = self._vectorized_string_concatenation(chunk_df, separator)
            results.append(chunk_result)
            
            # Force garbage collection for large chunks
            if chunk_size > 10000:
                gc.collect()
        
        # Efficient concatenation of results
        return pd.concat(results, ignore_index=False)
    
    def streaming_dataframe_concatenation(self,
                                        dataframe_chunks: List[pd.DataFrame],
                                        buffer_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Stream-based DataFrame concatenation to minimize memory usage.
        
        Args:
            dataframe_chunks: List of DataFrame chunks to concatenate
            buffer_limit: Maximum number of chunks to hold in memory
            
        Returns:
            Concatenated DataFrame
        """
        if not dataframe_chunks:
            return pd.DataFrame()
        
        if len(dataframe_chunks) == 1:
            return dataframe_chunks[0]
        
        # Calculate buffer limit based on memory constraints
        if buffer_limit is None:
            avg_chunk_memory = self._estimate_chunk_memory(dataframe_chunks[0])
            buffer_limit = max(2, int(self.buffer_size_mb / avg_chunk_memory))
        
        # Use streaming concatenation for large numbers of chunks
        if len(dataframe_chunks) > buffer_limit:
            return self._streaming_concat_with_buffer(dataframe_chunks, buffer_limit)
        else:
            # Direct concatenation for smaller datasets
            return pd.concat(dataframe_chunks, ignore_index=True)
    
    def _streaming_concat_with_buffer(self,
                                    chunks: List[pd.DataFrame],
                                    buffer_limit: int) -> pd.DataFrame:
        """Concatenate chunks using a streaming buffer approach."""
        result_chunks = []
        current_buffer = []
        buffer_memory = 0
        
        for chunk in chunks:
            chunk_memory = self._estimate_chunk_memory(chunk)
            
            # Add chunk to buffer
            current_buffer.append(chunk)
            buffer_memory += chunk_memory
            
            # Flush buffer if limit reached
            if len(current_buffer) >= buffer_limit or buffer_memory >= self.buffer_size_mb:
                # Concatenate current buffer
                buffer_result = pd.concat(current_buffer, ignore_index=True)
                result_chunks.append(buffer_result)
                
                # Clear buffer and force garbage collection
                current_buffer = []
                buffer_memory = 0
                gc.collect()
                
                self.concatenation_stats['buffer_flushes'] += 1
        
        # Process remaining chunks in buffer
        if current_buffer:
            buffer_result = pd.concat(current_buffer, ignore_index=True)
            result_chunks.append(buffer_result)
        
        # Final concatenation
        if len(result_chunks) == 1:
            return result_chunks[0]
        else:
            return pd.concat(result_chunks, ignore_index=True)
    
    def _estimate_chunk_memory(self, chunk: pd.DataFrame) -> float:
        """Estimate memory usage of a DataFrame chunk in MB."""
        return chunk.memory_usage(deep=True).sum() / (1024 * 1024)
    
    def lazy_concatenation_generator(self,
                                   dataframe_chunks: List[pd.DataFrame],
                                   batch_size: int = 5) -> Iterator[pd.DataFrame]:
        """
        Lazy generator for concatenating chunks in batches.
        Useful for processing very large datasets without loading everything into memory.
        
        Args:
            dataframe_chunks: List of DataFrame chunks
            batch_size: Number of chunks to concatenate per batch
            
        Yields:
            Concatenated DataFrame batches
        """
        for i in range(0, len(dataframe_chunks), batch_size):
            batch_chunks = dataframe_chunks[i:i + batch_size]
            
            if batch_chunks:
                batch_result = pd.concat(batch_chunks, ignore_index=True)
                yield batch_result
                
                # Clean up after yielding
                del batch_chunks, batch_result
                gc.collect()
    
    def optimized_hash_concatenation(self,
                                   processed_df: pd.DataFrame,
                                   separator: str = '|',
                                   use_streaming: bool = None) -> pd.Series:
        """
        Optimized concatenation specifically for hash generation.
        Combines efficient string operations with memory management.
        
        Args:
            processed_df: DataFrame to process for hash generation
            separator: Separator for concatenation
            use_streaming: Override streaming setting
            
        Returns:
            Series of concatenated strings ready for hashing
        """
        use_streaming = use_streaming if use_streaming is not None else self.enable_streaming
        
        # For hash generation, we can use more aggressive optimizations
        if use_streaming and len(processed_df) > 25000:
            return self._streaming_hash_concatenation(processed_df, separator)
        else:
            return self._optimized_batch_concatenation(processed_df, separator)
    
    def _streaming_hash_concatenation(self, df: pd.DataFrame, separator: str) -> pd.Series:
        """Streaming concatenation optimized for hash generation."""
        # Calculate optimal chunk size for hash generation
        estimated_row_size = sum(df.dtypes.apply(lambda x: 8 if x == 'object' else 4))
        chunk_size = max(5000, int(self.buffer_size_mb * 1024 * 1024 / estimated_row_size))
        
        hash_results = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            
            # Optimize for hash generation - convert directly to strings
            chunk_str = chunk.astype(str)
            concatenated = chunk_str.apply(lambda row: separator.join(row), axis=1)
            hash_results.append(concatenated)
            
            # Memory management for large chunks
            if chunk_size > 10000:
                del chunk, chunk_str, concatenated
                gc.collect()
        
        self.concatenation_stats['streaming_operations'] += 1
        return pd.concat(hash_results, ignore_index=False)
    
    def _optimized_batch_concatenation(self, df: pd.DataFrame, separator: str) -> pd.Series:
        """Optimized batch concatenation for smaller datasets."""
        # Use the most efficient method available
        return self._vectorized_string_concatenation(df, separator)
    
    def memory_efficient_result_combination(self,
                                          chunk_results: List[pd.DataFrame],
                                          final_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Efficiently combine chunk results with memory optimization.
        
        Args:
            chunk_results: List of processed DataFrame chunks
            final_columns: Optional list of columns for final result
            
        Returns:
            Combined DataFrame
        """
        if not chunk_results:
            return pd.DataFrame()
        
        # Estimate total memory requirements
        total_estimated_memory = sum(self._estimate_chunk_memory(chunk) for chunk in chunk_results)
        
        # Choose combination strategy based on memory requirements
        if total_estimated_memory > self.buffer_size_mb * 2:
            # Use streaming approach for large results
            result = self.streaming_dataframe_concatenation(chunk_results)
        else:
            # Direct concatenation for smaller results
            result = pd.concat(chunk_results, ignore_index=True)
        
        # Apply column selection if specified
        if final_columns:
            available_columns = [col for col in final_columns if col in result.columns]
            result = result[available_columns]
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_concatenation_statistics(self) -> Dict[str, Any]:
        """Get concatenation performance statistics."""
        stats = self.concatenation_stats.copy()
        
        # Calculate efficiency metrics
        if stats['total_concatenations'] > 0:
            stats['average_memory_per_operation'] = (
                stats['peak_memory_usage_mb'] / stats['total_concatenations']
            )
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset concatenation statistics."""
        self.concatenation_stats = {
            'total_concatenations': 0,
            'total_memory_saved_mb': 0,
            'peak_memory_usage_mb': 0,
            'streaming_operations': 0,
            'buffer_flushes': 0
        }


class OptimizedStringOperations:
    """
    Collection of optimized string operations for large-scale data processing.
    """
    
    @staticmethod
    def fast_string_join(series_list: List[pd.Series], separator: str = '|') -> pd.Series:
        """
        Fast string joining using vectorized operations.
        
        Args:
            series_list: List of pandas Series to join
            separator: Separator character
            
        Returns:
            Series with joined strings
        """
        if not series_list:
            return pd.Series([], dtype=str)
        
        if len(series_list) == 1:
            return series_list[0].astype(str)
        
        # Convert all series to string and align indices
        str_series = [s.astype(str) for s in series_list]
        
        # Use pandas string operations for efficiency
        result = str_series[0]
        for s in str_series[1:]:
            result = result.str.cat(s, sep=separator)
        
        return result
    
    @staticmethod
    def chunked_string_processing(text_series: pd.Series,
                                processing_func: Callable,
                                chunk_size: int = 10000) -> pd.Series:
        """
        Process string series in chunks to manage memory.
        
        Args:
            text_series: Series of strings to process
            processing_func: Function to apply to each chunk
            chunk_size: Size of each processing chunk
            
        Returns:
            Processed series
        """
        if len(text_series) <= chunk_size:
            return processing_func(text_series)
        
        results = []
        for i in range(0, len(text_series), chunk_size):
            chunk = text_series.iloc[i:i + chunk_size]
            processed_chunk = processing_func(chunk)
            results.append(processed_chunk)
        
        return pd.concat(results, ignore_index=False)
    
    @staticmethod
    def memory_aware_string_concatenation(df: pd.DataFrame,
                                        columns: List[str],
                                        separator: str = '|',
                                        max_memory_mb: float = 200) -> pd.Series:
        """
        Memory-aware string concatenation with automatic chunking.
        
        Args:
            df: DataFrame containing columns to concatenate
            columns: List of column names to concatenate
            separator: Separator character
            max_memory_mb: Maximum memory to use for concatenation
            
        Returns:
            Series of concatenated strings
        """
        # Estimate memory requirements
        subset_df = df[columns]
        estimated_memory = subset_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if estimated_memory <= max_memory_mb:
            # Direct concatenation for small datasets
            return subset_df.apply(lambda row: separator.join(row.astype(str)), axis=1)
        else:
            # Chunked processing for large datasets
            chunk_size = max(1000, int(len(df) * max_memory_mb / estimated_memory))
            
            concatenator = MemoryEfficientConcatenator(buffer_size_mb=max_memory_mb)
            return concatenator.efficient_string_concatenation(subset_df, separator, chunk_size)


def create_optimized_row_id_function(max_memory_mb: float = 300,
                                    enable_chunking: bool = True,
                                    enable_streaming: bool = True) -> Callable:
    """
    Create an optimized row ID generation function with memory-efficient concatenation.
    
    Args:
        max_memory_mb: Maximum memory to use for operations
        enable_chunking: Enable chunked processing
        enable_streaming: Enable streaming concatenation
        
    Returns:
        Optimized row ID generation function
    """
    # Initialize components
    concatenator = MemoryEfficientConcatenator(
        buffer_size_mb=max_memory_mb,
        enable_streaming=enable_streaming
    )
    
    if enable_chunking:
        chunk_processor = AdaptiveChunkProcessor(max_memory_mb=max_memory_mb)
    else:
        chunk_processor = None
    
    def optimized_generate_row_ids(df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 separator: str = '|',
                                 chunk_size: Optional[int] = None,
                                 **kwargs) -> pd.DataFrame:
        """
        Optimized row ID generation with memory-efficient processing.
        
        Args:
            df: Input DataFrame
            columns: Columns to use for ID generation
            separator: Separator for concatenation
            chunk_size: Optional chunk size override
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with generated row IDs
        """
        # Select columns
        if columns is None:
            columns = df.columns.tolist()
        
        selected_df = df[columns]
        
        # Choose processing strategy based on data size
        data_size_mb = selected_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if data_size_mb > max_memory_mb and chunk_processor:
            # Use chunked processing for large datasets
            def chunk_processing_func(chunk_df, **chunk_kwargs):
                # Generate concatenated strings for chunk
                concatenated = concatenator.optimized_hash_concatenation(chunk_df, separator)
                
                # Generate hashes
                hashes = concatenated.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
                
                # Return chunk with row IDs
                result_chunk = chunk_df.copy()
                result_chunk['row_id'] = hashes
                return result_chunk
            
            result_df = chunk_processor.process_dataframe_in_chunks(
                selected_df, chunk_processing_func, chunk_size
            )
        else:
            # Direct processing for smaller datasets
            concatenated = concatenator.optimized_hash_concatenation(selected_df, separator)
            hashes = concatenated.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
            
            result_df = df.copy()
            result_df['row_id'] = hashes
        
        return result_df
    
    return optimized_generate_row_ids


# ========================================
# Task 7 Performance Optimization - END
# ========================================