"""
Metrics collection system for the Row ID Generator system.

Provides comprehensive metrics tracking including counters, gauges, histograms,
and time-series data with tagging support for detailed performance analysis.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from statistics import mean, median, stdev
from enum import Enum
import json


class MetricType(Enum):
    """Types of metrics supported by the system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """
    Individual metric data point with timestamp and tags.
    
    Attributes:
        timestamp: When the metric was recorded (Unix timestamp)
        value: The numeric value of the metric
        tags: Optional key-value pairs for categorization
        metric_type: Type of metric (counter, gauge, histogram, timer)
    """
    timestamp: float
    value: float
    metric_type: MetricType
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Ensure tags is always a dict."""
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def matches_tags(self, filter_tags: Dict[str, str]) -> bool:
        """
        Check if this metric point matches the given tag filters.
        
        Args:
            filter_tags: Tags to filter by
            
        Returns:
            True if all filter tags match this point's tags
        """
        if not filter_tags:
            return True
        
        for key, value in filter_tags.items():
            if self.tags.get(key) != value:
                return False
        return True


@dataclass
class MetricSummary:
    """Summary statistics for a collection of metric points."""
    name: str
    metric_type: MetricType
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    latest_value: float
    latest_timestamp: float
    tags_summary: Dict[str, List[str]] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Features:
    - Counter metrics for counting events
    - Gauge metrics for current values
    - Histogram metrics for value distributions
    - Timer metrics for duration tracking
    - Tag-based filtering and aggregation
    - Thread-safe operations
    - Automatic retention management
    - Summary statistics calculation
    """
    
    def __init__(self, max_points_per_metric: int = 10000, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            max_points_per_metric: Maximum data points to keep per metric
            retention_hours: Hours to retain metric data
        """
        self.max_points_per_metric = max_points_per_metric
        self.retention_seconds = retention_hours * 3600
        
        # Thread-safe storage for metric points
        self._lock = threading.RLock()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_points_per_metric))
        
        # Quick access for current values
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        
        # Metric metadata
        self._metric_types: Dict[str, MetricType] = {}
        
        # Statistics cache
        self._stats_cache: Dict[str, MetricSummary] = {}
        self._stats_cache_timestamp = 0
        self._stats_cache_ttl = 60  # Cache for 60 seconds
    
    def increment_counter(
        self, 
        name: str, 
        value: float = 1.0, 
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to add (default: 1.0)
            tags: Optional tags for categorization
            timestamp: Optional timestamp (default: current time)
        """
        timestamp = timestamp or time.time()
        
        with self._lock:
            self._counters[name] += value
            self._metric_types[name] = MetricType.COUNTER
            
            point = MetricPoint(
                timestamp=timestamp,
                value=self._counters[name],
                metric_type=MetricType.COUNTER,
                tags=tags
            )
            
            self._metrics[name].append(point)
            self._cleanup_old_points(name)
            self._invalidate_stats_cache()
    
    def set_gauge(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags for categorization
            timestamp: Optional timestamp (default: current time)
        """
        timestamp = timestamp or time.time()
        
        with self._lock:
            self._gauges[name] = value
            self._metric_types[name] = MetricType.GAUGE
            
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                metric_type=MetricType.GAUGE,
                tags=tags
            )
            
            self._metrics[name].append(point)
            self._cleanup_old_points(name)
            self._invalidate_stats_cache()
    
    def record_histogram(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Record a value in a histogram metric.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags for categorization
            timestamp: Optional timestamp (default: current time)
        """
        timestamp = timestamp or time.time()
        
        with self._lock:
            self._metric_types[name] = MetricType.HISTOGRAM
            
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                tags=tags
            )
            
            self._metrics[name].append(point)
            self._cleanup_old_points(name)
            self._invalidate_stats_cache()
    
    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> 'TimerContext':
        """
        Start a timer context manager.
        
        Args:
            name: Timer name
            tags: Optional tags for categorization
            
        Returns:
            Timer context manager
        """
        return TimerContext(self, name, tags)
    
    def record_timer(
        self, 
        name: str, 
        duration: float, 
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Record a timer duration.
        
        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Optional tags for categorization
            timestamp: Optional timestamp (default: current time)
        """
        timestamp = timestamp or time.time()
        
        with self._lock:
            self._metric_types[name] = MetricType.TIMER
            
            point = MetricPoint(
                timestamp=timestamp,
                value=duration,
                metric_type=MetricType.TIMER,
                tags=tags
            )
            
            self._metrics[name].append(point)
            self._cleanup_old_points(name)
            self._invalidate_stats_cache()
    
    def get_metric_points(
        self, 
        name: str, 
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[MetricPoint]:
        """
        Get metric points for a specific metric.
        
        Args:
            name: Metric name
            tags: Optional tag filters
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            List of matching metric points
        """
        with self._lock:
            if name not in self._metrics:
                return []
            
            points = list(self._metrics[name])
            
            # Apply filters
            if tags:
                points = [p for p in points if p.matches_tags(tags)]
            
            if start_time:
                points = [p for p in points if p.timestamp >= start_time]
            
            if end_time:
                points = [p for p in points if p.timestamp <= end_time]
            
            return points
    
    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            tags: Optional tag filters
            
        Returns:
            MetricSummary or None if metric doesn't exist
        """
        points = self.get_metric_points(name, tags)
        if not points:
            return None
        
        values = [p.value for p in points]
        
        # Calculate all unique tag combinations
        tags_summary = defaultdict(set)
        for point in points:
            for key, value in point.tags.items():
                tags_summary[key].add(value)
        
        # Convert sets to lists for serialization
        tags_summary = {k: list(v) for k, v in tags_summary.items()}
        
        return MetricSummary(
            name=name,
            metric_type=points[0].metric_type,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=mean(values),
            median_value=median(values),
            std_dev=stdev(values) if len(values) > 1 else 0.0,
            latest_value=points[-1].value,
            latest_timestamp=points[-1].timestamp,
            tags_summary=tags_summary
        )
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all metric names."""
        with self._lock:
            return list(self._metrics.keys())
    
    def get_current_values(self) -> Dict[str, Any]:
        """
        Get current values for all metrics.
        
        Returns:
            Dictionary with current values organized by type
        """
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'metrics_count': len(self._metrics),
                'total_points': sum(len(points) for points in self._metrics.values())
            }
    
    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            String in Prometheus exposition format
        """
        lines = []
        
        with self._lock:
            for metric_name in self._metrics.keys():
                summary = self.get_metric_summary(metric_name)
                if not summary:
                    continue
                
                # Add help and type information
                lines.append(f"# HELP {metric_name} {summary.metric_type.value} metric")
                lines.append(f"# TYPE {metric_name} {summary.metric_type.value}")
                
                # Add the metric value
                if summary.metric_type in [MetricType.COUNTER, MetricType.GAUGE]:
                    lines.append(f"{metric_name} {summary.latest_value}")
                elif summary.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                    # For histograms and timers, export summary statistics
                    lines.append(f"{metric_name}_count {summary.count}")
                    lines.append(f"{metric_name}_sum {summary.mean_value * summary.count}")
                    lines.append(f"{metric_name}_min {summary.min_value}")
                    lines.append(f"{metric_name}_max {summary.max_value}")
                    lines.append(f"{metric_name}_mean {summary.mean_value}")
                    lines.append(f"{metric_name}_median {summary.median_value}")
        
        return '\n'.join(lines) + '\n'
    
    def export_json(self, include_points: bool = False) -> str:
        """
        Export metrics in JSON format.
        
        Args:
            include_points: Whether to include all data points
            
        Returns:
            JSON string with metrics data
        """
        data = {
            'timestamp': time.time(),
            'metrics': {},
            'summary': self.get_current_values()
        }
        
        with self._lock:
            for metric_name in self._metrics.keys():
                summary = self.get_metric_summary(metric_name)
                if not summary:
                    continue
                
                metric_data = asdict(summary)
                
                if include_points:
                    points = self.get_metric_points(metric_name)
                    metric_data['points'] = [p.to_dict() for p in points]
                
                data['metrics'][metric_name] = metric_data
        
        return json.dumps(data, indent=2, default=str)
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """
        Clear metrics data.
        
        Args:
            name: Optional specific metric to clear (clears all if None)
        """
        with self._lock:
            if name:
                if name in self._metrics:
                    self._metrics[name].clear()
                    self._counters.pop(name, None)
                    self._gauges.pop(name, None)
                    self._metric_types.pop(name, None)
            else:
                self._metrics.clear()
                self._counters.clear()
                self._gauges.clear()
                self._metric_types.clear()
            
            self._invalidate_stats_cache()
    
    def _cleanup_old_points(self, name: str) -> None:
        """Remove old metric points based on retention policy."""
        if name not in self._metrics:
            return
        
        cutoff_time = time.time() - self.retention_seconds
        points = self._metrics[name]
        
        # Remove old points from the left
        while points and points[0].timestamp < cutoff_time:
            points.popleft()
    
    def _invalidate_stats_cache(self) -> None:
        """Invalidate the statistics cache."""
        self._stats_cache.clear()
        self._stats_cache_timestamp = 0


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.metrics = metrics_collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_timer(self.name, duration, self.tags)


# Convenience functions for quick metric creation
def create_metrics_collector(max_points: int = 10000, retention_hours: int = 24) -> MetricsCollector:
    """
    Create a metrics collector with specified configuration.
    
    Args:
        max_points: Maximum points per metric
        retention_hours: Hours to retain data
        
    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(max_points, retention_hours) 