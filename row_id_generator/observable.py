"""
Observable wrapper for the Row ID Generator system.

This module provides an ObservableHashingEngine that integrates all observability
components (logging, metrics, monitoring, alerting, dashboards) with the existing
row ID generation functionality for comprehensive system visibility.
"""

import time
import pandas as pd
import psutil
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

# Import existing core functionality
from .core import (
    generate_unique_row_ids,
    generate_row_hash,
    generate_row_ids_vectorized,
    prepare_for_snowflake,
    load_to_snowflake
)
from .utils import (
    select_columns_for_hashing,
    prepare_data_for_hashing,
    normalize_string_data,
    handle_null_values,
    standardize_datetime,
    normalize_numeric_data
)

# Import observability components
from observability import (
    StructuredLogger,
    MetricsCollector,
    PerformanceMonitor,
    AlertManager,
    DashboardGenerator,
    ObservabilityConfig,
    AlertSeverity,
    AlertConditions,
    create_metrics_collector,
    create_performance_monitor,
    create_alert_manager,
    create_dashboard_generator,
    load_config
)


@dataclass
class ObservabilityMetrics:
    """Metrics collected during row ID generation operations."""
    operation_count: int = 0
    total_rows_processed: int = 0
    total_columns_analyzed: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_seconds: float = 0.0
    average_rows_per_second: float = 0.0
    collision_count: int = 0
    null_values_processed: int = 0
    data_preprocessing_time: float = 0.0
    hashing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for export."""
        return asdict(self)


class ObservableHashingEngine:
    """
    Observable wrapper for the Row ID Generator system.
    
    Provides comprehensive observability for all row ID generation operations
    including structured logging, metrics collection, performance monitoring,
    alerting, and dashboard capabilities.
    
    Features:
    - Automatic operation tracking and timing
    - Performance monitoring with system resource usage
    - Comprehensive metrics collection
    - Alert management with configurable rules
    - Dashboard generation for system visibility
    - Configuration-driven observability settings
    - Thread-safe operations
    - Error tracking and reporting
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        enable_monitoring: bool = True,
        enable_alerting: bool = True,
        enable_dashboards: bool = True
    ):
        """
        Initialize the observable hashing engine.
        
        Args:
            config_path: Path to observability configuration file
            enable_monitoring: Enable performance monitoring
            enable_alerting: Enable alerting system
            enable_dashboards: Enable dashboard generation
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else ObservabilityConfig()
        
        # Initialize observability components
        self.logger = StructuredLogger(
            name="observable_hashing_engine",
            level=self.config.get("logging.level", "INFO")
        )
        
        self.metrics = create_metrics_collector(
            max_points=self.config.get("metrics.max_points_per_metric", 10000),
            retention_hours=self.config.get("metrics.retention_days", 30) * 24  # Convert days to hours
        )
        
        self.monitor = create_performance_monitor(
            metrics_collector=self.metrics,
            enable_system_monitoring=True
        ) if enable_monitoring else None
        
        self.alert_manager = create_alert_manager() if enable_alerting else None
        
        self.dashboard = create_dashboard_generator(
            metrics_collector=self.metrics,
            performance_monitor=self.monitor,
            alert_manager=self.alert_manager
        ) if enable_dashboards else None
        
        # Internal state
        self._session_metrics = ObservabilityMetrics()
        self._operation_lock = threading.RLock()
        
        # Setup alert rules
        if self.alert_manager:
            self._setup_alert_rules()
        
        # Log initialization
        self.logger.info("ObservableHashingEngine initialized", {
            "monitoring_enabled": enable_monitoring,
            "alerting_enabled": enable_alerting,
            "dashboards_enabled": enable_dashboards,
            "config_loaded": config_path is not None
        })
    
    def generate_unique_row_ids(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        id_column_name: str = 'row_id',
        uniqueness_threshold: float = 0.95,
        separator: str = '|',
        show_progress: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Generate unique row IDs with comprehensive observability.
        
        Args:
            df: Input pandas DataFrame
            columns: Optional list of columns to use for hashing
            id_column_name: Name for the generated ID column
            uniqueness_threshold: Minimum uniqueness ratio for auto-selected columns
            separator: String separator for concatenating values
            show_progress: Whether to show progress indicators
            **kwargs: Additional arguments passed to core function
            
        Returns:
            Tuple of (DataFrame with row IDs, list of selected columns, audit trail)
        """
        operation_name = "generate_unique_row_ids"
        
        with self._track_operation(operation_name, df=df, columns=columns):
            try:
                # Log operation start
                self.logger.info(f"Starting {operation_name}", {
                    "input_rows": len(df),
                    "input_columns": len(df.columns),
                    "target_columns": columns,
                    "uniqueness_threshold": uniqueness_threshold,
                    "separator": separator
                })
                
                # Update metrics
                self._update_session_metrics("rows_processed", len(df))
                self._update_session_metrics("columns_analyzed", len(df.columns))
                
                # Call core function
                result = generate_unique_row_ids(
                    df=df,
                    columns=columns,
                    id_column_name=id_column_name,
                    uniqueness_threshold=uniqueness_threshold,
                    separator=separator,
                    show_progress=show_progress,
                    enable_monitoring=True,
                    return_audit_trail=True,  # Ensure we get the full dictionary
                    **kwargs
                )
                
                # Extract values from the returned dictionary
                if isinstance(result, dict):
                    result_df = result['result_dataframe']
                    selected_columns = result['column_selection']['selected_columns']
                    audit_trail = result['audit_trail']
                else:
                    # Fallback for unexpected return type
                    result_df = result
                    selected_columns = list(df.columns)
                    audit_trail = {}
                
                # Analyze results and collect metrics
                self._analyze_generation_results(result_df, selected_columns, audit_trail)
                
                # Check for alerts
                if self.alert_manager:
                    self._check_generation_alerts(df, result_df, selected_columns)
                
                # Log success
                self.logger.info(f"Completed {operation_name}", {
                    "output_rows": len(result_df),
                    "selected_columns": selected_columns,
                    "id_column": id_column_name,
                    "audit_trail": audit_trail
                })
                
                self._update_session_metrics("successful_operations", 1)
                return result_df, selected_columns, audit_trail
                
            except Exception as e:
                self._handle_operation_error(operation_name, e, df=df, columns=columns)
                raise
    
    def select_columns_for_hashing(
        self,
        df: pd.DataFrame,
        manual_columns: Optional[List[str]] = None,
        uniqueness_threshold: float = 0.95,
        include_email: bool = True
    ) -> List[str]:
        """
        Select columns for hashing with observability tracking.
        
        Args:
            df: Input pandas DataFrame
            manual_columns: Optional list of manually specified columns
            uniqueness_threshold: Minimum uniqueness ratio for column selection
            include_email: Whether to prioritize email columns
            
        Returns:
            List of selected column names
        """
        operation_name = "select_columns_for_hashing"
        
        with self._track_operation(operation_name, df=df):
            try:
                # Log operation start
                self.logger.info(f"Starting {operation_name}", {
                    "input_columns": len(df.columns),
                    "manual_columns": manual_columns,
                    "uniqueness_threshold": uniqueness_threshold,
                    "include_email": include_email
                })
                
                # Call core function
                selected_columns = select_columns_for_hashing(
                    df=df,
                    manual_columns=manual_columns,
                    uniqueness_threshold=uniqueness_threshold,
                    include_email=include_email
                )
                
                # Update metrics
                self.metrics.increment_counter("columns_selected", len(selected_columns))
                self.metrics.set_gauge("selection_ratio", len(selected_columns) / len(df.columns))
                
                # Check for alerts
                if self.alert_manager:
                    self._check_column_selection_alerts(df, selected_columns, uniqueness_threshold)
                
                # Log success
                self.logger.info(f"Completed {operation_name}", {
                    "selected_columns": selected_columns,
                    "selection_count": len(selected_columns),
                    "selection_ratio": len(selected_columns) / len(df.columns)
                })
                
                return selected_columns
                
            except Exception as e:
                self._handle_operation_error(operation_name, e, df=df)
                raise
    
    def prepare_data_for_hashing(
        self,
        df: pd.DataFrame,
        columns: List[str],
        normalize_strings: bool = True,
        handle_nulls: bool = True,
        standardize_dates: bool = True,
        normalize_numbers: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for hashing with comprehensive observability.
        
        Args:
            df: Input pandas DataFrame
            columns: List of columns to prepare for hashing
            normalize_strings: Whether to normalize string data
            handle_nulls: Whether to handle NULL values
            standardize_dates: Whether to standardize datetime data
            normalize_numbers: Whether to normalize numeric data
            
        Returns:
            DataFrame with preprocessed data ready for hashing
        """
        operation_name = "prepare_data_for_hashing"
        
        with self._track_operation(operation_name, df=df):
            try:
                start_time = time.time()
                
                # Log operation start
                self.logger.info(f"Starting {operation_name}", {
                    "input_rows": len(df),
                    "target_columns": columns,
                    "normalize_strings": normalize_strings,
                    "handle_nulls": handle_nulls,
                    "standardize_dates": standardize_dates,
                    "normalize_numbers": normalize_numbers
                })
                
                # Call core function (enhanced with observability)
                processed_df = self._prepare_data_with_tracking(
                    df, columns, normalize_strings, handle_nulls, 
                    standardize_dates, normalize_numbers
                )
                
                # Calculate preprocessing time
                preprocessing_time = time.time() - start_time
                self._update_session_metrics("data_preprocessing_time", preprocessing_time)
                
                # Update metrics
                self.metrics.record_histogram("preprocessing_duration", preprocessing_time)
                self.metrics.increment_counter("rows_preprocessed", len(processed_df))
                
                # Log success
                self.logger.info(f"Completed {operation_name}", {
                    "output_rows": len(processed_df),
                    "processing_time_seconds": preprocessing_time,
                    "columns_processed": len(columns)
                })
                
                return processed_df
                
            except Exception as e:
                self._handle_operation_error(operation_name, e, df=df, columns=columns)
                raise
    
    def generate_performance_dashboard(self) -> str:
        """
        Generate a comprehensive performance dashboard.
        
        Returns:
            HTML dashboard content
        """
        if not self.dashboard:
            raise RuntimeError("Dashboard generation is disabled")
        
        self.logger.info("Generating performance dashboard")
        
        # Generate dashboard
        dashboard_html = self.dashboard.generate_performance_dashboard()
        
        # Log dashboard generation
        self.logger.info("Performance dashboard generated", {
            "dashboard_size_chars": len(dashboard_html),
            "session_metrics": self._session_metrics.to_dict()
        })
        
        return dashboard_html
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system health report.
        
        Returns:
            System health metrics and status
        """
        health_report = {
            "timestamp": time.time(),
            "session_metrics": self._session_metrics.to_dict(),
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "metrics_summary": self.metrics.get_current_values(),
            "active_alerts": self.alert_manager.get_active_alerts() if self.alert_manager else [],
            "configuration": {
                "monitoring_enabled": self.monitor is not None,
                "alerting_enabled": self.alert_manager is not None,
                "dashboard_enabled": self.dashboard is not None
            }
        }
        
        self.logger.info("System health report generated", {
            "cpu_percent": health_report["system_resources"]["cpu_percent"],
            "memory_percent": health_report["system_resources"]["memory_percent"],
            "active_alerts_count": len(health_report["active_alerts"])
        })
        
        return health_report
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """
        Export current metrics in the specified format.
        
        Args:
            format_type: Export format ('prometheus' or 'json')
            
        Returns:
            Formatted metrics string
        """
        self.logger.info(f"Exporting metrics in {format_type} format")
        
        if format_type.lower() == "prometheus":
            return self.metrics.export_prometheus_format()
        elif format_type.lower() == "json":
            return self.metrics.export_json(include_points=True)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session metrics.
        
        Returns:
            Session summary dictionary
        """
        summary = self._session_metrics.to_dict()
        
        # Calculate derived metrics
        if summary["operation_count"] > 0:
            summary["success_rate"] = summary["successful_operations"] / summary["operation_count"]
            summary["failure_rate"] = summary["failed_operations"] / summary["operation_count"]
        
        if summary["total_duration_seconds"] > 0:
            summary["operations_per_second"] = summary["operation_count"] / summary["total_duration_seconds"]
        
        return summary
    
    @contextmanager
    def _track_operation(self, operation_name: str, **context):
        """
        Track operation with comprehensive monitoring.
        
        Args:
            operation_name: Name of the operation
            **context: Additional context for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Create child logger for this operation
        operation_logger = self.logger.create_child_logger(f"operation_{operation_id}")
        
        # Performance monitoring context
        monitor_context = None
        if self.monitor:
            monitor_context = self.monitor.track_operation(operation_name)
            monitor_context.__enter__()
        
        try:
            # Update operation count
            with self._operation_lock:
                self._session_metrics.operation_count += 1
            
            # Record operation start
            self.metrics.increment_counter(f"{operation_name}.started")
            operation_logger.info(f"Operation started: {operation_name}", context)
            
            yield operation_logger
            
        except Exception as e:
            # Record operation failure
            self.metrics.increment_counter(f"{operation_name}.failed")
            operation_logger.error(f"Operation failed: {operation_name}", {"error": str(e), **context})
            
            with self._operation_lock:
                self._session_metrics.failed_operations += 1
            
            raise
            
        else:
            # Record operation success
            self.metrics.increment_counter(f"{operation_name}.completed")
            operation_logger.info(f"Operation completed: {operation_name}", context)
            
            with self._operation_lock:
                self._session_metrics.successful_operations += 1
            
        finally:
            # Calculate duration
            duration = time.time() - start_time
            self.metrics.record_histogram(f"{operation_name}.duration", duration)
            
            with self._operation_lock:
                self._session_metrics.total_duration_seconds += duration
            
            # Close performance monitoring
            if monitor_context:
                monitor_context.__exit__(None, None, None)
    
    def _prepare_data_with_tracking(
        self,
        df: pd.DataFrame,
        columns: List[str],
        normalize_strings: bool,
        handle_nulls: bool,
        standardize_dates: bool,
        normalize_numbers: bool
    ) -> pd.DataFrame:
        """
        Prepare data with detailed tracking of each preprocessing step.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            normalize_strings: Whether to normalize strings
            handle_nulls: Whether to handle nulls
            standardize_dates: Whether to standardize dates
            normalize_numbers: Whether to normalize numbers
            
        Returns:
            Processed DataFrame
        """
        # Start with core function
        processed_df = prepare_data_for_hashing(df, columns)
        
        # Apply additional preprocessing with tracking
        null_count = 0
        
        for column in columns:
            if column not in processed_df.columns:
                continue
                
            series = processed_df[column]
            original_nulls = series.isnull().sum()
            null_count += original_nulls
            
            # Apply preprocessing based on data type and flags
            if normalize_strings and series.dtype == 'object':
                try:
                    processed_df[column] = normalize_string_data(series)
                    self.metrics.increment_counter("string_normalization.completed")
                except Exception as e:
                    self.logger.warning(f"String normalization failed for column {column}", {"error": str(e)})
                    self.metrics.increment_counter("string_normalization.failed")
            
            if handle_nulls and original_nulls > 0:
                try:
                    processed_df[column] = handle_null_values(series)
                    self.metrics.increment_counter("null_handling.completed")
                except Exception as e:
                    self.logger.warning(f"NULL handling failed for column {column}", {"error": str(e)})
                    self.metrics.increment_counter("null_handling.failed")
            
            if standardize_dates and pd.api.types.is_datetime64_any_dtype(series):
                try:
                    processed_df[column] = standardize_datetime(series)
                    self.metrics.increment_counter("datetime_standardization.completed")
                except Exception as e:
                    self.logger.warning(f"Datetime standardization failed for column {column}", {"error": str(e)})
                    self.metrics.increment_counter("datetime_standardization.failed")
            
            if normalize_numbers and pd.api.types.is_numeric_dtype(series):
                try:
                    processed_df[column] = normalize_numeric_data(series)
                    self.metrics.increment_counter("numeric_normalization.completed")
                except Exception as e:
                    self.logger.warning(f"Numeric normalization failed for column {column}", {"error": str(e)})
                    self.metrics.increment_counter("numeric_normalization.failed")
        
        # Update session metrics
        self._update_session_metrics("null_values_processed", null_count)
        
        return processed_df
    
    def _setup_alert_rules(self) -> None:
        """Setup standard alert rules for row ID generation."""
        if not self.alert_manager:
            return
        
        # High failure rate alert
        self.alert_manager.add_rule(
            name="high_failure_rate",
            condition=AlertConditions.error_rate_high("failed_operations", "operation_count", 10.0),
            severity=AlertSeverity.HIGH,
            message_template="High failure rate detected: {failure_rate:.2%} of operations failing",
            cooldown_seconds=300  # 5 minutes
        )
        
        # Low uniqueness alert
        self.alert_manager.add_rule(
            name="low_uniqueness",
            condition=lambda metrics: metrics.get("selection_ratio", 1.0) < 0.5,
            severity=AlertSeverity.MEDIUM,
            message_template="Low column selection ratio: {selection_ratio:.2%}",
            cooldown_seconds=600  # 10 minutes
        )
        
        # Performance degradation alert
        self.alert_manager.add_rule(
            name="performance_degradation",
            condition=AlertConditions.threshold_exceeded("avg_response_time", 5.0),
            severity=AlertSeverity.MEDIUM,
            message_template="Performance degradation detected: average response time {avg_response_time:.2f}s",
            cooldown_seconds=300  # 5 minutes
        )
        
        # System resource alerts
        self.alert_manager.add_rule(
            name="high_cpu_usage",
            condition=AlertConditions.threshold_exceeded("cpu_percent", self.config.get("monitoring.threshold_cpu_percent", 80.0)),
            severity=AlertSeverity.HIGH,
            message_template="High CPU usage: {cpu_percent:.1f}%",
            cooldown_seconds=180  # 3 minutes
        )
        
        self.alert_manager.add_rule(
            name="high_memory_usage",
            condition=AlertConditions.threshold_exceeded("memory_percent", self.config.get("monitoring.threshold_memory_percent", 85.0)),
            severity=AlertSeverity.HIGH,
            message_template="High memory usage: {memory_percent:.1f}%",
            cooldown_seconds=180  # 3 minutes
        )
        
        self.logger.info("Alert rules configured", {
            "rules_count": len(self.alert_manager._rules),
            "rules": list(self.alert_manager._rules.keys())
        })
    
    def _analyze_generation_results(
        self,
        result_df: pd.DataFrame,
        selected_columns: List[str],
        audit_trail: Dict[str, Any]
    ) -> None:
        """Analyze row ID generation results and update metrics."""
        # Calculate uniqueness metrics
        if 'row_id' in result_df.columns:
            unique_ids = result_df['row_id'].nunique()
            total_rows = len(result_df)
            uniqueness_ratio = unique_ids / total_rows if total_rows > 0 else 0.0
            
            self.metrics.set_gauge("id_uniqueness_ratio", uniqueness_ratio)
            self.metrics.set_gauge("total_generated_ids", unique_ids)
            
            # Check for collisions
            collisions = total_rows - unique_ids
            if collisions > 0:
                self._update_session_metrics("collision_count", collisions)
                self.metrics.increment_counter("hash_collisions", collisions)
                
                self.logger.warning("Hash collisions detected", {
                    "collisions": collisions,
                    "collision_rate": collisions / total_rows,
                    "total_rows": total_rows,
                    "unique_ids": unique_ids
                })
    
    def _check_generation_alerts(
        self,
        input_df: pd.DataFrame,
        result_df: pd.DataFrame,
        selected_columns: List[str]
    ) -> None:
        """Check for alerts related to row ID generation."""
        if not self.alert_manager:
            return
        
        # Prepare metrics for alert checking
        alert_metrics = {
            "input_rows": len(input_df),
            "output_rows": len(result_df),
            "selected_columns_count": len(selected_columns),
            "selection_ratio": len(selected_columns) / len(input_df.columns),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        # Add session metrics
        alert_metrics.update(self.get_session_summary())
        
        # Check alerts
        self.alert_manager.check_alerts(alert_metrics)
    
    def _check_column_selection_alerts(
        self,
        df: pd.DataFrame,
        selected_columns: List[str],
        uniqueness_threshold: float
    ) -> None:
        """Check for alerts related to column selection."""
        if not self.alert_manager:
            return
        
        alert_metrics = {
            "total_columns": len(df.columns),
            "selected_columns": len(selected_columns),
            "selection_ratio": len(selected_columns) / len(df.columns),
            "uniqueness_threshold": uniqueness_threshold
        }
        
        self.alert_manager.check_alerts(alert_metrics)
    
    def _handle_operation_error(self, operation_name: str, error: Exception, **context) -> None:
        """Handle operation errors with comprehensive logging and metrics."""
        error_context = {
            "operation": operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        
        # Log error
        self.logger.error(f"Operation failed: {operation_name}", error_context, error)
        
        # Update metrics
        self.metrics.increment_counter(f"errors.{operation_name}")
        self.metrics.increment_counter(f"errors.{type(error).__name__}")
        
        # Trigger error alert if alert manager is available
        if self.alert_manager:
            alert_metrics = {
                "error_count": 1,
                "operation": operation_name,
                "error_type": type(error).__name__
            }
            self.alert_manager.check_alerts(alert_metrics)
    
    def _update_session_metrics(self, metric_name: str, value: Union[int, float]) -> None:
        """Thread-safe update of session metrics."""
        with self._operation_lock:
            current_value = getattr(self._session_metrics, metric_name, 0)
            if isinstance(current_value, (int, float)):
                setattr(self._session_metrics, metric_name, current_value + value)
            else:
                setattr(self._session_metrics, metric_name, value)


# Convenience functions for easy instantiation
def create_observable_engine(
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> ObservableHashingEngine:
    """
    Create an ObservableHashingEngine with default settings.
    
    Args:
        config_path: Path to observability configuration file
        **kwargs: Additional arguments for ObservableHashingEngine
        
    Returns:
        Configured ObservableHashingEngine instance
    """
    return ObservableHashingEngine(config_path=config_path, **kwargs)


def create_minimal_observable_engine() -> ObservableHashingEngine:
    """
    Create a minimal ObservableHashingEngine with basic monitoring only.
    
    Returns:
        ObservableHashingEngine with minimal observability features
    """
    return ObservableHashingEngine(
        enable_monitoring=True,
        enable_alerting=False,
        enable_dashboards=False
    )


def create_full_observable_engine(config_path: Union[str, Path]) -> ObservableHashingEngine:
    """
    Create a fully-featured ObservableHashingEngine with all observability components.
    
    Args:
        config_path: Path to observability configuration file
        
    Returns:
        ObservableHashingEngine with all observability features enabled
    """
    return ObservableHashingEngine(
        config_path=config_path,
        enable_monitoring=True,
        enable_alerting=True,
        enable_dashboards=True
    ) 