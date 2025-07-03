"""
Observability package for the Row ID Generator system.

Provides comprehensive monitoring, logging, metrics collection, alerting,
and dashboard capabilities for monitoring system performance and health.
"""

# Import implemented modules
from .logger import StructuredLogger, LogLevel, get_logger
from .metrics import MetricsCollector, MetricPoint, MetricSummary, MetricType, TimerContext, create_metrics_collector
from .monitor import PerformanceMonitor, SystemSnapshot, OperationProfile, OperationTracker, create_performance_monitor
from .alerting import (
    AlertManager, AlertEvent, AlertRule, AlertSeverity, AlertStatus,
    NotificationChannel, EmailNotificationChannel, WebhookNotificationChannel, SlackNotificationChannel,
    AlertConditions, create_alert_manager, create_email_channel, create_slack_channel
)
from .dashboard import (
    DashboardGenerator, DashboardLayout, DashboardWidget, DashboardFormat, DashboardTheme,
    DashboardRenderer, HTMLDashboardRenderer, JSONDashboardRenderer, GrafanaDashboardRenderer,
    create_dashboard_generator, create_custom_dashboard, create_metric_widget
)
from .config import (
    ObservabilityConfig, ObservabilitySettings, ConfigurationError, ConfigFormat,
    LoggingConfig, MetricsConfig, AlertingConfig, MonitoringConfig, DashboardConfig,
    load_config, create_config_template, get_default_config, validate_config_file
)

# List currently available components
__all__ = [
    # Logging
    'StructuredLogger',
    'LogLevel',
    'get_logger',
    # Metrics
    'MetricsCollector',
    'MetricPoint',
    'MetricSummary',
    'MetricType',
    'TimerContext',
    'create_metrics_collector',
    # Performance Monitoring
    'PerformanceMonitor',
    'SystemSnapshot',
    'OperationProfile',
    'OperationTracker',
    'create_performance_monitor',
    # Alerting
    'AlertManager',
    'AlertEvent',
    'AlertRule',
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'EmailNotificationChannel',
    'WebhookNotificationChannel',
    'SlackNotificationChannel',
    'AlertConditions',
    'create_alert_manager',
    'create_email_channel',
    'create_slack_channel',
    # Dashboard and Visualization
    'DashboardGenerator',
    'DashboardLayout',
    'DashboardWidget',
    'DashboardFormat',
    'DashboardTheme',
    'DashboardRenderer',
    'HTMLDashboardRenderer',
    'JSONDashboardRenderer',
    'GrafanaDashboardRenderer',
    'create_dashboard_generator',
    'create_custom_dashboard',
    'create_metric_widget',
    # Configuration Management
    'ObservabilityConfig',
    'ObservabilitySettings',
    'ConfigurationError',
    'ConfigFormat',
    'LoggingConfig',
    'MetricsConfig',
    'AlertingConfig',
    'MonitoringConfig',
    'DashboardConfig',
    'load_config',
    'create_config_template',
    'get_default_config',
    'validate_config_file'
]

__version__ = '1.0.0'

# TODO: Add imports for other components as they are implemented:
# from .config import ObservabilityConfig 