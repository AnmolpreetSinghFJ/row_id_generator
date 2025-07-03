"""
Configuration management system for the Row ID Generator observability system.

Provides comprehensive configuration loading, validation, environment variable
overrides, and default settings management for all observability components.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime

# Optional YAML support for configuration files
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "json"
    output: str = "console"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5
    rotation: str = "size"  # size, time, or none
    json_indent: Optional[int] = None
    include_timestamp: bool = True
    include_session_id: bool = True
    include_system_info: bool = True


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    collection_interval: int = 60
    retention_days: int = 30
    max_points_per_metric: int = 10000
    export_prometheus: bool = True
    prometheus_port: int = 8000
    prometheus_path: str = "/metrics"
    export_json: bool = False
    json_export_path: Optional[str] = None
    tag_filtering: bool = True
    statistical_summaries: bool = True


@dataclass
class AlertingConfig:
    """Alerting system configuration."""
    enabled: bool = True
    email_notifications: bool = False
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    alert_recipients: List[str] = field(default_factory=list)
    slack_webhook: Optional[str] = None
    slack_channel: Optional[str] = None
    webhook_urls: List[str] = field(default_factory=list)
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 20
    deduplication: bool = True
    severity_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "low": 1,
        "medium": 5,
        "high": 10,
        "critical": 20
    })


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration."""
    enabled: bool = True
    performance_tracking: bool = True
    memory_monitoring: bool = True
    error_tracking: bool = True
    system_monitoring: bool = True
    monitoring_interval: int = 30
    threshold_cpu_percent: float = 80.0
    threshold_memory_percent: float = 85.0
    threshold_disk_percent: float = 90.0
    threshold_response_time_ms: float = 1000.0
    operation_history_size: int = 1000
    background_monitoring: bool = True


@dataclass
class DashboardConfig:
    """Dashboard and visualization configuration."""
    enabled: bool = True
    default_theme: str = "light"
    refresh_interval: int = 60
    cache_ttl: int = 30
    grafana_integration: bool = False
    grafana_url: Optional[str] = None
    grafana_api_key: Optional[str] = None
    export_html: bool = True
    html_output_dir: Optional[str] = None
    include_charts: bool = True
    responsive_design: bool = True


@dataclass
class ObservabilitySettings:
    """Complete observability configuration settings."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Global settings
    project_name: str = "row-id-generator"
    environment: str = "development"
    debug: bool = False
    config_version: str = "1.0"
    

class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ObservabilityConfig:
    """
    Comprehensive configuration management for observability system.
    
    Features:
    - YAML/JSON configuration file loading
    - Environment variable overrides
    - Default settings management
    - Configuration validation
    - Hot reloading
    - Configuration inheritance
    - Template generation
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        auto_reload: bool = False
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, test, prod)
            auto_reload: Enable automatic configuration reloading
        """
        self.config_path = Path(config_path) if config_path else None
        self.environment = environment or os.getenv("OBSERVABILITY_ENV", "development")
        self.auto_reload = auto_reload
        self._env_from_constructor = environment is not None
        
        # Configuration state
        self._settings: ObservabilitySettings = ObservabilitySettings()
        self._config_lock = threading.RLock()
        self._file_watchers: List[Callable] = []
        self._last_modified: Optional[float] = None
        
        # Environment variable mappings
        self._env_mappings = self._setup_env_mappings()
        
        # Load configuration
        self._load_configuration()
        
        # Setup auto-reload if enabled
        if auto_reload and self.config_path:
            self._setup_auto_reload()
    
    @property
    def settings(self) -> ObservabilitySettings:
        """Get current configuration settings."""
        with self._config_lock:
            return self._settings
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            parts = key.split('.')
            value = asdict(self._settings)
            
            for part in parts:
                value = value[part]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            value: Value to set
        """
        with self._config_lock:
            parts = key.split('.')
            settings_dict = asdict(self._settings)
            
            # Navigate to parent dict
            current = settings_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1]] = value
            
            # Reconstruct settings object
            self._settings = self._dict_to_settings(settings_dict)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary values.
        
        Args:
            config_dict: Configuration dictionary
        """
        with self._config_lock:
            current_dict = asdict(self._settings)
            merged_dict = self._deep_merge(current_dict, config_dict)
            self._settings = self._dict_to_settings(merged_dict)
    
    def reload(self) -> bool:
        """
        Reload configuration from file.
        
        Returns:
            True if configuration was reloaded, False otherwise
        """
        try:
            with self._config_lock:
                old_settings = asdict(self._settings)
                self._load_configuration()
                new_settings = asdict(self._settings)
                
                return old_settings != new_settings
        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            return False
    
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Validate logging configuration
            if hasattr(LogLevel, self._settings.logging.level.upper()):
                pass  # Valid log level
            else:
                errors.append(f"Invalid log level: {self._settings.logging.level}")
            
            # Validate metrics configuration
            if self._settings.metrics.collection_interval <= 0:
                errors.append("Metrics collection interval must be positive")
            
            if self._settings.metrics.retention_days <= 0:
                errors.append("Metrics retention days must be positive")
            
            # Validate monitoring thresholds
            monitoring = self._settings.monitoring
            if not (0 <= monitoring.threshold_cpu_percent <= 100):
                errors.append("CPU threshold must be between 0 and 100")
            
            if not (0 <= monitoring.threshold_memory_percent <= 100):
                errors.append("Memory threshold must be between 0 and 100")
            
            # Validate alerting configuration
            if self._settings.alerting.enabled and self._settings.alerting.email_notifications:
                if not self._settings.alerting.smtp_server:
                    errors.append("SMTP server required for email notifications")
                
                if not self._settings.alerting.alert_recipients:
                    errors.append("Alert recipients required for email notifications")
            
            # Validate dashboard configuration
            if self._settings.dashboard.refresh_interval <= 0:
                errors.append("Dashboard refresh interval must be positive")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors
    
    def export_config(self, format_type: ConfigFormat = ConfigFormat.YAML) -> str:
        """
        Export current configuration to string.
        
        Args:
            format_type: Export format
            
        Returns:
            Configuration as string
        """
        config_dict = asdict(self._settings)
        
        if format_type == ConfigFormat.YAML:
            if not HAS_YAML:
                raise ConfigurationError("PyYAML not available. Install with: pip install PyYAML")
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        elif format_type == ConfigFormat.JSON:
            return json.dumps(config_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Output file path (uses current config_path if None)
        """
        path = Path(output_path) if output_path else self.config_path
        
        if not path:
            raise ConfigurationError("No output path specified")
        
        # Determine format from file extension
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            format_type = ConfigFormat.YAML
        elif suffix == '.json':
            format_type = ConfigFormat.JSON
        else:
            format_type = ConfigFormat.YAML  # Default
        
        # Export and save
        config_content = self.export_config(format_type)
        
        # Ensure directory exists  
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(config_content)
    
    def create_template(self, output_path: Union[str, Path]) -> None:
        """
        Create a configuration template file.
        
        Args:
            output_path: Path for template file
        """
        template_settings = ObservabilitySettings()
        template_dict = asdict(template_settings)
        
        # Add comments/documentation
        template_content = self._generate_template_content(template_dict)
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(template_content)
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """
        Get configuration for specific component.
        
        Args:
            component: Component name (logging, metrics, alerting, monitoring, dashboard)
            
        Returns:
            Component configuration dictionary
        """
        component_map = {
            'logging': self._settings.logging,
            'metrics': self._settings.metrics,
            'alerting': self._settings.alerting,
            'monitoring': self._settings.monitoring,
            'dashboard': self._settings.dashboard
        }
        
        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")
        
        return asdict(component_map[component])
    
    def _load_configuration(self) -> None:
        """Load configuration from file and environment variables."""
        # Start with defaults
        config_dict = asdict(ObservabilitySettings())
        
        # Load from file if specified
        if self.config_path and self.config_path.exists():
            file_config = self._load_config_file(self.config_path)
            config_dict = self._deep_merge(config_dict, file_config)
            self._last_modified = self.config_path.stat().st_mtime
        
        # Apply environment variable overrides
        env_overrides = self._load_env_overrides()
        config_dict = self._deep_merge(config_dict, env_overrides)
        
        # Handle environment setting priority:
        # 1. Environment variable OBSERVABILITY_ENV
        # 2. Constructor parameter
        # 3. Config file value
        # 4. Default 'development'
        env_var = os.getenv("OBSERVABILITY_ENV")
        if env_var:
            config_dict['environment'] = env_var
        elif self._env_from_constructor:
            config_dict['environment'] = self.environment
        # else: keep the value from config file or default
        
        # Convert back to settings object
        self._settings = self._dict_to_settings(config_dict)
        
        # Validate configuration
        errors = self.validate()
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {', '.join(errors)}")
    
    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            suffix = path.suffix.lower()
            if suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ConfigurationError(f"PyYAML not available for loading {suffix} files. Install with: pip install PyYAML")
                return yaml.safe_load(content) or {}
            elif suffix == '.json':
                return json.loads(content)
            else:
                raise ConfigurationError(f"Unsupported config file format: {suffix}")
                
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {path}: {e}")
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        for env_var, config_path in self._env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                self._set_nested_value(overrides, config_path, converted_value)
        
        return overrides
    
    def _setup_env_mappings(self) -> Dict[str, str]:
        """Setup environment variable to configuration path mappings."""
        return {
            # Logging
            'OBSERVABILITY_LOG_LEVEL': 'logging.level',
            'OBSERVABILITY_LOG_FORMAT': 'logging.format',
            'OBSERVABILITY_LOG_FILE': 'logging.file_path',
            
            # Metrics
            'OBSERVABILITY_METRICS_ENABLED': 'metrics.enabled',
            'OBSERVABILITY_METRICS_INTERVAL': 'metrics.collection_interval',
            'OBSERVABILITY_METRICS_RETENTION': 'metrics.retention_days',
            'OBSERVABILITY_PROMETHEUS_PORT': 'metrics.prometheus_port',
            
            # Alerting
            'OBSERVABILITY_ALERTS_ENABLED': 'alerting.enabled',
            'OBSERVABILITY_SMTP_SERVER': 'alerting.smtp_server',
            'OBSERVABILITY_SMTP_USERNAME': 'alerting.smtp_username',
            'OBSERVABILITY_SMTP_PASSWORD': 'alerting.smtp_password',
            'OBSERVABILITY_SLACK_WEBHOOK': 'alerting.slack_webhook',
            'OBSERVABILITY_ALERT_RECIPIENTS': 'alerting.alert_recipients',
            
            # Monitoring
            'OBSERVABILITY_MONITORING_ENABLED': 'monitoring.enabled',
            'OBSERVABILITY_MONITORING_INTERVAL': 'monitoring.monitoring_interval',
            'OBSERVABILITY_CPU_THRESHOLD': 'monitoring.threshold_cpu_percent',
            'OBSERVABILITY_MEMORY_THRESHOLD': 'monitoring.threshold_memory_percent',
            
            # Dashboard
            'OBSERVABILITY_DASHBOARD_ENABLED': 'dashboard.enabled',
            'OBSERVABILITY_DASHBOARD_THEME': 'dashboard.default_theme',
            'OBSERVABILITY_GRAFANA_URL': 'dashboard.grafana_url',
            'OBSERVABILITY_GRAFANA_API_KEY': 'dashboard.grafana_api_key',
            
            # General
            'OBSERVABILITY_PROJECT_NAME': 'project_name',
            'OBSERVABILITY_ENV': 'environment',
            'OBSERVABILITY_DEBUG': 'debug'
        }
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String value
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation path."""
        parts = path.split('.')
        current = data
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_settings(self, config_dict: Dict[str, Any]) -> ObservabilitySettings:
        """Convert configuration dictionary to settings object."""
        # Extract component configurations
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        metrics_config = MetricsConfig(**config_dict.get('metrics', {}))
        alerting_config = AlertingConfig(**config_dict.get('alerting', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        dashboard_config = DashboardConfig(**config_dict.get('dashboard', {}))
        
        # Extract global settings
        global_settings = {
            'project_name': config_dict.get('project_name', 'row-id-generator'),
            'environment': config_dict.get('environment', 'development'),
            'debug': config_dict.get('debug', False),
            'config_version': config_dict.get('config_version', '1.0')
        }
        
        return ObservabilitySettings(
            logging=logging_config,
            metrics=metrics_config,
            alerting=alerting_config,
            monitoring=monitoring_config,
            dashboard=dashboard_config,
            **global_settings
        )
    
    def _generate_template_content(self, config_dict: Dict[str, Any]) -> str:
        """Generate configuration template with comments."""
        template = f"""# Observability Configuration Template
# Generated on {datetime.now().isoformat()}

# Project identification
project_name: "{config_dict['project_name']}"
environment: "{config_dict['environment']}"
debug: {str(config_dict['debug']).lower()}
config_version: "{config_dict['config_version']}"

# Logging Configuration
logging:
  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: "{config_dict['logging']['level']}"
  
  # Output format: json, text
  format: "{config_dict['logging']['format']}"
  
  # Output destination: console, file, both
  output: "{config_dict['logging']['output']}"
  
  # Optional file path for file output
  file_path: {config_dict['logging']['file_path'] or 'null'}
  
  # File rotation settings
  max_file_size: "{config_dict['logging']['max_file_size']}"
  backup_count: {config_dict['logging']['backup_count']}
  rotation: "{config_dict['logging']['rotation']}"
  
  # JSON formatting options
  json_indent: {config_dict['logging']['json_indent'] or 'null'}
  include_timestamp: {str(config_dict['logging']['include_timestamp']).lower()}
  include_session_id: {str(config_dict['logging']['include_session_id']).lower()}
  include_system_info: {str(config_dict['logging']['include_system_info']).lower()}

# Metrics Collection Configuration
metrics:
  # Enable metrics collection
  enabled: {str(config_dict['metrics']['enabled']).lower()}
  
  # Collection interval in seconds
  collection_interval: {config_dict['metrics']['collection_interval']}
  
  # Data retention in days
  retention_days: {config_dict['metrics']['retention_days']}
  
  # Maximum points per metric
  max_points_per_metric: {config_dict['metrics']['max_points_per_metric']}
  
  # Prometheus export settings
  export_prometheus: {str(config_dict['metrics']['export_prometheus']).lower()}
  prometheus_port: {config_dict['metrics']['prometheus_port']}
  prometheus_path: "{config_dict['metrics']['prometheus_path']}"
  
  # JSON export settings
  export_json: {str(config_dict['metrics']['export_json']).lower()}
  json_export_path: {config_dict['metrics']['json_export_path'] or 'null'}
  
  # Feature flags
  tag_filtering: {str(config_dict['metrics']['tag_filtering']).lower()}
  statistical_summaries: {str(config_dict['metrics']['statistical_summaries']).lower()}

# Alerting Configuration
alerting:
  # Enable alerting system
  enabled: {str(config_dict['alerting']['enabled']).lower()}
  
  # Email notification settings
  email_notifications: {str(config_dict['alerting']['email_notifications']).lower()}
  smtp_server: {config_dict['alerting']['smtp_server'] or 'null'}
  smtp_port: {config_dict['alerting']['smtp_port']}
  smtp_username: {config_dict['alerting']['smtp_username'] or 'null'}
  smtp_password: {config_dict['alerting']['smtp_password'] or 'null'}
  smtp_use_tls: {str(config_dict['alerting']['smtp_use_tls']).lower()}
  
  # Alert recipients (email addresses)
  alert_recipients: []
  
  # Slack integration
  slack_webhook: {config_dict['alerting']['slack_webhook'] or 'null'}
  slack_channel: {config_dict['alerting']['slack_channel'] or 'null'}
  
  # Webhook URLs for custom integrations
  webhook_urls: []
  
  # Alert rate limiting
  cooldown_minutes: {config_dict['alerting']['cooldown_minutes']}
  max_alerts_per_hour: {config_dict['alerting']['max_alerts_per_hour']}
  deduplication: {str(config_dict['alerting']['deduplication']).lower()}
  
  # Severity thresholds
  severity_thresholds:
    low: {config_dict['alerting']['severity_thresholds']['low']}
    medium: {config_dict['alerting']['severity_thresholds']['medium']}
    high: {config_dict['alerting']['severity_thresholds']['high']}
    critical: {config_dict['alerting']['severity_thresholds']['critical']}

# Performance Monitoring Configuration
monitoring:
  # Enable performance monitoring
  enabled: {str(config_dict['monitoring']['enabled']).lower()}
  
  # Feature flags
  performance_tracking: {str(config_dict['monitoring']['performance_tracking']).lower()}
  memory_monitoring: {str(config_dict['monitoring']['memory_monitoring']).lower()}
  error_tracking: {str(config_dict['monitoring']['error_tracking']).lower()}
  system_monitoring: {str(config_dict['monitoring']['system_monitoring']).lower()}
  background_monitoring: {str(config_dict['monitoring']['background_monitoring']).lower()}
  
  # Monitoring intervals and thresholds
  monitoring_interval: {config_dict['monitoring']['monitoring_interval']}
  threshold_cpu_percent: {config_dict['monitoring']['threshold_cpu_percent']}
  threshold_memory_percent: {config_dict['monitoring']['threshold_memory_percent']}
  threshold_disk_percent: {config_dict['monitoring']['threshold_disk_percent']}
  threshold_response_time_ms: {config_dict['monitoring']['threshold_response_time_ms']}
  
  # History settings
  operation_history_size: {config_dict['monitoring']['operation_history_size']}

# Dashboard Configuration
dashboard:
  # Enable dashboard system
  enabled: {str(config_dict['dashboard']['enabled']).lower()}
  
  # UI settings
  default_theme: "{config_dict['dashboard']['default_theme']}"
  refresh_interval: {config_dict['dashboard']['refresh_interval']}
  cache_ttl: {config_dict['dashboard']['cache_ttl']}
  
  # Grafana integration
  grafana_integration: {str(config_dict['dashboard']['grafana_integration']).lower()}
  grafana_url: {config_dict['dashboard']['grafana_url'] or 'null'}
  grafana_api_key: {config_dict['dashboard']['grafana_api_key'] or 'null'}
  
  # Export settings
  export_html: {str(config_dict['dashboard']['export_html']).lower()}
  html_output_dir: {config_dict['dashboard']['html_output_dir'] or 'null'}
  
  # Feature flags
  include_charts: {str(config_dict['dashboard']['include_charts']).lower()}
  responsive_design: {str(config_dict['dashboard']['responsive_design']).lower()}
"""
        return template
    
    def _setup_auto_reload(self) -> None:
        """Setup automatic configuration reloading."""
        # This would implement file watching for configuration changes
        # For now, we'll skip the actual implementation as it requires
        # additional dependencies like watchdog
        pass


# Convenience functions
def load_config(
    config_path: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> ObservabilityConfig:
    """
    Load observability configuration.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        ObservabilityConfig instance
    """
    return ObservabilityConfig(config_path=config_path, environment=environment)


def create_config_template(output_path: Union[str, Path]) -> None:
    """
    Create a configuration template file.
    
    Args:
        output_path: Output path for template
    """
    config = ObservabilityConfig()
    config.create_template(output_path)


def get_default_config() -> ObservabilitySettings:
    """
    Get default configuration settings.
    
    Returns:
        Default ObservabilitySettings
    """
    return ObservabilitySettings()


def validate_config_file(config_path: Union[str, Path]) -> List[str]:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation errors
    """
    try:
        config = ObservabilityConfig(config_path=config_path)
        return config.validate()
    except Exception as e:
        return [f"Failed to load configuration: {e}"] 