"""
Alerting and notification system for the Row ID Generator system.

Provides comprehensive alerting capabilities with configurable rules, severity levels,
multiple notification channels, and integration with metrics and monitoring systems.
"""

import time
import json
import smtplib
import threading
import requests
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import logging


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class AlertEvent:
    """
    Individual alert event with full context.
    
    Attributes:
        alert_id: Unique identifier for the alert
        rule_name: Name of the rule that triggered the alert
        severity: Alert severity level
        status: Current alert status
        message: Alert message
        timestamp: When the alert was triggered
        resolved_timestamp: When the alert was resolved (if applicable)
        acknowledged_by: Who acknowledged the alert (if applicable)
        metadata: Additional context about the alert
        metrics: Metric values that triggered the alert
        suppressed_until: Timestamp until which alert is suppressed
    """
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: float
    resolved_timestamp: Optional[float] = None
    acknowledged_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    suppressed_until: Optional[float] = None
    
    def __post_init__(self):
        """Ensure metadata and metrics are always dicts."""
        if self.metadata is None:
            self.metadata = {}
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        return data
    
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.ACTIVE
    
    def is_suppressed(self) -> bool:
        """Check if alert is currently suppressed."""
        if self.suppressed_until is None:
            return False
        return time.time() < self.suppressed_until


@dataclass
class AlertRule:
    """
    Configuration for an alert rule.
    
    Attributes:
        name: Unique name for the rule
        condition: Function that evaluates metrics and returns bool
        severity: Severity level for alerts triggered by this rule
        message_template: Template string for alert messages
        cooldown_seconds: Minimum time between alerts for the same rule
        enabled: Whether the rule is enabled
        metadata: Additional rule configuration
        escalation_rules: Rules for escalating alerts
        suppression_conditions: Conditions under which to suppress alerts
    """
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes default
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None
    escalation_rules: Optional[List[Dict[str, Any]]] = None
    suppression_conditions: Optional[List[Callable[[Dict[str, Any]], bool]]] = None
    
    def __post_init__(self):
        """Ensure optional fields are initialized."""
        if self.metadata is None:
            self.metadata = {}
        if self.escalation_rules is None:
            self.escalation_rules = []
        if self.suppression_conditions is None:
            self.suppression_conditions = []
    
    def should_suppress(self, metrics: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed based on conditions."""
        for condition in self.suppression_conditions:
            if condition(metrics):
                return True
        return False


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.logger = logging.getLogger(f"notification.{name}")
    
    @abstractmethod
    def send_alert(self, alert: AlertEvent) -> bool:
        """
        Send alert notification.
        
        Args:
            alert: Alert event to send
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        pass
    
    def format_message(self, alert: AlertEvent) -> str:
        """Format alert message for this channel."""
        return (
            f"🚨 ALERT: {alert.rule_name}\n"
            f"Severity: {alert.severity.value.upper()}\n"
            f"Message: {alert.message}\n"
            f"Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}\n"
            f"Alert ID: {alert.alert_id}\n"
        )


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel using SMTP."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_address = config.get('from_address', 'alerts@system.local')
        self.recipients = config.get('recipients', [])
        self.use_tls = config.get('use_tls', True)
    
    def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert via email."""
        if not self.enabled or not self.recipients:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # Format message body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert {alert.alert_id}: {e}")
            return False
    
    def _format_email_body(self, alert: AlertEvent) -> str:
        """Format email body with detailed alert information."""
        body = [
            f"Alert: {alert.rule_name}",
            f"Severity: {alert.severity.value.upper()}",
            f"Status: {alert.status.value.upper()}",
            f"Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}",
            f"Alert ID: {alert.alert_id}",
            "",
            f"Message: {alert.message}",
            ""
        ]
        
        if alert.metadata:
            body.append("Metadata:")
            for key, value in alert.metadata.items():
                body.append(f"  {key}: {value}")
            body.append("")
        
        if alert.metrics:
            body.append("Triggering Metrics:")
            for key, value in alert.metrics.items():
                body.append(f"  {key}: {value}")
            body.append("")
        
        body.append("This is an automated alert from the Row ID Generator system.")
        
        return '\n'.join(body)


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
        self.retry_count = config.get('retry_count', 3)
    
    def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert via webhook."""
        if not self.enabled or not self.webhook_url:
            return False
        
        payload = {
            'alert_id': alert.alert_id,
            'rule_name': alert.rule_name,
            'severity': alert.severity.value,
            'status': alert.status.value,
            'message': alert.message,
            'timestamp': alert.timestamp,
            'metadata': alert.metadata,
            'metrics': alert.metrics
        }
        
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                self.logger.info(f"Webhook alert sent for {alert.alert_id}")
                return True
                
            except Exception as e:
                self.logger.warning(f"Webhook attempt {attempt + 1} failed for {alert.alert_id}: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        self.logger.error(f"Failed to send webhook alert {alert.alert_id} after {self.retry_count} attempts")
        return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel using webhooks."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'AlertBot')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
    
    def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert to Slack."""
        if not self.enabled or not self.webhook_url:
            return False
        
        # Choose emoji based on severity
        severity_emojis = {
            AlertSeverity.LOW: ':information_source:',
            AlertSeverity.MEDIUM: ':warning:',
            AlertSeverity.HIGH: ':exclamation:',
            AlertSeverity.CRITICAL: ':fire:'
        }
        
        emoji = severity_emojis.get(alert.severity, ':warning:')
        
        # Format Slack message
        payload = {
            'channel': self.channel,
            'username': self.username,
            'icon_emoji': self.icon_emoji,
            'attachments': [{
                'color': self._get_color_for_severity(alert.severity),
                'title': f"{emoji} {alert.rule_name}",
                'text': alert.message,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                    {'title': 'Status', 'value': alert.status.value.upper(), 'short': True},
                    {'title': 'Alert ID', 'value': alert.alert_id, 'short': True},
                    {'title': 'Time', 'value': datetime.fromtimestamp(alert.timestamp).isoformat(), 'short': True}
                ],
                'footer': 'Row ID Generator Alerting',
                'ts': int(alert.timestamp)
            }]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert {alert.alert_id}: {e}")
            return False
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get color code for Slack attachment based on severity."""
        colors = {
            AlertSeverity.LOW: '#36a64f',      # Green
            AlertSeverity.MEDIUM: '#ff9900',   # Orange
            AlertSeverity.HIGH: '#ff0000',     # Red
            AlertSeverity.CRITICAL: '#8B0000'  # Dark Red
        }
        return colors.get(severity, '#ff9900')


class AlertManager:
    """
    Comprehensive alert management system.
    
    Features:
    - Rule-based alerting with configurable conditions
    - Multiple severity levels and notification channels
    - Alert deduplication and throttling
    - Alert history and management
    - Integration with metrics and monitoring systems
    - Escalation and suppression capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Setup logging first (before other initialization that may need it)
        self.logger = logging.getLogger("alert_manager")
        
        # Thread-safe storage
        self._lock = threading.RLock()
        
        # Alert management
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, AlertEvent] = {}
        self._alert_history = deque(maxlen=10000)  # Keep last 10k alerts
        self._last_alert_times: Dict[str, float] = {}
        
        # Notification channels
        self._notification_channels: List[NotificationChannel] = []
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_rule': defaultdict(int),
            'notification_failures': 0,
            'suppressed_alerts': 0
        }
        
        # Background processing
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_processing = threading.Event()
        
        # Initialize from config (after logger is set up)
        self._setup_from_config()
    
    def add_rule(
        self, 
        name: str, 
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        message_template: str,
        cooldown_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an alert rule.
        
        Args:
            name: Unique rule name
            condition: Function that evaluates metrics and returns bool
            severity: Alert severity level
            message_template: Template for alert messages (supports {key} formatting)
            cooldown_seconds: Minimum time between alerts for this rule
            metadata: Additional rule configuration
        """
        with self._lock:
            rule = AlertRule(
                name=name,
                condition=condition,
                severity=severity,
                message_template=message_template,
                cooldown_seconds=cooldown_seconds,
                metadata=metadata or {}
            )
            self._rules[name] = rule
        
        self.logger.info(f"Added alert rule: {name} (severity: {severity.value})")
    
    def remove_rule(self, name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            name: Rule name to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                self.logger.info(f"Removed alert rule: {name}")
                return True
            return False
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """
        Add a notification channel.
        
        Args:
            channel: Notification channel to add
        """
        with self._lock:
            self._notification_channels.append(channel)
        
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[AlertEvent]:
        """
        Check all alert rules against current metrics.
        
        Args:
            metrics: Current metric values
            
        Returns:
            List of newly triggered alerts
        """
        triggered_alerts = []
        current_time = time.time()
        
        with self._lock:
            for rule_name, rule in self._rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Check if rule condition is met
                    if rule.condition(metrics):
                        # Check cooldown
                        last_alert_time = self._last_alert_times.get(rule_name, 0)
                        if current_time - last_alert_time < rule.cooldown_seconds:
                            continue  # Still in cooldown
                        
                        # Check suppression conditions
                        if rule.should_suppress(metrics):
                            self._stats['suppressed_alerts'] += 1
                            continue
                        
                        # Create alert
                        alert = self._create_alert(rule, metrics, current_time)
                        triggered_alerts.append(alert)
                        
                        # Update tracking
                        self._last_alert_times[rule_name] = current_time
                        self._active_alerts[alert.alert_id] = alert
                        self._alert_history.append(alert)
                        
                        # Update statistics
                        self._stats['total_alerts'] += 1
                        self._stats['alerts_by_severity'][alert.severity.value] += 1
                        self._stats['alerts_by_rule'][rule_name] += 1
                        
                        # Send notifications
                        self._send_notifications(alert)
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who is acknowledging the alert
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved, False if not found
        """
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_timestamp = time.time()
                
                # Remove from active alerts
                del self._active_alerts[alert_id]
                
                self.logger.info(f"Alert {alert_id} resolved")
                return True
            return False
    
    def suppress_alert(self, alert_id: str, duration_seconds: int) -> bool:
        """
        Suppress an alert for a specified duration.
        
        Args:
            alert_id: ID of alert to suppress
            duration_seconds: How long to suppress the alert
            
        Returns:
            True if alert was suppressed, False if not found
        """
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.suppressed_until = time.time() + duration_seconds
                
                self.logger.info(f"Alert {alert_id} suppressed for {duration_seconds} seconds")
                return True
            return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[AlertEvent]:
        """
        Get list of active alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self._active_alerts.values())
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts
    
    def get_alert_history(self, limit: int = 100) -> List[AlertEvent]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        with self._lock:
            history = list(self._alert_history)
            return history[-limit:] if limit else history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            return {
                'total_alerts': self._stats['total_alerts'],
                'active_alerts': len(self._active_alerts),
                'alerts_by_severity': dict(self._stats['alerts_by_severity']),
                'alerts_by_rule': dict(self._stats['alerts_by_rule']),
                'notification_failures': self._stats['notification_failures'],
                'suppressed_alerts': self._stats['suppressed_alerts'],
                'total_rules': len(self._rules),
                'enabled_rules': sum(1 for r in self._rules.values() if r.enabled),
                'notification_channels': len(self._notification_channels)
            }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current alerting configuration."""
        with self._lock:
            return {
                'rules': {
                    name: {
                        'name': rule.name,
                        'severity': rule.severity.value,
                        'message_template': rule.message_template,
                        'cooldown_seconds': rule.cooldown_seconds,
                        'enabled': rule.enabled,
                        'metadata': rule.metadata
                    }
                    for name, rule in self._rules.items()
                },
                'notification_channels': [
                    {
                        'name': channel.name,
                        'type': type(channel).__name__,
                        'enabled': channel.enabled
                    }
                    for channel in self._notification_channels
                ],
                'statistics': self.get_statistics()
            }
    
    def reset_statistics(self) -> None:
        """Reset alerting statistics."""
        with self._lock:
            self._stats = {
                'total_alerts': 0,
                'alerts_by_severity': defaultdict(int),
                'alerts_by_rule': defaultdict(int),
                'notification_failures': 0,
                'suppressed_alerts': 0
            }
        
        self.logger.info("Alert statistics reset")
    
    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any], timestamp: float) -> AlertEvent:
        """Create an alert event from a rule and metrics."""
        # Generate unique alert ID
        alert_data = f"{rule.name}_{timestamp}_{json.dumps(metrics, sort_keys=True)}"
        alert_id = hashlib.md5(alert_data.encode()).hexdigest()[:12]
        
        # Format message using template
        try:
            message = rule.message_template.format(**metrics)
        except KeyError:
            message = rule.message_template
        
        return AlertEvent(
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            timestamp=timestamp,
            metadata=rule.metadata.copy(),
            metrics=metrics.copy()
        )
    
    def _send_notifications(self, alert: AlertEvent) -> None:
        """Send notifications for an alert."""
        for channel in self._notification_channels:
            try:
                if channel.enabled:
                    success = channel.send_alert(alert)
                    if not success:
                        self._stats['notification_failures'] += 1
            except Exception as e:
                self.logger.error(f"Error sending notification via {channel.name}: {e}")
                self._stats['notification_failures'] += 1
    
    def _setup_from_config(self) -> None:
        """Setup alert manager from configuration."""
        if not self.config:
            return
        
        # Setup notification channels from config
        channels_config = self.config.get('notification_channels', {})
        
        for channel_name, channel_config in channels_config.items():
            channel_type = channel_config.get('type', '').lower()
            
            try:
                if channel_type == 'email':
                    channel = EmailNotificationChannel(channel_name, channel_config)
                elif channel_type == 'webhook':
                    channel = WebhookNotificationChannel(channel_name, channel_config)
                elif channel_type == 'slack':
                    channel = SlackNotificationChannel(channel_name, channel_config)
                else:
                    self.logger.warning(f"Unknown channel type: {channel_type}")
                    continue
                
                self.add_notification_channel(channel)
                
            except Exception as e:
                self.logger.error(f"Failed to setup notification channel {channel_name}: {e}")


# Standard alert condition functions for common use cases
class AlertConditions:
    """Pre-built alert conditions for common scenarios."""
    
    @staticmethod
    def threshold_exceeded(metric_name: str, threshold: float) -> Callable[[Dict[str, Any]], bool]:
        """Alert when metric exceeds threshold."""
        def condition(metrics: Dict[str, Any]) -> bool:
            return metrics.get(metric_name, 0) > threshold
        return condition
    
    @staticmethod
    def threshold_below(metric_name: str, threshold: float) -> Callable[[Dict[str, Any]], bool]:
        """Alert when metric falls below threshold."""
        def condition(metrics: Dict[str, Any]) -> bool:
            return metrics.get(metric_name, float('inf')) < threshold
        return condition
    
    @staticmethod
    def rate_exceeded(metric_name: str, rate_threshold: float, time_window: int = 300) -> Callable[[Dict[str, Any]], bool]:
        """Alert when rate of change exceeds threshold."""
        last_values = {}
        
        def condition(metrics: Dict[str, Any]) -> bool:
            current_time = time.time()
            current_value = metrics.get(metric_name, 0)
            
            if metric_name in last_values:
                last_time, last_value = last_values[metric_name]
                time_diff = current_time - last_time
                
                if time_diff > 0:
                    rate = (current_value - last_value) / time_diff
                    if rate > rate_threshold:
                        return True
            
            last_values[metric_name] = (current_time, current_value)
            return False
        
        return condition
    
    @staticmethod
    def error_rate_high(error_metric: str, total_metric: str, threshold_percent: float) -> Callable[[Dict[str, Any]], bool]:
        """Alert when error rate exceeds percentage threshold."""
        def condition(metrics: Dict[str, Any]) -> bool:
            errors = metrics.get(error_metric, 0)
            total = metrics.get(total_metric, 0)
            
            if total == 0:
                return False
            
            error_rate = (errors / total) * 100
            return error_rate > threshold_percent
        
        return condition


# Convenience functions for quick setup
def create_alert_manager(config: Optional[Dict[str, Any]] = None) -> AlertManager:
    """
    Create an alert manager with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        AlertManager instance
    """
    return AlertManager(config)


def create_email_channel(
    name: str,
    smtp_server: str,
    recipients: List[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
    from_address: str = 'alerts@system.local'
) -> EmailNotificationChannel:
    """
    Create an email notification channel.
    
    Args:
        name: Channel name
        smtp_server: SMTP server address
        recipients: List of email recipients
        username: Optional SMTP username
        password: Optional SMTP password
        from_address: From email address
        
    Returns:
        EmailNotificationChannel instance
    """
    config = {
        'smtp_server': smtp_server,
        'recipients': recipients,
        'username': username,
        'password': password,
        'from_address': from_address,
        'enabled': True
    }
    return EmailNotificationChannel(name, config)


def create_slack_channel(name: str, webhook_url: str, channel: str = '#alerts') -> SlackNotificationChannel:
    """
    Create a Slack notification channel.
    
    Args:
        name: Channel name
        webhook_url: Slack webhook URL
        channel: Slack channel name
        
    Returns:
        SlackNotificationChannel instance
    """
    config = {
        'webhook_url': webhook_url,
        'channel': channel,
        'enabled': True
    }
    return SlackNotificationChannel(name, config) 