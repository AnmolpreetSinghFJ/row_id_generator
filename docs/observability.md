# Observability and Monitoring Guide

Complete guide to monitoring, logging, metrics, and operational insights for the row-id-generator package in production environments.

## Table of Contents

- [Overview](#overview)
- [Observable Engines](#observable-engines)
- [Logging](#logging)
- [Metrics and Monitoring](#metrics-and-monitoring)
- [Performance Monitoring](#performance-monitoring)
- [Configuration](#configuration)
- [Dashboards and Visualization](#dashboards-and-visualization)
- [Alerting](#alerting)
- [Troubleshooting](#troubleshooting)
- [Integration Examples](#integration-examples)

---

## Overview

The row-id-generator package provides comprehensive observability features designed for production environments. These capabilities enable you to monitor performance, track data quality, detect issues early, and optimize operations.

### Key Observability Features

- **📊 Real-time Metrics**: Performance, throughput, and quality metrics
- **📝 Structured Logging**: Configurable log levels with structured output
- **🔍 Audit Trails**: Detailed processing information and lineage tracking
- **📈 Performance Monitoring**: Resource usage and optimization insights
- **🚨 Alerting**: Configurable alerts for various conditions
- **📋 Dashboards**: HTML and JSON dashboard generation
- **🔗 Integration**: Export to Prometheus, DataDog, and other monitoring systems

### When to Use Observability

| Use Case | Recommendation |
|----------|----------------|
| Development & Testing | Basic monitoring for debugging |
| Staging Environment | Comprehensive monitoring without performance impact |
| Production (Low Volume) | Full observability with all features |
| Production (High Volume) | Selective monitoring optimized for performance |
| Critical Data Pipelines | Maximum observability with external integrations |

---

## Observable Engines

Observable engines provide the foundation for monitoring and observability. Choose the engine that matches your requirements.

### Engine Types

#### 1. Minimal Observable Engine

For basic monitoring with minimal overhead:

```python
from row_id_generator import create_minimal_observable_engine

# Basic monitoring
engine = create_minimal_observable_engine()

# Process with basic observability
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(df)

# Check basic metrics
session_summary = engine.get_session_summary()
print(f"Operations: {session_summary['operation_count']}")
print(f"Success Rate: {session_summary['success_rate']:.1%}")
```

**Features:**
- Session-level metrics
- Basic performance tracking
- Minimal resource overhead
- Simple error tracking

**Best for:** Development, testing, low-criticality applications

#### 2. Full Observable Engine

For comprehensive production monitoring:

```python
from row_id_generator import create_full_observable_engine

# Comprehensive monitoring
engine = create_full_observable_engine('config/observability.yaml')

# Process with full observability
result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
    df=df,
    show_progress=True
)

# Get comprehensive health report
health_report = engine.get_system_health_report()
```

**Features:**
- System resource monitoring
- Detailed audit trails
- Performance dashboards
- Alert management
- Metrics export
- Configuration management

**Best for:** Production environments, critical data pipelines

#### 3. Custom Observable Engine

For specific requirements:

```python
from row_id_generator import create_observable_engine

# Custom configuration
engine = create_observable_engine(
    config_path='config/custom_observability.yaml',
    enable_logging=True,
    enable_metrics=True,
    enable_alerts=False  # Disable alerts for this instance
)
```

### Engine Configuration

```yaml
# config/observability.yaml
logging:
  level: INFO
  format: structured
  handlers:
    - console
    - file

metrics:
  enabled: true
  retention_hours: 168  # 7 days
  export_interval: 60   # seconds
  
monitoring:
  system_resources: true
  performance_tracking: true
  audit_trails: true
  
alerts:
  enabled: true
  thresholds:
    cpu_percent: 80
    memory_percent: 85
    error_rate: 0.05
    
dashboard:
  auto_refresh: 30      # seconds
  export_path: "dashboards/"
```

---

## Logging

Comprehensive logging system with configurable levels and structured output.

### Log Levels

| Level | Purpose | Example Use Cases |
|-------|---------|-------------------|
| `DEBUG` | Detailed diagnostic information | Function entry/exit, variable values, step-by-step processing |
| `INFO` | General information | Processing start/completion, configuration changes, normal operations |
| `WARNING` | Warning conditions | Data quality issues, performance degradation, fallback usage |
| `ERROR` | Error conditions | Processing failures, validation errors, exception handling |
| `CRITICAL` | Critical conditions | System failures, data corruption, unrecoverable errors |

### Configuration

#### Environment Variables

```bash
# Logging level
ROWID_LOG_LEVEL=INFO

# Log format (structured or simple)
ROWID_LOG_FORMAT=structured

# Log output destinations
ROWID_LOG_HANDLERS=console,file

# Log file path
ROWID_LOG_FILE=/var/log/rowid-generator/app.log
```

#### Programmatic Configuration

```python
import logging
from row_id_generator import configure_logging

# Configure logging programmatically
configure_logging(
    level=logging.INFO,
    format_type='structured',
    handlers=['console', 'file'],
    file_path='/var/log/rowid-generator/app.log'
)
```

### Log Message Formats

#### Structured Format (JSON)

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "row_id_generator.core",
  "message": "Row ID generation completed",
  "context": {
    "session_id": "sess_abc123",
    "rows_processed": 10000,
    "processing_time": 2.34,
    "selected_columns": ["email", "user_id"],
    "uniqueness_ratio": 0.987
  }
}
```

#### Simple Format

```
2024-01-15 10:30:00,123 - INFO - row_id_generator.core - Row ID generation completed (10000 rows, 2.34s)
```

### Key Log Messages

#### Processing Events

```python
# Processing start
LOG.info("Starting row ID generation", extra={
    "rows": len(df),
    "columns": list(df.columns),
    "session_id": session_id
})

# Column selection
LOG.info("Selected columns for hashing", extra={
    "selected_columns": selected_columns,
    "uniqueness_scores": column_scores,
    "selection_criteria": criteria
})

# Processing completion
LOG.info("Row ID generation completed", extra={
    "rows_processed": len(result_df),
    "processing_time": elapsed_time,
    "uniqueness_ratio": uniqueness_ratio,
    "performance_metrics": metrics
})
```

#### Warning Events

```python
# Data quality warnings
LOG.warning("Low uniqueness detected", extra={
    "column": column_name,
    "uniqueness_ratio": ratio,
    "threshold": threshold,
    "recommendation": "Consider adding more columns"
})

# Performance warnings
LOG.warning("High memory usage detected", extra={
    "memory_usage_mb": current_memory,
    "threshold_mb": memory_threshold,
    "recommendation": "Consider using chunked processing"
})
```

#### Error Events

```python
# Processing errors
LOG.error("Row ID generation failed", extra={
    "error": str(exception),
    "error_type": type(exception).__name__,
    "context": error_context,
    "suggestions": error_suggestions
})
```

---

## Metrics and Monitoring

Comprehensive metrics collection and monitoring capabilities.

### Available Metrics

#### Processing Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `operations_total` | Counter | Total number of operations performed | count |
| `operations_success_total` | Counter | Number of successful operations | count |
| `operations_error_total` | Counter | Number of failed operations | count |
| `processing_duration_seconds` | Histogram | Time spent processing requests | seconds |
| `rows_processed_total` | Counter | Total number of rows processed | count |
| `throughput_rows_per_second` | Gauge | Current processing throughput | rows/sec |

#### Quality Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `uniqueness_ratio` | Gauge | Ratio of unique values to total values | ratio (0-1) |
| `null_ratio` | Gauge | Ratio of null values to total values | ratio (0-1) |
| `quality_score` | Gauge | Overall data quality score | score (0-100) |
| `selected_columns_count` | Gauge | Number of columns selected for hashing | count |

#### System Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `cpu_usage_percent` | Gauge | Current CPU usage | percentage |
| `memory_usage_mb` | Gauge | Current memory usage | megabytes |
| `memory_usage_percent` | Gauge | Current memory usage percentage | percentage |
| `active_sessions` | Gauge | Number of active processing sessions | count |

### Metrics Collection

#### Automatic Collection

```python
from row_id_generator import create_full_observable_engine

engine = create_full_observable_engine()

# Metrics are automatically collected during processing
result_df, cols, audit = engine.generate_unique_row_ids(df)

# Access collected metrics
metrics = engine.export_metrics("json")
print(f"Processing time: {metrics['processing_duration_seconds']}")
print(f"Throughput: {metrics['throughput_rows_per_second']}")
```

#### Manual Collection

```python
from row_id_generator.monitoring import MetricsCollector

collector = MetricsCollector()

# Collect system metrics
collector.collect_system_metrics()

# Collect processing metrics
collector.record_processing_start()
# ... your processing logic ...
collector.record_processing_complete(rows_processed=10000)

# Get metrics
metrics = collector.get_all_metrics()
```

### Metrics Export

#### JSON Export

```python
# Export as JSON
metrics_json = engine.export_metrics("json")

# Example output
{
  "timestamp": "2024-01-15T10:30:00Z",
  "operations_total": 42,
  "operations_success_total": 40,
  "operations_error_total": 2,
  "processing_duration_seconds": 15.67,
  "rows_processed_total": 50000,
  "throughput_rows_per_second": 3191.2,
  "uniqueness_ratio": 0.987,
  "quality_score": 92.5,
  "cpu_usage_percent": 45.2,
  "memory_usage_mb": 1024.5
}
```

#### Prometheus Export

```python
# Export in Prometheus format
prometheus_metrics = engine.export_metrics("prometheus")

# Example output
# HELP rowid_operations_total Total number of operations
# TYPE rowid_operations_total counter
rowid_operations_total 42

# HELP rowid_processing_duration_seconds Processing duration
# TYPE rowid_processing_duration_seconds histogram
rowid_processing_duration_seconds_bucket{le="1.0"} 5
rowid_processing_duration_seconds_bucket{le="5.0"} 15
rowid_processing_duration_seconds_bucket{le="10.0"} 35
rowid_processing_duration_seconds_count 42
rowid_processing_duration_seconds_sum 156.7

# HELP rowid_throughput_rows_per_second Current throughput
# TYPE rowid_throughput_rows_per_second gauge
rowid_throughput_rows_per_second 3191.2
```

#### CSV Export

```python
# Export as CSV for spreadsheet analysis
csv_metrics = engine.export_metrics("csv")

# Saves to file: metrics_YYYYMMDD_HHMMSS.csv
timestamp,operation,duration,rows,throughput,quality_score
2024-01-15T10:30:00Z,generate_row_ids,15.67,50000,3191.2,92.5
```

---

## Performance Monitoring

Track and optimize performance with detailed monitoring capabilities.

### Performance Dashboards

#### HTML Dashboard Generation

```python
# Generate performance dashboard
dashboard_html = engine.generate_performance_dashboard()

# Save to file
with open('performance_dashboard.html', 'w') as f:
    f.write(dashboard_html)

# Dashboard includes:
# - Real-time metrics
# - Performance trends
# - Resource utilization
# - Processing history
# - Quality metrics
# - System health status
```

#### Dashboard Features

1. **Real-time Metrics**
   - Current throughput
   - Processing latency
   - Error rates
   - Resource usage

2. **Historical Trends**
   - Performance over time
   - Throughput patterns
   - Error frequency
   - Quality trends

3. **Resource Monitoring**
   - CPU utilization
   - Memory usage
   - I/O statistics
   - Network metrics

4. **Quality Insights**
   - Data quality scores
   - Column selection patterns
   - Uniqueness trends
   - Null value statistics

### Performance Optimization

#### Recommendations Engine

```python
# Get performance recommendations
recommendations = engine.get_performance_recommendations()

# Example recommendations
{
  "memory_optimization": {
    "severity": "medium",
    "message": "Consider enabling chunked processing for datasets > 1M rows",
    "action": "Set chunk_size=50000 in processing parameters"
  },
  "column_selection": {
    "severity": "low", 
    "message": "Manual column selection may improve consistency",
    "action": "Specify columns=['email', 'user_id'] for better performance"
  },
  "caching": {
    "severity": "high",
    "message": "Enable result caching for repeated operations",
    "action": "Set enable_caching=True in configuration"
  }
}
```

#### Performance Baselines

```python
from row_id_generator.monitoring import PerformanceBaseline

# Establish baseline
baseline = PerformanceBaseline(
    operation_type="generate_row_ids",
    dataset_size=10000,
    baseline_time=2.5,
    baseline_throughput=4000
)

# Monitor against baseline
current_performance = engine.get_current_performance()
deviation = baseline.compare(current_performance)

if deviation.is_significant():
    LOG.warning("Performance regression detected", extra={
        "baseline_time": baseline.baseline_time,
        "current_time": current_performance.time,
        "deviation_percent": deviation.percent,
        "recommendation": deviation.recommendation
    })
```

---

## Configuration

Comprehensive configuration options for observability features.

### Configuration Files

#### Main Configuration (observability.yaml)

```yaml
observability:
  # Global settings
  enabled: true
  session_tracking: true
  
  # Logging configuration
  logging:
    level: INFO
    format: structured
    handlers:
      - type: console
        level: INFO
      - type: file
        level: DEBUG
        file: /var/log/rowid/app.log
        rotation:
          max_size: 100MB
          backup_count: 5
      - type: syslog
        level: WARNING
        facility: local0
  
  # Metrics configuration
  metrics:
    enabled: true
    collection_interval: 30  # seconds
    retention_period: 7      # days
    storage_path: /var/lib/rowid/metrics
    
    # Metric categories
    categories:
      processing: true
      quality: true
      system: true
      performance: true
    
    # Export settings
    export:
      prometheus:
        enabled: true
        port: 9090
        path: /metrics
      json:
        enabled: true
        interval: 300  # seconds
        path: /var/lib/rowid/metrics.json
  
  # Monitoring configuration
  monitoring:
    system_resources:
      enabled: true
      interval: 10  # seconds
      thresholds:
        cpu_percent: 80
        memory_percent: 85
        disk_usage_percent: 90
    
    performance:
      enabled: true
      baseline_tracking: true
      regression_detection: true
      thresholds:
        max_processing_time: 300  # seconds
        min_throughput: 1000      # rows/sec
        max_error_rate: 0.05      # 5%
  
  # Dashboard configuration
  dashboard:
    enabled: true
    auto_refresh: 30     # seconds
    history_retention: 24 # hours
    export_path: /var/www/rowid/dashboards
    
    # Dashboard components
    components:
      - real_time_metrics
      - performance_trends
      - quality_insights
      - system_health
      - error_tracking
  
  # Alerting configuration
  alerts:
    enabled: true
    cooldown_period: 300  # seconds
    
    # Alert channels
    channels:
      - type: log
        level: ERROR
      - type: webhook
        url: https://hooks.slack.com/your-webhook
        format: slack
      - type: email
        smtp_server: smtp.example.com
        recipients:
          - ops@example.com
          - data-team@example.com
    
    # Alert rules
    rules:
      high_error_rate:
        condition: error_rate > 0.05
        severity: high
        message: "High error rate detected: {error_rate:.2%}"
      
      performance_degradation:
        condition: throughput < baseline * 0.8
        severity: medium
        message: "Performance degradation: {throughput} rows/sec (baseline: {baseline})"
      
      system_resources:
        condition: cpu_percent > 80 OR memory_percent > 85
        severity: medium
        message: "High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
```

### Environment Variables

```bash
# Core observability settings
ROWID_OBSERVABILITY_ENABLED=true
ROWID_OBSERVABILITY_CONFIG=/etc/rowid/observability.yaml

# Logging settings
ROWID_LOG_LEVEL=INFO
ROWID_LOG_FORMAT=structured
ROWID_LOG_FILE=/var/log/rowid/app.log

# Metrics settings
ROWID_METRICS_ENABLED=true
ROWID_METRICS_RETENTION_HOURS=168
ROWID_METRICS_EXPORT_INTERVAL=60

# Monitoring settings
ROWID_MONITORING_SYSTEM_RESOURCES=true
ROWID_MONITORING_PERFORMANCE=true

# Dashboard settings
ROWID_DASHBOARD_ENABLED=true
ROWID_DASHBOARD_PATH=/var/www/rowid/dashboards

# Alert settings
ROWID_ALERTS_ENABLED=true
ROWID_ALERTS_WEBHOOK_URL=https://hooks.slack.com/your-webhook
```

### Runtime Configuration

```python
from row_id_generator import configure_observability

# Configure observability at runtime
configure_observability({
    'logging': {
        'level': 'INFO',
        'format': 'structured'
    },
    'metrics': {
        'enabled': True,
        'retention_hours': 24
    },
    'alerts': {
        'enabled': True,
        'thresholds': {
            'cpu_percent': 75,
            'error_rate': 0.03
        }
    }
})
```

---

## Dashboards and Visualization

Rich visualization capabilities for monitoring and analysis.

### HTML Dashboards

#### Generating Dashboards

```python
from row_id_generator import create_full_observable_engine

engine = create_full_observable_engine()

# Generate comprehensive dashboard
dashboard_html = engine.generate_performance_dashboard()

# Save to file
with open('rowid_dashboard.html', 'w') as f:
    f.write(dashboard_html)

# Dashboard includes:
# - Real-time metrics display
# - Performance trends and charts
# - Resource utilization graphs
# - Processing history timeline
# - Data quality indicators
# - Alert status panel
```

#### Dashboard Components

1. **Real-time Metrics Panel**
   ```html
   <!-- Displays current metrics -->
   <div class="metrics-panel">
     <div class="metric">
       <h3>Throughput</h3>
       <span class="value">3,191 rows/sec</span>
       <span class="trend up">↗ +12%</span>
     </div>
     <div class="metric">
       <h3>Quality Score</h3>
       <span class="value">92.5</span>
       <span class="status good">Good</span>
     </div>
   </div>
   ```

2. **Performance Charts**
   - Processing time trends
   - Throughput over time
   - Error rate tracking
   - Resource utilization

3. **System Health Panel**
   - CPU and memory usage
   - Active alerts
   - System status indicators

### Custom Dashboard Creation

```python
from row_id_generator.monitoring import DashboardBuilder

# Create custom dashboard
builder = DashboardBuilder()

# Add components
builder.add_metric_panel(['throughput', 'quality_score'])
builder.add_trend_chart('processing_time', hours=24)
builder.add_resource_monitor(['cpu_percent', 'memory_percent'])
builder.add_alert_panel()

# Generate dashboard
custom_dashboard = builder.build(
    title="Custom Row ID Dashboard",
    refresh_interval=30,
    theme="dark"
)

# Export
with open('custom_dashboard.html', 'w') as f:
    f.write(custom_dashboard.render())
```

### Dashboard Configuration

```yaml
dashboard:
  enabled: true
  auto_refresh: 30
  history_retention: 24
  export_path: "/var/www/dashboards"
  
  # Theme options
  theme:
    primary_color: "#007bff"
    success_color: "#28a745"
    warning_color: "#ffc107"
    danger_color: "#dc3545"
    
  # Component settings
  components:
    metrics_panel:
      enabled: true
      metrics: ['throughput', 'quality_score', 'error_rate']
    
    trend_charts:
      enabled: true
      charts: ['processing_time', 'throughput', 'quality_score']
      time_range: 24  # hours
      
    resource_monitor:
      enabled: true
      resources: ['cpu_percent', 'memory_percent', 'disk_usage']
      
    alert_panel:
      enabled: true
      show_history: true
      max_alerts: 10
```

---

## Alerting

Comprehensive alerting system for proactive monitoring.

### Alert Types

#### System Resource Alerts

```python
# High CPU usage alert
{
  "type": "system_resource",
  "condition": "cpu_percent > 80",
  "severity": "warning",
  "message": "High CPU usage detected: {cpu_percent:.1f}%",
  "cooldown": 300,  # 5 minutes
  "actions": ["log", "webhook"]
}

# Memory usage alert
{
  "type": "system_resource", 
  "condition": "memory_percent > 85",
  "severity": "critical",
  "message": "Critical memory usage: {memory_percent:.1f}%",
  "cooldown": 180,  # 3 minutes
  "actions": ["log", "webhook", "email"]
}
```

#### Performance Alerts

```python
# Throughput degradation
{
  "type": "performance",
  "condition": "throughput < baseline_throughput * 0.7",
  "severity": "warning",
  "message": "Performance degradation: {throughput:.0f} rows/sec (baseline: {baseline_throughput:.0f})",
  "cooldown": 600,  # 10 minutes
  "actions": ["log", "webhook"]
}

# High processing time
{
  "type": "performance",
  "condition": "processing_time > 300",  # 5 minutes
  "severity": "warning", 
  "message": "Long processing time: {processing_time:.1f}s",
  "cooldown": 300,
  "actions": ["log"]
}
```

#### Quality Alerts

```python
# Low data quality
{
  "type": "quality",
  "condition": "quality_score < 70",
  "severity": "warning",
  "message": "Low data quality detected: {quality_score:.1f}",
  "cooldown": 900,  # 15 minutes
  "actions": ["log", "webhook"]
}

# Low uniqueness ratio
{
  "type": "quality",
  "condition": "uniqueness_ratio < 0.8",
  "severity": "warning",
  "message": "Low uniqueness ratio: {uniqueness_ratio:.2%}",
  "cooldown": 600,
  "actions": ["log"]
}
```

### Alert Channels

#### Webhook Notifications

```python
# Slack webhook configuration
{
  "type": "webhook",
  "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
  "format": "slack",
  "template": {
    "text": "Row ID Generator Alert",
    "attachments": [{
      "color": "{severity_color}",
      "fields": [{
        "title": "Severity",
        "value": "{severity}",
        "short": true
      }, {
        "title": "Message", 
        "value": "{message}",
        "short": false
      }]
    }]
  }
}

# Generic webhook
{
  "type": "webhook",
  "url": "https://api.example.com/alerts",
  "format": "json",
  "headers": {
    "Authorization": "Bearer your-token",
    "Content-Type": "application/json"
  },
  "payload": {
    "alert_type": "{type}",
    "severity": "{severity}",
    "message": "{message}",
    "timestamp": "{timestamp}",
    "metrics": "{metrics}"
  }
}
```

#### Email Notifications

```python
# Email configuration
{
  "type": "email",
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "alerts@example.com",
  "password": "your-app-password",
  "use_tls": true,
  
  "recipients": [
    "ops-team@example.com",
    "data-team@example.com"
  ],
  
  "template": {
    "subject": "Row ID Generator Alert - {severity}",
    "body": """
    Alert Details:
    - Type: {type}
    - Severity: {severity} 
    - Message: {message}
    - Timestamp: {timestamp}
    
    System Metrics:
    - CPU Usage: {cpu_percent:.1f}%
    - Memory Usage: {memory_percent:.1f}%
    - Throughput: {throughput:.0f} rows/sec
    
    Dashboard: https://dashboard.example.com/rowid
    """
  }
}
```

### Alert Management

#### Programmatic Alert Handling

```python
from row_id_generator.monitoring import AlertManager

# Create alert manager
alert_manager = AlertManager(config_path='config/alerts.yaml')

# Register custom alert handler
@alert_manager.handler('custom_handler')
def handle_custom_alert(alert):
    """Custom alert handling logic."""
    if alert.severity == 'critical':
        # Send to PagerDuty
        send_to_pagerduty(alert)
    elif alert.severity == 'warning':
        # Log to monitoring system
        log_to_datadog(alert)

# Check for alerts
engine = create_full_observable_engine()
active_alerts = engine.get_active_alerts()

for alert in active_alerts:
    alert_manager.handle_alert(alert)
```

#### Alert Lifecycle

```python
# Alert lifecycle management
class AlertLifecycle:
    def __init__(self, alert_manager):
        self.alert_manager = alert_manager
    
    def trigger_alert(self, alert_config, current_metrics):
        """Trigger new alert if conditions are met."""
        if self.evaluate_condition(alert_config['condition'], current_metrics):
            alert = self.create_alert(alert_config, current_metrics)
            self.alert_manager.fire_alert(alert)
    
    def resolve_alert(self, alert_id):
        """Mark alert as resolved."""
        alert = self.alert_manager.get_alert(alert_id)
        alert.status = 'resolved'
        alert.resolved_at = datetime.utcnow()
        self.alert_manager.update_alert(alert)
    
    def silence_alert(self, alert_id, duration):
        """Temporarily silence an alert."""
        alert = self.alert_manager.get_alert(alert_id)
        alert.silenced_until = datetime.utcnow() + timedelta(seconds=duration)
        self.alert_manager.update_alert(alert)
```

---

## Troubleshooting

Common issues and diagnostic approaches using observability data.

### Common Issues

#### Issue 1: High Memory Usage

**Symptoms:**
- Memory usage consistently above 85%
- Slow processing times
- Out of memory errors

**Diagnostic Steps:**

```python
# Check memory metrics
health_report = engine.get_system_health_report()
memory_stats = health_report['system_resources']['memory']

print(f"Memory usage: {memory_stats['percent']:.1f}%")
print(f"Available memory: {memory_stats['available_mb']:.0f}MB")

# Check processing metrics
processing_stats = engine.get_processing_metrics()
print(f"Average rows per batch: {processing_stats['avg_batch_size']}")
print(f"Peak memory per operation: {processing_stats['peak_memory_mb']:.0f}MB")

# Get recommendations
recommendations = engine.get_performance_recommendations()
for rec in recommendations:
    if rec['category'] == 'memory':
        print(f"Recommendation: {rec['message']}")
```

**Solutions:**
1. Enable chunked processing
2. Reduce batch sizes
3. Use memory-optimized functions

```python
# Apply memory optimizations
from row_id_generator import create_optimized_row_id_function

optimized_generator = create_optimized_row_id_function(
    max_memory_mb=1000,
    enable_chunking=True,
    enable_streaming=True
)

result_df = optimized_generator(
    df=large_df,
    chunk_size=10000  # Smaller chunks
)
```

#### Issue 2: Low Throughput

**Symptoms:**
- Processing slower than baseline
- Throughput alerts firing
- Long processing times

**Diagnostic Steps:**

```python
# Analyze performance trends
performance_data = engine.get_performance_history(hours=24)

# Check for patterns
for hour_data in performance_data:
    if hour_data['throughput'] < baseline_throughput * 0.8:
        print(f"Low throughput at {hour_data['timestamp']}: {hour_data['throughput']:.0f}")

# Check system resource correlation
system_data = engine.get_system_resource_history(hours=24)
for sys_data in system_data:
    if sys_data['cpu_percent'] > 80:
        print(f"High CPU at {sys_data['timestamp']}: {sys_data['cpu_percent']:.1f}%")
```

**Solutions:**
1. Use performance-optimized functions
2. Optimize column selection
3. Disable unnecessary features

```python
# Performance optimization
result_df = generate_row_ids_fast(
    df,
    columns=['primary_key'],  # Fewer columns
    enable_monitoring=False,  # Disable monitoring
    enable_quality_checks=False  # Skip quality checks
)
```

#### Issue 3: Data Quality Warnings

**Symptoms:**
- Low uniqueness warnings
- Quality score below threshold
- Inconsistent results

**Diagnostic Steps:**

```python
# Analyze data quality
from row_id_generator import analyze_dataframe_quality

quality_metrics = analyze_dataframe_quality(df)
summary = quality_metrics.get_summary_report()

print(f"Overall quality score: {summary['overall_score']:.1f}")
print(f"Null ratio: {summary['null_ratio']:.2%}")
print(f"Uniqueness issues: {summary['uniqueness_issues']}")

# Check column quality individually
for column in df.columns:
    score = get_column_quality_score(df, column)
    if score['score'] < 70:
        print(f"Low quality column: {column} (score: {score['score']:.1f})")
```

**Solutions:**
1. Clean data before processing
2. Add more columns for uniqueness
3. Adjust quality thresholds

```python
# Data cleaning approach
df_clean = df.dropna().drop_duplicates()

# Add more columns
result_df = generate_unique_row_ids(
    df_clean,
    columns=['email', 'name', 'phone'],  # More columns
    uniqueness_threshold=0.8  # Lower threshold
)
```

### Diagnostic Tools

#### Audit Trail Analysis

```python
# Detailed audit trail examination
result_df = generate_unique_row_ids(
    df,
    return_audit_trail=True,
    enable_enhanced_lineage=True
)

if isinstance(result_df, dict):
    audit_trail = result_df['audit_trail']
    
    # Processing timeline
    print("Processing Timeline:")
    for step in audit_trail['processing_steps']:
        print(f"  {step['timestamp']}: {step['action']} ({step['duration']:.3f}s)")
    
    # Column selection details
    print("\nColumn Selection:")
    for col_info in audit_trail['column_selection']:
        print(f"  {col_info['column']}: uniqueness={col_info['uniqueness']:.3f}")
    
    # Performance metrics
    print(f"\nPerformance:")
    print(f"  Total time: {audit_trail['total_time']:.3f}s")
    print(f"  Throughput: {audit_trail['throughput']:.0f} rows/sec")
    print(f"  Memory peak: {audit_trail['peak_memory_mb']:.0f}MB")
```

#### Health Check Utilities

```python
def comprehensive_health_check(engine):
    """Perform comprehensive system health check."""
    
    # System resources
    health_report = engine.get_system_health_report()
    
    issues = []
    
    # Check CPU
    cpu_percent = health_report['system_resources']['cpu_percent']
    if cpu_percent > 80:
        issues.append(f"High CPU usage: {cpu_percent:.1f}%")
    
    # Check memory
    memory_percent = health_report['system_resources']['memory_percent']
    if memory_percent > 85:
        issues.append(f"High memory usage: {memory_percent:.1f}%")
    
    # Check active alerts
    active_alerts = health_report.get('active_alerts', [])
    if active_alerts:
        issues.append(f"{len(active_alerts)} active alerts")
    
    # Check performance trends
    performance_data = engine.get_performance_history(hours=1)
    if performance_data:
        recent_throughput = performance_data[-1]['throughput']
        baseline = engine.get_baseline_throughput()
        if recent_throughput < baseline * 0.8:
            issues.append(f"Low throughput: {recent_throughput:.0f} (baseline: {baseline:.0f})")
    
    return {
        'status': 'healthy' if not issues else 'issues_detected',
        'issues': issues,
        'recommendations': engine.get_performance_recommendations()
    }

# Run health check
health_status = comprehensive_health_check(engine)
print(f"System status: {health_status['status']}")
for issue in health_status['issues']:
    print(f"  ⚠️ {issue}")
```

---

## Integration Examples

Real-world integration patterns with monitoring systems.

### Prometheus Integration

```python
# Prometheus metrics export
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server

class PrometheusExporter:
    def __init__(self, port=9090):
        self.registry = CollectorRegistry()
        self.port = port
        
        # Define metrics
        self.operations_total = Counter(
            'rowid_operations_total',
            'Total operations performed',
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'rowid_processing_duration_seconds',
            'Processing duration in seconds',
            registry=self.registry
        )
        
        self.throughput = Gauge(
            'rowid_throughput_rows_per_second',
            'Current throughput',
            registry=self.registry
        )
        
        self.quality_score = Gauge(
            'rowid_quality_score',
            'Data quality score',
            registry=self.registry
        )
        
        # Start metrics server
        start_http_server(port, registry=self.registry)
    
    def update_metrics(self, engine):
        """Update Prometheus metrics from engine."""
        metrics = engine.export_metrics("json")
        
        self.operations_total._value._value = metrics.get('operations_total', 0)
        self.throughput.set(metrics.get('throughput_rows_per_second', 0))
        self.quality_score.set(metrics.get('quality_score', 0))
        
        # Record processing duration
        if 'processing_duration_seconds' in metrics:
            self.processing_duration.observe(metrics['processing_duration_seconds'])

# Usage
exporter = PrometheusExporter(port=9090)
engine = create_full_observable_engine()

# Process data and update metrics
result_df, cols, audit = engine.generate_unique_row_ids(df)
exporter.update_metrics(engine)
```

### DataDog Integration

```python
import requests
import json
import time

class DataDogExporter:
    def __init__(self, api_key, app_key=None):
        self.api_key = api_key
        self.app_key = app_key
        self.base_url = "https://api.datadoghq.com/api/v1"
    
    def send_metrics(self, engine, tags=None):
        """Send metrics to DataDog."""
        metrics = engine.export_metrics("json")
        
        if tags is None:
            tags = ['service:row-id-generator', 'environment:production']
        
        # Prepare metrics payload
        series = []
        
        # Add each metric
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                series.append({
                    'metric': f'rowid.{metric_name}',
                    'points': [[int(time.time()), value]],
                    'tags': tags
                })
        
        # Send to DataDog
        payload = {'series': series}
        
        response = requests.post(
            f"{self.base_url}/series",
            headers={
                'DD-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            },
            data=json.dumps(payload)
        )
        
        return response.status_code == 202
    
    def send_event(self, title, text, tags=None, alert_type='info'):
        """Send event to DataDog."""
        if tags is None:
            tags = ['service:row-id-generator']
        
        payload = {
            'title': title,
            'text': text,
            'tags': tags,
            'alert_type': alert_type,
            'date_happened': int(time.time())
        }
        
        response = requests.post(
            f"{self.base_url}/events",
            headers={
                'DD-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            },
            data=json.dumps(payload)
        )
        
        return response.status_code == 202

# Usage
datadog = DataDogExporter(api_key='your-datadog-api-key')
engine = create_full_observable_engine()

# Process data
result_df, cols, audit = engine.generate_unique_row_ids(df)

# Send metrics
datadog.send_metrics(engine, tags=['environment:prod', 'team:data'])

# Send completion event
datadog.send_event(
    title="Row ID Generation Completed",
    text=f"Processed {len(df)} rows successfully",
    alert_type='success'
)
```

### ELK Stack Integration

```python
import json
import requests
from datetime import datetime

class ElasticsearchExporter:
    def __init__(self, elasticsearch_url, index_prefix='rowid-metrics'):
        self.es_url = elasticsearch_url
        self.index_prefix = index_prefix
    
    def send_metrics(self, engine):
        """Send metrics to Elasticsearch."""
        metrics = engine.export_metrics("json")
        
        # Add timestamp
        metrics['@timestamp'] = datetime.utcnow().isoformat()
        metrics['service'] = 'row-id-generator'
        
        # Create index name with date
        index_name = f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"
        
        # Send to Elasticsearch
        response = requests.post(
            f"{self.es_url}/{index_name}/_doc",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(metrics)
        )
        
        return response.status_code in [200, 201]
    
    def send_log_entry(self, level, message, extra_data=None):
        """Send structured log entry to Elasticsearch."""
        log_entry = {
            '@timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'service': 'row-id-generator'
        }
        
        if extra_data:
            log_entry.update(extra_data)
        
        index_name = f"rowid-logs-{datetime.utcnow().strftime('%Y.%m.%d')}"
        
        response = requests.post(
            f"{self.es_url}/{index_name}/_doc",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(log_entry)
        )
        
        return response.status_code in [200, 201]

# Usage with custom logging handler
import logging

class ElasticsearchLogHandler(logging.Handler):
    def __init__(self, es_exporter):
        super().__init__()
        self.es_exporter = es_exporter
    
    def emit(self, record):
        log_data = {
            'logger': record.name,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread
        }
        
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        self.es_exporter.send_log_entry(
            level=record.levelname,
            message=record.getMessage(),
            extra_data=log_data
        )

# Setup
es_exporter = ElasticsearchExporter('http://localhost:9200')
es_handler = ElasticsearchLogHandler(es_exporter)

# Configure logger
logger = logging.getLogger('row_id_generator')
logger.addHandler(es_handler)
logger.setLevel(logging.INFO)
```

### Custom Monitoring Dashboard

```python
import flask
from flask import Flask, jsonify, render_template_string
import json

app = Flask(__name__)

# Store engine reference
observable_engine = None

def init_monitoring_server(engine, port=5000):
    """Initialize monitoring web server."""
    global observable_engine
    observable_engine = engine
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        health_report = observable_engine.get_system_health_report()
        return jsonify({
            'status': 'healthy' if not health_report.get('active_alerts') else 'degraded',
            'timestamp': health_report['timestamp'],
            'cpu_percent': health_report['system_resources']['cpu_percent'],
            'memory_percent': health_report['system_resources']['memory_percent']
        })
    
    @app.route('/metrics')
    def get_metrics():
        """Metrics endpoint."""
        metrics = observable_engine.export_metrics("json")
        return jsonify(metrics)
    
    @app.route('/dashboard')
    def dashboard():
        """Dashboard page."""
        dashboard_html = observable_engine.generate_performance_dashboard()
        return dashboard_html
    
    @app.route('/alerts')
    def get_alerts():
        """Active alerts endpoint."""
        alerts = observable_engine.get_active_alerts()
        return jsonify([{
            'id': alert.id,
            'type': alert.type,
            'severity': alert.severity,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat()
        } for alert in alerts])
    
    # Start server
    app.run(host='0.0.0.0', port=port, debug=False)

# Usage
engine = create_full_observable_engine()

# Start monitoring server in background thread
import threading
server_thread = threading.Thread(
    target=init_monitoring_server,
    args=(engine, 5000),
    daemon=True
)
server_thread.start()

# Now your monitoring endpoints are available:
# http://localhost:5000/health
# http://localhost:5000/metrics  
# http://localhost:5000/dashboard
# http://localhost:5000/alerts
```

This completes the comprehensive observability and monitoring documentation for the row-id-generator package. The documentation covers all aspects from basic logging to advanced integrations with external monitoring systems. 