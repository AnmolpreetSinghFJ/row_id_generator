"""
Dashboard and visualization system for the Row ID Generator system.

Provides comprehensive dashboard generation, system health visualization,
performance monitoring dashboards, and integration with external monitoring
systems like Grafana and Prometheus.
"""

import json
import time
import psutil
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import uuid
import html


class DashboardFormat(Enum):
    """Supported dashboard output formats."""
    HTML = "html"
    JSON = "json"
    GRAFANA = "grafana"
    PROMETHEUS = "prometheus"


class DashboardTheme(Enum):
    """Dashboard visual themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


@dataclass
class DashboardWidget:
    """
    Individual dashboard widget configuration.
    
    Attributes:
        widget_id: Unique identifier for the widget
        title: Widget display title
        widget_type: Type of widget (chart, table, gauge, etc.)
        data_source: Data source identifier
        query: Query or configuration for data retrieval
        position: Widget position (row, column)
        size: Widget size (width, height)
        config: Additional widget configuration
        refresh_interval: How often to refresh widget data (seconds)
    """
    widget_id: str
    title: str
    widget_type: str
    data_source: str
    query: Dict[str, Any]
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (1, 1)
    config: Optional[Dict[str, Any]] = None
    refresh_interval: int = 60
    
    def __post_init__(self):
        """Ensure config is always a dict."""
        if self.config is None:
            self.config = {}


@dataclass
class DashboardLayout:
    """
    Dashboard layout configuration.
    
    Attributes:
        dashboard_id: Unique dashboard identifier
        title: Dashboard title
        description: Dashboard description
        widgets: List of widgets in the dashboard
        theme: Visual theme
        refresh_interval: Default refresh interval for the dashboard
        metadata: Additional dashboard metadata
        filters: Global dashboard filters
        variables: Dashboard variables/parameters
    """
    dashboard_id: str
    title: str
    description: str = ""
    widgets: List[DashboardWidget] = field(default_factory=list)
    theme: DashboardTheme = DashboardTheme.LIGHT
    refresh_interval: int = 60
    metadata: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure optional attributes are initialized."""
        if self.metadata is None:
            self.metadata = {}
        if self.filters is None:
            self.filters = {}
        if self.variables is None:
            self.variables = {}


class DashboardRenderer(ABC):
    """Abstract base class for dashboard renderers."""
    
    @abstractmethod
    def render(self, layout: DashboardLayout, data: Dict[str, Any]) -> str:
        """
        Render dashboard to target format.
        
        Args:
            layout: Dashboard layout configuration
            data: Dashboard data
            
        Returns:
            Rendered dashboard as string
        """
        pass


class HTMLDashboardRenderer(DashboardRenderer):
    """HTML dashboard renderer with interactive charts."""
    
    def __init__(self, include_charts: bool = True):
        self.include_charts = include_charts
    
    def render(self, layout: DashboardLayout, data: Dict[str, Any]) -> str:
        """Render dashboard as HTML."""
        html_parts = [
            self._render_header(layout),
            self._render_styles(layout.theme),
            self._render_scripts() if self.include_charts else "",
            self._render_body_start(layout),
            self._render_widgets(layout, data),
            self._render_body_end(),
            self._render_footer()
        ]
        
        return '\n'.join(html_parts)
    
    def _render_header(self, layout: DashboardLayout) -> str:
        """Render HTML header."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(layout.title)} - Row ID Generator Dashboard</title>
    <meta name="description" content="{html.escape(layout.description)}">
    <meta name="refresh" content="{layout.refresh_interval}">
</head>
"""
    
    def _render_styles(self, theme: DashboardTheme) -> str:
        """Render CSS styles based on theme."""
        theme_colors = {
            DashboardTheme.LIGHT: {
                'bg': '#ffffff',
                'fg': '#333333',
                'border': '#dddddd',
                'accent': '#007bff'
            },
            DashboardTheme.DARK: {
                'bg': '#1a1a1a',
                'fg': '#ffffff',
                'border': '#444444',
                'accent': '#0d6efd'
            }
        }
        
        colors = theme_colors.get(theme, theme_colors[DashboardTheme.LIGHT])
        
        return f"""
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 0;
        padding: 20px;
        background-color: {colors['bg']};
        color: {colors['fg']};
        line-height: 1.6;
    }}
    
    .dashboard-header {{
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 2px solid {colors['border']};
    }}
    
    .dashboard-title {{
        font-size: 2.5em;
        margin: 0 0 10px 0;
        color: {colors['accent']};
    }}
    
    .dashboard-description {{
        font-size: 1.1em;
        color: {colors['fg']};
        opacity: 0.8;
    }}
    
    .dashboard-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }}
    
    .widget {{
        background: {colors['bg']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    
    .widget:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    
    .widget-title {{
        font-size: 1.3em;
        font-weight: 600;
        margin: 0 0 15px 0;
        color: {colors['accent']};
    }}
    
    .metric-value {{
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }}
    
    .metric-label {{
        font-size: 0.9em;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .status-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }}
    
    .status-healthy {{ background-color: #28a745; }}
    .status-warning {{ background-color: #ffc107; }}
    .status-critical {{ background-color: #dc3545; }}
    
    .chart-container {{
        width: 100%;
        height: 300px;
        margin-top: 15px;
    }}
    
    .table-container {{
        overflow-x: auto;
        margin-top: 15px;
    }}
    
    table {{
        width: 100%;
        border-collapse: collapse;
        background: {colors['bg']};
    }}
    
    th, td {{
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid {colors['border']};
    }}
    
    th {{
        background-color: {colors['accent']};
        color: white;
        font-weight: 600;
    }}
    
    .timestamp {{
        text-align: center;
        margin-top: 30px;
        font-size: 0.9em;
        opacity: 0.6;
    }}
    
    @media (max-width: 768px) {{
        .dashboard-grid {{
            grid-template-columns: 1fr;
        }}
        
        .dashboard-title {{
            font-size: 2em;
        }}
    }}
</style>
"""
    
    def _render_scripts(self) -> str:
        """Render JavaScript for interactive charts."""
        return """
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    // Auto-refresh functionality
    function refreshDashboard() {
        setTimeout(() => {
            window.location.reload();
        }, 60000); // Refresh every minute
    }
    
    // Initialize charts when page loads
    document.addEventListener('DOMContentLoaded', function() {
        refreshDashboard();
        initializeCharts();
    });
    
    function initializeCharts() {
        // Chart initialization will be added per widget
    }
</script>
"""
    
    def _render_body_start(self, layout: DashboardLayout) -> str:
        """Render body opening and header."""
        return f"""
<body>
    <div class="dashboard-header">
        <h1 class="dashboard-title">{html.escape(layout.title)}</h1>
        <p class="dashboard-description">{html.escape(layout.description)}</p>
    </div>
    <div class="dashboard-grid">
"""
    
    def _render_widgets(self, layout: DashboardLayout, data: Dict[str, Any]) -> str:
        """Render all widgets."""
        widgets_html = []
        
        for widget in layout.widgets:
            widget_data = data.get(widget.widget_id, {})
            widget_html = self._render_widget(widget, widget_data)
            widgets_html.append(widget_html)
        
        return '\n'.join(widgets_html)
    
    def _render_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Render individual widget."""
        widget_content = ""
        
        if widget.widget_type == "metric":
            widget_content = self._render_metric_widget(widget, data)
        elif widget.widget_type == "chart":
            widget_content = self._render_chart_widget(widget, data)
        elif widget.widget_type == "table":
            widget_content = self._render_table_widget(widget, data)
        elif widget.widget_type == "status":
            widget_content = self._render_status_widget(widget, data)
        else:
            widget_content = self._render_generic_widget(widget, data)
        
        return f"""
        <div class="widget" id="widget-{widget.widget_id}">
            <h3 class="widget-title">{html.escape(widget.title)}</h3>
            {widget_content}
        </div>
        """
    
    def _render_metric_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Render metric display widget."""
        value = data.get('value', 'N/A')
        label = data.get('label', widget.title)
        unit = data.get('unit', '')
        
        return f"""
        <div class="metric-display">
            <div class="metric-value">{value}{unit}</div>
            <div class="metric-label">{html.escape(label)}</div>
        </div>
        """
    
    def _render_chart_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Render chart widget."""
        chart_id = f"chart-{widget.widget_id}"
        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
        </div>
        <script>
            // Chart data would be injected here
            console.log('Chart data for {widget.widget_id}:', {json.dumps(data)});
        </script>
        """
    
    def _render_table_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Render table widget."""
        headers = data.get('headers', [])
        rows = data.get('rows', [])
        
        if not headers or not rows:
            return '<p>No data available</p>'
        
        table_html = ['<div class="table-container"><table>']
        
        # Headers
        table_html.append('<thead><tr>')
        for header in headers:
            table_html.append(f'<th>{html.escape(str(header))}</th>')
        table_html.append('</tr></thead>')
        
        # Rows
        table_html.append('<tbody>')
        for row in rows:
            table_html.append('<tr>')
            for cell in row:
                table_html.append(f'<td>{html.escape(str(cell))}</td>')
            table_html.append('</tr>')
        table_html.append('</tbody>')
        
        table_html.append('</table></div>')
        return ''.join(table_html)
    
    def _render_status_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Render status indicator widget."""
        status = data.get('status', 'unknown')
        message = data.get('message', 'Status unknown')
        
        status_class = {
            'healthy': 'status-healthy',
            'warning': 'status-warning', 
            'critical': 'status-critical'
        }.get(status, 'status-warning')
        
        return f"""
        <div class="status-display">
            <span class="status-indicator {status_class}"></span>
            {html.escape(message)}
        </div>
        """
    
    def _render_generic_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Render generic widget with JSON data."""
        return f"""
        <pre style="white-space: pre-wrap; font-size: 0.9em;">
        {html.escape(json.dumps(data, indent=2))}
        </pre>
        """
    
    def _render_body_end(self) -> str:
        """Render body closing."""
        return """
    </div>
    <div class="timestamp">
        Generated at: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') + """
    </div>
"""
    
    def _render_footer(self) -> str:
        """Render HTML footer."""
        return """
</body>
</html>
"""


class JSONDashboardRenderer(DashboardRenderer):
    """JSON dashboard renderer for API consumption."""
    
    def render(self, layout: DashboardLayout, data: Dict[str, Any]) -> str:
        """Render dashboard as JSON."""
        dashboard_json = {
            'dashboard': asdict(layout),
            'data': data,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'format': 'json',
                'version': '1.0'
            }
        }
        
        return json.dumps(dashboard_json, indent=2, default=str)


class GrafanaDashboardRenderer(DashboardRenderer):
    """Grafana dashboard configuration renderer."""
    
    def render(self, layout: DashboardLayout, data: Dict[str, Any]) -> str:
        """Render Grafana dashboard configuration."""
        grafana_config = {
            "dashboard": {
                "id": None,
                "title": layout.title,
                "description": layout.description,
                "tags": ["row-id-generator", "observability"],
                "timezone": "browser",
                "refresh": f"{layout.refresh_interval}s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": []
            }
        }
        
        # Convert widgets to Grafana panels
        panel_id = 1
        for widget in layout.widgets:
            panel = self._widget_to_grafana_panel(widget, panel_id)
            grafana_config["dashboard"]["panels"].append(panel)
            panel_id += 1
        
        return json.dumps(grafana_config, indent=2)
    
    def _widget_to_grafana_panel(self, widget: DashboardWidget, panel_id: int) -> Dict[str, Any]:
        """Convert widget to Grafana panel configuration."""
        base_panel = {
            "id": panel_id,
            "title": widget.title,
            "gridPos": {
                "x": widget.position[1] * 12,
                "y": widget.position[0] * 8,
                "w": widget.size[0] * 12,
                "h": widget.size[1] * 8
            },
            "targets": [{
                "expr": widget.query.get('prometheus_query', ''),
                "interval": f"{widget.refresh_interval}s",
                "legendFormat": widget.query.get('legend', widget.title)
            }]
        }
        
        # Configure panel type based on widget type
        if widget.widget_type == "metric":
            base_panel.update({
                "type": "stat",
                "fieldConfig": {
                    "defaults": {
                        "unit": widget.config.get('unit', 'short'),
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": widget.config.get('warning_threshold', 80)},
                                {"color": "red", "value": widget.config.get('critical_threshold', 95)}
                            ]
                        }
                    }
                }
            })
        elif widget.widget_type == "chart":
            base_panel.update({
                "type": "graph",
                "yAxes": [{
                    "label": widget.config.get('y_label', ''),
                    "min": widget.config.get('y_min'),
                    "max": widget.config.get('y_max')
                }]
            })
        elif widget.widget_type == "table":
            base_panel.update({
                "type": "table",
                "styles": [{
                    "pattern": "/.*/",
                    "type": "string",
                    "alias": "Value"
                }]
            })
        
        return base_panel


class DashboardGenerator:
    """
    Comprehensive dashboard generation and visualization system.
    
    Features:
    - Multiple dashboard formats (HTML, JSON, Grafana)
    - Real-time system health monitoring
    - Performance metrics visualization
    - Historical trend analysis
    - Customizable layouts and themes
    - Integration with existing observability components
    """
    
    def __init__(self, metrics_collector=None, performance_monitor=None, alert_manager=None):
        """
        Initialize dashboard generator.
        
        Args:
            metrics_collector: Optional metrics collector instance
            performance_monitor: Optional performance monitor instance
            alert_manager: Optional alert manager instance
        """
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        
        # Dashboard configurations
        self._dashboards: Dict[str, DashboardLayout] = {}
        self._renderers = {
            DashboardFormat.HTML: HTMLDashboardRenderer(),
            DashboardFormat.JSON: JSONDashboardRenderer(),
            DashboardFormat.GRAFANA: GrafanaDashboardRenderer()
        }
        
        # Data cache for performance
        self._data_cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 30  # 30 seconds
        
        # Initialize default dashboards
        self._setup_default_dashboards()
    
    def create_dashboard(self, dashboard_id: str, title: str, description: str = "") -> DashboardLayout:
        """
        Create a new dashboard layout.
        
        Args:
            dashboard_id: Unique dashboard identifier
            title: Dashboard title
            description: Dashboard description
            
        Returns:
            DashboardLayout instance
        """
        layout = DashboardLayout(
            dashboard_id=dashboard_id,
            title=title,
            description=description
        )
        
        self._dashboards[dashboard_id] = layout
        return layout
    
    def add_widget(self, dashboard_id: str, widget: DashboardWidget) -> None:
        """
        Add widget to dashboard.
        
        Args:
            dashboard_id: Dashboard to add widget to
            widget: Widget to add
        """
        if dashboard_id in self._dashboards:
            self._dashboards[dashboard_id].widgets.append(widget)
    
    def generate_dashboard(
        self, 
        dashboard_id: str, 
        format_type: DashboardFormat = DashboardFormat.HTML
    ) -> str:
        """
        Generate dashboard in specified format.
        
        Args:
            dashboard_id: Dashboard to generate
            format_type: Output format
            
        Returns:
            Generated dashboard as string
        """
        if dashboard_id not in self._dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        layout = self._dashboards[dashboard_id]
        data = self._collect_dashboard_data(layout)
        
        renderer = self._renderers.get(format_type)
        if not renderer:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return renderer.render(layout, data)
    
    def generate_system_health_dashboard(self) -> str:
        """Generate real-time system health dashboard."""
        return self.generate_dashboard("system_health", DashboardFormat.HTML)
    
    def generate_performance_dashboard(self) -> str:
        """Generate performance monitoring dashboard."""
        return self.generate_dashboard("performance", DashboardFormat.HTML)
    
    def generate_alerts_dashboard(self) -> str:
        """Generate alerts and notifications dashboard."""
        return self.generate_dashboard("alerts", DashboardFormat.HTML)
    
    def export_grafana_config(self, dashboard_id: str) -> str:
        """
        Export dashboard as Grafana configuration.
        
        Args:
            dashboard_id: Dashboard to export
            
        Returns:
            Grafana configuration JSON
        """
        return self.generate_dashboard(dashboard_id, DashboardFormat.GRAFANA)
    
    def export_prometheus_config(self) -> Dict[str, Any]:
        """Export Prometheus monitoring configuration."""
        return {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [{
                "job_name": "row-id-generator",
                "static_configs": [{
                    "targets": ["localhost:8000"]
                }],
                "scrape_interval": "5s",
                "metrics_path": "/metrics"
            }],
            "rule_files": [
                "alert_rules.yml"
            ]
        }
    
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards."""
        return [
            {
                'id': dashboard_id,
                'title': layout.title,
                'description': layout.description,
                'widget_count': len(layout.widgets),
                'theme': layout.theme.value,
                'refresh_interval': layout.refresh_interval
            }
            for dashboard_id, layout in self._dashboards.items()
        ]
    
    def _setup_default_dashboards(self) -> None:
        """Setup default dashboard configurations."""
        self._setup_system_health_dashboard()
        self._setup_performance_dashboard() 
        self._setup_alerts_dashboard()
        self._setup_overview_dashboard()
    
    def _setup_system_health_dashboard(self) -> None:
        """Setup system health monitoring dashboard."""
        dashboard = self.create_dashboard(
            "system_health",
            "System Health Monitor",
            "Real-time system resource monitoring and health status"
        )
        
        # CPU Usage widget
        dashboard.widgets.append(DashboardWidget(
            widget_id="cpu_usage",
            title="CPU Usage",
            widget_type="metric",
            data_source="system",
            query={"metric": "cpu_percent"},
            position=(0, 0),
            config={"unit": "%", "warning_threshold": 70, "critical_threshold": 90}
        ))
        
        # Memory Usage widget
        dashboard.widgets.append(DashboardWidget(
            widget_id="memory_usage",
            title="Memory Usage",
            widget_type="metric", 
            data_source="system",
            query={"metric": "memory_percent"},
            position=(0, 1),
            config={"unit": "%", "warning_threshold": 80, "critical_threshold": 95}
        ))
        
        # Disk Usage widget
        dashboard.widgets.append(DashboardWidget(
            widget_id="disk_usage",
            title="Disk Usage",
            widget_type="metric",
            data_source="system", 
            query={"metric": "disk_percent"},
            position=(0, 2),
            config={"unit": "%", "warning_threshold": 85, "critical_threshold": 95}
        ))
        
        # System Status widget
        dashboard.widgets.append(DashboardWidget(
            widget_id="system_status",
            title="Overall System Status",
            widget_type="status",
            data_source="system",
            query={"metric": "overall_status"},
            position=(1, 0),
            size=(1, 3)
        ))
    
    def _setup_performance_dashboard(self) -> None:
        """Setup performance monitoring dashboard."""
        dashboard = self.create_dashboard(
            "performance",
            "Performance Monitor", 
            "Application performance metrics and operation tracking"
        )
        
        # Operations per second
        dashboard.widgets.append(DashboardWidget(
            widget_id="operations_per_second",
            title="Operations/Second",
            widget_type="metric",
            data_source="performance",
            query={"metric": "operations_rate"},
            position=(0, 0)
        ))
        
        # Average response time
        dashboard.widgets.append(DashboardWidget(
            widget_id="avg_response_time",
            title="Avg Response Time",
            widget_type="metric",
            data_source="performance", 
            query={"metric": "avg_duration"},
            position=(0, 1),
            config={"unit": "ms"}
        ))
        
        # Error rate
        dashboard.widgets.append(DashboardWidget(
            widget_id="error_rate",
            title="Error Rate",
            widget_type="metric",
            data_source="performance",
            query={"metric": "error_rate"},
            position=(0, 2),
            config={"unit": "%", "warning_threshold": 1, "critical_threshold": 5}
        ))
        
        # Performance trends chart
        dashboard.widgets.append(DashboardWidget(
            widget_id="performance_trends",
            title="Performance Trends",
            widget_type="chart",
            data_source="performance",
            query={"metric": "duration_history"},
            position=(1, 0),
            size=(1, 3)
        ))
    
    def _setup_alerts_dashboard(self) -> None:
        """Setup alerts monitoring dashboard."""
        dashboard = self.create_dashboard(
            "alerts",
            "Alerts & Notifications",
            "Alert status, notification channels, and alert history"
        )
        
        # Active alerts count
        dashboard.widgets.append(DashboardWidget(
            widget_id="active_alerts",
            title="Active Alerts",
            widget_type="metric", 
            data_source="alerts",
            query={"metric": "active_count"},
            position=(0, 0)
        ))
        
        # Alerts by severity
        dashboard.widgets.append(DashboardWidget(
            widget_id="alerts_by_severity",
            title="Alerts by Severity",
            widget_type="chart",
            data_source="alerts",
            query={"metric": "severity_breakdown"},
            position=(0, 1),
            size=(1, 2)
        ))
        
        # Recent alerts table
        dashboard.widgets.append(DashboardWidget(
            widget_id="recent_alerts",
            title="Recent Alerts",
            widget_type="table",
            data_source="alerts",
            query={"metric": "recent_alerts", "limit": 10},
            position=(1, 0),
            size=(1, 3)
        ))
    
    def _setup_overview_dashboard(self) -> None:
        """Setup overview dashboard with key metrics."""
        dashboard = self.create_dashboard(
            "overview",
            "System Overview",
            "High-level overview of system health, performance, and alerts"
        )
        
        # Key metrics in a row
        metrics = [
            ("uptime", "System Uptime", "system", {"metric": "uptime"}),
            ("total_operations", "Total Operations", "performance", {"metric": "total_operations"}),
            ("success_rate", "Success Rate", "performance", {"metric": "success_rate", "unit": "%"}),
            ("active_alerts", "Active Alerts", "alerts", {"metric": "active_count"})
        ]
        
        for i, (widget_id, title, data_source, query) in enumerate(metrics):
            dashboard.widgets.append(DashboardWidget(
                widget_id=widget_id,
                title=title,
                widget_type="metric",
                data_source=data_source,
                query=query,
                position=(0, i)
            ))
    
    def _collect_dashboard_data(self, layout: DashboardLayout) -> Dict[str, Any]:
        """
        Collect data for all widgets in dashboard.
        
        Args:
            layout: Dashboard layout
            
        Returns:
            Dictionary of widget data
        """
        data = {}
        
        for widget in layout.widgets:
            # Check cache first
            cache_key = f"{widget.widget_id}_{widget.data_source}"
            cached_data = self._get_cached_data(cache_key)
            
            if cached_data is not None:
                data[widget.widget_id] = cached_data
            else:
                # Collect fresh data
                widget_data = self._collect_widget_data(widget)
                data[widget.widget_id] = widget_data
                
                # Cache the data
                self._cache_data(cache_key, widget_data)
        
        return data
    
    def _collect_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Collect data for a specific widget."""
        if widget.data_source == "system":
            return self._collect_system_data(widget)
        elif widget.data_source == "performance":
            return self._collect_performance_data(widget)
        elif widget.data_source == "alerts":
            return self._collect_alerts_data(widget)
        else:
            return {"error": f"Unknown data source: {widget.data_source}"}
    
    def _collect_system_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Collect system metrics data."""
        metric = widget.query.get("metric")
        
        if metric == "cpu_percent":
            return {
                "value": round(psutil.cpu_percent(), 1),
                "label": "CPU Usage",
                "unit": "%"
            }
        elif metric == "memory_percent":
            return {
                "value": round(psutil.virtual_memory().percent, 1),
                "label": "Memory Usage", 
                "unit": "%"
            }
        elif metric == "disk_percent":
            return {
                "value": round(psutil.disk_usage('/').percent, 1),
                "label": "Disk Usage",
                "unit": "%"
            }
        elif metric == "overall_status":
            # Determine overall system health
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            if cpu > 90 or memory > 95 or disk > 95:
                status = "critical"
                message = "System resources critically high"
            elif cpu > 70 or memory > 80 or disk > 85:
                status = "warning"
                message = "System resources elevated"
            else:
                status = "healthy"
                message = "All systems operating normally"
            
            return {"status": status, "message": message}
        elif metric == "uptime":
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_hours = int(uptime_seconds / 3600)
            return {
                "value": uptime_hours,
                "label": "System Uptime",
                "unit": "h"
            }
        
        return {"error": f"Unknown system metric: {metric}"}
    
    def _collect_performance_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Collect performance metrics data."""
        if not self.performance_monitor:
            return {"error": "Performance monitor not available"}
        
        metric = widget.query.get("metric")
        
        if metric == "operations_rate":
            # Calculate operations per second (simplified)
            return {
                "value": 0,  # Would be calculated from actual metrics
                "label": "Operations/Second"
            }
        elif metric == "avg_duration":
            return {
                "value": 0,  # Would be calculated from actual metrics
                "label": "Average Duration",
                "unit": "ms"
            }
        elif metric == "error_rate":
            return {
                "value": 0,  # Would be calculated from actual metrics
                "label": "Error Rate",
                "unit": "%"
            }
        elif metric == "total_operations":
            return {
                "value": 0,  # Would be calculated from actual metrics
                "label": "Total Operations"
            }
        elif metric == "success_rate":
            return {
                "value": 100,  # Would be calculated from actual metrics
                "label": "Success Rate",
                "unit": "%"
            }
        
        return {"error": f"Unknown performance metric: {metric}"}
    
    def _collect_alerts_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Collect alerts data."""
        if not self.alert_manager:
            return {"error": "Alert manager not available"}
        
        metric = widget.query.get("metric")
        
        if metric == "active_count":
            active_alerts = self.alert_manager.get_active_alerts()
            return {
                "value": len(active_alerts),
                "label": "Active Alerts"
            }
        elif metric == "severity_breakdown":
            stats = self.alert_manager.get_statistics()
            return {
                "labels": list(stats.get("alerts_by_severity", {}).keys()),
                "values": list(stats.get("alerts_by_severity", {}).values())
            }
        elif metric == "recent_alerts":
            recent_alerts = self.alert_manager.get_alert_history(limit=10)
            headers = ["Time", "Rule", "Severity", "Message"]
            rows = []
            
            for alert in recent_alerts:
                rows.append([
                    datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S'),
                    alert.rule_name,
                    alert.severity.value.upper(),
                    alert.message[:50] + "..." if len(alert.message) > 50 else alert.message
                ])
            
            return {"headers": headers, "rows": rows}
        
        return {"error": f"Unknown alerts metric: {metric}"}
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        if key in self._data_cache:
            timestamp, data = self._data_cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        return None
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with timestamp."""
        self._data_cache[key] = (time.time(), data)


# Convenience functions
def create_dashboard_generator(
    metrics_collector=None,
    performance_monitor=None, 
    alert_manager=None
) -> DashboardGenerator:
    """
    Create a dashboard generator with observability components.
    
    Args:
        metrics_collector: Optional metrics collector
        performance_monitor: Optional performance monitor
        alert_manager: Optional alert manager
        
    Returns:
        DashboardGenerator instance
    """
    return DashboardGenerator(
        metrics_collector=metrics_collector,
        performance_monitor=performance_monitor,
        alert_manager=alert_manager
    )


def create_custom_dashboard(
    dashboard_id: str,
    title: str,
    widgets: List[DashboardWidget],
    description: str = "",
    theme: DashboardTheme = DashboardTheme.LIGHT
) -> DashboardLayout:
    """
    Create a custom dashboard layout.
    
    Args:
        dashboard_id: Unique dashboard ID
        title: Dashboard title
        widgets: List of widgets
        description: Dashboard description
        theme: Visual theme
        
    Returns:
        DashboardLayout instance
    """
    layout = DashboardLayout(
        dashboard_id=dashboard_id,
        title=title,
        description=description,
        widgets=widgets,
        theme=theme
    )
    
    return layout


def create_metric_widget(
    widget_id: str,
    title: str,
    data_source: str,
    metric_name: str,
    position: Tuple[int, int] = (0, 0),
    unit: str = "",
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None
) -> DashboardWidget:
    """
    Create a metric display widget.
    
    Args:
        widget_id: Unique widget ID
        title: Widget title
        data_source: Data source identifier
        metric_name: Name of metric to display
        position: Widget position (row, col)
        unit: Metric unit
        warning_threshold: Warning threshold value
        critical_threshold: Critical threshold value
        
    Returns:
        DashboardWidget instance
    """
    config = {"unit": unit}
    if warning_threshold is not None:
        config["warning_threshold"] = warning_threshold
    if critical_threshold is not None:
        config["critical_threshold"] = critical_threshold
    
    return DashboardWidget(
        widget_id=widget_id,
        title=title,
        widget_type="metric",
        data_source=data_source,
        query={"metric": metric_name},
        position=position,
        config=config
    ) 