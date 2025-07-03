"""
Structured logging framework for the Row ID Generator system.

Provides JSON-formatted logging with contextual metadata, session tracking,
and operation-specific logging capabilities for comprehensive observability.
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from enum import Enum


class LogLevel(Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs with contextual metadata.
    
    Features:
    - JSON formatting for structured log analysis
    - Session tracking across operations
    - Operation-specific logging with metadata
    - Configurable log levels
    - Context managers for operation logging
    """
    
    def __init__(
        self, 
        name: str, 
        level: Union[str, LogLevel] = LogLevel.INFO,
        session_id: Optional[str] = None,
        include_system_info: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (usually module or component name)
            level: Logging level (string or LogLevel enum)
            session_id: Optional session ID for tracking related operations
            include_system_info: Whether to include system information in logs
        """
        self.name = name
        self.session_id = session_id or str(uuid.uuid4())
        self.include_system_info = include_system_info
        
        # Setup the underlying Python logger
        self.logger = logging.getLogger(name)
        
        # Convert level to string if enum provided
        level_str = level.value if isinstance(level, LogLevel) else level
        self.logger.setLevel(getattr(logging, level_str.upper()))
        
        # Clear any existing handlers to avoid duplication
        self.logger.handlers.clear()
        
        # Create JSON formatter
        self._setup_json_formatter()
        
        # Initialize operation context stack
        self._operation_stack = []
    
    def _setup_json_formatter(self):
        """Setup JSON formatter for structured logging."""
        
        class JSONFormatter(logging.Formatter):
            """Custom JSON formatter for structured logs."""
            
            def __init__(self, logger_instance):
                super().__init__()
                self.logger_instance = logger_instance
            
            def format(self, record):
                # Base log structure
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "session_id": self.logger_instance.session_id
                }
                
                # Add operation context if available
                if self.logger_instance._operation_stack:
                    log_entry["operation"] = self.logger_instance._operation_stack[-1]
                
                # Add any extra fields from the log record
                if hasattr(record, 'extra_fields'):
                    log_entry.update(record.extra_fields)
                
                # Add system info if enabled
                if self.logger_instance.include_system_info:
                    import os
                    import psutil
                    log_entry["system"] = {
                        "process_id": os.getpid(),
                        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                        "cpu_usage_percent": psutil.cpu_percent()
                    }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": self.formatException(record.exc_info)
                    }
                
                return json.dumps(log_entry, default=str)
        
        # Create and configure handler
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter(self))
        self.logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _log_with_metadata(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Internal method to log with metadata."""
        # Create a custom log record with extra fields
        extra_fields = metadata or {}
        
        # Get the appropriate log method
        log_method = getattr(self.logger, level.lower())
        
        # Create log record with extra fields
        if extra_fields:
            class MetadataLogRecord:
                def __init__(self, extra_fields):
                    self.extra_fields = extra_fields
            
            # Temporarily store extra fields in the logger
            old_makeRecord = self.logger.makeRecord
            
            def makeRecord(*args, **kwargs):
                record = old_makeRecord(*args, **kwargs)
                record.extra_fields = extra_fields
                return record
            
            self.logger.makeRecord = makeRecord
            log_method(message)
            self.logger.makeRecord = old_makeRecord
        else:
            log_method(message)
    
    def debug(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log debug message with optional metadata."""
        self._log_with_metadata('DEBUG', message, metadata)
    
    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log info message with optional metadata."""
        self._log_with_metadata('INFO', message, metadata)
    
    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log warning message with optional metadata."""
        self._log_with_metadata('WARNING', message, metadata)
    
    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message with optional metadata and exception info."""
        if exc_info:
            metadata = metadata or {}
            metadata['exception_captured'] = True
        self._log_with_metadata('ERROR', message, metadata)
    
    def critical(self, message: str, metadata: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message with optional metadata and exception info."""
        if exc_info:
            metadata = metadata or {}
            metadata['exception_captured'] = True
        self._log_with_metadata('CRITICAL', message, metadata)
    
    def log_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log an operation with contextual metadata.
        
        Args:
            operation: Name of the operation being performed
            metadata: Additional context about the operation
        """
        log_data = {
            'operation_type': 'operation_log',
            'operation_name': operation,
            'session_id': self.session_id,
            'metadata': metadata or {}
        }
        
        self.info(f"Operation: {operation}", log_data)
    
    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Additional performance metrics
        """
        perf_data = {
            'operation_type': 'performance_log',
            'operation_name': operation,
            'duration_seconds': duration,
            'session_id': self.session_id,
            'metadata': metadata or {}
        }
        
        self.info(f"Performance - {operation}: {duration:.4f}s", perf_data)
    
    def log_error_with_context(self, error: Exception, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log an error with full context information.
        
        Args:
            error: The exception that occurred
            operation: The operation during which the error occurred
            metadata: Additional context about the error
        """
        error_data = {
            'operation_type': 'error_log',
            'operation_name': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'session_id': self.session_id,
            'metadata': metadata or {}
        }
        
        self.error(f"Error in {operation}: {error}", error_data, exc_info=True)
    
    def operation_context(self, operation_name: str):
        """
        Context manager for tracking operations.
        
        Args:
            operation_name: Name of the operation to track
            
        Returns:
            Context manager that automatically logs operation start/end
        """
        return OperationContext(self, operation_name)
    
    def create_child_logger(self, child_name: str) -> 'StructuredLogger':
        """
        Create a child logger that inherits session ID and configuration.
        
        Args:
            child_name: Name for the child logger
            
        Returns:
            New StructuredLogger instance with inherited context
        """
        full_name = f"{self.name}.{child_name}"
        
        # Convert numeric level to LogLevel enum
        current_level = self.logger.level
        level_mapping = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARNING,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.CRITICAL
        }
        level_enum = level_mapping.get(current_level, LogLevel.INFO)
        
        return StructuredLogger(
            name=full_name,
            level=level_enum,
            session_id=self.session_id,
            include_system_info=self.include_system_info
        )


class OperationContext:
    """Context manager for operation tracking."""
    
    def __init__(self, logger: StructuredLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger._operation_stack.append(self.operation_name)
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger._operation_stack.pop()
        
        if exc_type:
            self.logger.log_error_with_context(
                exc_val, 
                self.operation_name,
                {'duration_seconds': duration}
            )
        else:
            self.logger.log_performance(
                self.operation_name,
                duration
            )


# Convenience function for quick logger creation
def get_logger(name: str, level: Union[str, LogLevel] = LogLevel.INFO) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        level: Log level
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level) 