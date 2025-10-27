"""
Comprehensive logging configuration for crew feedback parser system.
Provides detailed logging for debugging and monitoring with proper error handling.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LoggingConfig:
    """
    Configures comprehensive logging for the crew feedback parser system.
    Supports multiple log levels, file rotation, and structured logging.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional, defaults to logs/crew_feedback_parser.log)
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            max_file_size_mb: Maximum log file size in MB before rotation
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        
        # Set default log file path if not provided
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.log_file = log_dir / "crew_feedback_parser.log"
        else:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging with appropriate handlers and formatters."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
        
        # Log the logging configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured - Level: {logging.getLevelName(self.log_level)}, "
                   f"Console: {self.enable_console}, File: {self.enable_file}")
        if self.enable_file:
            logger.info(f"Log file: {self.log_file}")

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)

    def log_system_info(self) -> None:
        """Log system information for debugging purposes."""
        logger = logging.getLogger(__name__)
        logger.info("=== System Information ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Current working directory: {Path.cwd()}")
        logger.info(f"Log file location: {self.log_file}")
        logger.info("=== End System Information ===")

    def set_module_log_level(self, module_name: str, level: str) -> None:
        """
        Set log level for a specific module.
        
        Args:
            module_name: Name of the module
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        main_logger = logging.getLogger(__name__)
        main_logger.info(f"Set log level for {module_name} to {level.upper()}")


class ErrorHandler:
    """
    Comprehensive error handling utilities for the crew feedback parser system.
    Provides structured error logging and recovery mechanisms.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance (optional, creates default if not provided)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.error_history: list = []

    def handle_file_error(self, 
                         file_path: str, 
                         error: Exception, 
                         operation: str = "processing") -> Dict[str, Any]:
        """
        Handle file-related errors with appropriate logging and recovery suggestions.
        
        Args:
            file_path: Path to the file that caused the error
            error: The exception that occurred
            operation: Description of the operation being performed
            
        Returns:
            Dict containing error details and recovery suggestions
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log the error with context
        self.logger.error(f"File {operation} error for {file_path}: {error_type} - {error_message}")
        
        # Track error statistics
        self._track_error(error_type)
        
        # Determine recovery suggestions based on error type
        recovery_suggestions = self._get_file_error_recovery_suggestions(error, file_path)
        
        error_details = {
            'file_path': file_path,
            'error_type': error_type,
            'error_message': error_message,
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'recovery_suggestions': recovery_suggestions,
            'is_recoverable': self._is_recoverable_file_error(error)
        }
        
        # Add to error history
        self.error_history.append(error_details)
        
        return error_details

    def handle_api_error(self, 
                        error: Exception, 
                        file_name: str = None, 
                        retry_count: int = 0) -> Dict[str, Any]:
        """
        Handle API-related errors with appropriate logging and retry logic.
        
        Args:
            error: The API exception that occurred
            file_name: Name of file being processed (optional)
            retry_count: Current retry attempt number
            
        Returns:
            Dict containing error details and retry recommendations
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log with appropriate level based on error type
        if "rate limit" in error_message.lower() or "429" in error_message:
            self.logger.warning(f"API rate limit hit for {file_name or 'unknown file'} "
                              f"(attempt {retry_count + 1}): {error_message}")
        elif "timeout" in error_message.lower():
            self.logger.warning(f"API timeout for {file_name or 'unknown file'} "
                              f"(attempt {retry_count + 1}): {error_message}")
        else:
            self.logger.error(f"API error for {file_name or 'unknown file'} "
                            f"(attempt {retry_count + 1}): {error_type} - {error_message}")
        
        # Track error statistics
        self._track_error(f"api_{error_type.lower()}")
        
        # Determine retry recommendations
        should_retry, retry_delay = self._get_api_retry_recommendation(error, retry_count)
        
        error_details = {
            'error_type': error_type,
            'error_message': error_message,
            'file_name': file_name,
            'retry_count': retry_count,
            'timestamp': datetime.now().isoformat(),
            'should_retry': should_retry,
            'retry_delay_seconds': retry_delay,
            'is_rate_limit': "rate limit" in error_message.lower() or "429" in error_message,
            'is_timeout': "timeout" in error_message.lower()
        }
        
        # Add to error history
        self.error_history.append(error_details)
        
        return error_details

    def handle_extraction_error(self, 
                              error: Exception, 
                              file_name: str, 
                              text_length: int = 0) -> Dict[str, Any]:
        """
        Handle data extraction errors with appropriate logging and analysis.
        
        Args:
            error: The extraction exception that occurred
            file_name: Name of file being processed
            text_length: Length of text that was being processed
            
        Returns:
            Dict containing error details and analysis
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        self.logger.error(f"Data extraction error for {file_name}: {error_type} - {error_message}")
        
        # Track error statistics
        self._track_error(f"extraction_{error_type.lower()}")
        
        # Analyze potential causes
        potential_causes = self._analyze_extraction_error(error, text_length)
        
        error_details = {
            'error_type': error_type,
            'error_message': error_message,
            'file_name': file_name,
            'text_length': text_length,
            'timestamp': datetime.now().isoformat(),
            'potential_causes': potential_causes,
            'is_recoverable': False  # Extraction errors typically require manual review
        }
        
        # Add to error history
        self.error_history.append(error_details)
        
        return error_details

    def _track_error(self, error_type: str) -> None:
        """Track error statistics for monitoring."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def _get_file_error_recovery_suggestions(self, error: Exception, file_path: str) -> list:
        """Get recovery suggestions for file errors."""
        suggestions = []
        error_message = str(error).lower()
        
        if "permission" in error_message:
            suggestions.extend([
                "Check file permissions and ensure read access",
                "Verify the file is not locked by another application",
                "Run the application with appropriate user permissions"
            ])
        elif "not found" in error_message:
            suggestions.extend([
                "Verify the file path is correct",
                "Check if the file was moved or deleted",
                "Ensure the file exists before processing"
            ])
        elif "corrupted" in error_message or "invalid" in error_message:
            suggestions.extend([
                "Check if the file is corrupted or incomplete",
                "Try opening the file manually to verify integrity",
                "Re-scan or re-create the document if possible"
            ])
        elif "size" in error_message or "large" in error_message:
            suggestions.extend([
                "File may be too large for processing",
                "Consider compressing or optimizing the file",
                "Check system memory and disk space"
            ])
        else:
            suggestions.append("Review file format and ensure it's supported")
        
        return suggestions

    def _is_recoverable_file_error(self, error: Exception) -> bool:
        """Determine if a file error is potentially recoverable."""
        error_message = str(error).lower()
        
        # Recoverable errors (temporary issues)
        recoverable_indicators = ["timeout", "busy", "locked", "temporary"]
        
        # Non-recoverable errors (permanent issues)
        non_recoverable_indicators = ["not found", "permission denied", "corrupted", "invalid format"]
        
        if any(indicator in error_message for indicator in non_recoverable_indicators):
            return False
        
        if any(indicator in error_message for indicator in recoverable_indicators):
            return True
        
        # Default to non-recoverable for safety
        return False

    def _get_api_retry_recommendation(self, error: Exception, retry_count: int) -> tuple:
        """Get retry recommendation for API errors."""
        error_message = str(error).lower()
        max_retries = 3
        
        # Rate limiting - always retry with exponential backoff
        if "rate limit" in error_message or "429" in error_message:
            if retry_count < max_retries:
                delay = min(60, 2 ** retry_count)  # Exponential backoff, max 60s
                return True, delay
        
        # Timeout errors - retry with shorter timeout
        elif "timeout" in error_message:
            if retry_count < max_retries:
                delay = 5 + retry_count * 2  # Linear increase
                return True, delay
        
        # Server errors (5xx) - retry with backoff
        elif any(code in error_message for code in ["500", "502", "503", "504"]):
            if retry_count < max_retries:
                delay = 2 ** retry_count  # Exponential backoff
                return True, delay
        
        # Authentication errors - don't retry
        elif any(code in error_message for code in ["401", "403"]):
            return False, 0
        
        # Client errors (4xx except 429) - don't retry
        elif any(code in error_message for code in ["400", "404", "422"]):
            return False, 0
        
        # Unknown errors - retry once
        elif retry_count == 0:
            return True, 5
        
        return False, 0

    def _analyze_extraction_error(self, error: Exception, text_length: int) -> list:
        """Analyze potential causes of extraction errors."""
        causes = []
        error_message = str(error).lower()
        
        if text_length == 0:
            causes.append("No text was extracted from the document")
        elif text_length < 100:
            causes.append("Very little text extracted - document may be mostly images or corrupted")
        
        if "validation" in error_message:
            causes.append("Data validation failed - extracted values may be outside expected ranges")
        elif "pattern" in error_message or "regex" in error_message:
            causes.append("Pattern matching failed - document format may be different than expected")
        elif "encoding" in error_message:
            causes.append("Text encoding issues - document may contain special characters")
        else:
            causes.append("Unexpected extraction error - manual review required")
        
        return causes

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors encountered.
        
        Returns:
            Dict containing error statistics and recent errors
        """
        total_errors = sum(self.error_counts.values())
        recent_errors = self.error_history[-10:]  # Last 10 errors
        
        return {
            'total_errors': total_errors,
            'error_counts_by_type': self.error_counts.copy(),
            'recent_errors': recent_errors,
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }

    def clear_error_history(self) -> None:
        """Clear error history and statistics."""
        self.error_counts.clear()
        self.error_history.clear()
        self.logger.info("Error history and statistics cleared")

    def log_error_summary(self) -> None:
        """Log a summary of errors for monitoring."""
        summary = self.get_error_summary()
        
        if summary['total_errors'] == 0:
            self.logger.info("No errors encountered during processing")
            return
        
        self.logger.warning(f"Error Summary: {summary['total_errors']} total errors")
        
        for error_type, count in summary['error_counts_by_type'].items():
            self.logger.warning(f"  {error_type}: {count} occurrences")
        
        if summary['most_common_error']:
            self.logger.warning(f"Most common error: {summary['most_common_error']}")


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 enable_console: bool = True) -> tuple:
    """
    Convenience function to set up logging and error handling.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
        
    Returns:
        Tuple of (LoggingConfig, ErrorHandler)
    """
    logging_config = LoggingConfig(
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console
    )
    
    error_handler = ErrorHandler(logging_config.get_logger(__name__))
    
    # Log system information
    logging_config.log_system_info()
    
    return logging_config, error_handler