"""
Centralized logging configuration for the AI Trading Lab.

Features:
- Colored console output for development
- Structured JSON output for production
- Log levels configurable via environment
- Module-specific loggers with prefixes
"""

import logging
import os
import sys
from typing import Optional


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Log levels
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta
    
    # Component prefixes
    DEBATE = "\033[94m"     # Light blue
    CONSENSUS = "\033[95m"  # Light magenta
    SIGNALS = "\033[96m"    # Light cyan
    EXEC = "\033[93m"       # Light yellow
    TRADE = "\033[92m"      # Light green
    API = "\033[97m"        # White


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Get color for log level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")
        
        # Build the log line
        level = f"{color}{record.levelname:8}{Colors.RESET}"
        
        # Add component prefix if present in message
        message = record.getMessage()
        
        # Color component prefixes
        prefix_colors = {
            "[DEBATE]": Colors.DEBATE,
            "[CONSENSUS]": Colors.CONSENSUS,
            "[SIGNALS]": Colors.SIGNALS,
            "[EXEC]": Colors.EXEC,
            "[TRADE]": Colors.TRADE,
            "[API]": Colors.API,
        }
        
        for prefix, prefix_color in prefix_colors.items():
            if message.startswith(prefix):
                message = f"{prefix_color}{Colors.BOLD}{prefix}{Colors.RESET}" + \
                          message[len(prefix):]
                break
        
        # Format: TIME LEVEL MODULE MESSAGE
        module = record.name.split(".")[-1][:15]
        return f"{timestamp} {level} {module:15} {message}"


class JsonFormatter(logging.Formatter):
    """JSON formatter for production logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key in ["fund_id", "symbol", "action", "decision_id"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        return json.dumps(log_data)


def setup_logging(
    level: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var.
        json_format: Use JSON format (for production). Defaults to LOG_FORMAT env var.
    """
    # Get level from env or parameter
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Check if JSON format requested
    use_json = json_format or os.getenv("LOG_FORMAT", "").lower() == "json"
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ColoredFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    root_logger.addHandler(handler)
    
    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, format={'json' if use_json else 'colored'}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience loggers for common components
def debate_log(message: str, level: int = logging.INFO, **kwargs) -> None:
    """Log a debate-related message."""
    logger = logging.getLogger("debate")
    logger.log(level, f"[DEBATE] {message}", **kwargs)


def consensus_log(message: str, level: int = logging.INFO, **kwargs) -> None:
    """Log a consensus-related message."""
    logger = logging.getLogger("consensus")
    logger.log(level, f"[CONSENSUS] {message}", **kwargs)


def signals_log(message: str, level: int = logging.INFO, **kwargs) -> None:
    """Log a signals-related message."""
    logger = logging.getLogger("signals")
    logger.log(level, f"[SIGNALS] {message}", **kwargs)


def exec_log(message: str, level: int = logging.INFO, **kwargs) -> None:
    """Log an execution-related message."""
    logger = logging.getLogger("execution")
    logger.log(level, f"[EXEC] {message}", **kwargs)


def trade_log(message: str, level: int = logging.INFO, **kwargs) -> None:
    """Log a trade-related message."""
    logger = logging.getLogger("trade")
    logger.log(level, f"[TRADE] {message}", **kwargs)
