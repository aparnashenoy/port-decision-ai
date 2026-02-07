"""
Logging utilities for Port Decision AI system.

Provides structured logging configuration with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up structured logging with file and console handlers.
    
    Args:
        log_dir: Directory to store log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file name. If None, uses 'port_decision_ai.log'.
    
    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_file is None:
        log_file = "port_decision_ai.log"
    
    log_path = log_dir / log_file
    
    # Create logger
    logger = logging.getLogger("port_decision_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name. If None, returns root logger.
    
    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"port_decision_ai.{name}")
    return logging.getLogger("port_decision_ai")

