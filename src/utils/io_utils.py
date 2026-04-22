from pathlib import Path
import yaml
import json
import logging
from typing import Dict, Any


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file

    Returns:
        Dictionary containing the loaded configuration
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data: Dictionary to save
        path: Path where to save the JSON file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """
    Set up a logger that writes to both file and stdout.

    Args:
        name: Logger name
        log_file: Path to the log file

    Returns:
        Configured logger instance
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
