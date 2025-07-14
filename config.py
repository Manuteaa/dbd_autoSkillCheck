"""
Configuration management for DBD Auto Skill Check.

This module handles application configuration, defaults, and validation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration with defaults and validation."""
    
    # Model settings
    models_folder: str = "models"
    default_model: str = "model.onnx"
    
    # Performance settings
    default_cpu_threads: int = 4
    max_cpu_threads: int = 8
    target_fps: int = 60
    
    # Skill check timing
    default_ante_delay: int = 20  # milliseconds
    max_ante_delay: int = 50
    cooldown_duration: float = 0.5  # seconds
    key_press_duration: float = 0.005  # seconds
    
    # Display settings
    default_crop_size: int = 224
    debug_crop_size: int = 520
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "dbd_auto_skill_check.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    
    # UI settings
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860
    ui_theme: str = "soft"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.default_cpu_threads < 1 or self.default_cpu_threads > self.max_cpu_threads:
            raise ValueError(f"CPU threads must be between 1 and {self.max_cpu_threads}")
        
        if self.default_ante_delay < 0 or self.default_ante_delay > self.max_ante_delay:
            raise ValueError(f"Ante delay must be between 0 and {self.max_ante_delay}")
        
        if self.target_fps < 10 or self.target_fps > 120:
            raise ValueError("Target FPS must be between 10 and 120")
        
        if not os.path.exists(self.models_folder):
            logger.warning(f"Models folder does not exist: {self.models_folder}")


class ConfigManager:
    """Manages application configuration loading, saving, and access."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> AppConfig:
        """
        Load configuration from file or create default.
        
        Returns:
            AppConfig instance with loaded or default values
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return AppConfig(**config_data)
            else:
                logger.info("No config file found, using defaults")
                return AppConfig()
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return AppConfig()
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            config_dict = asdict(self.config)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = getattr(self.config, key, default)
            return value
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.config.validate()
                return True
            else:
                logger.warning(f"Unknown configuration key: {key}")
                return False
        except Exception as e:
            logger.error(f"Error setting config {key}={value}: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = AppConfig()
        logger.info("Configuration reset to defaults")


# Global configuration instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config_manager.config

def save_config() -> bool:
    """Save the global configuration."""
    return config_manager.save_config()