"""
Configuration management utility for the horse racing simulation app.

This module provides functionality to load and manage application configuration
from YAML files with proper error handling and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages application configuration from YAML files.
    
    Provides centralized access to configuration settings with
    proper error handling and environment-specific overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Get the project root directory (3 levels up from utils)
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "app_config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file) or {}
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration as fallback
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values as fallback."""
        return {
            'app': {
                'title': '競馬レース時間予測システム',
                'description': 'LightGBMを使用した競馬レース時間予測',
                'version': '1.0.0'
            },
            'ui': {
                'max_file_size_mb': 10,
                'allowed_file_types': ['csv'],
                'default_page_size': 20
            },
            'model': {
                'prediction_threshold': 0.95,
                'batch_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key: Configuration key in dot notation (e.g., 'app.title')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.warning(f"Failed to get config value for key '{key}': {e}")
            return default
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application-specific configuration."""
        return self.get('app', {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-specific configuration."""
        return self.get('ui', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.get('model', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def reload(self) -> bool:
        """
        Reload configuration from file.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self._load_config()
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ['app', 'ui', 'model']
            for section in required_sections:
                if section not in self._config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate specific values
            ui_config = self.get_ui_config()
            if ui_config.get('max_file_size_mb', 0) <= 0:
                logger.error("Invalid max_file_size_mb value")
                return False
            
            model_config = self.get_model_config()
            threshold = model_config.get('prediction_threshold', 0)
            if not (0 < threshold <= 1):
                logger.error("Invalid prediction_threshold value")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def reload_config() -> bool:
    """Reload the global configuration."""
    return config_manager.reload()
