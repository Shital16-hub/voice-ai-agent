"""
Configuration utility for the speech-to-text module.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration loader for the speech-to-text module.
    
    This class handles loading and validating configuration from
    JSON files and environment variables.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigLoader.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_data: Dict[str, Any] = {}
        
        # Load default configuration
        self._load_default_config()
        
        # Load user configuration if provided
        if config_path:
            self._load_user_config(config_path)
        
        # Override with environment variables
        self._load_env_vars()
        
        logger.debug(f"Loaded configuration: {self.config_data}")
    
    def _load_default_config(self):
        """Load default configuration."""
        default_config = {
            "speech_to_text": {
                "model": {
                    "path": "models/ggml-base.en.bin",
                    "language": "en",
                    "n_threads": 4,
                    "translate": False
                },
                "audio": {
                    "sample_rate": 16000,
                    "chunk_size_ms": 1000,
                    "overlap_ms": 200,
                    "silence_threshold": 0.01,
                    "min_silence_ms": 500,
                    "max_chunk_size_ms": 30000
                },
                "streaming": {
                    "vad_enabled": True,
                    "vad_threshold": 0.3,
                    "max_context_length": 5
                }
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "cors_origins": ["*"],
                "max_connections": 10
            }
        }
        
        self.config_data = default_config
    
    def _load_user_config(self, config_path: str):
        """
        Load user configuration.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            
            # Merge with default config
            self._merge_configs(self.config_data, user_config)
        except Exception as e:
            logger.warning(f"Failed to load user config from {config_path}: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """
        Recursively merge configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
        """
        for key, value in override.items():
            if (
                key in base and 
                isinstance(base[key], dict) and 
                isinstance(value, dict)
            ):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        # Look for environment variables with prefix STT_
        for key, value in os.environ.items():
            if key.startswith("STT_"):
                # Convert environment variable name to config path
                # e.g., STT_MODEL_LANGUAGE -> speech_to_text.model.language
                path = key[4:].lower().replace("_", ".")
                
                # Set value in config
                self._set_config_value(path, value)
    
    def _set_config_value(self, path: str, value: str):
        """
        Set configuration value by path.
        
        Args:
            path: Configuration path (e.g., "speech_to_text.model.language")
            value: Configuration value
        """
        try:
            # Split path into parts
            parts = path.split(".")
            
            # Navigate to the correct part of the config
            current = self.config_data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value, converting to appropriate type
            last_part = parts[-1]
            
            # Try to infer the correct type
            if value.lower() == "true":
                current[last_part] = True
            elif value.lower() == "false":
                current[last_part] = False
            elif value.isdigit():
                current[last_part] = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                current[last_part] = float(value)
            else:
                current[last_part] = value
        except Exception as e:
            logger.warning(f"Failed to set config value for {path}: {e}")
    
    def get_config(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            path: Configuration path (e.g., "speech_to_text.model.language")
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        if not path:
            return self.config_data
        
        try:
            # Split path into parts
            parts = path.split(".")
            
            # Navigate to the correct part of the config
            current = self.config_data
            for part in parts:
                current = current[part]
            
            return current
        except (KeyError, TypeError):
            return default
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write configuration to file
            with open(output_path, "w") as f:
                json.dump(self.config_data, f, indent=4)
                
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {output_path}: {e}")