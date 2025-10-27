"""
Configuration management for the crew feedback parser system.
Handles environment variables and system configuration validation.
"""

import os
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigManager:
    """Manages system configuration and environment variables."""
    
    def __init__(self):
        """Initialize configuration manager and load environment variables."""
        load_dotenv()
        self._validate_required_config()
    
    def _validate_required_config(self) -> None:
        """Validate that all required environment variables are set."""
        required_vars = ['LLAMAINDEX_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please set these variables in your .env file or environment."
            )
    
    def get_api_key(self) -> str:
        """Get the LlamaIndex API key."""
        api_key = os.getenv('LLAMAINDEX_API_KEY')
        if not api_key:
            raise ConfigurationError("LLAMAINDEX_API_KEY not found in environment variables")
        return api_key
    
    def get_api_base_url(self) -> str:
        """Get the LlamaIndex API base URL."""
        return os.getenv('LLAMAINDEX_API_URL', 'https://api.llamaindex.ai')
    
    def get_max_retries(self) -> int:
        """Get maximum number of API retries."""
        try:
            return int(os.getenv('MAX_API_RETRIES', '3'))
        except ValueError:
            return 3
    
    def get_request_timeout(self) -> int:
        """Get API request timeout in seconds."""
        try:
            return int(os.getenv('REQUEST_TIMEOUT', '30'))
        except ValueError:
            return 30
    
    def get_batch_size(self) -> int:
        """Get batch processing size."""
        try:
            return int(os.getenv('BATCH_SIZE', '10'))
        except ValueError:
            return 10
    
    def get_log_level(self) -> str:
        """Get logging level."""
        return os.getenv('LOG_LEVEL', 'INFO').upper()
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes')


# Global configuration instance
config = ConfigManager()