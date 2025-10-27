"""
Configuration manager for crew feedback parser system.
"""
import os
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigManager:
    """
    Manages environment variables and system configuration for the crew feedback parser.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            env_file: Optional path to .env file to load
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from default .env file if present
        
        self._config = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate all required configuration values."""
        # Required environment variables
        required_vars = {
            'LLAMAINDEX_API_KEY': 'LlamaIndex API key for document parsing'
        }
        
        # Optional but recommended environment variables
        recommended_vars = {
            'OPENAI_API_KEY': 'OpenAI API key for enhanced data extraction (optional but recommended)'
        }
        
        # Optional environment variables with defaults
        optional_vars = {
            'LLAMAINDEX_API_URL': 'https://api.llamaindex.ai',
            'MAX_RETRIES': '3',
            'RETRY_DELAY': '1',
            'REQUEST_TIMEOUT': '30',
            'MAX_FILE_SIZE_MB': '50',
            'SUPPORTED_FORMATS': 'pdf,png,jpg,jpeg,tiff',
            'LOG_LEVEL': 'INFO'
        }
        
        # Load required variables
        for var_name, description in required_vars.items():
            value = os.getenv(var_name)
            if not value:
                raise ConfigurationError(
                    f"Required environment variable '{var_name}' is not set. "
                    f"This variable is needed for: {description}"
                )
            self._config[var_name] = value
        
        # Load optional variables with defaults
        for var_name, default_value in optional_vars.items():
            self._config[var_name] = os.getenv(var_name, default_value)
        
        # Validate and convert numeric values
        self._validate_numeric_configs()

    def _validate_numeric_configs(self) -> None:
        """Validate and convert numeric configuration values."""
        numeric_configs = {
            'MAX_RETRIES': int,
            'RETRY_DELAY': float,
            'REQUEST_TIMEOUT': int,
            'MAX_FILE_SIZE_MB': int
        }
        
        for config_name, config_type in numeric_configs.items():
            try:
                self._config[config_name] = config_type(self._config[config_name])
            except ValueError as e:
                raise ConfigurationError(
                    f"Invalid value for {config_name}: {self._config[config_name]}. "
                    f"Expected {config_type.__name__}."
                ) from e

    def get_api_key(self) -> str:
        """
        Get the LlamaIndex API key.
        
        Returns:
            str: The API key for LlamaIndex service
            
        Raises:
            ConfigurationError: If API key is not configured
        """
        return self._config['LLAMAINDEX_API_KEY']

    def get_api_url(self) -> str:
        """
        Get the LlamaIndex API URL.
        
        Returns:
            str: The base URL for LlamaIndex API
        """
        return self._config['LLAMAINDEX_API_URL']

    def get_max_retries(self) -> int:
        """
        Get the maximum number of API request retries.
        
        Returns:
            int: Maximum retry attempts
        """
        return self._config['MAX_RETRIES']

    def get_retry_delay(self) -> float:
        """
        Get the base delay for retry attempts in seconds.
        
        Returns:
            float: Base retry delay in seconds
        """
        return self._config['RETRY_DELAY']

    def get_request_timeout(self) -> int:
        """
        Get the API request timeout in seconds.
        
        Returns:
            int: Request timeout in seconds
        """
        return self._config['REQUEST_TIMEOUT']

    def get_max_file_size_mb(self) -> int:
        """
        Get the maximum allowed file size in megabytes.
        
        Returns:
            int: Maximum file size in MB
        """
        return self._config['MAX_FILE_SIZE_MB']

    def get_supported_formats(self) -> list[str]:
        """
        Get the list of supported file formats.
        
        Returns:
            list[str]: List of supported file extensions (lowercase)
        """
        formats_str = self._config['SUPPORTED_FORMATS']
        return [fmt.strip().lower() for fmt in formats_str.split(',')]

    def get_log_level(self) -> str:
        """
        Get the logging level.
        
        Returns:
            str: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        return self._config['LOG_LEVEL']

    def validate_api_key(self) -> bool:
        """
        Validate that the API key is properly formatted.
        
        Returns:
            bool: True if API key appears valid, False otherwise
        """
        api_key = self.get_api_key()
        
        # Basic validation - check if it's not empty and has reasonable length
        if not api_key or len(api_key.strip()) < 10:
            return False
        
        # Check for common placeholder values
        placeholder_values = ['your_api_key', 'api_key_here', 'replace_me', 'xxx']
        if api_key.lower() in placeholder_values:
            return False
        
        return True

    def get_output_settings(self) -> dict:
        """
        Get Excel output configuration settings.
        
        Returns:
            dict: Configuration for Excel output formatting
        """
        return {
            'feedback_sheet_name': 'Feedback_Data',
            'processing_sheet_name': 'Processing_Log',
            'date_format': 'YYYY-MM-DD HH:MM:SS',
            'auto_fit_columns': True,
            'freeze_header_row': True
        }

    def get_all_config(self) -> dict:
        """
        Get all configuration values (excluding sensitive data).
        
        Returns:
            dict: All configuration values with API key masked
        """
        config_copy = self._config.copy()
        # Mask sensitive information
        if 'LLAMAINDEX_API_KEY' in config_copy:
            api_key = config_copy['LLAMAINDEX_API_KEY']
            config_copy['LLAMAINDEX_API_KEY'] = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        
        return config_copy