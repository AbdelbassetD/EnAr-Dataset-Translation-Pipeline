"""
Configuration loader with YAML and environment variable support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load and merge configuration from YAML files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML config file (defaults to config.yaml in project root)
        """
        self.base_dir = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else self.base_dir / "config.yaml"
        self.config = {}
        
        # Load environment variables from .env file
        load_dotenv(self.base_dir / ".env")
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML and environment variables.
        
        Returns:
            Merged configuration dictionary
        """
        # Load defaults
        self.config = self._get_defaults()
        
        # Load from YAML if exists
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
                self._merge_config(self.config, yaml_config)
        
        # Override with environment variables
        self._load_env_vars()
        
        return self.config
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'dataset': {
                'source': None,
                'file_name': None,
                'columns_to_translate': 'auto',
            },
            'translation': {
                'primary_api': 'nvidia',
                'enable_fallback': False,
                'normalize_provider_terms': True,
                'preserve_all_columns': True,
            },
            'apis': {
                'nvidia': {
                    'model': 'nvidia/riva-translate-4b-instruct-v1.1',
                    'temperature': 0.1,
                    'max_tokens': 512,
                    'top_p': 0.7,
                    'rate_limit_rpm': 40,
                },
                'fanar': {
                    'model': 'Fanar',
                    'temperature': 0.3,
                    'max_tokens': 1024,
                    'rate_limit_rpm': 40,
                }
            },
            'retry': {
                'max_retries': 3,
                'backoff_multiplier': 2,
                'request_timeout': 30,
                'respect_rate_limits': True,
            },
            'checkpoint': {
                'enabled': True,
                'interval': 50,
                'resume_from_last': False,
            },
            'output': {
                'path': 'data/translated_output.csv',
                'format': 'csv',
                'save_statistics': True,
            }
        }
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_env_vars(self):
        """Load API credentials from environment variables."""
        # NVIDIA API
        nvidia_key = os.getenv('NVIDIA_API_KEY')
        nvidia_url = os.getenv('NVIDIA_BASE_URL')
        if nvidia_key:
            self.config.setdefault('env', {})['nvidia_api_key'] = nvidia_key
        if nvidia_url:
            self.config.setdefault('env', {})['nvidia_base_url'] = nvidia_url
            
        # Fanar API
        fanar_key = os.getenv('FANAR_API_KEY')
        fanar_url = os.getenv('FANAR_API_URL')
        if fanar_key:
            self.config.setdefault('env', {})['fanar_api_key'] = fanar_key
        if fanar_url:
            self.config.setdefault('env', {})['fanar_api_url'] = fanar_url
            
        # Kaggle API
        kaggle_user = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        if kaggle_user:
            self.config.setdefault('env', {})['kaggle_username'] = kaggle_user
        if kaggle_key:
            self.config.setdefault('env', {})['kaggle_key'] = kaggle_key
    
    def validate(self) -> bool:
        """
        Validate required configuration.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for API keys
        env = self.config.get('env', {})
        primary_api = self.config['translation']['primary_api']
        
        if primary_api == 'nvidia':
            if not env.get('nvidia_api_key'):
                raise ValueError(
                    "NVIDIA_API_KEY not found in environment. "
                    "Please set it in .env file or environment variables."
                )
        elif primary_api == 'fanar':
            if not env.get('fanar_api_key'):
                raise ValueError(
                    "FANAR_API_KEY not found in environment. "
                    "Please set it in .env file or environment variables."
                )
        
        return True


if __name__ == "__main__":
    # Test configuration loading
    loader = ConfigLoader()
    config = loader.load()
    
    print("Configuration loaded successfully:")
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
