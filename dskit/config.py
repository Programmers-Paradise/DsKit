"""
Configuration module for dskit
"""
import os
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    'visualization_backend': 'matplotlib',  # 'matplotlib' or 'plotly'
    'auto_save_plots': False,
    'plot_directory': './plots/',
    'default_test_size': 0.2,
    'default_random_state': 42,
    'n_jobs': -1,
    'verbose': True,
    'warning_level': 'ignore',  # 'ignore', 'warn', 'raise'
    'cache_enabled': True,
    'max_categories_for_onehot': 10,
    'correlation_threshold': 0.05,
    'missing_threshold': 0.5,  # Drop columns with > 50% missing
    'outlier_method': 'iqr',
    'scaling_method': 'standard',
    'cv_folds': 5,
    'hyperopt_max_evals': 50,
    'shap_sample_size': 1000,
    'eda_sample_size': 10000,
}

# Global configuration
_CONFIG = DEFAULT_CONFIG.copy()

def set_config(config: Dict[str, Any]) -> None:
    """
    Set global configuration parameters.
    
    Parameters:
    -----------
    config : dict
        Dictionary of configuration parameters to update
    """
    global _CONFIG
    _CONFIG.update(config)
    
    # Apply some configurations immediately
    if config.get('warning_level') == 'ignore':
        import warnings
        warnings.filterwarnings('ignore')

def get_config(key: Optional[str] = None) -> Any:
    """
    Get configuration parameter(s).
    
    Parameters:
    -----------
    key : str, optional
        Specific configuration key. If None, returns all config.
    
    Returns:
    --------
    Any : Configuration value or entire configuration dict
    """
    global _CONFIG
    if key is None:
        return _CONFIG.copy()
    return _CONFIG.get(key)

def reset_config() -> None:
    """Reset configuration to defaults."""
    global _CONFIG
    _CONFIG = DEFAULT_CONFIG.copy()

def load_config_from_file(filepath: str) -> None:
    """
    Load configuration from a JSON or YAML file.
    
    Parameters:
    -----------
    filepath : str
        Path to configuration file
    """
    import json
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            config = json.load(f)
        elif filepath.endswith(('.yml', '.yaml')):
            try:
                import yaml
                config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configuration files")
        else:
            raise ValueError("Configuration file must be JSON or YAML")
    
    set_config(config)

def save_config_to_file(filepath: str) -> None:
    """
    Save current configuration to a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to save configuration file
    """
    import json
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(_CONFIG, f, indent=2)

def print_config() -> None:
    """Print current configuration."""
    print("dskit Configuration:")
    print("-" * 30)
    for key, value in _CONFIG.items():
        print(f"{key}: {value}")

# Configuration context manager
class config_context:
    """
    Context manager for temporary configuration changes.
    
    Example:
    --------
    with config_context({'verbose': False, 'random_state': 123}):
        # Code here runs with temporary config
        kit.train()
    # Config is restored after the block
    """
    
    def __init__(self, temp_config: Dict[str, Any]):
        self.temp_config = temp_config
        self.original_config = None
    
    def __enter__(self):
        global _CONFIG
        self.original_config = _CONFIG.copy()
        _CONFIG.update(self.temp_config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CONFIG
        _CONFIG = self.original_config

# Initialize configuration from environment variables
def _init_from_env():
    """Initialize configuration from environment variables."""
    env_mapping = {
        'dskit_VERBOSE': ('verbose', lambda x: x.lower() == 'true'),
        'dskit_RANDOM_STATE': ('default_random_state', int),
        'dskit_N_JOBS': ('n_jobs', int),
        'dskit_BACKEND': ('visualization_backend', str),
        'dskit_AUTO_SAVE': ('auto_save_plots', lambda x: x.lower() == 'true'),
    }
    
    for env_var, (config_key, converter) in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                _CONFIG[config_key] = converter(value)
            except (ValueError, TypeError):
                print(f"Warning: Invalid value for {env_var}: {value}")

# Initialize on import
_init_from_env()