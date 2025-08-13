"""
Configuration loader with environment variable support.
"""
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv


def load_config_with_env(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file and override with environment variables.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary with environment variable overrides
    """
    # Load .env file if it exists (for local development)
    load_dotenv()
    
    # Load base config from YAML
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override with environment variables
    alpaca_config = config.get('alpaca', {})
    alpaca_config['api_key'] = os.getenv('ALPACA_API_KEY', alpaca_config.get('api_key', ''))
    alpaca_config['secret_key'] = os.getenv('ALPACA_SECRET_KEY', alpaca_config.get('secret_key', ''))
    
    telegram_config = config.get('telegram', {})
    telegram_config['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', telegram_config.get('bot_token', ''))
    telegram_config['chat_id'] = os.getenv('TELEGRAM_CHAT_ID', telegram_config.get('chat_id', ''))
    
    return config