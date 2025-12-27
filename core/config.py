"""
Configuration settings for the translation pipeline.
DEPRECATED: Use config_loader.py and config.yaml instead.
This file is kept for backward compatibility.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration - loaded from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = "nvidia/riva-translate-4b-instruct-v1.1"

FANAR_API_KEY = os.getenv("FANAR_API_KEY", "")
FANAR_API_URL = os.getenv("FANAR_API_URL", "https://api.fanar.qa/v1/chat/completions")
FANAR_MODEL = "Fanar"

# Translation Parameters
TRANSLATION_PARAMS = {
    "nvidia": {
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.7,
    },
    "fanar": {
        "temperature": 0.3,
        "max_tokens": 1024,
    }
}

# Retry Configuration
MAX_RETRIES = 3
BACKOFF_MULTIPLIER = 2
REQUEST_TIMEOUT = 30

# Checkpoint Configuration
CHECKPOINT_INTERVAL = 50
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Output Configuration
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset Configuration
SUPPORTED_FORMATS = ["csv", "tsv", "json", "jsonl", "parquet"]
DEFAULT_ENCODING = "utf-8"
CSV_DELIMITER_AUTO_DETECT = True

# Provider Term Mappings (for preprocessing)
PROVIDER_TERMS = {
    # OpenAI-specific terms
    "ChatGPT": "the AI assistant",
    "chatgpt": "the AI assistant",
    "Chat GPT": "the AI assistant",
    
    "OpenAI": "the AI provider",
    "openai": "the AI provider",
    "OpenAI's": "the AI provider's",
    
    "GPT-4": "the advanced language model",
    "GPT-3.5": "the language model",
    "GPT-3": "the language model",
    "gpt-4": "the advanced language model",
    "gpt-3.5": "the language model",
    
    # Generic replacements
    "OpenAI API": "the AI API",
    "ChatGPT API": "the AI API",
}

# Translation Validation
MIN_TRANSLATION_LENGTH = 1
MAX_LENGTH_RATIO = 3.0
MIN_LENGTH_RATIO = 0.3
MIN_ARABIC_CHAR_RATIO = 0.5

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, REPORTS_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
