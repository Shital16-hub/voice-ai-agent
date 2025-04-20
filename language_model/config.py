"""
Configuration settings for the language model component.
"""
from typing import Dict, Any, Optional, List
import os

# Model configuration
DEFAULT_MODEL = "mistral:7b-instruct-v0.2-q4_0"  # Quantized 4-bit model
ALTERNATIVE_MODELS = {
    "larger": "mistral:7b-instruct-v0.2",  # Full precision
    "smaller": "mistral:7b-instruct-v0.2-q5_K_M",  # 5-bit quantized
}

# Inference parameters
DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 1024,
    "stop": ["\n\nHuman:", "\n\nUser:"],
    "repeat_penalty": 1.1,
    "num_predict": 512,
}

# Ollama API settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# Context settings
MAX_CONTEXT_LENGTH = 4096  # Maximum token length for context
MAX_HISTORY_MESSAGES = 10  # Maximum number of messages to keep in history

# Response formatting
RESPONSE_FORMAT = "text"  # Options: "text", "json"
INCLUDE_METADATA = False  # Include model metadata in responses

def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for specified model or default.
    
    Args:
        model_name: Optional name of model to use
        
    Returns:
        Dictionary with model configuration
    """
    model = model_name or DEFAULT_MODEL
    
    return {
        "model": model,
        "params": DEFAULT_INFERENCE_PARAMS.copy(),
        "api_host": OLLAMA_HOST,
        "timeout": OLLAMA_TIMEOUT,
    }