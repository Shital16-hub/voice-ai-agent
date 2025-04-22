"""
Configuration for the LangGraph-based Voice AI Agent.

This module provides configuration settings and constants
for the LangGraph integration.
"""
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

@dataclass
class LangGraphConfig:
    """Configuration for the LangGraph integration."""
    
    # STT configuration
    stt_model: str = "tiny.en"
    stt_language: str = "en"
    
    # KB configuration
    kb_temperature: float = 0.7
    kb_max_tokens: int = 1024
    kb_include_sources: bool = True
    
    # TTS configuration
    tts_voice: Optional[str] = None
    
    # Graph configuration
    enable_human_handoff: bool = True
    confidence_threshold: float = 0.7  # Threshold for human handoff
    
    # Debugging
    debug_mode: bool = False
    save_state_history: bool = False
    state_history_path: Optional[str] = None
    
    # Performance
    enable_streaming: bool = True
    
    # Custom node settings
    custom_node_settings: Dict[str, Any] = field(default_factory=dict)

# Default configuration
DEFAULT_CONFIG = LangGraphConfig()

# Router decision mapping
ROUTER_DECISIONS = {
    "stt_failure": "human_handoff",
    "kb_failure": "human_handoff",
    "tts_failure": "kb",  # Skip TTS on failure, return text response
    "low_confidence": "human_handoff",
    "default": "stt"
}

# Node mapping for state transitions
NODE_MAPPING = {
    "stt": "speech-to-text",
    "kb": "knowledge-base",
    "tts": "text-to-speech",
    "router": "router",
    "human_handoff": "human-handoff"
}