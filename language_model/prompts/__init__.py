"""
Prompt templates and system prompts for language model.
"""
from language_model.prompts.system_prompts import get_system_prompt, create_custom_system_prompt
from language_model.prompts.templates import (
    format_chat_history,
    create_chat_message,
    create_rag_prompt,
    create_rag_chat_messages,
    create_voice_optimized_prompt,
    format_json_response
)

__all__ = [
    "get_system_prompt",
    "create_custom_system_prompt",
    "format_chat_history",
    "create_chat_message",
    "create_rag_prompt",
    "create_rag_chat_messages",
    "create_voice_optimized_prompt",
    "format_json_response",
]