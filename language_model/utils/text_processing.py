"""
Text processing utilities for language model inputs and outputs.
"""
import re
from typing import List, Dict, Any, Optional

def clean_transcription(text: str) -> str:
    """
    Clean speech-to-text transcription output.
    
    Args:
        text: Raw transcription text
        
    Returns:
        Cleaned text
    """
    # Remove filler words
    fillers = [
        r'\bum\b', r'\buh\b', r'\ber\b', r'\blike\b(?! to)', 
        r'\byou know\b', r'\bI mean\b', r'\bso\b(?= )',
        r'\bkind of\b', r'\bsort of\b'
    ]
    for filler in fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)
    
    # Clean up double spaces
    text = re.sub(r' +', ' ', text)
    
    # Capitalize first letter of sentences
    text = re.sub(r'(?<=[\.\?\!]\s)([a-z])', lambda m: m.group(1).upper(), text)
    text = text[0].upper() + text[1:] if text else text
    
    return text.strip()

def truncate_text(text: str, max_tokens: int = 8192) -> str:
    """
    Truncate text to stay within token limits.
    
    This is a simplified estimator - in practice, you might want
    to use a proper tokenizer for the specific model.
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text
    """
    # Simple token estimation (assumes average ~4 chars per token)
    estimated_tokens = len(text) / 4
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate to approximate token count
    # Leave some margin to be safe
    safe_char_count = int(max_tokens * 3.8)
    
    # Try to truncate at sentence boundary
    truncated = text[:safe_char_count]
    last_sentence_end = max(
        truncated.rfind('.'), 
        truncated.rfind('!'), 
        truncated.rfind('?')
    )
    
    if last_sentence_end > 0:
        return text[:last_sentence_end + 1]
    
    # If no sentence boundary found, truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return text[:last_space] + "..."
    
    # Last resort - hard truncate
    return text[:safe_char_count] + "..."

def detect_language(text: str) -> str:
    """
    Simple language detection.
    
    Args:
        text: Input text
        
    Returns:
        ISO language code (en, es, fr, etc.)
    """
    try:
        # Try to use langdetect if available
        from langdetect import detect
        return detect(text)
    except ImportError:
        # Fallback to simple heuristics
        # Count characters that are common in different languages
        text = text.lower()
        
        # English
        if re.search(r'\b(the|and|is|in|to|it|of)\b', text):
            return "en"
        
        # Spanish
        if re.search(r'\b(el|la|los|las|es|en|por|que)\b', text):
            return "es"
        
        # French
        if re.search(r'\b(le|la|les|est|dans|pour|que|vous)\b', text):
            return "fr"
        
        # German
        if re.search(r'\b(der|die|das|ist|und|oder|für)\b', text):
            return "de"
        
        # Default to English
        return "en"

def extract_questions(text: str) -> List[str]:
    """
    Extract questions from text.
    
    Args:
        text: Input text
        
    Returns:
        List of identified questions
    """
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Identify questions
    questions = [
        sentence.strip() 
        for sentence in sentences 
        if sentence.strip().endswith('?')
    ]
    
    # Also look for implicit questions
    implicit_patterns = [
        r'(?i)can you', r'(?i)could you', r'(?i)would you',
        r'(?i)please tell me', r'(?i)i want to know'
    ]
    
    for sentence in sentences:
        if sentence not in questions:  # Skip already identified questions
            for pattern in implicit_patterns:
                if re.search(pattern, sentence):
                    questions.append(sentence.strip())
                    break
    
    return questions

def format_conversation_context(
    messages: List[Dict[str, str]], 
    max_length: Optional[int] = None
) -> str:
    """
    Format conversation history to string format.
    
    Args:
        messages: List of message dictionaries
        max_length: Optional maximum length
        
    Returns:
        Formatted conversation history
    """
    formatted = []
    
    for message in messages:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")
        
        if role == "System":
            continue  # Skip system messages in the formatted output
        
        formatted.append(f"{role}: {content}")
    
    result = "\n\n".join(formatted)
    
    if max_length and len(result) > max_length:
        return truncate_text(result, max_length // 4)  # Rough token estimate
    
    return result