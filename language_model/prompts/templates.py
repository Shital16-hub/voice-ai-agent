"""
Templates for constructing prompts in different formats.
"""
from typing import List, Dict, Any, Optional, Union
import json

def format_chat_history(
    messages: List[Dict[str, str]],
    include_system: bool = True
) -> List[Dict[str, str]]:
    """
    Format conversation history for the chat API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        include_system: Whether to include system messages
        
    Returns:
        Formatted message list for API
    """
    if not include_system:
        return [msg for msg in messages if msg["role"] != "system"]
    
    return messages

def create_chat_message(
    role: str,
    content: str
) -> Dict[str, str]:
    """
    Create a properly formatted chat message.
    
    Args:
        role: Message role (system, user, assistant)
        content: Message content
        
    Returns:
        Formatted message dictionary
    """
    valid_roles = ["system", "user", "assistant"]
    if role not in valid_roles:
        raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")
    
    return {
        "role": role,
        "content": content
    }

def create_rag_prompt(
    query: str,
    context: Union[str, List[str]],
    include_metadata: bool = False
) -> str:
    """
    Create a prompt for RAG (Retrieval Augmented Generation).
    
    Args:
        query: User query
        context: Retrieved context as string or list of strings
        include_metadata: Whether to include metadata about sources
        
    Returns:
        Formatted RAG prompt
    """
    # Format context
    if isinstance(context, list):
        formatted_context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
    else:
        formatted_context = context
    
    # Create prompt
    if include_metadata:
        prompt = f"""CONTEXT INFORMATION:
{formatted_context}

USER QUERY:
{query}

Based on the context information provided above, please answer the user query. If the answer cannot be determined from the context, say so."""
    else:
        prompt = f"""CONTEXT INFORMATION:
{formatted_context}

Using only the context information provided above, answer the following question:
{query}

If the answer cannot be determined from the context, say so."""
    
    return prompt

def create_rag_chat_messages(
    query: str,
    context: Union[str, List[str]],
    system_prompt: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Create formatted chat messages for RAG.
    
    Args:
        query: User query
        context: Retrieved context
        system_prompt: System prompt to use
        chat_history: Optional previous chat history
        
    Returns:
        List of formatted chat messages
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append(create_chat_message("system", system_prompt))
    
    # Add chat history if provided
    if chat_history:
        messages.extend(chat_history)
    
    # Format context
    if isinstance(context, list):
        context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
    else:
        context_str = context
    
    # Add context message
    context_message = create_chat_message(
        "user",
        f"Here is some relevant information to help answer the next question:\n\n{context_str}"
    )
    messages.append(context_message)
    
    # Add user query
    query_message = create_chat_message("user", query)
    messages.append(query_message)
    
    return messages

def create_voice_optimized_prompt(
    query: str, 
    conversation_context: Optional[str] = None
) -> str:
    """
    Create a prompt optimized for voice conversations.
    
    Args:
        query: Transcribed user query
        conversation_context: Optional context from the conversation
        
    Returns:
        Voice-optimized prompt
    """
    if conversation_context:
        prompt = f"""CONVERSATION CONTEXT:
{conversation_context}

USER QUERY (transcribed from voice):
{query}

Respond in a way that's natural to speak aloud. Keep responses concise and easy to understand when spoken."""
    else:
        prompt = f"""USER QUERY (transcribed from voice):
{query}

Respond in a way that's natural to speak aloud. Keep responses concise and easy to understand when spoken."""
    
    return prompt

def format_json_response(
    query: str,
    fields: List[str],
    example: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a prompt that requests a JSON-formatted response.
    
    Args:
        query: User query
        fields: List of fields to include in the JSON response
        example: Optional example of the expected format
        
    Returns:
        Prompt requesting JSON response
    """
    fields_str = ", ".join(fields)
    
    if example:
        example_json = json.dumps(example, indent=2)
        prompt = f"""USER QUERY:
{query}

Please provide your response in JSON format with the following fields: {fields_str}.

Example format:
{example_json}

Your JSON response:"""
    else:
        prompt = f"""USER QUERY:
{query}

Please provide your response in JSON format with the following fields: {fields_str}.

Your JSON response:"""
    
    return prompt