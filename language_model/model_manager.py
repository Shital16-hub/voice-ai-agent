"""
Model manager for handling language model conversations.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator, Callable, Union
import time

from language_model.inference.ollama_client import OllamaClient
from language_model.inference.response_formatter import ResponseFormatter
from language_model.prompts.system_prompts import get_system_prompt
from language_model.prompts.templates import (
    format_chat_history, 
    create_chat_message, 
    create_rag_chat_messages
)
from language_model.utils.text_processing import clean_transcription, truncate_text

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manager for language model inference and conversation handling.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        scenario: str = "customer_service",
        with_rag: bool = False,
    ):
        """
        Initialize the model manager.
        
        Args:
            model_name: Name of the model to use
            system_prompt: Custom system prompt (if None, uses scenario)
            scenario: Conversation scenario
            with_rag: Whether to use RAG-optimized prompts
        """
        self.client = OllamaClient(model_name=model_name)
        self.formatter = ResponseFormatter()
        
        # Set system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = get_system_prompt(scenario, with_rag)
        
        # Initialize conversation history
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append(
                create_chat_message("system", self.system_prompt)
            )
        
        # Metadata
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_messages = 0
        
        logger.info(f"Initialized ModelManager with model: {self.client.model}")
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
        """
        message = create_chat_message(role, content)
        self.conversation_history.append(message)
        self.total_messages += 1
    
    def get_conversation_history(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            List of conversation messages
        """
        return format_chat_history(self.conversation_history, include_system)
    
    def clear_history(self, keep_system_prompt: bool = True):
        """
        Clear conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt
        """
        if keep_system_prompt and self.system_prompt:
            self.conversation_history = [
                create_chat_message("system", self.system_prompt)
            ]
        else:
            self.conversation_history = []
        
        self.total_messages = len(self.conversation_history)
    
    def process_user_input(self, text: str) -> str:
        """
        Process user text input from speech-to-text.
        
        Args:
            text: Transcribed user input
            
        Returns:
            Processed text
        """
        # Clean transcription
        cleaned = clean_transcription(text)
        
        # Truncate if too long
        truncated = truncate_text(cleaned, max_tokens=1000)
        
        return truncated
    
    async def generate_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response to user input.
        
        Args:
            user_input: User input text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Process user input
        processed_input = self.process_user_input(user_input)
        
        # Add user message to history
        self.add_message("user", processed_input)
        
        # Generate response
        start_time = time.time()
        response = self.client.generate_chat(
            messages=self.get_conversation_history(),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        generation_time = time.time() - start_time
        
        # Extract and clean response text
        response_text = self.formatter.extract_content(response)
        cleaned_response = self.formatter.clean_response(response_text)
        
        # Add to conversation history
        self.add_message("assistant", cleaned_response)
        
        # Extract metadata
        metadata = self.formatter.format_metadata(response)
        metadata["generation_time"] = generation_time
        
        # Update token counts
        if "total_tokens" in metadata:
            self.total_tokens += metadata["total_tokens"]
        
        return {
            "response": cleaned_response,
            "original_input": user_input,
            "processed_input": processed_input,
            "metadata": metadata
        }
    
    async def generate_streaming_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a streaming response to user input.
        
        Args:
            user_input: User input text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callback: Optional callback function for chunks
            **kwargs: Additional parameters
            
        Returns:
            Async iterator of response chunks
        """
        # Process user input
        processed_input = self.process_user_input(user_input)
        
        # Add user message to history
        self.add_message("user", processed_input)
        
        # Initialize response tracking
        full_response = ""
        start_time = time.time()
        chunk_count = 0
        
        # Stream response
        async for chunk in self.client.generate_chat_stream(
            messages=self.get_conversation_history(),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            # Extract text from chunk
            chunk_text = self.formatter.extract_streaming_content(chunk)
            
            # Update full response
            full_response += chunk_text
            chunk_count += 1
            
            # Create result dictionary
            result = {
                "chunk": chunk_text,
                "full_response": full_response,
                "chunk_count": chunk_count,
                "done": False,
                "metadata": {}
            }
            
            # Call callback if provided
            if callback:
                callback(chunk_text)
            
            yield result
        
        # Complete the streaming response with final metadata
        generation_time = time.time() - start_time
        
        # Clean up final response
        cleaned_response = self.formatter.clean_response(full_response)
        
        # Add to conversation history
        self.add_message("assistant", cleaned_response)
        
        # Final chunk with metadata
        final_result = {
            "chunk": "",
            "full_response": cleaned_response,
            "chunk_count": chunk_count,
            "done": True,
            "metadata": {
                "generation_time": generation_time,
                "chunks": chunk_count
            }
        }
        
        yield final_result
    
    async def generate_rag_response(
        self,
        user_input: str,
        context: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response with RAG (Retrieval-Augmented Generation).
        
        Args:
            user_input: User input text
            context: Retrieved context (string or list of strings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Process user input
        processed_input = self.process_user_input(user_input)
        
        # Create RAG messages
        rag_messages = create_rag_chat_messages(
            query=processed_input,
            context=context,
            system_prompt=self.system_prompt,
            chat_history=self.get_conversation_history(include_system=False)
        )
        
        # Generate response
        start_time = time.time()
        response = self.client.generate_chat(
            messages=rag_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        generation_time = time.time() - start_time
        
        # Extract and clean response text
        response_text = self.formatter.extract_content(response)
        cleaned_response = self.formatter.clean_response(response_text)
        
        # Add user message and assistant response to history
        self.add_message("user", processed_input)
        self.add_message("assistant", cleaned_response)
        
        # Extract metadata
        metadata = self.formatter.format_metadata(response)
        metadata["generation_time"] = generation_time
        
        # Update token counts
        if "total_tokens" in metadata:
            self.total_tokens += metadata["total_tokens"]
        
        return {
            "response": cleaned_response,
            "original_input": user_input,
            "processed_input": processed_input,
            "metadata": metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        elapsed_time = time.time() - self.start_time
        
        return {
            "model": self.client.model,
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "elapsed_time": elapsed_time,
            "tokens_per_second": self.total_tokens / elapsed_time if elapsed_time > 0 else 0
        }