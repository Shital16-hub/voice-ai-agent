"""
Ollama API client for language model inference.
"""
import json
import aiohttp
import requests
from typing import Dict, List, Optional, Any, AsyncIterator, Union
import logging
import asyncio

from language_model.config import get_model_config

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama API.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_host: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the model to use (defaults to config)
            api_host: Ollama API host (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
        """
        config = get_model_config(model_name)
        self.model = config["model"]
        self.api_host = api_host or config["api_host"]
        self.timeout = timeout or config["timeout"]
        self.default_params = config["params"]
        
        # Ensure API host has correct format
        if not self.api_host.startswith(("http://", "https://")):
            self.api_host = f"http://{self.api_host}"
        
        logger.info(f"Initialized OllamaClient with model: {self.model}")
    
    def _get_api_url(self, endpoint: str) -> str:
        """Get full API URL for endpoint."""
        return f"{self.api_host}/api/{endpoint}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        try:
            url = self._get_api_url("show")
            response = requests.post(
                url,
                json={"name": self.model},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences to stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary with generated text and metadata
        """
        try:
            # Build request parameters
            params = self.default_params.copy()
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["num_predict"] = max_tokens
            if stop_sequences:
                params["stop"] = stop_sequences
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Build request body
            request_body = {
                "model": self.model,
                "prompt": prompt,
                **params
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_body["system"] = system_prompt
            
            # Make API request
            url = self._get_api_url("generate")
            response = requests.post(
                url,
                json=request_body,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the chat format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences to stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary with generated text and metadata
        """
        try:
            # Build request parameters
            params = self.default_params.copy()
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["num_predict"] = max_tokens
            if stop_sequences:
                params["stop"] = stop_sequences
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Build request body
            request_body = {
                "model": self.model,
                "messages": messages,
                **params
            }
            
            # Make API request
            url = self._get_api_url("chat")
            response = requests.post(
                url,
                json=request_body,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream text generation from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences to stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Async iterator yielding response chunks
        """
        # Build request parameters
        params = self.default_params.copy()
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["num_predict"] = max_tokens
        if stop_sequences:
            params["stop"] = stop_sequences
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Build request body
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **params
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_body["system"] = system_prompt
        
        # Make streaming API request
        url = self._get_api_url("generate")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_body,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    
                    # Process streaming response
                    async for line in response.content:
                        if not line:
                            continue
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {line}")
                            continue
        
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}")
            raise
    
    async def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat responses from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences to stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Async iterator yielding response chunks
        """
        # Build request parameters
        params = self.default_params.copy()
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["num_predict"] = max_tokens
        if stop_sequences:
            params["stop"] = stop_sequences
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Build request body
        request_body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **params
        }
        
        # Make streaming API request
        url = self._get_api_url("chat")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_body,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    
                    # Process streaming response
                    async for line in response.content:
                        if not line:
                            continue
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {line}")
                            continue
        
        except Exception as e:
            logger.error(f"Error in generate_chat_stream: {e}")
            raise