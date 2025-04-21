"""
Ollama API client for language model inference.
"""
import json
import aiohttp
import requests
import re
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
    
    def _extract_content_from_text(self, text: str) -> str:
        """
        Extract content from potentially malformed JSON response.
        
        Args:
            text: Response text from Ollama
            
        Returns:
            Extracted content or fallback message
        """
        try:
            # Try to find content in JSON
            content_match = re.search(r'"content"\s*:\s*"([^"]*)"', text)
            if content_match:
                return content_match.group(1)
                
            # Try to find response in JSON
            response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
            if response_match:
                return response_match.group(1)
                
            # If we have a reasonable text chunk, return it
            if len(text) > 10 and "{" in text and "}" in text:
                # Try to extract anything between quotes
                text_match = re.search(r'"([^"]{10,})"', text)
                if text_match:
                    return text_match.group(1)
        except Exception as e:
            logger.error(f"Error extracting content from text: {e}")
        
        # If all extraction attempts fail, return a portion of the text
        if len(text) > 20:
            return "Response parsing error. Raw response fragment: " + text[:100].replace('"', '\'')
        
        return "Unable to parse response"
    
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
            
            # Try to parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.debug(f"Raw response: {response.text[:500]}...")
                
                # Extract content from response text
                content = self._extract_content_from_text(response.text)
                
                # Return a simplified response
                return {"response": content}
        
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
            
            # Log the request for debugging
            logger.debug(f"Chat request: {json.dumps(request_body, indent=2)}")
            
            # Make API request
            url = self._get_api_url("chat")
            response = requests.post(
                url,
                json=request_body,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Try to parse the response as JSON
            try:
                json_response = response.json()
                logger.debug(f"Chat response parsed successfully")
                return json_response
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.debug(f"Raw response: {response.text[:500]}...")
                
                # Extract content from response text
                content = self._extract_content_from_text(response.text)
                
                # Create a simplified response dictionary
                return {"response": content}
        
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise
    
    def generate_from_messages(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use generate API instead of chat API as a fallback.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        # Extract system prompt if present
        system_prompt = None
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        # Convert remaining messages to prompt
        prompt_parts = []
        for msg in messages:
            if msg["role"] != "system":
                role = msg["role"].capitalize()
                content = msg["content"]
                prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # Call generate with converted prompt
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
    
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
                            # Try to extract content
                            content = self._extract_content_from_text(line.decode('utf-8'))
                            yield {"response": content}
        
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
                            logger.warning(f"Failed to parse JSON in stream: {line}")
                            # Try to extract content
                            content = self._extract_content_from_text(line.decode('utf-8'))
                            yield {"response": content}
        
        except Exception as e:
            logger.error(f"Error in generate_chat_stream: {e}")
            raise