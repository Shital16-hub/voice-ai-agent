"""
Formatter for model responses to ensure consistent output.
"""
from typing import Dict, Any, List, Optional, Union
import json
import re

class ResponseFormatter:
    """
    Format and process model responses.
    """
    
    @staticmethod
    def extract_content(response: Dict[str, Any]) -> str:
        """
        Extract content from model response.
        
        Args:
            response: Model response dictionary
            
        Returns:
            Text content
        """
        # Handle chat completions
        if "message" in response:
            return response.get("message", {}).get("content", "")
        
        # Handle regular completions
        return response.get("response", "")
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON as dict or None if no valid JSON found
        """
        # Match JSON pattern with or without code blocks
        json_pattern = r'```(?:json)?\s*({[\s\S]*?})```|({[\s\S]*?})'
        matches = re.findall(json_pattern, text)
        
        # Check each match
        for match in matches:
            # The match could be in either capture group
            json_str = match[0] if match[0] else match[1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON without regex if no matches found
        try:
            # Look for opening and closing braces
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return None
    
    @staticmethod
    def clean_response(text: str) -> str:
        """
        Clean up response text.
        
        Args:
            text: Raw model output
            
        Returns:
            Cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove markdown code blocks if they're wrapping the entire response
        if text.startswith("```") and text.endswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:  # Must have at least 3 lines for a valid code block
                # Check if it's a code block with a language specifier
                if not lines[0].startswith("```json") and not lines[0].startswith("```python"):
                    # Remove first and last line
                    text = "\n".join(lines[1:-1])
        
        # Remove AI self-references
        text = re.sub(r'As an AI (assistant|language model|agent),?\s*', '', text)
        
        return text
    
    @staticmethod
    def format_for_voice(text: str, max_length: Optional[int] = None) -> str:
        """
        Format response for voice output.
        
        Args:
            text: Response text
            max_length: Maximum length in characters (optional)
            
        Returns:
            Formatted text optimized for TTS
        """
        # Remove markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
        
        # Remove URLs but keep description
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Replace bullet points
        text = re.sub(r'^- ', '• ', text, flags=re.MULTILINE)
        
        # Simplify multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Limit length if specified
        if max_length and len(text) > max_length:
            # Try to cut at sentence boundary
            sentences = re.split(r'(?<=[.!?])\s+', text)
            truncated = ""
            
            for sentence in sentences:
                if len(truncated + sentence) <= max_length:
                    truncated += sentence + " "
                else:
                    break
            
            if truncated:
                text = truncated.strip()
            else:
                # If we can't find sentence boundary, just truncate
                text = text[:max_length] + "..."
        
        return text.strip()
    
    @staticmethod
    def create_ssml(text: str) -> str:
        """
        Convert text to SSML for text-to-speech.
        
        Args:
            text: Plain text response
            
        Returns:
            SSML formatted text
        """
        # Basic SSML wrapper
        ssml = f'<speak>{text}</speak>'
        
        # Add pauses after sentences
        ssml = re.sub(r'([.!?])\s+', r'\1<break time="500ms"/>', ssml)
        
        # Add emphasis for important parts (e.g., numbers, dates)
        ssml = re.sub(r'\b(\d+(?:[.,]\d+)?)\b', r'<emphasis>\1</emphasis>', ssml)
        
        # Convert lists to pauses
        ssml = re.sub(r'•\s+', r'<break time="300ms"/>• ', ssml)
        
        return ssml
    
    @staticmethod
    def extract_streaming_content(response_chunk: Dict[str, Any]) -> str:
        """
        Extract content from a streaming response chunk.
        
        Args:
            response_chunk: Streaming response chunk
            
        Returns:
            Text content from the chunk
        """
        # For chat API
        if "message" in response_chunk:
            content = response_chunk.get("message", {}).get("content", "")
            return content
        
        # For completion API
        if "response" in response_chunk:
            return response_chunk.get("response", "")
        
        # Fallback
        return ""
    
    @staticmethod
    def format_metadata(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format metadata from the model response.
        
        Args:
            response: Full model response dictionary
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Add model info
        if "model" in response:
            metadata["model"] = response["model"]
        
        # Add timing info
        if "eval_count" in response:
            metadata["eval_count"] = response["eval_count"]
        
        if "eval_duration" in response:
            metadata["eval_duration"] = response["eval_duration"]
            
            # Calculate tokens per second
            if "eval_count" in response and response["eval_duration"] > 0:
                metadata["tokens_per_second"] = response["eval_count"] / (response["eval_duration"] / 1000000000)
        
        # Add token counts
        if "prompt_eval_count" in response:
            metadata["prompt_tokens"] = response["prompt_eval_count"]
            
        if "eval_count" in response and "prompt_eval_count" in response:
            metadata["completion_tokens"] = response["eval_count"] - response["prompt_eval_count"]
            metadata["total_tokens"] = response["eval_count"]
        
        return metadata