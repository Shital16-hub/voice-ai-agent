"""
TTS Integration module for Voice AI Agent.

This module provides classes and functions for integrating text-to-speech
capabilities with the Voice AI Agent system.
"""
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

from text_to_speech import DeepgramTTS, RealTimeResponseHandler, AudioProcessor

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration for Voice AI Agent.
    
    Provides an abstraction layer for TTS functionality, handling initialization,
    single-text processing, and streaming capabilities.
    """
    
    def __init__(
        self,
        voice: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice: Voice ID to use for Deepgram TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice = voice
        self.enable_caching = enable_caching
        self.tts_client = None
        self.tts_handler = None
        self.initialized = False
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the DeepgramTTS client
            self.tts_client = DeepgramTTS(voice=self.voice, enable_caching=self.enable_caching)
            
            # Initialize the RealTimeResponseHandler
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
            
            self.initialized = True
            logger.info(f"Initialized TTS with voice: {self.voice or 'default'}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            audio_data = await self.tts_client.synthesize(text)
            return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            raise
    
    async def text_to_speech_streaming(
        self, 
        text_generator: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech conversion.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Yields:
            Audio data chunks
        """
        if not self.initialized:
            await self.init()
        
        try:
            async for audio_chunk in self.tts_client.synthesize_streaming(text_generator):
                yield audio_chunk
        except Exception as e:
            logger.error(f"Error in streaming text to speech: {e}")
            raise
    
    async def process_realtime_text(
        self,
        text_chunks: AsyncIterator[str],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process text chunks in real-time and generate speech.
        
        Args:
            text_chunks: Async iterator of text chunks
            audio_callback: Callback to handle audio data
            
        Returns:
            Statistics about the processing
        """
        if not self.initialized:
            await self.init()
        
        # Start measuring time
        start_time = time.time()
        
        # Reset the TTS handler for this new session
        if self.tts_handler:
            await self.tts_handler.stop()
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
        
        # Process each text chunk
        total_chunks = 0
        total_audio_bytes = 0
        
        try:
            async for chunk in text_chunks:
                if not chunk or not chunk.strip():
                    continue
                
                # Process the text chunk
                audio_data = await self.text_to_speech(chunk)
                
                # Track statistics
                total_chunks += 1
                total_audio_bytes += len(audio_data)
                
                # Send audio to callback
                await audio_callback(audio_data)
                
                # Log progress periodically
                if total_chunks % 10 == 0:
                    logger.debug(f"Processed {total_chunks} text chunks")
        
        except Exception as e:
            logger.error(f"Error processing realtime text: {e}")
            return {
                "error": str(e),
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "elapsed_time": time.time() - start_time
            }
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        
        return {
            "total_chunks": total_chunks,
            "total_audio_bytes": total_audio_bytes,
            "elapsed_time": elapsed_time,
            "avg_chunk_size": total_audio_bytes / total_chunks if total_chunks > 0 else 0
        }
    
    async def process_ssml(self, ssml: str) -> bytes:
        """
        Process SSML text and convert to speech.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            audio_data = await self.tts_client.synthesize_with_ssml(ssml)
            return audio_data
        except Exception as e:
            logger.error(f"Error in SSML processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_handler:
            try:
                await self.tts_handler.stop()
            except Exception as e:
                logger.error(f"Error during TTS cleanup: {e}")