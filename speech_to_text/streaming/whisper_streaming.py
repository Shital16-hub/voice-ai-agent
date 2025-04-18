"""
Streaming wrapper for Whisper.cpp using pywhispercpp.
"""

import time
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Awaitable, Any, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

from pywhispercpp.model import Model
from speech_to_text.streaming.chunker import AudioChunker, ChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float
    start_time: float
    end_time: float
    chunk_id: int

class StreamingWhisperASR:
    """
    Streaming speech recognition using Whisper.cpp via pywhispercpp.
    
    This class handles the real-time streaming of audio data,
    chunking, and recognition using the Whisper model.
    """
    
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        language: str = "en",
        n_threads: int = 4,
        chunk_size_ms: int = 1000,
        overlap_ms: int = 200,
        silence_threshold: float = 0.01,
        min_silence_ms: int = 500,
        max_chunk_size_ms: int = 30000,
        vad_enabled: bool = True,
        translate: bool = False,
    ):
        """
        Initialize StreamingWhisperASR.
        
        Args:
            model_path: Path to the Whisper model file
            sample_rate: Audio sample rate in Hz
            language: Language code for recognition
            n_threads: Number of CPU threads to use
            chunk_size_ms: Size of each audio chunk in milliseconds
            overlap_ms: Overlap between chunks in milliseconds
            silence_threshold: Threshold for silence detection
            min_silence_ms: Minimum silence duration for chunking
            max_chunk_size_ms: Maximum chunk size in milliseconds
            vad_enabled: Whether to use voice activity detection
            translate: Whether to translate non-English to English
        """
        self.sample_rate = sample_rate
        self.vad_enabled = vad_enabled
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Initialize the audio chunker
        self.chunker = AudioChunker(
            sample_rate=sample_rate,
            chunk_size_ms=chunk_size_ms,
            overlap_ms=overlap_ms,
            silence_threshold=silence_threshold,
            min_silence_ms=min_silence_ms,
            max_chunk_size_ms=max_chunk_size_ms,
        )

        # Initialize the Whisper model using pywhispercpp
        self.model = Model('base.en', print_realtime=False, print_progress=False)
        # self.model.params.n_threads = n_threads
        
        # Set language
        # if language != "auto":
        #     self.model.set_language(language)
        
        # Set translation if needed
        if translate:
            self.model.set_translate(True)
        
        # Tracking state
        self.is_streaming = False
        self.last_chunk_id = 0
        self.partial_text = ""
        self.streaming_start_time = 0
        self.stream_duration = 0
        
        logger.info(f"Initialized StreamingWhisperASR with model {model_path}")
    
    async def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk for streaming recognition.
        
        Args:
            audio_chunk: Audio data as numpy array
            callback: Optional async callback function for results
            
        Returns:
            StreamingTranscriptionResult or None if no result available
        """
        # Start streaming if not already started
        if not self.is_streaming:
            self.start_streaming()
        
        # Add the audio to the chunker
        has_chunk = self.chunker.add_audio(audio_chunk)
        
        if not has_chunk:
            return None
        
        # Get the next chunk for processing
        chunk = self.chunker.get_chunk()
        if chunk is None:
            return None
        
        # Process the chunk
        result = await self._process_chunk(chunk, callback)
        return result
    
    async def _process_chunk(
        self,
        chunk: np.ndarray,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a single audio chunk.
        
        Args:
            chunk: Audio chunk as numpy array
            callback: Optional async callback for results
            
        Returns:
            StreamingTranscriptionResult or None
        """
        # Generate chunk metadata
        self.last_chunk_id += 1
        chunk_id = self.last_chunk_id
        
        start_sample = self.chunker.processed_samples
        end_sample = start_sample + len(chunk)
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=self.sample_rate,
            is_first_chunk=(chunk_id == 1),
        )
        
        # Perform voice activity detection if enabled
        contains_speech = True
        if self.vad_enabled:
            contains_speech = self._detect_speech(chunk)
        
        # If no speech detected, skip transcription
        if not contains_speech:
            logger.debug(f"No speech detected in chunk {chunk_id}, skipping transcription")
            result = StreamingTranscriptionResult(
                text="",
                is_final=True,
                confidence=0.0,
                start_time=metadata.start_time,
                end_time=metadata.end_time,
                chunk_id=chunk_id,
            )
            
            if callback:
                await callback(result)
            
            return result
        
        # Process the chunk with Whisper
        start_time = time.time()
        
        # Run in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            self.executor, 
            lambda: self.model.transcribe(chunk)
        )
        
        processing_time = time.time() - start_time
        
        logger.debug(f"Processed chunk {chunk_id} in {processing_time:.3f}s")
        
        # Handle transcription results
        if not segments:
            return None
        
        # Combine results from all segments
        combined_text = " ".join(segment.text for segment in segments)
        
        # Create streaming result
        result = StreamingTranscriptionResult(
            text=combined_text,
            is_final=True,  # For now, all results are final
            confidence=1.0,  # pywhispercpp doesn't provide confidence values
            start_time=metadata.start_time,
            end_time=metadata.end_time,
            chunk_id=chunk_id,
        )
        
        # Update partial text
        self.partial_text += " " + combined_text if combined_text else ""
        
        # Call the callback if provided
        if callback and combined_text.strip():
            await callback(result)
        
        return result if combined_text.strip() else None
    
    def _detect_speech(self, audio: np.ndarray, threshold: float = 0.3) -> bool:
        """
        Simple voice activity detection.
        
        Args:
            audio: Audio data
            threshold: Energy threshold for speech detection
            
        Returns:
            True if speech is detected, False otherwise
        """
        # Simple energy-based VAD
        energy = np.abs(audio).mean()
        return energy > threshold
    
    def start_streaming(self):
        """Start a new streaming session."""
        self.is_streaming = True
        self.last_chunk_id = 0
        self.partial_text = ""
        self.chunker.reset()
        self.streaming_start_time = time.time()
        logger.info("Started new streaming session")
    
    async def stop_streaming(self) -> Tuple[str, float]:
        """
        Stop the current streaming session.
        
        Returns:
            Tuple of (final_text, stream_duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        # Process any remaining audio
        chunk = self.chunker.get_final_chunk()
        final_text = self.partial_text
        
        if chunk is not None and len(chunk) > 0:
            result = await self._process_chunk(chunk)
            if result:
                final_text += " " + result.text
        
        self.stream_duration = time.time() - self.streaming_start_time
        self.is_streaming = False
        
        logger.info(f"Stopped streaming session after {self.stream_duration:.2f}s")
        return final_text.strip(), self.stream_duration

    def __del__(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)