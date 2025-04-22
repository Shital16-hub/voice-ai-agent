"""
Enhanced Voice AI Agent module with complete TTS integration.
Combines speech-to-text, RAG, LLM, and text-to-speech for end-to-end voice interaction.
"""
import asyncio
import logging
import time
import os
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Tuple, Callable, Awaitable
from dotenv import load_dotenv

import numpy as np

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult
from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.llm_setup import setup_global_llm

# Import the TTS modules
from text_to_speech import DeepgramTTS, RealTimeResponseHandler, AudioProcessor

# Load environment variables for API keys
load_dotenv()

logger = logging.getLogger(__name__)

class VoiceAIAgentWithTTS:
    """
    Enhanced Voice AI Agent with complete text-to-speech integration.
    Provides end-to-end voice interaction capabilities.
    """
    
    def __init__(
        self,
        storage_dir: str = "./storage",
        model_name: Optional[str] = None,
        whisper_model_path: str = "base.en",
        language: str = "en",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1024,
        tts_voice: Optional[str] = None,
        use_langgraph: bool = False
    ):
        """
        Initialize the Voice AI Agent with TTS.
        
        Args:
            storage_dir: Storage directory for the knowledge base
            model_name: LLM model name to use
            whisper_model_path: Path to Whisper model
            language: Language code
            llm_temperature: Temperature for sampling
            llm_max_tokens: Maximum tokens to generate
            tts_voice: Voice for TTS (uses Deepgram default if None)
            use_langgraph: Whether to use LangGraph (for future implementation)
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.use_langgraph = use_langgraph
        self.tts_voice = tts_voice
        
        # Initialize components as None first
        self.index_manager = None
        self.query_engine = None
        self.conversation_manager = None
        self.speech_recognizer = None
        self.tts_client = None
        self.tts_handler = None
        
        # Speech recognition settings
        self.whisper_model_path = whisper_model_path
        self.language = language
        
        # State for LangGraph (will be expanded in future)
        self.agent_state = {
            "mode": "default",
            "components": {
                "stt": {"initialized": False},
                "rag": {"initialized": False},
                "llm": {"initialized": False},
                "tts": {"initialized": False}
            },
            "session_id": None,
            "metadata": {}
        }
        
        logger.info(f"Initialized Enhanced VoiceAIAgent with model: {model_name}, TTS voice: {tts_voice}, LangGraph: {use_langgraph}")
    
    async def init(self):
        """Initialize all components."""
        logger.info("Initializing VoiceAIAgent components...")
        
        # Set up global LLM
        self.agent_state["components"]["llm"]["initializing"] = True
        setup_global_llm(
            model_name=self.model_name,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens
        )
        self.agent_state["components"]["llm"]["initialized"] = True
        
        # Initialize index manager
        self.agent_state["components"]["rag"]["initializing"] = True
        self.index_manager = IndexManager(storage_dir=self.storage_dir)
        await self.index_manager.init()
        
        # Initialize query engine
        self.query_engine = QueryEngine(
            index_manager=self.index_manager,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            llm_max_tokens=self.llm_max_tokens
        )
        await self.query_engine.init()
        self.agent_state["components"]["rag"]["initialized"] = True
        
        # Initialize conversation manager with skip_greeting=True for voice interactions
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            use_langgraph=self.use_langgraph,
            skip_greeting=True  # Always skip greeting for voice interactions
        )
        await self.conversation_manager.init()
        
        # Store session ID
        self.agent_state["session_id"] = self.conversation_manager.session_id
        
        # Initialize speech recognizer if needed
        if self.whisper_model_path:
            await self.init_speech_recognition()
        
        # Initialize text-to-speech components
        await self.init_text_to_speech()
        
        logger.info("VoiceAIAgent initialization complete")
    
    async def init_speech_recognition(self):
        """Initialize speech recognition component if not already initialized."""
        if not self.speech_recognizer and self.whisper_model_path:
            try:
                self.agent_state["components"]["stt"]["initializing"] = True
                self.speech_recognizer = StreamingWhisperASR(
                    model_path=self.whisper_model_path,
                    language=self.language,
                    n_threads=4,
                    chunk_size_ms=2000,
                    vad_enabled=True,
                    single_segment=True,
                    temperature=0.0  # Start with 0 temperature, will adjust as needed
                )
                self.agent_state["components"]["stt"]["initialized"] = True
                logger.info(f"Initialized speech recognizer with model: {self.whisper_model_path}")
            except Exception as e:
                logger.error(f"Error initializing speech recognizer: {e}")
                self.agent_state["components"]["stt"]["error"] = str(e)
    
    async def init_text_to_speech(self):
        """Initialize text-to-speech components."""
        try:
            self.agent_state["components"]["tts"]["initializing"] = True
            
            # Initialize the DeepgramTTS client
            self.tts_client = DeepgramTTS(voice=self.tts_voice)
            
            # Initialize the RealTimeResponseHandler
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
            
            self.agent_state["components"]["tts"]["initialized"] = True
            logger.info(f"Initialized TTS with voice: {self.tts_voice or 'default'}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            self.agent_state["components"]["tts"]["error"] = str(e)
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input directly.
        
        Args:
            text: Text input
            
        Returns:
            Response dictionary
        """
        if not self.conversation_manager:
            await self.init()
        
        # Ensure we're in the correct state for queries
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # For future LangGraph implementation
        if self.use_langgraph:
            return await self._process_text_langgraph(text)
        
        # Process with current implementation
        try:
            response = await self.conversation_manager.handle_user_input(text)
            return response
        except Exception as e:
            logger.error(f"Error in process_text: {e}", exc_info=True)
            return {
                "response": "I'm sorry, I encountered an error processing your request.",
                "state": ConversationState.WAITING_FOR_QUERY,
                "requires_human": False,
                "context": None
            }
    
    async def _process_text_langgraph(self, text: str) -> Dict[str, Any]:
        """
        LangGraph implementation of text processing (placeholder for future).
        
        Args:
            text: Input text
            
        Returns:
            Response dictionary
        """
        # This will be implemented in the future LangGraph integration
        logger.info("LangGraph integration will be implemented in a future update")
        
        # For now, fall back to standard implementation
        response = await self.conversation_manager.handle_user_input(text)
        return response
    
    async def process_text_streaming(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Process text input with streaming response.
        
        Args:
            text: Text input
            
        Returns:
            Async iterator of response chunks
        """
        if not self.conversation_manager:
            await self.init()
        
        # Ensure we're in the correct state for queries
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # For future LangGraph implementation
        if self.use_langgraph:
            async for chunk in self._process_text_streaming_langgraph(text):
                yield chunk
            return
        
        # Process with current implementation
        try:
            async for chunk in self.conversation_manager.generate_streaming_response(text):
                yield chunk
        except Exception as e:
            logger.error(f"Error in process_text_streaming: {e}", exc_info=True)
            yield {
                "chunk": "I'm sorry, I encountered an error processing your request.",
                "done": True,
                "requires_human": False,
                "state": ConversationState.WAITING_FOR_QUERY
            }
    
    async def _process_text_streaming_langgraph(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        LangGraph implementation of streaming text processing (placeholder for future).
        
        Args:
            text: Input text
            
        Returns:
            Async iterator of response chunks
        """
        # This will be implemented in the future LangGraph integration
        logger.info("LangGraph streaming integration will be implemented in a future update")
        
        # For now, fall back to standard implementation
        async for chunk in self.conversation_manager.generate_streaming_response(text):
            yield chunk
    
    async def process_audio(
        self,
        audio_chunk: Union[bytes, np.ndarray, List[float]],
        callback: Optional[callable] = None,
        is_short_audio: bool = False
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio input for speech recognition.
        
        Args:
            audio_chunk: Audio data as bytes or numpy array
            callback: Optional callback for interim results
            is_short_audio: Flag to indicate short audio for optimized handling
            
        Returns:
            Recognition result or None for interim results
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
        
        if not self.speech_recognizer:
            logger.error("Speech recognizer could not be initialized")
            return None
        
        # Convert to numpy array if needed
        if isinstance(audio_chunk, bytes):
            # Convert bytes to float array (implementation depends on your audio format)
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        elif isinstance(audio_chunk, list):
            audio_data = np.array(audio_chunk, dtype=np.float32)
        else:
            audio_data = audio_chunk
        
        # Auto-detect short audio if not specified
        if not is_short_audio and len(audio_data) < 5 * 16000:  # Less than 5 seconds at 16kHz
            is_short_audio = True
            logger.debug(f"Auto-detected short audio: {len(audio_data)/16000:.2f}s")
        
        # For short audio files, use simplified processing to avoid parameter issues
        if is_short_audio:
            # Disable VAD for short audio (don't change model parameters)
            original_vad = self.speech_recognizer.vad_enabled
            self.speech_recognizer.vad_enabled = False
            
            try:
                # Process with speech recognizer using basic mode
                result = await self.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_data,
                    callback=callback
                )
                return result
            finally:
                # Restore original VAD setting
                self.speech_recognizer.vad_enabled = original_vad
        else:
            # Standard processing for normal-length audio
            result = await self.speech_recognizer.process_audio_chunk(
                audio_chunk=audio_data,
                callback=callback
            )
            return result
    
    async def end_audio_stream(self) -> Tuple[str, float]:
        """
        End the audio stream and get final transcription.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.speech_recognizer:
            return "", 0.0
        
        return await self.speech_recognizer.stop_streaming()
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Deepgram TTS.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.tts_client:
            await self.init_text_to_speech()
        
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
        if not self.tts_client:
            await self.init_text_to_speech()
        
        try:
            async for audio_chunk in self.tts_client.synthesize_streaming(text_generator):
                yield audio_chunk
        except Exception as e:
            logger.error(f"Error in streaming text to speech: {e}")
            raise
    
    async def process_speech_to_response(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        stream_response: bool = True,
        is_short_audio: bool = False
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Process speech input and generate response in a single call.
        
        Args:
            audio_chunk: Audio data
            stream_response: Whether to stream the response
            is_short_audio: Flag to indicate short audio for optimized handling
            
        Returns:
            Response or async iterator of response chunks
        """
        # For future LangGraph implementation
        if self.use_langgraph:
            return await self._process_speech_to_response_langgraph(audio_chunk, stream_response)
        
        # Process audio for transcription
        transcription_result = await self.process_audio(audio_chunk, is_short_audio=is_short_audio)
        
        if not transcription_result or not transcription_result.text.strip():
            return {"error": "No speech detected or transcription failed"}
        
        # Get the transcribed text
        text = transcription_result.text
        
        # Log the transcription
        logger.info(f"Transcription: {text}")
        
        # Ensure we're in WAITING_FOR_QUERY state
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            logger.info("Forcing conversation state to WAITING_FOR_QUERY for voice input")
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Generate response based on streaming preference
        if stream_response:
            return self.process_text_streaming(text)
        else:
            return await self.process_text(text)
    
    async def _process_speech_to_response_langgraph(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        stream_response: bool = True
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        LangGraph implementation of speech to response processing (placeholder for future).
        
        Args:
            audio_chunk: Audio data
            stream_response: Whether to stream the response
            
        Returns:
            Response or async iterator of response chunks
        """
        # This will be implemented in the future LangGraph integration
        logger.info("LangGraph speech-to-response integration will be implemented in a future update")
        
        # For now, fall back to standard implementation
        transcription_result = await self.process_audio(audio_chunk)
        
        if not transcription_result or not transcription_result.text.strip():
            return {"error": "No speech detected or transcription failed"}
        
        # Get the transcribed text
        text = transcription_result.text
        logger.info(f"Transcription: {text}")
        
        # Generate response based on streaming preference
        if stream_response:
            return self.process_text_streaming(text)
        else:
            return await self.process_text(text)
    
    async def end_to_end_pipeline(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline: STT -> KB -> TTS.
        
        This is a convenience method for testing the entire pipeline.
        
        Args:
            audio_file_path: Path to input audio file
            output_speech_file: Path to save output speech file (optional)
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info(f"Starting end-to-end pipeline with audio: {audio_file_path}")
        
        # Make sure all components are initialized
        if not self.speech_recognizer or not self.query_engine or not self.tts_client:
            await self.init()
        
        # Track timing for each stage
        timings = {}
        start_time = time.time()
        
        # STAGE 1: Speech-to-Text - Use the proven method that works
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Use the existing process_full_audio_file method which is known to work
        stt_result = await self.process_full_audio_file(audio_file_path)
        
        # Check if there was an error in STT
        if "error" in stt_result:
            return stt_result
        
        # Extract the transcription from the result
        transcription = stt_result.get("transcription", "")
        if not transcription.strip():
            return {"error": "No transcription detected"}
        
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query - Use the response from process_full_audio_file
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        # Extract the response that was already generated
        response = stt_result.get("response", "")
        if not response:
            return {"error": "No response generated from knowledge base"}
        
        timings["kb"] = time.time() - kb_start
        logger.info(f"Response generated: {response[:50]}...")
        
        # STAGE 3: Text-to-Speech
        logger.info("STAGE 3: Text-to-Speech")
        tts_start = time.time()
        
        try:
            # Convert response to speech
            speech_audio = await self.text_to_speech(response)
            
            # Save speech audio if output file specified
            if output_speech_file:
                with open(output_speech_file, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {output_speech_file}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response
            }
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"End-to-end pipeline completed in {total_time:.2f}s")
        
        # Compile results
        return {
            "transcription": transcription,
            "response": response,
            "speech_audio_size": len(speech_audio),
            "speech_audio": None if output_speech_file else speech_audio,
            "timings": timings,
            "total_time": total_time
        }
    
    async def process_full_audio_file(
        self,
        audio_file_path: str,
        chunk_size_ms: int = 1000,
        simulate_realtime: bool = False
    ) -> Dict[str, Any]:
        """
        Process an entire audio file and generate a response.
        
        Args:
            audio_file_path: Path to audio file
            chunk_size_ms: Size of chunks in milliseconds (ignored for short audio)
            simulate_realtime: Whether to simulate real-time processing
            
        Returns:
            Dictionary with transcription and response
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Reset conversation state to ensure we start fresh for this audio file
        if self.conversation_manager:
            self.conversation_manager.reset()
            # Force to WAITING_FOR_QUERY state to ensure proper handling
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {"error": f"Error loading audio file: {e}"}
        
        # Determine if this is a short audio file
        audio_duration = len(audio) / sample_rate
        is_short_audio = audio_duration < 5.0  # Less than 5 seconds
        
        if is_short_audio:
            logger.info(f"Detected short audio file: {audio_duration:.2f}s - using optimized processing")
            return await self._process_short_audio_file(audio)
        else:
            logger.info(f"Processing normal-length audio file: {audio_duration:.2f}s")
            return await self._process_normal_audio_file(audio, sample_rate, chunk_size_ms, simulate_realtime)
    
    async def process_audio_to_speech_stream(
        self,
        audio_file_path: str,
        tts_callback: Callable[[bytes], Awaitable[None]],
        chunk_size_ms: int = 1000
    ) -> Dict[str, Any]:
        """
        Process audio and stream the response directly to speech with real-time streaming.
        
        Args:
            audio_file_path: Path to audio file
            tts_callback: Async callback that will receive speech audio chunks
            chunk_size_ms: Size of chunks in milliseconds (ignored for short audio)
            
        Returns:
            Dictionary with stats about the process
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
            
        if not self.tts_client:
            await self.init_text_to_speech()
            
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Start measuring total time
        start_time = time.time()
        
        # Reset conversation state to ensure we start fresh for this audio file
        if self.conversation_manager:
            self.conversation_manager.reset()
            # Force to WAITING_FOR_QUERY state to ensure proper handling
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {"error": f"Error loading audio file: {e}"}
        
        # Get transcription
        logger.info("Starting transcription...")
        self.speech_recognizer.start_streaming()
        await self.speech_recognizer.process_audio_chunk(audio)
        transcription, duration = await self.speech_recognizer.stop_streaming()
        
        if not transcription.strip():
            return {"error": "No transcription detected"}
        
        # Log transcription time
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f}s: {transcription}")
        
        # Start retrieving context for the query
        logger.info("Retrieving relevant context...")
        retrieval_start = time.time()
        retrieval_results = await self.query_engine.retrieve_with_sources(transcription)
        context = self.query_engine.format_retrieved_context(retrieval_results.get("results", []))
        retrieval_time = time.time() - retrieval_start
        
        # Begin streaming response generation and TTS simultaneously
        total_text_chunks = 0
        total_audio_chunks = 0
        stream_start_time = time.time()
        full_response = ""
        
        logger.info("Starting streaming response generation with TTS...")
        
        # Reset the TTS handler for this new session
        if self.tts_handler:
            await self.tts_handler.stop()
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
        
        # Use the query engine's streaming method for direct access to knowledge base
        async for chunk in self.query_engine.query_with_streaming(transcription, None):
            chunk_text = chunk.get("chunk", "")
            
            if chunk_text:
                # Add to full response
                full_response += chunk_text
                total_text_chunks += 1
                
                # Process the text through TTS
                try:
                    # Convert text to speech
                    audio_data = await self.tts_client.synthesize(chunk_text)
                    
                    # Send the audio to the callback
                    await tts_callback(audio_data)
                    total_audio_chunks += 1
                    
                    # Optionally log chunks for debugging
                    if total_audio_chunks % 5 == 0:
                        logger.debug(f"Sent {total_audio_chunks} audio chunks")
                        
                except Exception as e:
                    logger.error(f"Error processing TTS for chunk: {e}")
        
        # Calculate stats
        stream_time = time.time() - stream_start_time
        total_time = time.time() - start_time
        
        logger.info(f"Streaming response with TTS completed in {stream_time:.2f}s")
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return {
            "transcription": transcription,
            "transcription_time": transcription_time,
            "retrieval_time": retrieval_time,
            "response_stream_time": stream_time,
            "total_time": total_time,
            "total_text_chunks": total_text_chunks,
            "total_audio_chunks": total_audio_chunks,
            "full_response": full_response
        }
    
    async def process_full_audio_file_with_tts(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None,
        chunk_size_ms: int = 1000,
        simulate_realtime: bool = False
    ) -> Dict[str, Any]:
        """
        Process an entire audio file, generate a response, and convert to speech.
        
        Args:
            audio_file_path: Path to input audio file
            output_speech_file: Path to save output speech file (optional)
            chunk_size_ms: Size of chunks in milliseconds (ignored for short audio)
            simulate_realtime: Whether to simulate real-time processing
            
        Returns:
            Dictionary with transcription, response, and output speech info
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
        
        if not self.tts_client:
            await self.init_text_to_speech()
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Start measuring total time
        start_time = time.time()
        
        # Reset conversation state to ensure we start fresh for this audio file
        if self.conversation_manager:
            self.conversation_manager.reset()
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {"error": f"Error loading audio file: {e}"}
        
        # Determine if this is a short audio file
        audio_duration = len(audio) / sample_rate
        is_short_audio = audio_duration < 5.0  # Less than 5 seconds
        
        # Process for transcription and response
        if is_short_audio:
            logger.info(f"Processing short audio file: {audio_duration:.2f}s")
            result = await self._process_short_audio_file(audio)
        else:
            logger.info(f"Processing normal-length audio file: {audio_duration:.2f}s")
            result = await self._process_normal_audio_file(audio, sample_rate, chunk_size_ms, simulate_realtime)
        
        # Check if we have a response to convert to speech
        if "response" in result and result["response"]:
            response_text = result["response"]
            
            # Convert response to speech
            logger.info(f"Converting response to speech: {response_text[:50]}...")
            try:
                tts_start_time = time.time()
                speech_audio = await self.text_to_speech(response_text)
                tts_time = time.time() - tts_start_time
                
                # Save speech audio if output file specified
                if output_speech_file:
                    with open(output_speech_file, "wb") as f:
                        f.write(speech_audio)
                    logger.info(f"Saved speech audio to {output_speech_file}")
                
                # Add speech info to result
                result["speech_audio_size"] = len(speech_audio)
                result["tts_time"] = tts_time
                result["speech_audio"] = speech_audio if not output_speech_file else None
                
            except Exception as e:
                logger.error(f"Error converting response to speech: {e}")
                result["speech_error"] = str(e)
        
        # Add total processing time
        result["total_processing_time"] = time.time() - start_time
        
        return result
    
    async def _process_short_audio_file(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Process a short audio file with optimized parameters.
        This method uses a simple approach that avoids parameter issues with pywhispercpp.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary with results
        """
        # Save original settings
        original_vad = self.speech_recognizer.vad_enabled
        
        # Use simple approach for short audio
        self.speech_recognizer.vad_enabled = False  # Disable VAD for short audio
        
        # Try multiple approaches to get transcription
        transcription = ""
        duration = 0
        
        # First attempt with default temperature
        try:
            self.speech_recognizer.start_streaming()
            await self.speech_recognizer.process_audio_chunk(audio)
            transcription, duration = await self.speech_recognizer.stop_streaming()
        except Exception as e:
            logger.warning(f"First transcription attempt failed: {e}")
        
        # If first attempt failed, try with higher temperature
        if not transcription or transcription.strip() == "":
            try:
                logger.info("First attempt yielded no transcription, trying with higher temperature")
                self.speech_recognizer.start_streaming()
                # Directly process audio without changing parameters
                await self.speech_recognizer.process_audio_chunk(audio)
                transcription, duration = await self.speech_recognizer.stop_streaming()
            except Exception as e:
                logger.warning(f"Second transcription attempt failed: {e}")
        
        # Restore original settings
        self.speech_recognizer.vad_enabled = original_vad
        
        # If we got a transcription, generate a response
        if transcription and transcription.strip():
            try:
                # Make sure we log the RAG process
                logger.info("-----------------------------------------------------------")
                logger.info("RAG (Retrieval Augmented Generation) VERIFICATION:")
                logger.info("-----------------------------------------------------------")
                logger.info(f"Retrieving documents for: '{transcription}'")
                
                # Get documents directly from query engine to verify RAG is working
                retrieval_results = await self.query_engine.retrieve_with_sources(transcription)
                results = retrieval_results.get("results", [])
                
                # Print retrieved documents for verification
                logger.info(f"Retrieved {len(results)} documents from knowledge base")
                for i, doc in enumerate(results):
                    metadata = doc.get("metadata", {})
                    source = metadata.get("source", f"Source {i+1}")
                    text = doc.get("text", "")
                    if len(text) > 100:
                        text = text[:97] + "..."
                    logger.info(f"Document {i+1}: Source={source}")
                    logger.info(f"Content: {text}")
                
                # Format context for verification
                context = self.query_engine.format_retrieved_context(results)
                logger.info(f"Formatted context ({len(context)} characters) created for LLM")
                
                # Log LLM verification
                logger.info("-----------------------------------------------------------")
                logger.info("LLM (Language Learning Model) VERIFICATION:")
                logger.info("-----------------------------------------------------------")
                logger.info(f"Using model: {self.model_name} with temperature: {self.llm_temperature}")
                
                # Get direct response from query engine to verify it's working
                direct_result = await self.query_engine.query(transcription)
                direct_response = direct_result.get("response", "")
                logger.info(f"Direct LLM response: {direct_response[:50]}...")
                logger.info("-----------------------------------------------------------")
                
                # Use the DIRECT result as the response, bypassing the conversation manager
                result = {
                    "transcription": transcription,
                    "duration": duration,
                    "response": direct_response
                }
            except Exception as e:
                logger.error(f"Error generating response: {e}", exc_info=True)
                result = {
                    "transcription": transcription,
                    "duration": duration,
                    "error": f"Error generating response: {e}"
                }
        else:
            result = {
                "transcription": "",
                "duration": duration,
                "error": "No valid transcription detected"
            }
        
        return result
    
    async def _process_normal_audio_file(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        chunk_size_ms: int, 
        simulate_realtime: bool
    ) -> Dict[str, Any]:
        """
        Process a normal-length audio file using chunking.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            chunk_size_ms: Chunk size in milliseconds
            simulate_realtime: Whether to simulate real-time processing
            
        Returns:
            Dictionary with results
        """
        # Calculate chunk size in samples
        chunk_size = int(sample_rate * chunk_size_ms / 1000)
        
        # Split audio into chunks
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        
        # Storage for transcriptions
        transcriptions = []
        
        # Process each chunk
        async def transcription_callback(result: StreamingTranscriptionResult):
            if result.text.strip():
                transcriptions.append(result.text)
                logger.info(f"Interim transcription: {result.text}")
        
        # Start streaming
        self.speech_recognizer.start_streaming()
        
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            # Process chunk
            await self.speech_recognizer.process_audio_chunk(
                audio_chunk=chunk,
                callback=transcription_callback
            )
            
            # Simulate real-time processing if requested
            if simulate_realtime and i < num_chunks - 1:
                await asyncio.sleep(chunk_size_ms / 1000)
        
        # Get final transcription
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # Generate response
        if final_text.strip():
            # Ensure conversation manager is in correct state
            if self.conversation_manager:
                self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
                
            # Get RAG verification info
            logger.info("-----------------------------------------------------------")
            logger.info("RAG (Retrieval Augmented Generation) VERIFICATION:")
            logger.info("-----------------------------------------------------------")
            logger.info(f"Retrieving documents for: '{final_text}'")
            
            # Show retrieved documents
            retrieval_results = await self.query_engine.retrieve_with_sources(final_text)
            results = retrieval_results.get("results", [])
            
            logger.info(f"Retrieved {len(results)} documents from knowledge base")
            for i, doc in enumerate(results):
                metadata = doc.get("metadata", {})
                source = metadata.get("source", f"Source {i+1}")
                text = doc.get("text", "")
                if len(text) > 100:
                    text = text[:97] + "..."
                logger.info(f"Document {i+1}: Source={source}")
                logger.info(f"Content: {text}")
            
            # Format context
            context = self.query_engine.format_retrieved_context(results)
            logger.info(f"Formatted context ({len(context)} characters) created for LLM")
            
            # LLM verification
            logger.info("-----------------------------------------------------------")
            logger.info("LLM (Language Learning Model) VERIFICATION:")
            logger.info("-----------------------------------------------------------")
            logger.info(f"Using model: {self.model_name} with temperature: {self.llm_temperature}")
            
            # Get direct LLM response to verify
            direct_result = await self.query_engine.query(final_text)
            direct_response = direct_result.get("response", "")
            logger.info(f"Direct LLM response: {direct_response[:50]}...")
            logger.info("-----------------------------------------------------------")
            
            # Use the DIRECT result as the response, bypassing the conversation manager
            result = {
                "transcription": final_text,
                "duration": duration,
                "response": direct_response
            }
        else:
            result = {
                "transcription": "",
                "duration": duration,
                "error": "No valid transcription detected"
            }
        
        return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent.
        
        Returns:
            Dictionary with statistics
        """
        if not self.query_engine:
            await self.init()
        
        # Get basic stats from query engine
        kb_stats = await self.query_engine.get_stats()
        
        # Add agent-specific stats
        stats = {
            "model_name": self.model_name,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "speech_recognizer_model": self.whisper_model_path,
            "speech_recognizer_language": self.language,
            "tts_voice": self.tts_voice,
            "knowledge_base": kb_stats,
            "use_langgraph": self.use_langgraph,
            "components_state": self.agent_state["components"]
        }
        
        # Add conversation stats if available
        if self.conversation_manager and self.conversation_manager.history:
            conversation_stats = {
                "total_turns": len(self.conversation_manager.history),
                "current_state": self.conversation_manager.current_state
            }
            stats["conversation"] = conversation_stats
        
        return stats