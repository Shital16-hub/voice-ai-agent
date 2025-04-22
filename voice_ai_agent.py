"""
Main Voice AI Agent module combining speech-to-text, RAG, and LLM capabilities.
Designed for future LangGraph integration with streaming response support.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Tuple, Callable, Awaitable

import numpy as np

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult
from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.llm_setup import setup_global_llm

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Unified Voice AI Agent integrating speech recognition, RAG, and LLM components.
    Designed with future LangGraph migration in mind.
    """
    
    def __init__(
        self,
        storage_dir: str = "./storage",
        model_name: Optional[str] = None,
        whisper_model_path: str = "base.en",
        language: str = "en",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1024,
        use_langgraph: bool = False  # Flag for future LangGraph implementation
    ):
        """
        Initialize the Voice AI Agent.
        
        Args:
            storage_dir: Storage directory for the knowledge base
            model_name: LLM model name to use
            whisper_model_path: Path to Whisper model
            language: Language code
            llm_temperature: Temperature for sampling
            llm_max_tokens: Maximum tokens to generate
            use_langgraph: Whether to use LangGraph (for future implementation)
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.use_langgraph = use_langgraph
        
        # Initialize components as None first
        self.index_manager = None
        self.query_engine = None
        self.conversation_manager = None
        self.speech_recognizer = None
        
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
        
        logger.info(f"Initialized VoiceAIAgent with model: {model_name}, LangGraph: {use_langgraph}")
    
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
        tts_callback: Callable[[str], Awaitable[None]],
        chunk_size_ms: int = 1000
    ) -> Dict[str, Any]:
        """
        Process audio and stream the response directly to text-to-speech with minimal latency.
        Optimized for real-time word-by-word output.
        
        Args:
            audio_file_path: Path to audio file
            tts_callback: Async callback that will receive text chunks for TTS
            chunk_size_ms: Size of chunks in milliseconds (ignored for short audio)
            
        Returns:
            Dictionary with stats about the process
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
            
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
        total_chunks = 0
        stream_start_time = time.time()
        full_response = ""
        
        logger.info("Starting streaming response generation...")
        
        # Use the query engine's streaming method for direct access to knowledge base
        async for chunk in self.query_engine.query_with_streaming(transcription, None):
            chunk_text = chunk.get("chunk", "")
            
            if chunk_text:
                # Add to full response
                full_response += chunk_text
                
                # Send chunk immediately to TTS without any batching or delay
                # This ensures minimum latency between generation and vocalization
                await tts_callback(chunk_text)
                total_chunks += 1
                
                # Optionally log chunks for debugging
                if total_chunks % 10 == 0:
                    logger.debug(f"Sent {total_chunks} text chunks to TTS")
        
        # Calculate stats
        stream_time = time.time() - stream_start_time
        total_time = time.time() - start_time
        
        logger.info(f"Streaming response completed in {stream_time:.2f}s")
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return {
            "transcription": transcription,
            "transcription_time": transcription_time,
            "retrieval_time": retrieval_time,
            "response_stream_time": stream_time,
            "total_time": total_time,
            "total_chunks": total_chunks,
            "full_response": full_response
        }
    
    async def process_realtime_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        tts_callback: Callable[[str], Awaitable[None]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process real-time audio stream and generate streaming responses for TTS.
        Optimized for word-by-word output to TTS with minimal latency.
        
        This method supports true real-time continuous conversation.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
            tts_callback: Async callback that will receive text chunks for TTS
            
        Returns:
            Async iterator with info about each processing cycle
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
        
        # Reset for new conversation
        if self.conversation_manager:
            self.conversation_manager.reset()
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Prepare the audio chunker
        self.speech_recognizer.start_streaming()
        
        # Track state
        is_speaking = False
        silence_frames = 0
        max_silence_frames = 30  # Adjust based on your frame rate
        
        # Process incoming audio stream
        try:
            async for audio_chunk in audio_stream:
                # Process the audio chunk
                result = await self.speech_recognizer.process_audio_chunk(audio_chunk)
                
                # Check for speech activity
                if not is_speaking:
                    # Detect start of speech
                    if result and result.text.strip():
                        is_speaking = True
                        silence_frames = 0
                        logger.info("Speech detected, beginning transcription")
                else:
                    # Check for end of utterance (silence after speech)
                    if not result or not result.text.strip():
                        silence_frames += 1
                    else:
                        silence_frames = 0
                
                # If we've detected enough silence after speech, process the utterance
                if is_speaking and silence_frames >= max_silence_frames:
                    is_speaking = False
                    
                    # Get final transcription
                    transcription, duration = await self.speech_recognizer.stop_streaming()
                    
                    if transcription.strip():
                        logger.info(f"Processing utterance: {transcription}")
                        
                        # Start a new streaming response
                        response_start = time.time()
                        
                        # Stream response directly to TTS with minimum latency
                        async for chunk in self.query_engine.query_with_streaming(transcription):
                            chunk_text = chunk.get("chunk", "")
                            if chunk_text:
                                # Send each chunk immediately to TTS without any batching
                                await tts_callback(chunk_text)
                        
                        response_time = time.time() - response_start
                        
                        # Yield processing information
                        yield {
                            "transcription": transcription,
                            "response_time": response_time,
                            "utterance_duration": duration
                        }
                    
                    # Reset for next utterance
                    self.speech_recognizer.start_streaming()
        
        except Exception as e:
            logger.error(f"Error in real-time audio processing: {e}", exc_info=True)
            yield {"error": str(e)}
        
        finally:
            # Clean up
            await self.speech_recognizer.stop_streaming()
    
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
    
    def reset_conversation(self):
        """Reset the conversation state."""
        if self.conversation_manager:
            self.conversation_manager.reset()
            # Always set to WAITING_FOR_QUERY for voice interaction
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        if not self.conversation_manager:
            return []
        
        return self.conversation_manager.get_history()
    
    async def query_knowledge_base(
        self,
        query: str,
        stream: bool = False,
        top_k: int = 3
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Query the knowledge base directly without conversation context.
        
        Args:
            query: Query text
            stream: Whether to stream the response
            top_k: Number of top results to retrieve
            
        Returns:
            Query result or async iterator of response chunks
        """
        if not self.query_engine:
            await self.init()
        
        if stream:
            return self.query_engine.query_with_streaming(query)
        else:
            return await self.query_engine.query(query)
    
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
    
    # LangGraph preparation - placeholder methods for future implementation
    def get_agent_state(self) -> Dict[str, Any]:
        """
        Get the current agent state for LangGraph.
        
        Returns:
            Current agent state
        """
        return self.agent_state