"""
Main Voice AI Agent module combining speech-to-text, RAG, and LLM capabilities.
Designed for future LangGraph integration.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Tuple

import numpy as np

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult
from knowledge_base.conversation_manager import ConversationManager
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
        use_langgraph: bool = False  # Flag for future LangGraph integration
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
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            use_langgraph=self.use_langgraph
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
                    single_segment=True
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
        
        # For future LangGraph implementation
        if self.use_langgraph:
            return await self._process_text_langgraph(text)
        
        # Process with current implementation
        response = await self.conversation_manager.handle_user_input(text)
        return response
    
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
        
        # For future LangGraph implementation
        if self.use_langgraph:
            async for chunk in self._process_text_streaming_langgraph(text):
                yield chunk
            return
        
        # Process with current implementation
        async for chunk in self.conversation_manager.generate_streaming_response(text):
            yield chunk
    
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
        callback: Optional[callable] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio input for speech recognition.
        
        Args:
            audio_chunk: Audio data as bytes or numpy array
            callback: Optional callback for interim results
            
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
        
        # Process with speech recognizer
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
        stream_response: bool = True
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Process speech input and generate response in a single call.
        
        Args:
            audio_chunk: Audio data
            stream_response: Whether to stream the response
            
        Returns:
            Response or async iterator of response chunks
        """
        # For future LangGraph implementation
        if self.use_langgraph:
            return await self._process_speech_to_response_langgraph(audio_chunk, stream_response)
        
        # Process audio for transcription
        transcription_result = await self.process_audio(audio_chunk)
        
        if not transcription_result or not transcription_result.text.strip():
            return {"error": "No speech detected or transcription failed"}
        
        # Get the transcribed text
        text = transcription_result.text
        
        # Log the transcription
        logger.info(f"Transcription: {text}")
        
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
            chunk_size_ms: Size of chunks in milliseconds
            simulate_realtime: Whether to simulate real-time processing
            
        Returns:
            Dictionary with transcription and response
        """
        if not self.speech_recognizer:
            await self.init_speech_recognition()
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {"error": f"Error loading audio file: {e}"}
        
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
            response = await self.process_text(final_text)
            result = {
                "transcription": final_text,
                "duration": duration,
                "response": response["response"]
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