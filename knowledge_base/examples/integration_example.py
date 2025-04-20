#!/usr/bin/env python3
"""
Example script for integrating speech-to-text, knowledge base, and language model.
"""
import os
import sys
import argparse
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import components
from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult
from speech_to_text.utils.audio_utils import load_audio_file

from knowledge_base.document_processor import DocumentProcessor
from knowledge_base.embedding_generator import EmbeddingGenerator, MockEmbeddingGenerator
from knowledge_base.vector_store import VectorStore, InMemoryVectorStore
from knowledge_base.retriever import Retriever
from knowledge_base.conversation_manager import ConversationManager, ConversationState

from language_model.model_manager import ModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """
    Integrated Voice AI Agent with speech recognition, knowledge retrieval, and language generation.
    """
    
    def __init__(
        self,
        whisper_model_path: str = "base.en",
        language: str = "en",
        use_mock_embeddings: bool = False,
        use_in_memory_store: bool = False,
        language_model_name: Optional[str] = None
    ):
        """
        Initialize Voice AI Agent.
        
        Args:
            whisper_model_path: Path to Whisper model
            language: Language code
            use_mock_embeddings: Whether to use mock embeddings (for testing)
            use_in_memory_store: Whether to use in-memory vector store
            language_model_name: Name of language model to use
        """
        self.whisper_model_path = whisper_model_path
        self.language = language
        self.use_mock_embeddings = use_mock_embeddings
        self.use_in_memory_store = use_in_memory_store
        self.language_model_name = language_model_name
        
        # Components will be initialized in init() method
        self.asr = None
        self.embedding_generator = None
        self.vector_store = None
        self.retriever = None
        self.conversation_manager = None
        self.language_model = None
        
        logger.info("Initialized VoiceAIAgent instance")
    
    async def init(self):
        """Initialize all components."""
        # Initialize speech recognition
        logger.info(f"Initializing speech recognition with model: {self.whisper_model_path}")
        self.asr = StreamingWhisperASR(
            model_path=self.whisper_model_path,
            language=self.language,
            n_threads=4,
            chunk_size_ms=2000,
            vad_enabled=True,
            single_segment=True
        )
        
        # Initialize embedding generator
        if self.use_mock_embeddings:
            logger.info("Using mock embedding generator")
            self.embedding_generator = MockEmbeddingGenerator()
        else:
            logger.info("Initializing embedding generator")
            self.embedding_generator = EmbeddingGenerator()
        
        # Initialize vector store
        if self.use_in_memory_store:
            logger.info("Using in-memory vector store")
            self.vector_store = InMemoryVectorStore()
        else:
            logger.info("Initializing Qdrant vector store")
            self.vector_store = VectorStore()
        
        await self.vector_store.init()
        
        # Initialize retriever
        logger.info("Initializing knowledge retriever")
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        await self.retriever.init()
        
        # Initialize language model
        logger.info(f"Initializing language model: {self.language_model_name or 'default'}")
        self.language_model = ModelManager(
            model_name=self.language_model_name,
            scenario="voice",
            with_rag=True
        )
        
        # Define language model callback for conversation manager
        async def language_model_callback(
            query: str, 
            context: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None
        ) -> Dict[str, Any]:
            if context:
                # Use RAG if context is provided
                response = await self.language_model.generate_rag_response(
                    user_input=query,
                    context=context,
                    temperature=0.7
                )
            else:
                # Use regular response if no context
                response = await self.language_model.generate_response(
                    user_input=query,
                    temperature=0.7
                )
            
            return response
        
        # Initialize conversation manager
        logger.info("Initializing conversation manager")
        self.conversation_manager = ConversationManager(
            retriever=self.retriever,
            language_model_callback=language_model_callback
        )
        await self.conversation_manager.init()
        
        logger.info("All components initialized successfully")
    
    async def process_audio_file(
        self, 
        audio_file: str,
        chunk_size_ms: int = 1000,
        simulate_realtime: bool = True
    ) -> Dict[str, Any]:
        """
        Process an audio file through the pipeline.
        
        Args:
            audio_file: Path to audio file
            chunk_size_ms: Size of audio chunks in milliseconds
            simulate_realtime: Whether to simulate real-time processing
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Processing audio file: {audio_file}")
        
        # Placeholder for accumulated transcription
        current_transcription = ""
        finalized_segments = []
        responses = []
        
        # Define callback for ASR results
        async def transcription_callback(result: StreamingTranscriptionResult):
            nonlocal current_transcription
            
            if result.text.strip():
                current_transcription = result.text
                logger.info(f"[{result.start_time:.2f}s-{result.end_time:.2f}s] {result.text}")
                
                # If segment is final, generate response
                if result.is_final:
                    finalized_segments.append(result.text)
                    print(f"\nTranscription: {result.text}")
                    
                    # Process through conversation manager
                    response = await self.conversation_manager.handle_user_input(result.text)
                    
                    # Print response
                    print(f"AI: {response['response']}")
                    responses.append(response)
        
        try:
            # Load audio file
            audio, sample_rate = load_audio_file(audio_file, target_sr=self.asr.sample_rate)
            
            # Calculate chunk size in samples
            chunk_size = int(sample_rate * chunk_size_ms / 1000)
            
            # Split audio into chunks
            num_chunks = (len(audio) + chunk_size - 1) // chunk_size
            
            # Process each chunk
            start_time = asyncio.get_event_loop().time()
            
            for i in range(num_chunks):
                # Get chunk
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(audio))
                chunk = audio[chunk_start:chunk_end]
                
                # Process chunk
                await self.asr.process_audio_chunk(chunk, callback=transcription_callback)
                
                # Simulate real-time processing
                if simulate_realtime and i < num_chunks - 1:
                    await asyncio.sleep(chunk_size_ms / 1000)
            
            # Get final transcription
            final_text, duration = await self.asr.stop_streaming()
            
            if final_text and final_text not in finalized_segments:
                print(f"\nFinal transcription: {final_text}")
                
                # Process through conversation manager
                response = await self.conversation_manager.handle_user_input(final_text)
                
                # Print response
                print(f"AI: {response['response']}")
                responses.append(response)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"\nProcessed {duration:.2f}s of audio in {processing_time:.2f}s "
                      f"({duration/processing_time:.2f}x real-time)")
            
            # Return results
            return {
                "transcriptions": finalized_segments,
                "final_text": final_text,
                "responses": responses,
                "processing_time": processing_time,
                "audio_duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def interactive_session(self):
        """
        Run an interactive session with text input/output.
        
        This simulates the full pipeline without speech input/output.
        """
        print("\n" + "="*50)
        print("Voice AI Agent Interactive Session")
        print("Type 'exit' or 'quit' to end the session")
        print("="*50 + "\n")
        
        # Get stats
        kb_stats = await self.retriever.get_stats()
        print(f"Knowledge base contains {kb_stats['document_count']} documents")
        
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(user_input)
            
            # Print response
            print(f"AI: {response['response']}")
            
            # If context was used, show it
            if response.get("context"):
                if "--verbose" in sys.argv:
                    print("\nContext used:")
                    print(response["context"])
    
    async def cleanup(self):
        """Clean up resources."""
        # Stop streaming
        if hasattr(self, 'asr') and self.asr:
            await self.asr.stop_streaming()
        
        logger.info("Cleaned up resources")

async def process_file(
    audio_file: str,
    whisper_model_path: str = "base.en",
    language: str = "en",
    use_mock: bool = False,
    in_memory: bool = False,
    language_model: Optional[str] = None
):
    """
    Process a single audio file through the agent.
    
    Args:
        audio_file: Path to audio file
        whisper_model_path: Path to Whisper model
        language: Language code
        use_mock: Whether to use mock embeddings
        in_memory: Whether to use in-memory vector store
        language_model: Language model name
    """
    # Initialize agent
    agent = VoiceAIAgent(
        whisper_model_path=whisper_model_path,
        language=language,
        use_mock_embeddings=use_mock,
        use_in_memory_store=in_memory,
        language_model_name=language_model
    )
    
    try:
        await agent.init()
        
        # Process audio file
        await agent.process_audio_file(
            audio_file=audio_file,
            simulate_realtime=True
        )
        
    finally:
        # Clean up
        await agent.cleanup()

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Voice AI Agent')
    parser.add_argument('--audio', type=str,
                      help='Path to audio file to process')
    parser.add_argument('--whisper-model', type=str, default="base.en",
                      help='Path to Whisper model')
    parser.add_argument('--language', type=str, default='en',
                      help='Language code (default: en)')
    parser.add_argument('--use-mock', action='store_true',
                      help='Use mock embedding generator (faster but less accurate)')
    parser.add_argument('--in-memory', action='store_true',
                      help='Use in-memory vector store (no Qdrant required)')
    parser.add_argument('--language-model', type=str,
                      help='Name of language model to use')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode (text input/output)')
    parser.add_argument('--verbose', action='store_true',
                      help='Show verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = VoiceAIAgent(
            whisper_model_path=args.whisper_model,
            language=args.language,
            use_mock_embeddings=args.use_mock,
            use_in_memory_store=args.in_memory,
            language_model_name=args.language_model
        )
        
        await agent.init()
        
        if args.interactive:
            # Run interactive session
            await agent.interactive_session()
        elif args.audio:
            # Process audio file
            await agent.process_audio_file(
                audio_file=args.audio,
                simulate_realtime=True
            )
        else:
            print("Please provide an audio file or use --interactive")
            return 1
        
        # Clean up
        await agent.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)