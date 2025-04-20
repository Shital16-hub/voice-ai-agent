#!/usr/bin/env python3
"""
Example of integrating speech-to-text with language model inference.
"""
import sys
import os
import argparse
import asyncio
import logging
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from language_model.model_manager import ModelManager
from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceAssistantPipeline:
    """
    Pipeline for connecting speech-to-text with language model.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        whisper_model_path: str = "base.en",
        language: str = "en",
        scenario: str = "voice"
    ):
        """
        Initialize the voice assistant pipeline.
        
        Args:
            model_name: Name of the language model to use
            whisper_model_path: Path to Whisper.cpp model
            language: Language code for speech recognition
            scenario: Conversation scenario
        """
        # Initialize language model
        self.model_manager = ModelManager(
            model_name=model_name,
            scenario=scenario
        )
        
        # Initialize speech-to-text
        self.asr = StreamingWhisperASR(
            model_path=whisper_model_path,
            language=language,
            n_threads=4,
            chunk_size_ms=2000,
            vad_enabled=True,
            single_segment=True
        )
        
        logger.info("Initialized VoiceAssistantPipeline")
    
    async def process_audio_file(
        self, 
        audio_file: str,
        chunk_size_ms: int = 1000,
        simulate_realtime: bool = True
    ):
        """
        Process an audio file through the pipeline.
        
        Args:
            audio_file: Path to audio file
            chunk_size_ms: Size of audio chunks in milliseconds
            simulate_realtime: Whether to simulate real-time processing
        """
        logger.info(f"Processing audio file: {audio_file}")
        
        # Placeholder for accumulated transcription
        current_transcription = ""
        finalized_segments = []
        
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
                    
                    # Generate response from language model
                    print("Generating response... ", end="", flush=True)
                    
                    # Use streaming response for better UX
                    full_response = ""
                    async for chunk in self.model_manager.generate_streaming_response(
                        user_input=result.text,
                        temperature=0.7
                    ):
                        chunk_text = chunk["chunk"]
                        print(chunk_text, end="", flush=True)
                        full_response += chunk_text
                    
                    print("\n")
        
        try:
            # Load audio file in chunks
            from speech_to_text.utils.audio_utils import load_audio_file
            
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
                
                # Generate final response if needed
                print("Generating final response... ", end="", flush=True)
                response = await self.model_manager.generate_response(
                    user_input=final_text,
                    temperature=0.7
                )
                print(response["response"])
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"\nProcessed {duration:.2f}s of audio in {processing_time:.2f}s "
                      f"({duration/processing_time:.2f}x real-time)")
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def process_audio_chunk(self, audio_chunk):
        """
        Process a single audio chunk (for real-time streaming).
        
        Args:
            audio_chunk: Audio data as numpy array
        """
        # Process audio chunk with streaming ASR
        current_transcription = ""
        
        async def transcription_callback(result: StreamingTranscriptionResult):
            nonlocal current_transcription
            if result.is_final and result.text.strip():
                current_transcription = result.text
                logger.info(f"Transcription: {result.text}")
                
                # In a real system, you'd generate the response and send to TTS
                
        await self.asr.process_audio_chunk(audio_chunk, callback=transcription_callback)
        
        return current_transcription
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'asr'):
            final_text, _ = await self.asr.stop_streaming()
            logger.info(f"Final transcription: {final_text}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Speech-to-text and language model integration')
    parser.add_argument('--audio', type=str, required=True,
                      help='Path to audio file')
    parser.add_argument('--whisper-model', type=str, default="base.en",
                      help='Path to Whisper model')
    parser.add_argument('--language', type=str, default='en',
                      help='Language code (default: en)')
    parser.add_argument('--language-model', type=str,
                      help='Name of language model to use')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Chunk size in milliseconds (default: 1000)')
    parser.add_argument('--no-realtime', action='store_true',
                      help='Disable real-time simulation')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = VoiceAssistantPipeline(
            model_name=args.language_model,
            whisper_model_path=args.whisper_model,
            language=args.language,
            scenario="voice"
        )
        
        # Process audio file
        await pipeline.process_audio_file(
            audio_file=args.audio,
            chunk_size_ms=args.chunk_size,
            simulate_realtime=not args.no_realtime
        )
        
        # Cleanup
        await pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)