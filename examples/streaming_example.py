#!/usr/bin/env python3
"""
Example script for streaming speech recognition and response generation.
"""
import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_ai_agent import VoiceAIAgent
from speech_to_text.utils.audio_utils import load_audio_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def streaming_audio_processing(
    agent: VoiceAIAgent, 
    audio_file: str, 
    chunk_size_ms: int = 1000,
    simulate_realtime: bool = True
):
    """
    Process an audio file in streaming chunks.
    
    Args:
        agent: Initialized VoiceAIAgent
        audio_file: Path to audio file
        chunk_size_ms: Size of each chunk in milliseconds
        simulate_realtime: Whether to simulate real-time processing
    """
    print("\n" + "="*50)
    print(f"Streaming Audio Processing: {audio_file}")
    print(f"Chunk size: {chunk_size_ms}ms, Simulate realtime: {simulate_realtime}")
    print("="*50 + "\n")
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    try:
        # Load audio
        audio, sample_rate = load_audio_file(audio_file, target_sr=16000)
        
        # Calculate chunk size in samples
        chunk_size = int(sample_rate * chunk_size_ms / 1000)
        
        # Split audio into chunks
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        
        print(f"Processing {len(audio)/sample_rate:.2f}s audio in {num_chunks} chunks...\n")
        
        # Initialize streaming variables
        current_transcription = ""
        
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            print(f"Chunk {i+1}/{num_chunks} ({len(chunk)/sample_rate:.2f}s):")
            
            # Define callback for interim results
            async def transcription_callback(result):
                nonlocal current_transcription
                if result.text and result.text.strip():
                    current_transcription = result.text
                    print(f"  Interim: {result.text}")
            
            # Process chunk
            result = await agent.process_audio(chunk, callback=transcription_callback)
            
            # Simulate real-time processing
            if simulate_realtime and i < num_chunks - 1:
                await asyncio.sleep(chunk_size_ms / 1000)
            
            # Check if we have a complete segment
            if result and result.is_final and result.text.strip():
                print(f"  Final segment: {result.text}")
                
                # Generate streaming response
                print("  AI response: ", end="", flush=True)
                full_response = ""
                
                async for chunk in agent.process_text_streaming(result.text):
                    if "chunk" in chunk and chunk["chunk"]:
                        chunk_text = chunk["chunk"]
                        full_response += chunk_text
                        print(chunk_text, end="", flush=True)
                
                print()  # New line after response
        
        # Get final transcription
        final_text, duration = await agent.end_audio_stream()
        
        print(f"\nFinal transcription ({duration:.2f}s): {final_text}")
        
        # Generate final response if different from last segment
        if final_text and final_text != current_transcription:
            print("\nFinal AI response: ", end="", flush=True)
            
            async for chunk in agent.process_text_streaming(final_text):
                if "chunk" in chunk and chunk["chunk"]:
                    print(chunk["chunk"], end="", flush=True)
            
            print()  # New line after response
        
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error processing audio file: {e}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Streaming Speech Recognition Example")
    parser.add_argument("--model", type=str, help="LLM model name")
    parser.add_argument("--whisper-model", type=str, default="base.en", help="Whisper model path")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file for processing")
    parser.add_argument("--storage-dir", type=str, default="./storage", help="Knowledge base storage directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in milliseconds")
    parser.add_argument("--language", type=str, default="en", help="Language code for speech recognition")
    parser.add_argument("--no-realtime", action="store_true", help="Disable real-time simulation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize agent
        agent = VoiceAIAgent(
            storage_dir=args.storage_dir,
            model_name=args.model,
            whisper_model_path=args.whisper_model,
            language=args.language
        )
        
        # Initialize components
        await agent.init()
        
        # Process audio in streaming mode
        await streaming_audio_processing(
            agent=agent,
            audio_file=args.audio,
            chunk_size_ms=args.chunk_size,
            simulate_realtime=not args.no_realtime
        )
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)