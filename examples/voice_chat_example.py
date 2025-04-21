#!/usr/bin/env python3
"""
Example script for voice-based conversation using the Voice AI Agent.
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

async def process_audio_file(agent: VoiceAIAgent, audio_file: str, simulate_realtime: bool = True):
    """
    Process an audio file and generate a response.
    
    Args:
        agent: Initialized VoiceAIAgent
        audio_file: Path to audio file
        simulate_realtime: Whether to simulate real-time processing
    """
    print("\n" + "="*50)
    print(f"Processing audio file: {audio_file}")
    print("="*50 + "\n")
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    try:
        # Process the full audio file
        result = await agent.process_full_audio_file(
            audio_file_path=audio_file,
            chunk_size_ms=1000,
            simulate_realtime=simulate_realtime
        )
        
        # Display results
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nTranscription ({result['duration']:.2f}s): {result['transcription']}")
        print(f"\nResponse: {result['response']}")
        
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error processing audio file: {e}")

async def interactive_text_chat(agent: VoiceAIAgent):
    """
    Interactive text chat with the Voice AI Agent.
    
    Args:
        agent: Initialized VoiceAIAgent
    """
    print("\n" + "="*50)
    print("Interactive Text Chat")
    print("Type 'exit', 'quit', or press Ctrl+C to end the chat")
    print("Type 'reset' to reset the conversation")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for commands
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        if user_input.lower() == "reset":
            agent.reset_conversation()
            print("Conversation reset.")
            continue
        
        # Process with streaming response
        print("\nAI: ", end="", flush=True)
        async for chunk in agent.process_text_streaming(user_input):
            if "chunk" in chunk and chunk["chunk"]:
                print(chunk["chunk"], end="", flush=True)
        
        print()  # New line after response

async def simulate_voice_conversation(agent: VoiceAIAgent, audio_directory: str):
    """
    Simulate a voice conversation using multiple audio files.
    
    Args:
        agent: Initialized VoiceAIAgent
        audio_directory: Directory containing audio files
    """
    print("\n" + "="*50)
    print(f"Simulating Voice Conversation from: {audio_directory}")
    print("="*50 + "\n")
    
    # Check if directory exists
    if not os.path.isdir(audio_directory):
        print(f"Error: Directory not found: {audio_directory}")
        return
    
    # Get audio files sorted by name
    audio_files = sorted([
        os.path.join(audio_directory, f) 
        for f in os.listdir(audio_directory)
        if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))
    ])
    
    if not audio_files:
        print(f"No audio files found in {audio_directory}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n--- Turn {i+1}/{len(audio_files)} ---")
        print(f"Processing: {os.path.basename(audio_file)}")
        
        try:
            # Load audio
            audio, sample_rate = load_audio_file(audio_file, target_sr=16000)
            
            # Define a callback to show interim results
            async def transcription_callback(result):
                if result.text.strip():
                    print(f"Interim transcription: {result.text}")
            
            # Process audio
            result = await agent.process_audio(audio, callback=transcription_callback)
            
            if result and result.text.strip():
                # Get final transcription
                print(f"\nTranscription: {result.text}")
                
                # Generate streaming response
                print("\nAI: ", end="", flush=True)
                async for chunk in agent.process_text_streaming(result.text):
                    if "chunk" in chunk and chunk["chunk"]:
                        print(chunk["chunk"], end="", flush=True)
                
                print()  # New line after response
            else:
                print("No speech detected in this audio file.")
            
            # Small pause between turns
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_file}: {e}")
            print(f"Error processing audio file: {e}")
    
    print("\nConversation simulation complete.")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Voice Chat Example")
    parser.add_argument("--model", type=str, help="LLM model name")
    parser.add_argument("--whisper-model", type=str, default="base.en", help="Whisper model path")
    parser.add_argument("--audio", type=str, help="Path to audio file for processing")
    parser.add_argument("--audio-dir", type=str, help="Directory with audio files for conversation simulation")
    parser.add_argument("--storage-dir", type=str, default="./storage", help="Knowledge base storage directory")
    parser.add_argument("--language", type=str, default="en", help="Language code for speech recognition")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable response streaming")
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
            language=args.language,
            llm_temperature=args.temperature
        )
        
        # Initialize components
        await agent.init()
        
        # Determine which mode to run
        if args.audio:
            await process_audio_file(
                agent=agent,
                audio_file=args.audio,
                simulate_realtime=not args.no_realtime
            )
        elif args.audio_dir:
            await simulate_voice_conversation(
                agent=agent,
                audio_directory=args.audio_dir
            )
        else:
            await interactive_text_chat(agent=agent)
            
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