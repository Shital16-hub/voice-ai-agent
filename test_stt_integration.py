#!/usr/bin/env python3
"""
Test speech-to-text integration with Voice AI Agent.
"""
import asyncio
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_ai_agent import VoiceAIAgent
from speech_to_text.utils.audio_utils import load_audio_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_stt_integration(audio_file: str):
    """Test speech-to-text integration."""
    print(f"\n=== Testing STT Integration with file: {audio_file} ===\n")
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Initialize the agent
    agent = VoiceAIAgent(
        storage_dir="./storage",
        model_name="mistral:7b-instruct-v0.2-q4_0",
        whisper_model_path="base.en"  # Adjust to your model path
    )
    
    # Initialize components
    await agent.init()
    
    # Process audio file
    result = await agent.process_full_audio_file(audio_file)
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Transcription: {result['transcription']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Response: {result['response']}")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test STT integration")
    parser.add_argument("--audio", type=str, required=True,
                      help="Path to audio file")
    
    args = parser.parse_args()
    
    try:
        await test_stt_integration(args.audio)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)