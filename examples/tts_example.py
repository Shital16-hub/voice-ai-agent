"""
Example usage of the Text-to-Speech module.

This script demonstrates how to use the TTS module for both batch
and streaming synthesis.
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TTS module
from text_to_speech import DeepgramTTS, RealTimeResponseHandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_batch_synthesis(tts_client, text, output_file):
    """
    Demonstrate batch synthesis of text to speech.
    
    Args:
        tts_client: DeepgramTTS client
        text: Text to synthesize
        output_file: Path to save the audio file
    """
    logger.info(f"Synthesizing text: {text}")
    
    # Synthesize the text
    audio_data = await tts_client.synthesize(text)
    
    # Save the audio data to a file
    with open(output_file, 'wb') as f:
        f.write(audio_data)
    
    logger.info(f"Saved audio to {output_file}")

async def word_generator(text):
    """
    Generate words from text with delays to simulate real-time output.
    
    Args:
        text: Text to split into words
        
    Yields:
        Words from the text with delays
    """
    words = text.split()
    for word in words:
        yield word
        # Simulate delay between words (100-300ms)
        await asyncio.sleep(0.1 + (len(word) * 0.01))

async def demo_streaming_synthesis(tts_client, text, output_file):
    """
    Demonstrate streaming synthesis of text to speech.
    
    Args:
        tts_client: DeepgramTTS client
        text: Text to synthesize
        output_file: Path to save the audio file
    """
    logger.info(f"Streaming synthesis of: {text}")
    
    # Create the streaming handler
    handler = RealTimeResponseHandler(tts_streamer=None, tts_client=tts_client)
    
    # Start the handler
    audio_stream = handler.start()
    audio_chunks = []
    
    # Process audio chunks in the background
    async def collect_audio():
        async for chunk in audio_stream:
            audio_chunks.append(chunk)
    
    # Start collecting audio
    collect_task = asyncio.create_task(collect_audio())
    
    # Generate words one by one
    async for word in word_generator(text):
        logger.info(f"Adding word: {word}")
        await handler.add_word(word)
    
    # Flush and stop the handler
    await handler.flush()
    await handler.stop()
    
    # Wait for audio collection to complete
    await collect_task
    
    # Combine all audio chunks and save to file
    with open(output_file, 'wb') as f:
        for chunk in audio_chunks:
            f.write(chunk)
    
    logger.info(f"Saved streaming audio to {output_file}")

async def main():
    """Main entry point for the example."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text-to-Speech Example')
    parser.add_argument('--mode', choices=['batch', 'streaming'], default='streaming',
                        help='Synthesis mode: batch or streaming')
    parser.add_argument('--text', default="Hello, this is a test of real-time text to speech synthesis for our Voice AI Agent project.",
                        help='Text to synthesize')
    parser.add_argument('--output', default='output.mp3',
                        help='Output audio file')
    args = parser.parse_args()
    
    # Create the TTS client
    tts_client = DeepgramTTS()
    
    # Run the appropriate demo
    if args.mode == 'batch':
        await demo_batch_synthesis(tts_client, args.text, args.output)
    else:
        await demo_streaming_synthesis(tts_client, args.text, args.output)

if __name__ == '__main__':
    asyncio.run(main())