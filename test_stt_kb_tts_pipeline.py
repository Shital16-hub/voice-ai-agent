#!/usr/bin/env python3
"""
Test script for the complete Voice AI Agent with TTS integration.
Tests the full pipeline: STT -> Knowledge Base -> TTS
"""
import asyncio
import logging
from pathlib import Path
import sys
import os
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced Voice AI Agent
from voice_ai_agent_with_tts import VoiceAIAgentWithTTS

async def test_full_pipeline(args):
    """Test the complete STT -> Knowledge Base -> TTS pipeline."""
    print("\n=== Testing Voice AI Agent Pipeline: STT -> KB -> TTS ===\n")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Output path for the generated speech
    if args.output_dir:
        output_speech_file = os.path.join(args.output_dir, "response.mp3")
    else:
        output_speech_file = "response.mp3"
    
    # Initialize the agent
    agent = VoiceAIAgentWithTTS(
        storage_dir=args.storage_dir,
        model_name=args.model_name,
        whisper_model_path=args.whisper_model,
        tts_voice=args.tts_voice,
        llm_temperature=args.temperature
    )
    
    # Initialize components
    print("Initializing Voice AI Agent components...")
    await agent.init()
    
    # Get agent stats
    stats = await agent.get_stats()
    print(f"Agent statistics:")
    print(f"- LLM Model: {stats['model_name']}")
    print(f"- Whisper Model: {stats['speech_recognizer_model']}")
    print(f"- TTS Voice: {stats['tts_voice'] or 'default'}")
    print(f"- Knowledge base documents: {stats['knowledge_base']['document_count']}")
    
    # Run the end-to-end pipeline
    print(f"\nProcessing audio file: {args.input_file}")
    start_time = time.time()
    
    # Run the pipeline
    result = await agent.end_to_end_pipeline(
        audio_file_path=args.input_file,
        output_speech_file=output_speech_file
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print("\n=== Pipeline Results ===")
    print(f"Transcription: {result['transcription']}")
    print(f"Response: {result['response']}")
    
    # Print timing information
    print("\n=== Timing Information ===")
    print(f"STT time: {result['timings']['stt']:.2f} seconds")
    print(f"Knowledge Base time: {result['timings']['kb']:.2f} seconds")
    print(f"TTS time: {result['timings']['tts']:.2f} seconds")
    print(f"Total pipeline time: {result['total_time']:.2f} seconds")
    
    # Print output information
    print(f"\nGenerated speech audio saved to: {output_speech_file}")
    print(f"Audio size: {result['speech_audio_size']} bytes")
    
    print("\nPipeline test completed successfully!")

async def test_with_streaming(args):
    """Test the STT -> KB -> TTS pipeline with streaming responses."""
    print("\n=== Testing Voice AI Agent Pipeline with Streaming: STT -> KB -> TTS ===\n")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the agent
    agent = VoiceAIAgentWithTTS(
        storage_dir=args.storage_dir,
        model_name=args.model_name,
        whisper_model_path=args.whisper_model,
        tts_voice=args.tts_voice,
        llm_temperature=args.temperature
    )
    
    # Initialize components
    print("Initializing Voice AI Agent components...")
    await agent.init()
    
    # Define a callback to handle TTS audio chunks
    chunk_counter = 0
    
    async def tts_audio_callback(audio_chunk):
        nonlocal chunk_counter
        chunk_counter += 1
        
        # Save the chunk for debugging/verification
        if args.output_dir:
            chunk_file = os.path.join(args.output_dir, f"chunk_{chunk_counter}.mp3")
            with open(chunk_file, "wb") as f:
                f.write(audio_chunk)
            
            if chunk_counter % 5 == 0:
                print(f"Saved {chunk_counter} audio chunks...")
    
    # Run the streaming pipeline
    print(f"\nProcessing audio file with streaming: {args.input_file}")
    start_time = time.time()
    
    # Run the streaming pipeline
    result = await agent.process_audio_to_speech_stream(
        audio_file_path=args.input_file,
        tts_callback=tts_audio_callback
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print("\n=== Streaming Pipeline Results ===")
    print(f"Transcription: {result['transcription']}")
    print(f"Response: {result['full_response']}")
    
    # Print timing information
    print("\n=== Timing Information ===")
    print(f"Transcription time: {result['transcription_time']:.2f} seconds")
    print(f"Response stream time: {result['response_stream_time']:.2f} seconds")
    print(f"Total processing time: {result['total_time']:.2f} seconds")
    
    # Print output information
    print(f"\nGenerated {result['total_audio_chunks']} audio chunks")
    if args.output_dir:
        print(f"Audio chunks saved to: {args.output_dir}")
    
    print("\nStreaming pipeline test completed successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the Voice AI Agent STT -> KB -> TTS pipeline')
    
    # Input and output options
    parser.add_argument('--input-file', required=True, help='Path to input audio file')
    parser.add_argument('--output-dir', default=None, help='Directory to save output files')
    
    # Model options
    parser.add_argument('--storage-dir', default='./storage', help='Storage directory for knowledge base')
    parser.add_argument('--model-name', default='mistral:7b-instruct-v0.2-q4_0', help='LLM model name')
    parser.add_argument('--whisper-model', default='base.en', help='Whisper model path')
    parser.add_argument('--tts-voice', default=None, help='TTS voice (Deepgram voice ID)')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    
    # Mode options
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Run the appropriate test
    try:
        if args.streaming:
            asyncio.run(test_with_streaming(args))
        else:
            asyncio.run(test_full_pipeline(args))
        return 0
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())