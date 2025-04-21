#!/usr/bin/env python3
"""
Test script for the complete Voice AI Agent.
"""
import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_ai_agent import VoiceAIAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_voice_ai_agent():
    """Test the complete Voice AI Agent."""
    print("\n=== Testing Voice AI Agent ===\n")
    
    # Initialize the agent
    agent = VoiceAIAgent(
        storage_dir="./storage",
        model_name="mistral:7b-instruct-v0.2-q4_0",
        llm_temperature=0.7
    )
    
    # Initialize components
    await agent.init()
    
    # Get agent stats
    stats = await agent.get_stats()
    print("Agent statistics:")
    print(f"- Model: {stats['model_name']}")
    print(f"- Temperature: {stats['llm_temperature']}")
    print(f"- Knowledge base documents: {stats['knowledge_base']['document_count']}")
    
    # Test queries
    test_queries = [
        "What are your pricing plans?",
        "Tell me about your product features",
        "Who is your company?",
        "Explain your system architecture"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        
        # Process query with streaming response
        print("AI: ", end="", flush=True)
        
        async for chunk in agent.process_text_streaming(query):
            if "chunk" in chunk and chunk["chunk"]:
                print(chunk["chunk"], end="", flush=True)
        
        print()  # New line after response
    
    print("\nTesting complete!")

async def main():
    """Main function."""
    try:
        await test_voice_ai_agent()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)