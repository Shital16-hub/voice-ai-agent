#!/usr/bin/env python3
"""
Example script for using the refactored RAG-integrated Voice AI Agent.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def interactive_query(agent: VoiceAIAgent, stream: bool = True):
    """
    Interactive RAG query interface.
    
    Args:
        agent: Initialized VoiceAIAgent
        stream: Whether to stream responses
    """
    print("\n" + "="*50)
    print("RAG Query Example")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session")
    print("Type 'stats' to see knowledge base statistics")
    print("="*50 + "\n")
    
    # Get stats
    stats = await agent.get_stats()
    doc_count = stats["knowledge_base"]["document_count"]
    print(f"Knowledge base contains {doc_count} document chunks")
    
    while True:
        # Get query
        query = input("\nEnter your query: ")
        
        # Check for exit
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Check for stats
        if query.lower() == "stats":
            stats = await agent.get_stats()
            print("\nKnowledge Base Statistics:")
            for key, value in stats["knowledge_base"].items():
                print(f"  {key}: {value}")
            continue
        
        # Process query
        start_time = asyncio.get_event_loop().time()
        
        if stream:
            print("\nResponse: ", end="", flush=True)
            async for chunk in agent.query_knowledge_base(query, stream=True):
                if "chunk" in chunk and chunk["chunk"]:
                    print(chunk["chunk"], end="", flush=True)
                
                # Check if done
                if chunk.get("done", False):
                    sources = chunk.get("sources", [])
                    
                    # Print sources
                    if sources:
                        print("\n\nSources:")
                        for i, source in enumerate(sources):
                            print(f"  {i+1}. {source.get('name', 'Unknown')}")
        else:
            result = await agent.query_knowledge_base(query, stream=False)
            print(f"\nResponse: {result['response']}")
            
            # Print sources
            if "sources" in result and result["sources"]:
                print("\nSources:")
                for i, source in enumerate(result["sources"]):
                    print(f"  {i+1}. {source.get('text', '')[:100]}...")
        
        query_time = asyncio.get_event_loop().time() - start_time
        print(f"\nQuery processed in {query_time:.3f}s")

async def batch_query(agent: VoiceAIAgent, queries: list):
    """
    Run a batch of queries.
    
    Args:
        agent: Initialized VoiceAIAgent
        queries: List of query strings
    """
    print("\n" + "="*50)
    print("Batch Query Mode")
    print("="*50 + "\n")
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")
        
        # Process query
        start_time = asyncio.get_event_loop().time()
        result = await agent.query_knowledge_base(query, stream=False)
        query_time = asyncio.get_event_loop().time() - start_time
        
        print(f"Response: {result['response']}")
        print(f"Query processed in {query_time:.3f}s")
        
        # Print sources
        if "sources" in result and result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"]):
                print(f"  {i+1}. {source.get('text', '')[:100]}...")
        
        print("-" * 40)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="RAG Query Example")
    parser.add_argument("--model", type=str, help="LLM model name")
    parser.add_argument("--storage-dir", type=str, default="./storage", help="Knowledge base storage directory")
    parser.add_argument("--no-stream", action="store_true", help="Disable response streaming")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--queries", type=str, help="Path to file with queries for batch processing")
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
            llm_temperature=args.temperature
        )
        
        # Initialize components
        await agent.init()
        
        # Run batch or interactive mode
        if args.queries:
            with open(args.queries, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            await batch_query(agent, queries)
        else:
            await interactive_query(agent, stream=not args.no_stream)
            
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