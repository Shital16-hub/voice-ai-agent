#!/usr/bin/env python3
"""
Test script for knowledge base with LLM integration.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.llm_setup import setup_global_llm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_rag():
    """Test RAG with LLM."""
    print("\n=== Testing Knowledge Base with LLM (Full RAG) ===\n")
    
    # Initialize LLM
    llm = setup_global_llm(model_name="mistral:7b-instruct-v0.2-q4_0")
    print(f"Initialized LLM: {llm.model}")
    
    # Initialize index manager
    storage_dir = "./storage"  # Change this to your storage directory
    index_manager = IndexManager(storage_dir=storage_dir)
    await index_manager.init()
    
    # Get document count
    doc_count = await index_manager.count_documents()
    print(f"Knowledge base contains {doc_count} document chunks")
    
    if doc_count == 0:
        print("No documents found. Please index some documents first.")
        return
    
    # Initialize query engine with LLM
    query_engine = QueryEngine(
        index_manager=index_manager,
        llm_model_name="mistral:7b-instruct-v0.2-q4_0",
        llm_temperature=0.7
    )
    await query_engine.init()
    
    # Test queries
    test_queries = [
        "What are the pricing plans?",
        "What features does the product have?",
        "Who is the company?",
        "What is the architecture of the system?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("Generating response...")
        
        # Get response with sources
        response = await query_engine.query(query)
        
        print(f"Response: {response['response']}")
        print(f"Based on {len(response['sources'])} sources")
        
        # Display sources
        if response['sources']:
            print("\nSources:")
            for i, source in enumerate(response['sources']):
                print(f"  {i+1}. {source.get('text', '')[:100]}...")

async def test_streaming_rag():
    """Test RAG with streaming LLM responses."""
    print("\n=== Testing Knowledge Base with Streaming LLM ===\n")
    
    # Initialize LLM
    llm = setup_global_llm(model_name="mistral:7b-instruct-v0.2-q4_0")
    
    # Initialize index manager
    storage_dir = "./storage"  # Change this to your storage directory
    index_manager = IndexManager(storage_dir=storage_dir)
    await index_manager.init()
    
    # Initialize query engine with LLM
    query_engine = QueryEngine(
        index_manager=index_manager,
        llm_model_name="mistral:7b-instruct-v0.2-q4_0",
        llm_temperature=0.7
    )
    await query_engine.init()
    
    # Test streaming with one query
    query = "Explain the system architecture in detail"
    print(f"\nQuery: {query}")
    print("Streaming response: ", end="", flush=True)
    
    # Stream response
    async for chunk in query_engine.query_with_streaming(query):
        if "chunk" in chunk and chunk["chunk"]:
            print(chunk["chunk"], end="", flush=True)
    
    print("\n\nStreaming complete")

async def main():
    """Main function."""
    try:
        # Test regular RAG
        await test_rag()
        
        # Test streaming RAG
        await test_streaming_rag()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)