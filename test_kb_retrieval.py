#!/usr/bin/env python3
"""
Test script for knowledge base retrieval without LLM.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_retrieval():
    """Test pure retrieval without LLM."""
    print("\n=== Testing Knowledge Base Retrieval (No LLM) ===\n")
    
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
    
    # Initialize query engine but don't initialize LLM
    query_engine = QueryEngine(index_manager=index_manager)
    
    # We'll only initialize the retriever part, not the LLM
    # Set is_initialized to make retrieve() work without calling init()
    query_engine.retriever = query_engine.index_manager.index.as_retriever(
        similarity_top_k=query_engine.top_k
    )
    query_engine.is_initialized = True
    
    # Test queries
    test_queries = [
        "What are the pricing plans?",
        "What features does the product have?",
        "Who is the company?",
        "What is the architecture of the system?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Retrieve documents
        docs = await query_engine.retrieve(query)
        
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"\n--- Result {i+1} ---")
            print(f"Text: {doc.text[:150]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            if 'score' in doc.metadata:
                print(f"Score: {doc.metadata['score']:.4f}")

async def main():
    """Main function."""
    try:
        await test_retrieval()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)