#!/usr/bin/env python3
"""
Example script for retrieving information from the knowledge base.
"""
import os
import sys
import argparse
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_base.embedding_generator import EmbeddingGenerator, MockEmbeddingGenerator
from knowledge_base.vector_store import VectorStore, InMemoryVectorStore
from knowledge_base.retriever import Retriever

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def retrieve_information(
    query: str,
    use_mock: bool = False,
    in_memory: bool = False,
    top_k: int = 3,
    min_score: float = 0.6
) -> Dict[str, Any]:
    """
    Retrieve information from knowledge base.
    
    Args:
        query: Query text
        use_mock: Whether to use mock embedding generator
        in_memory: Whether to use in-memory vector store
        top_k: Number of results to return
        min_score: Minimum similarity score
        
    Returns:
        Dictionary with retrieval results
    """
    logger.info(f"Retrieving information for query: {query}")
    
    # Choose embedding generator
    if use_mock:
        embedding_generator = MockEmbeddingGenerator()
        logger.info("Using MockEmbeddingGenerator")
    else:
        embedding_generator = EmbeddingGenerator()
        logger.info("Using EmbeddingGenerator")
    
    # Choose vector store
    if in_memory:
        vector_store = InMemoryVectorStore()
        logger.info("Using InMemoryVectorStore")
    else:
        vector_store = VectorStore()
        logger.info("Using Qdrant VectorStore")
    
    # Initialize components
    await vector_store.init()
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
    await retriever.init()
    
    # Retrieve information
    results = await retriever.retrieve_with_sources(
        query=query,
        top_k=top_k,
        min_score=min_score
    )
    
    return results

async def interactive_retrieval(use_mock: bool = False, in_memory: bool = False):
    """
    Run interactive retrieval session.
    
    Args:
        use_mock: Whether to use mock embedding generator
        in_memory: Whether to use in-memory vector store
    """
    # Choose embedding generator
    if use_mock:
        embedding_generator = MockEmbeddingGenerator()
        logger.info("Using MockEmbeddingGenerator")
    else:
        embedding_generator = EmbeddingGenerator()
        logger.info("Using EmbeddingGenerator")
    
    # Choose vector store
    if in_memory:
        vector_store = InMemoryVectorStore()
        logger.info("Using InMemoryVectorStore")
    else:
        vector_store = VectorStore()
        logger.info("Using Qdrant VectorStore")
    
    # Initialize components
    await vector_store.init()
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
    await retriever.init()
    
    # Check document count
    stats = await retriever.get_stats()
    doc_count = stats["document_count"]
    
    print("\n" + "="*50)
    print(f"Knowledge Base Retrieval")
    print(f"Documents in knowledge base: {doc_count}")
    print("Type 'exit' or 'quit' to end the session")
    print("="*50 + "\n")
    
    if doc_count == 0:
        print("Warning: No documents found in knowledge base.")
        print("Please index some documents first using indexing_example.py")
    
    while True:
        # Get query
        query = input("\nEnter your query: ")
        
        # Check for exit
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Retrieve information
        try:
            results = await retriever.retrieve_with_sources(query=query)
            
            # Display results
            print("\nRetrieved results:")
            
            if not results["results"]:
                print("  No relevant information found.")
            else:
                for i, doc in enumerate(results["results"]):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Score: {doc['score']:.4f}")
                    
                    # Show metadata
                    source = doc["metadata"].get("source", "Unknown")
                    print(f"Source: {source}")
                    
                    # Show text (truncated if long)
                    text = doc["text"]
                    if len(text) > 500:
                        text = text[:497] + "..."
                    print(f"Content: {text}")
            
            # Show sources
            if results["sources"]:
                print("\nSources:")
                for i, source in enumerate(results["sources"]):
                    print(f"  {i+1}. {source['name']} ({source['type']})")
            
        except Exception as e:
            logger.error(f"Error retrieving information: {e}")
            print(f"Error: {e}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Retrieve information from knowledge base')
    parser.add_argument('--query', type=str,
                      help='Query to search for')
    parser.add_argument('--use-mock', action='store_true',
                      help='Use mock embedding generator (faster but less accurate)')
    parser.add_argument('--in-memory', action='store_true',
                      help='Use in-memory vector store (no Qdrant required)')
    parser.add_argument('--top-k', type=int, default=3,
                      help='Number of results to return')
    parser.add_argument('--min-score', type=float, default=0.6,
                      help='Minimum similarity score')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            # Run interactive session
            await interactive_retrieval(
                use_mock=args.use_mock,
                in_memory=args.in_memory
            )
        elif args.query:
            # Single query
            results = await retrieve_information(
                query=args.query,
                use_mock=args.use_mock,
                in_memory=args.in_memory,
                top_k=args.top_k,
                min_score=args.min_score
            )
            
            # Display results
            print("\nQuery:", args.query)
            print("\nRetrieved results:")
            
            if not results["results"]:
                print("  No relevant information found.")
            else:
                for i, doc in enumerate(results["results"]):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Score: {doc['score']:.4f}")
                    print(f"Source: {doc['metadata'].get('source', 'Unknown')}")
                    print(f"Content: {doc['text'][:500]}...")
            
        else:
            print("Please provide a query or use --interactive")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)