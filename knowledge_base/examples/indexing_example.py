#!/usr/bin/env python3
"""
Example script for indexing documents into the knowledge base.
"""
import os
import sys
import argparse
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_base.document_processor import DocumentProcessor
from knowledge_base.embedding_generator import EmbeddingGenerator, MockEmbeddingGenerator
from knowledge_base.vector_store import VectorStore, InMemoryVectorStore
from knowledge_base.retriever import Retriever
from knowledge_base.utils.file_utils import list_documents

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def index_documents(
    directory: str,
    use_mock: bool = False,
    in_memory: bool = False,
    extensions: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    batch_size: int = 3  # Added batch processing
) -> int:
    """
    Index documents from a directory.
    
    Args:
        directory: Directory with documents
        use_mock: Whether to use mock embedding generator
        in_memory: Whether to use in-memory vector store
        extensions: File extensions to include
        max_files: Maximum number of files to process
        batch_size: Number of files to process in each batch
        
    Returns:
        Number of documents indexed
    """
    logger.info(f"Indexing documents from {directory}")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    
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
        logger.info("Using ChromaStore")
    
    # Initialize vector store
    try:
        await vector_store.init()
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
    
    try:
        await retriever.init()
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Get list of files
    file_paths = list_documents(directory, extensions=extensions)
    
    if max_files is not None and len(file_paths) > max_files:
        logger.info(f"Limiting to {max_files} files")
        file_paths = file_paths[:max_files]
    
    # Process and index documents in batches
    total_documents = 0
    
    # Process files in batches to manage memory
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
        
        for file_path in batch:
            try:
                logger.info(f"Processing file: {file_path}")
                
                # Load and chunk document
                documents = doc_processor.load_document(file_path)
                logger.info(f"Loaded document {file_path}: {len(documents)} chunks")
                
                # Add to vector store
                doc_ids = await retriever.add_documents(documents)
                
                logger.info(f"Indexed {len(doc_ids)} chunks from {file_path}")
                total_documents += len(doc_ids)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Get stats
    try:
        stats = await retriever.get_stats()
        logger.info(f"Indexing complete. Total documents: {stats['document_count']}")
    except Exception as e:
        logger.error(f"Error getting retriever stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return total_documents

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Index documents into knowledge base')
    parser.add_argument('--directory', type=str, required=True,
                      help='Directory containing documents to index')
    parser.add_argument('--use-mock', action='store_true',
                      help='Use mock embedding generator (faster but less accurate)')
    parser.add_argument('--in-memory', action='store_true',
                      help='Use in-memory vector store (no Qdrant required)')
    parser.add_argument('--extensions', type=str, nargs='+',
                      help='File extensions to include (e.g., .pdf .txt)')
    parser.add_argument('--max-files', type=int,
                      help='Maximum number of files to process')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Number of files to process in each batch (default: 3)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    
    try:
        # Convert extensions to list if provided
        extensions = args.extensions if args.extensions else None
        
        # Index documents
        total_docs = await index_documents(
            directory=args.directory,
            use_mock=args.use_mock,
            in_memory=args.in_memory,
            extensions=extensions,
            max_files=args.max_files,
            batch_size=args.batch_size
        )
        
        print(f"\nSuccessfully indexed {total_docs} document chunks.")
        return 0
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)