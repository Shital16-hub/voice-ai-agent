#!/usr/bin/env python3
"""
Test script for the updated document processor.
"""
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the updated document processor
from document_processor import DocumentProcessor

def test_document_processor(file_path):
    """Test document processing on a single file."""
    try:
        logger.info(f"Testing document processor on file: {file_path}")
        
        # Create document processor
        processor = DocumentProcessor()
        
        # Process document
        documents = processor.load_document(file_path)
        
        # Print results
        logger.info(f"Successfully processed document into {len(documents)} chunks")
        for i, doc in enumerate(documents):
            logger.info(f"Chunk {i+1}: {doc.text[:50]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_updated_processor.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = test_document_processor(file_path)
    
    if success:
        print("\nDocument processing successful!")
        sys.exit(0)
    else:
        print("\nDocument processing failed!")
        sys.exit(1)