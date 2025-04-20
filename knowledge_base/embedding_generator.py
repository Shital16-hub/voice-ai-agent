"""
Embedding generator for creating vector representations of documents.
"""
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import time

from .config import get_embedding_config
from .document_processor import Document

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate embeddings for documents using a sentence transformer model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_embedding_config()
        self.model_name = self.config["model_name"]
        self.device = self.config["device"]
        self.batch_size = self.config["batch_size"]
        self.dimension = self.config["dimension"]
        
        self.model = None
        logger.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")
        
        # Lazy initialization of model to avoid loading at import time
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded."""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
                
                # Check if sentence_transformers is installed
                try:
                    import sentence_transformers
                    logger.info(f"Using sentence_transformers version: {sentence_transformers.__version__}")
                except ImportError:
                    logger.error("sentence-transformers not installed!")
                    raise ImportError("sentence-transformers is not installed. "
                                "Please install it with: pip install sentence-transformers")
                
                # Check for PyTorch
                try:
                    import torch
                    logger.info(f"PyTorch version: {torch.__version__}")
                    logger.info(f"CUDA available: {torch.cuda.is_available()}")
                    if self.device == "cuda" and not torch.cuda.is_available():
                        logger.warning("CUDA requested but not available, falling back to CPU")
                        self.device = "cpu"
                except ImportError:
                    logger.warning("PyTorch not installed or CUDA not available")
                
                # Load the model with timing
                start_time = time.time()
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                load_time = time.time() - start_time
                logger.info(f"Loaded embedding model: {self.model_name} on {self.device} in {load_time:.2f}s")
                
            except ImportError as ie:
                logger.error(f"Import error loading model: {ie}")
                raise
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        self._ensure_model_loaded()
        
        try:
            logger.debug(f"Generating embedding for text: {text[:50]}...")
            
            # Generate embedding
            embedding = self.model.encode(text, show_progress_bar=False)
            
            # Convert numpy array to list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        self._ensure_model_loaded()
        
        if not texts:
            logger.warning("Empty list of texts provided to generate_embeddings")
            return []
            
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Generate embeddings with progress bar for larger batches
            start_time = time.time()
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 10
            )
            generation_time = time.time() - start_time
            
            # Convert numpy arrays to lists
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
            logger.info(f"Generated {len(embeddings_list)} embeddings in {generation_time:.2f}s")
            return embeddings_list
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of dictionaries with document data and embeddings
        """
        if not documents:
            logger.warning("Empty list of documents provided to embed_documents")
            return []
            
        # Extract texts
        texts = [doc.text for doc in documents]
        logger.info(f"Embedding {len(texts)} documents")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Combine documents with embeddings
        result = []
        for doc, embedding in zip(documents, embeddings):
            doc_dict = doc.to_dict()
            doc_dict["embedding"] = embedding
            result.append(doc_dict)
        
        return result

# No changes needed for MockEmbeddingGenerator
class MockEmbeddingGenerator:
    """
    Mock embedding generator for testing without dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MockEmbeddingGenerator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_embedding_config()
        self.dimension = self.config["dimension"]
        
        logger.info(f"Initialized MockEmbeddingGenerator with dimension: {self.dimension}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a mock embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of mock embedding values
        """
        # Generate a deterministic but unique vector based on the text
        import hashlib
        
        # Get hash of text
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()
        
        # Use hash to seed random generator
        import random
        random.seed(hash_hex)
        
        # Generate random vector with consistent dimensions
        vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
        
        # Normalize to unit length
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x/magnitude for x in vector]
        
        return vector
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of mock embeddings
        """
        return [self.generate_embedding(text) for text in texts]
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate mock embeddings for a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of dictionaries with document data and mock embeddings
        """
        # Extract texts
        texts = [doc.text for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Combine documents with embeddings
        result = []
        for doc, embedding in zip(documents, embeddings):
            doc_dict = doc.to_dict()
            doc_dict["embedding"] = embedding
            result.append(doc_dict)
        
        return result