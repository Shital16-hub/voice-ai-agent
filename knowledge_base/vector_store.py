"""
Vector database integration for storing and retrieving embeddings.
"""
import logging
import asyncio
import time
import os
from typing import List, Dict, Any, Optional, Tuple, Union

from .config import get_vector_db_config
from .document_processor import Document

logger = logging.getLogger(__name__)

class ChromaStore:
    """
    ChromaDB vector store for storing and retrieving document embeddings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaStore.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_vector_db_config()
        self.collection_name = self.config["collection_name"]
        self.vector_size = self.config["vector_size"]
        
        self.client = None
        self.collection = None
        
        # Create persistence directory if it doesn't exist
        self.persist_dir = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        logger.info(f"Initialized ChromaStore with collection: {self.collection_name}")
    
    async def init(self):
        """Initialize connection and ensure collection exists."""
        if self.client is None:
            try:
                import chromadb
                
                # Create persistent client
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                
                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name
                )
                
                logger.info(f"Connected to ChromaDB, collection: {self.collection_name}, persistent at {self.persist_dir}")
            except ImportError:
                raise ImportError("chromadb is not installed. "
                               "Please install it with: pip install chromadb")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of document dictionaries with text, metadata, and embedding
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        doc_ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = str(doc["id"])  # Ensure ID is string
            embedding = doc["embedding"]
            text = doc["text"]
            metadata = doc["metadata"]
            
            doc_ids.append(doc_id)
            embeddings.append(embedding)
            texts.append(text)
            metadatas.append(metadata)
        
        try:
            # Add in batches to ChromaDB (upsert to handle duplicates)
            self.collection.upsert(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection {self.collection_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 3,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k * 2, 10)  # Get extra results to filter by score
            )
            
            # Process results
            formatted_results = []
            
            # Check if we have any results
            if results["ids"] and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Extract document data
                    text = results["documents"][0][i] if "documents" in results and results["documents"] else ""
                    metadata = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {}
                    distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                    
                    # Convert distance to similarity score (ChromaDB returns distances, not similarities)
                    # For cosine distance, similarity = 1 - distance
                    score = 1.0 - min(1.0, max(0.0, distance))
                    
                    # Skip if below threshold
                    if score < min_score:
                        continue
                    
                    # Create document
                    document = {
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "score": score
                    }
                    
                    formatted_results.append(document)
            
            # Sort by score and limit to top_k
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        try:
            # Get by ID
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            # Check if we have any results
            if not result["ids"] or not result["ids"]:
                return None
            
            # Extract document data
            text = result["documents"][0] if "documents" in result and result["documents"] else ""
            metadata = result["metadatas"][0] if "metadatas" in result and result["metadatas"] else {}
            
            # Create document
            document = {
                "id": doc_id,
                "text": text,
                "metadata": metadata
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents by ID.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Number of documents deleted
        """
        try:
            # Delete documents
            self.collection.delete(ids=doc_ids)
            
            logger.info(f"Deleted {len(doc_ids)} documents from collection {self.collection_name}")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0
    
    async def count_documents(self) -> int:
        """
        Count documents in collection.
        
        Returns:
            Number of documents
        """
        try:
            # Get collection info
            count = self.collection.count()
            return count
            
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def reset_collection(self) -> bool:
        """
        Reset collection by recreating it.
        
        Returns:
            True if successful
        """
        try:
            # Delete collection
            self.client.delete_collection(self.collection_name)
            
            # Create collection
            self.collection = self.client.create_collection(
                name=self.collection_name
            )
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False


class InMemoryVectorStore:
    """
    In-memory vector store for testing and development.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize InMemoryVectorStore.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_vector_db_config()
        self.vector_size = self.config["vector_size"]
        self.collection_name = self.config["collection_name"]
        
        # Storage for documents
        self.documents = {}
        
        logger.info(f"Initialized InMemoryVectorStore")
    
    async def init(self):
        """Initialize (no-op for in-memory)."""
        pass
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a_array = np.array(a)
        b_array = np.array(b)
        
        # Cosine similarity
        similarity = np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))
        
        return float(similarity)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to in-memory store.
        
        Args:
            documents: List of document dictionaries with text, metadata, and embedding
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        for doc in documents:
            doc_id = doc["id"]
            self.documents[doc_id] = {
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "embedding": doc["embedding"]
            }
            doc_ids.append(doc_id)
        
        logger.info(f"Added {len(documents)} documents to in-memory store")
        return doc_ids
    
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 3,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of matching documents
        """
        results = []
        
        # Calculate similarity for each document
        for doc_id, doc in self.documents.items():
            score = self._cosine_similarity(query_vector, doc["embedding"])
            
            if score >= min_score:
                results.append({
                    "id": doc_id,
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        
        # Sort by score (descending) and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        doc = self.documents.get(doc_id)
        
        if doc:
            return {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"]
            }
        
        return None
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents by ID.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Number of documents deleted
        """
        deleted = 0
        
        for doc_id in doc_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted += 1
        
        logger.info(f"Deleted {deleted} documents from in-memory store")
        return deleted
    
    async def count_documents(self) -> int:
        """
        Count documents in store.
        
        Returns:
            Number of documents
        """
        return len(self.documents)
    
    async def reset_collection(self) -> bool:
        """
        Reset store by clearing all documents.
        
        Returns:
            True if successful
        """
        self.documents.clear()
        logger.info("Reset in-memory store")
        return True

# Default to ChromaStore instead of Qdrant's VectorStore
VectorStore = ChromaStore