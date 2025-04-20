"""
Retriever for fetching relevant documents from the knowledge base.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

from .config import get_retriever_config
from .document_processor import Document
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)

class Retriever:
    """
    Retrieve relevant documents from the knowledge base.
    """
    
    def __init__(
        self,
        vector_store: Optional[Union[VectorStore, InMemoryVectorStore]] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            config: Optional configuration dictionary
        """
        self.config = config or get_retriever_config()
        self.top_k = self.config["top_k"]
        self.min_score = self.config["min_score"]
        self.reranking_enabled = self.config["reranking_enabled"]
        
        # Initialize components
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        logger.info(f"Initialized Retriever with top_k={self.top_k}, min_score={self.min_score}")
    
    async def init(self):
        """Initialize components."""
        await self.vector_store.init()
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides config)
            min_score: Minimum similarity score (overrides config)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        # Use provided values or defaults
        top_k = top_k if top_k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                min_score=min_score
            )
            
            # Apply metadata filtering if provided
            if filter_metadata and results:
                filtered_results = []
                for doc in results:
                    metadata = doc["metadata"]
                    match = True
                    
                    # Check all filter conditions
                    for key, value in filter_metadata.items():
                        if key not in metadata or metadata[key] != value:
                            match = False
                            break
                    
                    if match:
                        filtered_results.append(doc)
                
                results = filtered_results
            
            # Apply reranking if enabled
            if self.reranking_enabled and len(results) > 1:
                results = self._rerank_results(query, results)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using a more sophisticated relevance model.
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        try:
            # Try to use cross-encoder if available
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare sentence pairs
            sentence_pairs = [[query, doc["text"]] for doc in results]
            
            # Get scores
            scores = model.predict(sentence_pairs)
            
            # Update scores
            for i, score in enumerate(scores):
                results[i]["score"] = float(score)
            
            # Sort by new scores
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
        
        except ImportError:
            logger.warning("CrossEncoder not available. Skipping reranking.")
            return results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results
    
    async def retrieve_with_sources(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents with source information.
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            Dictionary with results and sources
        """
        # Retrieve documents
        docs = await self.retrieve(query, top_k, min_score)
        
        if not docs:
            return {
                "query": query,
                "results": [],
                "sources": []
            }
        
        # Extract unique sources
        sources = []
        source_ids = set()
        
        for doc in docs:
            metadata = doc["metadata"]
            source = metadata.get("source")
            
            if source and source not in source_ids:
                source_ids.add(source)
                
                # Add source info
                source_info = {
                    "name": source,
                    "type": metadata.get("source_type", "unknown")
                }
                
                # Add file info if available
                if metadata.get("file_path"):
                    source_info["file_path"] = metadata.get("file_path")
                    source_info["file_type"] = metadata.get("file_type")
                
                sources.append(source_info)
        
        return {
            "query": query,
            "results": docs,
            "sources": sources
        }
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string for LLM.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, doc in enumerate(results):
            # Extract info
            text = doc["text"]
            score = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"Source {i+1}")
            
            # Add to context
            context_parts.append(f"[Document {i+1}] Source: {source} (Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(context_parts)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        # Generate embeddings
        doc_dicts = self.embedding_generator.embed_documents(documents)
        
        # Add to vector store
        doc_ids = await self.vector_store.add_documents(doc_dicts)
        
        return doc_ids
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with statistics
        """
        doc_count = await self.vector_store.count_documents()
        
        return {
            "document_count": doc_count,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "reranking_enabled": self.reranking_enabled
        }