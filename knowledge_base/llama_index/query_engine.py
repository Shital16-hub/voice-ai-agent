"""
Query engine for retrieving information from the vector index.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from llama_index.core.schema import Node, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from knowledge_base.config import get_retriever_config
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.schema import Document

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Retrieve and process information from the knowledge base.
    """
    
    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize QueryEngine.
        
        Args:
            index_manager: IndexManager instance
            config: Optional configuration dictionary
        """
        self.index_manager = index_manager
        self.config = config or get_retriever_config()
        self.top_k = self.config["top_k"]
        self.min_score = self.config["min_score"]
        
        self.retriever = None
        self.query_engine = None
        self.is_initialized = False
        
        logger.info(f"Initialized QueryEngine with top_k={self.top_k}, min_score={self.min_score}")
    
    async def init(self):
        """Initialize the query engine."""
        if self.is_initialized:

            return
    
        # Ensure index manager is initialized
        if not self.index_manager.is_initialized:
            await self.index_manager.init()
    
        # Create retriever
        self.retriever = VectorIndexRetriever(
        index=self.index_manager.index,
        similarity_top_k=self.top_k,
        filters=None
        )
    
        # Create response synthesizer without an LLM
        from llama_index.core.response_synthesizers import ResponseMode
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.NO_TEXT,
            llm=None  # Explicitly set to None to avoid OpenAI dependency
        )
    
    # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer
        )
    
        self.is_initialized = True
        logger.info("Query engine initialized")
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides default)
            min_score: Minimum similarity score (overrides default)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        if not self.is_initialized:
            await self.init()
        
        # Use provided values or defaults
        top_k = top_k if top_k is not None else self.top_k
        
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Update retriever if needed
            if top_k != self.retriever.similarity_top_k:
                self.retriever.similarity_top_k = top_k
            
            # Apply metadata filters if provided
            if filter_metadata:
                filters = {"metadata_filter": filter_metadata}
                self.retriever.filters = filters
            else:
                self.retriever.filters = None
            
            # Retrieve nodes
            nodes = self.retriever.retrieve(query_bundle)
            
            # Convert to Document objects and filter by score if needed
            documents = []
            for node in nodes:
                # Skip if below minimum score (if provided)
                if min_score is not None and node.score < min_score:
                    continue
                
                doc = Document(
                    text=node.text,
                    metadata=node.metadata,
                    doc_id=node.id_
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
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
        
        # Process documents for result format
        results = []
        for doc in docs:
            metadata = doc.metadata
            
            # Add to results
            result = {
                "id": doc.doc_id,
                "text": doc.text,
                "metadata": metadata,
                "score": metadata.get("score", 0.0)
            }
            results.append(result)
            
            # Extract source information
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
            "results": results,
            "sources": sources
        }
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string.
        
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
    
    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query_text: Query text
            
        Returns:
            Dictionary with response and source information
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Run query through the LlamaIndex query engine
            response = self.query_engine.query(query_text)
            
            # Get source nodes
            source_nodes = response.source_nodes
            
            # Extract source information
            sources = []
            for node in source_nodes:
                source = {
                    "text": node.text,
                    "score": node.score if hasattr(node, "score") else 0.0,
                    "metadata": node.metadata
                }
                sources.append(source)
            
            # Format response
            result = {
                "query": query_text,
                "response": str(response),
                "sources": sources
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "query": query_text,
                "response": "Error querying knowledge base.",
                "sources": []
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with statistics
        """
        doc_count = await self.index_manager.count_documents()
        
        return {
            "document_count": doc_count,
            "top_k": self.top_k,
            "min_score": self.min_score
        }