"""
Knowledge base component for the Voice AI Agent.
"""
from knowledge_base.document_processor import Document, DocumentProcessor
from knowledge_base.embedding_generator import EmbeddingGenerator, MockEmbeddingGenerator
from knowledge_base.vector_store import ChromaStore, InMemoryVectorStore
from knowledge_base.retriever import Retriever
from knowledge_base.conversation_manager import ConversationManager, ConversationState, ConversationTurn

__version__ = "0.1.0"

__all__ = [
    "Document",
    "DocumentProcessor",
    "EmbeddingGenerator",
    "MockEmbeddingGenerator",
    "ChromaStore",
    "InMemoryVectorStore",
    "Retriever",
    "ConversationManager",
    "ConversationState",
    "ConversationTurn",
]