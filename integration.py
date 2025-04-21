"""
Integration layer between knowledge base and language model.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional

from language_model.model_manager import ModelManager
from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

class VoiceAssistantIntegration:
    """
    Integrates knowledge base retrieval with language model generation.
    """
    
    def __init__(
        self,
        language_model: Optional[ModelManager] = None,
        query_engine: Optional[QueryEngine] = None,
        conversation_manager: Optional[ConversationManager] = None,
        storage_dir: str = "./storage",
        model_name: Optional[str] = None
    ):
        """
        Initialize the integration layer.
        
        Args:
            language_model: ModelManager instance
            query_engine: QueryEngine instance
            conversation_manager: ConversationManager instance
            storage_dir: Directory for persistent storage
            model_name: Name of language model to use
        """
        # Initialize language model if not provided
        self.language_model = language_model or ModelManager(
            model_name=model_name,
            scenario="customer_service",
            with_rag=True
        )
        
        # Initialize knowledge base components if not provided
        if query_engine is None:
            # Initialize index manager
            self.index_manager = IndexManager(storage_dir=storage_dir)
            
            # Initialize query engine
            self.query_engine = QueryEngine(index_manager=self.index_manager)
        else:
            self.query_engine = query_engine
            self.index_manager = query_engine.index_manager
        
        # Create language model callback for conversation manager
        async def language_model_callback(
            query: str, 
            context: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None
        ) -> Dict[str, Any]:
            if context:
                # Use RAG if context is provided
                response = await self.language_model.generate_rag_response(
                    user_input=query,
                    context=context,
                    temperature=0.7
                )
            else:
                # Use regular response if no context
                response = await self.language_model.generate_response(
                    user_input=query,
                    temperature=0.7
                )
            
            return response
        
        # Initialize conversation manager if not provided
        self.conversation_manager = conversation_manager or ConversationManager(
            query_engine=self.query_engine,
            language_model_callback=language_model_callback
        )
        
        logger.info("Initialized VoiceAssistantIntegration")
    
    async def init(self):
        """Initialize all components."""
        # Initialize index manager
        await self.index_manager.init()
        
        # Initialize query engine
        await self.query_engine.init()
        
        # Initialize conversation manager
        await self.conversation_manager.init()
        
        logger.info("All components initialized")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query text
            
        Returns:
            Response with text and metadata
        """
        # Process through conversation manager
        response = await self.conversation_manager.handle_user_input(query)
        
        return response
    
    async def interactive_session(self):
        """
        Run an interactive session with text input/output.
        """
        print("\n" + "="*50)
        print("Voice AI Agent Interactive Session")
        print("Type 'exit' or 'quit' to end the session")
        print("="*50 + "\n")
        
        # Get stats
        stats = await self.query_engine.get_stats()
        print(f"Knowledge base contains {stats['document_count']} documents")
        
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Process through conversation manager
            response = await self.process_query(user_input)
            
            # Print response
            print(f"AI: {response['response']}")