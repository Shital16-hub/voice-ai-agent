"""
Knowledge Base node for the LangGraph-based Voice AI Agent.

This module provides the KB node that processes queries
and generates responses within the LangGraph flow.
"""
import time
import logging
from typing import Dict, Any, AsyncIterator, Optional, List

from integration.kb_integration import KnowledgeBaseIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.query_engine import QueryEngine

from langgraph_integration.nodes.state import AgentState, NodeType, ConversationStatus

logger = logging.getLogger(__name__)

class KBNode:
    """
    Knowledge Base node for LangGraph.
    
    This node processes queries and generates responses.
    """
    
    def __init__(
        self,
        kb_integration: Optional[KnowledgeBaseIntegration] = None,
        query_engine: Optional[QueryEngine] = None,
        conversation_manager: Optional[ConversationManager] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        include_sources: bool = True
    ):
        """
        Initialize the KB node.
        
        Args:
            kb_integration: Existing KB integration to use
            query_engine: Query engine to use if creating new integration
            conversation_manager: Conversation manager if creating new integration
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response generation
            include_sources: Whether to include sources in the response
        """
        self.include_sources = include_sources
        
        if kb_integration:
            self.kb = kb_integration
        elif query_engine and conversation_manager:
            self.kb = KnowledgeBaseIntegration(
                query_engine=query_engine,
                conversation_manager=conversation_manager,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError("Either kb_integration or both query_engine and conversation_manager must be provided")
    
    async def process(self, state: AgentState) -> AsyncIterator[AgentState]:
        """
        Process the input state and generate a response.
        
        Args:
            state: The current agent state
            
        Yields:
            Updated agent state with response
        """
        # Update state
        state.current_node = NodeType.KB
        state.status = ConversationStatus.THINKING
        
        # Start timing
        start_time = time.time()
        
        try:
            # Check for query
            if not state.query:
                if state.transcription:
                    state.query = state.transcription
                else:
                    logger.error("No query provided to KB node")
                    state.error = "No query provided to KB node"
                    state.status = ConversationStatus.ERROR
                    yield state
                    return
            
            # Query the knowledge base
            result = await self.kb.query(state.query, include_context=self.include_sources)
            
            # Check for errors
            if "error" in result:
                state.error = result["error"]
                state.status = ConversationStatus.ERROR
                yield state
                return
            
            # Update state with results
            state.response = result.get("response", "")
            if not state.response:
                state.error = "No response generated"
                state.status = ConversationStatus.ERROR
                yield state
                return
            
            # Add context and sources if available
            if self.include_sources:
                state.context = result.get("context", "")
                if "sources" in result:
                    state.sources = result["sources"]
            
            # Update status
            state.status = ConversationStatus.RESPONDING
            state.next_node = NodeType.TTS
            
            # Save timing information
            state.timings["kb"] = time.time() - start_time
            state.timings["retrieval_time"] = result.get("retrieval_time", 0.0)
            state.timings["llm_time"] = result.get("llm_time", 0.0)
            
            # Add to history
            state.history.append({
                "role": "assistant",
                "content": state.response
            })
            
        except Exception as e:
            logger.error(f"Error in KB node: {e}")
            state.error = f"KB error: {str(e)}"
            state.status = ConversationStatus.ERROR
        
        # Return updated state
        yield state
    
    async def cleanup(self):
        """Clean up resources."""
        logger.debug("Cleaning up KB node")
        # Reset conversation state
        if hasattr(self, 'kb') and self.kb:
            self.kb.reset_conversation()