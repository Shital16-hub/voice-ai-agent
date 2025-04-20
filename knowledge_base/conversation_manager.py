"""
Conversation manager using LangGraph for state management.
"""
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from enum import Enum
import time

from .retriever import Retriever
from .document_processor import Document

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Enum for conversation states."""
    GREETING = "greeting"
    WAITING_FOR_QUERY = "waiting_for_query"
    RETRIEVING = "retrieving"
    GENERATING_RESPONSE = "generating_response"
    CLARIFYING = "clarifying"
    HUMAN_HANDOFF = "human_handoff"
    ENDED = "ended"

class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    def __init__(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        state: ConversationState = ConversationState.WAITING_FOR_QUERY,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ConversationTurn.
        
        Args:
            query: User query
            response: System response
            retrieved_context: Retrieved documents
            state: Conversation state
            metadata: Additional metadata
        """
        self.query = query
        self.response = response
        self.retrieved_context = retrieved_context or []
        self.state = state
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "retrieved_context": self.retrieved_context,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(
            query=data.get("query"),
            response=data.get("response"),
            retrieved_context=data.get("retrieved_context", []),
            state=data.get("state", ConversationState.WAITING_FOR_QUERY),
            metadata=data.get("metadata", {})
        )

class ConversationManager:
    """
    Manage conversation state and flow using LangGraph.
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        language_model_callback: Optional[Callable] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize ConversationManager.
        
        Args:
            retriever: Retriever instance
            language_model_callback: Callback for language model
            session_id: Unique session identifier
        """
        self.retriever = retriever
        self.language_model_callback = language_model_callback
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Initialize conversation state
        self.current_state = ConversationState.GREETING
        self.history: List[ConversationTurn] = []
        self.context_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Initialized ConversationManager with session_id: {self.session_id}")
    
    async def init(self):
        """Initialize dependencies."""
        if self.retriever:
            await self.retriever.init()
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and move conversation forward.
        
        Args:
            user_input: User input text
            
        Returns:
            Response with next state and response text
        """
        # Create new turn
        turn = ConversationTurn(
            query=user_input,
            state=self.current_state
        )
        
        # Process based on current state
        if self.current_state == ConversationState.GREETING:
            # Handle greeting
            response = await self._handle_greeting(turn)
        elif self.current_state == ConversationState.WAITING_FOR_QUERY:
            # Handle query
            response = await self._handle_query(turn)
        elif self.current_state == ConversationState.CLARIFYING:
            # Handle clarification
            response = await self._handle_clarification(turn)
        elif self.current_state == ConversationState.HUMAN_HANDOFF:
            # Handle already in human handoff
            response = {
                "response": "I'll let the human agent know about your message.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        else:
            # Default handling
            response = await self._handle_query(turn)
        
        # Update turn with response and add to history
        turn.response = response["response"]
        turn.state = response["state"]
        self.history.append(turn)
        
        # Update current state
        self.current_state = response["state"]
        
        return response
    
    async def _handle_greeting(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Handle greeting state.
        
        Args:
            turn: Current conversation turn
            
        Returns:
            Response dictionary
        """
        # Generate greeting response
        if self.language_model_callback:
            greeting_prompt = "Generate a friendly greeting for a customer service conversation."
            
            try:
                greeting_response = await self.language_model_callback(greeting_prompt)
                response_text = greeting_response["response"]
            except Exception as e:
                logger.error(f"Error generating greeting: {e}")
                response_text = "Hello! How can I assist you today?"
        else:
            response_text = "Hello! How can I assist you today?"
        
        # Move to waiting for query
        return {
            "response": response_text,
            "state": ConversationState.WAITING_FOR_QUERY,
            "requires_human": False,
            "context": None
        }
    
    async def _handle_query(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Handle user query.
        
        Args:
            turn: Current conversation turn
            
        Returns:
            Response dictionary
        """
        query = turn.query
        
        # Check for human handoff request
        if self._check_for_human_handoff(query):
            return {
                "response": "I'll connect you with a human agent shortly. Please wait a moment.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        
        # Retrieve relevant documents
        if self.retriever:
            # Set state to retrieving
            turn.state = ConversationState.RETRIEVING
            
            try:
                # Get relevant documents
                retrieval_results = await self.retriever.retrieve_with_sources(query)
                turn.retrieved_context = retrieval_results["results"]
                
                # Format context for LLM
                context = self.retriever.format_retrieved_context(turn.retrieved_context)
                
                # Check if we have enough context
                if not turn.retrieved_context:
                    # No relevant information found
                    if self._should_clarify(query):
                        # Need clarification
                        return {
                            "response": self._generate_clarification_question(query),
                            "state": ConversationState.CLARIFYING,
                            "requires_human": False,
                            "context": None
                        }
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = None
        else:
            context = None
        
        # Generate response
        if self.language_model_callback:
            try:
                # Set state to generating response
                turn.state = ConversationState.GENERATING_RESPONSE
                
                # Get conversation history for context
                conversation_history = self._format_conversation_history()
                
                # Generate response with LLM
                response_data = await self.language_model_callback(
                    query,
                    context=context,
                    conversation_history=conversation_history
                )
                
                response_text = response_data["response"]
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response_text = "I'm sorry, I'm having trouble processing your request right now."
        else:
            response_text = "I would answer your query, but I need language model integration to generate a response."
        
        # Return response
        return {
            "response": response_text,
            "state": ConversationState.WAITING_FOR_QUERY,
            "requires_human": False,
            "context": context
        }
    
    async def _handle_clarification(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Handle clarification response from user.
        
        Args:
            turn: Current conversation turn
            
        Returns:
            Response dictionary
        """
        # Get original query from previous turn
        original_query = self.history[-1].query if self.history else ""
        
        # Combine original query with clarification
        combined_query = f"{original_query} {turn.query}"
        
        # Create new turn with combined query
        new_turn = ConversationTurn(
            query=combined_query,
            state=ConversationState.WAITING_FOR_QUERY
        )
        
        # Handle as normal query
        return await self._handle_query(new_turn)
    
    def _check_for_human_handoff(self, query: str) -> bool:
        """
        Check if user is requesting human handoff.
        
        Args:
            query: User query
            
        Returns:
            True if human handoff requested
        """
        # Simple keyword matching
        handoff_keywords = [
            "speak to a human",
            "talk to a person",
            "talk to someone",
            "speak to an agent",
            "connect me with",
            "real person",
            "human agent",
            "customer service",
            "representative"
        ]
        
        query_lower = query.lower()
        for keyword in handoff_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    def _should_clarify(self, query: str) -> bool:
        """
        Determine if we need clarification for the query.
        
        Args:
            query: User query
            
        Returns:
            True if clarification needed
        """
        # Check query length
        if len(query.split()) < 3:
            return True
        
        # Check for vagueness
        vague_terms = ["this", "that", "it", "thing", "stuff", "something"]
        query_lower = query.lower()
        for term in vague_terms:
            if term in query_lower.split():
                return True
        
        return False
    
    def _generate_clarification_question(self, query: str) -> str:
        """
        Generate a clarification question.
        
        Args:
            query: Original query
            
        Returns:
            Clarification question
        """
        # Simple template-based generation
        templates = [
            "Could you please provide more details about what you're looking for?",
            "I'd like to help, but I need a bit more information. Can you elaborate on your question?",
            "To better assist you, could you be more specific about what you need?",
            "I'm not sure I understand completely. Could you explain what you're looking for in more detail?",
            "Could you clarify what specifically you'd like to know about this topic?"
        ]
        
        import random
        return random.choice(templates)
    
    def _format_conversation_history(self) -> List[Dict[str, str]]:
        """
        Format conversation history for language model.
        
        Returns:
            Formatted conversation history
        """
        formatted_history = []
        
        # Add recent turns (up to last 5 turns)
        for turn in self.history[-5:]:
            if turn.query:
                formatted_history.append({
                    "role": "user",
                    "content": turn.query
                })
            
            if turn.response:
                formatted_history.append({
                    "role": "assistant",
                    "content": turn.response
                })
        
        return formatted_history
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return [turn.to_dict() for turn in self.history]
    
    def get_latest_context(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get most recently retrieved context.
        
        Returns:
            Retrieved context documents or None
        """
        # Find most recent turn with context
        for turn in reversed(self.history):
            if turn.retrieved_context:
                return turn.retrieved_context
        
        return None
    
    def reset(self):
        """Reset conversation state."""
        self.current_state = ConversationState.GREETING
        self.history = []
        self.context_cache = {}
        
        logger.info(f"Reset conversation for session: {self.session_id}")
    
    def get_state_for_transfer(self) -> Dict[str, Any]:
        """
        Get conversation state for human handoff.
        
        Returns:
            Dictionary with conversation state for transfer
        """
        # Create transfer state with relevant information
        transfer_state = {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history_summary": self._generate_history_summary(),
            "last_query": self.history[-1].query if self.history else None,
            "last_response": self.history[-1].response if self.history else None,
            "recent_context": self.get_latest_context()
        }
        
        return transfer_state
    
    def _generate_history_summary(self) -> str:
        """
        Generate a summary of conversation history.
        
        Returns:
            Summary text
        """
        if not self.history:
            return "No conversation history."
        
        # Count turns
        num_turns = len(self.history) // 2
        
        # Get key exchanges
        summary_parts = [f"Conversation with {num_turns} exchanges:"]
        
        for i, turn in enumerate(self.history):
            if turn.query:
                summary_parts.append(f"User: {turn.query}")
            if turn.response:
                # Truncate long responses
                response = turn.response
                if len(response) > 100:
                    response = response[:97] + "..."
                summary_parts.append(f"AI: {response}")
        
        return "\n".join(summary_parts)