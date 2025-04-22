"""
Helper functions for the LangGraph integration.

This module provides utility functions for working with
the LangGraph-based Voice AI Agent.
"""
import os
import json
import time
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union

import numpy as np

from langgraph_integration.nodes.state import AgentState, ConversationStatus

logger = logging.getLogger(__name__)

def create_initial_state(
    audio_input: Optional[Union[bytes, np.ndarray]] = None,
    audio_file_path: Optional[str] = None,
    text_input: Optional[str] = None,
    conversation_id: Optional[str] = None,
    speech_output_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Create an initial state for the LangGraph.
    
    Args:
        audio_input: Audio input data
        audio_file_path: Path to audio file
        text_input: Direct text input
        conversation_id: Conversation ID (generated if None)
        speech_output_path: Path to save speech output
        metadata: Additional metadata
        
    Returns:
        Initial agent state
    """
    # Validate inputs
    if not any([audio_input, audio_file_path, text_input]):
        raise ValueError("At least one of audio_input, audio_file_path, or text_input must be provided")
    
    # Create state
    state = AgentState(
        audio_input=audio_input,
        audio_file_path=audio_file_path,
        text_input=text_input,
        conversation_id=conversation_id or str(uuid.uuid4()),
        speech_output_path=speech_output_path,
        metadata=metadata or {},
        status=ConversationStatus.IDLE,
        timings={"start_time": time.time()}
    )
    
    return state

async def save_state_history(
    state_history: List[AgentState],
    output_path: str
) -> None:
    """
    Save the state history to a file.
    
    Args:
        state_history: List of agent states
        output_path: Path to save the history
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert states to dictionaries
    state_dicts = []
    for state in state_history:
        # Skip binary data for readability
        state_dict = state.dict(exclude={"audio_input", "speech_output"})
        
        # Add metadata about binary fields
        if state.audio_input is not None:
            if isinstance(state.audio_input, np.ndarray):
                state_dict["audio_input_info"] = f"NumPy array with shape {state.audio_input.shape}"
            else:
                state_dict["audio_input_info"] = f"Binary data with size {len(state.audio_input)} bytes"
        
        if state.speech_output is not None:
            state_dict["speech_output_info"] = f"Binary data with size {len(state.speech_output)} bytes"
        
        state_dicts.append(state_dict)
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(state_dicts, f, indent=2, default=str)

def calculate_confidence(state: AgentState) -> float:
    """
    Calculate overall confidence score for the agent's response.
    
    Args:
        state: Current agent state
        
    Returns:
        Confidence score between 0 and 1
    """
    # Start with transcription confidence if available
    confidence = state.transcription_confidence or 0.8
    
    # Adjust based on other factors
    
    # 1. Short responses might indicate uncertainty
    if state.response and len(state.response.split()) < 5:
        confidence *= 0.8
    
    # 2. No context retrieval might indicate lack of knowledge
    if not state.context:
        confidence *= 0.9
    
    # 3. Error indicators in response
    uncertainty_phrases = ["i'm not sure", "i don't know", "i'm uncertain", "cannot", "unable to"]
    if state.response and any(phrase in state.response.lower() for phrase in uncertainty_phrases):
        confidence *= 0.7
    
    return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1

def should_handoff_to_human(state: AgentState, threshold: float = 0.7) -> bool:
    """
    Determine if the conversation should be handed off to a human.
    
    Args:
        state: Current agent state
        threshold: Confidence threshold for handoff
        
    Returns:
        Whether to hand off to a human
    """
    # Always hand off if explicitly required
    if state.requires_human:
        return True
    
    # Hand off on errors
    if state.error:
        return True
    
    # Hand off on low confidence
    confidence = calculate_confidence(state)
    if confidence < threshold:
        return True
    
    # Check for explicit requests for human in transcription
    human_request_phrases = ["speak to a human", "talk to a person", "speak to a person", "human operator"]
    if state.transcription and any(phrase in state.transcription.lower() for phrase in human_request_phrases):
        return True
    
    return False

class StateTracker:
    """
    Utility class for tracking state changes during graph execution.
    
    This is useful for debugging and analysis.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize the state tracker.
        
        Args:
            save_path: Path to save state history (if None, history is kept in memory only)
        """
        self.history = []
        self.save_path = save_path
    
    def add_state(self, state: AgentState) -> None:
        """
        Add a state to the history.
        
        Args:
            state: Agent state to add
        """
        # Create a copy to avoid reference issues
        state_dict = state.dict()
        self.history.append(state_dict)
    
    async def save_history(self) -> None:
        """Save the state history to a file if save_path is set."""
        if not self.save_path:
            return
        
        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Summary dictionary
        """
        if not self.history:
            return {"error": "No history available"}
        
        first_state = self.history[0]
        last_state = self.history[-1]
        
        return {
            "conversation_id": first_state.get("conversation_id"),
            "start_time": first_state.get("timings", {}).get("start_time"),
            "end_time": time.time(),
            "duration": time.time() - first_state.get("timings", {}).get("start_time", time.time()),
            "status": last_state.get("status"),
            "num_turns": len([s for s in self.history if s.get("current_node") == "STT"]),
            "final_state": last_state
        }