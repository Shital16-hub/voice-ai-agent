"""
LangGraph-based Voice AI Agent.

This module provides the main implementation of the Voice AI Agent
using LangGraph for orchestration, enabling more flexible and
powerful conversation flows.
"""
import os
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union, Tuple, AsyncIterator

import numpy as np
from langgraph.graph import StateGraph
# Update END to whatever is used in your version of LangGraph
try:
    from langgraph.graph import END
except ImportError:
    # If END is not defined in your version of LangGraph, use a string identifier
    END = "end"

from voice_ai_agent import VoiceAIAgent
from integration.tts_integration import TTSIntegration
from integration.kb_integration import KnowledgeBaseIntegration
from integration.stt_integration import STTIntegration

from langgraph_integration.nodes import (
    STTNode, 
    KBNode, 
    TTSNode, 
    AgentState, 
    NodeType, 
    ConversationStatus
)
from langgraph_integration.utils import (
    create_initial_state,
    save_state_history,
    should_handoff_to_human,
    StateTracker
)
from langgraph_integration.config import LangGraphConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class VoiceAILangGraph:
    """
    LangGraph-based Voice AI Agent.
    
    This class provides a LangGraph implementation of the Voice AI Agent,
    enabling more flexible and powerful conversation flows.
    """
    
    def __init__(
        self,
        voice_ai_agent: Optional[VoiceAIAgent] = None,
        stt_integration: Optional[STTIntegration] = None,
        kb_integration: Optional[KnowledgeBaseIntegration] = None,
        tts_integration: Optional[TTSIntegration] = None,
        config: Optional[LangGraphConfig] = None
    ):
        """
        Initialize the LangGraph-based Voice AI Agent.
        
        Args:
            voice_ai_agent: Existing VoiceAIAgent to use for components
            stt_integration: STT integration
            kb_integration: KB integration
            tts_integration: TTS integration
            config: Configuration for the LangGraph
        """
        self.voice_ai_agent = voice_ai_agent
        self.stt_integration = stt_integration
        self.kb_integration = kb_integration
        self.tts_integration = tts_integration
        self.config = config or DEFAULT_CONFIG
        
        # Nodes
        self.stt_node = None
        self.kb_node = None
        self.tts_node = None
        
        # Graph
        self.graph = None
        self.compiled_graph = None
        
        # State tracking
        self.state_tracker = StateTracker(self.config.state_history_path if self.config.save_state_history else None)
        
        # Telephony callbacks
        self.audio_callback = None
    
    async def init(self) -> None:
        """Initialize the LangGraph and all components."""
        logger.info("Initializing VoiceAILangGraph")
        
        # Initialize the base agent if provided and needed
        if self.voice_ai_agent and (not self.voice_ai_agent.speech_recognizer or
                                  not self.voice_ai_agent.query_engine or
                                  not self.voice_ai_agent.conversation_manager):
            await self.voice_ai_agent.init()
        
        # Initialize nodes
        await self._init_nodes()
        
        # Create the graph
        self._create_graph()
        
        logger.info("VoiceAILangGraph initialization complete")
    
    async def _init_nodes(self) -> None:
        """Initialize all nodes for the graph."""
        # STT Node
        if self.stt_integration:
            self.stt_node = STTNode(stt_integration=self.stt_integration)
        elif self.voice_ai_agent and self.voice_ai_agent.speech_recognizer:
            self.stt_node = STTNode(speech_recognizer=self.voice_ai_agent.speech_recognizer)
        else:
            self.stt_node = STTNode(model_path=self.config.stt_model, language=self.config.stt_language)
            await self.stt_node.stt.init(model_path=self.config.stt_model)
        
        # KB Node
        if self.kb_integration:
            self.kb_node = KBNode(kb_integration=self.kb_integration)
        elif self.voice_ai_agent and self.voice_ai_agent.query_engine and self.voice_ai_agent.conversation_manager:
            self.kb_node = KBNode(
                query_engine=self.voice_ai_agent.query_engine,
                conversation_manager=self.voice_ai_agent.conversation_manager,
                temperature=self.config.kb_temperature,
                max_tokens=self.config.kb_max_tokens,
                include_sources=self.config.kb_include_sources
            )
        else:
            raise ValueError("Either kb_integration or voice_ai_agent with query_engine and conversation_manager must be provided")
        
        # TTS Node with optional callback
        if self.tts_integration:
            self.tts_node = TTSNode(
                tts_integration=self.tts_integration,
                output_callback=self.audio_callback
            )
        else:
            self.tts_node = TTSNode(
                voice=self.config.tts_voice,
                output_callback=self.audio_callback
            )
            await self.tts_node.tts.init()
    
    def _create_graph(self) -> None:
        """Create the LangGraph."""
        # Create state graph
        self.graph = StateGraph(AgentState)
        
        # Add nodes
        self.graph.add_node("stt", self.stt_node.process)
        self.graph.add_node("kb", self.kb_node.process)
        self.graph.add_node("tts", self.tts_node.process)
        
        # Add conditional edges
        self.graph.add_conditional_edges(
            "stt",
            self._route_from_stt,
            {
                "kb": "kb",
                "error": END
            }
        )
        
        self.graph.add_conditional_edges(
            "kb",
            self._route_from_kb,
            {
                "tts": "tts",
                "error": END
            }
        )
        
        self.graph.add_conditional_edges(
            "tts",
            self._route_from_tts,
            {
                "end": END,
                "error": END
            }
        )
        
        # Set entry point
        self.graph.set_entry_point("stt")
        
        # Compile the graph
        self.compiled_graph = self.graph.compile()
    
    def _route_from_stt(self, state: AgentState) -> str:
        """
        Route from STT node based on state.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        if state.error or state.status == ConversationStatus.ERROR:
            return "error"
        return "kb"
    
    def _route_from_kb(self, state: AgentState) -> str:
        """
        Route from KB node based on state.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        if state.error or state.status == ConversationStatus.ERROR:
            return "error"
        return "tts"
    
    def _route_from_tts(self, state: AgentState) -> str:
        """
        Route from TTS node based on state.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        if state.error or state.status == ConversationStatus.ERROR:
            return "error"
        return "end"
    
    def set_audio_callback(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """
        Set a callback for audio output.
        
        This is useful for integration with telephony systems.
        
        Args:
            callback: Async function that receives audio data
        """
        self.audio_callback = callback
        
        # Update TTS node if already created
        if self.tts_node:
            self.tts_node.output_callback = callback
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to audio file
            speech_output_path: Path to save speech output
            metadata: Additional metadata
            
        Returns:
            Results dictionary
        """
        if not self.compiled_graph:
            await self.init()
        
        # Create initial state
        state = create_initial_state(
            audio_file_path=audio_file_path,
            speech_output_path=speech_output_path,
            metadata=metadata
        )
        
        # Run the graph
        try:
            final_state = await self.compiled_graph.ainvoke(state)
            
            # Save state history if enabled
            if self.config.save_state_history:
                await self.state_tracker.save_history()
            
            # Prepare results
            results = self._prepare_results(final_state)
            return results
            
        except Exception as e:
            logger.error(f"Error running LangGraph: {e}")
            
            # Create error results
            error_results = {
                "error": str(e),
                "status": "ERROR",
                "total_time": time.time() - state.timings.get("start_time", time.time())
            }
            
            # Add available state info if possible
            if hasattr(state, "transcription") and state.transcription:
                error_results["transcription"] = state.transcription
                
            if hasattr(state, "response") and state.response:
                error_results["response"] = state.response
                
            return error_results
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data through the complete pipeline.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            speech_output_path: Path to save speech output
            metadata: Additional metadata
            
        Returns:
            Results dictionary
        """
        if not self.compiled_graph:
            await self.init()
        
        # Create initial state
        state = create_initial_state(
            audio_input=audio_data,
            speech_output_path=speech_output_path,
            metadata=metadata
        )
        
        # Run the graph
        try:
            final_state = await self.compiled_graph.ainvoke(state)
            
            # Save state history if enabled
            if self.config.save_state_history:
                await self.state_tracker.save_history()
            
            # Prepare results
            results = self._prepare_results(final_state)
            return results
            
        except Exception as e:
            logger.error(f"Error running LangGraph: {e}")
            
            # Create error results
            error_results = {
                "error": str(e),
                "status": "ERROR",
                "total_time": time.time() - state.timings.get("start_time", time.time())
            }
            
            # Add available state info if possible
            if hasattr(state, "transcription") and state.transcription:
                error_results["transcription"] = state.transcription
                
            if hasattr(state, "response") and state.response:
                error_results["response"] = state.response
                
            return error_results

    def _get_status_name(self, state) -> str:
        """
        Safely get the status name from a state object.
        
        Args:
            state: Agent state
            
        Returns:
            Status name as string
        """
        if hasattr(state, "status"):
            status = state.status
            if hasattr(status, "name"):
                return status.name
            return str(status)
        return "UNKNOWN"    
    
    async def process_text(
        self,
        text: str,
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process text input through the pipeline (skipping STT).
        
        Args:
            text: Text input
            speech_output_path: Path to save speech output
            metadata: Additional metadata
            
        Returns:
            Results dictionary
        """
        if not self.compiled_graph:
            await self.init()
        
        # Create initial state
        state = create_initial_state(
            text_input=text,
            speech_output_path=speech_output_path,
            metadata=metadata
        )
        
        # Run the graph
        try:
            final_state = await self.compiled_graph.ainvoke(state)
            
            # Save state history if enabled
            if self.config.save_state_history:
                await self.state_tracker.save_history()
            
            # Prepare results
            results = self._prepare_results(final_state)
            return results
            
        except Exception as e:
            logger.error(f"Error running LangGraph: {e}")
            
            # Create error results
            error_results = {
                "error": str(e),
                "status": "ERROR",
                "total_time": time.time() - state.timings.get("start_time", time.time())
            }
            
            # Add available state info if possible
            if hasattr(state, "transcription") and state.transcription:
                error_results["transcription"] = state.transcription
                
            if hasattr(state, "response") and state.response:
                error_results["response"] = state.response
                
            return error_results
    
    async def process_streaming(
        self,
        stream_input: Union[AsyncIterator[np.ndarray], AsyncIterator[bytes], str],
        is_text: bool = False,
        speech_output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming input with streaming output.
        
        Args:
            stream_input: Audio stream or text
            is_text: Whether the input is text
            speech_output_path: Path to save final speech output
            metadata: Additional metadata
            
        Yields:
            Results for each processing step
        """
        # Not fully implemented yet - would need more complex streaming graph
        raise NotImplementedError("Streaming processing not implemented yet")
    
    def _prepare_results(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare results dictionary from final state.
        
        Args:
            state: Final agent state
            
        Returns:
            Results dictionary
        """
        # Create results dictionary
        results = {
            "status": self._get_status_name(state),
            "timings": getattr(state, "timings", {}),
            "total_time": time.time() - getattr(state, "timings", {}).get("start_time", time.time())
        }
        
        # Add transcription if present
        if hasattr(state, "transcription") and state.transcription:
            results["transcription"] = state.transcription
            
        # Add response if present
        if hasattr(state, "response") and state.response:
            results["response"] = state.response
            
        # Add error if present
        if hasattr(state, "error") and state.error:
            results["error"] = state.error
        
        # Add speech output info if applicable
        if hasattr(state, "speech_output") and state.speech_output:
            results["speech_audio_size"] = len(state.speech_output)
            # Exclude binary data from result
            results["speech_output"] = None
        
        if hasattr(state, "speech_output_path") and state.speech_output_path:
            results["speech_output_path"] = state.speech_output_path
        
        # Add conversation info
        if hasattr(state, "conversation_id") and state.conversation_id:
            results["conversation_id"] = state.conversation_id
        
        # Add sources if available
        if hasattr(state, "sources") and state.sources:
            results["sources"] = state.sources
        
        return results
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up VoiceAILangGraph")
        
        # Clean up nodes
        if hasattr(self.stt_node, 'cleanup'):
            await self.stt_node.cleanup()
        
        if hasattr(self.kb_node, 'cleanup'):
            await self.kb_node.cleanup()
        
        if hasattr(self.tts_node, 'cleanup'):
            await self.tts_node.cleanup()