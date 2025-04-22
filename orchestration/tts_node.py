"""
LangGraph node for Text-to-Speech integration.

This module provides the TTS node for the LangGraph orchestration system,
enabling seamless integration of the TTS functionality into the voice agent workflow.
"""
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator, Optional, List, Tuple
from langgraph.graph import StateGraph, Node
from langgraph.graph.message import MessageState

# Import the TTS module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_to_speech.streaming import RealTimeResponseHandler
from text_to_speech.deepgram_tts import DeepgramTTS

logger = logging.getLogger(__name__)

class TTSNode(Node):
    """
    LangGraph node for Text-to-Speech processing.
    
    This node takes text input from the knowledge base output
    and converts it to speech in real-time.
    """
    
    def __init__(
        self,
        tts_handler: Optional[RealTimeResponseHandler] = None,
        **tts_kwargs
    ):
        """
        Initialize the TTS node.
        
        Args:
            tts_handler: Existing RealTimeResponseHandler or one will be created
            **tts_kwargs: Arguments to pass to RealTimeResponseHandler if creating a new one
        """
        super().__init__()
        self.tts_handler = tts_handler or RealTimeResponseHandler(**tts_kwargs)
        self.audio_stream = None
        self.processing_task = None
    
    async def process(self, state: MessageState) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process the input state and generate speech.
        
        Args:
            state: The current state of the conversation
            
        Yields:
            Updated state with audio data
        """
        # Extract text from the state
        text = state.get("knowledge_output", "")
        
        if not text:
            logger.warning("No text found in state for TTS processing")
            return
        
        # Initialize the audio stream if not already done
        if not self.audio_stream:
            self.audio_stream = await self.tts_handler.start()
            self.processing_task = asyncio.create_task(self._process_audio_stream())
        
        # Send the text to the TTS handler
        if isinstance(text, str):
            # Handle single text string
            await self.tts_handler.add_text(text)
        elif isinstance(text, list):
            # Handle list of words/tokens
            for word in text:
                if isinstance(word, str):
                    await self.tts_handler.add_word(word)
        
        # Update the state with audio information
        state["tts_active"] = True
        yield state
    
    async def _process_audio_stream(self) -> None:
        """Process the audio stream from the TTS handler."""
        try:
            async for audio_chunk in self.audio_stream:
                # You could send this audio directly to FreeSWITCH here
                # or put it in a queue for the telephony system to consume
                logger.debug(f"Received audio chunk: {len(audio_chunk)} bytes")
        except Exception as e:
            logger.error(f"Error processing audio stream: {str(e)}")
        finally:
            logger.debug("Audio stream processing completed")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_handler:
            await self.tts_handler.stop()
        
        if self.processing_task:
            try:
                self.processing_task.cancel()
                await asyncio.gather(self.processing_task, return_exceptions=True)
            except:
                pass

# Example of how to integrate this node into a LangGraph
def create_tts_graph() -> StateGraph:
    """
    Create a LangGraph with TTS integration.
    
    Returns:
        Configured StateGraph with TTS node
    """
    # Create the graph
    graph = StateGraph(MessageState)
    
    # Create nodes
    tts_node = TTSNode()
    
    # Add nodes to the graph
    graph.add_node("tts", tts_node.process)
    
    # Add edges (this will depend on your overall graph structure)
    # For example, connecting knowledge output to TTS:
    graph.add_edge("knowledge", "tts")
    
    # Add cleanup
    async def cleanup():
        await tts_node.cleanup()
    
    graph.add_cleanup(cleanup)
    
    return graph

# Example usage in the main voice agent graph
def integrate_tts_in_voice_agent(main_graph: StateGraph) -> StateGraph:
    """
    Integrate the TTS functionality into the main voice agent graph.
    
    Args:
        main_graph: The main voice agent StateGraph
        
    Returns:
        Updated StateGraph with TTS integration
    """
    # Create the TTS node
    tts_node = TTSNode()
    
    # Add the TTS node to the main graph
    main_graph.add_node("tts", tts_node.process)
    
    # Connect the knowledge output to the TTS node
    main_graph.add_edge("knowledge", "tts")
    
    # Connect the TTS node to the output (telephony system)
    main_graph.add_edge("tts", "telephony")
    
    # Add cleanup for the TTS node
    async def cleanup():
        await tts_node.cleanup()
    
    main_graph.add_cleanup(cleanup)
    
    return main_graph