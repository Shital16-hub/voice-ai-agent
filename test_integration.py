"""
Test script for the complete Voice AI Agent.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the integration layer
from integration import VoiceAssistantIntegration
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from language_model.model_manager import ModelManager

async def test_integration():
    """Test the full integration."""
    print("Initializing Voice AI Agent...")
    
    # Initialize language model
    language_model = ModelManager(
        model_name=None,  # Use default model
        scenario="customer_service",
        with_rag=True
    )
    
    # Initialize index manager and query engine
    index_manager = IndexManager(storage_dir="./storage")
    await index_manager.init()
    
    query_engine = QueryEngine(index_manager=index_manager)
    await query_engine.init()
    
    # Initialize integration
    integration = VoiceAssistantIntegration(
        language_model=language_model,
        query_engine=query_engine
    )
    await integration.init()
    
    # Run interactive session
    await integration.interactive_session()

if __name__ == "__main__":
    asyncio.run(test_integration())