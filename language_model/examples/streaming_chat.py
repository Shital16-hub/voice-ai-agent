#!/usr/bin/env python3
"""
Streaming chat example with the language model.
"""
import sys
import os
import argparse
import asyncio
import logging
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from language_model.model_manager import ModelManager
from language_model.prompts.system_prompts import create_custom_system_prompt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def streaming_chat_session(
    model_name: Optional[str] = None,
    scenario: str = "customer_service",
    temperature: float = 0.7,
    max_messages: int = 10,
    company_name: Optional[str] = None,
    product_info: Optional[str] = None
):
    """
    Run an interactive streaming chat session.
    
    Args:
        model_name: Name of the model to use
        scenario: Conversation scenario
        temperature: Temperature for sampling
        max_messages: Maximum number of exchanges before ending
        company_name: Optional company name for prompt customization
        product_info: Optional product info for prompt customization
    """
    print("\n" + "="*50)
    print(f"Starting streaming chat with model: {model_name or 'default'}")
    print(f"Scenario: {scenario}")
    print("Type 'exit', 'quit', or press Ctrl+C to end the chat")
    print("="*50 + "\n")
    
    # Create custom system prompt if company info provided
    system_prompt = None
    if company_name or product_info:
        system_prompt = create_custom_system_prompt(
            base_scenario=scenario,
            company_name=company_name,
            product_info=product_info,
            tone="friendly and helpful"
        )
    
    # Initialize model manager
    manager = ModelManager(
        model_name=model_name,
        system_prompt=system_prompt,
        scenario=scenario
    )
    
    message_count = 0
    
    try:
        while message_count < max_messages:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Stream response
            print("\nAI: ", end="", flush=True)
            
            full_response = ""
            async for chunk in manager.generate_streaming_response(
                user_input=user_input,
                temperature=temperature
            ):
                chunk_text = chunk["chunk"]
                print(chunk_text, end="", flush=True)
                full_response += chunk_text
                
                if chunk["done"]:
                    # Print metadata at the end if verbose
                    if "--verbose" in sys.argv:
                        print("\n\nMetadata:")
                        for key, value in chunk["metadata"].items():
                            print(f"  {key}: {value}")
            
            print()  # New line after streaming completes
            message_count += 1
    
    except KeyboardInterrupt:
        print("\n\nChat session ended by user.")
    except Exception as e:
        logger.error(f"Error in chat session: {e}")
        print(f"\n\nAn error occurred: {e}")
    
    # Print stats
    if "--stats" in sys.argv or "--verbose" in sys.argv:
        stats = manager.get_stats()
        print("\nSession stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

async def integrated_example(
    transcription: str,
    model_name: Optional[str] = None,
    scenario: str = "voice"
):
    """
    Example of integration with speech-to-text.
    
    Args:
        transcription: Transcribed speech input
        model_name: Name of the model to use
        scenario: Conversation scenario
    """
    print("\nReceived transcription:", transcription)
    
    # Initialize model manager
    manager = ModelManager(
        model_name=model_name,
        scenario=scenario
    )
    
    # Generate streaming response
    print("\nGenerating response: ", end="", flush=True)
    
    full_response = ""
    async for chunk in manager.generate_streaming_response(
        user_input=transcription,
        temperature=0.7
    ):
        chunk_text = chunk["chunk"]
        print(chunk_text, end="", flush=True)
        full_response += chunk_text
    
    print("\n\nFull response:", full_response)
    print("This response would be sent to text-to-speech")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Streaming chat example')
    parser.add_argument('--model', type=str, 
                      help='Model name to use (default from config)')
    parser.add_argument('--scenario', type=str, default='customer_service',
                      choices=['customer_service', 'technical_support', 'sales', 'voice'],
                      help='Conversation scenario')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for sampling (higher=more creative)')
    parser.add_argument('--max-messages', type=int, default=10,
                      help='Maximum number of message exchanges')
    parser.add_argument('--company', type=str,
                      help='Company name for prompt customization')
    parser.add_argument('--product-info', type=str,
                      help='Product information for prompt customization')
    parser.add_argument('--verbose', action='store_true',
                      help='Show verbose output including metadata')
    parser.add_argument('--stats', action='store_true',
                      help='Show usage statistics at the end')
    parser.add_argument('--transcription', type=str,
                      help='Run integrated example with provided transcription')
    
    args = parser.parse_args()
    
    # Check if we should run integrated example with speech-to-text
    if args.transcription:
        asyncio.run(integrated_example(
            transcription=args.transcription,
            model_name=args.model,
            scenario='voice'
        ))
    else:
        # Run streaming chat session
        asyncio.run(streaming_chat_session(
            model_name=args.model,
            scenario=args.scenario,
            temperature=args.temperature,
            max_messages=args.max_messages,
            company_name=args.company,
            product_info=args.product_info
        ))

if __name__ == "__main__":
    main()