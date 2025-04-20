#!/usr/bin/env python3
"""
Basic chat example with the language model.
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

async def chat_session(
    model_name: Optional[str] = None,
    scenario: str = "customer_service",
    temperature: float = 0.7,
    max_messages: int = 10,
    company_name: Optional[str] = None,
    product_info: Optional[str] = None
):
    """
    Run an interactive chat session.
    
    Args:
        model_name: Name of the model to use
        scenario: Conversation scenario
        temperature: Temperature for sampling
        max_messages: Maximum number of exchanges before ending
        company_name: Optional company name for prompt customization
        product_info: Optional product info for prompt customization
    """
    print("\n" + "="*50)
    print(f"Starting chat with model: {model_name or 'default'}")
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
            
            # Generate response
            print("\nAI: ", end="", flush=True)
            
            response = await manager.generate_response(
                user_input=user_input,
                temperature=temperature
            )
            
            # Print response
            print(response["response"])
            
            # Show metadata (optional)
            if "--verbose" in sys.argv:
                print("\nMetadata:")
                for key, value in response["metadata"].items():
                    print(f"  {key}: {value}")
            
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Basic chat example')
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
    
    args = parser.parse_args()
    
    # Run chat session
    asyncio.run(chat_session(
        model_name=args.model,
        scenario=args.scenario,
        temperature=args.temperature,
        max_messages=args.max_messages,
        company_name=args.company,
        product_info=args.product_info
    ))

if __name__ == "__main__":
    main()