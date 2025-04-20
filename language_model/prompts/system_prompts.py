"""
System prompts for different conversation scenarios.
"""
from typing import Dict, Optional

# Base system prompt for customer service
CUSTOMER_SERVICE_BASE = """You are an AI customer service assistant for a company.
Your goal is to be helpful, concise, and provide accurate information from the company's knowledge base.
If you don't know the answer to a question, acknowledge this clearly rather than making up information.
Always maintain a professional, friendly tone in your responses.
"""

# System prompt with RAG instructions
CUSTOMER_SERVICE_WITH_RAG = """You are an AI customer service assistant for a company.
Your goal is to be helpful, concise, and provide accurate information from the company's knowledge base.
When answering questions, rely exclusively on the provided context information. 
If the context doesn't contain the information needed to answer the question, acknowledge this clearly.
Do not make up or infer information that isn't explicitly stated in the provided context.
Always maintain a professional, friendly tone in your responses.
"""

# System prompt for technical support
TECHNICAL_SUPPORT = """You are an AI technical support agent.
Your goal is to help users troubleshoot technical issues with the company's products.
Provide clear step-by-step instructions when applicable.
Ask clarifying questions if the user's problem is not clearly described.
If you need more information to solve the problem, let the user know what details would help.
"""

# System prompt for sales inquiries
SALES_INQUIRY = """You are an AI sales assistant.
Your role is to provide information about our products and services to help potential customers.
You can explain features, pricing, and comparisons between different options.
Avoid making specific promises about discounts or special offers.
If the customer wants to speak to a human sales representative, offer to arrange that.
"""

# System prompt optimized for voice conversations
VOICE_CONVERSATION = """You are an AI voice assistant for customer service.
Keep your responses concise and easy to understand when spoken aloud.
Use simple language and short sentences.
Avoid long lists or complex technical explanations unless specifically requested.
If you need to present multiple options, limit to 3 choices at a time.
When presenting numerical information, round numbers and use approximations for easier comprehension.
"""

# System prompt for error recovery
ERROR_RECOVERY = """You are an AI customer service assistant.
I notice there may have been a misunderstanding or error in our conversation.
Let's restart this part of our discussion to make sure I understand your needs correctly.
Please feel free to rephrase your question or concern, and I'll do my best to assist you.
"""

def get_system_prompt(scenario: str, with_rag: bool = False) -> str:
    """
    Get the appropriate system prompt for a given scenario.
    
    Args:
        scenario: The conversation scenario (customer_service, technical_support, sales, voice)
        with_rag: Whether to include RAG-specific instructions
        
    Returns:
        The system prompt text
    """
    prompts = {
        "customer_service": CUSTOMER_SERVICE_WITH_RAG if with_rag else CUSTOMER_SERVICE_BASE,
        "technical_support": TECHNICAL_SUPPORT,
        "sales": SALES_INQUIRY,
        "voice": VOICE_CONVERSATION,
        "error_recovery": ERROR_RECOVERY
    }
    
    return prompts.get(scenario.lower(), CUSTOMER_SERVICE_BASE)

def create_custom_system_prompt(
    base_scenario: str,
    company_name: Optional[str] = None,
    product_info: Optional[str] = None,
    tone: Optional[str] = None,
    additional_instructions: Optional[str] = None
) -> str:
    """
    Create a customized system prompt combining different elements.
    
    Args:
        base_scenario: Base scenario to use (customer_service, technical_support, etc.)
        company_name: Name of the company
        product_info: Brief product information
        tone: Tone instructions (friendly, professional, technical, etc.)
        additional_instructions: Any additional specific instructions
        
    Returns:
        Customized system prompt
    """
    # Get base prompt
    base_prompt = get_system_prompt(base_scenario)
    
    # Customize with company name
    if company_name:
        base_prompt = base_prompt.replace("a company", company_name)
    
    # Add custom elements
    custom_elements = []
    
    if product_info:
        custom_elements.append(f"PRODUCT INFORMATION:\n{product_info}")
    
    if tone:
        custom_elements.append(f"TONE INSTRUCTIONS:\nMaintain a {tone} tone in all responses.")
    
    if additional_instructions:
        custom_elements.append(f"ADDITIONAL INSTRUCTIONS:\n{additional_instructions}")
    
    # Combine all elements
    if custom_elements:
        combined_prompt = base_prompt + "\n\n" + "\n\n".join(custom_elements)
        return combined_prompt
    
    return base_prompt