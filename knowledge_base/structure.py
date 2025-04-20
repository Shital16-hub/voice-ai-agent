import os

# Define the folder structure
structure = {
    '__init__.py': None,
    'config.py': None,
    'document_processor.py': None,
    'embedding_generator.py': None,
    'retriever.py': None,
    'vector_store.py': None,
    'conversation_manager.py': None,
    'utils': {
        '__init__.py': None,
        'text_utils.py': None,
        'file_utils.py': None,
    },
    'examples': {
        'indexing_example.py': None,
        'retrieval_example.py': None,
        'integration_example.py': None,
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if content is None:  # It's a file
            with open(path, 'w') as f:
                pass  # Create empty file
        else:  # It's a directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

# Get current directory
current_dir = os.getcwd()

# Verify we're in the right place (optional check)
dir_name = os.path.basename(current_dir)
if dir_name != 'knowledge_base':
    print(f"Warning: Current directory is '{dir_name}', expected 'knowledge_base'")
    response = input("Continue anyway? (y/n): ").lower()
    if response != 'y':
        exit()

# Create the structure
create_structure(current_dir, structure)
print("Knowledge Base folder structure created successfully!")