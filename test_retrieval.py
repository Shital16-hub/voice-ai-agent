import asyncio
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add the parent directory to the path if needed
sys.path.append(str(Path(__file__).parent))

# Import necessary components
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

async def test_retrieval():
    print("Initializing retrieval test...")
    
    # Disable LLM usage by setting Settings.llm to None
    Settings.llm = None
    
    # Create embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    Settings.embed_model = embed_model
    
    # Connect to ChromaDB
    print("Connecting to ChromaDB...")
    storage_dir = "./storage"
    chroma_client = chromadb.PersistentClient(path=storage_dir)
    collection = chroma_client.get_or_create_collection("company_knowledge")
    
    # Create vector store and index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Create retriever with specified similarity_top_k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3  # Retrieve top 3 similar documents
    )
    
    print("\nRetrieval system is ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Create query bundle
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        print("Retrieving information...")
        nodes = retriever.retrieve(query_bundle)
        
        print(f"\nFound {len(nodes)} relevant documents:")
        
        for i, node in enumerate(nodes):
            print(f"\n--- Result {i+1} ---")
            score = getattr(node, 'score', 0)
            print(f"Score: {score:.4f}")
            
            if node.metadata:
                source = node.metadata.get('source', 'Unknown')
                print(f"Source: {source}")
            
            text = node.text
            if len(text) > 300:
                text = text[:297] + "..."
            print(f"Content: {text}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())