import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine

async def test_retrieval():
    print("Initializing index manager...")
    index_manager = IndexManager(storage_dir="./storage")
    await index_manager.init()
    
    print("Initializing query engine...")
    query_engine = QueryEngine(index_manager=index_manager)
    await query_engine.init()
    
    print("\nRetrieval system is ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        print("Retrieving information...")
        results = await query_engine.retrieve_with_sources(query)
        
        print(f"\nFound {len(results['results'])} relevant documents:")
        
        for i, doc in enumerate(results['results']):
            print(f"\n--- Result {i+1} ---")
            score = doc.get('score', 0)
            print(f"Score: {score:.4f}")
            
            if 'metadata' in doc:
                source = doc['metadata'].get('source', 'Unknown')
                print(f"Source: {source}")
            
            text = doc['text']
            if len(text) > 300:
                text = text[:297] + "..."
            print(f"Content: {text}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())