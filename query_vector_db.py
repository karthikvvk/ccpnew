"""
Simple Vector DB Query - View what's stored
"""
import sys
sys.path.insert(0, '/home/muruga/workspace/ccp6/lib/python-standalone/lib/python3.11/site-packages')

from pathlib import Path
import chromadb

# Your job ID
JOB_ID = "23e31f70-d218-4efa-b9bd-755070559097"

# Load the ChromaDB
db_path = Path(f"outputs/{JOB_ID}/vector_db")
client = chromadb.PersistentClient(path=str(db_path))

# Get the collection
collection_name = f"frames_{JOB_ID}"
collection = client.get_collection(collection_name)

print(f"📊 Vector DB: {db_path}")
print(f"Collection: {collection_name}")
print(f"Total frames: {collection.count()}")
print("=" * 60)

# Get all stored frames
results = collection.get(include=['metadatas'])

print("\n📋 All frames in database:\n")
for i, metadata in enumerate(results['metadatas'], 1):
    print(f"{i}. Frame #{metadata['frame_number']:03d}")
    print(f"   Path: {metadata['frame_path']}")
    print(f"   Embedding dimension: {len(collection.get(ids=[results['ids'][i-1]], include=['embeddings'])['embeddings'][0])}")
    print()

print("=" * 60)
print("\n💡 This vector DB can be queried with:")
print("   - Text queries (e.g., 'person talking')")
print("   - Image queries (another frame)")
print("   - To find similar frames based on visual content")
print("\nThe CLIP model converts both text and images to the same")
print("embedding space, making semantic search possible!")
