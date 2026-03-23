"""
Standalone script for RAG Embedding (CLIP + ChromaDB).
This script demonstrates how to generate embeddings for images using CLIP and store/query them using ChromaDB.

Dependencies:
    pip install sentence-transformers chromadb pillow numpy
"""
import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: Missing dependencies. Install with 'pip install sentence-transformers chromadb pillow numpy'")
    sys.exit(1)

def setup_model(model_name="clip-ViT-B-32"):
    """
    Load the embedding model (CLIP).
    """
    print(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def setup_db(db_path="./chroma_db", collection_name="demo_frames"):
    """
    Initialize ChromaDB.
    """
    print(f"Initializing ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def embed_image(model, image_path):
    """
    Generate embedding for an image.
    """
    print(f"Generating embedding for: {image_path}")
    image = Image.open(image_path).convert('RGB')
    embedding = model.encode(image)
    return embedding

def add_to_db(collection, image_path, embedding):
    """
    Add embedding to ChromaDB.
    """
    frame_name = Path(image_path).name
    print(f"Adding '{frame_name}' to database...")
    
    collection.add(
        ids=[frame_name],
        embeddings=[embedding.tolist()],
        metadatas=[{"path": str(image_path)}]
    )

def query_db(collection, query_embedding, n_results=1):
    """
    Query database for similar images.
    """
    print("Querying database...")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Standalone RAG Embedding Demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Add Image
    p_add = subparsers.add_parser("add", help="Add image embedding to DB")
    p_add.add_argument("image", help="Path to image file")
    
    # Query Image
    p_query = subparsers.add_parser("query", help="Query similar images")
    p_query.add_argument("image", help="Path to query image")
    
    args = parser.parse_args()
    
    model = setup_model()
    collection = setup_db()
    
    if args.command == "add":
        embedding = embed_image(model, args.image)
        add_to_db(collection, args.image, embedding)
        print("Done.")
        
    elif args.command == "query":
        embedding = embed_image(model, args.image)
        results = query_db(collection, embedding)
        
        print("-" * 50)
        print("Query Results:")
        if results['ids']:
            ids = results['ids'][0]
            distances = results['distances'][0]
            for i, (id, dist) in enumerate(zip(ids, distances)):
                print(f"{i+1}. {id} (Distance: {dist:.4f})")
        else:
            print("No results found.")
        print("-" * 50)

if __name__ == "__main__":
    main()
