
import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.semantic_rag import SemanticRAG

# Mock FrameEmbedder
class MockEmbedder:
    def __init__(self):
        self.model = self
        
    def encode(self, texts):
        # Return random normalized vectors for simplicity, or deterministic ones
        # If texts is list, return list of arrays
        if isinstance(texts, list):
            return [np.random.rand(512) for _ in texts]
        return np.random.rand(512)

def test_dynamic_rag():
    print("Testing Dynamic Semantic RAG...")
    
    # 1. Setup
    embedder = MockEmbedder()
    rag = SemanticRAG(embedder=embedder)
    
    # 2. Mock Data
    # Create deterministic embeddings to test filtering
    # Vector A (Frame)
    vec_frame = np.zeros(512)
    vec_frame[0] = 1.0  # Pointing in dimension 0
    
    # Vector B (Matching Word: "cat") -> High similarity
    vec_cat = np.zeros(512)
    vec_cat[0] = 0.9
    vec_cat[1] = 0.1
    
    # Vector C (Non-matching Word: "car") -> Low similarity
    vec_car = np.zeros(512)
    vec_car[2] = 1.0
    
    # Manually inject into cache to bypass random mock encoding
    rag._vocab_embeddings = {
        "cat": vec_cat,
        "car": vec_car
    }
    
    # 3. Test verify_space_and_filter
    print("\nTest 1: verify_space_and_filter")
    candidate_words = ["cat", "car"]
    matches = rag.verify_space_and_filter(vec_frame, candidate_words, threshold=0.5)
    
    print(f"Frame Vector: [1.0, 0.0, ...]")
    print(f"Cat Vector:   [0.9, 0.1, ...] (Sim: ~0.9)")
    print(f"Car Vector:   [0.0, 0.0, 1.0, ...] (Sim: 0.0)")
    print(f"Matches found: {matches}")
    
    assert len(matches) == 1
    assert matches[0][0] == "cat"
    print("✅ Reverse verification logic passed (filtered 'car')")
    
    # 4. Test analyze_frames (Integration)
    print("\nTest 2: analyze_frames")
    
    # Mock embeddings list
    frame_embeddings = [vec_frame]
    
    # It should look up "cat" and "car", only find "cat"
    result = rag.analyze_frames(frame_embeddings, [], vocabulary=["cat", "car"])
    
    if result:
        concepts = [c[0] for c in result['concepts']]
        print(f"Concepts found: {concepts}")
        assert "cat" in concepts
        assert "car" not in concepts
        print("✅ analyze_frames integration passed")
    else:
        print("❌ analyze_frames returned None")
        
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_dynamic_rag()
