"""
Standalone script for Semantic RAG Analysis of Video Frames.
This script demonstrates the "Reverse Apply" logic where keywords are verified against frame embeddings.

Features:
1. Extracts frames from video.
2. Generates CLIP embeddings for each frame.
3. reverse_verify(): Checks if candidate keywords actually exist in the frame's semantic space.
4. Classifies content into Domains (Historical, Scientific, etc.).
5. Outputs a single JSON with frame-by-frame analysis.

Dependencies:
    pip install sentence-transformers numpy pillow ffmpeg-python
"""
import argparse
import sys
import json
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

# Try imports
try:
    from sentence_transformers import SentenceTransformer
    import ffmpeg
except ImportError:
    print("Error: Missing dependencies. Install with 'pip install sentence-transformers numpy pillow ffmpeg-python'")
    sys.exit(1)

# --- Configuration ---
META_CATEGORIES = {
    "domain": [
        "historical", "religious", "cultural", "scientific",
        "architectural", "natural", "tourism", "art", "technology"
    ],
    "tone": [
        "informative", "narrative", "analytical", "exploratory",
        "serious", "casual", "formal"
    ]
}

# Sample vocabulary to test against (Dynamic input in real usage)
DEFAULT_VOCABULARY = [
    "temple", "ancient", "architecture", "river", "mountain", "people", 
    "crowd", "worship", "technology", "computer", "coding", "forest", "sky",
    "building", "ruins", "sculpture", "painting", "festival"
]

class SemanticAnalyzer:
    def __init__(self, model_name="clip-ViT-B-32"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.vocab_cache = {}
        self.category_cache = {}
        self._precompute_categories()
        print("Model loaded.")

    def _precompute_categories(self):
        """Pre-compute embeddings for meta-categories."""
        for cat_type, categories in META_CATEGORIES.items():
            self.category_cache[cat_type] = {}
            embeddings = self.model.encode(categories)
            for cat, emb in zip(categories, embeddings):
                self.category_cache[cat_type][cat] = emb

    def extract_frames(self, video_path, output_dir, fps=0.5):
        """Extract frames from video using FFmpeg."""
        print(f"Extracting frames from {video_path} to {output_dir} at {fps} fps...")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.filter(stream, 'fps', fps=fps)
            stream = ffmpeg.output(stream, str(Path(output_dir) / 'frame_%04d.jpg'), **{'qscale:v': 2})
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            frames = sorted(Path(output_dir).glob('*.jpg'))
            print(f"Extracted {len(frames)} frames.")
            return frames
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return []

    def embed_frame(self, frame_path):
        """Generate embedding for a frame."""
        image = Image.open(frame_path).convert('RGB')
        return self.model.encode(image)

    def _cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)

    def reverse_verify_keywords(self, frame_embedding, candidates, threshold=0.25):
        """
        The "Reverse Apply" logic.
        Verifies if candidate keywords match the visual content of the frame.
        """
        # Encode new candidates
        new_terms = [c for c in candidates if c not in self.vocab_cache]
        if new_terms:
            embeddings = self.model.encode(new_terms)
            for term, emb in zip(new_terms, embeddings):
                self.vocab_cache[term] = emb
        
        verified = []
        for term in candidates:
            term_emb = self.vocab_cache[term]
            score = self._cosine_similarity(frame_embedding, term_emb)
            if score >= threshold:
                verified.append({"term": term, "score": float(score)})
        
        # Sort by score
        verified.sort(key=lambda x: x["score"], reverse=True)
        return verified

    def classify_domain(self, frame_embedding):
        """Classify frame into domains/tones."""
        results = {}
        for cat_type, categories in self.category_cache.items():
            best_cat = None
            best_score = -1.0
            
            for cat, cat_emb in categories.items():
                score = self._cosine_similarity(frame_embedding, cat_emb)
                if score > best_score:
                    best_score = score
                    best_cat = cat
            
            results[cat_type] = {"category": best_cat, "score": float(best_score)}
        return results

    def analyze_video(self, video_path, vocabulary):
        # 1. Extract Frames
        frame_dir = Path("temp_frames_analysis")
        frames = self.extract_frames(video_path, frame_dir)
        
        if not frames:
            print("No frames extracted.")
            return None

        analysis_results = []
        global_domain_votes = defaultdict(float)

        print("Analyzing frames...")
        for i, frame_path in enumerate(frames):
            # 2. Embed Frame
            frame_emb = self.embed_frame(frame_path)
            
            # 3. Reverse Verify Keywords (RAG Logic)
            verified_keywords = self.reverse_verify_keywords(frame_emb, vocabulary)
            
            # 4. Classify Domain
            classification = self.classify_domain(frame_emb)
            
            # Vote for global domain
            domain = classification.get("domain", {}).get("category")
            score = classification.get("domain", {}).get("score", 0)
            if domain:
                global_domain_votes[domain] += score

            analysis_results.append({
                "frame_id": i,
                "path": str(frame_path),
                "verified_keywords": verified_keywords[:5], # Top 5
                "classification": classification
            })
            
            if (i+1) % 5 == 0:
                print(f"Analyzed {i+1}/{len(frames)} frames...")

        # Determine Global Domain
        best_global_domain = max(global_domain_votes.items(), key=lambda x: x[1])[0] if global_domain_votes else "unknown"
        
        final_output = {
            "video_path": str(video_path),
            "global_domain": best_global_domain,
            "frames_analyzed": len(frames),
            "frame_analysis": analysis_results
        }
        
        # Cleanup
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
            
        return final_output

def main():
    parser = argparse.ArgumentParser(description="Semantic RAG Frame Analyzer")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", default="analysis_result.json", help="Path to output JSON")
    parser.add_argument("--vocab", nargs="+", help="Custom vocabulary list (space separated)", default=DEFAULT_VOCABULARY)
    
    args = parser.parse_args()
    
    analyzer = SemanticAnalyzer()
    result = analyzer.analyze_video(args.video_path, args.vocab)
    
    if result:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Analysis saved to {args.output}")
        print(f"Global Domain Detected: {result['global_domain']}")

if __name__ == "__main__":
    main()
