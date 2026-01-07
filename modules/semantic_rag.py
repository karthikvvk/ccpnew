"""
Semantic RAG - Embedding-native visual context extraction with self-pruning
Based on hierarchical semantic grounding with temporal reasoning
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
from utils.logger import setup_logger

logger = setup_logger("semantic_rag")


class SemanticRAG:
    """
    Embedding-native semantic grounding for video frames.
    
    Key features:
    - Self-pruning when frames are too unique (< 0.3 base score)
    - Weighted temporal voting
    - Multi-label outputs (not argmax)
    - Cached embeddings (no redundant encoding)
    - Embedding-native until final output
    """
    
    # Base confidence threshold - below this, skip RAG entirely
    BASE_SCORE_THRESHOLD = 0.3
    
    # Scene vocabulary (semantic anchors)
    SCENE_VOCABULARY = [
        # Objects & Structures
        "monument", "temple", "mosque", "church", "palace", "fort",
        "iron pillar", "sculpture", "artifact", "ancient structure",
        "building", "architecture", "ruins", "tower", "dome",
        
        # People & Actions
        "person speaking", "presenter", "crowd", "people walking",
        "close-up face", "interview", "demonstration",
        
        # Visual Elements
        "text on screen", "title card", "landscape", "cityscape",
        "aerial view", "interior", "exterior", "detail shot",
        
        # Natural Elements
        "sky", "water", "trees", "mountains", "sunset", "clouds"
    ]
    
    # Meta-categories for hierarchical grounding
    META_CATEGORIES = {
        "content_type": [
            "documentary", "educational", "tutorial", "news",
            "entertainment", "vlog", "presentation"
        ],
        "domain": [
            "historical", "religious", "cultural", "scientific",
            "architectural", "natural", "tourism", "art"
        ],
        "tone": [
            "informative", "narrative", "analytical", "exploratory",
            "serious", "casual", "formal"
        ]
    }
    
    def __init__(self, embedder=None):
        """
        Initialize semantic RAG
        
        Args:
            embedder: FrameEmbedder instance (for encoding text/images)
        """
        self.embedder = embedder
        self._vocab_embeddings = None  # Cached vocabulary embeddings
        self._category_embeddings = None  # Cached category embeddings
        
        logger.info("Semantic RAG initialized")
    
    def _ensure_vocab_cached(self):
        """Pre-encode scene vocabulary to avoid redundant encoding"""
        if self._vocab_embeddings is not None:
            return
        
        logger.info("Caching scene vocabulary embeddings...")
        self._vocab_embeddings = {}
        
        for term in self.SCENE_VOCABULARY:
            self._vocab_embeddings[term] = self.embedder.model.encode(term)
        
        logger.info(f"Cached {len(self._vocab_embeddings)} vocabulary embeddings")
    
    def _ensure_categories_cached(self):
        """Pre-encode meta-categories"""
        if self._category_embeddings is not None:
            return
        
        logger.info("Caching category embeddings...")
        self._category_embeddings = {}
        
        for cat_type, categories in self.META_CATEGORIES.items():
            self._category_embeddings[cat_type] = {}
            for cat in categories:
                self._category_embeddings[cat_type][cat] = self.embedder.model.encode(cat)
        
        logger.info(f"Cached category embeddings for {len(self.META_CATEGORIES)} types")
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _match_to_vocabulary(
        self,
        frame_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Match frame embedding to scene vocabulary
        
        Returns:
            List of (term, confidence_score) tuples
        """
        self._ensure_vocab_cached()
        
        scores = {}
        for term, vocab_emb in self._vocab_embeddings.items():
            scores[term] = self._cosine_similarity(frame_embedding, vocab_emb)
        
        # Sort by score, return top-k
        sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches[:top_k]
    
    def _deduplicate_concepts(
        self,
        concepts: List[Tuple[str, float]],
        threshold: float = 0.85
    ) -> List[Tuple[str, float]]:
        """
        Remove semantically similar concepts using embedding clustering
        
        Args:
            concepts: List of (concept, score) tuples
            threshold: Similarity threshold for deduplication
            
        Returns:
            Deduplicated list
        """
        if not concepts:
            return []
        
        self._ensure_vocab_cached()
        
        unique = []
        for concept, score in concepts:
            concept_emb = self._vocab_embeddings[concept]
            
            # Check if similar to any existing unique concept
            is_duplicate = False
            for unique_concept, _ in unique:
                unique_emb = self._vocab_embeddings[unique_concept]
                if self._cosine_similarity(concept_emb, unique_emb) > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append((concept, score))
        
        return unique
    
    def _classify_to_categories(
        self,
        concepts: List[Tuple[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Map concepts to meta-categories using semantic similarity
        
        Returns:
            Dict of {category_type: [(category, weighted_score), ...]}
        """
        self._ensure_categories_cached()
        
        results = {}
        
        for cat_type, cat_embeddings in self._category_embeddings.items():
            category_votes = defaultdict(float)
            
            # For each concept, vote for categories
            for concept, concept_score in concepts:
                concept_emb = self._vocab_embeddings[concept]
                
                for category, cat_emb in cat_embeddings.items():
                    similarity = self._cosine_similarity(concept_emb, cat_emb)
                    # Weight by concept confidence
                    category_votes[category] += similarity * concept_score
            
            # Sort and keep multi-label (top 3 per type)
            sorted_categories = sorted(
                category_votes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            results[cat_type] = sorted_categories
        
        return results
    
    def analyze_frames(
        self,
        frame_embeddings: List[np.ndarray],
        frame_paths: List[Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze frames with self-pruning
        
        Args:
            frame_embeddings: List of CLIP frame embeddings
            frame_paths: Corresponding frame paths
            
        Returns:
            Analysis dict or None if self-pruned
        """
        if not frame_embeddings:
            logger.warning("No frames to analyze")
            return None
        
        logger.info(f"Analyzing {len(frame_embeddings)} frames")
        
        # Step 1: Match each frame to vocabulary (embedding-native)
        frame_matches = []
        all_scores = []
        
        for i, frame_emb in enumerate(frame_embeddings):
            matches = self._match_to_vocabulary(frame_emb, top_k=3)
            frame_matches.append(matches)
            all_scores.extend([score for _, score in matches])
        
        # Self-pruning check (if enabled)
        avg_confidence = np.mean(all_scores)
        logger.info(f"Average frame-to-concept confidence: {avg_confidence:.3f}")
        
        from config import settings
        if settings.rag_enable_self_pruning:
            if avg_confidence < self.BASE_SCORE_THRESHOLD:
                logger.warning(
                    f"Confidence {avg_confidence:.3f} below threshold {self.BASE_SCORE_THRESHOLD}. "
                    "Frames too unique - skipping RAG (Whisper alone is better)"
                )
                return None
        else:
            logger.info(f"Self-pruning DISABLED - continuing with RAG (confidence: {avg_confidence:.3f})")
        
        # Step 2: Aggregate with temporal weighting
        concept_votes = defaultdict(lambda: {"score": 0.0, "count": 0})
        
        for frame_idx, matches in enumerate(frame_matches):
            # Temporal weight (keyframes > duplicates)
            # Simple approach: weight by frame position uniqueness
            temporal_weight = 1.0  # Could be enhanced with scene detection
            
            for concept, score in matches:
                weighted_score = score * temporal_weight
                concept_votes[concept]["score"] += weighted_score
                concept_votes[concept]["count"] += 1
        
        # Normalize and sort
        aggregated_concepts = [
            (concept, data["score"] / data["count"])
            for concept, data in concept_votes.items()
        ]
        aggregated_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Deduplicate
        unique_concepts = self._deduplicate_concepts(aggregated_concepts[:10])
        
        logger.info(f"Unique concepts: {[c for c, _ in unique_concepts]}")
        
        # Step 4: Classify to meta-categories
        categories = self._classify_to_categories(unique_concepts[:5])
        
        # Step 5: Build natural language context (only at the end)
        context = self._build_context(unique_concepts, categories)
        
        return {
            "confidence": avg_confidence,
            "concepts": unique_concepts[:5],
            "categories": categories,
            "context": context,
            "self_pruned": False
        }
    
    def _build_context(
        self,
        concepts: List[Tuple[str, float]],
        categories: Dict[str, List[Tuple[str, float]]]
    ) -> str:
        """Build natural language context from analysis"""
        # Top concepts
        concept_str = ", ".join([c for c, _ in concepts[:3]])
        
        # Top category from each type
        cat_strs = []
        for cat_type, cat_list in categories.items():
            if cat_list:
                cat_strs.append(cat_list[0][0])
        
        category_str = ", ".join(cat_strs)
        
        context = f"Visual content: {concept_str}. Context: {category_str}."
        
        logger.info(f"Generated context: {context}")
        return context
