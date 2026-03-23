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
    # Scene vocabulary (Removed static list - now dynamic)
    # SCENE_VOCABULARY removed
    
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
        self._vocab_embeddings = {}  # Cached vocabulary embeddings: {term: embedding}
        self._category_embeddings = None  # Cached category embeddings
        
        logger.info("Semantic RAG initialized")
    
    def _ensure_vocab_cached(self, vocabulary: List[str]):
        """
        Ensure provided vocabulary terms are cached.
        Only encodes words that are not already in the cache.
        
        Args:
            vocabulary: List of string terms to ensure are in cache
        """
        if not vocabulary:
            return
            
        new_terms = [term for term in vocabulary if term not in self._vocab_embeddings]
        
        if new_terms:
            logger.info(f"Encoding {len(new_terms)} new vocabulary terms...")
            # Batch encode for efficiency if supported, otherwise loop
            if hasattr(self.embedder.model, 'encode'):
                embeddings = self.embedder.model.encode(new_terms)
                for term, emb in zip(new_terms, embeddings):
                    self._vocab_embeddings[term] = emb
            else:
                # Fallback if specific embedder API is different
                for term in new_terms:
                     self._vocab_embeddings[term] = self.embedder.model.encode(term)
                
            logger.info(f"Updated vocabulary cache. Total terms: {len(self._vocab_embeddings)}")
    
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
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)

    def verify_space_and_filter(
        self,
        frame_embedding: np.ndarray,
        candidate_words: List[str],
        threshold: float = 0.25
    ) -> List[Tuple[str, float]]:
        """
        Reverse logical check:
        1. Embed the candidate words (if not cached).
        2. Verify if they fall into the semantic space of the frame.
        3. Filter those that don't meet the threshold.
        
        Args:
            frame_embedding: The CLIP embedding of the video frame.
            candidate_words: List of words/phrases to verify.
            threshold: Cosine similarity threshold.
            
        Returns:
            List of (word, score) tuples that passed the filter.
        """
        self._ensure_vocab_cached(candidate_words)
        
        # Filter words that have embeddings
        valid_words = [w for w in candidate_words if w in self._vocab_embeddings]
        if not valid_words:
            return []
            
        # Stack embeddings: Shape (N_words, Emb_dim)
        word_matrix = np.stack([self._vocab_embeddings[w] for w in valid_words])
        
        # Normalize word embeddings (if not already normalized, but let's be safe)
        word_norms = np.linalg.norm(word_matrix, axis=1, keepdims=True)
        word_matrix = word_matrix / (word_norms + 1e-9)
        
        # Normalize frame embedding
        frame_norm = np.linalg.norm(frame_embedding)
        frame_vec = frame_embedding / (frame_norm + 1e-9)
        
        # Batch dot product: (N_words, D) @ (D,) -> (N_words,)
        scores = np.dot(word_matrix, frame_vec)
        
        # Filter results
        verified_matches = []
        for word, score in zip(valid_words, scores):
            if score >= threshold:
                verified_matches.append((word, float(score)))
        
        # Sort by score descending
        verified_matches.sort(key=lambda x: x[1], reverse=True)
        return verified_matches
    
    def _match_to_vocabulary(
        self,
        frame_embedding: np.ndarray,
        vocabulary: List[str],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Match frame embedding to provided vocabulary using the verification logic.
        """
        matches = self.verify_space_and_filter(frame_embedding, vocabulary, threshold=0.2)
        return matches[:top_k]
    
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
        
        self._ensure_vocab_cached([c for c, _ in concepts])
        
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
        frame_paths: List[Path],
        vocabulary: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze frames with dynamic vocabulary
        
        Args:
            frame_embeddings: List of CLIP frame embeddings
            frame_paths: Corresponding frame paths
            vocabulary: Dynamic list of concepts to look for
            
        Returns:
            Analysis dict or None if self-pruned
        """
        if not frame_embeddings:
            logger.warning("No frames to analyze")
            return None
            
        if not vocabulary:
            logger.warning("No vocabulary provided for analysis")
            return None
        
        logger.info(f"Analyzing {len(frame_embeddings)} frames against {len(vocabulary)} concepts")
        
        # Step 1: Match each frame to vocabulary (embedding-native)
        frame_matches = []
        all_scores = []
        
        for i, frame_emb in enumerate(frame_embeddings):
            matches = self._match_to_vocabulary(frame_emb, vocabulary, top_k=3)
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
    
    def analyze_global(
        self,
        all_frame_embeddings: List[np.ndarray],
        vocabulary: List[str],
        sample_rate: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Perform video-level global RAG analysis with dynamic vocabulary.
        """
        if not all_frame_embeddings:
            logger.warning("No frames for global analysis")
            return None
            
        if not vocabulary:
             logger.warning("No vocabulary provided for global analysis")
             return None
        
        # Sample frames for efficiency
        sampled = all_frame_embeddings[::sample_rate]
        logger.info(f"Global RAG: analyzing {len(sampled)} sampled frames (from {len(all_frame_embeddings)})")
        
        # Aggregate all frame-to-concept matches
        all_concept_scores = defaultdict(list)
        
        self._ensure_vocab_cached(vocabulary)
        
        for frame_emb in sampled:
            matches = self._match_to_vocabulary(frame_emb, vocabulary, top_k=5)
            for concept, score in matches:
                all_concept_scores[concept].append(score)
        
        # Calculate mean score and frequency for each concept
        global_concepts = []
        for concept, scores in all_concept_scores.items():
            mean_score = np.mean(scores)
            frequency = len(scores) / len(sampled)  # How often this concept appears
            combined_score = mean_score * (0.5 + 0.5 * frequency)  # Blend score and frequency
            global_concepts.append((concept, combined_score))
        
        global_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate
        unique_global = self._deduplicate_concepts(global_concepts[:15])
        
        # Get global categories
        global_categories = self._classify_to_categories(unique_global[:7])
        
        # Extract domain and terminology
        domain = None
        content_type = None
        
        if 'domain' in global_categories and global_categories['domain']:
            domain = global_categories['domain'][0][0]
        if 'content_type' in global_categories and global_categories['content_type']:
            content_type = global_categories['content_type'][0][0]
        
        # Build global context
        global_context = self._build_context(unique_global, global_categories)
        
        # Extract key terminology (concepts that appear frequently)
        terminology = [c for c, s in unique_global[:5]]
        
        result = {
            'concepts': unique_global[:10],
            'categories': global_categories,
            'domain': domain,
            'content_type': content_type,
            'terminology': terminology,
            'context': global_context,
            'frame_count': len(all_frame_embeddings),
            'sampled_count': len(sampled)
        }
        
        logger.info(f"Global RAG: domain={domain}, content_type={content_type}, "
                   f"terminology={terminology}")
        
        return result
    
    def analyze_chunk(
        self,
        chunk_frame_embeddings: List[np.ndarray],
        vocabulary: List[str],
        global_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Perform chunk-level RAG analysis constrained by global context.
        """
        if not chunk_frame_embeddings:
            logger.warning("No frames for chunk analysis")
            return None
        
        logger.info(f"Chunk RAG: analyzing {len(chunk_frame_embeddings)} frames")
        
        # Match chunk frames to vocabulary
        chunk_concepts = defaultdict(list)
        
        self._ensure_vocab_cached(vocabulary)
        
        for frame_emb in chunk_frame_embeddings:
            matches = self._match_to_vocabulary(frame_emb, vocabulary, top_k=3)
            for concept, score in matches:
                chunk_concepts[concept].append(score)
        
        # Score and rank
        scored_concepts = []
        for concept, scores in chunk_concepts.items():
            mean_score = np.mean(scores)
            
            # Boost if concept appears in global terminology
            if global_context and concept in global_context.get('terminology', []):
                mean_score *= 1.3  # 30% boost for global consistency
            
            scored_concepts.append((concept, mean_score))
        
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate
        unique_chunk = self._deduplicate_concepts(scored_concepts[:8])
        
        # Get chunk-specific categories, constrained by global
        chunk_categories = self._classify_to_categories(unique_chunk[:5])
        
        # Merge with global context
        merged_context_parts = []
        
        if global_context:
            # Include global domain
            if global_context.get('domain'):
                merged_context_parts.append(f"Domain: {global_context['domain']}")
            if global_context.get('content_type'):
                merged_context_parts.append(f"Type: {global_context['content_type']}")
        
        # Add chunk-specific concepts
        chunk_concept_str = ", ".join([c for c, _ in unique_chunk[:3]])
        merged_context_parts.append(f"Scene: {chunk_concept_str}")
        
        chunk_context = ". ".join(merged_context_parts) + "."
        
        result = {
            'concepts': unique_chunk[:5],
            'categories': chunk_categories,
            'context': chunk_context,
            'global_aligned': global_context is not None
        }
        
        logger.info(f"Chunk RAG context: {chunk_context}")
        
        return result
