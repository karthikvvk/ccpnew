"""
Text deduplication module for handling overlapping audio chunk boundaries
"""
from typing import List, Dict, Any, Optional
import re
from utils.logger import setup_logger

logger = setup_logger("text_dedup")


class TextDeduplicator:
    """
    Handles text overlap deduplication for chunks processed with audio overlap.
    Uses token-level comparison and optional embedding similarity.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize text deduplicator
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicate detection
        """
        self.similarity_threshold = similarity_threshold
        self._tokenizer = None
        self._embedder = None
    
    def merge_overlapping_segments(
        self, 
        segments: List[Dict[str, Any]],
        overlap_tokens: int = 20,
        use_embedding: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Merge segments from overlapping audio sub-chunks.
        Keeps earlier timestamp for duplicates.
        
        Args:
            segments: List of segment dictionaries with 'start', 'end', 'text'
            overlap_tokens: Number of tokens to check for overlap
            use_embedding: Whether to use embedding similarity (slower but more accurate)
            
        Returns:
            Deduplicated and merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        
        if len(sorted_segments) <= 1:
            return sorted_segments
        
        logger.info(f"Deduplicating {len(sorted_segments)} segments with {overlap_tokens} token overlap check")
        
        merged = [sorted_segments[0]]
        duplicates_removed = 0
        
        for current in sorted_segments[1:]:
            previous = merged[-1]
            
            # Check for time overlap
            if current['start'] < previous['end']:
                # Potential duplicate due to audio overlap
                similarity = self.compute_similarity(
                    previous.get('text', ''),
                    current.get('text', ''),
                    overlap_tokens,
                    use_embedding
                )
                
                if similarity >= self.similarity_threshold:
                    # Duplicate detected - merge or skip
                    # Keep earlier timestamp, extend end if needed
                    if current['end'] > previous['end']:
                        # Extend previous segment with new content
                        merged[-1] = self._merge_segments(previous, current, overlap_tokens)
                    duplicates_removed += 1
                    continue
            
            # No overlap or not similar enough - add as new segment
            merged.append(current)
        
        logger.info(f"Removed {duplicates_removed} duplicate segments, {len(merged)} remaining")
        return merged
    
    def _merge_segments(
        self, 
        seg1: Dict[str, Any], 
        seg2: Dict[str, Any],
        overlap_tokens: int
    ) -> Dict[str, Any]:
        """
        Merge two overlapping segments
        
        Args:
            seg1: Earlier segment
            seg2: Later segment
            overlap_tokens: Number of tokens in overlap region
            
        Returns:
            Merged segment
        """
        text1 = seg1.get('text', '')
        text2 = seg2.get('text', '')
        
        # Find the overlap point
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        # Find best overlap alignment
        best_overlap = 0
        for i in range(min(len(tokens1), overlap_tokens)):
            overlap_len = len(tokens1) - i
            if overlap_len <= len(tokens2):
                if tokens1[i:] == tokens2[:overlap_len]:
                    best_overlap = overlap_len
                    break
        
        if best_overlap > 0:
            # Merge with detected overlap
            merged_text = ' '.join(tokens1 + tokens2[best_overlap:])
        else:
            # No clear overlap, concatenate
            merged_text = text1 + ' ' + text2
        
        return {
            'start': seg1['start'],
            'end': max(seg1['end'], seg2['end']),
            'text': merged_text.strip(),
            'merged': True,
            'original_segments': [seg1, seg2]
        }
    
    def compute_similarity(
        self, 
        text1: str, 
        text2: str,
        overlap_tokens: int = 20,
        use_embedding: bool = False
    ) -> float:
        """
        Compute similarity between text segments
        
        Args:
            text1: First text
            text2: Second text
            overlap_tokens: Number of tokens to compare
            use_embedding: Whether to use embedding similarity
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        if use_embedding:
            return self._embedding_similarity(text1, text2)
        else:
            return self._token_similarity(text1, text2, overlap_tokens)
    
    def _token_similarity(self, text1: str, text2: str, overlap_tokens: int) -> float:
        """
        Compute token-level similarity between text endings and beginnings
        
        Args:
            text1: First text (check ending)
            text2: Second text (check beginning)
            overlap_tokens: Number of tokens to compare
            
        Returns:
            Similarity score
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Get the overlap regions
        end_tokens = tokens1[-overlap_tokens:] if len(tokens1) >= overlap_tokens else tokens1
        start_tokens = tokens2[:overlap_tokens] if len(tokens2) >= overlap_tokens else tokens2
        
        # Count matching tokens
        matches = 0
        min_len = min(len(end_tokens), len(start_tokens))
        
        for i in range(min_len):
            if end_tokens[-(min_len-i):] == start_tokens[:min_len-i]:
                matches = min_len - i
                break
        
        if min_len == 0:
            return 0.0
        
        return matches / min_len
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Compute embedding-based cosine similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self._embedder.encode([text1, text2])
            
            # Cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            
            cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
            return float(cos_sim)
            
        except Exception as e:
            logger.warning(f"Embedding similarity failed, falling back to token: {e}")
            return self._token_similarity(text1, text2, 20)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple word-level tokenization
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Simple tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def align_timestamps(
        self, 
        segments: List[Dict[str, Any]], 
        audio_offset: float
    ) -> List[Dict[str, Any]]:
        """
        Adjust segment timestamps by audio offset
        
        Args:
            segments: List of segments with 'start' and 'end'
            audio_offset: Offset in seconds to add to timestamps
            
        Returns:
            Segments with adjusted timestamps
        """
        aligned = []
        
        for segment in segments:
            aligned_segment = segment.copy()
            aligned_segment['start'] = segment.get('start', 0) + audio_offset
            aligned_segment['end'] = segment.get('end', 0) + audio_offset
            aligned_segment['original_start'] = segment.get('start', 0)
            aligned_segment['original_end'] = segment.get('end', 0)
            aligned.append(aligned_segment)
        
        return aligned
    
    def validate_continuity(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate timestamp continuity across segments
        
        Args:
            segments: List of segments
            
        Returns:
            Validation result with any issues found
        """
        issues = []
        
        for i in range(1, len(segments)):
            prev = segments[i-1]
            curr = segments[i]
            
            # Check for timestamp regression
            if curr['start'] < prev['end']:
                gap = curr['start'] - prev['end']
                if gap < -0.5:  # More than 0.5s regression
                    issues.append({
                        'type': 'regression',
                        'segment_index': i,
                        'gap': gap,
                        'prev_end': prev['end'],
                        'curr_start': curr['start']
                    })
            
            # Check for large gaps
            elif curr['start'] - prev['end'] > 2.0:  # More than 2s gap
                issues.append({
                    'type': 'gap',
                    'segment_index': i,
                    'gap': curr['start'] - prev['end'],
                    'prev_end': prev['end'],
                    'curr_start': curr['start']
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'segment_count': len(segments)
        }
