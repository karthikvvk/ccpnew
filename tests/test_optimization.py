import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.semantic_rag import SemanticRAG
from services.pipeline import TranslationPipeline
from modules.frame_embedder import FrameEmbedder

class TestOptimization(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.rag = SemanticRAG(embedder=self.mock_embedder)

    def test_vectorized_similarity(self):
        """Verify vectorized similarity matches manual calculation"""
        # Dictionary of fake embeddings
        vocab = ["cat", "dog", "car"]
        self.rag._vocab_embeddings = {
            "cat": np.array([1.0, 0.0, 0.0]),
            "dog": np.array([0.9, 0.1, 0.0]),
            "car": np.array([0.0, 1.0, 0.0])
        }
        
        # Frame embedding (close to cat/dog)
        frame_emb = np.array([0.95, 0.05, 0.0])
        
        # Run verification
        results = self.rag.verify_space_and_filter(frame_emb, vocab, threshold=0.1)
        
        # Check results
        # Expected scores:
        # cat: dot([1,0,0], [0.95, 0.05, 0]) / norms = 0.95
        # dog: dot([0.9,0.1,0], [0.95, 0.05, 0]) / norms approx (0.855 + 0.005) = 0.86
        # car: dot([0,1,0], [0.95, 0.05, 0]) / norms = 0.05 (should be filtered if threshold > 0.05)
        
        found_words = [r[0] for r in results]
        self.assertIn("cat", found_words)
        self.assertIn("dog", found_words)
        
        # Check ordering
        self.assertEqual(results[0][0], "cat")
        
        print("\nVectorized RAG test passed!")

    @patch('services.pipeline.FrameEmbedder')
    @patch('services.pipeline.SpeechToText')
    @patch('services.pipeline.Translator')
    @patch('services.pipeline.VideoProcessor')
    @patch('services.pipeline.FileManager')
    @patch('services.pipeline.ChunkManager')
    @patch('services.pipeline.TextDeduplicator')
    def test_pipeline_embedder_passing(self, MockTextDedup, MockChunkMgr, MockFileMgr, MockVP, MockTranslator, MockSTT, MockEmbedder):
        """Verify FrameEmbedder is initialized once and passed down"""
        
        pipeline = TranslationPipeline("test_job")
        
        # Mock file manager tracking
        pipeline.file_manager.track_file = MagicMock()
        pipeline.file_manager.job_dir = Path("/tmp/test_job")
        
        # Mock video splitting to return 1 chunk
        mock_vp_instance = MockVP.return_value
        mock_vp_instance.split_video.return_value = [{'chunk_id': 0, 'path': Path('chunk.mp4'), 'start_time': 0, 'duration': 10}]
        mock_vp_instance.get_video_info.return_value = {'duration': 10}
        
        # Mock global RAG to return None (simplify)
        pipeline._perform_global_rag = MagicMock(return_value=None)
        
        # Mock process_chunk to capture arguments
        pipeline.process_chunk = MagicMock(return_value={
            'chunk_id': 0, 
            'status': 'completed', 
            'tts_path': '/tmp/tts.wav'
        })
        
        # Run process
        pipeline.process(Path("video.mp4"), "es", use_rag=True)
        
        # Verify Embedder initialized once
        MockEmbedder.assert_called_once()
        
        # Verify passed to process_chunk
        args, kwargs = pipeline.process_chunk.call_args
        self.assertIn('embedder', kwargs)
        self.assertIsNotNone(kwargs['embedder'])
        
        print("Pipeline embedder passing test passed!")

from pathlib import Path

if __name__ == '__main__':
    unittest.main()
