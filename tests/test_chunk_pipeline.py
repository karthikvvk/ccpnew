"""
Test script for chunk-based pipeline logic (Mocked)
"""
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock heavy dependencies BEFORE importing pipeline
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['ffmpeg'] = MagicMock()
sys.modules['whisper'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['bitsandbytes'] = MagicMock()
# Complex mocking for chromadb package structure
chromadb = MagicMock()
sys.modules['chromadb'] = chromadb
sys.modules['chromadb.config'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pydub'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.io'] = MagicMock()
sys.modules['scipy.io.wavfile'] = MagicMock()
sys.modules['gtts'] = MagicMock()
sys.modules['deep_translator'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from services.pipeline import TranslationPipeline
from config import settings

def test_pipeline():
    print("Setting up test environment...")
    
    # Setup paths
    test_dir = Path("./test_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    settings.output_dir = test_dir
    
    # Create dummy video
    video_path = test_dir / "test_video.mp4"
    with open(video_path, "wb") as f:
        f.write(b"dummy video content")
        
    print(f"Created dummy video at {video_path}")

    # Mock dependencies
    with patch('services.pipeline.VideoProcessor') as MockVP, \
         patch('services.pipeline.SpeechToText') as MockSTT, \
         patch('services.pipeline.Translator') as MockTrans, \
         patch('services.pipeline.TextToSpeech') as MockTTS, \
         patch('modules.chunk_manager.ChunkManager') as MockCM, \
         patch('modules.text_dedup.TextDeduplicator') as MockDedup, \
         patch('modules.frame_embedder.FrameEmbedder') as MockEmbedder, \
         patch('modules.semantic_rag.SemanticRAG') as MockRAG:
        
        # Configure VideoProcessor Mock
        vp_instance = MockVP.return_value
        vp_instance.get_video_info.return_value = {'duration': 600.0} # 10 min video
        
        # Mock split_video to return 2 chunks
        MockVP.split_video.return_value = [
            {'path': test_dir / 'chunk_0.mp4', 'start_time': 0, 'duration': 300, 'chunk_id': 0},
            {'path': test_dir / 'chunk_1.mp4', 'start_time': 300, 'duration': 300, 'chunk_id': 1}
        ]
        
        # Mock split_audio to return subchunks
        MockVP.split_audio.return_value = [
            {'path': test_dir / 'sub_0.wav', 'start_time': 0, 'end_time': 30, 'subchunk_id': 0}
        ]
        
        # Configure STT Mock
        stt_instance = MockSTT.return_value
        stt_instance.transcribe_with_confidence.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Hello world'}],
            'confidence': {'avg_logprob': -0.5},
            'text': 'Hello world'
        }
        
        # Configure Translator Mock
        trans_instance = MockTrans.return_value
        trans_instance.translate_segments.return_value = [
            {'start': 0, 'end': 5, 'text': 'Hola mundo'}
        ]

        # Configure actual pipeline to use mocked classes
        # Note: We need to ensure the pipeline uses our mocks. 
        # Since we patched the classes imported in pipeline.py, instances created inside pipeline.py will be mocks.
        
        pipeline = TranslationPipeline("test_job")
        
        # Run process
        print("Running pipeline.process()...")
        try:
            result = pipeline.process(
                video_path=video_path,
                target_language="spanish",
                use_rag=True
            )
            print("Pipeline completed successfully!")
            print(f"Result status: {result['status']}")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
