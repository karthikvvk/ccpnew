"""
Main orchestration pipeline for video translation
FLOW: Video Split -> Global RAG -> Chunked (STT -> Local RAG -> Translate -> TTS) -> Merge -> Reconstruct
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
from utils.logger import setup_logger
from utils.file_manager import FileManager
from modules.video_processor import VideoProcessor
from modules.frame_embedder import FrameEmbedder
from modules.vector_store import VectorStore
from modules.speech_to_text import SpeechToText
from modules.simple_translator import Translator
from modules.text_to_speech import TextToSpeech
from modules.chunk_manager import ChunkManager
from modules.text_dedup import TextDeduplicator
from modules.voice_cloner import VoiceCloner, get_language_code
from config import settings

logger = setup_logger("pipeline")


class TranslationPipeline:
    """
    Orchestrates the complete video translation workflow with multi-level chunking
    """
    
    def __init__(self, job_id: str = None):
        """
        Initialize pipeline
        
        Args:
            job_id: Job identifier (generated if not provided)
        """
        self.job_id = job_id
        self.file_manager = FileManager(job_id)
        logger.info(f"Initialized translation pipeline for job: {self.file_manager.job_id}")
        self.job_id = self.file_manager.job_id
        
        # Initialize sub-modules
        self.chunk_manager = ChunkManager(self.file_manager.job_dir)
        self.text_dedup = TextDeduplicator(similarity_threshold=settings.dedup_similarity_threshold)
        
        # Models (lazy loaded in methods, but we can init lightweight ones here)
        self.video_processor = VideoProcessor
    
    def process(self,
                video_path: Path,
                target_language: str,
                source_language: str = "auto",
                use_rag: bool = True) -> Dict[str, Any]:
        """
        Process video translation using robust chunking pipeline
        """
        try:
            logger.info(f"Starting chunk-based translation pipeline")
            logger.info(f"  Video: {video_path}")
            logger.info(f"  Target: {target_language}")
            
            # 1. Get video info and split
            logger.info("Step 1: Analyzing and splitting video...")
            video_processor = self.video_processor(video_path)
            video_info = video_processor.get_video_info()
            
            video_chunks = self.video_processor.split_video(
                video_path,
                duration=settings.video_chunk_duration,
                overlap=0, # Video chunks don't overlap, audio does inside chunks
                output_dir=self.file_manager.job_dir / 'video_chunks'
            )
            
            # Track chunks
            self.file_manager.track_file('video_chunks_dir', self.file_manager.job_dir / 'video_chunks')
            
            # 2. Global RAG (Video-Level)
            global_context = None
            if use_rag:
                logger.info("Step 2: Performing Global RAG analysis...")
                global_context = self._perform_global_rag(video_path)
                
                if global_context:
                    rag_path = self.file_manager.job_dir / 'global_rag.json'
                    with open(rag_path, 'w', encoding='utf-8') as f:
                        json.dump(global_context, f, indent=2, ensure_ascii=False)
                    self.file_manager.track_file('global_rag', rag_path)
            
            # 3. Process Chunks
            logger.info(f"Step 3: Processing {len(video_chunks)} chunks...")
            chunk_results = []
            previous_context = {}  # For context carryover
            
            # Initialize STT/TTS models once
            stt = SpeechToText()
            translator = Translator()
            
            if settings.use_voice_cloning:
                # We need voice cloner, but maybe we only init it inside process_chunk or pass it
                pass 
            
            for i, chunk_info in enumerate(video_chunks):
                chunk_id = chunk_info['chunk_id']
                logger.info(f"--- Processing Chunk {chunk_id+1}/{len(video_chunks)} ---")
                
                result = self.process_chunk(
                    chunk_id=chunk_id,
                    chunk_video_path=chunk_info['path'],
                    target_language=target_language,
                    source_language=source_language,
                    global_context=global_context,
                    previous_context=previous_context,
                    stt_model=stt,
                    translator_model=translator
                )
                
                chunk_results.append(result)
                
                # Update context for next chunk
                if result.get('metadata', {}).get('last_sentence'):
                    previous_context['last_sentence'] = result['metadata']['last_sentence']
            
            # 4. Timeline-Aware Merge
            logger.info("Step 4: Merging processed audio...")
            
            # Gather audio chunks to merge
            audio_chunks_to_merge = []
            for res in chunk_results:
                if res.get('tts_path'):
                    # Map chunk index to timeline position based on video split
                    chunk_idx = res['chunk_id']
                    video_chunk_info = video_chunks[chunk_idx]
                    
                    audio_chunks_to_merge.append({
                        'path': res['tts_path'],
                        'start_time': video_chunk_info['start_time'], # Align to video chunk start
                        'duration': VideoProcessor.get_audio_duration(res['tts_path'])
                    })
            
            merged_audio_path = self.file_manager.get_path('translated_audio')
            self.video_processor.merge_audios_timeline(
                audio_chunks_to_merge,
                merged_audio_path,
                crossfade_ms=settings.crossfade_ms,
                validate_duration=False # Duration might vary due to translation/speed changes
            )
            self.file_manager.track_file('translated_audio', merged_audio_path)
            
            # 5. Sync Audio Speed (Global Sync)
            logger.info("Step 5: Synchronizing audio duration...")
            synced_audio_path = self.video_processor.adjust_audio_speed(
                merged_audio_path,
                video_info['duration'],
                output_path=self.file_manager.job_dir / 'synced_audio.wav'
            )
            self.file_manager.track_file('synced_audio', synced_audio_path)
            
            # 6. Reconstruct Video
            logger.info("Step 6: Reconstructing final video...")
            final_video_path = self.file_manager.get_path('final_video')
            self.video_processor.reconstruct_video(video_path, synced_audio_path, final_video_path)
            self.file_manager.track_file('final_video', final_video_path)
            
            # 7. Safe Cleanup
            if not settings.keep_temp_files:
                logger.info("Step 7: Cleaning up temporary chunk files...")
                self.chunk_manager.cleanup_chunks(keep_failed=True)
            
            # Save Manifest
            manifest_path = self.file_manager.save_manifest()
            
            logger.info("Pipeline completed successfully!")
            return {
                'status': 'completed',
                'job_id': self.job_id,
                'files': self.file_manager.tracked_files,
                'manifest': str(manifest_path)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            # Try to save partial results/manifest
            try:
                self.file_manager.save_manifest()
            except:
                pass
            raise
            
    def _perform_global_rag(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Perform global RAG analysis on the video"""
        try:
            # We need to extract some frames globally
            # For efficiency, we can reuse the first chunk's frames if video is short,
            # but for long videos we should probably extract frames sparsely from whole video.
            # Here we'll do a sparse extraction from the whole video.
            
            global_frames_dir = self.file_manager.job_dir / 'global_frames'
            vp = self.video_processor(video_path)
            
            # Extract at very low FPS (e.g. 0.2 FPS = 1 frame every 5 sec)
            vp.extract_frames(global_frames_dir, fps=0.2)
            
            embedder = FrameEmbedder()
            frame_embeddings_list = embedder.embed_frames(global_frames_dir)
            embeddings = [fe[1] for fe in frame_embeddings_list]
            
            from modules.semantic_rag import SemanticRAG
            rag = SemanticRAG(embedder=embedder)
            
            return rag.analyze_global(embeddings)
            
        except Exception as e:
            logger.warning(f"Global RAG failed, proceeding without it: {e}")
            return None

    def process_chunk(self,
                      chunk_id: int,
                      chunk_video_path: Path,
                      target_language: str,
                      source_language: str,
                      global_context: Dict[str, Any],
                      previous_context: Dict[str, Any],
                      stt_model: SpeechToText,
                      translator_model: Translator) -> Dict[str, Any]:
        """
        Process a single video chunk
        """
        chunk_dir = self.chunk_manager.create_chunk(chunk_id)
        
        # Check if already completed
        status = self.chunk_manager.get_chunk_status(chunk_id)
        if status.get('completed'):
            logger.info(f"Chunk {chunk_id} already completed, skipping.")
            return {
                'chunk_id': chunk_id,
                'status': 'completed',
                'tts_path': self.chunk_manager.get_chunk_audio_path(chunk_id),
                'metadata': status.get('metadata', {})
            }
            
        try:
            self.chunk_manager.update_chunk_status(chunk_id, 'started', True)
            
            # 1. Extract Audio
            vp = self.video_processor(chunk_video_path)
            chunk_audio_path = chunk_dir / 'audio.wav'
            vp.extract_audio(chunk_audio_path)
            
            # 2. Extract Frames for Local RAG
            chunk_frames_dir = chunk_dir / 'frames'
            vp.extract_frames(chunk_frames_dir, fps=settings.frame_extract_fps)
            
            # 3. Split Audio into Sub-chunks
            audio_subchunks = self.video_processor.split_audio(
                chunk_audio_path,
                duration=settings.audio_subchunk_duration,
                overlap=settings.audio_overlap,
                output_dir=chunk_dir / 'audio_subchunks'
            )
            
            # 4. Transcribe Sub-chunks
            all_segments = []
            context_prompt = previous_context.get('last_sentence')
            
            for sub_info in audio_subchunks:
                sub_path = sub_info['path']
                start_offset = sub_info['start_time']
                
                transcription = stt_model.transcribe_with_confidence(
                    sub_path,
                    language=source_language,
                    context_prompt=context_prompt
                )
                
                # Check confidence
                if stt_model.is_low_confidence(transcription['confidence']):
                    logger.warning(f"Chunk {chunk_id}, subchunk {sub_info['subchunk_id']}: Low confidence transcription")
                    # Could implement retry logic here (e.g. with different temp)
                
                # Align timestamps relative to chunk start
                aligned_segments = self.text_dedup.align_timestamps(
                    transcription['segments'],
                    audio_offset=start_offset
                )
                
                all_segments.extend(aligned_segments)
                
                # Update prompt for next sub-chunk
                if transcription.get('text'):
                    context_prompt = transcription['text']  # Use full text as prompt
            
            # 5. Deduplicate Text
            merged_segments = self.text_dedup.merge_overlapping_segments(
                all_segments,
                overlap_tokens=settings.text_overlap_tokens
            )
            
            # 6. Local RAG
            local_context = None
            if global_context: # Only do local if RAG enabled (implied by global_context presence if use_rag=True)
                # Need embedding for frames
                embedder = FrameEmbedder() # Initialize locally to save VRAM if needed, or pass
                frame_embeddings = embedder.embed_frames(chunk_frames_dir)
                embeddings = [fe[1] for fe in frame_embeddings]
                
                from modules.semantic_rag import SemanticRAG
                rag = SemanticRAG(embedder=embedder)
                
                local_analysis = rag.analyze_chunk(embeddings, global_context)
                if local_analysis:
                    local_context = local_analysis['context']
            
            # 7. Translation
            detected_lang = source_language # Or from STT result
            if not merged_segments:
                logger.warning(f"Chunk {chunk_id}: No speech detected.")
                translated_segments = []
            else:
                translated_segments = translator_model.translate_segments(
                    merged_segments,
                    target_language,
                    source_language=detected_lang,
                    context=local_context
                )
            
            # Save results
            with open(chunk_dir / 'stt.json', 'w') as f:
                json.dump(merged_segments, f, indent=2)
            with open(chunk_dir / 'translation.json', 'w') as f:
                json.dump(translated_segments, f, indent=2)
                
            # 8. TTS
            tts_path = chunk_dir / 'tts.wav'
            
            if not translated_segments:
                # Create silent audio matches chunk duration? 
                # Or just empty file? Timeline merge should handle it.
                # For now create empty/silent file 
                # (actually, simple TTS handles empty segments list by producing nothing or silence)
                pass 
            
            if settings.use_voice_cloning:
                voice_cloner = VoiceCloner()
                voice_cloner.segments_to_dubbed_audio(
                    segments=translated_segments,
                    reference_audio=chunk_audio_path, # Use chunk audio as ref
                    output_path=tts_path,
                    language=get_language_code(target_language)
                )
            else:
                tts = TextToSpeech(language=self._get_tts_language_code(target_language))
                tts.segments_to_speech(translated_segments, tts_path)
            
            # Update status
            metadata = {
                'segments_count': len(merged_segments),
                'last_sentence': merged_segments[-1]['text'] if merged_segments else None,
                'tts_path': str(tts_path)
            }
            self.chunk_manager.update_chunk_status(chunk_id, 'completed', True, metadata)
            
            return {
                'chunk_id': chunk_id,
                'status': 'completed',
                'tts_path': tts_path,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_id}: {e}", exc_info=True)
            self.chunk_manager.update_chunk_status(chunk_id, 'failed', False, {'error': str(e)})
            raise

    def _get_tts_language_code(self, language: str) -> str:
        """Helper for gTTS language codes"""
        # (Same as before)
        language_map = {
            'spanish': 'es', 'french': 'fr', 'german': 'de', 'italian': 'it',
            'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja', 'korean': 'ko',
            'chinese': 'zh-cn', 'arabic': 'ar', 'hindi': 'hi', 'tamil': 'ta', 'english': 'en'
        }
        return language_map.get(language.lower(), 'en')
