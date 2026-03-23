"""
FastAPI endpoints for video translation API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from pathlib import Path
import shutil
from typing import Optional
from models.schemas import (
    TranslationRequest, 
    TranslationResponse, 
    StatusResponse,
    ProcessingStatus,
    FileInfo
)
from services.pipeline import TranslationPipeline
from utils.logger import setup_logger
from config import settings
import uuid

logger = setup_logger("api_endpoints")
router = APIRouter()

# Job status tracking (in-memory for now)
job_status = {}


def process_video_background(job_id: str, 
                             video_path: Path,
                             target_language: str,
                             source_language: str,
                             use_rag: bool):
    """
    Background task for video processing
    
    Args:
        job_id: Job identifier
        video_path: Path to video file
        target_language: Target language
        source_language: Source language
        use_rag: Whether to use RAG
    """
    try:
        logger.info(f"Starting background processing for job: {job_id}")
        
        # Update status
        job_status[job_id] = {
            'status': ProcessingStatus.PROCESSING,
            'progress': 'Processing video...'
        }
        
        # Run pipeline
        pipeline = TranslationPipeline(job_id)
        result = pipeline.process(
            video_path,
            target_language,
            source_language,
            use_rag
        )
        
        # Update status
        job_status[job_id] = {
            'status': ProcessingStatus.COMPLETED,
            'result': result,
            'progress': 'Completed'
        }
        
        logger.info(f"Background processing completed for job: {job_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}", exc_info=True)
        job_status[job_id] = {
            'status': ProcessingStatus.FAILED,
            'error': str(e),
            'progress': 'Failed'
        }


@router.post("/translate", response_model=TranslationResponse)
async def translate_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str = Form("auto"),
    use_rag: bool = Form(False)
):
    """
    Translate video to target language
    
    Args:
        background_tasks: FastAPI background tasks
        video: Video file upload
        target_language: Target language
        source_language: Source language (auto-detect if 'auto')
        use_rag: Whether to use RAG context
        
    Returns:
        Translation response with job ID
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        logger.info(f"Received translation request for job: {job_id}")
        logger.info(f"  Target language: {target_language}")
        logger.info(f"  Source language: {source_language}")
        logger.info(f"  Use RAG: {use_rag}")
        
        # Save uploaded video
        upload_path = settings.upload_dir / job_id
        upload_path.mkdir(parents=True, exist_ok=True)
        
        video_path = upload_path / video.filename
        with open(video_path, 'wb') as f:
            shutil.copyfileobj(video.file, f)
        
        logger.info(f"Saved uploaded video: {video_path}")
        
        # Initialize job status
        job_status[job_id] = {
            'status': ProcessingStatus.PENDING,
            'progress': 'Queued for processing'
        }
        
        # Add background task
        background_tasks.add_task(
            process_video_background,
            job_id,
            video_path,
            target_language,
            source_language,
            use_rag
        )
        
        return TranslationResponse(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            message="Translation job queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to queue translation job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Get processing status for a job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Status response
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status[job_id]
    
    # Prepare file info if completed
    files = None
    if status_info['status'] == ProcessingStatus.COMPLETED:
        result = status_info.get('result', {})
        tracked_files = result.get('files', {})
        
        files = [
            FileInfo(
                file_type=file_type,
                file_path=str(file_path),
                exists=Path(file_path).exists()
            )
            for file_type, file_path in tracked_files.items()
        ]
    
    return StatusResponse(
        job_id=job_id,
        status=status_info['status'],
        progress=status_info.get('progress'),
        files=files,
        error=status_info.get('error')
    )


@router.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """
    Download a specific file from a job
    
    Args:
        job_id: Job identifier
        file_type: Type of file to download
        
    Returns:
        File response
    """
    from fastapi.responses import FileResponse
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status[job_id]
    
    if status_info['status'] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result = status_info.get('result', {})
    tracked_files = result.get('files', {})
    
    if file_type not in tracked_files:
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found")
    
    file_path = Path(tracked_files[file_type])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type='application/octet-stream'
    )


def generate_srt_content(segments: list) -> str:
    """Generate SRT subtitle content from translated segments"""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('translated', segment.get('text', ''))
        
        # Convert seconds to SRT time format (HH:MM:SS,mmm)
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_lines.append(str(i))
        srt_lines.append(f"{format_time(start)} --> {format_time(end)}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)


@router.get("/jobs/{job_id}/download/audio")
async def download_audio(job_id: str):
    """Download dubbed audio file"""
    from fastapi.responses import FileResponse
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status[job_id]
    if status_info['status'] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result = status_info.get('result', {})
    tracked_files = result.get('files', {})
    
    # Try dubbed_audio first, then original_audio
    audio_path = tracked_files.get('dubbed_audio') or tracked_files.get('original_audio')
    
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_path),
        filename=f"{job_id}_audio.wav",
        media_type='audio/wav'
    )


@router.get("/jobs/{job_id}/download/srt")
async def download_srt(job_id: str):
    """Download SRT subtitle file"""
    from fastapi.responses import Response
    import json
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status[job_id]
    if status_info['status'] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result = status_info.get('result', {})
    tracked_files = result.get('files', {})
    
    # Try to load translation JSON to generate SRT
    translation_json = tracked_files.get('translation_json')
    
    if translation_json and Path(translation_json).exists():
        with open(translation_json, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        srt_content = generate_srt_content(segments)
        
        return Response(
            content=srt_content,
            media_type='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename="{job_id}_subtitles.srt"'
            }
        )
    
    # Fallback to translation_txt
    translation_txt = tracked_files.get('translation_txt')
    if translation_txt and Path(translation_txt).exists():
        with open(translation_txt, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Response(
            content=content,
            media_type='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename="{job_id}_translation.txt"'
            }
        )
    
    raise HTTPException(status_code=404, detail="Translation file not found")


@router.get("/jobs/{job_id}/download/video")
async def download_video(job_id: str):
    """Download final translated video"""
    from fastapi.responses import FileResponse
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status[job_id]
    if status_info['status'] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result = status_info.get('result', {})
    tracked_files = result.get('files', {})
    
    video_path = tracked_files.get('final_video')
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Final video not found")
    
    return FileResponse(
        path=str(video_path),
        filename=f"{job_id}_translated.mp4",
        media_type='video/mp4'
    )


@router.get("/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List all available files for a job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status[job_id]
    if status_info['status'] != ProcessingStatus.COMPLETED:
        return {"job_id": job_id, "status": str(status_info['status']), "files": []}
    
    result = status_info.get('result', {})
    tracked_files = result.get('files', {})
    
    files_info = []
    for file_type, file_path in tracked_files.items():
        path = Path(file_path)
        files_info.append({
            "type": file_type,
            "path": str(file_path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "download_url": f"/api/v1/download/{job_id}/{file_type}"
        })
    
    return {
        "job_id": job_id,
        "status": "completed",
        "files": files_info,
        "quick_downloads": {
            "audio": f"/api/v1/jobs/{job_id}/download/audio",
            "srt": f"/api/v1/jobs/{job_id}/download/srt",
            "video": f"/api/v1/jobs/{job_id}/download/video"
        }
    }


# ─────────────────────────────────────────────────────────
# API MODE HELPER ROUTES  (no ML models involved)
# Used by the frontend "API Mode" fast path for demo
# ─────────────────────────────────────────────────────────

@router.get("/settings")
async def get_settings():
    """Return api_keys + demo sections from settings.json for the frontend API mode."""
    import json
    settings_path = Path(__file__).parent.parent / "settings.json"
    try:
        with open(settings_path, "r") as f:
            raw = json.load(f)
        return {
            "api_keys": raw.get("api_keys", {}),
            "demo":     raw.get("demo", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read settings: {e}")


@router.post("/extract-audio")
async def extract_audio(video: UploadFile = File(...)):
    """Extract audio from uploaded video as 16kHz mono WAV for Whisper STT."""
    import subprocess, tempfile, os
    from fastapi.responses import Response

    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.TemporaryDirectory() as tmp:
        in_path  = os.path.join(tmp, f"input{suffix}")
        out_path = os.path.join(tmp, "audio.wav")
        with open(in_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg extraction failed: {result.stderr.decode()}")
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
    return Response(content=audio_bytes, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=audio.wav"})


@router.post("/mux-audio")
async def mux_audio(
    video:  UploadFile = File(...),
    audio:  UploadFile = File(...),
    job_id: Optional[str] = Form(None),
):
    """Replace audio track of video with dubbed audio blob; returns MP4."""
    import subprocess, tempfile, os, uuid
    from fastapi.responses import FileResponse

    jid      = job_id or str(uuid.uuid4())
    v_suffix = Path(video.filename).suffix or ".mp4"
    a_suffix = Path(audio.filename).suffix or ".mp3"
    out_dir  = settings.output_dir / jid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "api_dubbed.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        in_v = os.path.join(tmp, f"video{v_suffix}")
        in_a = os.path.join(tmp, f"audio{a_suffix}")
        with open(in_v, "wb") as f:
            shutil.copyfileobj(video.file, f)
        with open(in_a, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_v, "-i", in_a,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg mux failed: {result.stderr.decode()}")

    return FileResponse(path=str(out_path), filename=f"{jid}_api_dubbed.mp4", media_type="video/mp4")

@router.post("/tts")
async def generate_tts(
    text:           str   = Form(...),
    lang:           str   = Form("EN"),
    video_duration: float = Form(0.0),   # seconds — 0 means no speed adjustment
):
    """
    Generate TTS with edge-tts, then optionally speed-match to video_duration
    using ffmpeg atempo filter (same algorithm as the local pipeline).
    Clamped to 0.5×–2.0× like the backend does.
    """
    import subprocess, tempfile, os, uuid
    from fastapi.responses import FileResponse

    voice_map = {
        "TA": "ta-IN-ValluvarNeural",
        "JP": "ja-JP-KeitaNeural",
        "DE": "de-DE-KillianNeural",
        "EN": "en-US-ChristopherNeural",
        "ES": "es-ES-AlvaroNeural",
        "FR": "fr-FR-HenriNeural",
        "ZH": "zh-CN-YunxiNeural",
        "KO": "ko-KR-InJoonNeural",
        "HI": "hi-IN-MadhurNeural",
        "AR": "ar-SA-HamedNeural",
    }
    voice = voice_map.get(lang.upper(), "en-US-ChristopherNeural")
    uid   = str(uuid.uuid4())
    out_dir = settings.output_dir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = out_dir / f"{uid}_raw.mp3"
    final_path   = out_dir / f"{uid}_final.mp3"

    # ── 1. Generate TTS audio with edge-tts ──────────────────────────
    result = subprocess.run(
        ["edge-tts", "--voice", voice, "--text", text, "--write-media", str(raw_path)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"edge-tts failed: {result.stderr.decode()}")

    # ── 2. Speed-match to video_duration if provided ─────────────────
    if video_duration > 0.1:
        # Probe generated TTS duration with ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(raw_path)],
            capture_output=True, text=True,
        )
        try:
            tts_duration = float(probe.stdout.strip())
        except ValueError:
            tts_duration = 0.0

        if tts_duration > 0.1:
            speed_factor = tts_duration / video_duration     # >1 → speed up, <1 → slow down
            speed_factor = max(0.5, min(2.0, speed_factor))  # clamp same as backend

            if abs(speed_factor - 1.0) > 0.01:
                # Build chained atempo filters (each step limited to 0.5–2.0)
                # e.g. 2.0× is fine; if we needed 3× we'd chain two 1.5× but clamp handles it
                atempo_filter = f"atempo={speed_factor:.4f}"

                adjust = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(raw_path),
                     "-filter:a", atempo_filter,
                     "-c:a", "libmp3lame", "-q:a", "2",
                     str(final_path)],
                    capture_output=True,
                )
                if adjust.returncode == 0:
                    raw_path.unlink(missing_ok=True)
                    return FileResponse(path=str(final_path), media_type="audio/mpeg")
                # fallthrough to return raw if atempo fails

    return FileResponse(path=str(raw_path), media_type="audio/mpeg")

