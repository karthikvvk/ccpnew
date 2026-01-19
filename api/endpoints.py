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
    use_rag: bool = Form(True)
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

