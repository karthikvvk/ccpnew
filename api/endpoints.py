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
