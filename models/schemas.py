"""
Pydantic models and schemas for API requests/responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationRequest(BaseModel):
    """Request model for video translation"""
    target_language: str = Field(..., description="Target language for translation (e.g., 'Spanish', 'French')")
    source_language: Optional[str] = Field(default="auto", description="Source language (auto-detect if not specified)")
    frame_fps: Optional[int] = Field(default=1, description="Frames per second to extract")
    use_rag: Optional[bool] = Field(default=True, description="Whether to use RAG context")
    whisper_model: Optional[str] = Field(default="medium", description="Whisper model size")
    

class FileInfo(BaseModel):
    """File information model"""
    file_type: str
    file_path: str
    exists: bool


class TranslationResponse(BaseModel):
    """Response model for translation"""
    job_id: str
    status: ProcessingStatus
    message: str
    files: Optional[List[FileInfo]] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for status check"""
    job_id: str
    status: ProcessingStatus
    progress: Optional[str] = None
    files: Optional[List[FileInfo]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
