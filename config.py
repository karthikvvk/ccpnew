"""
Configuration management using Pydantic settings
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "VideoTranslationAPI"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File Storage
    output_dir: Path = Path("./outputs")
    upload_dir: Path = Path("./uploads")
    max_file_size: int = 1073741824  # 1GB
    
    # Video Processing
    frame_extract_fps: int = 1
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    
    # Whisper Settings
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_language: str = "auto"
    
    # LLM Settings
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_device: str = "cpu"
    llm_max_length: int = 2048
    llm_temperature: float = 0.7
    
    # Vector Database
    vector_db_path: Path = Path("./vector_db")
    embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    
    # TTS Settings
    tts_slow: bool = False
    tts_lang: str = "en"
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/app.log")
    
    # Colab GPU Server (Optional)
    use_colab_gpu: bool = False
    colab_api_url: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
