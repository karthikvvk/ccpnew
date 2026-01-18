"""
Configuration management using JSON settings file
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Project root directory (where config.py is located)
PROJECT_ROOT = Path(__file__).parent.resolve()


def _load_json_config(config_path: Path = None) -> dict:
    """Load configuration from JSON file"""
    if config_path is None:
        config_path = PROJECT_ROOT / "settings.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class Settings:
    """Application settings loaded from settings.json"""
    
    # Application
    app_name: str = "VideoTranslationAPI"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File Storage
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    upload_dir: Path = field(default_factory=lambda: Path("./uploads"))
    max_file_size: int = 1073741824  # 1GB
    
    # Video Processing
    frame_extract_fps: int = 2
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    
    # Whisper Settings (GPU-only)
    whisper_model: str = "medium"
    whisper_device: str = "cuda"
    whisper_language: str = "auto"
    
    # Translation Settings (NLLB-200)
    translation_model: str = "facebook/nllb-200-1.3B"
    translation_device: str = "cuda"
    translation_max_length: int = 256
    
    # Vector Database
    vector_db_path: Path = field(default_factory=lambda: Path("./vector_db"))
    embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    
    # TTS Settings
    tts_slow: bool = False
    tts_lang: str = "en"
    
    # Voice Cloning Settings (XTTS-v2)
    use_voice_cloning: bool = False
    tts_device: str = "cuda"
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("./logs/app.log"))
    
    # Streaming Settings
    streaming_enabled: bool = False
    streaming_interval: float = 2.0
    streaming_buffer_size: int = 10
    streaming_sample_rate: int = 48000
    streaming_channels: int = 1
    
    # RAG configuration
    rag_enable_self_pruning: bool = False
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_json(cls, config_path: Path = None) -> "Settings":
        """Load settings from JSON file"""
        config = _load_json_config(config_path)
        
        return cls(
            # App settings
            app_name=config.get("app", {}).get("name", "VideoTranslationAPI"),
            debug=config.get("app", {}).get("debug", True),
            host=config.get("app", {}).get("host", "0.0.0.0"),
            port=config.get("app", {}).get("port", 8000),
            
            # Storage settings
            output_dir=Path(config.get("storage", {}).get("output_dir", "./outputs")),
            upload_dir=Path(config.get("storage", {}).get("upload_dir", "./uploads")),
            max_file_size=config.get("storage", {}).get("max_file_size", 1073741824),
            
            # Video settings
            frame_extract_fps=config.get("video", {}).get("frame_extract_fps", 2),
            video_codec=config.get("video", {}).get("video_codec", "libx264"),
            audio_codec=config.get("video", {}).get("audio_codec", "aac"),
            
            # Whisper settings
            whisper_model=config.get("whisper", {}).get("model", "medium"),
            whisper_device=config.get("whisper", {}).get("device", "cuda"),
            whisper_language=config.get("whisper", {}).get("language", "auto"),
            
            # Translation settings
            translation_model=config.get("translation", {}).get("model", "facebook/nllb-200-1.3B"),
            translation_device=config.get("translation", {}).get("device", "cuda"),
            translation_max_length=config.get("translation", {}).get("max_length", 256),
            
            # Vector DB settings
            vector_db_path=Path(config.get("vector_db", {}).get("path", "./vector_db")),
            embedding_model=config.get("vector_db", {}).get("embedding_model", "sentence-transformers/clip-ViT-B-32"),
            
            # TTS settings
            tts_slow=config.get("tts", {}).get("slow", False),
            tts_lang=config.get("tts", {}).get("lang", "en"),
            use_voice_cloning=config.get("tts", {}).get("use_voice_cloning", False),
            tts_device=config.get("tts", {}).get("device", "cuda"),
            
            # Logging settings
            log_level=config.get("logging", {}).get("level", "INFO"),
            log_file=Path(config.get("logging", {}).get("file", "./logs/app.log")),
            
            # Streaming settings
            streaming_enabled=config.get("streaming", {}).get("enabled", False),
            streaming_interval=config.get("streaming", {}).get("interval", 2.0),
            streaming_buffer_size=config.get("streaming", {}).get("buffer_size", 10),
            streaming_sample_rate=config.get("streaming", {}).get("sample_rate", 48000),
            streaming_channels=config.get("streaming", {}).get("channels", 1),
            
            # RAG settings
            rag_enable_self_pruning=config.get("rag", {}).get("enable_self_pruning", False),
        )


# Global settings instance
settings = Settings.from_json()
