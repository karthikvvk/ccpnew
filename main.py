"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router
from api.streaming import router as streaming_router
from models.schemas import HealthResponse
from config import settings
from utils.logger import setup_logger

logger = setup_logger("main")

app = FastAPI(
    title=settings.app_name,
    description="Video Translation API with RAG-enhanced transcription",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["translation"])
app.include_router(streaming_router, prefix="/api/v1")


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Whisper model: {settings.whisper_model}")
    logger.info(f"Translation model: {settings.translation_model}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
