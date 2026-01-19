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

from pyngrok import ngrok
import os

logger = setup_logger("main")

# Global tunnel handle
ngrok_tunnel = None

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Video Translation API with RAG-enhanced transcription",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["translation"])
app.include_router(streaming_router, prefix="/api/v1", tags=["streaming"])


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")


@app.on_event("startup")
async def startup_event():
    global ngrok_tunnel

    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Whisper model: {settings.whisper_model}")
    logger.info(f"Translation model: {settings.translation_model}")

    # ---- ngrok tunnel ----
    if "NGROK_AUTHTOKEN" in os.environ:
        ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))

        ngrok_tunnel = ngrok.connect(
            addr=settings.port,
            bind_tls=True
        )

        logger.info(f"ngrok tunnel started")
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")
    else:
        logger.warning("NGROK_AUTHTOKEN not set, ngrok tunnel skipped")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")

    if ngrok_tunnel:
        ngrok.disconnect(ngrok_tunnel.public_url)
        ngrok.kill()
        logger.info("ngrok tunnel closed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
