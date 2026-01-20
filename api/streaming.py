"""
Streaming API Endpoints
Real-time audio streaming and transcription endpoints
"""

from typing import Optional, List, Dict, Any
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse

from utils.logger import setup_logger
from config import settings

logger = setup_logger("streaming_api")

router = APIRouter(prefix="/stream", tags=["streaming"])

# Global instances (created on first use)
_streamer = None
_pipeline = None


def get_streamer():
    """Get or create AudioStreamer instance."""
    global _streamer
    if _streamer is None:
        from modules.streaming import AudioStreamer
        _streamer = AudioStreamer(
            interval_seconds=getattr(settings, 'streaming_interval', 0.5),
            buffer_size=getattr(settings, 'streaming_buffer_size', 10),
            sample_rate=getattr(settings, 'streaming_sample_rate', 48000),
            channels=getattr(settings, 'streaming_channels', 1)
        )
    return _streamer


def get_pipeline():
    """Get or create StreamingPipeline instance."""
    global _pipeline
    if _pipeline is None:
        from modules.streaming import StreamingPipeline
        _pipeline = StreamingPipeline(
            interval_seconds=getattr(settings, 'streaming_interval', 2.0),
            sample_rate=getattr(settings, 'streaming_sample_rate', 48000)
        )
    return _pipeline


# ==================== REST Endpoints ====================

@router.get("/devices")
async def list_devices():
    """List available audio input devices."""
    try:
        from modules.device_tester import list_input_devices
        devices = list_input_devices(include_all=True)
        return {"devices": devices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detect")
async def detect_device(duration: float = 1.0):
    """Detect audio device with active audio."""
    try:
        from modules.device_tester import find_working_device
        device = find_working_device(test_duration=duration)
        if device is not None:
            return {"device": device, "status": "detected"}
        else:
            return {"device": None, "status": "no_audio_detected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_streaming(device: Optional[int] = None):
    """Start audio capture."""
    try:
        streamer = get_streamer()
        if streamer.is_streaming():
            return {"status": "already_running", "queue_size": streamer.queue_size()}
        
        streamer.start(device=device)
        return {"status": "started", "device": device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_streaming():
    """Stop audio capture."""
    try:
        streamer = get_streamer()
        streamer.stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def stream_status():
    """Get streaming status."""
    try:
        streamer = get_streamer()
        return {
            "streaming": streamer.is_streaming(),
            "queue_size": streamer.queue_size()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pop")
async def pop_chunk(timeout: float = 0.5):
    """Pop next audio chunk from queue."""
    import base64
    
    try:
        streamer = get_streamer()
        chunk = streamer.pop(timeout=timeout)
        
        if chunk is None:
            return {"chunk": None, "queue_size": streamer.queue_size()}
        
        # Encode as base64 for JSON transport
        chunk_b64 = base64.b64encode(chunk).decode('utf-8')
        return {
            "chunk": chunk_b64,
            "queue_size": streamer.queue_size(),
            "samples": len(chunk) // 4
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_queue():
    """Clear the audio queue."""
    try:
        streamer = get_streamer()
        streamer.clear_queue()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WebSocket Endpoint ====================

@router.websocket("/ws")
async def websocket_transcription(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.
    
    Send: {"action": "start", "device": 0} to start
    Send: {"action": "stop"} to stop
    Receive: {"text": "...", "language": "en", "timestamp": ...}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    pipeline = None
    transcription_task = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get('action', '')
            
            if action == 'start':
                device = message.get('device')
                
                if pipeline is not None:
                    await websocket.send_json({"error": "Already running"})
                    continue
                
                # Create and start pipeline
                from modules.streaming import StreamingPipeline
                pipeline = StreamingPipeline(
                    interval_seconds=message.get('interval', 2.0),
                    sample_rate=message.get('sample_rate', 48000)
                )
                pipeline.start(device=device)
                
                await websocket.send_json({"status": "started", "device": device})
                
                # Start transcription loop in background
                async def transcription_loop():
                    try:
                        for result in pipeline.stream():
                            if not pipeline.is_running():
                                break
                            await websocket.send_json({
                                "text": result['text'],
                                "language": result['language'],
                                "timestamp": asyncio.get_event_loop().time()
                            })
                    except Exception as e:
                        logger.error(f"Transcription error: {e}")
                
                transcription_task = asyncio.create_task(transcription_loop())
            
            elif action == 'stop':
                if pipeline is not None:
                    pipeline.stop()
                    pipeline = None
                if transcription_task is not None:
                    transcription_task.cancel()
                    transcription_task = None
                await websocket.send_json({"status": "stopped"})
            
            elif action == 'devices':
                from modules.device_tester import list_input_devices
                devices = list_input_devices(include_all=True)
                await websocket.send_json({"devices": devices})
            
            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if pipeline is not None:
            pipeline.stop()
        if transcription_task is not None:
            transcription_task.cancel()
