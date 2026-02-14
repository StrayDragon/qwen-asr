"""
OpenAI-Compatible Audio API Server for Qwen ASR.

Provides /v1/audio/transcriptions endpoint compatible with OpenAI API format.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, status, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from libqwen import QwenContext
from audio_utils import load_and_convert_audio, estimate_token_count
from models import (
    TranscriptionResponse,
    StreamingDelta,
    StreamingDone,
    ErrorResponse,
    TokenUsage,
)


# ============================================================================
# Model Pool Management
# ============================================================================

class ModelPool:
    """Pool of model contexts with lazy loading and auto-unload."""

    def __init__(self):
        self._contexts: dict[str, list[QwenContext]] = {}
        self._lock: dict[str, asyncio.Lock] = {}
        self._model_dirs: dict[str, str] = {}  # model_name -> model_dir mapping
        self._last_used: dict[str, float] = {}  # model_name -> last_used_timestamp
        self._initialized = False
        self._unload_task: Optional[asyncio.Task] = None
        self._unload_stop_event: Optional[asyncio.Event] = None

    async def initialize(self):
        """Initialize pool structure (lazy load on first use)."""
        if self._initialized:
            return

        available_models = settings.get_available_models()
        if not available_models:
            raise RuntimeError(
                "No models found! Please download at least one model:\n"
                "  - qwen3-asr-0.6b\n"
                "  - qwen3-asr-1.7b\n"
                "Use: bash download_model.sh"
            )

        # Initialize structures for all available models
        for model_name in available_models:
            model_dir = settings.get_model_dir(model_name)
            self._contexts[model_name] = []
            self._lock[model_name] = asyncio.Lock()
            self._model_dirs[model_name] = model_dir
            self._last_used[model_name] = 0.0  # Never used

        self._initialized = True
        print(f"Model pool initialized for {len(available_models)} model(s) (lazy loading enabled)")
        if settings.MODEL_IDLE_TIMEOUT > 0:
            print(f"Models will auto-unload after {settings.MODEL_IDLE_TIMEOUT}s of inactivity")
            # Start background monitor task
            self._unload_stop_event = asyncio.Event()
            self._unload_task = asyncio.create_task(self._unload_monitor())
        else:
            print("Model auto-unload disabled (MODEL_IDLE_TIMEOUT=0)")

    async def shutdown(self):
        """Cleanup resources."""
        if self._unload_task:
            self._unload_stop_event.set()
            try:
                await asyncio.wait_for(self._unload_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._unload_task.cancel()
            self._unload_task = None

        # Unload all models
        for model_name in list(self._contexts.keys()):
            await self._unload_model(model_name)

    async def _ensure_loaded(self, model_name: str):
        """Ensure model is loaded, load if necessary."""
        if not self._contexts[model_name]:
            # Load model instances
            model_dir = self._model_dirs[model_name]
            print(f"[{model_name}] Loading model (lazy load)...")
            for i in range(settings.MODEL_POOL_SIZE):
                ctx = await asyncio.to_thread(QwenContext, model_dir)
                self._contexts[model_name].append(ctx)
            print(f"[{model_name}] Model ready ({settings.MODEL_POOL_SIZE} instances)")

    async def _unload_model(self, model_name: str):
        """Unload all instances of a model to free memory."""
        if self._contexts[model_name]:
            print(f"[{model_name}] Unloading model...")
            # Clear contexts (Python GC should free the C memory)
            self._contexts[model_name].clear()
            self._last_used[model_name] = 0.0
            print(f"[{model_name}] Model unloaded")

    async def _unload_monitor(self):
        """Background task to monitor and unload idle models."""
        while not self._unload_stop_event.is_set():
            try:
                # Check every 30 seconds
                await asyncio.wait_for(self._unload_stop_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                # Check for idle models
                current_time = asyncio.get_event_loop().time()
                for model_name in list(self._contexts.keys()):
                    if self._last_used[model_name] > 0:  # Has been used at least once
                        idle_time = current_time - self._last_used[model_name]
                        if idle_time > settings.MODEL_IDLE_TIMEOUT:
                            # Only unload if no active requests (lock is free)
                            lock = self._lock[model_name]
                            if lock.locked():
                                # Model is in use, skip
                                continue
                            async with lock:
                                # Double-check after acquiring lock
                                idle_time = asyncio.get_event_loop().time() - self._last_used[model_name]
                                if idle_time > settings.MODEL_IDLE_TIMEOUT:
                                    await self._unload_model(model_name)

    def _update_last_used(self, model_name: str):
        """Update last used timestamp."""
        self._last_used[model_name] = asyncio.get_event_loop().time()
    
    @asynccontextmanager
    async def acquire(self, model_name: str):
        """Acquire a model context from the pool.

        Usage:
            async with pool.acquire(model_name) as ctx:
                # Use ctx
                pass
        """
        if model_name not in self._contexts:
            available = list(self._contexts.keys())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_name}' not available. Available: {available}"
            )

        # Lazy load model if needed
        await self._ensure_loaded(model_name)

        # Update last used time
        self._update_last_used(model_name)

        lock = self._lock[model_name]
        contexts = self._contexts[model_name]

        async with lock:
            if not contexts:
                # Pool exhausted - wait briefly for a context to be returned
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=30.0)
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"All model instances for '{model_name}' are busy. Try again later."
                    )

            ctx = contexts.pop()
            try:
                yield ctx
            finally:
                # Return context to pool
                contexts.append(ctx)
                # Update last used time after request completion
                self._update_last_used(model_name)


# Global model pool
model_pool = ModelPool()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup: initialize pool (lazy load on first request)
    print("Starting Qwen ASR OpenAI-Compatible API Server...")
    await model_pool.initialize()
    print(f"\nServer ready at http://{settings.HOST}:{settings.PORT}")
    print(f"API docs: http://{settings.HOST}:{settings.PORT}/docs")

    yield

    # Shutdown: cleanup
    print("Shutting down...")
    await model_pool.shutdown()


app = FastAPI(
    title="Qwen ASR OpenAI-Compatible API",
    version="1.0.0",
    description="Audio transcription API compatible with OpenAI format",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_token(authorization: Optional[str] = Header(None)) -> str:
    """Verify Bearer token authentication."""
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format. Expected: 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split(" ", 1)[1]
    if token != settings.API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API token",
        )
    
    return token


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "name": "Qwen ASR OpenAI-Compatible API Server",
        "version": "1.0.0",
        "models": settings.get_available_models(),
        "endpoints": {
            "transcriptions": "/v1/audio/transcriptions",
        },
    }


@app.get("/v1/models")
async def list_models(_token: str = Depends(verify_api_token)):
    """List available models (OpenAI-compatible format)."""
    models = []
    for name in settings.get_available_models():
        models.append({
            "id": name,
            "name": name,
            "owned_by": "qwen",
        })
    return {"object": "list", "data": models}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = Form(...),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    stream: bool = Form(False),
    _token: str = Depends(verify_api_token),
):
    """Transcribe audio file (OpenAI-compatible endpoint).

    Request:
        - file: Audio file (multipart/form-data)
        - model: Model ID (e.g., "qwen-asr-1.7b")
        - language: Optional ISO-639-1 language code
        - prompt: Optional system prompt
        - response_format: "json" (default) or "text"
        - stream: Enable SSE streaming (default: false)

    Response (stream=false):
        {
            "text": "transcribed text",
            "usage": {
                "type": "tokens",
                "input_tokens": 14,
                "output_tokens": 45,
                "total_tokens": 59
            }
        }

    Response (stream=true):
        SSE events with incremental text updates.
    """
    
    # Validate model
    if model not in settings.get_available_models():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model}' not available. Available: {settings.get_available_models()}"
        )
    
    # Read uploaded file
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )
    
    # Convert audio to target format
    samples = await load_and_convert_audio(file_bytes, file.filename)
    
    # Choose streaming vs non-streaming path
    if stream:
        return StreamingResponse(
            _stream_transcription(model, language, prompt, samples),
            media_type="text/event-stream",
        )
    else:
        return await _non_streaming_transcription(model, language, prompt, samples)


async def _non_streaming_transcription(
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    samples,
) -> JSONResponse:
    """Non-streaming transcription path."""
    
    async with model_pool.acquire(model) as ctx:
        # Set language if provided
        if language:
            result = await asyncio.to_thread(ctx.set_force_language, language)
            if result != 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported language: '{language}'",
                )
        
        # Set prompt if provided
        if prompt:
            await asyncio.to_thread(ctx.set_prompt, prompt)
        
        # Run transcription
        text, perf = await asyncio.to_thread(
            ctx.transcribe_audio,
            samples,
            streaming=False,
        )
        
        # Calculate usage stats
        input_tokens = estimate_token_count(samples)
        output_tokens = perf.perf_text_tokens
        usage = TokenUsage.from_audio_tokens(input_tokens, output_tokens)
        
        return JSONResponse(
            content=TranscriptionResponse(text=text, usage=usage).model_dump()
        )


async def _stream_transcription(
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    samples,
):
    """Streaming transcription path with SSE events."""
    
    text_buffer = []
    token_deltas = []
    
    def token_callback(piece: str):
        """Capture token fragments for streaming."""
        if piece:
            token_deltas.append(piece)
            text_buffer.append(piece)
    
    async with model_pool.acquire(model) as ctx:
        # Set language if provided
        if language:
            result = await asyncio.to_thread(ctx.set_force_language, language)
            if result != 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported language: '{language}'",
                )
        
        # Set prompt if provided
        if prompt:
            await asyncio.to_thread(ctx.set_prompt, prompt)
        
        # Install token callback
        ctx.set_token_callback(token_callback)
        
        # Run transcription in background thread
        # Note: For streaming mode, we use the non-streaming transcription function
        # but capture tokens via callback. This is simpler than using qwen_transcribe_stream
        # which requires chunk-by-chunk audio growth.
        task = asyncio.create_task(
            asyncio.to_thread(
                ctx.transcribe_audio,
                samples,
                streaming=False,  # But callback will be invoked
            )
        )
        
        # Stream token deltas as they arrive
        # In a real implementation, we'd yield as tokens arrive
        # For now, we'll wait for completion and then yield deltas
        text, perf = await task
        
        # Yield all token deltas
        for delta in token_deltas:
            event = StreamingDelta(delta=delta)
            yield f"data: {event.model_dump_json()}\n\n"
        
        # Yield completion event
        input_tokens = estimate_token_count(samples)
        output_tokens = perf.perf_text_tokens
        usage = TokenUsage.from_audio_tokens(input_tokens, output_tokens)
        full_text = ''.join(text_buffer)
        
        event = StreamingDone(text=full_text, usage=usage)
        yield f"data: {event.model_dump_json()}\n\n"


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Standard HTTP error handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.from_detail(exc.detail).model_dump(),
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # Enable auto-reload for development
    )
