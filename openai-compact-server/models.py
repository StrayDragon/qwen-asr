"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel
from typing import Optional


class TranscriptionRequest(BaseModel):
    """Request model for /v1/audio/transcriptions endpoint."""
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: str = "json"
    stream: bool = False


class TokenUsage(BaseModel):
    """Token usage statistics."""
    type: str = "tokens"
    input_tokens: int
    output_tokens: int
    total_tokens: int

    @classmethod
    def from_audio_tokens(cls, input_tokens: int, output_tokens: int) -> "TokenUsage":
        """Create usage stats from audio and output token counts."""
        return cls(
            type="tokens",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )


class TranscriptionResponse(BaseModel):
    """Non-streaming transcription response."""
    text: str
    usage: TokenUsage


class StreamingDelta(BaseModel):
    """Streaming text delta event."""
    type: str = "transcript.text.delta"
    delta: str


class StreamingDone(BaseModel):
    """Streaming completion event."""
    type: str = "transcript.text.done"
    text: str
    usage: TokenUsage


class ErrorResponse(BaseModel):
    """Error response."""
    error: dict

    @classmethod
    def from_detail(cls, detail: str, status: int = 500) -> "ErrorResponse":
        """Create error response from error detail string."""
        return cls(error={"message": detail, "code": status})


class ValidationError(BaseModel):
    """Validation error detail."""
    loc: list[str | int]
    msg: str
    type: str
