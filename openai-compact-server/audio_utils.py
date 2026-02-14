"""
Audio processing utilities for loading and converting audio files.
"""
import tempfile
import asyncio
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa


# Target format for Qwen ASR
QWEN_SAMPLE_RATE = 16000
QWEN_DTYPE = np.float32


async def load_and_convert_audio(file_bytes: bytes, filename: str) -> np.ndarray:
    """Load audio and convert to float32 16kHz mono.
    
    Args:
        file_bytes: Raw file content
        filename: Original filename (used to determine format)
        
    Returns:
        numpy array of float32 samples at 16kHz mono, shape (n_samples,)
    """
    # Get file extension
    ext = Path(filename).suffix.lower()
    
    # Write to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        # Run audio loading in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        samples = await loop.run_in_executor(
            None,
            _load_audio_sync,
            tmp_path,
            ext
        )
        return samples
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def _load_audio_sync(file_path: str, ext: str) -> np.ndarray:
    """Synchronous audio loading (runs in thread pool).
    
    Priority:
    1. soundfile for WAV/FLAC (fastest, zero-copy when possible)
    2. librosa for other formats (MP3, M4A, OGG, etc.)
    """
    # Try soundfile first for supported formats
    if ext in ['.wav', '.flac', '.ogg']:
        try:
            audio, sr = sf.read(file_path, dtype='float32')
            
            # Convert to mono if needed
            if len(audio.shape) == 2:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sr != QWEN_SAMPLE_RATE:
                # Use librosa for high-quality resampling
                audio = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=QWEN_SAMPLE_RATE,
                    res_type='kaiser_best'
                )
            
            # Ensure float32
            if audio.dtype != QWEN_DTYPE:
                audio = audio.astype(QWEN_DTYPE)
            
            return audio
        except Exception as e:
            # Fall through to librosa
            pass
    
    # Use librosa for all other formats (MP3, M4A, etc.)
    try:
        # librosa.load returns float64 by default, specify float32
        audio, sr = librosa.load(
            file_path,
            sr=QWEN_SAMPLE_RATE,
            dtype=QWEN_DTYPE,
            mono=True
        )
        return audio
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}: {e}")


def estimate_token_count(samples: np.ndarray, sample_rate: int = QWEN_SAMPLE_RATE) -> int:
    """Estimate input token count from audio duration.
    
    Approximation: ~1 token per 0.02 seconds of audio.
    This is a rough estimate; OpenAI's actual tokenization differs.
    
    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz (default 16000)
        
    Returns:
        Estimated token count (int)
    """
    duration_sec = len(samples) / sample_rate
    # ~50 tokens per second
    return int(duration_sec * 50)
