"""
C library bindings for qwen_asr using ctypes.
"""
import ctypes
import ctypes.util
from pathlib import Path
from typing import Optional

# Find and load the shared library
# Search in parent directory (qwen-asr root)
_lib_path = Path(__file__).parent.parent / "libqwen_asr.so"
if not _lib_path.exists():
    # Try system library path
    _lib_path = ctypes.util.find_library("qwen_asr")
    if _lib_path is None:
        raise ImportError(
            "Cannot find libqwen_asr.so. "
            "Run 'make libqwen_asr.so' in the qwen-asr directory first."
        )
    _lib_path = Path(_lib_path)

_lib = ctypes.CDLL(str(_lib_path))

# ============================================================================
# Type definitions
# ============================================================================

# Token callback function type
# void (*qwen_token_cb)(const char *piece, void *userdata)
TOKEN_CALLBACK_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_void_p
)

# Performance stats structure (matching qwen_ctx_t fields)
class QwenPerfStats(ctypes.Structure):
    _fields_ = [
        ("perf_total_ms", ctypes.c_double),
        ("perf_text_tokens", ctypes.c_int),
        ("perf_audio_ms", ctypes.c_double),
        ("perf_encode_ms", ctypes.c_double),
        ("perf_decode_ms", ctypes.c_double),
    ]

# ============================================================================
# Function signatures
# ============================================================================

# qwen_ctx_t *qwen_load(const char *model_dir)
_lib.qwen_load.restype = ctypes.c_void_p
_lib.qwen_load.argtypes = [ctypes.c_char_p]

# void qwen_free(qwen_ctx_t *ctx)
_lib.qwen_free.restype = None
_lib.qwen_free.argtypes = [ctypes.c_void_p]

# void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata)
_lib.qwen_set_token_callback.restype = None
_lib.qwen_set_token_callback.argtypes = [
    ctypes.c_void_p, TOKEN_CALLBACK_FUNC, ctypes.c_void_p
]

# int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt)
_lib.qwen_set_prompt.restype = ctypes.c_int
_lib.qwen_set_prompt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# int qwen_set_force_language(qwen_ctx_t *ctx, const char *language)
_lib.qwen_set_force_language.restype = ctypes.c_int
_lib.qwen_set_force_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# const char *qwen_supported_languages_csv(void)
_lib.qwen_supported_languages_csv.restype = ctypes.c_char_p
_lib.qwen_supported_languages_csv.argtypes = []

# char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples)
_lib.qwen_transcribe_audio.restype = ctypes.c_char_p
_lib.qwen_transcribe_audio.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

# char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples)
_lib.qwen_transcribe_stream.restype = ctypes.c_char_p
_lib.qwen_transcribe_stream.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

# ============================================================================
# Python wrapper functions
# ============================================================================

class QwenContext:
    """Wrapper for qwen_ctx_t with automatic resource management."""

    def __init__(self, model_dir: str):
        """Load model from directory."""
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self._ctx = _lib.qwen_load(model_dir.encode('utf-8'))
        if self._ctx is None:
            raise RuntimeError(f"Failed to load model from {model_dir}")
        self._model_dir = model_dir
        self._token_callback = None

    def __del__(self):
        """Automatically free resources."""
        if hasattr(self, '_ctx') and self._ctx is not None:
            _lib.qwen_free(self._ctx)
            self._ctx = None

    @property
    def ctx_ptr(self) -> int:
        """Get raw context pointer for ctypes calls."""
        if self._ctx is None:
            raise RuntimeError("Context has been freed")
        return self._ctx

    def set_token_callback(self, callback):
        """Set token callback for streaming.
        
        callback: function(piece: str) -> None
        """
        if callback is None:
            _lib.qwen_set_token_callback(self.ctx_ptr, None, None)
            self._token_callback = None
        else:
            # Create a ctypes callback wrapper
            def _wrapper(piece_ptr, userdata):
                piece = ctypes.cast(piece_ptr, ctypes.c_char_p).value.decode('utf-8')
                callback(piece)
            
            self._token_callback = TOKEN_CALLBACK_FUNC(_wrapper)
            _lib.qwen_set_token_callback(self.ctx_ptr, self._token_callback, None)

    def set_prompt(self, prompt: Optional[str]) -> int:
        """Set system prompt. Returns 0 on success, -1 on error."""
        if prompt is None:
            prompt = ""
        return _lib.qwen_set_prompt(self.ctx_ptr, prompt.encode('utf-8'))

    def set_force_language(self, language: Optional[str]) -> int:
        """Set forced language (ISO-639-1 code or language name). Returns 0 on success, -1 on error."""
        if language is None:
            language = ""
        return _lib.qwen_set_force_language(self.ctx_ptr, language.encode('utf-8'))

    def transcribe_audio(self, samples, streaming: bool = False) -> tuple[str, QwenPerfStats]:
        """Transcribe audio from float32 16kHz mono samples.
        
        Args:
            samples: numpy array or list of float32 samples at 16kHz
            streaming: if True, use streaming mode with chunk-by-chunk processing
            
        Returns:
            (transcribed_text, perf_stats)
        """
        import numpy as np
        
        # Convert to numpy array if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # Ensure contiguous array
        if not samples.flags['C_CONTIGUOUS']:
            samples = np.ascontiguousarray(samples)

        n_samples = len(samples)
        samples_ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Choose transcription function
        transcribe_func = _lib.qwen_transcribe_stream if streaming else _lib.qwen_transcribe_audio

        # Call transcription
        result_ptr = transcribe_func(self.ctx_ptr, samples_ptr, n_samples)
        if result_ptr is None:
            raise RuntimeError("Transcription failed (returned NULL)")

        # Extract result string
        result = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('utf-8')

        # Get performance stats from context structure
        # Note: This assumes the context structure layout matches qwen_ctx_t
        # We only need the stats fields
        perf = QwenPerfStats.from_address(self.ctx_ptr)

        return result, perf

    @staticmethod
    def get_supported_languages() -> str:
        """Get comma-separated list of supported language names."""
        result = _lib.qwen_supported_languages_csv()
        return result.decode('utf-8') if result else ""


# Direct functions for convenience
def qwen_load(model_dir: str) -> QwenContext:
    """Load model and return QwenContext wrapper."""
    return QwenContext(model_dir)

def qwen_free(ctx: QwenContext):
    """Free model resources (also called automatically by __del__)."""
    if ctx is not None:
        ctx.__del__()

def get_supported_languages() -> str:
    """Get comma-separated list of supported language names."""
    return QwenContext.get_supported_languages()
