"""
Audio transcription for mRAG.

Uses OpenAI Whisper (local) to transcribe extracted audio so video summaries
can include spoken content. INSTRUCTIONS mention Faster-Whisper; we use
openai-whisper for now (already in requirements); can swap to faster-whisper later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def transcribe_audio(
    audio_path: Union[str, Path],
    *,
    model_size: str = "base",
    language: Optional[str] = None,
    return_segments: bool = False,
):
    """
    Transcribe an audio file (e.g. WAV extracted from video) using Whisper.

    Returns:
        If return_segments is False: the full transcript text (str).
        If return_segments is True: a list of dicts with "start", "end", "text".
    """
    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Transcription requires openai-whisper. Install with: pip install openai-whisper"
        ) from e

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    model = whisper.load_model(model_size)
    kwargs = {"language": language} if language else {}
    result = model.transcribe(str(audio_path), **kwargs)

    if return_segments and "segments" in result:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", "").strip()}
            for s in result["segments"]
        ]
    return (result.get("text") or "").strip()
