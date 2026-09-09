import io
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_FORMATS = {
    "wav",
    "flac",
    "ogg",
    "oga",
    "opus",
    "mp3",
    "m4a",
    "mp4",
    "aac",
    "webm",
}


def _get_extension(filename: str | None) -> str | None:
    if not filename:
        return None

    _, ext = os.path.splitext(filename.lower())
    if not ext:
        return None

    return ext.lstrip(".")


def _validate_extension(filename: str | None) -> None:
    ext = _get_extension(filename)

    # If no filename is available, do not reject the file.
    # Some clients may upload bytes without a reliable filename.
    if ext is None:
        return

    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported audio format '.{ext}'. "
            f"Supported formats: {supported}"
        )


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data

    # soundfile usually returns shape: samples x channels
    if data.shape[0] >= data.shape[1]:
        return np.mean(data, axis=1)

    # librosa may return shape: channels x samples
    return np.mean(data, axis=0)


def _load_with_soundfile(file_bytes: bytes, target_sr: int) -> np.ndarray:
    data, sr = sf.read(
        io.BytesIO(file_bytes),
        dtype="float32",
        always_2d=False,
    )

    data = np.asarray(data, dtype=np.float32)
    data = _to_mono(data)

    if sr != target_sr:
        data = librosa.resample(
            data,
            orig_sr=sr,
            target_sr=target_sr,
        )

    return data.astype(np.float32)


def _load_with_ffmpeg(file_bytes: bytes, target_sr: int) -> np.ndarray:
    """
    Decode audio bytes with ffmpeg into mono float32 PCM.
    This is more robust for compressed/container formats such as mp3, m4a,
    mp4, webm, opus and aac.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        input=file_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"ffmpeg failed to decode audio: {err}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)

    return audio.astype(np.float32)


def load_audio(
    file_bytes: bytes,
    target_sr: int = 16000,
    filename: str | None = None,
) -> np.ndarray:
    """
    Load audio bytes and return mono float32 PCM at target_sr.

    Supported filename extensions:
    wav, flac, ogg, oga, opus, mp3, m4a, mp4, aac, webm.

    Decoding strategy:
    1. validate extension when a filename is available;
    2. try soundfile first;
    3. fall back to ffmpeg for compressed/container formats.
    """
    if not file_bytes:
        raise ValueError("Empty audio file")

    _validate_extension(filename)

    try:
        audio = _load_with_soundfile(file_bytes, target_sr=target_sr)
    except Exception:
        audio = _load_with_ffmpeg(file_bytes, target_sr=target_sr)

    if audio.size == 0:
        raise ValueError("Decoded audio is empty")

    return audio.astype(np.float32)