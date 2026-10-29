"""
src/augment.py
==============
Audio augmentation pipeline for training-time and held-out-evaluation-time
degradations.

All augmentations that rely on FFmpeg (Opus, MP3, G.711) call
``src.utils.probe_ffmpeg`` at import time within their respective functions
and raise ``RuntimeError`` immediately if the required codec is absent.
They do NOT fall back to any substitute codec (working rule 6).

Training-time augmentations (applied to train split only)
----------------------------------------------------------
apply_clean(waveform, sr)          No-op; returns waveform unchanged.
apply_noise(waveform, sr, cfg)     Additive Gaussian white noise.
apply_echo(waveform, sr, cfg)      Synthetic room impulse response.
apply_opus(waveform, sr, cfg)      Opus encode→decode via FFmpeg.

Held-out augmentations (applied to eval split at scoring time only)
-------------------------------------------------------------------
apply_mp3(waveform, sr, cfg)       MP3 encode→decode via FFmpeg.
apply_g711(waveform, sr, cfg)      G.711 mu-law encode→decode via FFmpeg.
apply_phone_chain(waveform, sr, cfg)  noise + echo + G.711, chained.

Dispatcher
----------
apply_augmentation(waveform, sr, condition, cfg)
    Route to the correct function by condition name string.
"""

from __future__ import annotations

import numpy as np


def apply_clean(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Return ``waveform`` unchanged (no-op clean condition).

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array, values in [-1, 1].
    sr:
        Sample rate in Hz (unused but kept for API consistency).

    Returns
    -------
    np.ndarray
        The original waveform, unmodified.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_noise(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Add Gaussian white noise at a uniformly sampled SNR.

    The SNR is drawn uniformly from ``cfg.augmentation.noise.snr_db_range``
    for each call, so each utterance receives a different noise level.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array.
    sr:
        Sample rate in Hz.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    np.ndarray
        Noisy waveform clipped to [-1, 1].
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_echo(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Convolve waveform with a synthetic single-reflection RIR.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array.
    sr:
        Sample rate in Hz.
    cfg:
        Loaded ``Config`` instance.  Uses ``cfg.augmentation.echo``.

    Returns
    -------
    np.ndarray
        Echo-augmented waveform, same length as input, clipped to [-1, 1].
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_opus(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Encode and decode audio using the Opus codec via FFmpeg.

    Bitrate is drawn uniformly from ``cfg.augmentation.opus.bitrate_kbps_range``.

    Requires FFmpeg with ``libopus`` encoder/decoder.
    Raises ``RuntimeError`` immediately if the codec is missing — does NOT
    substitute an alternative codec.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array at ``sr`` Hz.
    sr:
        Sample rate in Hz.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    np.ndarray
        Opus-degraded waveform.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_mp3(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Encode and decode audio using the MP3 codec via FFmpeg (held-out).

    Requires FFmpeg with ``libmp3lame`` encoder and ``mp3`` decoder.
    Raises ``RuntimeError`` if missing.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array at ``sr`` Hz.
    sr:
        Sample rate in Hz.
    cfg:
        Loaded ``Config`` instance.  Uses ``cfg.augmentation.mp3``.

    Returns
    -------
    np.ndarray
        MP3-degraded waveform.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_g711(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Encode and decode audio using G.711 mu-law via FFmpeg (held-out).

    Resamples to 8 kHz, encodes with ``pcm_mulaw``, decodes, and resamples
    back to ``sr``.  Requires FFmpeg with ``pcm_mulaw`` encoder/decoder.
    Raises ``RuntimeError`` if missing.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array at ``sr`` Hz.
    sr:
        Sample rate in Hz.
    cfg:
        Loaded ``Config`` instance.  Uses ``cfg.augmentation.g711``.

    Returns
    -------
    np.ndarray
        G.711-degraded waveform resampled back to ``sr``.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_phone_chain(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Apply chained phone degradation: noise → echo → G.711 (held-out).

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array at ``sr`` Hz.
    sr:
        Sample rate in Hz.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    np.ndarray
        Fully degraded waveform.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def apply_augmentation(
    waveform: np.ndarray,
    sr: int,
    condition: str,
    cfg,
) -> np.ndarray:
    """Dispatcher: route to the correct augmentation function by name.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array.
    sr:
        Sample rate in Hz.
    condition:
        One of ``'clean'``, ``'noise'``, ``'echo'``, ``'opus'``,
        ``'mp3'``, ``'g711'``, ``'phone_chain'``.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    np.ndarray
        Augmented waveform.

    Raises
    ------
    ValueError
        If ``condition`` is not a recognised augmentation name.
    """
    raise NotImplementedError("Implemented in Stage 3.")
