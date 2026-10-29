"""
src/extract.py
==============
Batched, frozen-encoder feature extraction for HuBERT and wav2vec 2.0.

All extraction runs under ``torch.no_grad()`` — encoders are never trained.
Embeddings are mean-pooled over the time dimension to produce one fixed-size
vector per utterance.

Results are cached to disk as ``.pt`` files.  If a cache file already exists
for a given (encoder, split, condition) combination, extraction is skipped
and the cache is loaded directly (idempotent, safe to re-run).

Public API (implemented in Stage 4, after pilot approval)
----------------------------------------------------------
load_encoder(encoder_cfg)
    Download (first run) or load from HuggingFace cache a frozen encoder.
    Returns (model, feature_extractor).

extract_embeddings(manifest_df, encoder_cfg, condition, cfg)
    Run batched forward passes and return a (N, D) float32 tensor.

extract_and_cache(manifest_df, encoder_cfg, split, condition, cfg)
    Calls extract_embeddings and saves to cache; loads cache if present.

load_cached_embeddings(encoder_cfg, split, condition, cfg)
    Load a previously cached embedding tensor from disk.

cache_path(encoder_cfg, split, condition, cfg) -> Path
    Return the expected path for a given cache file.
"""

from __future__ import annotations

from pathlib import Path


def cache_path(encoder_cfg, split: str, condition: str, cfg) -> Path:
    """Return the expected ``.pt`` cache file path for a given combination.

    Parameters
    ----------
    encoder_cfg:
        An ``EncoderConfig`` instance (from ``cfg.encoders``).
    split:
        ``'train'``, ``'dev'``, or ``'eval'``.
    condition:
        Augmentation condition string (e.g. ``'clean'``, ``'noise'``).
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    Path
        E.g. ``data/features/hubert_base/train_clean.pt``.
    """
    raise NotImplementedError("Implemented in Stage 4.")


def load_encoder(encoder_cfg):
    """Load a pretrained, frozen encoder from HuggingFace Hub.

    The encoder's parameters are frozen (``requires_grad=False``).
    The model is placed on CPU.

    Parameters
    ----------
    encoder_cfg:
        An ``EncoderConfig`` instance.

    Returns
    -------
    (model, feature_extractor)
        A HuggingFace model and its corresponding feature extractor /
        processor, both ready for inference.

    Raises
    ------
    OSError
        If the model cannot be downloaded or loaded from the HF cache.
    """
    raise NotImplementedError("Implemented in Stage 4.")


def extract_embeddings(manifest_df, encoder_cfg, condition: str, cfg):
    """Run batched frozen-encoder inference and return mean-pooled embeddings.

    Parameters
    ----------
    manifest_df:
        DataFrame as returned by ``src.dataset.load_manifest``.
    encoder_cfg:
        An ``EncoderConfig`` instance.
    condition:
        Augmentation condition name applied to each waveform before encoding.
    cfg:
        Loaded ``Config`` instance.  Uses ``encoder_cfg.batch_size`` for
        batching and ``cfg.dataset.sample_rate`` for audio loading.

    Returns
    -------
    torch.Tensor
        Shape ``(N, D)`` float32, where N = number of utterances and D is
        the encoder's hidden size (768 for base models).

    Notes
    -----
    * Runs entirely under ``torch.no_grad()``.
    * Augmentation is applied on-the-fly per batch — augmented audio is
      never persisted to disk.
    * Progress is reported via tqdm.
    """
    raise NotImplementedError("Implemented in Stage 4.")


def extract_and_cache(manifest_df, encoder_cfg, split: str, condition: str, cfg):
    """Extract embeddings and save to cache, or load existing cache.

    Parameters
    ----------
    manifest_df:
        DataFrame for the requested split.
    encoder_cfg:
        An ``EncoderConfig`` instance.
    split:
        ``'train'``, ``'dev'``, or ``'eval'``.
    condition:
        Augmentation condition name.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(N, D)`` float32 embedding tensor.
    """
    raise NotImplementedError("Implemented in Stage 4.")


def load_cached_embeddings(encoder_cfg, split: str, condition: str, cfg):
    """Load a previously cached embedding tensor from disk.

    Parameters
    ----------
    encoder_cfg:
        An ``EncoderConfig`` instance.
    split:
        ``'train'``, ``'dev'``, or ``'eval'``.
    condition:
        Augmentation condition name.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(N, D)`` float32.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """
    raise NotImplementedError("Implemented in Stage 4.")
