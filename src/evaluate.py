"""
src/evaluate.py
===============
Metric computation for the Deepfake Call Shield experiment.

Primary metric: Equal Error Rate (EER) — the threshold where the false
acceptance rate (FAR) equals the false rejection rate (FRR).

Secondary metric: minimum Detection Cost Function (min-DCF), the standard
ASVspoof 2019 metric.

Bootstrap confidence intervals (95%) are computed over 10 000 resamples
of the evaluation score list.

IMPORTANT: threshold selection uses DEV scores only.
           Eval scores are computed once, in script 07_evaluate_eval.py.

Public API (implemented in Stage 5)
-------------------------------------
compute_eer(labels, scores)
    Return (eer, threshold) from binary labels and real-valued scores.

compute_mindcf(labels, scores, p_target, c_miss, c_fa)
    Return min-DCF for the given cost parameters.

bootstrap_eer_ci(labels, scores, n_resamples, alpha, seed)
    Return (eer_mean, ci_low, ci_high) via bootstrap resampling.

save_scores(scores, labels, utt_ids, encoder_name, split, condition, cfg)
    Persist raw scores to ``results/scores/``.

load_scores(encoder_name, split, condition, cfg)
    Load raw scores from ``results/scores/``.
"""

from __future__ import annotations

from pathlib import Path


def compute_eer(labels, scores):
    """Compute Equal Error Rate from binary labels and real-valued scores.

    Parameters
    ----------
    labels:
        1-D array-like of int; 0 = bonafide (target), 1 = spoof (non-target).
    scores:
        1-D array-like of float; higher score → more likely spoof.

    Returns
    -------
    (eer: float, threshold: float)
        ``eer`` is in the range [0, 1].  ``threshold`` is the score value
        at the EER operating point (for use on dev only).

    Notes
    -----
    Uses linear interpolation between adjacent FAR/FRR values to find
    the exact crossing point.
    """
    raise NotImplementedError("Implemented in Stage 5.")


def compute_mindcf(
    labels,
    scores,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute minimum Detection Cost Function (min-DCF).

    Parameters
    ----------
    labels:
        1-D array-like of int (0 = bonafide, 1 = spoof).
    scores:
        1-D array-like of float.
    p_target:
        Prior probability of a target trial (default 0.05, ASVspoof standard).
    c_miss:
        Cost of a miss (default 1.0).
    c_fa:
        Cost of a false alarm (default 1.0).

    Returns
    -------
    float
        min-DCF value.
    """
    raise NotImplementedError("Implemented in Stage 5.")


def bootstrap_eer_ci(
    labels,
    scores,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Compute bootstrap confidence interval for EER.

    Parameters
    ----------
    labels:
        1-D array-like of int (0 = bonafide, 1 = spoof).
    scores:
        1-D array-like of float.
    n_resamples:
        Number of bootstrap resamples.
    alpha:
        Significance level; CI width = 1 - alpha (default 0.05 → 95% CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (eer_mean: float, ci_low: float, ci_high: float)
        All values in [0, 1].
    """
    raise NotImplementedError("Implemented in Stage 5.")


def save_scores(
    scores,
    labels,
    utt_ids,
    encoder_name: str,
    split: str,
    condition: str,
    cfg,
) -> Path:
    """Save raw scores, labels, and utterance IDs to a CSV file.

    Parameters
    ----------
    scores:
        1-D float array of model scores.
    labels:
        1-D int array of ground-truth labels.
    utt_ids:
        1-D string array of utterance IDs.
    encoder_name:
        Encoder identifier string.
    split:
        ``'train'``, ``'dev'``, or ``'eval'``.
    condition:
        Augmentation condition string.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    Path
        Path of the saved CSV file.
    """
    raise NotImplementedError("Implemented in Stage 5.")


def load_scores(encoder_name: str, split: str, condition: str, cfg):
    """Load raw scores from a previously saved CSV file.

    Parameters
    ----------
    encoder_name:
        Encoder identifier string.
    split:
        ``'train'``, ``'dev'``, or ``'eval'``.
    condition:
        Augmentation condition string.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    (scores: np.ndarray, labels: np.ndarray, utt_ids: np.ndarray)

    Raises
    ------
    FileNotFoundError
        If the score file does not exist.
    """
    raise NotImplementedError("Implemented in Stage 5.")
