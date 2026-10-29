"""
src/train.py
============
Logistic Regression classifier training for the Deepfake Call Shield
experiment.

One model is trained per configuration (encoder × training condition).
The regularisation strength ``C`` is selected by grid search over
``cfg.classifier.C_candidates`` using the **development set EER** — never
the evaluation set.

Public API (implemented in Stage 5)
-------------------------------------
train_logistic_regression(X_train, y_train, cfg)
    Fit a LogisticRegression on the provided features and return the model.

select_C(X_train, y_train, X_dev, y_dev, cfg)
    Grid-search C over ``cfg.classifier.C_candidates``; return best C and
    its dev EER.

save_model(model, encoder_name, condition, cfg)
    Persist the trained sklearn model to ``results/models/``.

load_model(encoder_name, condition, cfg)
    Load a previously saved model from ``results/models/``.
"""

from __future__ import annotations

from pathlib import Path


def train_logistic_regression(X_train, y_train, C: float, cfg):
    """Fit a LogisticRegression classifier.

    Parameters
    ----------
    X_train:
        2-D float32 array of shape ``(N_train, D)``.
    y_train:
        1-D int array of labels: 0 = bonafide, 1 = spoof.
    C:
        Regularisation inverse strength (selected on dev set, not here).
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted classifier.
    """
    raise NotImplementedError("Implemented in Stage 5.")


def select_C(X_train, y_train, X_dev, y_dev, cfg):
    """Grid-search C using development-set EER as the selection criterion.

    Parameters
    ----------
    X_train, y_train:
        Training features and labels.
    X_dev, y_dev:
        Development features and labels.  **Never use eval data here.**
    cfg:
        Loaded ``Config`` instance.  Candidates come from
        ``cfg.classifier.C_candidates``.

    Returns
    -------
    (best_C: float, best_dev_eer: float, results: list[dict])
        ``results`` contains one dict per candidate with keys
        ``'C'`` and ``'dev_eer'``.
    """
    raise NotImplementedError("Implemented in Stage 5.")


def save_model(model, encoder_name: str, condition: str, cfg) -> Path:
    """Persist a trained sklearn model to disk.

    Parameters
    ----------
    model:
        Fitted ``LogisticRegression`` instance.
    encoder_name:
        Encoder identifier string, e.g. ``'hubert_base'``.
    condition:
        Training condition string, e.g. ``'clean'`` or ``'augmented'``.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    Path
        The path where the model was saved.
    """
    raise NotImplementedError("Implemented in Stage 5.")


def load_model(encoder_name: str, condition: str, cfg):
    """Load a previously saved sklearn model from disk.

    Parameters
    ----------
    encoder_name:
        Encoder identifier string.
    condition:
        Training condition string.
    cfg:
        Loaded ``Config`` instance.

    Returns
    -------
    sklearn.linear_model.LogisticRegression

    Raises
    ------
    FileNotFoundError
        If no saved model exists for the given encoder + condition.
    """
    raise NotImplementedError("Implemented in Stage 5.")
