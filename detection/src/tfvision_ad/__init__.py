"""TFVision-AD: training-free vision (CLIP) anomaly detection for
quasi-periodic time series.

A beat is rendered to an image and matched against a phase-aligned memory bank
of normal beats along three disentangled channels (amp_fine / amp_ng / sh).
No training, no LLM.
"""
from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
