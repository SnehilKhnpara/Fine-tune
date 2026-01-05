"""
Arabic Diffusion Fine-tuning Module (Approach 3)

This module implements fine-tuning of diffusion models for improved Arabic text rendering.
It is a RESEARCH PHASE and should not be used as a production dependency.

Key components:
- dataset: Dataset classes for synthetic Arabic and EvArEST data
- losses: Loss functions including OCR-guided and RTL directionality losses
- ocr_loss: OCR integration for training-time evaluation
- train_lora: Main training script
"""

__version__ = "0.1.0"

from .dataset import (
    SyntheticArabicDataset,
    EvArESTDataset,
    CombinedArabicDataset,
)

from .losses import (
    OCRGuidedLoss,
    RTLDirectionalityLoss,
    CombinedLoss,
)

from .ocr_loss import OCRWrapper

__all__ = [
    "SyntheticArabicDataset",
    "EvArESTDataset",
    "CombinedArabicDataset",
    "OCRGuidedLoss",
    "RTLDirectionalityLoss",
    "CombinedLoss",
    "OCRWrapper",
]

