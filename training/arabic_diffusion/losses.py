"""
Loss functions for Arabic diffusion fine-tuning.

Includes:
1. Standard diffusion loss
2. OCR-guided loss (Arabic-aware)
3. RTL directionality penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import difflib


class OCRGuidedLoss(nn.Module):
    """
    OCR-guided loss for Arabic text.
    
    Generates image, runs Arabic OCR, compares OCR output vs prompt Arabic,
    and penalizes mismatch (edit distance, order errors).
    
    NOTE: This loss is computed during training only, OCR is NOT part of inference.
    """
    
    def __init__(
        self,
        ocr_model=None,  # PaddleOCR or Tesseract wrapper
        weight: float = 0.1,
        edit_distance_weight: float = 1.0,
        order_penalty_weight: float = 0.5,
    ):
        super().__init__()
        self.ocr_model = ocr_model
        self.weight = weight
        self.edit_distance_weight = edit_distance_weight
        self.order_penalty_weight = order_penalty_weight
    
    def _run_ocr(self, image: torch.Tensor) -> str:
        """
        Run OCR on image.
        
        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W), values in [-1, 1]
        
        Returns:
            OCR text string
        """
        if self.ocr_model is None:
            # Return empty string if OCR not available
            return ""
        
        # Convert tensor to PIL Image
        from torchvision import transforms
        from PIL import Image
        
        # Denormalize
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0, 1)
        
        # Convert to PIL
        if image.dim() == 4:
            image = image[0]  # Take first batch item
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image.cpu())
        
        # Run OCR through the wrapper (handles PaddleOCR / Tesseract internally)
        try:
            return self.ocr_model.ocr(pil_image)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def _compute_edit_distance(self, s1: str, s2: str) -> float:
        """Compute normalized edit distance (Levenshtein distance)."""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        # Use difflib for edit distance
        similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
        return 1.0 - similarity
    
    def _compute_rtl_order_penalty(self, predicted: str, target: str) -> float:
        """
        Penalize LTR ordering of Arabic sequences.
        
        Arabic should be read RTL. If OCR detects LTR ordering,
        this is a strong signal of incorrect rendering.
        """
        # Simple heuristic: check if characters are in reverse order
        # This is a simplified check; real RTL detection is more complex
        if len(predicted) < 2 or len(target) < 2:
            return 0.0
        
        # Check if predicted text is reversed compared to target
        # (This is a simplified check)
        predicted_reversed = predicted[::-1]
        similarity_reversed = difflib.SequenceMatcher(None, predicted_reversed, target).ratio()
        similarity_normal = difflib.SequenceMatcher(None, predicted, target).ratio()
        
        # If reversed is more similar, penalize
        if similarity_reversed > similarity_normal:
            return 0.5  # Moderate penalty
        return 0.0
    
    def forward(
        self,
        generated_image: torch.Tensor,
        target_text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute OCR-guided loss.
        
        Args:
            generated_image: Generated image tensor (B, C, H, W) or (C, H, W)
            target_text: Target Arabic text string
        
        Returns:
            Dictionary with loss components
        """
        if self.ocr_model is None or self.weight == 0.0:
            return {
                "ocr_loss": torch.tensor(0.0, device=generated_image.device),
                "edit_distance": torch.tensor(0.0, device=generated_image.device),
                "rtl_penalty": torch.tensor(0.0, device=generated_image.device),
            }
        
        # Run OCR
        ocr_text = self._run_ocr(generated_image)
        
        # Compute edit distance
        edit_distance = self._compute_edit_distance(ocr_text, target_text)
        
        # Compute RTL penalty
        rtl_penalty = self._compute_rtl_order_penalty(ocr_text, target_text)
        
        # Total loss
        ocr_loss = (
            self.edit_distance_weight * edit_distance +
            self.order_penalty_weight * rtl_penalty
        ) * self.weight
        
        return {
            "ocr_loss": torch.tensor(ocr_loss, device=generated_image.device, dtype=generated_image.dtype),
            "edit_distance": torch.tensor(edit_distance, device=generated_image.device, dtype=generated_image.dtype),
            "rtl_penalty": torch.tensor(rtl_penalty, device=generated_image.device, dtype=generated_image.dtype),
        }


class RTLDirectionalityLoss(nn.Module):
    """
    Directionality penalty for Arabic text.
    
    Penalizes LTR ordering of Arabic sequences.
    This is a simpler version that doesn't require OCR.
    """
    
    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        text: str,
    ) -> torch.Tensor:
        """
        Compute RTL directionality penalty.
        
        This is a placeholder that could be enhanced with:
        - Character-level attention analysis
        - Token ordering analysis
        - Embedding space analysis
        
        For now, returns zero (to be implemented based on research).
        """
        # TODO: Implement based on attention patterns or token ordering
        # For now, return zero loss
        device = prompt_embeds.device
        return torch.tensor(0.0, device=device, dtype=prompt_embeds.dtype) * self.weight


class CombinedLoss(nn.Module):
    """
    Combined loss function for Arabic diffusion training.
    
    Combines:
    1. Standard diffusion loss
    2. OCR-guided loss (optional)
    3. RTL directionality penalty (optional)
    """
    
    def __init__(
        self,
        ocr_loss_fn: Optional[OCRGuidedLoss] = None,
        rtl_loss_fn: Optional[RTLDirectionalityLoss] = None,
        ocr_loss_weight: float = 0.1,
        rtl_loss_weight: float = 0.05,
    ):
        super().__init__()
        self.ocr_loss_fn = ocr_loss_fn
        self.rtl_loss_fn = rtl_loss_fn
        self.ocr_loss_weight = ocr_loss_weight
        self.rtl_loss_weight = rtl_loss_weight
    
    def forward(
        self,
        diffusion_loss: torch.Tensor,
        generated_image: Optional[torch.Tensor] = None,
        target_text: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            diffusion_loss: Standard diffusion loss
            generated_image: Generated image (for OCR loss)
            target_text: Target text (for OCR loss)
            prompt_embeds: Prompt embeddings (for RTL loss)
        
        Returns:
            Dictionary with all loss components
        """
        total_loss = diffusion_loss
        loss_dict = {"diffusion_loss": diffusion_loss}
        
        # OCR-guided loss
        if self.ocr_loss_fn is not None and generated_image is not None and target_text is not None:
            ocr_loss_dict = self.ocr_loss_fn(generated_image, target_text)
            ocr_loss = ocr_loss_dict["ocr_loss"]
            total_loss = total_loss + self.ocr_loss_weight * ocr_loss
            loss_dict.update({f"ocr_{k}": v for k, v in ocr_loss_dict.items()})
        
        # RTL directionality loss
        if self.rtl_loss_fn is not None and prompt_embeds is not None and target_text is not None:
            rtl_loss = self.rtl_loss_fn(prompt_embeds, target_text)
            total_loss = total_loss + self.rtl_loss_weight * rtl_loss
            loss_dict["rtl_loss"] = rtl_loss
        
        loss_dict["total_loss"] = total_loss
        return loss_dict

