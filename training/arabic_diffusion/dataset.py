"""
Dataset classes for Arabic diffusion fine-tuning.

Supports:
1. Synthetic Arabic scene text dataset (primary)
2. EvArEST dataset (secondary, for OCR evaluation)
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display


class SyntheticArabicDataset(Dataset):
    """
    Generates synthetic Arabic text images on simple backgrounds.
    
    This is the PRIMARY dataset for training.
    Features:
    - Clean Arabic words rendered on simple backgrounds
    - Multiple fonts, sizes, colors
    - Correct RTL and glyph connections
    - Generated programmatically
    """
    
    def __init__(
        self,
        arabic_words: List[str],
        size: int = 1024,
        num_samples: int = 10000,
        font_paths: Optional[List[str]] = None,
        background_colors: Optional[List[Tuple[int, int, int]]] = None,
        text_colors: Optional[List[Tuple[int, int, int]]] = None,
        font_sizes: Optional[List[int]] = None,
    ):
        self.arabic_words = arabic_words
        self.size = size
        self.num_samples = num_samples
        
        # Default fonts (user should provide Arabic fonts)
        self.font_paths = font_paths or []
        if not self.font_paths:
            # Try to find system Arabic fonts
            self._find_arabic_fonts()
        
        # Default colors
        self.background_colors = background_colors or [
            (255, 255, 255),  # White
            (240, 240, 240),  # Light gray
            (250, 250, 250),  # Off-white
        ]
        self.text_colors = text_colors or [
            (0, 0, 0),        # Black
            (50, 50, 50),    # Dark gray
            (20, 20, 20),    # Very dark gray
        ]
        self.font_sizes = font_sizes or [32, 40, 48, 56, 64]
        
    def _find_arabic_fonts(self):
        """Try to find Arabic fonts on the system."""
        # Common Arabic font paths
        common_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "C:/Windows/Fonts/tahoma.ttf",  # Windows (has Arabic support)
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.font_paths.append(path)
                break
        
        if not self.font_paths:
            # Fallback: use default PIL font (may not support Arabic well)
            self.font_paths = [None]
    
    def _render_arabic_text(
        self,
        text: str,
        font_path: Optional[str],
        font_size: int,
        text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
    ) -> Image.Image:
        """Render Arabic text on a background image."""
        # Reshape Arabic text for proper display
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        
        # Create image
        img = Image.new('RGB', (self.size, self.size), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Load font
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), bidi_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (self.size - text_width) // 2
        y = (self.size - text_height) // 2
        
        # Draw text
        draw.text((x, y), bidi_text, fill=text_color, font=font)
        
        return img
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Randomly select word, font, colors, size
        word = random.choice(self.arabic_words)
        font_path = random.choice(self.font_paths) if self.font_paths else None
        font_size = random.choice(self.font_sizes)
        bg_color = random.choice(self.background_colors)
        text_color = random.choice(self.text_colors)
        
        # Render image
        image = self._render_arabic_text(word, font_path, font_size, text_color, bg_color)
        
        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pixel_values = transform(image)
        
        return {
            "pixel_values": pixel_values,
            "prompt": word,  # Ground truth Arabic text
            "text": word,   # For OCR evaluation
        }


class EvArESTDataset(Dataset):
    """
    EvArEST dataset loader.
    
    This is the SECONDARY dataset, used ONLY for:
    - OCR-based evaluation
    - OCR-guided loss
    - Weak supervision
    
    DO NOT treat EvArEST as a text-to-image dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" or "test"
        size: int = 1024,
        recognition_only: bool = True,  # Only use recognition dataset
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.size = size
        self.recognition_only = recognition_only
        
        if recognition_only:
            # Load recognition dataset (cropped word images)
            self.images_dir = self.data_dir / "Recognition" / split
            self.gt_file = self.data_dir / "Recognition" / f"{split}_gt.txt"
            
            if not self.images_dir.exists():
                raise ValueError(f"Recognition images directory not found: {self.images_dir}")
            if not self.gt_file.exists():
                raise ValueError(f"Ground truth file not found: {self.gt_file}")
            
            # Load ground truth
            self.samples = []
            with open(self.gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        img_name, text = parts
                        img_path = self.images_dir / img_name
                        if img_path.exists():
                            self.samples.append({
                                "image_path": str(img_path),
                                "text": text,
                            })
        else:
            # Detection dataset (full images with annotations)
            # This is more complex and not recommended for training
            raise NotImplementedError("Detection dataset not implemented. Use recognition_only=True.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Resize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pixel_values = transform(image)
        
        return {
            "pixel_values": pixel_values,
            "prompt": sample["text"],  # Ground truth text
            "text": sample["text"],    # For OCR evaluation
            "is_evarest": True,        # Flag to indicate this is from EvArEST
        }


class CombinedArabicDataset(Dataset):
    """
    Combines synthetic and EvArEST datasets.
    
    Synthetic data is primary, EvArEST is used for validation/weak supervision.
    """
    
    def __init__(
        self,
        synthetic_dataset: SyntheticArabicDataset,
        evarest_dataset: Optional[EvArESTDataset] = None,
        evarest_weight: float = 0.1,  # Weight of EvArEST samples
    ):
        self.synthetic_dataset = synthetic_dataset
        self.evarest_dataset = evarest_dataset
        self.evarest_weight = evarest_weight
        
        self.synthetic_len = len(synthetic_dataset)
        self.evarest_len = len(evarest_dataset) if evarest_dataset else 0
        
        # Calculate effective lengths
        if evarest_dataset:
            # Scale EvArEST to match weight
            self.evarest_effective_len = int(self.synthetic_len * evarest_weight)
            self.total_len = self.synthetic_len + self.evarest_effective_len
        else:
            self.evarest_effective_len = 0
            self.total_len = self.synthetic_len
    
    def __len__(self) -> int:
        return self.total_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.synthetic_len:
            return self.synthetic_dataset[idx]
        else:
            # Sample from EvArEST
            evarest_idx = (idx - self.synthetic_len) % self.evarest_len
            return self.evarest_dataset[evarest_idx]

