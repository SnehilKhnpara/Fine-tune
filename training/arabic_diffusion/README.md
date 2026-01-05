# Arabic Diffusion Fine-tuning (Approach 3) - Experimental

**⚠️ RESEARCH PHASE - NOT PRODUCTION DEPENDENCY**

This module implements **Approach 3: Fine-tuned Arabic Diffusion** as a separate, optional, research phase.

## ⚠️ Critical Safety Notes

- **This approach MUST NOT break Approach 1 (Arabic mask/ControlNet) or Approach 2 (Layout → Render)**
- **Production correctness MUST remain guaranteed by Approach 1**
- **This is an enhancement layer, not a replacement**
- **Model output is NEVER trusted blindly - fallback to mask/overlay is always available**

## Overview

This approach fine-tunes a diffusion model (using LoRA) to:
- Understand Arabic text semantically from prompts
- Render Arabic glyphs more accurately than base models
- Reduce (but not eliminate) dependence on masks

**IMPORTANT**: This approach is NOT expected to achieve perfect Arabic spelling alone. It is meant to:
- Improve Arabic awareness
- Improve consistency
- Work in combination with mask-based enforcement

## Dataset Strategy

### 1. Synthetic Arabic Scene Dataset (Primary)
- Clean Arabic words rendered on simple backgrounds
- Multiple fonts, sizes, colours
- Correct RTL and glyph connections
- Generated programmatically

### 2. EvArEST Dataset (Secondary)
- Used ONLY for:
  - OCR-based evaluation
  - OCR-guided loss
  - Weak supervision
- **DO NOT treat EvArEST as a text-to-image dataset**

### 3. Optional External Synthetic Datasets
- Arabic text overlays on random backgrounds
- Layout-aware generation

## Model Strategy

- **Base Model**: SD3 (preferred for fine-tuning stability)
- **Training Method**: LoRA or QLoRA ONLY (no full model fine-tuning)
- **Target Components**:
  - Text encoder
  - Cross-attention layers
  - (Optional) decoder layers

## Training Objectives

### Primary Objective
Improve alignment between Arabic tokens and visual glyphs

### Secondary Objectives
- Reduce character hallucination
- Improve spacing and connection consistency

### Loss Functions

1. **Standard diffusion loss** - Standard flow matching loss
2. **OCR-guided loss (Arabic-aware)**
   - Generate image
   - Run Arabic OCR (PaddleOCR or Tesseract)
   - Compare OCR output vs prompt Arabic
   - Penalise mismatch (edit distance, order errors)
3. **Directionality penalty**
   - Penalise LTR ordering of Arabic sequences

## Installation

### Prerequisites

```bash
# Install base requirements (from project root)
pip install -r requirements.txt
pip install -e ./diffusers-amo
```

### Additional Dependencies for Training

```bash
# For OCR support
pip install paddlepaddle paddleocr  # Preferred for Arabic
# OR
pip install pytesseract  # Fallback

# For Arabic text reshaping
pip install arabic-reshaper python-bidi

# For training
pip install peft accelerate wandb tensorboard
```

### Arabic Fonts

You need Arabic fonts for synthetic data generation. Common options:
- Noto Sans Arabic
- Tahoma (Windows)
- Arial Unicode MS

Place font files in a directory and specify paths in config.

## Usage

### 1. Prepare Arabic Words File

Create a text file with Arabic words (one per line):

```bash
# data/arabic_words.txt
مرحبا
شكرا
سلام
...
```

### 2. Prepare EvArEST Dataset (Optional)

Download EvArEST dataset from: https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text

Extract and point to the directory in config.

### 3. Configure Training

Edit `configs/arabic_lora.yaml` or pass arguments directly.

### 4. Run Training

```bash
# Using config file
python training/arabic_diffusion/train_lora.py \
    --arabic_words_file data/arabic_words.txt \
    --evarest_data_dir path/to/evarest \
    --output_dir arabic-lora-output \
    --enable_ocr_loss \
    --ocr_type paddleocr \
    --enable_rtl_loss

# Or with more options
python training/arabic_diffusion/train_lora.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --arabic_words_file data/arabic_words.txt \
    --output_dir arabic-lora-output \
    --train_batch_size 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --rank 4 \
    --enable_ocr_loss \
    --ocr_loss_weight 0.1 \
    --enable_rtl_loss \
    --rtl_loss_weight 0.05
```

### 5. Use Trained LoRA in Inference

```bash
# Hybrid mode (recommended - model + mask fallback)
python run.py \
    --model_type sd3 \
    --arabic_diffusion_mode hybrid \
    --arabic_lora_path arabic-lora-output \
    --prompt_file prompts.txt

# Model-only mode (research/testing only)
python run.py \
    --model_type sd3 \
    --arabic_diffusion_mode model_only \
    --arabic_lora_path arabic-lora-output \
    --prompt_file prompts.txt
```

## Inference Rules (CRITICAL)

During inference:
- Model output is NEVER trusted blindly
- If Arabic is detected:
  1. Try model-native generation
  2. Validate using OCR (optional, fast check)
  3. If confidence < threshold → fallback to mask/overlay

This ensures:
- No regression in correctness
- No repeated "WINE-like" failures

## CLI Extensions

Added to `run.py`:

```bash
--arabic_diffusion_mode [off|hybrid|model_only]
  off        → default (Approach 1 / 2 only)
  hybrid     → model + mask fallback (recommended)
  model_only → research/testing only

--arabic_lora_path path_to_lora
```

Defaults:
- `arabic_diffusion_mode = off`

## Logging & Metrics

The training script logs:
- OCR accuracy
- Character error rate
- RTL correctness rate
- Fallback frequency

Metrics are saved per epoch to TensorBoard/W&B.

## Expected Timeline

- Dataset preparation: 2–4 weeks
- LoRA training experiments: 4–6 weeks
- Evaluation & tuning: 2–4 weeks

**Total: ~2–3 months minimum**

## Quality Bar

Success is defined as:
- Arabic text accuracy improves vs base model
- Fewer hallucinated characters
- Mask fallback still guarantees correctness

Failure is acceptable if:
- The system safely falls back to deterministic rendering

## File Structure

```
training/
  arabic_diffusion/
    train_lora.py      # Main training script
    dataset.py          # Dataset classes (Synthetic, EvArEST, Combined)
    losses.py           # Loss functions (OCR, RTL, Combined)
    ocr_loss.py         # OCR integration wrapper
    README.md           # This file

configs/
  arabic_lora.yaml      # Training configuration
```

## Troubleshooting

### OCR Not Available
- Install PaddleOCR: `pip install paddlepaddle paddleocr`
- Or install Tesseract with Arabic language data
- Training will continue without OCR loss if OCR is unavailable

### Font Issues
- Ensure Arabic fonts are installed on your system
- Specify font paths in config or use system defaults
- Some fonts may not support all Arabic characters

### Memory Issues
- Reduce `train_batch_size`
- Enable `gradient_checkpointing`
- Use `mixed_precision: fp16` or `bf16`

### Training Instability
- Reduce learning rate
- Increase `gradient_accumulation_steps`
- Check that Arabic words file is valid UTF-8

## Citation

If you use this training approach, please cite:

```bibtex
@article{hassan2021arabic,
  title={Arabic Scene Text Recognition in the Deep Learning Era: Analysis on A Novel Dataset},
  author={Hassan, Heba and El-Mahdy, Ahmed and Hussein, Mohamed E},
  journal={IEEE Access},
  year={2021},
  publisher={IEEE}
}
```

## License

This training code follows the same license as the main project (Apache 2.0).

## Contributing

This is a research phase. Contributions should:
- Not break existing approaches (Approach 1/2)
- Maintain safety fallbacks
- Include proper evaluation metrics
- Document limitations clearly

