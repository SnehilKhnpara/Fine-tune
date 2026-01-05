# EvArEST Dataset Setup

The EvArEST (Everyday Arabic-English Scene Text) dataset is used as a **secondary dataset** for OCR-based evaluation and weak supervision during training.

## Important Notes

⚠️ **EvArEST is NOT used as a text-to-image dataset**
- It is used ONLY for:
  - OCR-based evaluation
  - OCR-guided loss
  - Weak supervision

## Dataset Source

Repository: https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text

Paper: "Arabic Scene Text Recognition in the Deep Learning Era: Analysis on A Novel Dataset"
- Hassan, H., El-Mahdy, A., & Hussein, M. E. (2021). IEEE Access.

## Dataset Structure

The EvArEST dataset contains:

### Detection Dataset
- 510 images with text detection annotations
- Four-point polygon annotations
- Text files with polygon coordinates and language labels

### Recognition Dataset (Primary for Training)
- 7232 cropped word images
- Both Arabic and English
- Ground truth text files

### Synthetic Data
- ~200k synthetic images with segmentation maps
- Code available at: https://github.com/HGamal11/Arabic_Synthetic_Data_Generator

## Setup Instructions

1. **Clone the EvArEST repository**:
   ```bash
   git clone https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text.git
   cd EvArEST-dataset-for-Arabic-scene-text
   ```

2. **Download the dataset**:
   - Follow instructions in the repository README
   - Download the recognition dataset (cropped word images)
   - Extract to a directory

3. **Organize the directory structure**:
   ```
   evarest_data_dir/
     Recognition/
       train/
         image1.jpg
         image2.jpg
         ...
       train_gt.txt
       test/
         image1.jpg
         ...
       test_gt.txt
   ```

4. **Ground truth file format**:
   ```
   image1.jpg text_in_image
   image2.jpg another_text
   ...
   ```

5. **Use in training**:
   ```bash
   python training/arabic_diffusion/train_lora.py \
       --evarest_data_dir /path/to/evarest_data_dir \
       --evarest_weight 0.1 \
       ...
   ```

## Citation

If you use the EvArEST dataset, please cite:

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

Please check the EvArEST repository for license information.

