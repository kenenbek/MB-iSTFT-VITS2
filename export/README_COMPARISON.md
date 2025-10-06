# Export and Comparison Guide

## Files Overview

### Fixed Issues

All three files now have proper path handling since they were moved into the `export/` subdirectory:

1. **onnx_export.py** - Exports PyTorch checkpoint to ONNX format
2. **onnx_inference.py** - Runs inference using ONNX model
3. **inference.py** - Runs inference using PyTorch checkpoint
4. **compare_outputs.py** - NEW: Compares checkpoint vs ONNX outputs side-by-side

All files now include:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
This adds the parent directory to Python path so imports work correctly.

### Important Note: Scale Order

The scale parameters are in this order: `[noise_scale, length_scale, noise_scale_w]`
- **noise_scale**: Controls synthesis quality (default: 0.667)
- **length_scale**: Controls speaking speed (1.0 = normal)
- **noise_scale_w**: Controls duration prediction (default: 0.8)

## Usage Guide

### Step 1: Export to ONNX

Edit configuration in `onnx_export.py`:
```python
PATH_TO_CONFIG = "/mnt/d/VITS2/config.json"
PATH_TO_MODEL = "/mnt/d/VITS2/G_91000.pth"
```

Run:
```bash
cd /home/kenenbek/PycharmProjects/MB-iSTFT-VITS2/export
python onnx_export.py
```

This creates `model.onnx` in the export directory.

### Step 2: Compare Outputs (RECOMMENDED)

Edit configuration in `compare_outputs.py`:
```python
PATH_TO_CONFIG = "/mnt/d/VITS2/config.json"
PATH_TO_MODEL = "/mnt/d/VITS2/G_91000.pth"
PATH_TO_ONNX = "model.onnx"
TEXT = "Your text here"
```

Run:
```bash
python compare_outputs.py
```

This will:
1. Generate audio from PyTorch checkpoint
2. Generate audio from ONNX model
3. Compare both outputs statistically
4. Save three files in `comparison_output/`:
   - `checkpoint_output.wav` - Audio from PyTorch model
   - `onnx_output.wav` - Audio from ONNX model
   - `difference.wav` - Audible difference (amplified)

### Step 3: Analyze Results

The comparison tool shows:
- **Absolute difference** (mean, max, std)
- **Relative error** (percentage differences)
- **Correlation coefficient** (should be close to 1.0)
- **MSE/MAE** (mean squared/absolute error)
- **SNR** (Signal-to-Noise Ratio in dB)

#### What to Look For:

**Good ONNX export:**
- Correlation > 0.999
- SNR > 40 dB
- Mean absolute difference < 0.001

**Potential issues if:**
- Correlation < 0.99
- SNR < 30 dB
- Large audible differences

### Common Issues and Fixes

#### Issue 1: Big Differences Between Outputs

**Possible causes:**
1. Weight normalization not removed properly
2. Random seed differences (dropout layers)
3. Numerical precision differences (float32 vs float64)

**Check:**
- Ensure `is_onnx=True` is set in `onnx_export.py`
- Verify `remove_weight_norm()` is called
- Use same random seed if needed

#### Issue 2: ONNX Model Not Found

Make sure you run `onnx_export.py` first to create `model.onnx`.

#### Issue 3: Import Errors

All files now have proper path handling, but ensure you run from the export directory:
```bash
cd /home/kenenbek/PycharmProjects/MB-iSTFT-VITS2/export
```

## Alternative: Run Individual Scripts

### Run PyTorch Inference Only

Edit `inference.py` and run:
```bash
python inference.py
```

### Run ONNX Inference Only

Edit inline config in `onnx_inference.py`:
```python
USE_INLINE = True
MODEL_PATH = "model.onnx"
CONFIG = "/mnt/d/VITS2/config.json"
TEXT = "Your text here"
```

Run:
```bash
python onnx_inference.py
```

Or use command line:
```bash
python onnx_inference.py \
  --model model.onnx \
  --config /mnt/d/VITS2/config.json \
  --text "Your text here" \
  --out output.wav \
  --sid 0 \
  --tid 0
```

## Troubleshooting Audio Differences

If you hear big differences:

1. **Check the comparison statistics** - Run `compare_outputs.py` to get numerical data
2. **Listen to difference.wav** - This amplifies the differences
3. **Verify same parameters** - Ensure noise_scale, noise_scale_w, length_scale match
4. **Check model architecture** - Some layers may not export properly to ONNX
5. **Test with shorter text** - Start with simple text to isolate issues

## Performance Notes

- ONNX models are typically faster for inference
- Use `USE_GPU = True` for faster generation (requires onnxruntime-gpu)
- PyTorch checkpoint is more flexible but slower
- ONNX is better for deployment/production

