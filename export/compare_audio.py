"""
Compare audio outputs from PyTorch checkpoint and ONNX model.
This script helps verify that the ONNX export produces identical results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt

import utils
import commons
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

try:
    import onnxruntime as ort
except ImportError:
    print("WARNING: onnxruntime not installed. Install with: pip install onnxruntime")
    ort = None

# Configuration
PATH_TO_CONFIG = "/mnt/d/VITS2/config.json"
PATH_TO_MODEL = "/mnt/d/VITS2/G_91000.pth"
PATH_TO_ONNX = "model.onnx"
TEST_TEXT = "Алар кой таштардан, арасында кум-шагыл ширелген майда жумуру таштардан."

NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
LENGTH_SCALE = 1.0
SID = 0
TID = 0


def get_text(text, hps):
    """Convert text to phoneme sequence."""
    import re
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return text_norm


def inference_pytorch(config_path, model_path, text):
    """Run inference with PyTorch checkpoint."""
    print("\n=== PyTorch Checkpoint Inference ===")
    
    # Load config
    hps = utils.get_hparams_from_file(config_path)
    
    # Setup model
    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder:
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    
    # Process text
    text_norm = get_text(text, hps)
    text_norm = torch.LongTensor(text_norm).to(device).unsqueeze(0)
    text_lengths = torch.LongTensor([text_norm.size(1)]).to(device)
    
    sid = torch.LongTensor([SID]).to(device) if hps.data.n_speakers > 1 else None
    tid = torch.LongTensor([TID]).to(device) if hasattr(hps.data, 'n_tones') and hps.data.n_tones > 1 else None
    
    # Run inference
    with torch.no_grad():
        audio = net_g.infer(
            text_norm, 
            text_lengths, 
            sid=sid, 
            tid=tid,
            noise_scale=NOISE_SCALE, 
            noise_scale_w=NOISE_SCALE_W, 
            length_scale=LENGTH_SCALE
        )[0][0, 0].data.cpu().float().numpy()
    
    print(f"PyTorch output shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"PyTorch audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"PyTorch audio mean: {audio.mean():.6f}, std: {audio.std():.6f}")
    
    return audio, hps.data.sampling_rate


def inference_onnx(onnx_path, config_path, text):
    """Run inference with ONNX model."""
    print("\n=== ONNX Model Inference ===")
    
    if ort is None:
        print("ERROR: onnxruntime not available")
        return None, None
    
    # Load config
    hps = utils.get_hparams_from_file(config_path)
    
    # Process text
    text_norm = get_text(text, hps)
    text_norm = np.array(text_norm, dtype=np.int64).reshape(1, -1)
    text_lengths = np.array([text_norm.shape[1]], dtype=np.int64)
    
    # Prepare inputs
    scales = np.array([NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W], dtype=np.float32)
    sid_input = np.array([SID], dtype=np.int64)
    tid_input = np.array([TID], dtype=np.int64)
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Run inference
    inputs = {
        'input': text_norm,
        'input_lengths': text_lengths,
        'scales': scales,
        'sid': sid_input,
        'tid': tid_input
    }
    
    outputs = session.run(None, inputs)
    audio = outputs[0]
    
    print(f"ONNX raw output shape: {audio.shape}, dtype: {audio.dtype}")
    
    # Extract audio
    if audio.ndim == 3:
        audio = audio[0, 0, :]
    elif audio.ndim == 2:
        audio = audio[0, :]
    else:
        audio = audio.squeeze()
    
    audio = audio.astype(np.float32)
    
    print(f"ONNX output shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"ONNX audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"ONNX audio mean: {audio.mean():.6f}, std: {audio.std():.6f}")
    
    return audio, hps.data.sampling_rate


def compare_audio(audio1, audio2, sr, output_dir="comparison_output"):
    """Compare two audio arrays and save results."""
    print("\n=== Comparison Results ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio files
    write(f"{output_dir}/pytorch.wav", sr, audio1)
    write(f"{output_dir}/onnx.wav", sr, audio2)
    print(f"Saved audio files to {output_dir}/")
    
    # Align lengths (in case they differ slightly)
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Calculate differences
    diff = audio1 - audio2
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))
    
    print(f"Length: PyTorch={len(audio1)}, ONNX={len(audio2)}")
    print(f"Mean Squared Error (MSE): {mse:.10f}")
    print(f"Mean Absolute Error (MAE): {mae:.10f}")
    print(f"Max Absolute Difference: {max_diff:.10f}")
    
    # Calculate correlation
    correlation = np.corrcoef(audio1, audio2)[0, 1]
    print(f"Correlation coefficient: {correlation:.10f}")
    
    # Save difference audio
    diff_normalized = diff / (np.abs(diff).max() + 1e-8) * 0.5
    write(f"{output_dir}/difference.wav", sr, diff_normalized.astype(np.float32))
    
    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    time = np.arange(min_len) / sr
    
    axes[0].plot(time, audio1, alpha=0.7, label='PyTorch')
    axes[0].set_title('PyTorch Checkpoint Output')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, audio2, alpha=0.7, label='ONNX', color='orange')
    axes[1].set_title('ONNX Model Output')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, diff, alpha=0.7, color='red')
    axes[2].set_title(f'Difference (MAE: {mae:.6f}, Max: {max_diff:.6f})')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # Histogram of differences
    axes[3].hist(diff, bins=100, alpha=0.7, color='red')
    axes[3].set_title('Difference Distribution')
    axes[3].set_xlabel('Difference Value')
    axes[3].set_ylabel('Count')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.png", dpi=150)
    print(f"Saved comparison plot to {output_dir}/comparison.png")
    
    # Interpretation
    print("\n=== Interpretation ===")
    if correlation > 0.9999:
        print("✓ Excellent match! The outputs are nearly identical.")
    elif correlation > 0.999:
        print("✓ Very good match. Minor differences exist but should be imperceptible.")
    elif correlation > 0.99:
        print("⚠ Good match, but noticeable differences may exist.")
    else:
        print("✗ Significant differences detected. Check the export process.")


def main():
    print("Audio Comparison Tool for PyTorch vs ONNX")
    print("=" * 60)
    
    # Run PyTorch inference
    pytorch_audio, sr = inference_pytorch(PATH_TO_CONFIG, PATH_TO_MODEL, TEST_TEXT)
    
    # Run ONNX inference
    if not os.path.exists(PATH_TO_ONNX):
        print(f"\nERROR: ONNX model not found at {PATH_TO_ONNX}")
        print("Please run onnx_export.py first to generate the ONNX model.")
        return
    
    onnx_audio, _ = inference_onnx(PATH_TO_ONNX, PATH_TO_CONFIG, TEST_TEXT)
    
    if onnx_audio is not None:
        # Compare outputs
        compare_audio(pytorch_audio, onnx_audio, sr)


if __name__ == "__main__":
    main()

