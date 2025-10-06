"""
Comparison script to generate audio from both PyTorch checkpoint and ONNX model
and compare the outputs to identify differences.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import re
from scipy.io.wavfile import write

import utils
import commons
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not found. Install with: pip install onnxruntime")
    exit(1)

# -------------------- Configuration --------------------
PATH_TO_CONFIG = "/mnt/d/VITS2/config.json"
PATH_TO_MODEL = "/mnt/d/VITS2/G_91000.pth"
PATH_TO_ONNX = "model.onnx"

TEXT = "Алар кой таштардан, арасында кум-шагыл ширелген майда жумуру таштардан, ар кандай бүртүктөгү кумдардын жана кумдуу чопонун кабатчалары менен топторунан түзүлгөн аллүбий-пролүбий чөкмө тоо тектерден турат."

SID = 0
TID = 0
NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
LENGTH_SCALE = 1.0

OUTPUT_DIR = "comparison_output"
USE_GPU = True
# -------------------------------------------------------


def get_text(text, hps):
    """Convert text to phoneme sequence."""
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return text_norm


def generate_checkpoint_audio(text, config_path, model_path, sid, tid, 
                               noise_scale, noise_scale_w, length_scale, use_gpu):
    """Generate audio using PyTorch checkpoint."""
    print("\n" + "="*60)
    print("GENERATING AUDIO FROM PYTORCH CHECKPOINT")
    print("="*60)
    
    device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load config
    print(f"Loading config from {config_path}...")
    hps = utils.get_hparams_from_file(config_path)
    
    # Determine posterior channels
    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder:
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using linear posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False
    
    # Load model
    print(f"Loading model from {model_path}...")
    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    net_g.eval()
    
    _ = utils.load_checkpoint(model_path, net_g, None)
    
    # Process text
    print(f"Processing text: {text[:50]}...")
    text_norm = get_text(text, hps)
    text_tensor = torch.LongTensor(text_norm).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([text_tensor.size(1)]).to(device)
    
    sid_tensor = torch.LongTensor([sid]).to(device)
    tid_tensor = torch.LongTensor([tid]).to(device)
    
    # Generate audio
    print("Generating audio...")
    with torch.no_grad():
        audio = net_g.infer(
            text_tensor,
            text_lengths,
            sid=sid_tensor,
            tid=tid_tensor,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale
        )[0][0, 0].data.cpu().float().numpy()
    
    print(f"Generated audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / hps.data.sampling_rate:.2f} seconds")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"Audio mean: {audio.mean():.4f}, std: {audio.std():.4f}")
    
    return audio, hps.data.sampling_rate, text_norm


def generate_onnx_audio(text_norm, config_path, onnx_path, sid, tid,
                        noise_scale, noise_scale_w, length_scale, use_gpu):
    """Generate audio using ONNX model."""
    print("\n" + "="*60)
    print("GENERATING AUDIO FROM ONNX MODEL")
    print("="*60)
    
    # Load config
    print(f"Loading config from {config_path}...")
    hps = utils.get_hparams_from_file(config_path)
    
    # Prepare inputs
    text_norm = np.array(text_norm, dtype=np.int64).reshape(1, -1)
    text_lengths = np.array([text_norm.shape[1]], dtype=np.int64)
    scales = np.array([noise_scale, length_scale, noise_scale_w], dtype=np.float32)
    sid_input = np.array([sid], dtype=np.int64)
    tid_input = np.array([tid], dtype=np.int64)
    
    # Create ONNX session
    print(f"Loading ONNX model from {onnx_path}...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"Model inputs: {[inp.name for inp in session.get_inputs()]}")
    print(f"Model outputs: {[out.name for out in session.get_outputs()]}")
    
    # Run inference
    print("Generating audio...")
    inputs = {
        'input': text_norm,
        'input_lengths': text_lengths,
        'scales': scales,
        'sid': sid_input,
        'tid': tid_input
    }
    
    outputs = session.run(None, inputs)
    audio = outputs[0].squeeze()
    
    print(f"Generated audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / hps.data.sampling_rate:.2f} seconds")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"Audio mean: {audio.mean():.4f}, std: {audio.std():.4f}")
    
    return audio, hps.data.sampling_rate


def compare_audios(audio_checkpoint, audio_onnx, sample_rate):
    """Compare two audio arrays and print statistics."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Check if lengths match
    if len(audio_checkpoint) != len(audio_onnx):
        print(f"WARNING: Audio lengths differ!")
        print(f"  Checkpoint length: {len(audio_checkpoint)}")
        print(f"  ONNX length: {len(audio_onnx)}")
        min_len = min(len(audio_checkpoint), len(audio_onnx))
        audio_checkpoint = audio_checkpoint[:min_len]
        audio_onnx = audio_onnx[:min_len]
        print(f"  Comparing first {min_len} samples")
    else:
        print(f"Audio lengths match: {len(audio_checkpoint)} samples")
    
    # Calculate differences
    diff = audio_checkpoint - audio_onnx
    abs_diff = np.abs(diff)
    
    print(f"\nAbsolute difference statistics:")
    print(f"  Mean: {abs_diff.mean():.6f}")
    print(f"  Max: {abs_diff.max():.6f}")
    print(f"  Min: {abs_diff.min():.6f}")
    print(f"  Std: {abs_diff.std():.6f}")
    
    # Calculate relative error
    epsilon = 1e-8
    rel_error = abs_diff / (np.abs(audio_checkpoint) + epsilon)
    print(f"\nRelative error statistics:")
    print(f"  Mean: {rel_error.mean():.6f}")
    print(f"  Max: {rel_error.max():.6f}")
    
    # Calculate correlation
    correlation = np.corrcoef(audio_checkpoint, audio_onnx)[0, 1]
    print(f"\nCorrelation coefficient: {correlation:.8f}")
    
    # Calculate MSE and MAE
    mse = np.mean(diff ** 2)
    mae = np.mean(abs_diff)
    print(f"\nMSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    
    # Signal-to-noise ratio
    signal_power = np.mean(audio_checkpoint ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"\nSNR: {snr:.2f} dB")
    
    return diff


def main():
    """Main comparison function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("CHECKPOINT VS ONNX COMPARISON TOOL")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Config: {PATH_TO_CONFIG}")
    print(f"  Checkpoint: {PATH_TO_MODEL}")
    print(f"  ONNX model: {PATH_TO_ONNX}")
    print(f"  Speaker ID: {SID}")
    print(f"  Tone ID: {TID}")
    print(f"  Noise scale: {NOISE_SCALE}")
    print(f"  Noise scale W: {NOISE_SCALE_W}")
    print(f"  Length scale: {LENGTH_SCALE}")
    print(f"  Text: {TEXT[:50]}...")
    
    # Generate checkpoint audio
    audio_checkpoint, sample_rate, text_norm = generate_checkpoint_audio(
        TEXT, PATH_TO_CONFIG, PATH_TO_MODEL, SID, TID,
        NOISE_SCALE, NOISE_SCALE_W, LENGTH_SCALE, USE_GPU
    )
    
    # Generate ONNX audio
    audio_onnx, _ = generate_onnx_audio(
        text_norm, PATH_TO_CONFIG, PATH_TO_ONNX, SID, TID,
        NOISE_SCALE, NOISE_SCALE_W, LENGTH_SCALE, USE_GPU
    )
    
    # Compare
    diff = compare_audios(audio_checkpoint, audio_onnx, sample_rate)
    
    # Save all outputs
    print(f"\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint_output.wav")
    onnx_path = os.path.join(OUTPUT_DIR, "onnx_output.wav")
    diff_path = os.path.join(OUTPUT_DIR, "difference.wav")
    
    write(checkpoint_path, sample_rate, audio_checkpoint)
    print(f"Saved checkpoint audio: {checkpoint_path}")
    
    write(onnx_path, sample_rate, audio_onnx)
    print(f"Saved ONNX audio: {onnx_path}")
    
    # Normalize difference for saving
    if diff.max() > 0:
        diff_normalized = diff / diff.max() * 0.5
        write(diff_path, sample_rate, diff_normalized.astype(np.float32))
        print(f"Saved difference audio: {diff_path}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

