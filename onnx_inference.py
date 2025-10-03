# ONNX Inference Script for MB-iSTFT-VITS2
#
# Usage example:
#   python onnx_inference.py \
#     --model ./model.onnx \
#     --config ./configs/istft_vits2_base.json \
#     --text "Your text here" \
#     --out output.wav \
#     --sid 0 \
#     --tid 0
#
# Notes:
# - Requires onnxruntime: pip install onnxruntime or onnxruntime-gpu
# - sid/tid are optional (speaker and tone IDs)

import argparse
import os
import numpy as np
import re

import utils
import commons
from text import text_to_sequence
from scipy.io.wavfile import write

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not found. Install with: pip install onnxruntime")
    exit(1)

# -------------------- Inline config (set USE_INLINE=True to ignore CLI) --------------------
USE_INLINE = True  # If True, use the variables below instead of argparse
MODEL_PATH = "model.onnx"                    # Path to exported ONNX model
CONFIG = "/mnt/d/VITS2/config.json"          # Path to config.json
TEXT = "Алар кой таштардан, арасында кум-шагыл ширелген майда жумуру таштардан, ар кандай бүртүктөгү кумдардын жана кумдуу чопонун кабатчалары менен топторунан түзүлгөн аллүбий-пролүбий чөкмө тоо тектерден турат."
OUT = "output_onnx.wav"                      # Output wav file
SID = 0                                       # Speaker ID
TID = 0                                       # Tone ID
NOISE_SCALE = 0.667                          # Noise scale for synthesis
NOISE_SCALE_W = 0.8                          # Noise scale for duration
LENGTH_SCALE = 1.0                           # Speed control (1.0 = normal, <1.0 = faster, >1.0 = slower)
USE_GPU = True                               # Set to True to use GPU (requires onnxruntime-gpu)
# -----------------------------------------------------------------------------------------


def get_text(text, hps):
    """Convert text to phoneme sequence."""
    # Remove special characters
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    
    # Convert to phoneme sequence
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    
    # Add blank tokens between phonemes if required
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    
    return text_norm


def inference_onnx(model_path, config_path, text, output_path, 
                   sid=0, tid=0, 
                   noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0,
                   use_gpu=False):
    """
    Run ONNX inference for TTS synthesis.
    
    Args:
        model_path: Path to ONNX model file
        config_path: Path to config.json
        text: Input text to synthesize
        output_path: Output wav file path
        sid: Speaker ID (for multi-speaker models)
        tid: Tone ID (for multi-tone models)
        noise_scale: Noise scale for synthesis quality
        noise_scale_w: Noise scale for duration prediction
        length_scale: Speaking speed (1.0=normal, <1.0=faster, >1.0=slower)
        use_gpu: Whether to use GPU for inference
    """
    
    # Load config
    print(f"Loading config from {config_path}...")
    hps = utils.get_hparams_from_file(config_path)
    
    # Convert text to phoneme sequence
    print(f"Processing text: {text[:50]}...")
    text_norm = get_text(text, hps)
    text_norm = np.array(text_norm, dtype=np.int64).reshape(1, -1)
    text_lengths = np.array([text_norm.shape[1]], dtype=np.int64)
    
    # Prepare input tensors
    scales = np.array([noise_scale, length_scale, noise_scale_w], dtype=np.float32)
    sid_input = np.array([sid], dtype=np.int64)
    tid_input = np.array([tid], dtype=np.int64)
    
    # Create ONNX Runtime session
    print(f"Loading ONNX model from {model_path}...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Print model info
    print(f"Model inputs: {[input.name for input in session.get_inputs()]}")
    print(f"Model outputs: {[output.name for output in session.get_outputs()]}")
    
    # Prepare inputs dict
    inputs = {
        'input': text_norm,
        'input_lengths': text_lengths,
        'scales': scales,
        'sid': sid_input,
        'tid': tid_input
    }
    
    # Run inference
    print("Running inference...")
    outputs = session.run(None, inputs)
    audio = outputs[0].squeeze()  # Remove batch dimension
    
    # Save audio
    print(f"Saving audio to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write(output_path, hps.data.sampling_rate, audio)
    print(f"Done! Audio saved to {output_path}")
    print(f"Sample rate: {hps.data.sampling_rate} Hz")
    print(f"Duration: {len(audio) / hps.data.sampling_rate:.2f} seconds")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="ONNX Inference for MB-iSTFT-VITS2")
    p.add_argument("--model", required=True, help="Path to ONNX model file")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--text", required=True, help="Input text to synthesize")
    p.add_argument("--out", default="output.wav", help="Output wav file path")
    p.add_argument("--sid", type=int, default=0, help="Speaker ID (for multi-speaker models)")
    p.add_argument("--tid", type=int, default=0, help="Tone ID (for multi-tone models)")
    p.add_argument("--noise-scale", type=float, default=0.667, help="Noise scale for synthesis")
    p.add_argument("--noise-scale-w", type=float, default=0.8, help="Noise scale for duration")
    p.add_argument("--length-scale", type=float, default=1.0, help="Speed control (1.0=normal)")
    p.add_argument("--gpu", action="store_true", help="Use GPU for inference (requires onnxruntime-gpu)")
    return p.parse_args()


def main():
    """Main entry point."""
    if USE_INLINE:
        # Use inline configuration
        args = argparse.Namespace(
            model=MODEL_PATH,
            config=CONFIG,
            text=TEXT,
            out=OUT,
            sid=SID,
            tid=TID,
            noise_scale=NOISE_SCALE,
            noise_scale_w=NOISE_SCALE_W,
            length_scale=LENGTH_SCALE,
            gpu=USE_GPU,
        )
    else:
        # Use command line arguments
        args = parse_args()
    
    # Run inference
    inference_onnx(
        model_path=args.model,
        config_path=args.config,
        text=args.text,
        output_path=args.out,
        sid=args.sid,
        tid=args.tid,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()

