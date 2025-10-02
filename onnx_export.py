# from @nshmyrev's fork (adapted): robust ONNX export for MB-iSTFT-VITS2
#
# Usage example:
#   python onnx_export.py \
#     --config ./configs/istft_vits2_base.json \
#     --checkpoint ./G_91000.pth \
#     --out ./model.onnx \
#     --opset 15 \
#     --sid 0 \
#     --tid 0
#
# Notes:
# - sid/tid are optional; they'll be ignored if the model wasn't trained with speakers/tones.
# - Export uses dynamic axes for batch and sequence/time dimensions.

import argparse
import os
import torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols

# -------------------- Inline config (set USE_INLINE=True to ignore CLI) --------------------
USE_INLINE = True  # If True, use the variables below instead of argparse
CONFIG = ""         # e.g. "./configs/istft_vits2_base.json"
CHECKPOINT = ""     # e.g. "./G_91000.pth"
OUT = "model.onnx"
OPSET = 15
SID = 0
TID = 0
SEQ_LEN = 50
DEVICE = "cpu"      # "cpu" or "cuda"
# -----------------------------------------------------------------------------------------


def build_model(hps, checkpoint_path: str, device: torch.device) -> SynthesizerTrn:
    """Instantiate SynthesizerTrn with ONNX-friendly flags and load weights."""
    # Decide posterior channels
    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder is True:
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    # Some configs may omit n_tones; default to 1
    n_tones = getattr(hps.data, "n_tones", 1)

    net_g = SynthesizerTrn(
        n_vocab=len(symbols),
        spec_channels=posterior_channels,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_tones=n_tones,
        is_onnx=True,
        **hps.model,
    ).to(device)

    _ = utils.load_checkpoint(checkpoint_path, net_g, None)

    net_g.eval()
    with torch.no_grad():
        # Remove weight norms for export-stable graph
        if hasattr(net_g, "dec") and hasattr(net_g.dec, "remove_weight_norm"):
            net_g.dec.remove_weight_norm()
        if hasattr(net_g, "flow") and hasattr(net_g.flow, "remove_weight_norm"):
            net_g.flow.remove_weight_norm()

    return net_g


class OnnxWrapper(torch.nn.Module):
    """Wrap model.infer into an exportable forward(text, text_lengths, scales, sid, tid)."""

    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor, scales: torch.Tensor,
                sid: torch.Tensor = None, tid: torch.Tensor = None) -> torch.Tensor:
        # Scales: [noise_scale, length_scale, noise_scale_w]
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        o, _, _, _, _ = self.model.infer(
            text, text_lengths,
            sid=sid, tid=tid,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
        )
        # o shape: [B, 1, T] -> export [B, T]
        return o.squeeze(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MB-iSTFT-VITS2 to ONNX")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--checkpoint", required=True, help="Path to G_XXXX.pth")
    p.add_argument("--out", default="model.onnx", help="Output ONNX path")
    p.add_argument("--opset", type=int, default=15, help="ONNX opset version")
    p.add_argument("--sid", type=int, default=0, help="Speaker id (optional)")
    p.add_argument("--tid", type=int, default=0, help="Tone id (optional)")
    p.add_argument("--seq-len", type=int, default=50, help="Dummy input phoneme length for tracing")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for export")
    return p.parse_args()


def main():
    # Choose source of arguments
    if USE_INLINE:
        args = argparse.Namespace(
            config=CONFIG,
            checkpoint=CHECKPOINT,
            out=OUT,
            opset=OPSET,
            sid=SID,
            tid=TID,
            seq_len=SEQ_LEN,
            device=DEVICE,
        )
    else:
        args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    hps = utils.get_hparams_from_file(args.config)
    model = build_model(hps, args.checkpoint, device)
    wrapper = OnnxWrapper(model).to(device)

    # Dummy inputs for tracing (batch=1)
    num_symbols = model.n_vocab
    dmy_text = torch.randint(low=0, high=num_symbols, size=(1, args.seq_len), dtype=torch.long, device=device)
    dmy_text_length = torch.LongTensor([dmy_text.size(1)]).to(device)
    dmy_scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32, device=device)
    dmy_sid = torch.LongTensor([args.sid]).to(device)
    dmy_tid = torch.LongTensor([args.tid]).to(device)

    # Export
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        model=wrapper,
        args=(dmy_text, dmy_text_length, dmy_scales, dmy_sid, dmy_tid),
        f=args.out,
        verbose=False,
        opset_version=args.opset,
        input_names=["input", "input_lengths", "scales", "sid", "tid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "sid": {0: "batch_size"},
            "tid": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )
    print(f"Exported ONNX model to {args.out}")


if __name__ == "__main__":
    main()
