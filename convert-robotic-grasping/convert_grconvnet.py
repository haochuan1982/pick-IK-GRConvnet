"""Convert GR-ConvNet weights from the Hugging Face Hub to OpenVINO IR.

optimum-cli can't be used because the repo is not a transformers model.
This loads the weights into the original GR-ConvNet class and converts directly.

Setup:
    git clone https://github.com/skumra/robotic-grasping /path/to/robotic-grasping
    python convert_grconvnet.py --repo-path /path/to/robotic-grasping

The class is imported from inference/models/grconvnet3.py of that repo.
"""
import argparse
import sys
from pathlib import Path

import openvino as ov
import torch
from huggingface_hub import list_repo_files, snapshot_download

WEIGHT_SUFFIXES = (".pth", ".pt", ".bin", ".ckpt")


def find_weights(local_dir: Path) -> Path:
    candidates = [p for p in local_dir.rglob("*") if p.suffix in WEIGHT_SUFFIXES]
    if not candidates:
        sys.exit(f"No weight file ({', '.join(WEIGHT_SUFFIXES)}) found in {local_dir}")
    if len(candidates) > 1:
        print("Multiple weight files found, using the first:", *candidates, sep="\n  ")
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="alvin0523/grconvnet-jacquard-v2")
    ap.add_argument("--repo-path", required=True, help="Clone of skumra/robotic-grasping")
    ap.add_argument("--weights", help="Local weight file; skips the Hub download")
    ap.add_argument("--output", default="models/grconvnet-jacquard-v2/model.xml")
    ap.add_argument("--input-channels", type=int, default=4, help="4 = RGB-D, 3 = RGB, 1 = depth")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--no-fp16", action="store_true", help="Keep FP32 weights in the IR")
    args = ap.parse_args()

    if args.weights:
        weights = Path(args.weights)
    else:
        print("Repo files:", *list_repo_files(args.repo_id), sep="\n  ")
        weights = find_weights(Path(snapshot_download(args.repo_id)))
    print(f"Using weights: {weights}")

    sys.path.insert(0, args.repo_path)
    from inference.models.grconvnet3 import GenerativeResnet

    # weights_only=True is safe; fall back to a full pickle only if the file needs it.
    # Only allow the fallback for files you trust.
    try:
        obj = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"weights_only load failed ({type(e).__name__}); retrying with full unpickling")
        obj = torch.load(weights, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        model = GenerativeResnet(input_channels=args.input_channels)
        model.load_state_dict(state)
    model.eval()

    example = torch.randn(1, args.input_channels, args.height, args.width)
    ov_model = ov.convert_model(model, example_input=example)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, out, compress_to_fp16=not args.no_fp16)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
