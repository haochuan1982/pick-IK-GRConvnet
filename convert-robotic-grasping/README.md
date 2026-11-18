# GR-ConvNet to OpenVINO

`convert_grconvnet.py` converts GR-ConvNet grasp-detection weights from the Hugging Face Hub to an OpenVINO IR (`model.xml` + `model.bin`). `infer_grconvnet.py` runs the exported model on a fake RGB-D image to check the data flow.

`optimum-cli` can't be used because the repo is not a transformers model, so the script loads the weights into the original GR-ConvNet class (`inference/models/grconvnet3.py` from [skumra/robotic-grasping](https://github.com/skumra/robotic-grasping)) and converts directly.

## Requirements

```bash
uv venv --python=3.12 .export-venv
source .export-venv/bin/activate
uv pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/tags/v2026.4.1/demos/common/export_models/requirements.txt --index-strategy unsafe-best-match
```

- Weights: [alvin0523/grconvnet-jacquard-v2](https://huggingface.co/alvin0523/grconvnet-jacquard-v2/tree/main) (download `model.pt`, or let the script fetch it)
- A clone of the original repo:

```bash
git clone https://github.com/skumra/robotic-grasping
```

## Usage

### 1. Convert

```bash
python convert_grconvnet.py --weights model.pt --repo-path robotic-grasping/
```

Without `--weights`, the script downloads the checkpoint from the Hub.

Options:

| Option | Description | Default |
|---|---|---|
| `--repo-path PATH` | Clone of `skumra/robotic-grasping` (required) | |
| `--repo-id ID` | Hugging Face repo to download from | `alvin0523/grconvnet-jacquard-v2` |
| `--weights FILE` | Local weight file; skips the Hub download | |
| `--output FILE` | Output IR path | `models/grconvnet-jacquard-v2/model.xml` |
| `--input-channels N` | 4 = RGB-D, 3 = RGB, 1 = depth | 4 |
| `--height`, `--width` | Example input size used for tracing | 480, 640 |
| `--no-fp16` | Keep FP32 weights in the IR | off (FP16) |

### 2. Test inference

```bash
python infer_grconvnet.py [path/to/model.xml]
```

The input is fake, so the predicted grasp is meaningless. It only shows the preprocessing and output flow.

## Notes

- Weight files are loaded with `weights_only=True` first. The script falls back to full unpickling only if that fails, so use it only on files you trust.
- The input shape is NCHW, e.g. `[1, 4, 480, 640]`. The channel count must match the weights (`--input-channels`).
- Run both scripts from the `convert-robotic-grasping` directory so the default model path resolves.
