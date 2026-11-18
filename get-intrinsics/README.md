# RealSense intrinsics

`save_intrinsics.py` reads the intrinsics of a connected Intel RealSense D435i or D435 and saves them to an `.npz` file.

## Requirements

```
pip install pyrealsense2 numpy
```

## Usage

```
python save_intrinsics.py --list                 # show connected cameras and serials
python save_intrinsics.py --model D435i          # default; saves intrinsics_d435i.npz
python save_intrinsics.py --model D435           # saves intrinsics_d435.npz
python save_intrinsics.py --serial 123456789012  # select by serial number
```

Options:

| Option | Description | Default |
|---|---|---|
| `--model {D435i,D435}` | Camera model to use | `D435i` |
| `--serial SERIAL` | Select by serial number (overrides `--model`) | |
| `--list` | List connected devices and exit | |
| `--output FILE` | Output file name | `intrinsics_<model>.npz` |
| `--width`, `--height`, `--fps` | Stream resolution and frame rate | 640, 480, 30 |

If two cameras of the same model are connected, use `--serial`.

## Saved contents

| Key | Meaning |
|---|---|
| `color_K`, `depth_K` | 3x3 camera matrices `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `color_dist`, `depth_dist` | Distortion coefficients (usually zeros on D400 series) |
| `color_size`, `depth_size` | Stream `[width, height]` |
| `depth_scale` | Multiply raw depth values by this to get meters |
| `model`, `serial` | Camera name and serial number |

## Loading

```python
import numpy as np
d = np.load("intrinsics_d435i.npz")
print(d["color_K"], d["color_dist"], d["model"])
```

## Notes

- Intrinsics depend on the resolution. Re-run if you change `--width`/`--height`.
- Color and depth have different intrinsics. If you align depth to color (`rs.align(rs.stream.color)`), use `color_K`.
- Each camera has its own calibration, so keep the files separate.
