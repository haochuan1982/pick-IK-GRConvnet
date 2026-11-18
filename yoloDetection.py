"""YOLO (Ultralytics export, OpenVINO IR) detector. Runs on NPU, falls back to CPU."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


@dataclass
class Detection:
    class_id: int
    label: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) in original-image pixels

    @property
    def center(self):
        return (0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3]))


def _load_names(model_xml: Path, model) -> dict:
    """Class names from Ultralytics' metadata.yaml next to the IR, or from the IR's rt_info."""
    meta = model_xml.parent / "metadata.yaml"
    if meta.exists():
        import yaml
        names = yaml.safe_load(meta.read_text()).get("names")
        if names:
            return {int(k): str(v) for k, v in dict(names).items()}
    try:
        labels = model.get_rt_info(["model_info", "labels"]).astype(str).split()
        return dict(enumerate(labels))
    except Exception:
        return {}


class YoloDetector:
    def __init__(self, model_xml, device="NPU", imgsz=640, conf=0.4, iou=0.5):
        self.model_xml = Path(model_xml)
        core = ov.Core()
        model = core.read_model(str(self.model_xml))
        self.names = _load_names(self.model_xml, model)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        model.reshape([1, 3, imgsz, imgsz])  # NPU needs static shapes
        try:
            self.compiled = core.compile_model(model, device)
            self.device = device
        except Exception as e:
            print(f"[yolo] cannot compile on {device} ({e}); falling back to CPU")
            self.compiled = core.compile_model(model, "CPU")
            self.device = "CPU"
        print(f"[yolo] running on {self.device}; classes: {len(self.names)}")

    def class_id_of(self, name: str) -> int:
        for k, v in self.names.items():
            if v == name:
                return k
        raise ValueError(f"class {name!r} not in model classes {sorted(set(self.names.values()))[:20]}...; "
                         "use --class-id, or a model trained on your brick")

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def detect(self, rgb, class_id=None):
        """rgb: HxWx3 uint8 (RGB). Returns detections sorted by confidence (optionally one class only)."""
        h, w = rgb.shape[:2]
        canvas, r, left, top = self._letterbox(rgb)  # Ultralytics models take RGB
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.compiled(x)[self.compiled.output(0)][0].T  # (N, 4+nc)
        boxes, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if class_id is not None:
            keep &= cls == class_id
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if len(boxes) == 0:
            return []
        xywh = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                                boxes[:, 2], boxes[:, 3]])
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.conf, self.iou)).reshape(-1)
        dets = []
        for i in idx:
            cx, cy, bw, bh = boxes[i]
            x1, x2 = np.clip([(cx - bw / 2 - left) / r, (cx + bw / 2 - left) / r], 0, w - 1)
            y1, y2 = np.clip([(cy - bh / 2 - top) / r, (cy + bh / 2 - top) / r], 0, h - 1)
            dets.append(Detection(int(cls[i]), self.names.get(int(cls[i]), str(int(cls[i]))),
                                  float(conf[i]), (float(x1), float(y1), float(x2), float(y2))))
        return sorted(dets, key=lambda d: -d.conf)
