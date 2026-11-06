"""Object detection with a YOLO model.

By default ONNX Runtime on the CPU (.onnx); optionally a Google Coral (Edge TPU, .tflite) or
OpenVINO (.xml) for the iGPU. Works with YOLOv8/YOLO11 exports. The output has the shape
[1, 4 + classes, N]. One shared detector for all cameras.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import LABEL_CLASSES

log = logging.getLogger(__name__)

CLASS_TO_LABEL = {cls: label for label, classes in LABEL_CLASSES.items() for cls in classes}
# Google's SSD models number the classes with the gaps of the original COCO list (90 ids);
# YOLO uses the 80 classes without gaps. Only the classes Uitkijk uses.
SSD_TO_COCO80 = {0: 0, 2: 2, 3: 3, 5: 5, 7: 7, 15: 14, 16: 15, 17: 16}


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels of the input frame


_OV_PROBE = r"""
import sys, numpy as np, openvino as ov
core = ov.Core()
dev, s = sys.argv[2], int(sys.argv[3])
if dev == "GPU" and not any(d.startswith("GPU") for d in core.available_devices):
    sys.exit(3)
m = core.read_model(sys.argv[1])
m.reshape([1, 3, s, s])
c = core.compile_model(m, dev, {"PERFORMANCE_HINT": "LATENCY"})
c.create_infer_request().infer({0: np.zeros((1, 3, s, s), np.float32)})
"""


def openvino_works(model_path: str | Path, device: str, input_size: int) -> bool:
    """Test OpenVINO on a device in a separate process. A broken driver or an OpenVINO version
    that cannot handle the CPU/container may crash (segfault); that must not happen to the
    service itself."""
    try:
        r = subprocess.run([sys.executable, "-c", _OV_PROBE, str(model_path), device, str(input_size)],
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        log.warning("OpenVINO test on %s took too long", device)
        return False
    if r.returncode == 0:
        return True
    if r.returncode == 3:
        log.info("No GPU found for OpenVINO")
    else:
        detail = (r.stderr.strip().splitlines() or [f"exitcode {r.returncode}"])[-1]
        log.warning("OpenVINO on %s not usable (%s)", device, detail)
    return False


class ObjectDetector:
    """Loads a YOLO model: .onnx via ONNX Runtime (CPU), .tflite on a Coral (Edge TPU),
    .xml via OpenVINO (GPU/CPU)."""

    def __init__(self, model_path: str | Path, device: str = "AUTO", input_size: int = 320):
        self.input_size = int(input_size)
        self._lock = threading.Lock()
        self.accelerated = False   # True = the model does not run on the CPU
        path = Path(model_path)
        if path.suffix == ".onnx":
            self._load_onnx(path)
        elif path.suffix == ".tflite":
            self._load_edgetpu(path)
        else:
            self._load_openvino(path, device.upper())
        log.info("Object model %s loaded via %s on %s (input %dpx)",
                 path.name, self.backend, self.device, self.input_size)

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if isinstance(inp.shape[2], int):
            self.input_size = int(inp.shape[2])   # fixed size from the export
        self.backend, self.device = "ONNX Runtime", "CPU"
        self._run = lambda blob: self.session.run(None, {self.input_name: blob})[0]

    def _load_edgetpu(self, path: Path) -> None:
        """Model for the Coral (Ultralytics `format=edgetpu`, int8). Raises without a Coral."""
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        delegate = load_delegate("libedgetpu.so.1")   # ValueError when there is no Coral
        interp = Interpreter(model_path=str(path), experimental_delegates=[delegate])
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        outs = interp.get_output_details()
        self.backend, self.device = "TFLite", "Edge TPU"
        self.accelerated = True
        if len(outs) == 4:
            self._setup_ssd(interp, inp, outs)
            return
        out = outs[0]
        self.input_size = size = int(inp["shape"][1])   # NHWC
        in_scale, in_zero = inp["quantization"]
        out_scale, out_zero = out["quantization"]
        in_dtype = inp["dtype"]
        in_range = np.iinfo(in_dtype) if np.issubdtype(in_dtype, np.integer) else None

        def run(blob: np.ndarray) -> np.ndarray:
            x = blob.transpose(0, 2, 3, 1)               # NCHW → NHWC
            if in_scale:
                x = x / in_scale + in_zero
            if in_range is not None:
                x = np.clip(np.round(x), in_range.min, in_range.max)
            interp.set_tensor(inp["index"], x.astype(in_dtype))
            interp.invoke()
            y = interp.get_tensor(out["index"]).astype(np.float32)
            if out_scale:
                y = (y - out_zero) * out_scale
            y[:, :4] *= size     # TFLite exports return normalised boxes (0-1)
            return y

        self._run = run

    def _setup_ssd(self, interp, inp, outs) -> None:
        """Google's ready-made Coral models (SSDLite MobileDet, SSD MobileNet; also used by
        Frigate): uint8 RGB input and the detections already worked out by the model
        (boxes, classes, scores, count)."""
        self.input_size = size = int(inp["shape"][1])
        # usual order: boxes [1,N,4], classes [1,N], scores [1,N], count [1]
        boxes_i = next(o["index"] for o in outs if len(o["shape"]) == 3)
        count_i = next(o["index"] for o in outs if int(np.prod(o["shape"])) == 1)
        classes_i, scores_i = (o["index"] for o in outs if o["index"] not in (boxes_i, count_i))

        def run_ssd(rgb: np.ndarray):
            interp.set_tensor(inp["index"], rgb[np.newaxis].astype(np.uint8))
            interp.invoke()
            n = int(interp.get_tensor(count_i)[0])
            return (interp.get_tensor(boxes_i)[0][:n], interp.get_tensor(classes_i)[0][:n].astype(int),
                    interp.get_tensor(scores_i)[0][:n])

        self._ssd = run_ssd
        self._ssd_size = size

    def _load_openvino(self, path: Path, wanted: str) -> None:
        candidates = {"AUTO": ["GPU", "CPU"], "GPU": ["GPU", "CPU"]}.get(wanted, [wanted])
        chosen = next((d for d in candidates if openvino_works(path, d, self.input_size)), None)
        if chosen is None:
            raise RuntimeError("OpenVINO does not work on any device in this container; "
                               "use an .onnx model (see the README)")
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(path))
        model.reshape([1, 3, self.input_size, self.input_size])
        compiled = core.compile_model(model, chosen, {"PERFORMANCE_HINT": "LATENCY"})
        request = compiled.create_infer_request()
        self.backend, self.device = "OpenVINO", chosen
        self.accelerated = chosen == "GPU"
        self._run = lambda blob: next(iter(request.infer({0: blob}).values()))

    def _letterbox(self, bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = bgr.shape[:2]
        s = self.input_size
        scale = min(s / w, s / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, scale, px, py

    def detect(self, bgr: np.ndarray, labels: list[str], min_score: float) -> list[Detection]:
        if not labels:
            return []
        wanted = {cls for lbl in labels for cls in LABEL_CLASSES[lbl]}
        img, scale, px, py = self._letterbox(bgr)
        if getattr(self, "_ssd", None) is not None:
            return self._detect_ssd(bgr, img, scale, px, py, wanted, min_score)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)  # NCHW, RGB, 0-1
        with self._lock:
            pred = self._run(blob)[0]                # [4 + C, N]
        pred = pred.T                                # [N, 4 + C]
        class_scores = pred[:, 4:]
        cls_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(cls_ids)), cls_ids]
        keep = (scores >= min_score) & np.isin(cls_ids, list(wanted))
        if not keep.any():
            return []
        boxes_cxcywh = pred[keep, :4]
        scores, cls_ids = scores[keep], cls_ids[keep]
        h, w = bgr.shape[:2]
        x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - px) / scale
        y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - py) / scale
        bw = boxes_cxcywh[:, 2] / scale
        bh = boxes_cxcywh[:, 3] / scale
        rects = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
        idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), min_score, 0.45)
        result = []
        for i in np.array(idx).flatten():
            x, y, bw_, bh_ = rects[i]
            box = (max(int(x), 0), max(int(y), 0), min(int(x + bw_), w - 1), min(int(y + bh_), h - 1))
            result.append(Detection(CLASS_TO_LABEL[int(cls_ids[i])], float(scores[i]), box))
        return result

    def _detect_ssd(self, bgr, img, scale, px, py, wanted, min_score) -> list[Detection]:
        with self._lock:
            boxes, classes, scores = self._ssd(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w = bgr.shape[:2]
        s = self._ssd_size
        result = []
        for (y1, x1, y2, x2), c, score in zip(boxes, classes, scores):
            cls = SSD_TO_COCO80.get(int(c))
            if cls is None or cls not in wanted or score < min_score:
                continue
            box = (max(int((x1 * s - px) / scale), 0), max(int((y1 * s - py) / scale), 0),
                   min(int((x2 * s - px) / scale), w - 1), min(int((y2 * s - py) / scale), h - 1))
            if box[2] > box[0] and box[3] > box[1]:
                result.append(Detection(CLASS_TO_LABEL[cls], float(score), box))
        return result


def draw_detections(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        text = f"{d.label} {d.score:.0%}"
        cv2.putText(out, text, (x1 + 3, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return out
