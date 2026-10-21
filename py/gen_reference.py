"""Generate golden reference detections for the YOLO ONNX model.

Runs the exported model with onnxruntime on the test images and dumps reference
JSON, raw model output and visualization overlays into assets/. Both ultralytics
export layouts are supported:

  end2end   (1, N, 6)     xyxy + conf + cls, NMS already applied by the model
  standard  (1, 4+nc, A)  xywh + per-class scores, decoded + class-aware NMS here
                          (mirrors ultralytics' non_max_suppression)

Run:  py/.venv/Scripts/python.exe py/gen_reference.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ASSETS = ROOT / "assets"
MODEL_PATH = ROOT / "yolo26n.onnx"

IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.7  # ultralytics `iou` default used by predictors
MAX_DET = 300
CLASS_OFFSET = 7680.0  # ultralytics `max_wh`, separates classes during NMS

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def letterbox(img: np.ndarray, new_shape: int = IMGSZ, color: int = 114):
    """Ultralytics-style letterbox: resize preserving aspect ratio, pad to square."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = round(w * r), round(h * r)
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (new_shape - new_w) / 2.0
    dh = (new_shape - new_h) / 2.0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    assert left + new_w + right == new_shape, (left, new_w, right)
    assert top + new_h + bottom == new_shape, (top, new_h, bottom)

    out = np.full((new_shape, new_shape, 3), color, dtype=img.dtype)
    out[top : top + new_h, left : left + new_w] = img
    return out, new_w, new_h, left, top


def preprocess(img: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> RGB CHW float32 [0,1] with batch dim."""
    x = img[..., ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None]


def load_names(session: ort.InferenceSession) -> list[str]:
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict) and parsed:
                n = max(parsed) + 1
                return [str(parsed.get(i, f"class_{i}")) for i in range(n)]
        except (ValueError, SyntaxError) as exc:
            print(f"[warn] failed to parse 'names' metadata ({exc}); using COCO80 table")
    else:
        print("[warn] no 'names' metadata; using COCO80 table")
    return list(COCO80)


def detect_layout(shape: tuple[int, ...]) -> str:
    """Classify the model output: end2end (1,N,6) vs standard (1,4+nc,anchors)."""
    if len(shape) == 3 and shape[2] == 6:
        return "end2end"
    if len(shape) == 3 and shape[1] > 5:
        return "standard"
    raise ValueError(f"unexpected output shape {shape}")


def decode_standard(pred: np.ndarray, iou_thres: float = IOU_THRES) -> np.ndarray:
    """Decode a standard (4+nc, anchors) prediction into (K, 6) xyxy+conf+cls rows.

    Mirrors ultralytics' non_max_suppression: best class per anchor, xywh -> xyxy,
    and class-aware NMS where boxes are shifted by class_id * 7680 so only boxes
    of the same class compete. Scores are used as they come out of the graph
    (ultralytics exports already apply the sigmoid).
    """
    scores = pred[4:]  # (nc, anchors)
    best = scores.max(axis=0)
    cls = scores.argmax(axis=0)
    cand = best > CONF_THRES

    cx, cy = pred[0, cand], pred[1, cand]
    w, h = pred[2, cand], pred[3, cand]
    boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    conf = best[cand]
    cls = cls[cand]

    shifted = boxes + np.repeat(cls[:, None], 4, axis=1) * CLASS_OFFSET
    order = np.argsort(-conf)
    suppressed = np.zeros(len(conf), dtype=bool)
    picked: list[int] = []
    for rank, i in enumerate(order):
        if suppressed[i]:
            continue
        picked.append(i)
        if len(picked) >= MAX_DET:
            break
        rest = order[rank + 1 :]
        if len(rest) == 0:
            break
        ix1 = np.maximum(shifted[i, 0], shifted[rest, 0])
        iy1 = np.maximum(shifted[i, 1], shifted[rest, 1])
        ix2 = np.minimum(shifted[i, 2], shifted[rest, 2])
        iy2 = np.minimum(shifted[i, 3], shifted[rest, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_i = (shifted[i, 2] - shifted[i, 0]) * (shifted[i, 3] - shifted[i, 1])
        area_r = (shifted[rest, 2] - shifted[rest, 0]) * (shifted[rest, 3] - shifted[rest, 1])
        ious = inter / np.maximum(area_i + area_r - inter, 1e-9)
        suppressed[rest[ious > iou_thres]] = True

    picked_arr = np.asarray(picked, dtype=int)
    return np.column_stack([boxes[picked_arr], conf[picked_arr], cls[picked_arr].astype(np.float64)])


def postprocess(raw: np.ndarray, layout: str, names: list[str], w: int, h: int,
                new_w: int, new_h: int, left: int, top: int) -> list[dict]:
    if layout == "end2end":
        dets = raw[0]
        if dets[:, 4].max() > 1.0:
            raise ValueError("end2end confidences above 1.0; wrong output tensor selected?")
    else:
        dets = decode_standard(raw[0])

    dets = dets[dets[:, 4] > CONF_THRES][:MAX_DET]
    if len(dets) == 0:
        return []

    boxes = dets[:, :4].astype(np.float64).copy()
    gain_x, gain_y = new_w / w, new_h / h
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= gain_x
    boxes[:, [1, 3]] /= gain_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    out = []
    for (x1, y1, x2, y2), row in zip(boxes, dets):
        cid = int(round(float(row[5])))
        out.append({
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "confidence": round(float(row[4]), 6),
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else f"class_{cid}",
        })
    return out


def draw(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        p1 = (int(round(d["x1"])), int(round(d["y1"])))
        p2 = (int(round(d["x2"])), int(round(d["y2"])))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(vis, label, (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def run(image_path: Path, session: ort.InferenceSession, names: list[str]) -> dict:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]

    lb, new_w, new_h, left, top = letterbox(img)
    x = preprocess(lb)
    out_names = [o.name for o in session.get_outputs()]
    raw = session.run(["output0"] if "output0" in out_names else None, {session.get_inputs()[0].name: x})[0]
    raw = np.asarray(raw, dtype=np.float32)
    layout = detect_layout(raw.shape)

    dets = postprocess(raw, layout, names, w, h, new_w, new_h, left, top)

    stem = image_path.stem
    np.save(ASSETS / f"raw_{stem}.npy", raw)

    payload = {
        "count": len(dets),
        "detections": dets,
        "meta": {
            "image": image_path.name,
            "width": w,
            "height": h,
            "layout": layout,
            "letterbox": {"new_w": new_w, "new_h": new_h, "pad_x": left, "pad_y": top},
        },
    }
    (ASSETS / f"reference_{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cv2.imwrite(str(ASSETS / f"overlay_{stem}.jpg"), draw(img, dets))

    print(f"\n== {image_path.name} ({w}x{h}) layout={layout} letterbox new_w={new_w} "
          f"new_h={new_h} pad_x={left} pad_y={top} -> {len(dets)} detections")
    for i, d in enumerate(dets[:3], 1):
        print(f"  top{i}: {json.dumps(d, ensure_ascii=False)}")
    return payload


def main() -> None:
    print(f"onnxruntime={ort.__version__} opencv-python={cv2.__version__} numpy={np.__version__}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    names = load_names(session)
    print(f"model={MODEL_PATH.name} outputs={[o.name for o in session.get_outputs()]} "
          f"names={len(names)} first={names[:3]}")
    for img_name in ("bus.jpg", "zidane.jpg"):
        run(ASSETS / img_name, session, names)


if __name__ == "__main__":
    main()
