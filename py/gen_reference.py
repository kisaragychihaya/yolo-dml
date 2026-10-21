"""Generate golden reference detections for the end2end YOLO ONNX model.

Runs the exported end2end model (output (1,300,6) = xyxy + conf + cls, NMS already
applied) with onnxruntime on the test images and dumps reference JSON, raw model
output and visualization overlays into assets/.

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
MAX_DET = 300

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


def postprocess(raw: np.ndarray, names: list[str], w: int, h: int,
                new_w: int, new_h: int, left: int, top: int) -> list[dict]:
    preds = raw[0]  # (300, 6) xyxy, conf, cls
    keep = preds[:, 4] > CONF_THRES
    preds = preds[keep][:MAX_DET]
    if len(preds) == 0:
        return []

    boxes = preds[:, :4].astype(np.float64).copy()
    gain_x, gain_y = new_w / w, new_h / h
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= gain_x
    boxes[:, [1, 3]] /= gain_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    dets = []
    for (x1, y1, x2, y2), row in zip(boxes, preds):
        cid = int(round(float(row[5])))
        dets.append({
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "confidence": round(float(row[4]), 6),
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else f"class_{cid}",
        })
    return dets


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

    dets = postprocess(raw, names, w, h, new_w, new_h, left, top)

    stem = image_path.stem
    np.save(ASSETS / f"raw_{stem}.npy", raw)

    payload = {
        "count": len(dets),
        "detections": dets,
        "meta": {
            "image": image_path.name,
            "width": w,
            "height": h,
            "letterbox": {"new_w": new_w, "new_h": new_h, "pad_x": left, "pad_y": top},
        },
    }
    (ASSETS / f"reference_{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cv2.imwrite(str(ASSETS / f"overlay_{stem}.jpg"), draw(img, dets))

    print(f"\n== {image_path.name} ({w}x{h}) letterbox new_w={new_w} new_h={new_h} "
          f"pad_x={left} pad_y={top} -> {len(dets)} detections")
    for i, d in enumerate(dets[:3], 1):
        print(f"  top{i}: {json.dumps(d, ensure_ascii=False)}")
    return payload


def main() -> None:
    print(f"onnxruntime={ort.__version__} opencv-python={cv2.__version__} numpy={np.__version__}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    names = load_names(session)
    print(f"model={MODEL_PATH.name} names={len(names)} first={names[:3]}")
    for img_name in ("bus.jpg", "zidane.jpg"):
        run(ASSETS / img_name, session, names)


if __name__ == "__main__":
    main()
