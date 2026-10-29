"""Generate golden reference detections for the YOLO ONNX model.

Runs the exported model with onnxruntime on the test images and dumps reference
JSON, raw model output and visualization overlays into assets/. The ultralytics
export layouts supported are:

  end2end         (1, N, 6)     xyxy + conf + cls, NMS already applied by the model
  standard        (1, 4+nc, A)  xywh + per-class scores, decoded + class-aware NMS here
                                (mirrors ultralytics' non_max_suppression)
  standard-pose   (1, 4+nc+K*D, A)
                                standard detect channels plus K keypoints with D
                                values, already decoded to letterbox pixels by the
                                exported graph (see Pose/Pose26.kpts_decode export
                                branch); rows become xyxy+conf+cls+kpts here
  standard-obb    (1, 4+nc+1, A)  cx cy w h + per-class scores + angle (radians);
                                decoded here with probIoU rotated NMS (mirrors the
                                rotated=True branch of ultralytics' non_max_suppression)
  end2end-obb     (1, N, 7)       cx cy w h conf cls angle rows, NMS baked into the graph

Run:  py/.venv/Scripts/python.exe py/gen_reference.py               # detect goldens
      py/.venv/Scripts/python.exe py/gen_reference.py --task pose   # pose goldens
      py/.venv/Scripts/python.exe py/gen_reference.py --task pose --verify-e2e
      py/.venv/Scripts/python.exe py/gen_reference.py --task obb    # obb goldens (yolo26n-obb, end2end)
      py/.venv/Scripts/python.exe py/gen_reference.py --task obb --model yolo11n-obb.onnx
                                                                    # obb goldens, standard layout
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# absolute(), not resolve(): cv2.imread/open need the path the user invoked, not the
# resolved one — resolve() chases directory junctions, which can turn an ASCII working
# path into a non-ASCII real one that the native side cannot open.
SCRIPT_DIR = Path(__file__).absolute().parent
ROOT = SCRIPT_DIR.parent
ASSETS = ROOT / "assets"

MODELS = {
    "detect": ROOT / "yolo26n.onnx",
    "pose": ROOT / "yolo26n-pose.onnx",
    # yolo26 OBB exports are end-to-end (1, N, 7); pass --model yolo11n-obb.onnx to
    # generate goldens for the standard (1, 4+nc+1, anchors) layout instead.
    "obb": ROOT / "yolo26n-obb.onnx",
}
E2E_POSE_MODEL = ROOT / "yolo26n-pose-e2e.onnx"

IMAGES = ("bus.jpg", "zidane.jpg")
# OBB models are DOTA-trained; the COCO street scenes yield few to no boxes, so the
# obb goldens use the aerial sample image instead.
OBB_IMAGES = ("boats.jpg",)

IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.7  # ultralytics `iou` default used by predictors
MAX_DET = 300
CLASS_OFFSET = 7680.0  # ultralytics `max_wh`, separates classes during NMS

# Clip bounds mirror the ultralytics helpers the predictor calls: `scale_boxes`
# clips boxes to [0, w] (clip_boxes), `scale_coords` clips keypoints to [0, w]
# too (clip_coords). The pre-existing detect goldens were generated with the
# [0, w-1] bound, so that path keeps it to stay byte-identical.
DETECT_CLIP_MARGIN = 1
POSE_CLIP_MARGIN = 0

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


def letterbox(img: np.ndarray, new_shape: int | tuple[int, int] = IMGSZ, color: int = 114):
    """Ultralytics-style letterbox: resize preserving aspect ratio, pad to the canvas.

    `new_shape` is a square side or a (canvas_h, canvas_w) pair taken from the model's
    input shape (OBB checkpoints are exported at 1024, detect/pose at 640).
    """
    h, w = img.shape[:2]
    canvas_h, canvas_w = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
    r = min(canvas_h / h, canvas_w / w)
    new_w, new_h = round(w * r), round(h * r)
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (canvas_w - new_w) / 2.0
    dh = (canvas_h - new_h) / 2.0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    assert left + new_w + right == canvas_w, (left, new_w, right)
    assert top + new_h + bottom == canvas_h, (top, new_h, bottom)

    out = np.full((canvas_h, canvas_w, 3), color, dtype=img.dtype)
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


def load_kpt_shape(session: ort.InferenceSession) -> tuple[int, int] | None:
    """Parse the `kpt_shape` metadata ultralytics writes for pose exports, e.g. "(17, 3)"."""
    raw = session.get_modelmeta().custom_metadata_map.get("kpt_shape")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
        k, d = int(parsed[0]), int(parsed[1])
        return k, d
    except (ValueError, SyntaxError, TypeError, IndexError) as exc:
        raise ValueError(f"cannot parse 'kpt_shape' metadata {raw!r} ({exc})") from exc


def format_keypoints(kpts: np.ndarray) -> list[list[float]]:
    """(K, D) keypoint block -> [[x, y, v], ...], x/y at 2dp and v at 4dp."""
    out = []
    for kp in kpts:
        v = float(kp[2]) if kp.shape[0] > 2 else 1.0
        out.append([round(float(kp[0]), 2), round(float(kp[1]), 2), round(v, 4)])
    return out


def detect_layout(shape: tuple[int, ...], task: str = "detect",
                  kpt_shape: tuple[int, int] | None = None,
                  nc: int | None = None) -> str:
    """Classify the model output: end2end (1,N,6) vs standard (1,4+nc,anchors).

    With a pose model the keypoint channels are folded into the detect row, so both
    layouts are matched on their exact width rather than on a > test: standard is
    (1, 4+nc+K*D, anchors), end2end is (1, max_det, 6+K*D). OBB adds one angle
    channel: standard is (1, 4+nc+1, anchors), end2end is (1, max_det, 7) with
    cx cy w h conf cls angle rows (the rotated heads decode to xywh, not xyxy).
    """
    if len(shape) != 3:
        raise ValueError(f"unexpected output shape {shape}")
    if task == "pose":
        k, d = kpt_shape
        nc = 1 if nc is None else nc
        if shape[1] == 4 + nc + k * d:
            return "standard-pose"
        if shape[2] == 6 + k * d:
            return "end2end-pose"
        raise ValueError(f"unexpected pose output shape {shape} for kpt_shape={kpt_shape}, nc={nc}")
    if task == "obb":
        if shape[2] == 7:
            return "end2end-obb"
        if shape[1] > 5:
            return "standard-obb"
        raise ValueError(f"unexpected obb output shape {shape}")
    if shape[2] == 6:
        return "end2end"
    if shape[1] > 5:
        return "standard"
    raise ValueError(f"unexpected output shape {shape}")


def probiou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    """Probabilistic IoU of two (cx, cy, w, h, angle) boxes (ultralytics `batch_probiou`)."""
    def cov(r: np.ndarray) -> tuple[float, float, float]:
        var_w, var_h = r[2] ** 2 / 12.0, r[3] ** 2 / 12.0
        c, s = np.cos(r[4]), np.sin(r[4])
        return (var_w * c * c + var_h * s * s,
                var_w * s * s + var_h * c * c,
                (var_w - var_h) * s * c)

    a1, b1, c1 = cov(a)
    a2, b2, c2 = cov(b)
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5
    t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
                / (4 * np.sqrt(max(a1 * b1 - c1 ** 2, 0.0) * max(a2 * b2 - c2 ** 2, 0.0)) + eps)
                + eps) * 0.5
    bd = min(max(t1 + t2 + t3, eps), 100.0)
    return float(1.0 - np.sqrt(1.0 - np.exp(-bd) + eps))


def decode_standard_obb(pred: np.ndarray, nc: int,
                        iou_thres: float = IOU_THRES) -> np.ndarray:
    """Decode a standard (4+nc+1, anchors) OBB prediction into cx cy w h conf cls angle rows.

    Mirrors the rotated=True branch of ultralytics' non_max_suppression: best class per
    anchor, then class-aware rotated NMS with the probabilistic IoU and TorchNMS.fast_nms
    semantics — a candidate is suppressed by ANY higher-scored candidate (kept or not),
    class centres are offset by class_id * 7680.
    """
    scores = pred[4 : 4 + nc]
    best = scores.max(axis=0)
    cls = scores.argmax(axis=0)
    cand = best > CONF_THRES

    cx, cy = pred[0, cand], pred[1, cand]
    w, h = pred[2, cand], pred[3, cand]
    angle = pred[4 + nc, cand]
    conf, cls = best[cand], cls[cand]

    order = np.argsort(-conf)
    shift = cls.astype(np.float64) * CLASS_OFFSET
    boxes = np.stack([cx + shift, cy + shift, w, h, angle], axis=1)  # xywhr, offset centres
    suppressed = np.zeros(len(order), dtype=bool)
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            if not suppressed[b] and probiou(boxes[order[a]], boxes[order[b]]) > iou_thres:
                suppressed[b] = True
    picked = order[~suppressed][:MAX_DET]
    return np.column_stack([cx[picked], cy[picked], w[picked], h[picked],
                            conf[picked], cls[picked].astype(np.float64), angle[picked]])


def xywhr2xyxyxyxy(x: np.ndarray) -> np.ndarray:
    """(N, 5) xywhr -> (N, 4, 2) corner points (ultralytics `xywhr2xyxyxyxy`)."""
    cos, sin = np.cos(x[:, 4]), np.sin(x[:, 4])
    vec1 = np.stack([x[:, 2] / 2 * cos, x[:, 2] / 2 * sin], axis=1)
    vec2 = np.stack([-x[:, 3] / 2 * sin, x[:, 3] / 2 * cos], axis=1)
    ctr = x[:, :2]
    return np.stack([ctr + vec1 + vec2, ctr + vec1 - vec2,
                     ctr - vec1 - vec2, ctr - vec1 + vec2], axis=1)


def decode_standard(pred: np.ndarray, nc: int | None = None,
                    kpt_shape: tuple[int, int] | None = None,
                    iou_thres: float = IOU_THRES) -> np.ndarray:
    """Decode a standard (4+nc[+K*D], anchors) prediction into xyxy+conf+cls[+kpts] rows.

    Mirrors ultralytics' non_max_suppression: best class per anchor, xywh -> xyxy,
    and class-aware NMS where boxes are shifted by class_id * 7680 so only boxes
    of the same class compete. Scores are used as they come out of the graph
    (ultralytics exports already apply the sigmoid). For pose, everything past
    channel 4+nc is the K*D keypoint block, carried through NMS unchanged.
    """
    nc = pred.shape[0] - 4 if nc is None else nc
    scores = pred[4 : 4 + nc]  # (nc, anchors)
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
    rows = np.column_stack([boxes[picked_arr], conf[picked_arr], cls[picked_arr].astype(np.float64)])
    if kpt_shape is not None:
        k, d = kpt_shape
        extra = pred[4 + nc : 4 + nc + k * d].reshape(k * d, -1).T  # (anchors, K*D)
        rows = np.concatenate([rows, extra[cand][picked_arr].astype(np.float64)], axis=1)
    return rows


def postprocess_obb(raw: np.ndarray, layout: str, names: list[str], w: int, h: int,
                    new_w: int, new_h: int, left: int, top: int) -> list[dict]:
    """Decode an OBB output into detections with xywhr/corners in source-image pixels.

    Geometry mirrors ultralytics: the `xywh` branch of `scale_boxes` (padding off the
    centre, per-axis gains, NO clipping) and `xywhr2xyxyxyxy` for the corners; `x1..y2`
    is the minimal axis-aligned box enclosing the rotated one.
    """
    if layout == "end2end-obb":
        dets = raw[0]
        if dets[:, 4].max() > 1.0:
            raise ValueError("end2end confidences above 1.0; wrong output tensor selected?")
        dets = dets[dets[:, 4] > CONF_THRES][:MAX_DET]
    else:  # standard-obb
        nc = raw.shape[1] - 5
        if nc != len(names):
            raise ValueError(f"obb channel split gives nc={nc} but metadata names has {len(names)}")
        dets = decode_standard_obb(raw[0], nc=nc)
    if len(dets) == 0:
        return []

    gain_x, gain_y = new_w / w, new_h / h
    rboxes = dets[:, [0, 1, 2, 3, 6]].astype(np.float64).copy()  # cx cy w h angle
    rboxes[:, 0] = (rboxes[:, 0] - left) / gain_x
    rboxes[:, 1] = (rboxes[:, 1] - top) / gain_y
    rboxes[:, 2] /= gain_x
    rboxes[:, 3] /= gain_y
    corners = xywhr2xyxyxyxy(rboxes)

    out = []
    for rbox, pts, row in zip(rboxes, corners, dets):
        cid = int(round(float(row[5])))
        out.append({
            "x1": round(float(pts[:, 0].min()), 2),
            "y1": round(float(pts[:, 1].min()), 2),
            "x2": round(float(pts[:, 0].max()), 2),
            "y2": round(float(pts[:, 1].max()), 2),
            "confidence": round(float(row[4]), 6),
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else f"class_{cid}",
            "xywhr": [round(float(v), 4) for v in rbox[:4]] + [round(float(rbox[4]), 6)],
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in pts],
        })
    return out


def postprocess(raw: np.ndarray, layout: str, names: list[str], w: int, h: int,
                new_w: int, new_h: int, left: int, top: int,
                kpt_shape: tuple[int, int] | None = None) -> list[dict]:
    if layout in ("end2end", "end2end-pose", "end2end-obb"):
        dets = raw[0]
        if dets[:, 4].max() > 1.0:
            raise ValueError("end2end confidences above 1.0; wrong output tensor selected?")
    elif layout == "standard-pose":
        k, d = kpt_shape
        nc = raw.shape[1] - 4 - k * d
        if nc != len(names):
            raise ValueError(f"pose channel split gives nc={nc} but metadata names has {len(names)}")
        dets = decode_standard(raw[0], nc=nc, kpt_shape=kpt_shape)
    else:
        dets = decode_standard(raw[0])

    dets = dets[dets[:, 4] > CONF_THRES][:MAX_DET]
    if len(dets) == 0:
        return []

    is_pose = layout.endswith("pose")
    margin = POSE_CLIP_MARGIN if is_pose else DETECT_CLIP_MARGIN

    boxes = dets[:, :4].astype(np.float64).copy()
    gain_x, gain_y = new_w / w, new_h / h
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= gain_x
    boxes[:, [1, 3]] /= gain_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - margin)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - margin)

    kpts = None
    if is_pose:
        k, d = kpt_shape
        if d < 2:
            raise ValueError(f"unexpected kpt_shape {kpt_shape}")
        kpts = dets[:, 6 : 6 + k * d].reshape(len(dets), k, d).astype(np.float64).copy()
        kpts[..., 0] = (kpts[..., 0] - left) / gain_x
        kpts[..., 1] = (kpts[..., 1] - top) / gain_y
        kpts[..., 0] = kpts[..., 0].clip(0, w - POSE_CLIP_MARGIN)
        kpts[..., 1] = kpts[..., 1].clip(0, h - POSE_CLIP_MARGIN)

    out = []
    for i, ((x1, y1, x2, y2), row) in enumerate(zip(boxes, dets)):
        cid = int(round(float(row[5])))
        det = {
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "confidence": round(float(row[4]), 6),
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else f"class_{cid}",
        }
        if kpts is not None:
            det["keypoints"] = format_keypoints(kpts[i])
        out.append(det)
    return out


def draw(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        label = f"{d['class_name']} {d['confidence']:.2f}"
        corners = d.get("corners")
        if corners:  # obb: draw the rotated box
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            anchor = (int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min()))
        else:
            p1 = (int(round(d["x1"])), int(round(d["y1"])))
            p2 = (int(round(d["x2"])), int(round(d["y2"])))
            cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
            anchor = p1
        cv2.putText(vis, label, (anchor[0], max(anchor[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        for kx, ky, kv in d.get("keypoints", []):
            color = (0, 200, 0) if kv >= 0.5 else (160, 160, 160)  # green / grey
            cv2.circle(vis, (int(round(kx)), int(round(ky))), 3, color, -1, cv2.LINE_AA)
    return vis


def infer(image_path: Path, session: ort.InferenceSession, names: list[str], task: str = "detect",
          kpt_shape: tuple[int, int] | None = None):
    """Letterbox, run, postprocess. Returns (img, raw, layout, dets, letterbox geometry)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]

    input_shape = session.get_inputs()[0].shape  # (N, C, H, W); dynamic dims are not ints
    imgsz = ((int(input_shape[2]), int(input_shape[3]))
             if len(input_shape) == 4 and isinstance(input_shape[2], int)
             and isinstance(input_shape[3], int) else IMGSZ)
    lb, new_w, new_h, left, top = letterbox(img, imgsz)
    x = preprocess(lb)
    out_names = [o.name for o in session.get_outputs()]
    raw = session.run(["output0"] if "output0" in out_names else None, {session.get_inputs()[0].name: x})[0]
    raw = np.asarray(raw, dtype=np.float32)
    layout = detect_layout(raw.shape, task, kpt_shape, len(names))

    if layout.endswith("obb"):
        dets = postprocess_obb(raw, layout, names, w, h, new_w, new_h, left, top)
    else:
        dets = postprocess(raw, layout, names, w, h, new_w, new_h, left, top, kpt_shape)
    return img, raw, layout, dets, (new_w, new_h, left, top)


def run(image_path: Path, session: ort.InferenceSession, names: list[str], task: str = "detect",
        kpt_shape: tuple[int, int] | None = None, stem_prefix: str | None = None) -> dict:
    img, raw, layout, dets, (new_w, new_h, left, top) = infer(
        image_path, session, names, task, kpt_shape)
    h, w = img.shape[:2]

    prefix = stem_prefix if stem_prefix is not None else (task if task != "detect" else "")
    stem = f"{prefix}_{image_path.stem}" if prefix else image_path.stem
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
    if kpt_shape is not None:  # pose
        payload["meta"]["task"] = task
        payload["meta"]["kpt_shape"] = list(kpt_shape)
    elif task != "detect":
        payload["meta"]["task"] = task
    (ASSETS / f"reference_{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cv2.imwrite(str(ASSETS / f"overlay_{stem}.jpg"), draw(img, dets))

    print(f"\n== {image_path.name} ({w}x{h}) layout={layout} letterbox new_w={new_w} "
          f"new_h={new_h} pad_x={left} pad_y={top} -> {len(dets)} detections")
    for i, d in enumerate(dets[:3], 1):
        print(f"  top{i}: {json.dumps(d, ensure_ascii=False)}")
    return payload


def _iou(a: dict, b: dict) -> float:
    def area(d):
        return max(0.0, d["x2"] - d["x1"]) * max(0.0, d["y2"] - d["y1"])

    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(area(a) + area(b) - inter, 1e-9)


def compare_detections(preds: list[dict], refs: list[dict], label: str) -> None:
    """Greedy same-class IoU matching of two detection lists, then report max deltas."""
    pairs, unmatched_pred, used = [], [], set()
    for p in preds:
        best, best_j = 0.0, -1
        for j, r in enumerate(refs):
            if j in used or r["class_id"] != p["class_id"]:
                continue
            v = _iou(p, r)
            if v > best:
                best, best_j = v, j
        if best_j < 0 or best < 0.5:
            unmatched_pred.append(p)
            continue
        used.add(best_j)
        pairs.append((p, refs[best_j], best))
    unmatched_ref = [r for j, r in enumerate(refs) if j not in used]

    box_d = max((max(abs(p[k] - r[k]) for k in ("x1", "y1", "x2", "y2"))
                 for p, r, _ in pairs), default=float("nan"))
    conf_d = max((abs(p["confidence"] - r["confidence"]) for p, r, _ in pairs), default=float("nan"))
    kpt_d = vis_d = float("nan")
    if pairs and "keypoints" in pairs[0][0]:
        kpt_d = vis_d = 0.0
        for p, r, _ in pairs:
            for (px, py, pv), (rx, ry, rv) in zip(p["keypoints"], r["keypoints"]):
                kpt_d = max(kpt_d, abs(px - rx), abs(py - ry))
                vis_d = max(vis_d, abs(pv - rv))

    print(f"[{label}] preds={len(preds)} refs={len(refs)} matched={len(pairs)} "
          f"unmatched_pred={len(unmatched_pred)} unmatched_ref={len(unmatched_ref)}")
    print(f"[{label}] min_iou={min((v for *_, v in pairs), default=float('nan')):.6f} "
          f"max|dbox|={box_d:.4f}px max|dconf|={conf_d:.6f}")
    if not np.isnan(kpt_d):
        print(f"[{label}] max|dkpt|={kpt_d:.4f}px max|dvis|={vis_d:.6f}")


def print_metadata(session: ort.InferenceSession) -> None:
    meta = session.get_modelmeta()
    for k, v in sorted(meta.custom_metadata_map.items()):
        shown = v if len(v) <= 160 else f"{v[:160]}..."
        print(f"  {k} = {shown}")


def verify_e2e() -> None:
    """Decode the NMS-free pose export and diff it against the standard-layout goldens."""
    session = ort.InferenceSession(str(E2E_POSE_MODEL), providers=["CPUExecutionProvider"])
    out = session.get_outputs()[0]
    print(f"e2e model={E2E_POSE_MODEL.name} input={session.get_inputs()[0].name}"
          f"{session.get_inputs()[0].shape} output={out.name}{out.shape}")
    print_metadata(session)
    names = load_names(session)
    kpt_shape = load_kpt_shape(session)
    for img_name in IMAGES:
        img, raw, layout, dets, _ = infer(ASSETS / img_name, session, names, "pose", kpt_shape)
        ref_path = ASSETS / f"reference_pose_{Path(img_name).stem}.json"
        refs = json.loads(ref_path.read_text(encoding="utf-8"))["detections"]
        print(f"\n== {img_name} layout={layout} raw{raw.shape} rows={len(dets)}")
        compare_detections(dets, refs, f"e2e-vs-standard {img_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden reference detections from ONNX exports.")
    parser.add_argument("--task", choices=("detect", "pose", "obb"), default="detect",
                        help="golden set to regenerate (default: detect)")
    parser.add_argument("--model", type=Path, default=None,
                        help="ONNX model path (default: yolo26n.onnx / yolo26n-pose.onnx / "
                             "yolo26n-obb.onnx); with --task obb a custom model changes the "
                             "output prefix to obb_<family>, e.g. obb_yolo11n_boats.json")
    parser.add_argument("--verify-e2e", action="store_true",
                        help="decode yolo26n-pose-e2e.onnx and diff it against the pose goldens")
    args = parser.parse_args()

    print(f"onnxruntime={ort.__version__} opencv-python={cv2.__version__} numpy={np.__version__}")
    ASSETS.mkdir(parents=True, exist_ok=True)

    if args.verify_e2e:
        verify_e2e()
        return

    model_path = args.model or MODELS[args.task]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = load_names(session)
    kpt_shape = load_kpt_shape(session) if args.task == "pose" else None
    if args.task == "pose" and kpt_shape is None:
        raise SystemExit(f"{model_path.name}: no 'kpt_shape' metadata, not a pose export?")
    # a non-default obb model gets its family name in the prefix so the goldens of
    # yolo26n-obb.onnx (end2end) and yolo11n-obb.onnx (standard) do not collide
    stem_prefix = None
    if args.task == "obb" and args.model is not None:
        stem_prefix = f"obb_{model_path.stem.split('-')[0]}"
    print(f"model={model_path.name} outputs={[o.name for o in session.get_outputs()]} "
          f"names={len(names)} first={names[:3]} kpt_shape={kpt_shape}")
    images = OBB_IMAGES if args.task == "obb" else IMAGES
    for img_name in images:
        run(ASSETS / img_name, session, names, args.task, kpt_shape, stem_prefix)


if __name__ == "__main__":
    main()
