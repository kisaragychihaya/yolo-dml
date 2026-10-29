"""End-to-end smoke test for the `yolo_dml` cdylib through ctypes.

Loads target/<profile>/yolo_dml.dll, runs the C ABI on the sample images, writes the
returned JSON and a visualisation next to the Python reference in assets/, and prints a
short summary. Run from the repository root:

    python examples/demo.py             # debug build, detect model
    python examples/demo.py --release   # release build, detect model
    python examples/demo.py --pose      # pose model, draws the keypoints too
    python examples/demo.py --obb       # OBB model, draws the rotated boxes (boats.jpg)

`--pose` loads `yolo26n-pose.onnx` (exported by scripts/fetch-test-assets.ps1) and writes
`dll_pose_<image>.json` / `dll_overlay_pose_<image>.jpg`, so the detect artefacts stay
intact. `--obb` loads `yolo26n-obb.onnx` and writes `dll_obb_*`; the DLL target directory
can be overridden with YOLO_DML_TARGET when it is not target/<profile>. `--model` runs
any other exported model instead.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

# absolute(), not resolve(): keep the caller's path spelling — resolve() chases directory
# junctions, which can turn an ASCII working path into a non-ASCII real one (cv2.imread
# and the DLLs' file opens cannot handle those).
ROOT = Path(__file__).absolute().parent.parent
ASSETS = ROOT / "assets"
MODEL = ROOT / "yolo26n.onnx"
POSE_MODEL = ROOT / "yolo26n-pose.onnx"
OBB_MODEL = ROOT / "yolo26n-obb.onnx"
IMAGES = ("bus.jpg", "zidane.jpg")
OBB_IMAGES = ("boats.jpg",)
CONF = 0.25
IOU = 0.7

# COCO 17-keypoint skeleton (ultralytics `kpt_line=True`), drawn only between keypoints that
# are visible enough.
KPT_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
KPT_VISIBLE = 0.5


def load_library(release: bool) -> ctypes.CDLL:
    profile = "release" if release else "debug"
    target = Path(os.environ.get("YOLO_DML_TARGET", ROOT / "target"))
    dll = target / profile / "yolo_dml.dll"
    if not dll.is_file():
        sys.exit(f"missing {dll}; run `cargo build{' --release' if release else ''}` first")

    # Let the OS resolve the DLLs that sit next to the cdylib (opencv_world4100.dll, ...).
    os.add_dll_directory(str(dll.parent))
    # ort (load-dynamic) looks "onnxruntime.dll" up next to the *host* executable, so point it
    # at the copy build.rs placed beside our own library.
    ort_dll = dll.parent / "onnxruntime.dll"
    if ort_dll.is_file():
        os.environ.setdefault("YOLO_ORT_PATH", str(ort_dll))

    lib = ctypes.CDLL(str(dll))
    lib.yolo_version.restype = ctypes.c_char_p
    lib.yolo_version.argtypes = []
    lib.yolo_create.restype = ctypes.c_int64
    lib.yolo_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.yolo_detect_file.restype = ctypes.c_void_p
    lib.yolo_detect_file.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_float, ctypes.c_float]
    lib.yolo_free_string.restype = None
    lib.yolo_free_string.argtypes = [ctypes.c_void_p]
    lib.yolo_destroy.restype = None
    lib.yolo_destroy.argtypes = [ctypes.c_int64]
    lib.yolo_last_error.restype = ctypes.c_char_p
    lib.yolo_last_error.argtypes = []
    return lib


def ensure_cv2() -> None:
    """Re-run under `py/.venv` when the calling interpreter cannot import cv2."""
    try:
        import cv2  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = ROOT / "py" / ".venv" / "Scripts" / "python.exe"
    if os.environ.get("YOLO_DEMO_REEXEC") or not venv_python.is_file():
        sys.exit("opencv-python is required for the overlays; run this with "
                 "py/.venv/Scripts/python.exe")
    env = dict(os.environ, YOLO_DEMO_REEXEC="1")
    sys.exit(subprocess.call([str(venv_python), *sys.argv], env=env))


def last_error(lib: ctypes.CDLL) -> str:
    return (lib.yolo_last_error() or b"").decode("utf-8", "replace")


def detect_file(lib: ctypes.CDLL, handle: int, image: Path) -> dict:
    ptr = lib.yolo_detect_file(handle, os.fsencode(str(image)), CONF, IOU)
    if not ptr:
        raise RuntimeError(f"yolo_detect_file({image.name}) failed: {last_error(lib)}")
    try:
        payload = ctypes.cast(ptr, ctypes.c_char_p).value
    finally:
        lib.yolo_free_string(ptr)
    if payload is None:
        raise RuntimeError("detector returned an empty string")
    return json.loads(payload.decode("utf-8"))


def draw(image_path: Path, detections: list[dict], out_path: Path) -> None:
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {image_path}")
    for d in detections:
        label = f"{d['class_name']} {d['confidence']:.2f}"
        corners = d.get("corners")
        if corners:  # obb: rotated box
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            anchor = (int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min()))
        else:
            p1 = (int(round(d["x1"])), int(round(d["y1"])))
            p2 = (int(round(d["x2"])), int(round(d["y2"])))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            anchor = p1
        cv2.putText(img, label, (anchor[0], max(anchor[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        draw_keypoints(cv2, img, d.get("keypoints"))
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"cannot write {out_path}")


def kpt_visible(point: list[float]) -> bool:
    """A keypoint is drawn when it has no visibility channel or the visibility clears 0.5."""
    return len(point) < 3 or point[2] >= KPT_VISIBLE


def draw_keypoints(cv2, img, keypoints: list[list[float]] | None) -> None:
    """Draw the COCO skeleton and the keypoint dots of one detection."""
    if not keypoints:
        return
    points = [(int(round(k[0])), int(round(k[1]))) for k in keypoints]
    visible = [kpt_visible(k) for k in keypoints]
    for start, end in KPT_SKELETON:
        if start < len(points) and end < len(points) and visible[start] and visible[end]:
            cv2.line(img, points[start], points[end], (0, 200, 255), 2, cv2.LINE_AA)
    for point, shown in zip(points, visible):
        if shown:
            cv2.circle(img, point, 3, (0, 0, 255), -1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="load target/release/yolo_dml.dll")
    ap.add_argument("--pose", action="store_true",
                    help="run yolo26n-pose.onnx and draw the keypoints")
    ap.add_argument("--obb", action="store_true",
                    help="run yolo26n-obb.onnx on boats.jpg and draw the rotated boxes")
    ap.add_argument("--model", type=Path, default=None,
                    help="exported .onnx model to run (default: yolo26n.onnx, or yolo26n-pose.onnx "
                         "with --pose, yolo26n-obb.onnx with --obb)")
    args = ap.parse_args()

    model = args.model or (POSE_MODEL if args.pose else OBB_MODEL if args.obb else MODEL)
    if not model.is_file():
        sys.exit(f"missing model {model}; run scripts/fetch-test-assets.ps1 first")
    prefix = "dll_pose" if args.pose else "dll_obb" if args.obb else "dll"
    images = OBB_IMAGES if args.obb else IMAGES

    ensure_cv2()
    lib = load_library(args.release)
    print(f"yolo_version: {lib.yolo_version().decode()}")

    handle = lib.yolo_create(os.fsencode(str(model)), None)
    if handle <= 0:
        sys.exit(f"yolo_create failed: {last_error(lib)}")
    print(f"handle: {handle} model={model.name}")

    try:
        for name in images:
            image = ASSETS / name
            payload = detect_file(lib, handle, image)
            json_path = ASSETS / f"{prefix}_{image.stem}.json"
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            overlay = ASSETS / f"{prefix}_overlay_{image.stem}.jpg"
            draw(image, payload["detections"], overlay)

            meta = payload["meta"]
            print(f"\n== {name} ({meta['width']}x{meta['height']}) "
                  f"count={payload['count']} device={meta['device']} task={meta['task']} "
                  f"inference_ms={meta['inference_ms']}")
            for i, det in enumerate(payload["detections"][:3], 1):
                keypoints = det.get("keypoints")
                corners = det.get("corners")
                summary = dict(det)
                if keypoints is not None:
                    # keypoint arrays are long: print how many instead of all of them
                    summary["keypoints"] = f"<{len(keypoints)} points>"
                if corners is not None:
                    summary["corners"] = "<4 points>"
                print(f"  top{i}: {json.dumps(summary, ensure_ascii=False)}")
            print(f"  -> {json_path.relative_to(ROOT)}, {overlay.relative_to(ROOT)}")
    finally:
        lib.yolo_destroy(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
