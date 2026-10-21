"""End-to-end smoke test for the `yolo_dml` cdylib through ctypes.

Loads target/<profile>/yolo_dml.dll, runs the C ABI on the sample images, writes the
returned JSON and a visualisation next to the Python reference in assets/, and prints a
short summary. Run from the repository root:

    python examples/demo.py             # debug build
    python examples/demo.py --release   # release build
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MODEL = ROOT / "yolo26n.onnx"
IMAGES = ("bus.jpg", "zidane.jpg")
CONF = 0.25
IOU = 0.7


def load_library(release: bool) -> ctypes.CDLL:
    profile = "release" if release else "debug"
    dll = ROOT / "target" / profile / "yolo_dml.dll"
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

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {image_path}")
    for d in detections:
        p1 = (int(round(d["x1"])), int(round(d["y1"])))
        p2 = (int(round(d["x2"])), int(round(d["y2"])))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(img, label, (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"cannot write {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="load target/release/yolo_dml.dll")
    args = ap.parse_args()

    ensure_cv2()
    lib = load_library(args.release)
    print(f"yolo_version: {lib.yolo_version().decode()}")

    handle = lib.yolo_create(os.fsencode(str(MODEL)), None)
    if handle <= 0:
        sys.exit(f"yolo_create failed: {last_error(lib)}")
    print(f"handle: {handle}")

    try:
        for name in IMAGES:
            image = ASSETS / name
            payload = detect_file(lib, handle, image)
            json_path = ASSETS / f"dll_{image.stem}.json"
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            overlay = ASSETS / f"dll_overlay_{image.stem}.jpg"
            draw(image, payload["detections"], overlay)

            meta = payload["meta"]
            print(f"\n== {name} ({meta['width']}x{meta['height']}) "
                  f"count={payload['count']} device={meta['device']} "
                  f"inference_ms={meta['inference_ms']}")
            for i, det in enumerate(payload["detections"][:3], 1):
                print(f"  top{i}: {json.dumps(det, ensure_ascii=False)}")
            print(f"  -> {json_path.relative_to(ROOT)}, {overlay.relative_to(ROOT)}")
    finally:
        lib.yolo_destroy(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
