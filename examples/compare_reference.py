"""QC tool: compare DLL output JSON against the Python golden reference.

Usage:
    python examples/compare_reference.py <dll_json> <reference_json> [--conf-tol 0.01] [--iou-tol 0.99]

Exit code 0 = pass, 1 = fail. Prints a per-detection match table.
The DLL schema has extra timing fields; only geometry/confidence/class are compared.
"""
import argparse
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll_json")
    ap.add_argument("reference_json")
    ap.add_argument("--conf-tol", type=float, default=0.01)
    ap.add_argument("--iou-tol", type=float, default=0.99)
    args = ap.parse_args()

    dll = load(args.dll_json)
    ref = load(args.reference_json)
    dets_dll = dll["detections"]
    dets_ref = ref["detections"]

    ok = True
    if len(dets_dll) != len(dets_ref):
        print(f"FAIL: count mismatch dll={len(dets_dll)} ref={len(dets_ref)}")
        ok = False

    used = set()
    rows = []
    for r in dets_ref:
        best, best_iou = None, 0.0
        for i, d in enumerate(dets_dll):
            if i in used or d["class_id"] != r["class_id"]:
                continue
            v = iou(d, r)
            if v > best_iou:
                best, best_iou = i, v
        if best is None:
            rows.append((r["class_name"], r["confidence"], None, 0.0, "NO MATCH"))
            ok = False
            continue
        used.add(best)
        d = dets_dll[best]
        conf_diff = abs(d["confidence"] - r["confidence"])
        passed = best_iou >= args.iou_tol and conf_diff <= args.conf_tol
        rows.append((r["class_name"], r["confidence"], d["confidence"], best_iou,
                     "ok" if passed else f"BAD(iou={best_iou:.4f},dconf={conf_diff:.4f})"))
        if not passed:
            ok = False

    for i, d in enumerate(dets_dll):
        if i not in used:
            rows.append((d["class_name"], d["confidence"], None, 0.0, "EXTRA"))
            ok = False

    print(f"class_name | ref_conf | dll_conf | IoU    | verdict")
    for name, rc, dc, v, verdict in rows:
        print(f"{name:<10} | {rc:8.4f} | {('%.4f' % dc) if dc is not None else '  --  ':>8} | {v:.4f} | {verdict}")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
