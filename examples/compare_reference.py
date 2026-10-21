"""QC tool: compare DLL output JSON against the Python golden reference.

Usage:
    python examples/compare_reference.py <dll_json> <reference_json>
        [--conf-tol 0.01] [--iou-tol 0.99] [--kpt-tol 1.5] [--vis-tol 0.02]

Exit code 0 = pass, 1 = fail. Prints a per-detection match table.
The DLL schema has extra timing fields; only geometry/confidence/class are compared. When the
reference carries pose `keypoints` (and the DLL output does too), every keypoint of a matched
detection is compared as well: `--kpt-tol` pixels on x/y and `--vis-tol` on the visibility.
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


def point(keypoint):
    """`(x, y, visibility_or_None)` from `[x, y]`, `[x, y, v]` or `{"x":..,"y":..,"v":..}`."""
    if isinstance(keypoint, dict):
        visibility = keypoint.get("v")
        return (float(keypoint["x"]), float(keypoint["y"]),
                None if visibility is None else float(visibility))
    values = [float(value) for value in keypoint]
    return (values[0], values[1], values[2] if len(values) > 2 else None)


def compare_keypoints(reference, detection, kpt_tol, vis_tol):
    """Compare two keypoint lists; returns `(worst_px, worst_vis, verdict_or_None)`."""
    if len(reference) != len(detection):
        return 0.0, 0.0, f"KPT COUNT {len(detection)} vs {len(reference)}"
    worst_px = 0.0
    worst_vis = 0.0
    for index, (ref_point, dll_point) in enumerate(zip(reference, detection)):
        ref_x, ref_y, ref_v = point(ref_point)
        dll_x, dll_y, dll_v = point(dll_point)
        deviation = max(abs(dll_x - ref_x), abs(dll_y - ref_y))
        worst_px = max(worst_px, deviation)
        if deviation > kpt_tol:
            return worst_px, worst_vis, f"BAD(kpt{index} dxy={deviation:.2f})"
        if ref_v is not None and dll_v is not None:
            vis_deviation = abs(dll_v - ref_v)
            worst_vis = max(worst_vis, vis_deviation)
            if vis_deviation > vis_tol:
                return worst_px, worst_vis, f"BAD(kpt{index} dv={vis_deviation:.4f})"
    return worst_px, worst_vis, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll_json")
    ap.add_argument("reference_json")
    ap.add_argument("--conf-tol", type=float, default=0.01)
    ap.add_argument("--iou-tol", type=float, default=0.99)
    ap.add_argument("--kpt-tol", type=float, default=1.5, help="pose keypoint tolerance in px")
    ap.add_argument("--vis-tol", type=float, default=0.02, help="keypoint visibility tolerance")
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
            rows.append((r["class_name"], r["confidence"], None, 0.0, 0.0, 0.0,
                         "NO MATCH"))
            ok = False
            continue
        used.add(best)
        d = dets_dll[best]
        conf_diff = abs(d["confidence"] - r["confidence"])
        verdict = "ok" if best_iou >= args.iou_tol and conf_diff <= args.conf_tol \
            else f"BAD(iou={best_iou:.4f},dconf={conf_diff:.4f})"
        passed = verdict == "ok"

        worst_px, worst_vis = 0.0, 0.0
        ref_kpts, dll_kpts = r.get("keypoints"), d.get("keypoints")
        if ref_kpts is not None or dll_kpts is not None:
            if ref_kpts is None or dll_kpts is None:
                verdict = "BAD(keypoints missing on the " + \
                    ("reference" if ref_kpts is None else "dll") + " side)"
                passed = False
            else:
                worst_px, worst_vis, kpt_verdict = compare_keypoints(
                    ref_kpts, dll_kpts, args.kpt_tol, args.vis_tol)
                if kpt_verdict is not None:
                    verdict = kpt_verdict
                    passed = False

        rows.append((r["class_name"], r["confidence"], d["confidence"], best_iou,
                     worst_px, worst_vis, verdict))
        if not passed:
            ok = False

    for i, d in enumerate(dets_dll):
        if i not in used:
            rows.append((d["class_name"], d["confidence"], None, 0.0, 0.0, 0.0, "EXTRA"))
            ok = False

    pose = any(r.get("keypoints") is not None for r in dets_ref)
    header = "class_name | ref_conf | dll_conf | IoU    "
    header += "| kpt_px | kpt_v  | verdict" if pose else "| verdict"
    print(header)
    for name, rc, dc, v, kpt_px, kpt_v, verdict in rows:
        line = (f"{name:<10} | {rc:8.4f} | "
                f"{('%.4f' % dc) if dc is not None else '  --  ':>8} | {v:.4f} ")
        if pose:
            line += f"| {kpt_px:6.2f} | {kpt_v:6.4f} "
        line += f"| {verdict}"
        print(line)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
