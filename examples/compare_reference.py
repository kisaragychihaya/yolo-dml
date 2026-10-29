"""QC tool: compare DLL output JSON against the Python golden reference.

Usage:
    python examples/compare_reference.py <dll_json> <reference_json>
        [--conf-tol 0.01] [--iou-tol 0.99] [--kpt-tol 1.5] [--vis-tol 0.02]
        [--probiou-tol 0.9] [--angle-tol 0.02] [--corner-tol 2.0]

Exit code 0 = pass, 1 = fail. Prints a per-detection match table.
The DLL schema has extra timing fields; only geometry/confidence/class are compared. When the
reference carries pose `keypoints` (and the DLL output does too), every keypoint of a matched
detection is compared as well: `--kpt-tol` pixels on x/y and `--vis-tol` on the visibility.
When the reference carries obb `xywhr`, matching uses the probabilistic IoU (`--probiou-tol`)
and the angle (`--angle-tol` radians) and corner (`--corner-tol` px) deviations are compared too.
"""
import argparse
import json
import math
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


def probiou(a, b, eps=1e-7):
    """Probabilistic IoU of two `xywhr` boxes ([cx, cy, w, h, angle], ultralytics batch_probiou)."""
    def cov(r):
        var_w, var_h = r[2] ** 2 / 12.0, r[3] ** 2 / 12.0
        c, s = math.cos(r[4]), math.sin(r[4])
        return (var_w * c * c + var_h * s * s,
                var_w * s * s + var_h * c * c,
                (var_w - var_h) * s * c)

    a1, b1, c1 = cov(a)
    a2, b2, c2 = cov(b)
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5
    det1 = max(a1 * b1 - c1 ** 2, 0.0)
    det2 = max(a2 * b2 - c2 ** 2, 0.0)
    t3 = math.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
                  / (4 * math.sqrt(det1 * det2) + eps) + eps) * 0.5
    bd = min(max(t1 + t2 + t3, eps), 100.0)
    return 1.0 - math.sqrt(1.0 - math.exp(-bd) + eps)


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


def compare_obb(reference, detection, angle_tol, corner_tol):
    """Compare the `xywhr`/`corners` of a matched obb pair; returns
    `(worst_angle, worst_corner_px, verdict_or_None)`."""
    if "xywhr" not in detection or "corners" not in detection:
        return 0.0, 0.0, "BAD(xywhr/corners missing on the dll side)"
    worst_angle = abs(float(detection["xywhr"][4]) - float(reference["xywhr"][4]))
    if worst_angle > angle_tol:
        return worst_angle, 0.0, f"BAD(dangle={worst_angle:.4f})"
    worst_corner = 0.0
    for index, (ref_pt, dll_pt) in enumerate(zip(reference["corners"], detection["corners"])):
        deviation = max(abs(float(dll_pt[0]) - float(ref_pt[0])),
                        abs(float(dll_pt[1]) - float(ref_pt[1])))
        worst_corner = max(worst_corner, deviation)
        if deviation > corner_tol:
            return worst_angle, worst_corner, f"BAD(corner{index} dxy={deviation:.2f})"
    return worst_angle, worst_corner, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll_json")
    ap.add_argument("reference_json")
    ap.add_argument("--conf-tol", type=float, default=0.01)
    ap.add_argument("--iou-tol", type=float, default=0.99)
    ap.add_argument("--kpt-tol", type=float, default=1.5, help="pose keypoint tolerance in px")
    ap.add_argument("--vis-tol", type=float, default=0.02, help="keypoint visibility tolerance")
    ap.add_argument("--probiou-tol", type=float, default=0.9,
                    help="obb probabilistic IoU gate (probiou of identical boxes is ~0.9995)")
    ap.add_argument("--angle-tol", type=float, default=0.02, help="obb angle tolerance in radians")
    ap.add_argument("--corner-tol", type=float, default=2.0, help="obb corner tolerance in px")
    args = ap.parse_args()

    dll = load(args.dll_json)
    ref = load(args.reference_json)
    dets_dll = dll["detections"]
    dets_ref = ref["detections"]
    obb = any(r.get("xywhr") is not None for r in dets_ref)

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
            if obb:
                if "xywhr" not in d:
                    continue
                v = probiou(r["xywhr"], d["xywhr"])
            else:
                v = iou(d, r)
            if v > best_iou:
                best, best_iou = i, v
        if best is None:
            rows.append((r["class_name"], r["confidence"], None, 0.0, 0.0, 0.0, 0.0, 0.0,
                         "NO MATCH"))
            ok = False
            continue
        used.add(best)
        d = dets_dll[best]
        conf_diff = abs(d["confidence"] - r["confidence"])
        iou_gate = args.probiou_tol if obb else args.iou_tol
        verdict = "ok" if best_iou >= iou_gate and conf_diff <= args.conf_tol \
            else f"BAD(iou={best_iou:.4f},dconf={conf_diff:.4f})"
        passed = verdict == "ok"

        worst_px, worst_vis, worst_angle = 0.0, 0.0, 0.0
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
        if obb:
            worst_angle, worst_px, obb_verdict = compare_obb(
                r, d, args.angle_tol, args.corner_tol)
            if obb_verdict is not None:
                verdict = obb_verdict
                passed = False

        rows.append((r["class_name"], r["confidence"], d["confidence"], best_iou,
                     worst_px, worst_vis, worst_angle, conf_diff if obb else 0.0, verdict))
        if not passed:
            ok = False

    for i, d in enumerate(dets_dll):
        if i not in used:
            rows.append((d["class_name"], d["confidence"], None, 0.0, 0.0, 0.0, 0.0, 0.0, "EXTRA"))
            ok = False

    pose = any(r.get("keypoints") is not None for r in dets_ref)
    iou_name = "probIoU" if obb else "IoU    "
    header = f"class_name | ref_conf | dll_conf | {iou_name} "
    if pose:
        header += "| kpt_px | kpt_v  "
    if obb:
        header += "| dangle | crn_px "
    header += "| verdict"
    print(header)
    for name, rc, dc, v, kpt_px, kpt_v, angle, _dc, verdict in rows:
        line = (f"{name:<10} | {rc:8.4f} | "
                f"{('%.4f' % dc) if dc is not None else '  --  ':>8} | {v:.4f} ")
        if pose:
            line += f"| {kpt_px:6.2f} | {kpt_v:6.4f} "
        if obb:
            line += f"| {angle:6.4f} | {kpt_px:6.2f} "
        line += f"| {verdict}"
        print(line)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
