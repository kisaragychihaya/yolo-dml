//! Postprocessing: decode raw YOLO output tensors -> boxes -> NMS.
//!
//! Two output layouts are supported by the single-tensor decoders: the end-to-end export
//! (`1 x N x 6`, NMS already applied by the model, [`decode_end2end`]) and the standard export
//! (`1 x (4 + nc) x anchors`, needs transposing, thresholding and class-aware NMS here,
//! [`decode_and_nms`]). A third layout — the four-tensor `torchvision` style export
//! (`num_detections`, `detection_boxes`, `detection_scores`, `detection_classes`, NMS baked
//! into the graph) — is decoded by [`decode_nms_applied`].
//!
//! Pose models reuse those layouts with keypoint channels appended: the standard export becomes
//! `1 x (4 + nc + K*D) x anchors` ([`decode_pose`]) and the end-to-end one `1 x N x (6 + K*D)`
//! ([`decode_pose_end2end`]). Both share the box NMS of the detect path — the keypoints of a
//! surviving candidate are carried along by its index (see [`nms_indices`]).

use crate::detector::{Detection, Keypoint, Obb};
use crate::error::{Error, Result};
use crate::preprocess::Preprocessed;

/// Number of classes in the standard COCO checkpoints.
pub const COCO_CLASSES: usize = 80;

/// Maximum number of detections kept per image (ultralytics `max_det`).
pub const MAX_DET: usize = 300;

/// Column count of an end-to-end (`NMS already applied`) output row: `x1 y1 x2 y2 conf cls`.
const END2END_COLUMNS: usize = 6;

/// Keypoint layout of a pose model, parsed from the `kpt_shape` custom metadata.
///
/// Ultralytics stores it as the Python `repr` of a tuple, e.g. `"(17, 3)"`: 17 COCO person
/// keypoints with `(x, y, visibility)` each. A `(K, 2)` shape carries just `(x, y)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KptShape {
    /// Number of keypoints per detection (`K`).
    pub count: usize,
    /// Values per keypoint (`D`): 2 (`x y`) or 3 (`x y visibility`).
    pub dim: usize,
}

impl KptShape {
    /// COCO person keypoints (`kpt_shape` `(17, 3)`).
    pub const COCO: Self = Self { count: 17, dim: 3 };

    /// Number of output channels the keypoints occupy (`K * D`).
    pub const fn channels(self) -> usize {
        self.count * self.dim
    }

    /// Parse the `kpt_shape` metadata, tolerating tuple/list reprs and stray whitespace.
    ///
    /// `"(17, 3)"`, `"[17,3]"`, `"17, 3"` all read as 17 keypoints of 3 values; anything that
    /// does not yield exactly two positive numbers with `dim` in `2..=3` is rejected so that a
    /// malformed metadata entry falls back to the shape heuristics instead of mis-decoding.
    pub fn parse(raw: &str) -> Option<Self> {
        let numbers: Vec<usize> = raw
            .split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .filter_map(|part| part.parse().ok())
            .collect();
        match numbers.as_slice() {
            [count, dim] if *count > 0 && (2..=3).contains(dim) => Some(Self {
                count: *count,
                dim: *dim,
            }),
            _ => None,
        }
    }
}

/// `ultralytics` `max_wh`: boxes are shifted by `class_id * 7680` before NMS so that boxes
/// of different classes never suppress each other while a single NMS pass suffices.
const CLASS_OFFSET: f32 = 7680.0;

/// Class names used when the model metadata carries no (parsable) `names` entry.
pub const COCO80: [&str; COCO_CLASSES] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

/// Parse the `names` custom metadata of a model (`"{0: 'person', 1: 'bicycle', ...}"`, the
/// Python `repr` of a dict). Returns `None` when the payload is not in that shape.
pub fn parse_names(raw: &str) -> Option<Vec<String>> {
    let body = raw.trim();
    let body = body
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(body);
    if body.trim().is_empty() {
        return None;
    }

    let mut entries: Vec<(usize, String)> = Vec::new();
    for part in body.split(',') {
        let (key, value) = part.split_once(':')?;
        let id: usize = key.trim().parse().ok()?;
        let name = value
            .trim()
            .trim_matches(|c| c == '\'' || c == '"')
            .to_string();
        entries.push((id, name));
    }
    let count = entries.iter().map(|(id, _)| *id).max()? + 1;
    let mut names = vec![String::new(); count];
    for (id, name) in entries {
        names[id] = name;
    }
    for (id, name) in names.iter_mut().enumerate() {
        if name.is_empty() {
            *name = format!("class_{id}");
        }
    }
    Some(names)
}

/// Resolve a class id to a human readable name.
///
/// `class_count` is the number of classes the model declares (unknown for end-to-end
/// outputs); the built-in COCO table is only used when the metadata had no names and the
/// model either reports 80 classes or does not report a count at all.
fn class_name(names: &[String], class_count: Option<usize>, class_id: i32) -> String {
    if let Some(name) = usize::try_from(class_id)
        .ok()
        .and_then(|id| names.get(id))
    {
        return name.clone();
    }
    let coco = match class_count {
        Some(nc) => nc == COCO_CLASSES,
        None => true,
    };
    if names.is_empty() && coco {
        if let Some(name) = usize::try_from(class_id)
            .ok()
            .and_then(|id| COCO80.get(id))
        {
            return (*name).to_string();
        }
    }
    format!("class_{class_id}")
}

/// Convert a center/size box to corner coordinates.
pub fn xywh2xyxy(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let half_w = w / 2.0;
    let half_h = h / 2.0;
    (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
}

/// One candidate box while decoding, in network-input coordinates.
///
/// Public, with public fields, only so that the `bench_nms` example can build candidate sets and
/// compare the two NMS implementations; [`decode_and_nms`] is the single in-crate producer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: i32,
}

impl Candidate {
    /// Box shifted by the class offset, i.e. what NMS actually compares.
    fn suppression_box(&self) -> (f32, f32, f32, f32) {
        let shift = self.class_id as f32 * CLASS_OFFSET;
        (self.x1 + shift, self.y1 + shift, self.x2 + shift, self.y2 + shift)
    }
}

/// Greedy NMS. Candidates are returned in descending confidence order, capped at `max_det`.
///
/// Structure-of-arrays scan: the candidates are ordered once (stable, by descending confidence)
/// and then copied into five flat `f32` vectors — corners plus precomputed area — with the
/// class offset already added. The pair loop therefore touches contiguous memory and never
/// re-derives the suppression box or a box area, and a winner only compares against the
/// candidates *after* it in that order.
///
/// A fully branch-free variant (compute the whole IoU row into a reused buffer, then
/// `suppressed[j] |= row[j] > threshold`) was measured to be *slower* on realistic candidate
/// sets: it recomputes IoU for every candidate an earlier winner already suppressed, which is
/// most of them. What is kept from it is the branch-free flag write (`iou > threshold` as `u8`)
/// and the skip of suppressed candidates, which `examples/bench_nms.rs` measures.
///
/// Semantics are bit-for-bit identical to [`nms_reference`]: the arithmetic of [`iou`] is
/// reproduced in the same order, with the same `min`/`max` arguments, the same
/// `area_a + area_b - inter` union and the same `union <= 0` guard, on the same shifted
/// coordinates. The array-of-structs version stays the executable definition of those semantics.
pub fn nms(candidates: &[Candidate], iou_threshold: f32, max_det: usize) -> Vec<Candidate> {
    nms_indices(candidates, iou_threshold, max_det)
        .into_iter()
        .map(|index| candidates[index])
        .collect()
}

/// [`nms`], returning the *positions* of the kept candidates in descending confidence order.
///
/// This is the same scan as [`nms`] (which is a thin mapping over it); the pose decoders need it
/// because a candidate's keypoints live in a parallel array and must follow it through NMS.
pub fn nms_indices(candidates: &[Candidate], iou_threshold: f32, max_det: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| candidates[b].confidence.total_cmp(&candidates[a].confidence));
    nms_ordered_indices(candidates, &order, iou_threshold, max_det)
}

/// [`nms`] body, with the confidence ordering already computed.
fn nms_ordered_indices(
    candidates: &[Candidate],
    order: &[usize],
    iou_threshold: f32,
    max_det: usize,
) -> Vec<usize> {
    let count = order.len();
    let mut x1 = Vec::with_capacity(count);
    let mut y1 = Vec::with_capacity(count);
    let mut x2 = Vec::with_capacity(count);
    let mut y2 = Vec::with_capacity(count);
    let mut area = Vec::with_capacity(count);
    for &index in order {
        let candidate = candidates[index];
        let shift = candidate.class_id as f32 * CLASS_OFFSET;
        let (sx1, sy1, sx2, sy2) = (
            candidate.x1 + shift,
            candidate.y1 + shift,
            candidate.x2 + shift,
            candidate.y2 + shift,
        );
        x1.push(sx1);
        y1.push(sy1);
        x2.push(sx2);
        y2.push(sy2);
        area.push((sx2 - sx1).max(0.0) * (sy2 - sy1).max(0.0));
    }

    // Slice them down to `count` explicitly: that length assertion lets LLVM drop the bounds
    // checks inside the scan below.
    let (x1, y1, x2, y2, area) = (
        &x1[..count],
        &y1[..count],
        &x2[..count],
        &y2[..count],
        &area[..count],
    );

    let mut suppressed = vec![0u8; count];
    let mut kept: Vec<usize> = Vec::new();
    for rank in 0..count {
        if suppressed[rank] != 0 {
            continue;
        }
        kept.push(order[rank]);
        if kept.len() >= max_det {
            break;
        }
        let (ax1, ay1, ax2, ay2) = (x1[rank], y1[rank], x2[rank], y2[rank]);
        let winner_area = area[rank];
        for j in rank + 1..count {
            if suppressed[j] != 0 {
                continue;
            }
            let inter_w = (ax2.min(x2[j]) - ax1.max(x1[j])).max(0.0);
            let inter_h = (ay2.min(y2[j]) - ay1.max(y1[j])).max(0.0);
            let inter = inter_w * inter_h;
            let union = winner_area + area[j] - inter;
            let iou = if union <= 0.0 { 0.0 } else { inter / union };
            suppressed[j] = (iou > iou_threshold) as u8;
        }
    }
    kept
}

/// Reference implementation of [`nms`]: the straightforward array-of-structs greedy loop.
///
/// Kept as the executable definition of the NMS semantics (and as the benchmark baseline) —
/// [`nms`] must return exactly the same sequence, in the same order.
#[doc(hidden)]
pub fn nms_reference(
    candidates: &[Candidate],
    iou_threshold: f32,
    max_det: usize,
) -> Vec<Candidate> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| candidates[b].confidence.total_cmp(&candidates[a].confidence));

    let mut suppressed = vec![false; candidates.len()];
    let mut kept: Vec<Candidate> = Vec::new();
    for (rank, &i) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        let winner = candidates[i];
        kept.push(winner);
        if kept.len() >= max_det {
            break;
        }
        let winner_box = winner.suppression_box();
        for &j in &order[rank + 1..] {
            if suppressed[j] {
                continue;
            }
            if iou(winner_box, candidates[j].suppression_box()) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

/// Decode the end-to-end export: `1 x N x 6` rows of `x1 y1 x2 y2 confidence class_id`
/// with the NMS already done by the model and the rows sorted by descending confidence.
pub fn decode_end2end(
    raw: &[f32],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
) -> Result<Vec<Detection>> {
    if raw.len() % END2END_COLUMNS != 0 {
        return Err(Error::invalid(format!(
            "end2end output has {} values, not a multiple of {END2END_COLUMNS}",
            raw.len()
        )));
    }

    let mut detections = Vec::new();
    for row in raw.chunks_exact(END2END_COLUMNS) {
        if !(row[4] > conf_threshold) {
            continue;
        }
        if detections.len() >= MAX_DET {
            break;
        }
        let (x1, y1, x2, y2) = pre.to_source_box(row[0], row[1], row[2], row[3]);
        let class_id = row[5].round() as i32;
        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            confidence: row[4],
            class_id,
            class_name: class_name(class_names, None, class_id),
            keypoints: None,
            obb: None,
        });
    }
    Ok(detections)
}

/// Element type of a `detection_classes` tensor.
///
/// `torchvision` exports the class ids either as the integer tensor the `NonMaxSuppression` op
/// produces or as a float tensor some exporters derive from it; both denote the same index.
pub trait ClassIndex: Copy {
    /// Class id stored in the tensor.
    fn class_id(self) -> i32;
}

impl ClassIndex for f32 {
    fn class_id(self) -> i32 {
        self.round() as i32
    }
}

impl ClassIndex for i32 {
    fn class_id(self) -> i32 {
        self
    }
}

impl ClassIndex for i64 {
    fn class_id(self) -> i32 {
        self as i32
    }
}

/// Decode the four-tensor `torchvision` style export, where the graph already ran NMS.
///
/// `boxes` is `1 x N x 4` (`x1 y1 x2 y2` in network-input coordinates, i.e. the letterboxed
/// `640 x 640` canvas), `scores` and `classes` are `1 x N`, and `num_detections` is the number
/// of valid rows, so only the first `min(num_detections, N)` rows are looked at. Rows below
/// `conf_threshold` are dropped: the graph's NMS kept them for its own `max_det`, not for ours.
///
/// The ONNX `NonMaxSuppression` op walks batches and classes before scores, so the rows it
/// emits are *not* in global score order; the surviving detections are re-sorted by descending
/// confidence to match what every other layout returns.
pub fn decode_nms_applied<C: ClassIndex>(
    num_detections: i64,
    boxes: &[f32],
    scores: &[f32],
    classes: &[C],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
) -> Result<Vec<Detection>> {
    if boxes.len() % 4 != 0 {
        return Err(Error::invalid(format!(
            "detection_boxes holds {} values, not a multiple of 4",
            boxes.len()
        )));
    }
    let rows = boxes.len() / 4;
    if scores.len() < rows || classes.len() < rows {
        return Err(Error::invalid(format!(
            "NMS-applied outputs disagree: {rows} boxes, {} scores, {} classes",
            scores.len(),
            classes.len()
        )));
    }
    let count = if num_detections <= 0 {
        0
    } else {
        (num_detections as u64).min(rows as u64) as usize
    };

    let mut detections = Vec::new();
    for index in 0..count {
        let confidence = scores[index];
        if !(confidence > conf_threshold) {
            continue;
        }
        let (x1, y1, x2, y2) = pre.to_source_box(
            boxes[4 * index],
            boxes[4 * index + 1],
            boxes[4 * index + 2],
            boxes[4 * index + 3],
        );
        let class_id = classes[index].class_id();
        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            confidence,
            class_id,
            class_name: class_name(class_names, None, class_id),
            keypoints: None,
            obb: None,
        });
    }
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    Ok(detections)
}

/// Decode raw model output (`1 x 4+nc x 8400` for YOLOv8/v11-style exporters) into
/// detections expressed in source-image pixel coordinates, after confidence
/// thresholding and class-wise NMS.
pub fn decode_and_nms(
    raw: &[f32],
    shape: &[i64],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    let (channels, anchors) = match shape {
        [_, channels, anchors] if *channels >= 5 && *anchors > 0 => {
            (*channels as usize, *anchors as usize)
        }
        _ => {
            return Err(Error::invalid(format!(
                "unexpected output shape {shape:?} for the standard decoder"
            )))
        }
    };
    let class_count = channels - 4;
    if raw.len() < channels * anchors {
        return Err(Error::invalid(format!(
            "output holds {} values, need {} for shape {shape:?}",
            raw.len(),
            channels * anchors
        )));
    }

    // Channel-major layout: attribute row `c` occupies `raw[c * anchors ..]`, so one anchor
    // is `raw[c * anchors + a]` — transposing the tensor would need a full copy.
    let mut candidates: Vec<Candidate> = Vec::new();
    for anchor in 0..anchors {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_class = 0usize;
        for class in 0..class_count {
            let score = raw[(4 + class) * anchors + anchor];
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        if !(best_score > conf_threshold) {
            continue;
        }
        let (x1, y1, x2, y2) = xywh2xyxy(
            raw[anchor],
            raw[anchors + anchor],
            raw[2 * anchors + anchor],
            raw[3 * anchors + anchor],
        );
        candidates.push(Candidate {
            x1,
            y1,
            x2,
            y2,
            confidence: best_score,
            class_id: best_class as i32,
        });
    }

    let kept = nms(&candidates, iou_threshold, MAX_DET);
    Ok(kept
        .into_iter()
        .map(|c| {
            let (x1, y1, x2, y2) = pre.to_source_box(c.x1, c.y1, c.x2, c.y2);
            Detection {
                x1,
                y1,
                x2,
                y2,
                confidence: c.confidence,
                class_id: c.class_id,
                class_name: class_name(class_names, Some(class_count), c.class_id),
                keypoints: None,
                obb: None,
            }
        })
        .collect())
}

/// Decode one run of `K*D` network-space keypoint values into source-image keypoints.
///
/// `values` is `[x0, y0(, v0), x1, y1(, v1), ...]`; the visibility is already sigmoid'd by the
/// exporter and is passed through unchanged. `x`/`y` get the same de-padding, gain division and
/// clipping as [`Preprocessed::to_source_box`].
fn keypoints_from_network(pre: &Preprocessed, kpt: KptShape, values: &[f32]) -> Vec<Keypoint> {
    values
        .chunks_exact(kpt.dim)
        .map(|point| {
            let (x, y) = pre.to_source_point(point[0], point[1]);
            Keypoint {
                x,
                y,
                visibility: if kpt.dim >= 3 { Some(point[2]) } else { None },
            }
        })
        .collect()
}

/// Decode the standard pose export: `1 x (4 + nc + K*D) x anchors`.
///
/// The first `4` channels are the box in `cx cy w h` form, the next `nc` ones the class scores
/// (the best one is the confidence, exactly like [`decode_and_nms`]) and the remaining `K*D` the
/// keypoints: `x`, `y` in network-canvas pixels and, for `D == 3`, the sigmoid'd visibility.
///
/// Keypoints are mapped to source-image pixels and clipped *before* NMS, and they follow their
/// candidate through the suppression by index, so a kept box always carries its own keypoints.
pub fn decode_pose(
    raw: &[f32],
    shape: &[i64],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    kpt: KptShape,
) -> Result<Vec<Detection>> {
    let (channels, anchors) = match shape {
        [_, channels, anchors] if *anchors > 0 => (*channels as usize, *anchors as usize),
        _ => {
            return Err(Error::invalid(format!(
                "unexpected output shape {shape:?} for the pose decoder"
            )))
        }
    };
    let kpt_channels = kpt.channels();
    if kpt.count == 0 || kpt.dim < 2 {
        return Err(Error::invalid(format!(
            "unusable keypoint shape: {} keypoints of dim {}",
            kpt.count, kpt.dim
        )));
    }
    if channels < 5 + kpt_channels {
        return Err(Error::invalid(format!(
            "pose output has {channels} channels, need at least {} for {} keypoints of dim {}",
            5 + kpt_channels,
            kpt.count,
            kpt.dim
        )));
    }
    let class_count = channels - 4 - kpt_channels;
    if raw.len() < channels * anchors {
        return Err(Error::invalid(format!(
            "output holds {} values, need {} for shape {shape:?}",
            raw.len(),
            channels * anchors
        )));
    }

    // Channel-major layout, as in `decode_and_nms`: attribute row `c` starts at `c * anchors`.
    // The keypoints of candidate `i` occupy `points[i * K .. (i + 1) * K]`.
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut points: Vec<Keypoint> = Vec::new();
    for anchor in 0..anchors {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_class = 0usize;
        for class in 0..class_count {
            let score = raw[(4 + class) * anchors + anchor];
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        if !(best_score > conf_threshold) {
            continue;
        }
        let (x1, y1, x2, y2) = xywh2xyxy(
            raw[anchor],
            raw[anchors + anchor],
            raw[2 * anchors + anchor],
            raw[3 * anchors + anchor],
        );
        candidates.push(Candidate {
            x1,
            y1,
            x2,
            y2,
            confidence: best_score,
            class_id: best_class as i32,
        });
        for point in 0..kpt.count {
            let base = (4 + class_count + point * kpt.dim) * anchors + anchor;
            let (x, y) = pre.to_source_point(raw[base], raw[base + anchors]);
            points.push(Keypoint {
                x,
                y,
                visibility: if kpt.dim >= 3 {
                    Some(raw[base + 2 * anchors])
                } else {
                    None
                },
            });
        }
    }

    let kept = nms_indices(&candidates, iou_threshold, MAX_DET);
    Ok(kept
        .into_iter()
        .map(|index| {
            let candidate = candidates[index];
            let (x1, y1, x2, y2) =
                pre.to_source_box(candidate.x1, candidate.y1, candidate.x2, candidate.y2);
            Detection {
                x1,
                y1,
                x2,
                y2,
                confidence: candidate.confidence,
                class_id: candidate.class_id,
                class_name: class_name(class_names, Some(class_count), candidate.class_id),
                keypoints: Some(points[index * kpt.count..(index + 1) * kpt.count].to_vec()),
                obb: None,
            }
        })
        .collect())
}

/// Decode the end-to-end pose export: `1 x N x (6 + K*D)` rows of
/// `x1 y1 x2 y2 confidence class_id keypoints`.
///
/// The graph already ran NMS and sorted the rows by descending confidence, so this only filters
/// by confidence and caps at [`MAX_DET`] — the same contract as [`decode_end2end`], with the
/// keypoints appended to each row (ultralytics inserts the `cls` column inside NMS, i.e. before
/// the keypoints).
pub fn decode_pose_end2end(
    raw: &[f32],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    kpt: KptShape,
) -> Result<Vec<Detection>> {
    if kpt.count == 0 || kpt.dim < 2 {
        return Err(Error::invalid(format!(
            "unusable keypoint shape: {} keypoints of dim {}",
            kpt.count, kpt.dim
        )));
    }
    let row_len = END2END_COLUMNS + kpt.channels();
    if raw.len() % row_len != 0 {
        return Err(Error::invalid(format!(
            "end2end pose output has {} values, not a multiple of {row_len} (6 + {}*{})",
            raw.len(),
            kpt.count,
            kpt.dim
        )));
    }

    let mut detections = Vec::new();
    for row in raw.chunks_exact(row_len) {
        if !(row[4] > conf_threshold) {
            continue;
        }
        if detections.len() >= MAX_DET {
            break;
        }
        let (x1, y1, x2, y2) = pre.to_source_box(row[0], row[1], row[2], row[3]);
        let class_id = row[5].round() as i32;
        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            confidence: row[4],
            class_id,
            class_name: class_name(class_names, None, class_id),
            keypoints: Some(keypoints_from_network(pre, kpt, &row[END2END_COLUMNS..])),
            obb: None,
        });
    }
    Ok(detections)
}

/// Column count of an end-to-end OBB (`NMS already applied`) row: `cx cy w h conf cls angle`.
///
/// Unlike the detect end-to-end layout the box stays in center/size form: the `dist2rbox` head
/// decode always produces `xywh`, and neither the graph nor the predictor converts it.
const OBB_END2END_COLUMNS: usize = 7;

/// One rotated-box candidate while decoding, in network-input coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObbCandidate {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    /// Rotation in radians, as it comes out of the graph (see [`decode_obb`]).
    pub angle: f32,
    pub confidence: f32,
    pub class_id: i32,
}

impl ObbCandidate {
    /// Box with the class offset added to the center, i.e. what rotated NMS actually compares.
    fn suppression_box(&self) -> (f32, f32, f32, f32, f32) {
        let shift = self.class_id as f32 * CLASS_OFFSET;
        (self.cx + shift, self.cy + shift, self.w, self.h, self.angle)
    }
}

/// Probabilistic IoU of two `(cx, cy, w, h, angle)` rotated boxes (ultralytics `batch_probiou`).
///
/// Models each box as a Gaussian with covariance `[[w²/12, 0], [0, h²/12]]` rotated by `angle`
/// and derives the IoU from the Bhattacharyya distance; the result is `0` for disjoint boxes
/// and approaches `1` for identical ones. Angle in radians.
pub fn probiou(a: (f32, f32, f32, f32, f32), b: (f32, f32, f32, f32, f32)) -> f32 {
    const EPS: f32 = 1e-7;
    // `_get_covariance_matrix`: [[a, c], [c, b]] of the rotated Gaussian.
    let covariance = |r: (f32, f32, f32, f32, f32)| {
        let var_w = r.2 * r.2 / 12.0;
        let var_h = r.3 * r.3 / 12.0;
        let (sin, cos) = r.4.sin_cos();
        (
            var_w * cos * cos + var_h * sin * sin,
            var_w * sin * sin + var_h * cos * cos,
            (var_w - var_h) * sin * cos,
        )
    };
    let (a1, b1, c1) = covariance(a);
    let (a2, b2, c2) = covariance(b);
    let (x1, y1) = (a.0, a.1);
    let (x2, y2) = (b.0, b.1);
    let denom = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2) + EPS;
    let t1 = ((a1 + a2) * (y1 - y2) * (y1 - y2) + (b1 + b2) * (x1 - x2) * (x1 - x2)) / denom * 0.25;
    let t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5;
    let det1 = (a1 * b1 - c1 * c1).max(0.0);
    let det2 = (a2 * b2 - c2 * c2).max(0.0);
    let t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2))
        / (4.0 * (det1 * det2).sqrt() + EPS)
        + EPS)
        .ln()
        * 0.5;
    let bd = (t1 + t2 + t3).clamp(EPS, 100.0);
    let hd = (1.0 - (-bd).exp() + EPS).sqrt();
    1.0 - hd
}

/// Rotated NMS with the semantics of ultralytics' `TorchNMS.fast_nms` (the rotated branch of
/// `non_max_suppression` uses it with [`probiou`]).
///
/// Candidates are sorted by descending confidence; a candidate is suppressed when *any*
/// higher-ranked candidate — kept or not — overlaps it with probIoU above `iou_threshold`.
/// Unlike greedy [`nms`], suppressed boxes keep suppressing. Returns the positions of the kept
/// candidates in descending confidence order, capped at `max_det` (mirrors `i[:max_det]`).
pub fn nms_rotated(candidates: &[ObbCandidate], iou_threshold: f32, max_det: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| candidates[b].confidence.total_cmp(&candidates[a].confidence));
    let shifted: Vec<(f32, f32, f32, f32, f32)> =
        order.iter().map(|&index| candidates[index].suppression_box()).collect();

    let mut suppressed = vec![false; order.len()];
    for j in 1..order.len() {
        for i in 0..j {
            if probiou(shifted[i], shifted[j]) > iou_threshold {
                suppressed[j] = true;
                break;
            }
        }
    }
    let mut kept = Vec::new();
    for (rank, index) in order.into_iter().enumerate() {
        if suppressed[rank] {
            continue;
        }
        kept.push(index);
        if kept.len() >= max_det {
            break;
        }
    }
    kept
}

/// Map one decoded OBB candidate to a [`Detection`] in source-image coordinates.
///
/// `cx`/`cy`/`w`/`h` go through [`Preprocessed::to_source_xywh`] (padding removed, per-axis
/// gains, no clipping — the `xywh` branch of ultralytics' `scale_boxes`), the angle passes
/// through unchanged. `x1..y2` is the minimal axis-aligned box enclosing the rotated one.
fn obb_detection(
    pre: &Preprocessed,
    class_names: &[String],
    class_count: Option<usize>,
    candidate: ObbCandidate,
) -> Detection {
    let (cx, cy, w, h) = pre.to_source_xywh(candidate.cx, candidate.cy, candidate.w, candidate.h);
    let obb = Obb {
        cx,
        cy,
        w,
        h,
        angle: candidate.angle,
    };
    let (x1, y1, x2, y2) = obb.enclosing_box();
    Detection {
        x1,
        y1,
        x2,
        y2,
        confidence: candidate.confidence,
        class_id: candidate.class_id,
        class_name: class_name(class_names, class_count, candidate.class_id),
        keypoints: None,
        obb: Some(obb),
    }
}

/// Decode the standard OBB export: `1 x (4 + nc + 1) x anchors`.
///
/// The first `4` channels are the box in `cx cy w h` form, the next `nc` the class scores (the
/// best one is the confidence, exactly like [`decode_and_nms`]) and the last one the rotation in
/// radians. The angle needs no further decoding: v8/v11 heads bake `(sigmoid(a) - 0.25) * pi`
/// into the exported graph, and yolo26 (`OBB26`) heads regress radians directly (their angle
/// loss supervises the raw value), so every export emits radians.
///
/// NMS is [`nms_rotated`] on the class-shifted centers, mirroring the `rotated=True` branch of
/// ultralytics' `non_max_suppression`.
pub fn decode_obb(
    raw: &[f32],
    shape: &[i64],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    let (channels, anchors) = match shape {
        [_, channels, anchors] if *channels >= 6 && *anchors > 0 => {
            (*channels as usize, *anchors as usize)
        }
        _ => {
            return Err(Error::invalid(format!(
                "unexpected output shape {shape:?} for the OBB decoder"
            )))
        }
    };
    let class_count = channels - 5;
    if raw.len() < channels * anchors {
        return Err(Error::invalid(format!(
            "output holds {} values, need {} for shape {shape:?}",
            raw.len(),
            channels * anchors
        )));
    }

    // Channel-major layout, as in `decode_and_nms`: attribute row `c` starts at `c * anchors`.
    let mut candidates: Vec<ObbCandidate> = Vec::new();
    for anchor in 0..anchors {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_class = 0usize;
        for class in 0..class_count {
            let score = raw[(4 + class) * anchors + anchor];
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        if !(best_score > conf_threshold) {
            continue;
        }
        candidates.push(ObbCandidate {
            cx: raw[anchor],
            cy: raw[anchors + anchor],
            w: raw[2 * anchors + anchor],
            h: raw[3 * anchors + anchor],
            angle: raw[(4 + class_count) * anchors + anchor],
            confidence: best_score,
            class_id: best_class as i32,
        });
    }

    let kept = nms_rotated(&candidates, iou_threshold, MAX_DET);
    Ok(kept
        .into_iter()
        .map(|index| obb_detection(pre, class_names, Some(class_count), candidates[index]))
        .collect())
}

/// Decode the end-to-end OBB export: `1 x N x 7` rows of `cx cy w h confidence class_id angle`.
///
/// The graph already ran (rotated) NMS and picked the top rows, so this only filters by
/// confidence and caps at [`MAX_DET`] — the same contract as [`decode_end2end`], except the box
/// stays in `xywh` form (`dist2rbox` never produces corners).
pub fn decode_obb_end2end(
    raw: &[f32],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
) -> Result<Vec<Detection>> {
    if raw.len() % OBB_END2END_COLUMNS != 0 {
        return Err(Error::invalid(format!(
            "end2end OBB output has {} values, not a multiple of {OBB_END2END_COLUMNS}",
            raw.len()
        )));
    }

    let mut detections = Vec::new();
    for row in raw.chunks_exact(OBB_END2END_COLUMNS) {
        if !(row[4] > conf_threshold) {
            continue;
        }
        if detections.len() >= MAX_DET {
            break;
        }
        let class_id = row[5].round() as i32;
        detections.push(obb_detection(
            pre,
            class_names,
            None,
            ObbCandidate {
                cx: row[0],
                cy: row[1],
                w: row[2],
                h: row[3],
                angle: row[6],
                confidence: row[4],
                class_id,
            },
        ));
    }
    Ok(detections)
}

/// Decode whichever OBB layout the model produced.
///
/// A last dimension of [`OBB_END2END_COLUMNS`] means the end-to-end export (mirroring the
/// last-dimension-`6` rule of [`decode_output`]); anything else 3-D is the standard
/// channel-major export. A standard OBB tensor whose anchor count happens to equal `7` is read
/// as end-to-end — that would need 7 anchors, which no real export has.
pub fn decode_obb_output(
    raw: &[f32],
    shape: &[i64],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    match shape {
        [_, _, last] if *last == OBB_END2END_COLUMNS as i64 => {
            decode_obb_end2end(raw, pre, class_names, conf_threshold)
        }
        [_, _, _] => decode_obb(raw, shape, pre, class_names, conf_threshold, iou_threshold),
        _ => Err(Error::invalid(format!(
            "unsupported output shape {shape:?}; expected a 3-D (batch, channels, anchors) tensor"
        ))),
    }
}

/// Keypoint layout implied by the output shape alone, for models whose metadata carries no
/// `kpt_shape`.
///
/// Only the two COCO person layouts are recognised: the standard `(1, 56, A)`
/// (`4 + 1 + 17*3`) and the end-to-end `(1, N, 57)` (`6 + 17*3`). Any other pose export needs
/// the metadata — guessing `K`/`D` from a shape would decode noise.
fn infer_kpt_shape(shape: &[i64]) -> Option<KptShape> {
    match shape {
        [_, channels, _] if *channels == (4 + 1 + KptShape::COCO.channels()) as i64 => {
            Some(KptShape::COCO)
        }
        [_, _, last] if *last == (END2END_COLUMNS + KptShape::COCO.channels()) as i64 => {
            Some(KptShape::COCO)
        }
        _ => None,
    }
}

/// Decode whichever detect layout the model produced.
///
/// [`decode_output_with_kpt`] is the same dispatch with pose support; this entry point is kept
/// for detect-only callers (and stays the definition of the detect routing).
pub fn decode_output(
    raw: &[f32],
    shape: &[i64],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    match shape {
        // (1, N, 6): the exporter already ran NMS, the last dimension identifies it.
        [_, _, last] if *last == END2END_COLUMNS as i64 => {
            decode_end2end(raw, pre, class_names, conf_threshold)
        }
        [_, _, _] => decode_and_nms(
            raw,
            shape,
            pre,
            class_names,
            conf_threshold,
            iou_threshold,
        ),
        _ => Err(Error::invalid(format!(
            "unsupported output shape {shape:?}; expected a 3-D (batch, channels, anchors) tensor"
        ))),
    }
}

/// Decode whichever layout the model produced, pose models included.
///
/// `kpt` is the model's `kpt_shape` metadata ([`KptShape::parse`]); `None` means "not declared"
/// and falls back to [`infer_kpt_shape`]. Routing, in order:
///
/// * `kpt` known and the last dimension is `6 + K*D` → end-to-end pose
///   ([`decode_pose_end2end`]);
/// * `kpt` known otherwise → standard pose ([`decode_pose`]);
/// * `kpt` unknown → the detect routing of [`decode_output`] (last dimension `6` → end-to-end,
///   anything else 3-D → standard).
///
/// A standard pose tensor whose anchor count happens to equal `6 + K*D` is read as end-to-end —
/// that would need 8..=57 anchors, which no real export has.
pub fn decode_output_with_kpt(
    raw: &[f32],
    shape: &[i64],
    pre: &Preprocessed,
    class_names: &[String],
    conf_threshold: f32,
    iou_threshold: f32,
    kpt: Option<KptShape>,
) -> Result<Vec<Detection>> {
    let kpt = kpt.or_else(|| infer_kpt_shape(shape));
    let last = match shape {
        [_, _, last] => *last,
        _ => {
            return Err(Error::invalid(format!(
                "unsupported output shape {shape:?}; expected a 3-D (batch, channels, anchors) tensor"
            )))
        }
    };
    match kpt {
        Some(kpt) if last == (END2END_COLUMNS + kpt.channels()) as i64 => {
            decode_pose_end2end(raw, pre, class_names, conf_threshold, kpt)
        }
        Some(kpt) => decode_pose(
            raw,
            shape,
            pre,
            class_names,
            conf_threshold,
            iou_threshold,
            kpt,
        ),
        None => decode_output(raw, shape, pre, class_names, conf_threshold, iou_threshold),
    }
}

/// Intersection over union of two `(x1, y1, x2, y2)` boxes.
pub fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let inter_w = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let inter_h = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let area_b = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity geometry: source and network space are the same 640x640 canvas.
    fn identity_pre() -> Preprocessed {
        Preprocessed {
            tensor: Vec::new(),
            input_width: 640,
            input_height: 640,
            src_width: 640,
            src_height: 640,
            gain_x: 1.0,
            gain_y: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    #[test]
    fn iou_basics() {
        let a = (0.0, 0.0, 10.0, 10.0);
        assert!((iou(a, a) - 1.0).abs() < 1e-6);
        assert_eq!(iou(a, (20.0, 20.0, 30.0, 30.0)), 0.0);
        // half-overlapping boxes: 50 / (100 + 100 - 50)
        let b = (5.0, 0.0, 15.0, 10.0);
        assert!((iou(a, b) - 50.0 / 150.0).abs() < 1e-6);
    }

    #[test]
    fn xywh_to_xyxy() {
        assert_eq!(xywh2xyxy(50.0, 60.0, 20.0, 40.0), (40.0, 40.0, 60.0, 80.0));
    }

    #[test]
    fn nms_suppresses_same_class_overlaps_in_confidence_order() {
        let pre = identity_pre();
        // 2 classes x 4 anchors, channel-major (4 + nc) x anchors.
        let anchors = 4usize;
        let channels = 6usize;
        let mut raw = vec![0f32; channels * anchors];
        let boxes = [
            // (cx, cy, w, h, conf, class)
            (100.0f32, 100.0, 40.0, 40.0, 0.90, 0usize),
            (105.0, 100.0, 40.0, 40.0, 0.80, 0), // overlaps #0, same class -> suppressed
            (400.0, 400.0, 40.0, 40.0, 0.70, 0),
            (405.0, 400.0, 40.0, 40.0, 0.60, 0), // overlaps #2, same class -> suppressed
        ];
        for (anchor, &(cx, cy, w, h, conf, class)) in boxes.iter().enumerate() {
            raw[anchor] = cx;
            raw[anchors + anchor] = cy;
            raw[2 * anchors + anchor] = w;
            raw[3 * anchors + anchor] = h;
            raw[(4 + class) * anchors + anchor] = conf;
        }

        let dets =
            decode_and_nms(&raw, &[1, channels as i64, anchors as i64], &pre, &[], 0.25, 0.7)
                .unwrap();
        assert_eq!(dets.len(), 2, "{dets:?}");
        assert!((dets[0].confidence - 0.90).abs() < 1e-6);
        assert!((dets[1].confidence - 0.70).abs() < 1e-6);
        assert!(dets[0].confidence > dets[1].confidence);
        // confidence 0.60 candidate stays below threshold? no: it is filtered by NMS, and
        // it must not appear even though it passed the confidence threshold.
        assert!(dets.iter().all(|d| (d.confidence - 0.60).abs() > 1e-6));
        assert!(dets[0].x1 < dets[1].x1);
    }

    #[test]
    fn nms_keeps_overlaps_of_different_classes() {
        let pre = identity_pre();
        let anchors = 2usize;
        let channels = 6usize;
        let mut raw = vec![0f32; channels * anchors];
        // same box, different classes (class ids are shifted by 7680 before NMS)
        for (anchor, class) in [(0usize, 0usize), (1, 1)] {
            raw[anchor] = 200.0;
            raw[anchors + anchor] = 200.0;
            raw[2 * anchors + anchor] = 40.0;
            raw[3 * anchors + anchor] = 40.0;
            raw[(4 + class) * anchors + anchor] = 0.5;
        }
        let dets =
            decode_and_nms(&raw, &[1, channels as i64, anchors as i64], &pre, &[], 0.25, 0.7)
                .unwrap();
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].class_name, "class_0");
        assert_eq!(dets[1].class_name, "class_1");
    }

    #[test]
    fn end2end_filters_and_preserves_order() {
        let pre = identity_pre();
        let mut raw = vec![0f32; 300 * END2END_COLUMNS];
        let rows = [
            (0usize, 10.0f32, 20.0, 30.0, 40.0, 0.9, 0.0),
            (1, 100.0, 200.0, 130.0, 240.0, 0.1, 5.0), // below threshold
            (2, 50.0, 60.0, 70.0, 80.0, 0.5, 2.0),
            (3, 5.0, 6.0, 7.0, 8.0, 0.3, 7.0),
        ];
        for (row, x1, y1, x2, y2, conf, class) in rows {
            let base = row * END2END_COLUMNS;
            raw[base] = x1;
            raw[base + 1] = y1;
            raw[base + 2] = x2;
            raw[base + 3] = y2;
            raw[base + 4] = conf;
            raw[base + 5] = class;
        }

        let dets = decode_end2end(&raw, &pre, &[], 0.25).unwrap();
        assert_eq!(dets.len(), 3, "{dets:?}");
        assert!((dets[0].confidence - 0.9).abs() < 1e-6);
        assert!((dets[1].confidence - 0.5).abs() < 1e-6);
        assert!((dets[2].confidence - 0.3).abs() < 1e-6);
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[1].class_id, 2);
        assert_eq!(dets[2].class_id, 7);
        // no `names` metadata at all: id < 80 falls back to the built-in COCO table
        assert_eq!(dets[0].class_name, "person");
        assert_eq!(dets[1].class_name, "car");
        assert_eq!(dets[2].class_name, "truck");
        assert_eq!((dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2), (10.0, 20.0, 30.0, 40.0));
    }

    #[test]
    fn parse_names_handles_python_repr() {
        let names = parse_names("{0: 'person', 1: 'bicycle', 3: 'car'}").unwrap();
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "person");
        assert_eq!(names[2], "class_2");
        assert_eq!(names[3], "car");
    }

    #[test]
    fn parse_names_rejects_garbage() {
        assert!(parse_names("").is_none());
        assert!(parse_names("{}").is_none());
        assert!(parse_names("person,bicycle").is_none());
    }

    #[test]
    fn decode_output_routes_by_shape() {
        let pre = identity_pre();
        let end2end = vec![10.0f32, 20.0, 30.0, 40.0, 0.9, 1.0];
        let dets = decode_output(&end2end, &[1, 1, 6], &pre, &[], 0.25, 0.7).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);

        let mut standard = vec![0f32; 6 * 1];
        standard[0] = 100.0;
        standard[1] = 100.0;
        standard[2] = 20.0;
        standard[3] = 20.0;
        standard[4] = 0.9;
        let dets = decode_output(&standard, &[1, 6, 1], &pre, &[], 0.25, 0.7).unwrap();
        assert_eq!(dets.len(), 1);

        assert!(decode_output(&[], &[1, 2], &pre, &[], 0.25, 0.7).is_err());
    }

    /// Deterministic `xorshift64*` PRNG, so the randomized test stays dependency free.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }

        /// Uniform in `[0, 1)`.
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
        }

        fn below(&mut self, bound: usize) -> usize {
            self.next_u32() as usize % bound
        }
    }

    /// Adversarial candidate set: a few dense clusters (heavy same-class overlap), boxes whose
    /// geometry is copied from another candidate (cross-class and same-class duplicates),
    /// quantised confidences (many exact ties for the stable sort), degenerate zero/negative area
    /// boxes and negative coordinates.
    fn synthetic_candidates(seed: u64, count: usize) -> Vec<Candidate> {
        let mut rng = Rng::new(seed);
        let clusters: Vec<(f32, f32)> = (0..4)
            .map(|_| (rng.next_f32() * 800.0 - 100.0, rng.next_f32() * 800.0 - 100.0))
            .collect();
        let mut candidates: Vec<Candidate> = Vec::with_capacity(count);
        while candidates.len() < count {
            let (x1, y1, mut x2, mut y2) = if rng.below(4) == 0 && !candidates.is_empty() {
                let same = candidates[rng.below(candidates.len())];
                (same.x1, same.y1, same.x2, same.y2)
            } else {
                let (cx, cy) = clusters[rng.below(clusters.len())];
                let x1 = cx + rng.next_f32() * 40.0 - 20.0;
                let y1 = cy + rng.next_f32() * 40.0 - 20.0;
                (x1, y1, x1 + rng.next_f32() * 120.0, y1 + rng.next_f32() * 120.0)
            };
            match rng.below(16) {
                0 => x2 = x1,       // zero area: vertical line
                1 => y2 = y1,       // zero area: horizontal line
                2 => x2 = x1 - 5.0, // negative area
                _ => {}
            }
            let confidence = rng.below(8) as f32 / 8.0 + 0.125;
            let class_id = rng.below(3) as i32;
            candidates.push(Candidate {
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id,
            });
        }
        candidates
    }

    #[test]
    fn nms_matches_reference_on_synthetic_data() {
        assert_eq!(nms(&[], 0.7, MAX_DET), Vec::new());

        let mut suppressed_somewhere = false;
        for seed in 0..6u64 {
            for count in [50usize, 500] {
                let candidates = synthetic_candidates(seed, count);
                for iou_threshold in [0.05f32, 0.45, 0.7, 1.0] {
                    for max_det in [0usize, 17, MAX_DET] {
                        let fast = nms(&candidates, iou_threshold, max_det);
                        let reference = nms_reference(&candidates, iou_threshold, max_det);
                        assert_eq!(
                            fast, reference,
                            "seed {seed}, count {count}, iou {iou_threshold}, max_det {max_det}"
                        );
                        assert!(
                            fast.windows(2).all(|pair| pair[0].confidence >= pair[1].confidence),
                            "kept candidates must stay in descending confidence order"
                        );
                        suppressed_somewhere |= fast.len() < candidates.len();
                    }
                }
            }
        }
        // Guard against a vacuous test: some of those candidate sets must actually be pruned.
        assert!(suppressed_somewhere);
    }

    /// Letterboxed `640 x 640` canvas mapped back onto a `320 x 240` source (gain 0.5 / 0.375).
    fn letterboxed_pre() -> Preprocessed {
        Preprocessed {
            tensor: Vec::new(),
            input_width: 640,
            input_height: 640,
            src_width: 320,
            src_height: 240,
            gain_x: 0.5,
            gain_y: 0.375,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    /// Four-output payload: rows are grouped by class (not by score), one row is below the
    /// confidence threshold, and the last two rows are padding the graph never counted.
    fn nms_applied_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let boxes = vec![
            100.0f32, 200.0, 300.0, 400.0, // row 0, class 3, conf 0.60
            10.0, 20.0, 30.0, 40.0, // row 1, class 1, conf 0.90
            400.0, 400.0, 480.0, 480.0, // row 2, class 2, conf 0.30
            50.0, 60.0, 70.0, 80.0, // row 3, class 0, conf 0.10 (below threshold)
            0.0, 0.0, 1.0, 1.0, // padding
            0.0, 0.0, 1.0, 1.0, // padding
        ];
        let scores = vec![0.60f32, 0.90, 0.30, 0.10, 0.0, 0.0];
        let classes = vec![3.0f32, 1.0, 2.0, 0.0, 0.0, 0.0];
        (boxes, scores, classes)
    }

    #[test]
    fn decode_nms_applied_filters_sorts_and_maps_back() {
        let pre = letterboxed_pre();
        let (boxes, scores, classes) = nms_applied_inputs();

        let detections =
            decode_nms_applied(4, &boxes, &scores, &classes, &pre, &[], 0.25).unwrap();
        assert_eq!(detections.len(), 3, "{detections:?}");
        // re-sorted by confidence although the rows came class-grouped
        let confidences: Vec<f32> = detections.iter().map(|d| d.confidence).collect();
        assert_eq!(confidences, vec![0.90, 0.60, 0.30]);
        assert_eq!(
            detections.iter().map(|d| d.class_id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
        // no `names` metadata: ids < 80 fall back to the built-in COCO table
        assert_eq!(detections[0].class_name, "bicycle");
        assert_eq!(detections[1].class_name, "motorcycle");
        assert_eq!(detections[2].class_name, "car");
        // network canvas -> source pixels
        let first = &detections[0];
        assert!((first.x1 - 20.0).abs() < 1e-4, "{first:?}");
        assert!((first.y1 - 20.0 / 0.375).abs() < 1e-4, "{first:?}");
        assert!((first.x2 - 60.0).abs() < 1e-4, "{first:?}");
        assert!((first.y2 - 40.0 / 0.375).abs() < 1e-4, "{first:?}");

        // `num_detections` truncates the rows: only row 0 and 1 are valid, then 0.90 sorts first
        let truncated =
            decode_nms_applied(2, &boxes, &scores, &classes, &pre, &[], 0.25).unwrap();
        assert_eq!(truncated.len(), 2);
        assert_eq!(truncated[0].confidence, 0.90);
        assert_eq!(truncated[1].confidence, 0.60);

        // nothing counted, and a count above the row count
        assert!(decode_nms_applied(0, &boxes, &scores, &classes, &pre, &[], 0.25)
            .unwrap()
            .is_empty());
        assert!(decode_nms_applied(-3, &boxes, &scores, &classes, &pre, &[], 0.25)
            .unwrap()
            .is_empty());
        assert_eq!(
            decode_nms_applied(9_999, &boxes, &scores, &classes, &pre, &[], 0.25)
                .unwrap()
                .len(),
            3
        );

        // a confidence exactly on the threshold is rejected (`score > conf_threshold`)
        let on_threshold =
            decode_nms_applied(4, &boxes, &scores, &classes, &pre, &[], 0.60).unwrap();
        assert_eq!(on_threshold.len(), 1);
    }

    #[test]
    fn decode_nms_applied_accepts_integer_classes() {
        let pre = letterboxed_pre();
        let (boxes, scores, classes_f32) = nms_applied_inputs();
        let classes_i32: Vec<i32> = classes_f32.iter().map(|&c| c as i32).collect();
        let classes_i64: Vec<i64> = classes_f32.iter().map(|&c| c as i64).collect();

        let from_f32 =
            decode_nms_applied(4, &boxes, &scores, &classes_f32, &pre, &[], 0.25).unwrap();
        let from_i32 =
            decode_nms_applied(4, &boxes, &scores, &classes_i32, &pre, &[], 0.25).unwrap();
        let from_i64 =
            decode_nms_applied(4, &boxes, &scores, &classes_i64, &pre, &[], 0.25).unwrap();
        assert_eq!(from_f32, from_i32);
        assert_eq!(from_f32, from_i64);
    }

    #[test]
    fn decode_nms_applied_rejects_malformed_tensors() {
        let pre = letterboxed_pre();
        // boxes not a multiple of 4
        assert!(decode_nms_applied::<f32>(1, &[0.0, 1.0], &[], &[], &pre, &[], 0.25).is_err());
        // fewer scores / classes than boxes
        let boxes = [0.0f32, 0.0, 10.0, 10.0];
        assert!(decode_nms_applied::<f32>(1, &boxes, &[], &[], &pre, &[], 0.25).is_err());
        assert!(decode_nms_applied::<f32>(1, &boxes, &[0.9], &[], &pre, &[], 0.25).is_err());
        // ... and the well-formed case works
        assert_eq!(
            decode_nms_applied(1, &boxes, &[0.9], &[1i32], &pre, &[], 0.25)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn kpt_shape_parses_python_reprs() {
        assert_eq!(KptShape::parse("(17, 3)"), Some(KptShape::COCO));
        assert_eq!(KptShape::parse("(17,3)"), Some(KptShape::COCO));
        assert_eq!(KptShape::parse("[17,3]"), Some(KptShape::COCO));
        assert_eq!(KptShape::parse(" 17 , 3 "), Some(KptShape::COCO));
        assert_eq!(
            KptShape::parse("(4, 2)"),
            Some(KptShape {
                count: 4,
                dim: 2
            })
        );
        assert_eq!(KptShape::COCO.channels(), 51);

        // anything that is not exactly two usable numbers is rejected
        assert_eq!(KptShape::parse(""), None);
        assert_eq!(KptShape::parse("(17)"), None);
        assert_eq!(KptShape::parse("person"), None);
        assert_eq!(KptShape::parse("(0, 3)"), None);
        assert_eq!(KptShape::parse("(17, 1)"), None);
        assert_eq!(KptShape::parse("(17, 4)"), None);
    }

    #[test]
    fn nms_indices_match_nms_output() {
        for seed in 0..3u64 {
            let candidates = synthetic_candidates(seed, 200);
            for max_det in [5usize, MAX_DET] {
                let kept = nms(&candidates, 0.45, max_det);
                let indices = nms_indices(&candidates, 0.45, max_det);
                let mapped: Vec<Candidate> =
                    indices.iter().map(|&index| candidates[index]).collect();
                assert_eq!(mapped, kept, "seed {seed}, max_det {max_det}");
            }
        }
        assert!(nms_indices(&[], 0.7, MAX_DET).is_empty());
    }

    /// `2` keypoints of `3` values each, `1` class: `1 x 11 x anchors`.
    fn pose_standard_inputs() -> (Vec<f32>, usize, usize) {
        let anchors = 3usize;
        let kpt = KptShape { count: 2, dim: 3 };
        let channels = 4 + 1 + kpt.channels();
        let mut raw = vec![0f32; channels * anchors];
        let mut write = |anchor: usize, cx: f32, cy: f32, conf: f32, points: [(f32, f32, f32); 2]| {
            let rows = [cx, cy, 40.0, 40.0];
            for (row, value) in rows.iter().enumerate() {
                raw[row * anchors + anchor] = *value;
            }
            raw[4 * anchors + anchor] = conf;
            for (point, (x, y, v)) in points.iter().enumerate() {
                let base = (5 + point * 3) * anchors + anchor;
                raw[base] = *x;
                raw[base + anchors] = *y;
                raw[base + 2 * anchors] = *v;
            }
        };
        // anchor 0 (kept, conf 0.9) overlaps anchor 1 (suppressed, conf 0.8);
        // anchor 2 (kept, conf 0.3) sits elsewhere.
        write(0, 100.0, 100.0, 0.9, [(100.0, 100.0, 0.1), (120.0, 120.0, 0.2)]);
        write(1, 105.0, 100.0, 0.8, [(400.0, 400.0, 0.9), (420.0, 420.0, 0.8)]);
        write(2, 400.0, 400.0, 0.3, [(500.0, 500.0, 0.5), (520.0, 520.0, 0.6)]);
        (raw, channels, anchors)
    }

    #[test]
    fn decode_pose_keeps_keypoints_of_the_surviving_candidates() {
        let pre = letterboxed_pre();
        let (raw, channels, anchors) = pose_standard_inputs();
        let kpt = KptShape { count: 2, dim: 3 };
        let dets = decode_pose(
            &raw,
            &[1, channels as i64, anchors as i64],
            &pre,
            &[],
            0.25,
            0.7,
            kpt,
        )
        .unwrap();
        assert_eq!(dets.len(), 2, "{dets:?}");
        assert!((dets[0].confidence - 0.9).abs() < 1e-6);
        assert!((dets[1].confidence - 0.3).abs() < 1e-6);
        assert_eq!(dets[0].class_id, 0);
        // one class and no `names` metadata: the built-in COCO table only applies to 80-class
        // models, so the name falls back to `class_<id>`
        assert_eq!(dets[0].class_name, "class_0");

        // The keypoints must be the ones of the *kept* candidate (100, 100) / (120, 120),
        // not the suppressed overlapping one at (400, 400), mapped back through the letterbox
        // (gain_x 0.5, gain_y 0.375) and clipped to the 320x240 source.
        let first = dets[0].keypoints.as_ref().expect("pose detections carry keypoints");
        assert_eq!(first.len(), 2);
        assert!((first[0].x - 200.0).abs() < 1e-3, "{first:?}");
        assert!((first[0].y - 240.0).abs() < 1e-3, "{first:?}"); // 100/0.375 clips to 240
        assert_eq!(first[0].visibility, Some(0.1));
        assert!((first[1].x - 240.0).abs() < 1e-3, "{first:?}");
        assert!((first[1].y - 240.0).abs() < 1e-3, "{first:?}"); // 320 clips to 240
        assert_eq!(first[1].visibility, Some(0.2));

        // x beyond the source width clamps on the right edge
        let second = dets[1].keypoints.as_ref().expect("pose detections carry keypoints");
        assert!((second[0].x - 320.0).abs() < 1e-3, "{second:?}");
        assert!((second[0].y - 240.0).abs() < 1e-3, "{second:?}");
        assert_eq!(second[1].visibility, Some(0.6));
    }

    #[test]
    fn decode_pose_supports_kpt_dim_two() {
        let pre = letterboxed_pre();
        let kpt = KptShape { count: 2, dim: 2 };
        let anchors = 1usize;
        let channels = 4 + 1 + kpt.channels();
        let mut raw = vec![0f32; channels * anchors];
        // one box at (100, 75) with cm size 40x40, conf 0.8, two (x, y) keypoints
        raw[0] = 100.0;
        raw[anchors] = 75.0;
        raw[2 * anchors] = 40.0;
        raw[3 * anchors] = 40.0;
        raw[4 * anchors] = 0.8;
        raw[5 * anchors] = 100.0;
        raw[6 * anchors] = 75.0;
        raw[7 * anchors] = 125.0;
        raw[8 * anchors] = 150.0;

        let dets = decode_pose(&raw, &[1, channels as i64, anchors as i64], &pre, &[], 0.25, 0.7, kpt)
            .unwrap();
        assert_eq!(dets.len(), 1, "{dets:?}");
        let points = dets[0].keypoints.as_ref().unwrap();
        assert_eq!(points.len(), 2);
        assert!(points.iter().all(|point| point.visibility.is_none()), "{points:?}");
        assert!((points[0].x - 200.0).abs() < 1e-3, "{points:?}");
        assert!((points[0].y - 200.0).abs() < 1e-3, "{points:?}");
        assert!((points[1].x - 250.0).abs() < 1e-3, "{points:?}");
        assert!((points[1].y - 240.0).abs() < 1e-3, "{points:?}"); // 150/0.375 = 400 -> clipped
    }

    #[test]
    fn decode_pose_end2end_filters_and_preserves_order() {
        let pre = letterboxed_pre();
        let kpt = KptShape { count: 2, dim: 3 };
        let row_len = END2END_COLUMNS + kpt.channels();
        let mut raw = vec![0f32; 4 * row_len];
        let rows = [
            (0usize, (10.0f32, 20.0, 30.0, 40.0), 0.9f32, 0.0f32),
            (1, (100.0, 200.0, 130.0, 240.0), 0.1, 5.0), // below threshold
            (2, (50.0, 60.0, 70.0, 80.0), 0.5, 2.0),
            (3, (5.0, 6.0, 7.0, 8.0), 0.3, 7.0),
        ];
        for (row, (x1, y1, x2, y2), conf, class) in rows {
            let base = row * row_len;
            raw[base] = x1;
            raw[base + 1] = y1;
            raw[base + 2] = x2;
            raw[base + 3] = y2;
            raw[base + 4] = conf;
            raw[base + 5] = class;
            // keypoints: (x, y, v) triples derived from the row so mixups are visible
            for point in 0..kpt.count {
                let offset = base + END2END_COLUMNS + point * 3;
                raw[offset] = 20.0 * (row + point) as f32;
                raw[offset + 1] = 10.0 * (row + point) as f32;
                raw[offset + 2] = 0.5 + 0.1 * point as f32;
            }
        }

        let dets = decode_pose_end2end(&raw, &pre, &[], 0.25, kpt).unwrap();
        assert_eq!(dets.len(), 3, "{dets:?}");
        assert_eq!(
            dets.iter().map(|d| d.class_id).collect::<Vec<_>>(),
            vec![0, 2, 7]
        );
        // no `names` metadata: ids < 80 fall back to the built-in COCO table
        assert_eq!(dets[0].class_name, "person");
        assert_eq!(dets[1].class_name, "car");
        assert_eq!((dets[0].x1, dets[0].y1), (20.0, 20.0 / 0.375));
        let points = dets[1].keypoints.as_ref().unwrap();
        assert_eq!(points.len(), 2);
        assert!((points[0].x - 80.0).abs() < 1e-3, "{points:?}"); // 20 * row 2 / gain 0.5
        assert!((points[0].y - 20.0 / 0.375).abs() < 1e-3, "{points:?}"); // 10 * row 2
        assert_eq!(points[0].visibility, Some(0.5));
        assert_eq!(points[1].visibility, Some(0.6));

        // a tensor that is not a whole number of rows is rejected
        assert!(decode_pose_end2end(&raw[..row_len - 1], &pre, &[], 0.25, kpt).is_err());
    }

    #[test]
    fn decode_pose_rejects_shapes_without_room_for_keypoints() {
        let pre = identity_pre();
        let kpt = KptShape::COCO;
        // 56 channels fit 17x3 keypoints and one class; 50 do not
        assert!(decode_pose(&vec![0f32; 50 * 2], &[1, 50, 2], &pre, &[], 0.25, 0.7, kpt).is_err());
        assert!(
            decode_pose(&vec![0f32; 56 * 2], &[1, 56, 2], &pre, &[], 0.25, 0.7, kpt)
                .unwrap()
                .is_empty()
        );
        // not 3-D
        assert!(decode_pose(&[], &[1, 56], &pre, &[], 0.25, 0.7, kpt).is_err());
    }

    #[test]
    fn decode_output_routes_pose_by_metadata_and_shape() {
        let pre = identity_pre();
        let coco = KptShape::COCO;
        let end2end_row = 6 + coco.channels();

        // metadata says (17, 3) and the last dimension is 6 + 17*3 -> end-to-end pose
        let mut raw = vec![0f32; 2 * end2end_row];
        raw[0] = 10.0;
        raw[1] = 20.0;
        raw[2] = 30.0;
        raw[3] = 40.0;
        raw[4] = 0.9;
        raw[5] = 1.0;
        raw[6] = 10.0; // first keypoint x
        raw[7] = 20.0; // first keypoint y
        raw[8] = 0.7; // first keypoint v
        let dets =
            decode_output_with_kpt(&raw, &[1, 2, end2end_row as i64], &pre, &[], 0.25, 0.7, Some(coco))
                .unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        let points = dets[0].keypoints.as_ref().unwrap();
        assert_eq!(points.len(), coco.count);
        assert_eq!((points[0].x, points[0].y), (10.0, 20.0));
        assert_eq!(points[0].visibility, Some(0.7));
        assert_eq!(points[16].visibility, Some(0.0)); // padded row values

        // metadata says (17, 3) and the last dimension is the anchor count -> standard pose
        let anchors = 2usize;
        let mut raw = vec![0f32; 56 * anchors];
        raw[0] = 100.0;
        raw[anchors] = 100.0;
        raw[2 * anchors] = 20.0;
        raw[3 * anchors] = 20.0;
        raw[4 * anchors] = 0.9; // the single class score
        raw[5 * anchors] = 50.0; // keypoint 0 x
        raw[6 * anchors] = 60.0; // keypoint 0 y
        raw[7 * anchors] = 0.4; // keypoint 0 v
        let dets =
            decode_output_with_kpt(&raw, &[1, 56, anchors as i64], &pre, &[], 0.25, 0.7, Some(coco))
                .unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].keypoints.as_ref().unwrap()[0].x, 50.0);

        // no metadata: both COCO pose shapes are recognised ...
        let dets =
            decode_output_with_kpt(&raw, &[1, 56, anchors as i64], &pre, &[], 0.25, 0.7, None)
                .unwrap();
        assert_eq!(dets.len(), 1);
        assert!(dets[0].keypoints.is_some());
        let mut raw = vec![0f32; 2 * end2end_row];
        raw[4] = 0.9;
        let dets = decode_output_with_kpt(
            &raw,
            &[1, 2, end2end_row as i64],
            &pre,
            &[],
            0.25,
            0.7,
            None,
        )
        .unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].keypoints.as_ref().unwrap().len(), 17);

        // ... while a shape that declares nothing stays a detect model
        let mut raw = vec![0f32; 6 * anchors];
        raw[0] = 100.0;
        raw[anchors] = 100.0;
        raw[2 * anchors] = 20.0;
        raw[3 * anchors] = 20.0;
        raw[4 * anchors] = 0.9;
        let dets =
            decode_output_with_kpt(&raw, &[1, 6, anchors as i64], &pre, &[], 0.25, 0.7, None)
                .unwrap();
        assert_eq!(dets.len(), 1);
        assert!(dets[0].keypoints.is_none());

        // metadata wins over the shape heuristic: (4, 2) keypoints need 4+1+8 channels
        let kpt = KptShape { count: 4, dim: 2 };
        let anchors = 1usize;
        let mut raw = vec![0f32; 13 * anchors];
        raw[0] = 100.0;
        raw[anchors] = 100.0;
        raw[2 * anchors] = 20.0;
        raw[3 * anchors] = 20.0;
        raw[4 * anchors] = 0.9;
        let dets =
            decode_output_with_kpt(&raw, &[1, 13, anchors as i64], &pre, &[], 0.25, 0.7, Some(kpt))
                .unwrap();
        assert_eq!(dets.len(), 1);
        let points = dets[0].keypoints.as_ref().unwrap();
        assert_eq!(points.len(), 4);
        assert!(points.iter().all(|point| point.visibility.is_none()));
    }

    fn obb_candidate(
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        angle: f32,
        confidence: f32,
        class_id: i32,
    ) -> ObbCandidate {
        ObbCandidate {
            cx,
            cy,
            w,
            h,
            angle,
            confidence,
            class_id,
        }
    }

    #[test]
    fn probiou_scores_overlap_and_alignment() {
        let a = (100.0, 100.0, 40.0, 20.0, 0.0);
        // identical boxes approach 1 (bounded by eps), disjoint boxes score 0
        assert!(probiou(a, a) > 0.99, "{}", probiou(a, a));
        assert!(probiou(a, (500.0, 500.0, 40.0, 20.0, 0.0)) < 1e-3);

        // a half-overlap along x: in the ballpark of the plain IoU (0.6)
        let overlap = probiou(a, (110.0, 100.0, 40.0, 20.0, 0.0));
        assert!((overlap - 0.6).abs() < 0.2, "{overlap}");
        // rotating the second box by 45 degrees lowers the score
        let rotated = probiou(a, (110.0, 100.0, 40.0, 20.0, std::f32::consts::FRAC_PI_4));
        assert!(rotated < overlap, "{rotated} vs {overlap}");
        // a 90-degree rotation of a 2:1 box around the same center drops the score hard
        let perpendicular = probiou(a, (100.0, 100.0, 40.0, 20.0, std::f32::consts::FRAC_PI_2));
        assert!(perpendicular < 0.6, "{perpendicular}");
    }

    #[test]
    fn nms_rotated_is_class_aware_and_not_greedy() {
        let candidates = vec![
            obb_candidate(100.0, 100.0, 40.0, 20.0, 0.0, 0.9, 0),
            obb_candidate(105.0, 100.0, 40.0, 20.0, 0.0, 0.8, 0), // overlaps #0 -> suppressed
            obb_candidate(100.0, 100.0, 40.0, 20.0, 0.0, 0.7, 1), // same box, other class: kept
            obb_candidate(400.0, 400.0, 40.0, 20.0, 0.0, 0.6, 0), // far away: kept
        ];
        assert_eq!(nms_rotated(&candidates, 0.7, MAX_DET), vec![0, 2, 3]);
        assert!(nms_rotated(&[], 0.7, MAX_DET).is_empty());

        // fast_nms semantics: a suppressed box still suppresses. #0 overlaps #1 (suppressed by
        // #0) and #1 overlaps #2, but #0 does not overlap #2 — greedy NMS would keep #2.
        let chain = vec![
            obb_candidate(0.0, 0.0, 40.0, 20.0, 0.0, 0.9, 0),
            obb_candidate(8.0, 0.0, 40.0, 20.0, 0.0, 0.8, 0),
            obb_candidate(16.0, 0.0, 40.0, 20.0, 0.0, 0.7, 0),
        ];
        let p01 = probiou(chain[0].suppression_box(), chain[1].suppression_box());
        let p12 = probiou(chain[1].suppression_box(), chain[2].suppression_box());
        let p02 = probiou(chain[0].suppression_box(), chain[2].suppression_box());
        assert!(p01 > 0.6 && p12 > 0.6 && p02 <= 0.6, "{p01} {p12} {p02}");
        assert_eq!(nms_rotated(&chain, 0.6, MAX_DET), vec![0]);

        // descending confidence order, capped at max_det
        let far = vec![
            obb_candidate(0.0, 0.0, 10.0, 10.0, 0.0, 0.5, 0),
            obb_candidate(100.0, 0.0, 10.0, 10.0, 0.0, 0.9, 0),
            obb_candidate(200.0, 0.0, 10.0, 10.0, 0.0, 0.7, 0),
        ];
        assert_eq!(nms_rotated(&far, 0.7, 2), vec![1, 2]);
    }

    #[test]
    fn decode_obb_decodes_angle_and_runs_rotated_nms() {
        let pre = letterboxed_pre();
        let anchors = 3usize;
        let channels = 6usize; // 4 box + 1 class + 1 angle
        let mut raw = vec![0f32; channels * anchors];
        let mut write = |anchor: usize, cx: f32, cy: f32, w: f32, h: f32, conf: f32, angle: f32| {
            raw[anchor] = cx;
            raw[anchors + anchor] = cy;
            raw[2 * anchors + anchor] = w;
            raw[3 * anchors + anchor] = h;
            raw[4 * anchors + anchor] = conf;
            raw[5 * anchors + anchor] = angle;
        };
        write(0, 100.0, 100.0, 40.0, 20.0, 0.9, 0.5);
        write(1, 105.0, 100.0, 40.0, 20.0, 0.8, 0.5); // overlaps #0 -> suppressed
        write(2, 400.0, 400.0, 30.0, 30.0, 0.3, -0.25);
        let dets = decode_obb(&raw, &[1, channels as i64, anchors as i64], &pre, &[], 0.25, 0.7)
            .unwrap();
        assert_eq!(dets.len(), 2, "{dets:?}");

        let first = &dets[0];
        assert!((first.confidence - 0.9).abs() < 1e-6);
        assert_eq!(first.class_name, "class_0"); // one class, no metadata: no COCO fallback
        assert!(first.keypoints.is_none());
        let obb = first.obb.expect("OBB detections carry a rotated box");
        // scale_boxes' xywh branch: padding off the center, per-axis gains, no clipping
        assert!((obb.cx - 200.0).abs() < 1e-3, "{obb:?}");
        assert!((obb.cy - 100.0 / 0.375).abs() < 1e-3, "{obb:?}");
        assert!((obb.w - 80.0).abs() < 1e-3, "{obb:?}");
        assert!((obb.h - 20.0 / 0.375).abs() < 1e-3, "{obb:?}");
        assert_eq!(obb.angle, 0.5);
        // 100 / 0.375 = 266.67 exceeds the 240px source height and is *not* clipped
        assert!(obb.cy > 240.0);
        // x1..y2 encloses the rotated box
        let (x1, y1, x2, y2) = obb.enclosing_box();
        assert_eq!((first.x1, first.y1, first.x2, first.y2), (x1, y1, x2, y2));
        for (cx, cy) in obb.corners() {
            assert!(cx >= x1 - 1e-3 && cx <= x2 + 1e-3 && cy >= y1 - 1e-3 && cy <= y2 + 1e-3);
        }

        assert_eq!(dets[1].obb.unwrap().angle, -0.25);

        // malformed shapes are rejected
        assert!(decode_obb(&raw, &[1, channels as i64], &pre, &[], 0.25, 0.7).is_err());
        assert!(decode_obb(&raw[..raw.len() - 1], &[1, channels as i64, anchors as i64], &pre, &[], 0.25, 0.7).is_err());
    }

    #[test]
    fn decode_obb_end2end_filters_rows_and_keeps_xywh() {
        let pre = identity_pre();
        let rows = [
            (100.0f32, 120.0, 40.0, 20.0, 0.9, 2.0, 0.3),
            (300.0, 300.0, 30.0, 30.0, 0.1, 0.0, 0.0), // below threshold
            (500.0, 500.0, 50.0, 10.0, 0.5, 7.0, -0.5),
        ];
        let mut raw = Vec::new();
        for (cx, cy, w, h, conf, class, angle) in rows {
            raw.extend_from_slice(&[cx, cy, w, h, conf, class, angle]);
        }

        let dets = decode_obb_end2end(&raw, &pre, &[], 0.25).unwrap();
        assert_eq!(dets.len(), 2, "{dets:?}");
        assert_eq!(dets[0].class_id, 2);
        assert_eq!(dets[0].class_name, "car"); // end2end: no class count, COCO fallback
        let obb = dets[0].obb.unwrap();
        assert_eq!(
            (obb.cx, obb.cy, obb.w, obb.h, obb.angle),
            (100.0, 120.0, 40.0, 20.0, 0.3)
        );
        assert_eq!(dets[1].obb.unwrap().angle, -0.5);

        // a tensor that is not a whole number of rows is rejected
        assert!(decode_obb_end2end(&raw[..raw.len() - 1], &pre, &[], 0.25).is_err());
    }

    #[test]
    fn decode_obb_output_routes_by_shape() {
        let pre = identity_pre();
        // end-to-end (1, N, 7): rows cx cy w h conf cls angle
        let mut e2e = vec![0f32; 14];
        e2e[0] = 100.0;
        e2e[1] = 100.0;
        e2e[2] = 40.0;
        e2e[3] = 20.0;
        e2e[4] = 0.9;
        e2e[5] = 1.0;
        e2e[6] = 0.7;
        let dets = decode_obb_output(&e2e, &[1, 2, 7], &pre, &[], 0.25, 0.7).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].obb.unwrap().angle, 0.7);

        // standard (1, 4+nc+1, A), channel-major
        let anchors = 2usize;
        let mut standard = vec![0f32; 6 * anchors];
        standard[0] = 100.0;
        standard[anchors] = 100.0;
        standard[2 * anchors] = 40.0;
        standard[3 * anchors] = 20.0;
        standard[4 * anchors] = 0.9;
        standard[5 * anchors] = 0.7;
        let dets = decode_obb_output(&standard, &[1, 6, anchors as i64], &pre, &[], 0.25, 0.7)
            .unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].obb.unwrap().angle, 0.7);

        assert!(decode_obb_output(&[], &[1, 2], &pre, &[], 0.25, 0.7).is_err());
    }
}
