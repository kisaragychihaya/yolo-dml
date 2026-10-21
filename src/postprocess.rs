//! Postprocessing: decode raw YOLO output tensors -> boxes -> NMS.
//!
//! Two output layouts are supported: the end-to-end export (`1 x N x 6`, NMS already
//! applied by the model) and the standard export (`1 x (4 + nc) x anchors`, needs
//! transposing, thresholding and class-aware NMS here).

use crate::detector::Detection;
use crate::error::{Error, Result};
use crate::preprocess::Preprocessed;

/// Number of classes in the standard COCO checkpoints.
pub const COCO_CLASSES: usize = 80;

/// Maximum number of detections kept per image (ultralytics `max_det`).
pub const MAX_DET: usize = 300;

/// Column count of an end-to-end (`NMS already applied`) output row: `x1 y1 x2 y2 conf cls`.
const END2END_COLUMNS: usize = 6;

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
#[derive(Debug, Clone, Copy)]
struct Candidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: i32,
}

impl Candidate {
    /// Box shifted by the class offset, i.e. what NMS actually compares.
    fn suppression_box(&self) -> (f32, f32, f32, f32) {
        let shift = self.class_id as f32 * CLASS_OFFSET;
        (self.x1 + shift, self.y1 + shift, self.x2 + shift, self.y2 + shift)
    }
}

/// Greedy NMS. Candidates are returned in descending confidence order, capped at `max_det`.
fn nms(candidates: &[Candidate], iou_threshold: f32, max_det: usize) -> Vec<Candidate> {
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
        });
    }
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
            }
        })
        .collect())
}

/// Decode whichever layout the model produced.
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
}
