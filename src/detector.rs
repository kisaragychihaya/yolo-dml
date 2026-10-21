//! Detector: owns the ONNX Runtime session (DirectML or CPU) behind an opaque handle.

use std::any::Any;
use std::borrow::Cow;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::postprocess::{self, parse_names, KptShape};
use crate::preprocess::{self, Preprocessed, INPUT_HEIGHT, INPUT_WIDTH};

/// Confidence threshold used when the caller passes `0` or a negative value.
pub const DEFAULT_CONF_THRESHOLD: f32 = 0.25;
/// NMS IoU threshold used when the caller passes `0` or a negative value.
pub const DEFAULT_IOU_THRESHOLD: f32 = 0.7;

/// Execution provider requested by the caller through `options_json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// DirectML execution provider (`{"device":"dml"}`, the default).
    Dml,
    /// CPU execution provider (`{"device":"cpu"}`).
    Cpu,
}

impl Device {
    /// Name echoed back in the JSON `meta.device` field.
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Dml => "dml",
            Device::Cpu => "cpu",
        }
    }
}

/// Parsed `options_json` accepted by `yolo_create`.
///
/// Recognised shape: `{"device":"dml"|"cpu","num_threads":n}`; an empty string or
/// `null` pointer means defaults (`dml`, system thread count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectorOptions {
    pub device: Device,
    pub num_threads: Option<i32>,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        Self {
            device: Device::Dml,
            num_threads: None,
        }
    }
}

impl DetectorOptions {
    /// Parse the optional `options_json` argument of `yolo_create`.
    pub fn from_json(json: Option<&str>) -> Result<Self> {
        let Some(json) = json.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(Self::default());
        };
        let value: Value = serde_json::from_str(json)
            .map_err(|e| Error::invalid(format!("options_json is not valid JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| Error::invalid("options_json must be a JSON object"))?;

        let device = match obj.get("device") {
            None | Some(Value::Null) => Device::Dml,
            Some(Value::String(s)) if s.eq_ignore_ascii_case("dml") => Device::Dml,
            Some(Value::String(s)) if s.eq_ignore_ascii_case("cpu") => Device::Cpu,
            Some(other) => {
                return Err(Error::invalid(format!(
                    "options_json.device must be \"dml\" or \"cpu\", got {other}"
                )))
            }
        };

        let num_threads = match obj.get("num_threads") {
            None | Some(Value::Null) => None,
            Some(v) => {
                let n = v.as_i64().ok_or_else(|| {
                    Error::invalid("options_json.num_threads must be an integer")
                })?;
                if n < 1 || n > i32::MAX as i64 {
                    return Err(Error::invalid(format!(
                        "options_json.num_threads out of range: {n}"
                    )));
                }
                Some(n as i32)
            }
        };

        Ok(Self {
            device,
            num_threads,
        })
    }
}

/// Raw detection in pixel coordinates of the source image (top-left origin).
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: i32,
    pub class_name: String,
    /// Pose keypoints, in source-image pixels; `None` for detect models.
    pub keypoints: Option<Vec<Keypoint>>,
}

/// One pose keypoint in pixel coordinates of the source image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    /// Visibility as it came out of the graph (already sigmoid'd by the exporter), or `None` for
    /// `(K, 2)` keypoint shapes that carry no visibility channel.
    pub visibility: Option<f32>,
}

/// What a model predicts: plain boxes, or boxes plus keypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Object detection (`meta.task` `"detect"`).
    Detect,
    /// Pose estimation (`meta.task` `"pose"`).
    Pose,
}

impl Task {
    /// Name echoed back in the JSON `meta.task` field.
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Detect => "detect",
            Task::Pose => "pose",
        }
    }
}

/// Inference result for one image, before JSON serialisation.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
    pub width: i32,
    pub height: i32,
    pub device: Device,
    /// Overrides [`Device::as_str`] in `meta.device`, e.g. when DirectML was requested but the
    /// session had to fall back to the CPU provider.
    pub device_detail: Option<String>,
    pub inference_ms: f32,
    /// Reported in `meta.task`; `detect` unless the model predicts keypoints.
    pub task: Task,
}

impl DetectionResult {
    /// Serialise to the JSON document returned by `yolo_detect_*`.
    ///
    /// Schema: `{"count":N,"detections":[{"x1":f,"y1":f,"x2":f,"y2":f,"confidence":f,
    /// "class_id":i,"class_name":s[,"keypoints":[[x,y,v],...]]}],
    /// "meta":{"width":i,"height":i,"device":s,"inference_ms":f,"task":s}}`
    ///
    /// The `keypoints` array is only present for pose models; a `(K, 2)` model emits `[x, y]`
    /// pairs instead of `[x, y, v]` triples.
    pub fn to_json(&self) -> String {
        let detections: Vec<Value> = self
            .detections
            .iter()
            .map(|d| {
                let mut value = serde_json::json!({
                    "x1": round3(d.x1),
                    "y1": round3(d.y1),
                    "x2": round3(d.x2),
                    "y2": round3(d.y2),
                    "confidence": round3(d.confidence),
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                });
                if let (Some(keypoints), Some(object)) = (&d.keypoints, value.as_object_mut()) {
                    object.insert("keypoints".to_string(), keypoints_json(keypoints));
                }
                value
            })
            .collect();
        serde_json::json!({
            "count": detections.len(),
            "detections": detections,
            "meta": {
                "width": self.width,
                "height": self.height,
                "device": self.device_detail.as_deref().unwrap_or_else(|| self.device.as_str()),
                "inference_ms": round3(self.inference_ms),
                "task": self.task.as_str(),
            }
        })
        .to_string()
    }
}

/// `[[x, y, v], ...]` for `(K, 3)` keypoints, `[[x, y], ...]` when there is no visibility.
fn keypoints_json(keypoints: &[Keypoint]) -> Value {
    Value::Array(
        keypoints
            .iter()
            .map(|point| match point.visibility {
                Some(visibility) => {
                    serde_json::json!([round3(point.x), round3(point.y), round3(visibility)])
                }
                None => serde_json::json!([round3(point.x), round3(point.y)]),
            })
            .collect(),
    )
}

fn round3(value: f32) -> f64 {
    ((value as f64) * 1000.0).round() / 1000.0
}

/// Resolve the bundled `onnxruntime.dll` / initialise ONNX Runtime once per process.
///
/// `ort` loads ONNX Runtime lazily through `libloading` and *panics* when the library is
/// missing, so the initialisation is wrapped in [`std::panic::catch_unwind`] and cached.
fn init_ort() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    let outcome = INIT.get_or_init(|| {
        let attempt = std::panic::catch_unwind(|| {
            let builder = match std::env::var("YOLO_ORT_PATH") {
                Ok(path) if !path.trim().is_empty() => ort::init_from(path),
                // No explicit path: `ort` resolves `onnxruntime.dll` next to the host executable
                // and then through the normal DLL search order (`ORT_DYLIB_PATH`, PATH).
                _ => ort::init(),
            };
            builder.with_name("yolo_dml").commit().map(|_| ())
        });
        match attempt {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e.to_string()),
            Err(payload) => Err(panic_message(payload)),
        }
    });
    outcome.clone().map_err(|message| Error::Ort(message.into()))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic while initialising ONNX Runtime".to_string()
    }
}

fn ort_err(err: ort::Error) -> Error {
    Error::Ort(err.to_string().into())
}

/// Parse the `imgsz` custom metadata (`"[640, 640]"`) into `(height, width)`.
fn parse_imgsz(raw: &str) -> Option<(u32, u32)> {
    let nums: Vec<u32> = raw
        .trim()
        .trim_matches(|c: char| !c.is_ascii_digit() && c != ',' && c != ' ')
        .split(',')
        .filter_map(|part| {
            let digits: String = part.chars().filter(char::is_ascii_digit).collect();
            digits.parse::<u32>().ok()
        })
        .collect();
    match nums.as_slice() {
        [h, w] => Some((*h, *w)),
        [side] => Some((*side, *side)),
        _ => None,
    }
}

/// How the session's output tensors are laid out, decided once at load time from
/// [`Session::outputs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputLayout {
    /// A single output tensor: either the standard `1 x (4 + nc) x anchors` export or the
    /// end-to-end `1 x N x 6` one, told apart by shape in [`postprocess::decode_output`]. The
    /// `iou` argument of the detect entry points is honoured here.
    Single,
    /// Four tensors in the `torchvision` `PostprocessDetections` style, with the NMS already
    /// applied by the graph: `num_detections` `1 x 1`, `detection_boxes` `1 x N x 4`,
    /// `detection_scores` `1 x N` and `detection_classes` `1 x N`. The `iou` argument of the
    /// detect entry points is ignored for these models — there is nothing left to suppress.
    NmsApplied {
        num: String,
        boxes: String,
        scores: String,
        classes: String,
    },
}

/// Names the four `torchvision` style outputs are recognised by: `(slot label, needles)`.
///
/// Matching is case-insensitive substring matching against the model's output names. The order
/// is the assignment priority, so that e.g. `detection_classes` is not stolen by the `dets`
/// needle meant for `num_detections`.
const FOUR_OUTPUT_NEEDLES: [(&str, &[&str]); 4] = [
    ("boxes", &["box"]),
    ("scores", &["score"]),
    ("classes", &["class", "label"]),
    ("num_detections", &["num", "dets"]),
];

/// Case-insensitive "does `name` contain one of `needles`".
fn name_matches(name: &str, needles: &[&str]) -> bool {
    let lowered = name.to_ascii_lowercase();
    needles.iter().any(|needle| lowered.contains(needle))
}

/// Render declared outputs as `\`name\` (Float32 [1, 100, 4]), ...` for error messages.
fn describe_outputs(outputs: &[(&str, &ValueType)]) -> String {
    if outputs.is_empty() {
        return "the model exposes no outputs".to_string();
    }
    outputs
        .iter()
        .map(|(name, output_type)| match output_type {
            ValueType::Tensor { ty, dimensions, .. } => format!("`{name}` ({ty:?} {dimensions:?})"),
            other => format!("`{name}` ({other:?})"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Index of the first not-yet-taken output whose name matches `needles`.
fn take_slot(
    outputs: &[(&str, &ValueType)],
    taken: &mut [bool],
    needles: &[&str],
) -> Option<usize> {
    let index = (0..outputs.len()).find(|&i| !taken[i] && name_matches(outputs[i].0, needles))?;
    taken[index] = true;
    Some(index)
}

/// Match the four `torchvision` style outputs to their roles.
///
/// Returns the output indices of `boxes`, `scores`, `classes` and `num_detections`, in that
/// order. A single unrecognised name is filled in by exclusion (one name left for one open
/// slot); anything less certain is an error rather than a guess.
fn select_four(outputs: &[(&str, &ValueType)]) -> Result<[usize; 4]> {
    let mut taken = vec![false; outputs.len()];
    let mut selected: [Option<usize>; 4] = [None; 4];
    for (slot, (_, needles)) in FOUR_OUTPUT_NEEDLES.iter().enumerate() {
        selected[slot] = take_slot(outputs, &mut taken, needles);
    }

    let open: Vec<usize> = (0..4).filter(|&slot| selected[slot].is_none()).collect();
    let remaining: Vec<usize> = (0..outputs.len()).filter(|&i| !taken[i]).collect();
    if let ([slot], [index]) = (open.as_slice(), remaining.as_slice()) {
        selected[*slot] = Some(*index);
    }

    for (slot, (label, _)) in FOUR_OUTPUT_NEEDLES.iter().enumerate() {
        if selected[slot].is_none() {
            return Err(Error::Ort(
                format!(
                    "model has four outputs ({}) but none of them looks like `{label}`",
                    describe_outputs(outputs)
                )
                .into(),
            ));
        }
    }
    Ok(std::array::from_fn(|slot| {
        selected[slot].expect("every slot was resolved above")
    }))
}

/// Decide the output layout from the declared session outputs, in declaration order.
///
/// One output means the single-tensor decoders; four means the `torchvision`-style layout;
/// any other count is rejected instead of silently reading `outputs[0]`, which would decode a
/// model whose tensors are not what this library expects.
fn resolve_layout(outputs: &[(&str, &ValueType)]) -> Result<OutputLayout> {
    match outputs.len() {
        1 => Ok(OutputLayout::Single),
        4 => {
            let [boxes, scores, classes, num] = select_four(outputs)?;
            for (label, index) in [
                ("boxes", boxes),
                ("scores", scores),
                ("classes", classes),
                ("num_detections", num),
            ] {
                let (name, output_type) = outputs[index];
                let ValueType::Tensor { ty, .. } = output_type else {
                    return Err(Error::Ort(
                        format!(
                            "output `{name}` is used as `{label}` and must be a tensor: {}",
                            describe_outputs(outputs)
                        )
                        .into(),
                    ));
                };
                if matches!(label, "boxes" | "scores") && *ty != TensorElementType::Float32 {
                    return Err(Error::Ort(
                        format!(
                            "output `{name}` is used as `{label}` and must be float32, got {ty:?}: {}",
                            describe_outputs(outputs)
                        )
                        .into(),
                    ));
                }
            }
            Ok(OutputLayout::NmsApplied {
                num: outputs[num].0.to_string(),
                boxes: outputs[boxes].0.to_string(),
                scores: outputs[scores].0.to_string(),
                classes: outputs[classes].0.to_string(),
            })
        }
        count => Err(Error::Ort(
            format!(
                "unsupported output count {count}, expected 1 or 4 ({})",
                describe_outputs(outputs)
            )
            .into(),
        )),
    }
}

/// `num_detections` scalar, whose dtype exporters disagree on (`int32`, `int64` or `float`).
fn extract_num_detections(value: &DynValue) -> Result<i64> {
    if let Ok((_, values)) = value.try_extract_raw_tensor::<i32>() {
        return Ok(values.first().copied().unwrap_or(0) as i64);
    }
    if let Ok((_, values)) = value.try_extract_raw_tensor::<i64>() {
        return Ok(values.first().copied().unwrap_or(0));
    }
    if let Ok((_, values)) = value.try_extract_raw_tensor::<f32>() {
        return Ok(values.first().copied().unwrap_or(0.0).round() as i64);
    }
    Err(Error::Ort(
        format!(
            "`num_detections` must be an int32, int64 or float32 tensor, got {:?}",
            value.dtype()
        )
        .into(),
    ))
}

/// Element type of a tensor handed to [`Detector::decode_external`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    /// `float32` (`yolo_decode_tensors` dtype `0`).
    Float32,
    /// `int32` (`yolo_decode_tensors` dtype `1`).
    Int32,
    /// `int64` (`yolo_decode_tensors` dtype `2`).
    Int64,
}

impl TensorDtype {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            TensorDtype::Float32 | TensorDtype::Int32 => 4,
            TensorDtype::Int64 => 8,
        }
    }
}

/// One raw output tensor borrowed from the caller of [`Detector::decode_external`].
///
/// `data` is the tensor payload as the graph produced it (row-major, no padding between rows,
/// native little-endian on Windows x86-64); `dims` is its shape, batch first, exactly as ORT
/// reports it. Nothing is copied unless the buffer is not aligned for its element type.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// Output name, used to match the four `torchvision` tensors to their roles.
    pub name: Option<&'a str>,
    /// Raw payload; see [`Self::elements`] for how many bytes are read.
    pub data: &'a [u8],
    /// Shape as emitted by the model.
    pub dims: &'a [i64],
    pub dtype: TensorDtype,
}

impl<'a> TensorView<'a> {
    /// Name used in error messages.
    fn label(&self) -> &str {
        self.name.unwrap_or("<unnamed>")
    }

    /// Number of elements, validated against the length of [`Self::data`].
    fn elements(&self) -> Result<usize> {
        let elements = self
            .dims
            .iter()
            .try_fold(1usize, |acc, &dim| {
                usize::try_from(dim).ok().and_then(|dim| acc.checked_mul(dim))
            })
            .ok_or_else(|| {
                Error::invalid(format!(
                    "tensor `{}` has unusable dims {:?}",
                    self.label(),
                    self.dims
                ))
            })?;
        let expected = elements.checked_mul(self.dtype.size()).ok_or_else(|| {
            Error::invalid(format!("tensor `{}` is too large to address", self.label()))
        })?;
        if self.data.len() < expected {
            return Err(Error::invalid(format!(
                "tensor `{}` holds {} bytes, {:?} need {expected}",
                self.label(),
                self.data.len(),
                self.dims
            )));
        }
        Ok(elements)
    }

    /// Reinterpret the payload as `T`, copying only when the bytes are not aligned for `T`.
    ///
    /// # Safety contract (internal)
    /// Only called with `f32` / `i32` / `i64`: every bit pattern is a valid value of those, and
    /// `elements()` has already checked that the payload is a whole number of elements.
    fn data_as<T: Copy>(&self) -> Result<Cow<'a, [T]>> {
        self.elements()?;
        let size = std::mem::size_of::<T>();
        let data = &self.data[..self.data.len() / size * size];
        if (data.as_ptr() as usize) % std::mem::align_of::<T>() == 0 {
            // SAFETY: the pointer is aligned for `T`, the length is a whole number of `T`s and
            // `T` is one of the scalar types documented above.
            Ok(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(data.as_ptr().cast::<T>(), data.len() / size)
            }))
        } else {
            // SAFETY: `read_unaligned` has no alignment requirement and `chunk` is exactly
            // `size_of::<T>()` bytes long.
            Ok(Cow::Owned(
                data.chunks_exact(size)
                    .map(|chunk| unsafe { std::ptr::read_unaligned(chunk.as_ptr().cast::<T>()) })
                    .collect(),
            ))
        }
    }

    /// The payload as `f32`; the tensor must be `float32`.
    fn floats(&self) -> Result<Cow<'a, [f32]>> {
        if self.dtype != TensorDtype::Float32 {
            return Err(Error::invalid(format!(
                "tensor `{}` must be float32, got {:?}",
                self.label(),
                self.dtype
            )));
        }
        self.data_as::<f32>()
    }

    /// First element as an integer, accepting the `int32` / `int64` / `float32` dtypes exporters
    /// use for `num_detections`.
    fn scalar_i64(&self) -> Result<i64> {
        if self.elements()? == 0 {
            return Ok(0);
        }
        match self.dtype {
            TensorDtype::Float32 => Ok(self.data_as::<f32>()?[0].round() as i64),
            TensorDtype::Int32 => Ok(self.data_as::<i32>()?[0] as i64),
            TensorDtype::Int64 => Ok(self.data_as::<i64>()?[0]),
        }
    }
}

/// Slot of each `FOUR_OUTPUT_NEEDLES` label in the `[num, boxes, scores, classes]` arrays.
fn role_slot(label: &str) -> usize {
    match label {
        "num_detections" => 0,
        "boxes" => 1,
        "scores" => 2,
        "classes" => 3,
        other => unreachable!("unknown role {other}"),
    }
}

/// Render caller tensors as `\`name\` (Float32 [1, 100, 4]), ...` for error messages.
fn describe_views(tensors: &[TensorView<'_>]) -> String {
    tensors
        .iter()
        .map(|tensor| format!("`{}` ({:?} {:?})", tensor.label(), tensor.dtype, tensor.dims))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Match the caller's tensors to the `num_detections` / `boxes` / `scores` / `classes` roles of
/// the `torchvision` layout.
///
/// `declared` holds the names the session declared for those roles, in that order. Matching runs
/// in passes, each one only filling slots the previous passes left open: exact declared names,
/// then the role needles of [`FOUR_OUTPUT_NEEDLES`], then exclusion (one free slot, one free
/// tensor), then shape — a one-element tensor is `num_detections`, a rank-3 tensor ending in 4
/// is `boxes`, and the two remaining matrices are told apart by dtype (`scores` is float,
/// `classes` integer). Two remaining float matrices are indistinguishable and rejected instead
/// of guessed.
fn select_four_views(tensors: &[TensorView<'_>], declared: [&str; 4]) -> Result<[usize; 4]> {
    if tensors.len() != 4 {
        return Err(Error::invalid(format!(
            "this model has a torchvision four-output layout, expected 4 tensors, got {}",
            tensors.len()
        )));
    }
    let mut taken = [false; 4];
    let mut selected: [Option<usize>; 4] = [None; 4];

    for (slot, declared_name) in declared.iter().enumerate() {
        if let Some(index) =
            (0..tensors.len()).find(|&i| !taken[i] && tensors[i].name == Some(*declared_name))
        {
            selected[slot] = Some(index);
            taken[index] = true;
        }
    }

    for (label, needles) in FOUR_OUTPUT_NEEDLES {
        let slot = role_slot(label);
        if selected[slot].is_some() {
            continue;
        }
        if let Some(index) = (0..tensors.len()).find(|&i| {
            !taken[i] && tensors[i].name.is_some_and(|name| name_matches(name, needles))
        }) {
            selected[slot] = Some(index);
            taken[index] = true;
        }
    }

    let open: Vec<usize> = (0..4).filter(|&slot| selected[slot].is_none()).collect();
    let remaining: Vec<usize> = (0..tensors.len()).filter(|&i| !taken[i]).collect();
    if let ([slot], [index]) = (open.as_slice(), remaining.as_slice()) {
        selected[*slot] = Some(*index);
        taken[*index] = true;
    }

    // Shape fallback for `num_detections` and `boxes`.
    for slot in [0usize, 1] {
        if selected[slot].is_some() {
            continue;
        }
        let found = (0..tensors.len()).find(|&i| {
            if taken[i] {
                return false;
            }
            let tensor = &tensors[i];
            match slot {
                0 => tensor.elements().map(|count| count == 1).unwrap_or(false),
                _ => tensor.dims.len() == 3 && tensor.dims[2] == 4,
            }
        });
        if let Some(index) = found {
            selected[slot] = Some(index);
            taken[index] = true;
        }
    }

    // `scores` / `classes`: with both open, only the dtype separates them.
    let free: Vec<usize> = (0..tensors.len()).filter(|&i| !taken[i]).collect();
    if selected[2].is_none() && selected[3].is_none() && free.len() == 2 {
        let [a, b] = [free[0], free[1]];
        let a_is_float = tensors[a].dtype == TensorDtype::Float32;
        let b_is_float = tensors[b].dtype == TensorDtype::Float32;
        match (a_is_float, b_is_float) {
            (true, false) => {
                selected[2] = Some(a);
                selected[3] = Some(b);
            }
            (false, true) => {
                selected[2] = Some(b);
                selected[3] = Some(a);
            }
            _ => {
                return Err(Error::invalid(format!(
                    "cannot tell the scores from the classes among tensors {a} and {b} ({}); \
                     pass their names",
                    describe_views(&[tensors[a], tensors[b]])
                )))
            }
        }
    }

    for (slot, role) in ["num_detections", "boxes", "scores", "classes"]
        .iter()
        .enumerate()
    {
        if selected[slot].is_none() {
            return Err(Error::invalid(format!(
                "cannot map the caller's tensors onto the four-output layout: `{role}` is missing \
                 ({})",
                describe_views(tensors)
            )));
        }
    }
    Ok(selected.map(|slot| slot.expect("every slot is resolved above")))
}

/// One loaded YOLO model.
pub struct Detector {
    pub options: DetectorOptions,
    pub model_path: String,
    session: Session,
    input_name: String,
    input_width: u32,
    input_height: u32,
    batch: u32,
    class_names: Vec<String>,
    /// Keypoint layout from the `kpt_shape` metadata; `None` for detect models (and for pose
    /// exports whose metadata was stripped, which then rely on the shape heuristics).
    kpt_shape: Option<KptShape>,
    /// Task reported in `meta.task`.
    task: Task,
    /// Output layout, resolved once from the session's declared outputs.
    layout: OutputLayout,
    /// Provider the session really runs on (differs from `options.device` after a DML fallback).
    resolved_device: Device,
    device_detail: Option<String>,
}

impl std::fmt::Debug for Detector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Detector")
            .field("model_path", &self.model_path)
            .field("options", &self.options)
            .field("input_name", &self.input_name)
            .field("input_size", &(self.input_width, self.input_height))
            .field("batch", &self.batch)
            .field("classes", &self.class_names.len())
            .field("task", &self.task)
            .field("kpt_shape", &self.kpt_shape)
            .field("outputs", &self.layout)
            .field("device", &self.device_label())
            .finish()
    }
}

impl Detector {
    /// Load `model_path` and prepare the ORT session with the requested EP.
    pub fn new(model_path: &str, options: DetectorOptions) -> Result<Self> {
        init_ort()?;
        let opened =
            std::panic::catch_unwind(AssertUnwindSafe(|| Self::open(model_path, &options)));
        match opened {
            Ok(result) => result,
            Err(payload) => Err(Error::Ort(panic_message(payload).into())),
        }
    }

    fn open(model_path: &str, options: &DetectorOptions) -> Result<Self> {
        if !Path::new(model_path).is_file() {
            return Err(Error::Io(
                format!("model file not found: {model_path}").into(),
            ));
        }

        let mut builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        if let Some(threads) = options.num_threads {
            builder = builder.with_intra_threads(threads as usize).map_err(ort_err)?;
        }

        let mut resolved_device = options.device;
        let mut device_detail = None;
        if options.device == Device::Dml {
            // A DirectML failure (no DirectML.dll, no capable adapter, unsupported op) must not
            // make the whole detector unusable, so retry the very same options on the CPU EP
            // and record why the GPU was not used.
            match builder
                .clone()
                .with_execution_providers([DirectMLExecutionProvider::default()
                    .build()
                    .error_on_failure()])
            {
                Ok(with_dml) => builder = with_dml,
                Err(e) => {
                    resolved_device = Device::Cpu;
                    device_detail = Some(format!("cpu (dml unavailable: {e})"));
                }
            }
        }

        let session = builder
            .commit_from_file(Path::new(model_path))
            .map_err(ort_err)?;

        let input = session
            .inputs
            .first()
            .ok_or_else(|| Error::Ort("model exposes no inputs".into()))?;
        let input_name = input.name.clone();
        let dimensions = match &input.input_type {
            ValueType::Tensor { ty, dimensions, .. } => {
                if *ty != TensorElementType::Float32 {
                    return Err(Error::Ort(
                        format!(
                            "model input `{input_name}` is {ty:?}, only float32 inputs are supported"
                        )
                        .into(),
                    ));
                }
                dimensions.clone()
            }
            other => {
                return Err(Error::Ort(
                    format!("model input `{input_name}` is not a tensor ({other:?})").into(),
                ))
            }
        };
        if dimensions.len() != 4 {
            return Err(Error::Ort(
                format!("expected an NCHW input, got shape {dimensions:?}").into(),
            ));
        }

        let (class_names, imgsz, kpt_shape, declares_pose) = {
            let metadata = session.metadata().ok();
            let names = metadata.as_ref().and_then(|m| m.custom("names").ok().flatten());
            let imgsz = metadata.as_ref().and_then(|m| m.custom("imgsz").ok().flatten());
            let kpt_shape = metadata
                .as_ref()
                .and_then(|m| m.custom("kpt_shape").ok().flatten());
            let task = metadata.as_ref().and_then(|m| m.custom("task").ok().flatten());
            (
                names.as_deref().and_then(parse_names).unwrap_or_default(),
                imgsz.as_deref().and_then(parse_imgsz),
                kpt_shape.as_deref().and_then(KptShape::parse),
                task.as_deref()
                    .is_some_and(|task| task.trim().eq_ignore_ascii_case("pose")),
            )
        };
        // `kpt_shape` is what actually drives the decoding, so it decides the task; the `task`
        // metadata alone (without a shape) still marks the model as pose, its keypoint layout is
        // then taken from the output shape at decode time.
        let task = if kpt_shape.is_some() || declares_pose {
            Task::Pose
        } else {
            Task::Detect
        };

        // Dynamic dimensions are reported as <= 0: batch is treated as 1 and the spatial size
        // falls back to the model metadata / the 640x640 default.
        let batch = if dimensions[0] > 0 {
            dimensions[0] as u32
        } else {
            1
        };
        if batch != 1 {
            // `detect_preprocessed` feeds exactly one image per call; a static batch > 1
            // would fail inside ORT with a shape error, so reject it up front.
            return Err(Error::Ort(
                format!("model declares a static batch of {batch}, only batch-1 models are supported")
                    .into(),
            ));
        }
        if dimensions[1] > 0 && dimensions[1] != 3 {
            return Err(Error::Ort(
                format!(
                    "model expects {} input channels, only 3-channel RGB is supported",
                    dimensions[1]
                )
                .into(),
            ));
        }
        let (input_height, input_width) = if dimensions[2] > 0 && dimensions[3] > 0 {
            (dimensions[2] as u32, dimensions[3] as u32)
        } else {
            imgsz.unwrap_or((INPUT_HEIGHT, INPUT_WIDTH))
        };

        let declared_outputs: Vec<(&str, &ValueType)> = session
            .outputs
            .iter()
            .map(|output| (output.name.as_str(), &output.output_type))
            .collect();
        let layout = resolve_layout(&declared_outputs)?;
        if task == Task::Pose && matches!(layout, OutputLayout::NmsApplied { .. }) {
            // The four-tensor layout has no slot for keypoints, so decoding it as pose would
            // silently drop them; refuse it instead of returning boxes without keypoints.
            return Err(Error::Ort(
                format!(
                    "pose model `{model_path}` has a torchvision four-output layout, which this \
                     library only decodes as detect ({})",
                    describe_outputs(&declared_outputs)
                )
                .into(),
            ));
        }

        Ok(Self {
            options: options.clone(),
            model_path: model_path.to_string(),
            session,
            input_name,
            input_width,
            input_height,
            batch,
            class_names,
            kpt_shape,
            task,
            layout,
            resolved_device,
            device_detail,
        })
    }

    /// Task reported in `meta.task`.
    pub fn task(&self) -> Task {
        self.task
    }

    /// Keypoint layout declared by the model metadata, if any.
    pub fn kpt_shape(&self) -> Option<KptShape> {
        self.kpt_shape
    }

    /// Value reported in `meta.device`.
    pub fn device_label(&self) -> &str {
        self.device_detail
            .as_deref()
            .unwrap_or_else(|| self.resolved_device.as_str())
    }

    /// Run inference on a file on disk (decoded through the `imgcodecs` module).
    pub fn detect_file(&mut self, image_path: &str, conf: f32, iou: f32) -> Result<DetectionResult> {
        let pre = preprocess::preprocess_file(
            Path::new(image_path),
            self.input_width,
            self.input_height,
        )?;
        self.detect_preprocessed(pre, conf, iou)
    }

    /// Run inference on an interleaved 8-bit BGR buffer of `width * height * 3` bytes.
    pub fn detect_bgr(
        &mut self,
        data: &[u8],
        width: i32,
        height: i32,
        conf: f32,
        iou: f32,
    ) -> Result<DetectionResult> {
        if width <= 0 || height <= 0 {
            return Err(Error::invalid(format!(
                "invalid image size: {width}x{height}"
            )));
        }
        let pre = preprocess::preprocess_bgr(
            data,
            width as u32,
            height as u32,
            self.input_width,
            self.input_height,
        )?;
        self.detect_preprocessed(pre, conf, iou)
    }

    /// Run inference on an interleaved 8-bit BGR buffer whose rows are `row_stride` bytes apart.
    ///
    /// `row_stride` must be at least `width * 3`; scanline padding is skipped, which is what a
    /// caller holding an OpenCV `Mat` (or any non-tightly-packed image) has.
    pub fn detect_bgr_strided(
        &mut self,
        data: &[u8],
        width: i32,
        height: i32,
        row_stride: usize,
        conf: f32,
        iou: f32,
    ) -> Result<DetectionResult> {
        if width <= 0 || height <= 0 {
            return Err(Error::invalid(format!(
                "invalid image size: {width}x{height}"
            )));
        }
        let pre = preprocess::preprocess_bgr_strided(
            data,
            width as u32,
            height as u32,
            row_stride,
            self.input_width,
            self.input_height,
        )?;
        self.detect_preprocessed(pre, conf, iou)
    }

    /// Run the session and decode whichever layout [`Self::layout`] resolved to.
    ///
    /// `iou` only reaches the standard single-output decoder: the end-to-end layout and the
    /// four-output `torchvision` layout both arrive with the NMS already applied inside the
    /// graph.
    fn detect_preprocessed(
        &mut self,
        mut pre: Preprocessed,
        conf: f32,
        iou: f32,
    ) -> Result<DetectionResult> {
        let conf_threshold = if conf > 0.0 {
            conf
        } else {
            DEFAULT_CONF_THRESHOLD
        };
        let iou_threshold = if iou > 0.0 {
            iou
        } else {
            DEFAULT_IOU_THRESHOLD
        };

        let shape = [
            self.batch as i64,
            3,
            self.input_height as i64,
            self.input_width as i64,
        ];
        let tensor = Tensor::from_array((shape, std::mem::take(&mut pre.tensor)))
            .map_err(ort_err)?;

        let started = Instant::now();
        let outputs = self
            .session
            .run(vec![(self.input_name.as_str(), tensor)])
            .map_err(ort_err)?;
        let inference_ms = started.elapsed().as_secs_f64() as f32 * 1000.0;

        if outputs.len() == 0 {
            return Err(Error::Ort("session produced no outputs".into()));
        }
        let detections = match &self.layout {
            OutputLayout::Single => {
                let output = outputs.get("output0").unwrap_or_else(|| &outputs[0]);
                let (dims, raw) = output
                    .try_extract_raw_tensor::<f32>()
                    .map_err(ort_err)?;
                postprocess::decode_output_with_kpt(
                    raw,
                    dims,
                    &pre,
                    &self.class_names,
                    conf_threshold,
                    iou_threshold,
                    self.kpt_shape,
                )?
            }
            OutputLayout::NmsApplied {
                num,
                boxes,
                scores,
                classes,
            } => {
                let missing =
                    |name: &str| Error::Ort(format!("session output `{name}` is missing").into());
                let num_value = outputs.get(num).ok_or_else(|| missing(num))?;
                let boxes_value = outputs.get(boxes).ok_or_else(|| missing(boxes))?;
                let scores_value = outputs.get(scores).ok_or_else(|| missing(scores))?;
                let classes_value = outputs.get(classes).ok_or_else(|| missing(classes))?;

                let num_detections = extract_num_detections(num_value)?;
                let (_, boxes_raw) = boxes_value
                    .try_extract_raw_tensor::<f32>()
                    .map_err(|e| {
                        Error::Ort(format!("output `{boxes}` is not float32: {e}").into())
                    })?;
                let (_, scores_raw) = scores_value
                    .try_extract_raw_tensor::<f32>()
                    .map_err(|e| {
                        Error::Ort(format!("output `{scores}` is not float32: {e}").into())
                    })?;

                // `detection_classes` carries whatever dtype the exporter picked for the labels.
                if let Ok((_, raw)) = classes_value.try_extract_raw_tensor::<f32>() {
                    postprocess::decode_nms_applied::<f32>(
                        num_detections,
                        boxes_raw,
                        scores_raw,
                        raw,
                        &pre,
                        &self.class_names,
                        conf_threshold,
                    )?
                } else if let Ok((_, raw)) = classes_value.try_extract_raw_tensor::<i32>() {
                    postprocess::decode_nms_applied::<i32>(
                        num_detections,
                        boxes_raw,
                        scores_raw,
                        raw,
                        &pre,
                        &self.class_names,
                        conf_threshold,
                    )?
                } else if let Ok((_, raw)) = classes_value.try_extract_raw_tensor::<i64>() {
                    postprocess::decode_nms_applied::<i64>(
                        num_detections,
                        boxes_raw,
                        scores_raw,
                        raw,
                        &pre,
                        &self.class_names,
                        conf_threshold,
                    )?
                } else {
                    return Err(Error::Ort(
                        format!(
                            "output `{classes}` must be a float32, int32 or int64 tensor, got {:?}",
                            classes_value.dtype()
                        )
                        .into(),
                    ));
                }
            }
        };

        Ok(DetectionResult {
            detections,
            width: pre.src_width as i32,
            height: pre.src_height as i32,
            device: self.resolved_device,
            device_detail: self.device_detail.clone(),
            inference_ms,
            task: self.task,
        })
    }

    /// Decode raw model outputs produced *outside* this library (the caller ran the ONNX session
    /// itself) into the same [`DetectionResult`] the detect entry points return.
    ///
    /// The caller's preprocessing must match this library's letterbox — resize preserving the
    /// aspect ratio into `input_width x input_height` with `114` grey padding centred in the
    /// canvas, BGR→RGB, `÷255`. From `src_width`/`src_height` the geometry is then rebuilt with
    /// [`preprocess::letterbox_params`], i.e. exactly the mapping [`Self::detect_file`] uses.
    ///
    /// `tensors` follows the model's output layout: one tensor for `OutputLayout::Single` (decoded
    /// by shape, pose included) or the four `torchvision` tensors for `OutputLayout::NmsApplied`.
    /// The four are matched by the names the model declares, then by role names, then by shape
    /// (see [`select_four_views`]). `conf`/`iou` `<= 0` select the same defaults as the FFI.
    ///
    /// `meta.inference_ms` is `0.0`: no inference happened here.
    pub fn decode_external(
        &self,
        tensors: &[TensorView<'_>],
        src_width: i32,
        src_height: i32,
        conf: f32,
        iou: f32,
    ) -> Result<DetectionResult> {
        if src_width <= 0 || src_height <= 0 {
            return Err(Error::invalid(format!(
                "invalid source size: {src_width}x{src_height}"
            )));
        }
        let pre = preprocess::geometry(
            src_width as u32,
            src_height as u32,
            self.input_width,
            self.input_height,
        )?;
        let conf_threshold = if conf > 0.0 {
            conf
        } else {
            DEFAULT_CONF_THRESHOLD
        };
        let iou_threshold = if iou > 0.0 {
            iou
        } else {
            DEFAULT_IOU_THRESHOLD
        };

        let detections = match &self.layout {
            OutputLayout::Single => {
                if tensors.len() != 1 {
                    return Err(Error::invalid(format!(
                        "this model has a single output, expected 1 tensor, got {}",
                        tensors.len()
                    )));
                }
                let tensor = &tensors[0];
                let raw = tensor.floats()?;
                postprocess::decode_output_with_kpt(
                    &raw,
                    tensor.dims,
                    &pre,
                    &self.class_names,
                    conf_threshold,
                    iou_threshold,
                    self.kpt_shape,
                )?
            }
            OutputLayout::NmsApplied {
                num,
                boxes,
                scores,
                classes,
            } => {
                let selected = select_four_views(
                    tensors,
                    [num.as_str(), boxes.as_str(), scores.as_str(), classes.as_str()],
                )?;
                let [num_view, boxes_view, scores_view, classes_view] =
                    std::array::from_fn::<_, 4, _>(|slot| &tensors[selected[slot]]);
                let num_detections = num_view.scalar_i64()?;
                let boxes_raw = boxes_view.floats()?;
                let scores_raw = scores_view.floats()?;
                match classes_view.dtype {
                    TensorDtype::Float32 => postprocess::decode_nms_applied::<f32>(
                        num_detections,
                        &boxes_raw,
                        &scores_raw,
                        &classes_view.data_as::<f32>()?,
                        &pre,
                        &self.class_names,
                        conf_threshold,
                    )?,
                    TensorDtype::Int32 => postprocess::decode_nms_applied::<i32>(
                        num_detections,
                        &boxes_raw,
                        &scores_raw,
                        &classes_view.data_as::<i32>()?,
                        &pre,
                        &self.class_names,
                        conf_threshold,
                    )?,
                    TensorDtype::Int64 => postprocess::decode_nms_applied::<i64>(
                        num_detections,
                        &boxes_raw,
                        &scores_raw,
                        &classes_view.data_as::<i64>()?,
                        &pre,
                        &self.class_names,
                        conf_threshold,
                    )?,
                }
            }
        };

        Ok(DetectionResult {
            detections,
            width: src_width,
            height: src_height,
            device: self.resolved_device,
            device_detail: self.device_detail.clone(),
            inference_ms: 0.0,
            task: self.task,
        })
    }
}

/// Point `ort` at the `onnxruntime.dll` that `build.rs` copies next to the cdylib.
///
/// Shared by the unit tests and the FFI tests, which need a real session.
#[cfg(test)]
pub(crate) fn point_ort_at_bundled_dll() {
    if std::env::var_os("YOLO_ORT_PATH").is_some() || std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let dll = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(profile)
        .join("onnxruntime.dll");
    if dll.is_file() {
        std::env::set_var("ORT_DYLIB_PATH", &dll);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn load_bus_detector() -> Detector {
        point_ort_at_bundled_dll();
        Detector::new(
            workspace().join("yolo26n.onnx").to_str().unwrap(),
            DetectorOptions::default(),
        )
        .expect("detector should load")
    }

    #[test]
    fn detect_bus_matches_python_reference() {
        let mut detector = load_bus_detector();
        let result = detector
            .detect_file(workspace().join("assets/bus.jpg").to_str().unwrap(), 0.25, 0.7)
            .expect("inference should succeed");

        assert_eq!(result.width, 810);
        assert_eq!(result.height, 1080);
        assert_eq!(result.detections.len(), 5, "{:?}", result.detections);

        assert!(!result.detections.is_empty());
        assert!(
            result
                .detections
                .iter()
                .all(|d| (0.0..=1.0).contains(&d.confidence)),
            "confidences must be probabilities: {:?}",
            result.detections
        );
        assert!(result.detections.iter().all(|d| d.x1 >= 0.0 && d.x2 <= 810.0));
        assert!(result.inference_ms > 0.0);

        // Same golden gate as examples/compare_reference.py: the reference JSON is
        // authoritative. Every reference detection must be reproduced with the same
        // class, IoU >= 0.99 and |dconf| < 0.01. The total count may differ by at
        // most one: a detection sitting exactly on the confidence threshold can flip
        // between execution providers (CPU vs DirectML) and across model re-exports,
        // which is expected and not a geometry bug (those shift boxes by tens of
        // pixels and fail the IoU gate instead).
        let reference: Value = serde_json::from_str(
            &std::fs::read_to_string(workspace().join("assets/reference_bus.json"))
                .expect("assets/reference_bus.json should be committed"),
        )
        .expect("reference JSON should parse");
        let expected = reference["detections"]
            .as_array()
            .expect("reference should carry a detections array");
        let mut unmatched: Vec<&Detection> = result.detections.iter().collect();
        for reference in expected {
            let class_id = reference["class_id"].as_i64().unwrap() as i32;
            let r = |key: &str| reference[key].as_f64().unwrap() as f32;
            let position = unmatched.iter().position(|detection| {
                detection.class_id == class_id
                    && postprocess::iou(
                        (detection.x1, detection.y1, detection.x2, detection.y2),
                        (r("x1"), r("y1"), r("x2"), r("y2")),
                    ) >= 0.99
            });
            let detection = position
                .map(|index| unmatched.remove(index))
                .unwrap_or_else(|| {
                    panic!("reference detection {reference} has no Rust match in {unmatched:?}")
                });
            let confidence = reference["confidence"].as_f64().unwrap() as f32;
            assert!(
                (detection.confidence - confidence).abs() < 0.01,
                "confidence {} vs reference {confidence}",
                detection.confidence
            );
        }
        assert!(
            unmatched.len() <= 1,
            "{} unmatched extra detections: {unmatched:?}",
            unmatched.len()
        );
    }

    #[test]
    fn detect_bgr_accepts_raw_buffers() {
        let mut detector = load_bus_detector();
        let (width, height) = (64i32, 48i32);
        let data = vec![0u8; (width * height * 3) as usize];
        let result = detector
            .detect_bgr(&data, width, height, 0.0, 0.0)
            .expect("inference should succeed");
        assert_eq!((result.width, result.height), (width, height));
        assert!(result.detections.is_empty(), "{:?}", result.detections);

        let err = detector
            .detect_bgr(&data, 0, height, 0.25, 0.7)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    }

    #[test]
    fn cpu_execution_provider_path_works() {
        point_ort_at_bundled_dll();
        let mut detector = Detector::new(
            workspace().join("yolo26n.onnx").to_str().unwrap(),
            DetectorOptions {
                device: Device::Cpu,
                num_threads: Some(2),
            },
        )
        .expect("cpu detector should load");
        assert_eq!(detector.device_label(), "cpu");

        let result = detector
            .detect_file(workspace().join("assets/bus.jpg").to_str().unwrap(), 0.25, 0.7)
            .expect("inference should succeed");
        assert_eq!(result.device, Device::Cpu);
        assert_eq!(result.detections.len(), 5, "{:?}", result.detections);
        assert!(result.to_json().contains("\"device\":\"cpu\""));
    }

    #[test]
    fn options_defaults_and_parsing() {
        assert_eq!(DetectorOptions::from_json(None).unwrap(), DetectorOptions::default());
        assert_eq!(
            DetectorOptions::from_json(Some("{\"device\":\"cpu\",\"num_threads\":4}")).unwrap(),
            DetectorOptions {
                device: Device::Cpu,
                num_threads: Some(4),
            }
        );
        assert!(DetectorOptions::from_json(Some("{\"device\":\"tpu\"}")).is_err());
    }

    #[test]
    fn json_schema_is_stable() {
        let result = DetectionResult {
            detections: vec![Detection {
                x1: 1.5,
                y1: 2.0,
                x2: 3.0,
                y2: 4.0,
                confidence: 0.5,
                class_id: 7,
                class_name: "truck".to_string(),
                keypoints: None,
            }],
            width: 64,
            height: 32,
            device: Device::Cpu,
            device_detail: None,
            inference_ms: 1.5,
            task: Task::Detect,
        };
        let parsed: Value = serde_json::from_str(&result.to_json()).unwrap();
        assert_eq!(parsed["count"], 1);
        assert_eq!(parsed["detections"][0]["class_name"], "truck");
        assert_eq!(parsed["meta"]["device"], "cpu");
        assert_eq!(parsed["meta"]["width"], 64);
        // detect results carry no keypoints, and every result names its task
        assert!(parsed["detections"][0].get("keypoints").is_none());
        assert_eq!(parsed["meta"]["task"], "detect");
    }

    #[test]
    fn pose_json_carries_keypoints_and_task() {
        let keypoints = vec![
            Keypoint {
                x: 10.0,
                y: 20.0,
                visibility: Some(0.875),
            },
            Keypoint {
                x: 30.1234,
                y: 40.9876,
                visibility: None,
            },
        ];
        let result = DetectionResult {
            detections: vec![Detection {
                x1: 1.0,
                y1: 2.0,
                x2: 3.0,
                y2: 4.0,
                confidence: 0.75,
                class_id: 0,
                class_name: "person".to_string(),
                keypoints: Some(keypoints),
            }],
            width: 64,
            height: 32,
            device: Device::Cpu,
            device_detail: None,
            inference_ms: 0.0,
            task: Task::Pose,
        };
        let parsed: Value = serde_json::from_str(&result.to_json()).unwrap();
        assert_eq!(parsed["meta"]["task"], "pose");
        let points = parsed["detections"][0]["keypoints"]
            .as_array()
            .expect("pose detections carry a keypoints array");
        // `(K, 3)` points are [x, y, v], rounded to 3 decimals; a missing visibility is
        // emitted as a plain [x, y] pair rather than a null
        assert_eq!(points[0], serde_json::json!([10.0, 20.0, 0.875]));
        assert_eq!(points[1], serde_json::json!([30.123, 40.988]));
    }

    #[test]
    fn parse_imgsz_reads_metadata() {
        assert_eq!(parse_imgsz("[640, 640]"), Some((640, 640)));
        assert_eq!(parse_imgsz("[480, 640]"), Some((480, 640)));
        assert_eq!(parse_imgsz("640"), Some((640, 640)));
        assert_eq!(parse_imgsz("[]"), None);
    }

    fn tensor(ty: TensorElementType, dimensions: Vec<i64>) -> ValueType {
        let dimension_symbols = vec![None; dimensions.len()];
        ValueType::Tensor {
            ty,
            dimensions,
            dimension_symbols,
        }
    }

    fn layout_of(outputs: &[(&str, ValueType)]) -> Result<OutputLayout> {
        resolve_layout(&outputs.iter().map(|(name, ty)| (*name, ty)).collect::<Vec<_>>())
    }

    fn four_output_layout() -> OutputLayout {
        OutputLayout::NmsApplied {
            num: "num_detections".to_string(),
            boxes: "detection_boxes".to_string(),
            scores: "detection_scores".to_string(),
            classes: "detection_classes".to_string(),
        }
    }

    #[test]
    fn single_output_models_use_the_single_layout() {
        let layout = layout_of(&[(
            "output0",
            tensor(TensorElementType::Float32, vec![1, 84, 8400]),
        )])
        .unwrap();
        assert_eq!(layout, OutputLayout::Single);

        // ... and so does the bundled detector
        let detector = load_bus_detector();
        assert_eq!(detector.layout, OutputLayout::Single);
    }

    #[test]
    fn four_output_models_are_matched_by_name() {
        let layout = layout_of(&[
            ("num_detections", tensor(TensorElementType::Int32, vec![1, 1])),
            ("detection_boxes", tensor(TensorElementType::Float32, vec![1, 100, 4])),
            ("detection_scores", tensor(TensorElementType::Float32, vec![1, 100])),
            ("detection_classes", tensor(TensorElementType::Int64, vec![1, 100])),
        ])
        .unwrap();
        assert_eq!(layout, four_output_layout());

        // matching is case-insensitive and tolerates other prefixes/suffixes
        let layout = layout_of(&[
            ("OutputNum", tensor(TensorElementType::Float32, vec![1, 1])),
            ("NMS_Boxes", tensor(TensorElementType::Float32, vec![1, 16, 4])),
            ("NMS_Scores", tensor(TensorElementType::Float32, vec![1, 16])),
            ("NMS_Labels", tensor(TensorElementType::Float32, vec![1, 16])),
        ])
        .unwrap();
        assert_eq!(
            layout,
            OutputLayout::NmsApplied {
                num: "OutputNum".to_string(),
                boxes: "NMS_Boxes".to_string(),
                scores: "NMS_Scores".to_string(),
                classes: "NMS_Labels".to_string(),
            }
        );
    }

    #[test]
    fn four_output_layout_falls_back_to_the_leftover_name() {
        // `count` matches no needle; it is the only name left, so it must be `num_detections`
        let layout = layout_of(&[
            ("count", tensor(TensorElementType::Int64, vec![1, 1])),
            ("boxes", tensor(TensorElementType::Float32, vec![1, 8, 4])),
            ("scores", tensor(TensorElementType::Float32, vec![1, 8])),
            ("classes", tensor(TensorElementType::Int32, vec![1, 8])),
        ])
        .unwrap();
        assert_eq!(
            layout,
            OutputLayout::NmsApplied {
                num: "count".to_string(),
                boxes: "boxes".to_string(),
                scores: "scores".to_string(),
                classes: "classes".to_string(),
            }
        );
    }

    #[test]
    fn unresolvable_outputs_are_rejected_with_the_actual_names() {
        let two = layout_of(&[
            ("output0", tensor(TensorElementType::Float32, vec![1, 84, 8400])),
            ("output1", tensor(TensorElementType::Float32, vec![1, 32])),
        ])
        .unwrap_err();
        assert!(two.to_string().contains("unsupported output count 2"), "{two}");
        assert!(two.to_string().contains("output1"), "{two}");

        // four outputs, but two of the names mean nothing to the heuristics
        let ambiguous = layout_of(&[
            ("boxes", tensor(TensorElementType::Float32, vec![1, 8, 4])),
            ("scores", tensor(TensorElementType::Float32, vec![1, 8])),
            ("mystery_a", tensor(TensorElementType::Float32, vec![1, 8])),
            ("mystery_b", tensor(TensorElementType::Float32, vec![1, 8])),
        ])
        .unwrap_err();
        assert!(ambiguous.to_string().contains("classes"), "{ambiguous}");
        assert!(ambiguous.to_string().contains("mystery_a"), "{ambiguous}");

        // boxes/scores must be float32
        let wrong_dtype = layout_of(&[
            ("num_detections", tensor(TensorElementType::Int32, vec![1, 1])),
            ("detection_boxes", tensor(TensorElementType::Int64, vec![1, 8, 4])),
            ("detection_scores", tensor(TensorElementType::Float32, vec![1, 8])),
            ("detection_classes", tensor(TensorElementType::Int32, vec![1, 8])),
        ])
        .unwrap_err();
        assert!(wrong_dtype.to_string().contains("must be float32"), "{wrong_dtype}");
    }

    /// `bus.jpg` as a tightly packed BGR8 buffer, as if decoded by the caller.
    fn bus_bgr() -> (Vec<u8>, i32, i32) {
        use opencv::prelude::{MatTraitConst, MatTraitConstManual};
        let path = workspace().join("assets/bus.jpg");
        let mat = opencv::imgcodecs::imread(
            path.to_str().unwrap(),
            opencv::imgcodecs::IMREAD_COLOR,
        )
        .expect("bus.jpg should load");
        let data = mat.data_bytes().expect("a decoded Mat is continuous").to_vec();
        (data, mat.cols(), mat.rows())
    }

    #[test]
    fn detect_bgr_strided_matches_tight_buffers() {
        let mut detector = load_bus_detector();
        let (bytes, width, height) = bus_bgr();
        let row_bytes = width as usize * 3;
        // the same pixels, with 16 bytes of scanline padding after every row
        let stride = row_bytes + 16;
        let mut padded = vec![0u8; height as usize * stride];
        for row in 0..height as usize {
            let from = row * row_bytes;
            let to = row * stride;
            padded[to..to + row_bytes].copy_from_slice(&bytes[from..from + row_bytes]);
        }

        let tight = detector
            .detect_bgr(&bytes, width, height, 0.25, 0.7)
            .expect("tight buffer should decode");
        let strided = detector
            .detect_bgr_strided(&padded, width, height, stride, 0.25, 0.7)
            .expect("strided buffer should decode");
        assert_eq!(tight.detections.len(), 5, "{:?}", tight.detections);
        assert_eq!(tight.detections, strided.detections);
        assert_eq!((strided.width, strided.height), (width, height));

        // a stride that does not cover a full row is rejected
        let err = detector
            .detect_bgr_strided(&padded, width, height, row_bytes - 1, 0.25, 0.7)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
        // ... and so is a buffer that stops short of the last row
        let short = &padded[..padded.len() - stride];
        let err = detector
            .detect_bgr_strided(short, width, height, stride, 0.25, 0.7)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    }

    /// Run the session on `image` and return the first output as `(dims, values)`.
    fn raw_output(detector: &mut Detector, image: &Path) -> (Vec<i64>, Vec<f32>) {
        let pre = preprocess::preprocess_file(image, detector.input_width, detector.input_height)
            .expect("preprocessing should succeed");
        let shape = [1, 3, detector.input_height as i64, detector.input_width as i64];
        let tensor = Tensor::from_array((shape, pre.tensor)).expect("input tensor");
        let outputs = detector
            .session
            .run(vec![(detector.input_name.as_str(), tensor)])
            .expect("the session should run");
        let output = outputs.get("output0").unwrap_or_else(|| &outputs[0]);
        let (dims, raw) = output
            .try_extract_raw_tensor::<f32>()
            .expect("the output should be float32");
        (dims.to_vec(), raw.to_vec())
    }

    /// Borrow `raw` as the raw bytes of a float32 tensor view.
    fn f32_view<'a>(name: &'a str, dims: &'a [i64], raw: &'a [f32]) -> TensorView<'a> {
        // SAFETY: the slice is read-only for the lifetime of the view and `f32` has no padding.
        let data = unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<u8>(), raw.len() * 4) };
        TensorView {
            name: Some(name),
            data,
            dims,
            dtype: TensorDtype::Float32,
        }
    }

    /// The pose model is exported by `scripts/fetch-test-assets.ps1` and not committed.
    fn pose_model_path() -> Option<std::path::PathBuf> {
        let path = workspace().join("yolo26n-pose.onnx");
        path.is_file().then_some(path)
    }

    #[test]
    fn decode_external_reproduces_detect_file() {
        let mut detector = load_bus_detector();
        let image = workspace().join("assets/bus.jpg");
        let direct = detector
            .detect_file(image.to_str().unwrap(), 0.25, 0.7)
            .expect("inference should succeed");
        let (dims, raw) = raw_output(&mut detector, &image);

        let external = detector
            .decode_external(
                &[f32_view("output0", &dims, &raw)],
                direct.width,
                direct.height,
                0.25,
                0.7,
            )
            .expect("decoding the same tensor must succeed");
        assert_eq!(external.detections, direct.detections);
        assert!(!external.detections.is_empty());
        assert_eq!(external.inference_ms, 0.0);
        assert_eq!(external.task, Task::Detect);

        // `conf`/`iou` <= 0 select the same defaults as the FFI entry points
        let defaults = detector
            .decode_external(
                &[f32_view("output0", &dims, &raw)],
                direct.width,
                direct.height,
                0.0,
                0.0,
            )
            .expect("default thresholds must decode");
        let direct_defaults = detector
            .detect_file(image.to_str().unwrap(), 0.0, 0.0)
            .expect("inference should succeed");
        assert_eq!(defaults.detections, direct_defaults.detections);

        let parsed: Value = serde_json::from_str(&external.to_json()).unwrap();
        assert_eq!(parsed["meta"]["inference_ms"], 0.0);
        assert_eq!(parsed["meta"]["task"], "detect");
        assert!(parsed["detections"][0].get("keypoints").is_none());

        // this model has one output, so a second tensor is a caller error
        let err = detector
            .decode_external(
                &[
                    f32_view("output0", &dims, &raw),
                    f32_view("output1", &dims, &raw),
                ],
                direct.width,
                direct.height,
                0.25,
                0.7,
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    }

    #[test]
    fn decode_external_reproduces_pose_detect_file() {
        let Some(model) = pose_model_path() else {
            eprintln!("skipping: yolo26n-pose.onnx is not exported in this checkout");
            return;
        };
        point_ort_at_bundled_dll();
        let mut detector = Detector::new(model.to_str().unwrap(), DetectorOptions::default())
            .expect("the pose detector should load");
        assert_eq!(detector.task(), Task::Pose);
        assert!(
            detector.kpt_shape().is_some(),
            "a pose export carries kpt_shape metadata: {detector:?}"
        );

        let image = workspace().join("assets/bus.jpg");
        let direct = detector
            .detect_file(image.to_str().unwrap(), 0.25, 0.7)
            .expect("inference should succeed");
        assert!(
            direct.detections.iter().all(|d| d.keypoints.is_some()),
            "{:?}",
            direct.detections
        );
        let (dims, raw) = raw_output(&mut detector, &image);
        let external = detector
            .decode_external(
                &[f32_view("output0", &dims, &raw)],
                direct.width,
                direct.height,
                0.25,
                0.7,
            )
            .expect("decoding the same tensor must succeed");
        assert_eq!(external.detections, direct.detections);
        assert_eq!(external.task, Task::Pose);
        assert_eq!(detector.kpt_shape(), Some(postprocess::KptShape::COCO));
    }

    /// `[x, y(, v)]` or `{"x":.., "y":..(, "v":..)}` from a reference JSON.
    fn reference_point(value: &Value) -> Option<(f32, f32, Option<f32>)> {
        if let Some(array) = value.as_array() {
            let x = array.first()?.as_f64()? as f32;
            let y = array.get(1)?.as_f64()? as f32;
            let v = array.get(2).and_then(Value::as_f64).map(|v| v as f32);
            return Some((x, y, v));
        }
        let x = value.get("x")?.as_f64()? as f32;
        let y = value.get("y")?.as_f64()? as f32;
        let v = value.get("v").and_then(Value::as_f64).map(|v| v as f32);
        Some((x, y, v))
    }

    #[test]
    fn pose_bus_matches_python_reference() {
        let Some(model) = pose_model_path() else {
            eprintln!("skipping the pose golden gate: yolo26n-pose.onnx is not exported");
            return;
        };
        let reference_path = workspace().join("assets/reference_pose_bus.json");
        if !reference_path.is_file() {
            eprintln!(
                "skipping the pose golden gate: {} is not generated yet",
                reference_path.display()
            );
            return;
        }

        point_ort_at_bundled_dll();
        let mut detector = Detector::new(model.to_str().unwrap(), DetectorOptions::default())
            .expect("the pose detector should load");
        let result = detector
            .detect_file(workspace().join("assets/bus.jpg").to_str().unwrap(), 0.25, 0.7)
            .expect("inference should succeed");
        assert_eq!((result.width, result.height), (810, 1080));
        assert_eq!(result.task, Task::Pose);

        // Same golden gate as `detect_bus_matches_python_reference` — same class, IoU >= 0.99,
        // |dconf| < 0.01 — plus the keypoints of every matched detection: 1.5 px on x/y and
        // 0.02 on the visibility. The count may differ by one for the same reason as there.
        let reference: Value = serde_json::from_str(
            &std::fs::read_to_string(&reference_path).expect("the reference should be readable"),
        )
        .expect("the reference JSON should parse");
        let expected = reference["detections"]
            .as_array()
            .expect("the reference should carry a detections array");
        assert!(!expected.is_empty(), "the reference carries no detections");

        let mut unmatched: Vec<&Detection> = result.detections.iter().collect();
        for reference in expected {
            let class_id = reference["class_id"].as_i64().unwrap() as i32;
            let r = |key: &str| reference[key].as_f64().unwrap() as f32;
            let position = unmatched.iter().position(|detection| {
                detection.class_id == class_id
                    && postprocess::iou(
                        (detection.x1, detection.y1, detection.x2, detection.y2),
                        (r("x1"), r("y1"), r("x2"), r("y2")),
                    ) >= 0.99
            });
            let detection = position.map(|index| unmatched.remove(index)).unwrap_or_else(|| {
                panic!("reference detection {reference} has no Rust match in {unmatched:?}")
            });
            assert!(
                (detection.confidence - r("confidence")).abs() < 0.01,
                "confidence {} vs reference {}",
                detection.confidence,
                r("confidence")
            );

            let points = detection
                .keypoints
                .as_ref()
                .unwrap_or_else(|| panic!("detection {reference} matched a box without keypoints"));
            let reference_points = reference["keypoints"]
                .as_array()
                .expect("pose references carry a keypoints array");
            assert_eq!(
                points.len(),
                reference_points.len(),
                "keypoint count of the detection matching {reference}"
            );
            for (index, (point, expected_point)) in
                points.iter().zip(reference_points.iter()).enumerate()
            {
                let (x, y, v) = reference_point(expected_point).unwrap_or_else(|| {
                    panic!("reference keypoint {expected_point} is not [x, y, v]")
                });
                assert!(
                    (point.x - x).abs() <= 1.5 && (point.y - y).abs() <= 1.5,
                    "keypoint {index}: ({}, {}) vs reference ({x}, {y})",
                    point.x,
                    point.y
                );
                if let (Some(rust_visibility), Some(reference_visibility)) =
                    (point.visibility, v)
                {
                    assert!(
                        (rust_visibility - reference_visibility).abs() <= 0.02,
                        "keypoint {index} visibility {rust_visibility} vs reference \
                         {reference_visibility}"
                    );
                }
            }
        }
        assert!(
            unmatched.len() <= 1,
            "{} unmatched extra detections: {unmatched:?}",
            unmatched.len()
        );
    }

    fn bytes_of_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    fn bytes_of_i32(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    fn bytes_of_i64(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    #[test]
    fn four_output_views_are_matched_by_name_then_shape() {
        let declared = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ];
        let num = bytes_of_i32(&[2]);
        let num_i64 = bytes_of_i64(&[2]);
        let boxes = bytes_of_f32(&[0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
        let scores = bytes_of_f32(&[0.9, 0.8]);
        let classes = bytes_of_i64(&[1, 2]);
        let num_dims = [1i64, 1];
        let boxes_dims = [1i64, 2, 4];
        let vector_dims = [1i64, 2];

        // declared names, but handed over in a different order
        let views = [
            TensorView {
                name: Some("detection_classes"),
                data: &classes,
                dims: &vector_dims,
                dtype: TensorDtype::Int64,
            },
            TensorView {
                name: Some("num_detections"),
                data: &num,
                dims: &num_dims,
                dtype: TensorDtype::Int32,
            },
            TensorView {
                name: Some("detection_scores"),
                data: &scores,
                dims: &vector_dims,
                dtype: TensorDtype::Float32,
            },
            TensorView {
                name: Some("detection_boxes"),
                data: &boxes,
                dims: &boxes_dims,
                dtype: TensorDtype::Float32,
            },
        ];
        assert_eq!(select_four_views(&views, declared).unwrap(), [1, 3, 2, 0]);

        // role names are enough (other prefixes/suffixes, different case)
        let views = [
            TensorView {
                name: Some("NMS_Boxes"),
                data: &boxes,
                dims: &boxes_dims,
                dtype: TensorDtype::Float32,
            },
            TensorView {
                name: Some("OutputNum"),
                data: &num,
                dims: &num_dims,
                dtype: TensorDtype::Int32,
            },
            TensorView {
                name: Some("NMS_Labels"),
                data: &classes,
                dims: &vector_dims,
                dtype: TensorDtype::Int64,
            },
            TensorView {
                name: Some("NMS_Scores"),
                data: &scores,
                dims: &vector_dims,
                dtype: TensorDtype::Float32,
            },
        ];
        assert_eq!(select_four_views(&views, declared).unwrap(), [1, 0, 3, 2]);

        // no names at all: a scalar is `num_detections`, the rank-3 tensor ending in 4 is
        // `boxes`, and the remaining float/integer pair is `scores`/`classes`
        let views = [
            TensorView {
                name: None,
                data: &classes,
                dims: &vector_dims,
                dtype: TensorDtype::Int32,
            },
            TensorView {
                name: None,
                data: &scores,
                dims: &vector_dims,
                dtype: TensorDtype::Float32,
            },
            TensorView {
                name: None,
                data: &boxes,
                dims: &boxes_dims,
                dtype: TensorDtype::Float32,
            },
            TensorView {
                name: None,
                data: &num_i64,
                dims: &num_dims,
                dtype: TensorDtype::Int64,
            },
        ];
        assert_eq!(select_four_views(&views, declared).unwrap(), [3, 2, 1, 0]);

        // two indistinguishable float matrices: refuse to guess
        let err = select_four_views(
            &[
                TensorView {
                    name: None,
                    data: &num,
                    dims: &num_dims,
                    dtype: TensorDtype::Int32,
                },
                TensorView {
                    name: None,
                    data: &boxes,
                    dims: &boxes_dims,
                    dtype: TensorDtype::Float32,
                },
                TensorView {
                    name: Some("mystery_a"),
                    data: &scores,
                    dims: &vector_dims,
                    dtype: TensorDtype::Float32,
                },
                TensorView {
                    name: Some("mystery_b"),
                    data: &scores,
                    dims: &vector_dims,
                    dtype: TensorDtype::Float32,
                },
            ],
            declared,
        )
        .unwrap_err();
        assert!(err.to_string().contains("scores"), "{err}");

        // three tensors cannot be a four-output layout
        let err = select_four_views(&views[..3], declared).unwrap_err();
        assert!(err.to_string().contains("expected 4 tensors"), "{err}");
    }

    #[test]
    fn tensor_views_validate_their_payload() {
        let bytes = bytes_of_f32(&[1.0, 2.0, 3.0]);
        let dims = [1i64, 3];
        let view = TensorView {
            name: Some("output0"),
            data: &bytes,
            dims: &dims,
            dtype: TensorDtype::Float32,
        };
        assert_eq!(view.elements().unwrap(), 3);
        assert_eq!(&*view.floats().unwrap(), &[1.0, 2.0, 3.0]);

        // a short payload is an error, not a panic
        let short = TensorView {
            data: &bytes[..8],
            ..view
        };
        assert!(short.floats().is_err());

        // a float tensor is not accepted where an integer one is required
        let wrong = TensorView {
            dtype: TensorDtype::Int32,
            ..view
        };
        let err = wrong.floats().unwrap_err();
        assert!(err.to_string().contains("must be float32"), "{err}");

        // `num_detections` round-trips through every dtype the exporters use
        let num_dims = [1i64, 1];
        for (dtype, bytes) in [
            (TensorDtype::Float32, bytes_of_f32(&[7.0])),
            (TensorDtype::Int32, bytes_of_i32(&[7])),
            (TensorDtype::Int64, bytes_of_i64(&[7])),
        ] {
            let view = TensorView {
                name: None,
                data: &bytes,
                dims: &num_dims,
                dtype,
            };
            assert_eq!(view.scalar_i64().unwrap(), 7, "{dtype:?}");
        }
    }
}
