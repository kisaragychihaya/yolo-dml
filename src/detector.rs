//! Detector: owns the ONNX Runtime session (DirectML or CPU) behind an opaque handle.

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::postprocess::{self, parse_names};
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
}

impl DetectionResult {
    /// Serialise to the JSON document returned by `yolo_detect_*`.
    ///
    /// Schema: `{"count":N,"detections":[{"x1":f,"y1":f,"x2":f,"y2":f,"confidence":f,
    /// "class_id":i,"class_name":s}],"meta":{"width":i,"height":i,"device":s,"inference_ms":f}}`
    pub fn to_json(&self) -> String {
        let detections: Vec<Value> = self
            .detections
            .iter()
            .map(|d| {
                serde_json::json!({
                    "x1": round3(d.x1),
                    "y1": round3(d.y1),
                    "x2": round3(d.x2),
                    "y2": round3(d.y2),
                    "confidence": round3(d.confidence),
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                })
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
            }
        })
        .to_string()
    }
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

        let (class_names, imgsz) = {
            let metadata = session.metadata().ok();
            let names = metadata.as_ref().and_then(|m| m.custom("names").ok().flatten());
            let imgsz = metadata.as_ref().and_then(|m| m.custom("imgsz").ok().flatten());
            (
                names.as_deref().and_then(parse_names).unwrap_or_default(),
                imgsz.as_deref().and_then(parse_imgsz),
            )
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

        Ok(Self {
            options: options.clone(),
            model_path: model_path.to_string(),
            session,
            input_name,
            input_width,
            input_height,
            batch,
            class_names,
            resolved_device,
            device_detail,
        })
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
        let output = outputs
            .get("output0")
            .unwrap_or_else(|| &outputs[0]);
        let (dims, raw) = output
            .try_extract_raw_tensor::<f32>()
            .map_err(ort_err)?;
        let detections = postprocess::decode_output(
            raw,
            dims,
            &pre,
            &self.class_names,
            conf_threshold,
            iou_threshold,
        )?;

        Ok(DetectionResult {
            detections,
            width: pre.src_width as i32,
            height: pre.src_height as i32,
            device: self.resolved_device,
            device_detail: self.device_detail.clone(),
            inference_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// Point `ort` at the `onnxruntime.dll` that `build.rs` copies next to the cdylib.
    fn point_ort_at_bundled_dll() {
        if std::env::var_os("YOLO_ORT_PATH").is_some() || std::env::var_os("ORT_DYLIB_PATH").is_some()
        {
            return;
        }
        let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
        let dll = workspace()
            .join("target")
            .join(profile)
            .join("onnxruntime.dll");
        if dll.is_file() {
            std::env::set_var("ORT_DYLIB_PATH", &dll);
        }
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

        let first = &result.detections[0];
        assert_eq!(first.class_id, 5);
        assert_eq!(first.class_name, "bus");
        assert!(first.confidence > 0.9, "confidence={}", first.confidence);
        assert!(
            (first.x1 - 5.58).abs() < 0.5,
            "x1={} expected ~5.58",
            first.x1
        );
        assert!(
            (first.y2 - 749.24).abs() < 0.5,
            "y2={} expected ~749.24",
            first.y2
        );
        assert!(result.detections.iter().all(|d| d.x1 >= 0.0 && d.x2 <= 810.0));
        assert!(result.inference_ms > 0.0);
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
            }],
            width: 64,
            height: 32,
            device: Device::Cpu,
            device_detail: None,
            inference_ms: 1.5,
        };
        let parsed: Value = serde_json::from_str(&result.to_json()).unwrap();
        assert_eq!(parsed["count"], 1);
        assert_eq!(parsed["detections"][0]["class_name"], "truck");
        assert_eq!(parsed["meta"]["device"], "cpu");
        assert_eq!(parsed["meta"]["width"], 64);
    }

    #[test]
    fn parse_imgsz_reads_metadata() {
        assert_eq!(parse_imgsz("[640, 640]"), Some((640, 640)));
        assert_eq!(parse_imgsz("[480, 640]"), Some((480, 640)));
        assert_eq!(parse_imgsz("640"), Some((640, 640)));
        assert_eq!(parse_imgsz("[]"), None);
    }
}
