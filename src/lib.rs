//! `yolo_dml` — C ABI wrapper around OpenCV preprocessing + ONNX Runtime (DirectML) YOLO inference.
//!
//! # C API
//!
//! ```c
//! const char* yolo_version(void);
//! int64_t yolo_create(const char* model_path, const char* options_json);
//! char* yolo_detect_file(int64_t handle, const char* image_path, float conf, float iou);
//! char* yolo_detect_bgr(int64_t handle, const uint8_t* data, int32_t width, int32_t height, float conf, float iou);
//! char* yolo_detect_bgr_strided(int64_t handle, const uint8_t* data, int32_t width, int32_t height,
//!                               int32_t row_stride_bytes, float conf, float iou);
//!
//! typedef struct YoloTensorView {
//!     const char* name;      // optional, NUL-terminated UTF-8
//!     const uint8_t* data;   // raw tensor payload
//!     const int64_t* dims;   // ndim shape entries, batch first
//!     int32_t ndim;
//!     int32_t dtype;         // 0 = float32, 1 = int32, 2 = int64
//! } YoloTensorView;
//!
//! char* yolo_decode_tensors(int64_t handle, const YoloTensorView* tensors, int32_t count,
//!                           int32_t src_width, int32_t src_height, float conf, float iou);
//! void yolo_free_string(char* s);
//! void yolo_destroy(int64_t handle);
//! const char* yolo_last_error(void);
//! ```
//!
//! All C strings are UTF-8 and NUL-terminated.
//!
//! # JSON schema returned by `yolo_detect_file` / `yolo_detect_bgr` / `yolo_detect_bgr_strided`
//!
//! ```json
//! {
//!   "count": 2,
//!   "detections": [
//!     {"x1": 12.5, "y1": 40.0, "x2": 220.0, "y2": 380.25,
//!      "confidence": 0.87, "class_id": 0, "class_name": "person"}
//!   ],
//!   "meta": {"width": 1280, "height": 720, "device": "dml", "inference_ms": 12.34, "task": "detect"}
//! }
//! ```
//!
//! Pose models (`task` `"pose"`) add a `keypoints` array of `[x, y, v]` triples — `[x, y]` for
//! `(K, 2)` keypoint shapes — per detection:
//!
//! ```json
//! {
//!   "count": 1,
//!   "detections": [
//!     {"x1": 12.5, "y1": 40.0, "x2": 220.0, "y2": 380.25,
//!      "confidence": 0.87, "class_id": 0, "class_name": "person",
//!      "keypoints": [[120.5, 44.25, 0.93], [131.0, 47.5, 0.88]]}
//!   ],
//!   "meta": {"width": 1280, "height": 720, "device": "dml", "inference_ms": 12.34, "task": "pose"}
//! }
//! ```
//!
//! Box and keypoint coordinates are in source-image pixels, top-left origin, `x2`/`y2` exclusive;
//! keypoints are clipped to the image like the boxes, the visibility is the sigmoid'd value the
//! graph produced. [`yolo_decode_tensors`] returns the same document with `meta.inference_ms`
//! `0.0`, because the inference happened in the caller.
//!
//! OBB models (`task` `"obb"`) add a `xywhr` array (`[cx, cy, w, h, angle]`, angle in radians)
//! and the four `corners` of the rotated box per detection; `x1`..`y2` then hold the minimal
//! axis-aligned box enclosing the rotated one. OBB geometry is **not** clipped to the image
//! (mirroring the `xywh` branch of ultralytics' `scale_boxes`):
//!
//! ```json
//! {
//!   "count": 1,
//!   "detections": [
//!     {"x1": 96.0, "y1": 218.0, "x2": 176.0, "y2": 278.0,
//!      "confidence": 0.91, "class_id": 0, "class_name": "plane",
//!      "xywhr": [136.0, 248.0, 80.0, 40.0, 0.785],
//!      "corners": [[120.2, 234.2], [176.8, 262.8], [151.8, 261.8], [95.2, 233.2]]}
//!   ],
//!   "meta": {"width": 1024, "height": 768, "device": "dml", "inference_ms": 12.34, "task": "obb"}
//! }
//! ```
//!
//! Both single-output OBB layouts are supported: the standard export `(1, 4+nc+1, anchors)`
//! (`cx cy w h`, class scores, angle) and the end-to-end export `(1, N, 7)`
//! (`cx cy w h conf cls angle`). OBB models are recognised by the `task` metadata; NMS runs on
//! the rotated boxes with the probabilistic IoU (ultralytics `batch_probiou`).
//!
//! The returned string is heap-allocated by the library and must be released with
//! [`yolo_free_string`]. A `conf`/`iou` argument `<= 0` selects the defaults `0.25` / `0.7`.
//!
//! `meta.device` normally echoes the requested provider (`"dml"` / `"cpu"`); when DirectML
//! could not be initialised the session falls back to the CPU provider and the field reads
//! `"cpu (dml unavailable: <reason>)"`.
//!
//! # ONNX Runtime binary
//!
//! ONNX Runtime is loaded dynamically. `build.rs` copies `onnxruntime.dll` (+ `DirectML.dll`)
//! next to this library, which is where the C ABI callers load from; set `YOLO_ORT_PATH` to an
//! explicit path to override the search (it is passed to `ort::init_from`).
//!
//! Detectors are kept in a process-wide table: [`yolo_create`] returns a positive handle and
//! every other entry point looks it up, so callers may create several detectors (different
//! models or devices) and use them concurrently — each handle is serialised by its own mutex.

pub mod detector;
pub mod error;
pub mod postprocess;
pub mod preprocess;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use detector::{Detector, DetectorOptions, TensorDtype, TensorView};
use error::{Error, Result};
use once_cell::sync::Lazy;

/// Semver string reported by [`yolo_version`].
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const VERSION_CSTR: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
const NO_ERROR_CSTR: &[u8] = b"no error\0";

type SharedDetector = Arc<Mutex<Detector>>;

/// Handle -> detector table shared by all FFI entry points.
static DETECTORS: Lazy<RwLock<HashMap<i64, SharedDetector>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Next handle to hand out; handles start at `1` so `0` can mean "failure".
static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);

thread_local! {
    /// Last error message produced on this thread. `None` means "no error".
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn record<T>(result: Result<T>) -> Option<T> {
    match result {
        Ok(value) => {
            clear_last_error();
            Some(value)
        }
        Err(err) => {
            set_last_error(err.to_string());
            None
        }
    }
}

fn registry_poisoned(what: &str) -> Error {
    Error::Ort(format!("detector registry is poisoned while {what}").into())
}

/// Register `detector` under a fresh handle.
fn insert_detector(detector: Detector) -> Result<i64> {
    let shared = Arc::new(Mutex::new(detector));
    let mut table = DETECTORS.write().map_err(|_| registry_poisoned("creating"))?;
    for _ in 0..i64::MAX {
        let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        if handle <= 0 {
            NEXT_HANDLE.store(1, Ordering::Relaxed);
            continue;
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = table.entry(handle) {
            slot.insert(shared);
            return Ok(handle);
        }
    }
    Err(Error::Ort("out of detector handles".into()))
}

/// Run `f` on the detector behind `handle`.
fn with_detector<T>(handle: i64, f: impl FnOnce(&mut Detector) -> Result<T>) -> Result<T> {
    if handle <= 0 {
        return Err(Error::InvalidHandle(handle));
    }
    let shared = {
        let table = DETECTORS.read().map_err(|_| registry_poisoned("reading"))?;
        table.get(&handle).cloned()
    }
    .ok_or(Error::InvalidHandle(handle))?;

    // Recover from a poisoned lock: a panic in a previous call must not disable the handle.
    let mut detector = match shared.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut detector)
}

fn to_c_string(json: String) -> Result<*mut c_char> {
    CString::new(json)
        .map(|c| c.into_raw())
        .map_err(|e| Error::Ort(format!("result JSON is not NUL-free: {e}").into()))
}

/// Borrow a required UTF-8 C string argument.
///
/// # Safety
/// `ptr` must be null or point to a NUL-terminated C string valid for the duration of the call.
unsafe fn required_str<'a>(ptr: *const c_char, arg_name: &str) -> Result<&'a str> {
    if ptr.is_null() {
        return Err(Error::invalid(format!("{arg_name} must not be null")));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|e| Error::invalid(format!("{arg_name} is not valid UTF-8: {e}")))
}

/// Borrow an optional UTF-8 C string argument.
///
/// # Safety
/// `ptr` must be null or point to a NUL-terminated C string valid for the duration of the call.
unsafe fn optional_str<'a>(ptr: *const c_char, arg_name: &str) -> Result<Option<&'a str>> {
    if ptr.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map(Some)
        .map_err(|e| Error::invalid(format!("{arg_name} is not valid UTF-8: {e}")))
}

/// Library version, e.g. `"0.1.0"`.
///
/// The returned pointer refers to static storage, it must not be freed and stays
/// valid for the lifetime of the process.
#[no_mangle]
pub extern "C" fn yolo_version() -> *const c_char {
    VERSION_CSTR.as_ptr().cast()
}

/// Create a detector for `model_path`.
///
/// `options_json` may be null/empty (defaults: `{"device":"dml"}`) or a JSON object
/// `{"device": "dml" | "cpu", "num_threads": n}` where `num_threads` is optional.
///
/// Returns a positive handle on success, or `0` on failure (see [`yolo_last_error`]).
#[no_mangle]
pub extern "C" fn yolo_create(model_path: *const c_char, options_json: *const c_char) -> i64 {
    let created = record((|| -> Result<i64> {
        // SAFETY: the caller guarantees the pointers are null or NUL-terminated UTF-8 strings.
        let model_path = unsafe { required_str(model_path, "model_path") }?;
        let options_json = unsafe { optional_str(options_json, "options_json") }?;
        let options = DetectorOptions::from_json(options_json)?;
        let detector = Detector::new(model_path, options)?;
        insert_detector(detector)
    })());
    created.unwrap_or(0)
}

/// Run detection on an image file (`png`/`jpg`/`bmp`/`webp`, decoded via `imgcodecs`).
///
/// `conf` is the confidence threshold, `iou` the NMS IoU threshold (both in `0.0..=1.0`;
/// values `<= 0` mean "use the default" of `0.25` / `0.7`).
/// Returns a heap-allocated UTF-8 JSON document (see the crate docs for the schema)
/// that must be released with [`yolo_free_string`], or `null` on failure
/// (see [`yolo_last_error`]).
#[no_mangle]
pub extern "C" fn yolo_detect_file(
    handle: i64,
    image_path: *const c_char,
    conf: f32,
    iou: f32,
) -> *mut c_char {
    let outcome = record((|| -> Result<*mut c_char> {
        if handle <= 0 {
            return Err(Error::InvalidHandle(handle));
        }
        // SAFETY: the caller guarantees the pointer is null or a NUL-terminated UTF-8 string.
        let image_path = unsafe { required_str(image_path, "image_path") }?;
        let result = with_detector(handle, |detector| detector.detect_file(image_path, conf, iou))?;
        to_c_string(result.to_json())
    })());
    outcome.unwrap_or(std::ptr::null_mut())
}

/// Run detection on an interleaved 8-bit BGR buffer of exactly `width * height * 3` bytes.
///
/// `data` is tightly packed (stride == width * 3) and stays owned by the caller; the
/// buffer is only read for the duration of the call.
/// Returns a heap-allocated UTF-8 JSON document (see the crate docs for the schema)
/// that must be released with [`yolo_free_string`], or `null` on failure
/// (see [`yolo_last_error`]).
#[no_mangle]
pub extern "C" fn yolo_detect_bgr(
    handle: i64,
    data: *const u8,
    width: i32,
    height: i32,
    conf: f32,
    iou: f32,
) -> *mut c_char {
    let outcome = record((|| -> Result<*mut c_char> {
        if handle <= 0 {
            return Err(Error::InvalidHandle(handle));
        }
        if data.is_null() {
            return Err(Error::invalid("data must not be null"));
        }
        if width <= 0 || height <= 0 {
            return Err(Error::invalid(format!(
                "invalid image size: {width}x{height}"
            )));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| Error::invalid("image size overflows usize"))?;
        // SAFETY: the caller guarantees `data` points to `width * height * 3` readable bytes.
        let bgr = unsafe { std::slice::from_raw_parts(data, expected) };
        let result =
            with_detector(handle, |detector| detector.detect_bgr(bgr, width, height, conf, iou))?;
        to_c_string(result.to_json())
    })());
    outcome.unwrap_or(std::ptr::null_mut())
}

/// Run detection on an interleaved 8-bit BGR buffer whose rows are `row_stride_bytes` apart.
///
/// `row_stride_bytes` must be at least `width * 3` (the bytes between the end of a row and the
/// start of the next one are skipped), and `data` must hold
/// `(height - 1) * row_stride_bytes + width * 3` readable bytes — which is what a caller holding
/// an OpenCV `Mat` or any padded bitmap has. The buffer stays owned by the caller and is only
/// read for the duration of the call.
/// Returns a heap-allocated UTF-8 JSON document (see the crate docs for the schema)
/// that must be released with [`yolo_free_string`], or `null` on failure
/// (see [`yolo_last_error`]).
#[no_mangle]
pub extern "C" fn yolo_detect_bgr_strided(
    handle: i64,
    data: *const u8,
    width: i32,
    height: i32,
    row_stride_bytes: i32,
    conf: f32,
    iou: f32,
) -> *mut c_char {
    let outcome = record((|| -> Result<*mut c_char> {
        if handle <= 0 {
            return Err(Error::InvalidHandle(handle));
        }
        if data.is_null() {
            return Err(Error::invalid("data must not be null"));
        }
        if width <= 0 || height <= 0 {
            return Err(Error::invalid(format!(
                "invalid image size: {width}x{height}"
            )));
        }
        if row_stride_bytes <= 0 {
            return Err(Error::invalid(format!(
                "row_stride_bytes must be positive, got {row_stride_bytes}"
            )));
        }
        let row_stride = row_stride_bytes as usize;
        let row_bytes = (width as usize)
            .checked_mul(3)
            .ok_or_else(|| Error::invalid("image size overflows usize"))?;
        if row_stride < row_bytes {
            return Err(Error::invalid(format!(
                "row_stride_bytes {row_stride} is smaller than one {width}px BGR row \
                 ({row_bytes} bytes)"
            )));
        }
        let length = (height as usize - 1)
            .checked_mul(row_stride)
            .and_then(|offset| offset.checked_add(row_bytes))
            .ok_or_else(|| Error::invalid("image size overflows usize"))?;
        // SAFETY: the caller guarantees `length` readable bytes from `data`.
        let bgr = unsafe { std::slice::from_raw_parts(data, length) };
        let result = with_detector(handle, |detector| {
            detector.detect_bgr_strided(bgr, width, height, row_stride, conf, iou)
        })?;
        to_c_string(result.to_json())
    })());
    outcome.unwrap_or(std::ptr::null_mut())
}

/// One raw output tensor handed to [`yolo_decode_tensors`] (C layout, see the crate docs).
///
/// `data` must hold `dims[0] * .. * dims[ndim-1] * sizeof(dtype)` readable bytes; all pointers
/// only need to stay valid for the duration of the [`yolo_decode_tensors`] call.
#[repr(C)]
pub struct YoloTensorView {
    /// Optional NUL-terminated UTF-8 output name (null is allowed).
    pub name: *const c_char,
    /// Tensor payload, row-major, native endian.
    pub data: *const u8,
    /// Shape, `ndim` entries, batch first.
    pub dims: *const i64,
    /// Number of shape entries.
    pub ndim: i32,
    /// Element type: `0` = float32, `1` = int32, `2` = int64.
    pub dtype: i32,
}

/// Borrow the caller's tensor descriptors as [`TensorView`]s.
///
/// # Safety
/// `tensors` must point to `count` initialised [`YoloTensorView`] values, and every `data` /
/// `dims` / `name` pointer inside them must be valid for the duration of the caller's C call.
unsafe fn tensor_views<'a>(
    tensors: *const YoloTensorView,
    count: i32,
) -> Result<Vec<TensorView<'a>>> {
    if count <= 0 {
        return Err(Error::invalid(format!(
            "tensor count must be positive, got {count}"
        )));
    }
    if tensors.is_null() {
        return Err(Error::invalid("tensors must not be null"));
    }
    let views = std::slice::from_raw_parts(tensors, count as usize);
    let mut borrowed = Vec::with_capacity(views.len());
    for (index, view) in views.iter().enumerate() {
        let label = format!("tensors[{index}]");
        if view.ndim <= 0 {
            return Err(Error::invalid(format!(
                "{label}.ndim must be positive, got {}",
                view.ndim
            )));
        }
        if view.dims.is_null() {
            return Err(Error::invalid(format!("{label}.dims must not be null")));
        }
        if view.data.is_null() {
            return Err(Error::invalid(format!("{label}.data must not be null")));
        }
        let dtype = match view.dtype {
            0 => TensorDtype::Float32,
            1 => TensorDtype::Int32,
            2 => TensorDtype::Int64,
            other => {
                return Err(Error::invalid(format!(
                    "{label}.dtype must be 0 (float32), 1 (int32) or 2 (int64), got {other}"
                )))
            }
        };
        let dims = std::slice::from_raw_parts(view.dims, view.ndim as usize);
        let elements = dims
            .iter()
            .try_fold(1usize, |acc, &dim| {
                usize::try_from(dim).ok().and_then(|dim| acc.checked_mul(dim))
            })
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{label}.dims must be positive and must not overflow: {dims:?}"
                ))
            })?;
        let bytes = elements
            .checked_mul(dtype.size())
            .ok_or_else(|| Error::invalid(format!("{label} is too large to address")))?;
        let data = std::slice::from_raw_parts(view.data, bytes);
        let name = optional_str(view.name, &format!("{label}.name"))?;
        borrowed.push(TensorView {
            name,
            data,
            dims,
            dtype,
        });
    }
    Ok(borrowed)
}

/// Decode raw model outputs that the caller produced outside this library.
///
/// Use this when the ONNX session runs in the caller (any runtime): the DLL only does the
/// decoding, confidence filtering and NMS. `tensors` must hold the model's outputs in the layout
/// described by the loaded model — one tensor for a single-output export (routed by shape,
/// pose included) or the four `torchvision` tensors, matched by name first and by shape
/// otherwise. `src_width`/`src_height` are the **source image** dimensions: the letterbox
/// geometry is rebuilt from them and from the model's input size, so the caller's preprocessing
/// must match this library's letterbox (aspect-preserving resize into `input_width x
/// input_height`, `114` grey padding, BGR→RGB, `÷255`).
///
/// Returns the same JSON document as the `yolo_detect_*` entry points — with
/// `meta.inference_ms` `0.0` — which must be released with [`yolo_free_string`], or `null` on
/// failure (see [`yolo_last_error`]).
///
/// # Safety
/// `tensors` must point to `count` valid [`YoloTensorView`] descriptors whose buffers stay alive
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn yolo_decode_tensors(
    handle: i64,
    tensors: *const YoloTensorView,
    count: i32,
    src_width: i32,
    src_height: i32,
    conf: f32,
    iou: f32,
) -> *mut c_char {
    let outcome = record((|| -> Result<*mut c_char> {
        if handle <= 0 {
            return Err(Error::InvalidHandle(handle));
        }
        let views = tensor_views(tensors, count)?;
        let result = with_detector(handle, |detector| {
            detector.decode_external(&views, src_width, src_height, conf, iou)
        })?;
        to_c_string(result.to_json())
    })());
    outcome.unwrap_or(std::ptr::null_mut())
}

/// Release a string previously returned by [`yolo_detect_file`] / [`yolo_detect_bgr`] /
/// [`yolo_detect_bgr_strided`] / [`yolo_decode_tensors`].
///
/// Passing `null` is a no-op. Passing any other pointer (a literal, a string not
/// produced by this library, or the same pointer twice) is undefined behaviour.
#[no_mangle]
pub extern "C" fn yolo_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: per the contract above `s` originates from `CString::into_raw` inside this library.
    unsafe { drop(CString::from_raw(s)) };
}

/// Destroy a detector handle created by [`yolo_create`] and release its ORT session.
///
/// Unknown or already destroyed handles are ignored.
#[no_mangle]
pub extern "C" fn yolo_destroy(handle: i64) {
    let removed = DETECTORS
        .write()
        .map(|mut table| table.remove(&handle))
        .unwrap_or(None);
    // Drop the session outside the table lock.
    drop(removed);
}

/// Message describing the last failure **on the calling thread**, or `"no error"`.
///
/// The pointer is valid until the next `yolo_*` call on the same thread.
#[no_mangle]
pub extern "C" fn yolo_last_error() -> *const c_char {
    LAST_ERROR
        .with(|slot| {
            slot.borrow()
                .as_ref()
                .map(|message| message.as_ptr().cast::<c_char>())
        })
        .unwrap_or_else(|| NO_ERROR_CSTR.as_ptr().cast())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn last_error() -> String {
        LAST_ERROR
            .with(|slot| {
                slot.borrow()
                    .as_ref()
                    .map(|message| message.to_string_lossy().into_owned())
            })
            .unwrap_or_default()
    }

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("yolo26n.onnx")
    }

    #[test]
    fn entry_points_reject_bad_arguments() {
        // an unknown handle must fail on every entry point instead of crashing
        assert!(yolo_detect_file(-7, std::ptr::null(), 0.25, 0.7).is_null());
        assert!(last_error().contains("invalid handle"), "{}", last_error());

        assert!(yolo_detect_bgr(-7, std::ptr::null(), 4, 4, 0.25, 0.7).is_null());
        assert!(yolo_detect_bgr_strided(-7, std::ptr::null(), 4, 4, 12, 0.25, 0.7).is_null());
        assert!(unsafe {
            yolo_decode_tensors(-7, std::ptr::null(), 0, 4, 4, 0.25, 0.7)
        }
        .is_null());
        assert!(last_error().contains("invalid handle"), "{}", last_error());
    }

    #[test]
    fn ffi_decodes_strided_buffers_and_external_tensors() {
        crate::detector::point_ort_at_bundled_dll();
        let model = CString::new(model_path().to_str().unwrap()).unwrap();
        let handle = yolo_create(model.as_ptr(), std::ptr::null());
        assert!(handle > 0, "yolo_create failed: {}", last_error());
        // SAFETY: `yolo_version` returns static storage.
        let version = unsafe { CStr::from_ptr(yolo_version()) }.to_str().unwrap();
        assert_eq!(version, VERSION);

        // a padded, all-black 64x48 image: no detections, but the whole FFI path runs
        let (width, height) = (64i32, 48i32);
        let row_bytes = width as usize * 3;
        let stride = row_bytes + 8;
        let padded = vec![0u8; height as usize * stride];
        let ptr = yolo_detect_bgr_strided(
            handle,
            padded.as_ptr(),
            width,
            height,
            stride as i32,
            0.25,
            0.7,
        );
        assert!(!ptr.is_null(), "strided detection failed: {}", last_error());
        // SAFETY: `ptr` comes from this library and is freed right below.
        let json = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        yolo_free_string(ptr);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["count"], 0);
        assert_eq!(parsed["meta"]["width"], 64);
        assert_eq!(parsed["meta"]["task"], "detect");

        // a stride that does not cover a full row is reported through `yolo_last_error`
        let ptr = yolo_detect_bgr_strided(
            handle,
            padded.as_ptr(),
            width,
            height,
            (row_bytes - 1) as i32,
            0.25,
            0.7,
        );
        assert!(ptr.is_null());
        assert!(last_error().contains("smaller"), "{}", last_error());

        // external tensors: one end-to-end `(1, 1, 6)` row that spans the whole 640x640 canvas
        // of the letterboxed 810x1080 source, so the decoded box is the whole image
        let raw = [80.0f32, 0.0, 560.0, 640.0, 0.9, 0.0];
        let dims = [1i64, 1, 6];
        let name = CString::new("output0").unwrap();
        let tensor = YoloTensorView {
            name: name.as_ptr(),
            data: raw.as_ptr().cast::<u8>(),
            dims: dims.as_ptr(),
            ndim: 3,
            dtype: 0,
        };
        let ptr = unsafe { yolo_decode_tensors(handle, &tensor, 1, 810, 1080, 0.25, 0.7) };
        assert!(!ptr.is_null(), "external decode failed: {}", last_error());
        let json = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        yolo_free_string(ptr);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["count"], 1);
        assert_eq!(parsed["meta"]["inference_ms"], 0.0);
        assert_eq!(parsed["detections"][0]["class_name"], "person");
        assert!(parsed["detections"][0].get("keypoints").is_none());
        let r = |key: &str| parsed["detections"][0][key].as_f64().unwrap();
        assert!((r("x1") - 0.0).abs() < 1e-3, "{parsed}");
        assert!((r("x1") - 0.0).abs() < 1e-3);
        assert!((r("y1") - 0.0).abs() < 1e-3);
        assert!((r("x2") - 810.0).abs() < 1e-3, "{parsed}");
        assert!((r("y2") - 1080.0).abs() < 1e-3, "{parsed}");

        // a dtype outside 0..=2 is rejected before anything is read
        let bad = YoloTensorView {
            dtype: 7,
            ..tensor
        };
        let ptr = unsafe { yolo_decode_tensors(handle, &bad, 1, 810, 1080, 0.25, 0.7) };
        assert!(ptr.is_null());
        assert!(last_error().contains("dtype"), "{}", last_error());

        yolo_destroy(handle);
    }
}
