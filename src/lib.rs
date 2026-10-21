//! `yolo_dml` — C ABI wrapper around OpenCV preprocessing + ONNX Runtime (DirectML) YOLO inference.
//!
//! # C API
//!
//! ```c
//! const char* yolo_version(void);
//! int64_t yolo_create(const char* model_path, const char* options_json);
//! char* yolo_detect_file(int64_t handle, const char* image_path, float conf, float iou);
//! char* yolo_detect_bgr(int64_t handle, const uint8_t* data, int32_t width, int32_t height, float conf, float iou);
//! void yolo_free_string(char* s);
//! void yolo_destroy(int64_t handle);
//! const char* yolo_last_error(void);
//! ```
//!
//! All C strings are UTF-8 and NUL-terminated.
//!
//! # JSON schema returned by `yolo_detect_file` / `yolo_detect_bgr`
//!
//! ```json
//! {
//!   "count": 2,
//!   "detections": [
//!     {"x1": 12.5, "y1": 40.0, "x2": 220.0, "y2": 380.25,
//!      "confidence": 0.87, "class_id": 0, "class_name": "person"}
//!   ],
//!   "meta": {"width": 1280, "height": 720, "device": "dml", "inference_ms": 12.34}
//! }
//! ```
//!
//! Box coordinates are in source-image pixels, top-left origin, `x2`/`y2` exclusive.
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

use detector::{Detector, DetectorOptions};
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

/// Release a string previously returned by [`yolo_detect_file`] / [`yolo_detect_bgr`].
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
