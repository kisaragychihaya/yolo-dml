//! Unified error type for the `yolo_dml` crate.
//!
//! All fallible internals return [`Result`]; the C ABI in [`crate`] converts an
//! error into a `null` return value plus a thread-local message that can be read
//! back through `yolo_last_error()`.

use std::borrow::Cow;
use std::fmt;

/// Error type shared by every internal module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A caller supplied an unusable argument (null pointer, bad UTF-8, negative size, ...).
    InvalidArgument(Cow<'static, str>),
    /// Filesystem / IO failure.
    Io(Cow<'static, str>),
    /// `opencv` call failed (imread returned empty, unsupported depth, ...).
    OpenCv(Cow<'static, str>),
    /// ONNX Runtime failure (session creation, tensor binding, inference, ...).
    Ort(Cow<'static, str>),
    /// Handle passed to a C entry point is unknown or already destroyed.
    InvalidHandle(i64),
    /// Entry point or output layout that this build does not implement yet.
    NotImplemented(&'static str),
}

impl Error {
    /// Convenience constructor for [`Error::InvalidArgument`].
    pub fn invalid<S: Into<Cow<'static, str>>>(msg: S) -> Self {
        Error::InvalidArgument(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::Io(msg) => write!(f, "io error: {msg}"),
            Error::OpenCv(msg) => write!(f, "opencv error: {msg}"),
            Error::Ort(msg) => write!(f, "onnxruntime error: {msg}"),
            Error::InvalidHandle(handle) => write!(f, "invalid handle: {handle}"),
            Error::NotImplemented(what) => write!(f, "not implemented: {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string().into())
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
