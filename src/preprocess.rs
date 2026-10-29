//! Image preprocessing: decode (imgcodecs) -> letterbox -> NCHW f32 RGB tensor.
//!
//! The letterbox geometry mirrors `ultralytics`' `LetterBox` so that boxes produced by
//! the model map back onto exactly the same source pixels as the Python reference.

use std::path::Path;

use opencv::core::{Mat, Scalar, Size, CV_8UC3};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};

use crate::error::{Error, Result};

/// Fallback network input width, used when the model does not fix its input shape.
pub const INPUT_WIDTH: u32 = 640;
/// Fallback network input height, used when the model does not fix its input shape.
pub const INPUT_HEIGHT: u32 = 640;

/// Value written into the letterbox padding columns/rows.
const PAD_VALUE: u8 = 114;

/// Network input plus everything needed to map boxes back to source pixels.
#[derive(Debug, Clone)]
pub struct Preprocessed {
    /// `1 x 3 x INPUT_HEIGHT x INPUT_WIDTH`, RGB, `0.0..=1.0`, row-major NCHW.
    pub tensor: Vec<f32>,
    /// Tensor width.
    pub input_width: u32,
    /// Tensor height.
    pub input_height: u32,
    /// Source image width in pixels.
    pub src_width: u32,
    /// Source image height in pixels.
    pub src_height: u32,
    /// Horizontal letterbox gain (`new_w / src_w`), undoes the horizontal resize.
    pub gain_x: f32,
    /// Vertical letterbox gain (`new_h / src_h`), undoes the vertical resize. Differs from
    /// `gain_x` by up to one rounding pixel, exactly like the separate gains in
    /// ultralytics' `scale_boxes`.
    pub gain_y: f32,
    /// Left padding (pixels) inside the letterboxed canvas.
    pub pad_x: f32,
    /// Top padding (pixels) inside the letterboxed canvas.
    pub pad_y: f32,
}

impl Preprocessed {
    /// Map one network-space point back to source-image pixel coordinates.
    ///
    /// This is the per-axis part of [`Self::to_source_box`]: subtract the padding, divide by the
    /// axis gain, then clip to `[0, src_width]` x `[0, src_height]`. Pose keypoints use it
    /// directly, so boxes and keypoints land on exactly the same pixel grid.
    pub fn to_source_point(&self, x: f32, y: f32) -> (f32, f32) {
        let unpad_x = |v: f32| (v - self.pad_x) / self.gain_x;
        let unpad_y = |v: f32| (v - self.pad_y) / self.gain_y;
        let clamp_x = |v: f32| v.clamp(0.0, self.src_width as f32);
        let clamp_y = |v: f32| v.clamp(0.0, self.src_height as f32);
        (clamp_x(unpad_x(x)), clamp_y(unpad_y(y)))
    }

    /// Map a network-space box back to source-image pixel coordinates.
    pub fn to_source_box(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32, f32) {
        let (x1, y1) = self.to_source_point(x1, y1);
        let (x2, y2) = self.to_source_point(x2, y2);
        (x1, y1, x2, y2)
    }

    /// Map a network-space center/size box back to source-image pixel coordinates.
    ///
    /// This is the `xywh` branch of ultralytics' `scale_boxes`: the padding is subtracted from
    /// the center only, all four components are divided by their axis gain, and — unlike
    /// [`Self::to_source_box`] — nothing is clipped to the image.
    pub fn to_source_xywh(&self, cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        (
            (cx - self.pad_x) / self.gain_x,
            (cy - self.pad_y) / self.gain_y,
            w / self.gain_x,
            h / self.gain_y,
        )
    }
}

/// Geometry produced by `ultralytics`' letterbox.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterBox {
    /// Width of the resized image before padding.
    pub new_w: u32,
    /// Height of the resized image before padding.
    pub new_h: u32,
    /// Left padding in pixels.
    pub pad_x: u32,
    /// Top padding in pixels.
    pub pad_y: u32,
    /// Horizontal scale factor (`new_w / src_w`).
    pub gain_x: f32,
    /// Vertical scale factor (`new_h / src_h`), may differ from `gain_x` by rounding.
    pub gain_y: f32,
}

/// Compute the letterbox geometry for a `src_w x src_h` image.
pub fn letterbox_params(src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Result<LetterBox> {
    if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
        return Err(Error::invalid(format!(
            "letterbox on degenerate size {src_w}x{src_h} -> {out_w}x{out_h}"
        )));
    }
    // Same arithmetic as ultralytics `LetterBox`: r = min(out_h/h, out_w/w), the result is
    // rounded (no stride alignment) and the odd padding pixel is pushed to the far side.
    let r = (out_h as f64 / src_h as f64).min(out_w as f64 / src_w as f64);
    let new_w = ((src_w as f64 * r).round() as u32).clamp(1, out_w);
    let new_h = ((src_h as f64 * r).round() as u32).clamp(1, out_h);
    let dw = (out_w - new_w) as f64 / 2.0;
    let dh = (out_h - new_h) as f64 / 2.0;
    let pad_x = (dw - 0.1).round().max(0.0) as u32;
    let pad_y = (dh - 0.1).round().max(0.0) as u32;
    if pad_x + new_w > out_w || pad_y + new_h > out_h {
        return Err(Error::invalid(format!(
            "letterbox padding overflow: {new_w}x{new_h} at ({pad_x},{pad_y}) inside {out_w}x{out_h}"
        )));
    }
    Ok(LetterBox {
        new_w,
        new_h,
        pad_x,
        pad_y,
        gain_x: new_w as f32 / src_w as f32,
        gain_y: new_h as f32 / src_h as f32,
    })
}

/// Letterbox geometry for a `src_w x src_h` image resized into an `out_w x out_h` canvas.
///
/// The returned [`Preprocessed`] carries no tensor: this is what a caller needs to map decoded
/// network-space output back onto its own pixels when the inference happened outside this
/// library (see [`crate::detector::Detector::decode_external`]).
pub fn geometry(src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Result<Preprocessed> {
    let lb = letterbox_params(src_w, src_h, out_w, out_h)?;
    Ok(Preprocessed {
        tensor: Vec::new(),
        input_width: out_w,
        input_height: out_h,
        src_width: src_w,
        src_height: src_h,
        gain_x: lb.gain_x,
        gain_y: lb.gain_y,
        pad_x: lb.pad_x as f32,
        pad_y: lb.pad_y as f32,
    })
}

/// Decode `image_path` and letterbox it into network input format.
///
/// The file is read with `std::fs` and decoded from memory (`imdecode`) instead of
/// `imread`: OpenCV opens files with the narrow system codepage, which cannot open
/// paths outside it (e.g. a non-ASCII directory on a Western Windows), while Rust
/// paths are full Unicode.
pub fn preprocess_file(image_path: &Path, width: u32, height: u32) -> Result<Preprocessed> {
    let bytes = std::fs::read(image_path).map_err(|e| {
        Error::Io(format!("cannot read image {}: {e}", image_path.display()).into())
    })?;
    let mat = imgcodecs::imdecode(
        &opencv::core::Vector::<u8>::from_iter(bytes),
        imgcodecs::IMREAD_COLOR,
    )
        .map_err(|e| {
            Error::OpenCv(format!("imdecode({}) failed: {e}", image_path.display()).into())
        })?;
    if mat.empty() {
        return Err(Error::OpenCv(
            format!("cannot decode image: {}", image_path.display()).into(),
        ));
    }
    letterbox_mat(&mat, width, height)
}

/// Letterbox an interleaved 8-bit BGR buffer into network input format.
pub fn preprocess_bgr(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Preprocessed> {
    if width == 0 || height == 0 {
        return Err(Error::invalid(format!(
            "invalid source size: {width}x{height}"
        )));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| Error::invalid("image size overflows usize"))?;
    if data.len() < expected {
        return Err(Error::invalid(format!(
            "bgr buffer holds {} bytes, need {expected} for {width}x{height}",
            data.len()
        )));
    }

    let mat = bgr_to_mat(data, width, height, width as usize * 3)?;
    letterbox_mat(&mat, out_w, out_h)
}

/// Letterbox an interleaved 8-bit BGR buffer whose rows are `row_stride` bytes apart.
///
/// The rows are copied one by one into a contiguous matrix, then follow exactly the path of
/// [`preprocess_bgr`]. `row_stride` must be at least `width * 3`; the bytes between the end of a
/// row and the start of the next one (e.g. an OpenCV `Mat` whose `step` exceeds `cols * 3`, or a
/// caller's scanline padding) are ignored.
pub fn preprocess_bgr_strided(
    data: &[u8],
    width: u32,
    height: u32,
    row_stride: usize,
    out_w: u32,
    out_h: u32,
) -> Result<Preprocessed> {
    if width == 0 || height == 0 {
        return Err(Error::invalid(format!(
            "invalid source size: {width}x{height}"
        )));
    }
    let row_bytes = (width as usize)
        .checked_mul(3)
        .ok_or_else(|| Error::invalid("image size overflows usize"))?;
    if row_stride < row_bytes {
        return Err(Error::invalid(format!(
            "row stride {row_stride} is smaller than one {width}px BGR row ({row_bytes} bytes)"
        )));
    }
    let required = (height as usize - 1)
        .checked_mul(row_stride)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| Error::invalid("image size overflows usize"))?;
    if data.len() < required {
        return Err(Error::invalid(format!(
            "bgr buffer holds {} bytes, need {required} for {width}x{height} with stride {row_stride}",
            data.len()
        )));
    }

    let mat = bgr_to_mat(data, width, height, row_stride)?;
    letterbox_mat(&mat, out_w, out_h)
}

/// Copy a (possibly strided) BGR8 buffer into a contiguous `height x width` matrix.
fn bgr_to_mat(data: &[u8], width: u32, height: u32, row_stride: usize) -> Result<Mat> {
    let row_bytes = width as usize * 3;
    let mut mat = Mat::new_rows_cols_with_default(
        height as i32,
        width as i32,
        CV_8UC3,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
    )
    .map_err(|e| Error::OpenCv(format!("cannot allocate {width}x{height} Mat: {e}").into()))?;
    let storage = mat
        .data_bytes_mut()
        .map_err(|e| Error::OpenCv(format!("cannot access Mat storage: {e}").into()))?;
    if row_stride == row_bytes {
        storage.copy_from_slice(&data[..row_bytes * height as usize]);
    } else {
        for row in 0..height as usize {
            let from = row * row_stride;
            let to = row * row_bytes;
            storage[to..to + row_bytes].copy_from_slice(&data[from..from + row_bytes]);
        }
    }
    Ok(mat)
}

/// Resize + pad `src` and pack it into an NCHW f32 RGB tensor.
fn letterbox_mat(src: &Mat, out_w: u32, out_h: u32) -> Result<Preprocessed> {
    if src.empty() {
        return Err(Error::OpenCv("source image is empty".into()));
    }
    if src.channels() != 3 {
        return Err(Error::OpenCv(
            format!("expected a 3-channel BGR image, got {} channels", src.channels()).into(),
        ));
    }
    let src_w = src.cols();
    let src_h = src.rows();
    if src_w <= 0 || src_h <= 0 {
        return Err(Error::OpenCv(
            format!("invalid image size: {src_w}x{src_h}").into(),
        ));
    }
    // OpenCV matrices are (rows = height, cols = width).
    let (src_w, src_h) = (src_w as u32, src_h as u32);
    let lb = letterbox_params(src_w, src_h, out_w, out_h)?;

    let mut resized = Mat::default();
    imgproc::resize(
        src,
        &mut resized,
        Size::new(lb.new_w as i32, lb.new_h as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .map_err(|e| Error::OpenCv(format!("resize failed: {e}").into()))?;
    let pixels = resized
        .data_bytes()
        .map_err(|e| Error::OpenCv(format!("cannot read resized image: {e}").into()))?;

    let new_w = lb.new_w as usize;
    let new_h = lb.new_h as usize;
    let row_bytes = new_w * 3;
    if pixels.len() < row_bytes * new_h {
        return Err(Error::OpenCv(
            format!(
                "resized image has {} bytes, expected {}",
                pixels.len(),
                row_bytes * new_h
            )
            .into(),
        ));
    }

    let canvas_w = out_w as usize;
    let plane = canvas_w * out_h as usize;
    let pad = PAD_VALUE as f32 / 255.0;
    let mut tensor = vec![pad; plane * 3];
    let (pad_x, pad_y) = (lb.pad_x as usize, lb.pad_y as usize);
    for row in 0..new_h {
        let src_row = &pixels[row * row_bytes..(row + 1) * row_bytes];
        let dst_row = (pad_y + row) * canvas_w + pad_x;
        for col in 0..new_w {
            let bgr = &src_row[col * 3..col * 3 + 3];
            let i = dst_row + col;
            // Split the packed BGR pixels into planes while swapping to RGB: this replaces
            // `cvt_color(BGR2RGB)` + a second pass, and yields the same values.
            tensor[i] = bgr[2] as f32 / 255.0;
            tensor[plane + i] = bgr[1] as f32 / 255.0;
            tensor[2 * plane + i] = bgr[0] as f32 / 255.0;
        }
    }

    Ok(Preprocessed {
        tensor,
        input_width: out_w,
        input_height: out_h,
        src_width: src_w,
        src_height: src_h,
        gain_x: lb.gain_x,
        gain_y: lb.gain_y,
        pad_x: lb.pad_x as f32,
        pad_y: lb.pad_y as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pre(src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Preprocessed {
        let lb = letterbox_params(src_w, src_h, out_w, out_h).unwrap();
        Preprocessed {
            tensor: Vec::new(),
            input_width: out_w,
            input_height: out_h,
            src_width: src_w,
            src_height: src_h,
            gain_x: lb.gain_x,
            gain_y: lb.gain_y,
            pad_x: lb.pad_x as f32,
            pad_y: lb.pad_y as f32,
        }
    }

    #[test]
    fn letterbox_matches_ultralytics() {
        let portrait = letterbox_params(810, 1080, 640, 640).unwrap();
        assert_eq!((portrait.new_w, portrait.new_h), (480, 640));
        assert_eq!((portrait.pad_x, portrait.pad_y), (80, 0));

        let landscape = letterbox_params(1280, 720, 640, 640).unwrap();
        assert_eq!((landscape.new_w, landscape.new_h), (640, 360));
        assert_eq!((landscape.pad_x, landscape.pad_y), (0, 140));

        let exact = letterbox_params(640, 640, 640, 640).unwrap();
        assert_eq!(
            (exact.new_w, exact.new_h, exact.pad_x, exact.pad_y),
            (640, 640, 0, 0)
        );
    }

    #[test]
    fn to_source_box_round_trip() {
        let p = pre(100, 200, 640, 640);
        assert_eq!((p.pad_x, p.pad_y), (160.0, 0.0));
        assert!((p.gain_x - 3.2).abs() < 1e-6);
        assert!((p.gain_y - 3.2).abs() < 1e-6);

        // full network-space box maps back onto the whole source image
        let (x1, y1, x2, y2) = p.to_source_box(160.0, 0.0, 480.0, 640.0);
        assert!((x1 - 0.0).abs() < 1e-3, "x1={x1}");
        assert!((y1 - 0.0).abs() < 1e-3, "y1={y1}");
        assert!((x2 - 100.0).abs() < 1e-3, "x2={x2}");
        assert!((y2 - 200.0).abs() < 1e-3, "y2={y2}");

        // padding is subtracted and the result never leaves the source image
        let (x1, y1, x2, y2) = p.to_source_box(0.0, 0.0, 640.0, 640.0);
        assert_eq!((x1, y1), (0.0, 0.0));
        assert!((x2 - 100.0).abs() < 1e-3 && (y2 - 200.0).abs() < 1e-3, "x2={x2} y2={y2}");
    }

    #[test]
    fn to_source_box_uses_per_axis_gains() {
        // 333x777 into 640x640 rounds new_w to 274 while new_h hits 640 exactly, so
        // gain_x != gain_y and the Y axis must be rescaled by gain_y, not gain_x.
        let lb = letterbox_params(333, 777, 640, 640).unwrap();
        assert_eq!((lb.new_w, lb.new_h), (274, 640));
        assert!((lb.gain_x - 274.0 / 333.0).abs() < 1e-6);
        assert!((lb.gain_y - 640.0 / 777.0).abs() < 1e-6);
        assert_eq!((lb.pad_x, lb.pad_y), (183, 0));

        let p = pre(333, 777, 640, 640);
        let (_, y1, _, _) = p.to_source_box(183.0, 400.0, 457.0, 400.0);
        let expect_y = 400.0 / (640.0 / 777.0);
        assert!((y1 - expect_y).abs() < 0.05, "y1={y1} expected {expect_y}");

        // the full network canvas maps back onto the whole source image on both axes
        let (x1, y1, x2, y2) = p.to_source_box(183.0, 0.0, 457.0, 640.0);
        assert!((x1 - 0.0).abs() < 1e-3, "x1={x1}");
        assert!((y1 - 0.0).abs() < 1e-3, "y1={y1}");
        assert!((x2 - 333.0).abs() < 0.5, "x2={x2}");
        assert!((y2 - 777.0).abs() < 0.5, "y2={y2}");
    }

    #[test]
    fn preprocess_bgr_letterboxes_and_pads() {
        // 3x3 all-black image into a 9x9 canvas: letterbox pads with 114 grey
        let src_w = 3u32;
        let src_h = 3u32;
        let data = vec![0u8; (src_w * src_h * 3) as usize];
        let out = preprocess_bgr(&data, src_w, src_h, 9, 9).unwrap();
        assert_eq!((out.input_width, out.input_height), (9, 9));
        assert_eq!((out.pad_x, out.pad_y), (0.0, 0.0));
        assert_eq!(out.tensor.len(), 3 * 9 * 9);
        assert!(out.tensor.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn preprocess_bgr_rejects_short_buffers() {
        let err = preprocess_bgr(&[0u8; 8], 2, 2, 4, 4).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    }
}
