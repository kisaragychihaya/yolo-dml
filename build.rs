use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

const ONNXRUNTIME_DLLS: [&str; 2] = ["onnxruntime.dll", "DirectML.dll"];

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set for build scripts"),
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is always set"));
    let profile_dir = profile_dir_from_out_dir(&out_dir);

    let ort_dir = manifest_dir.join("thirdparty").join("onnxruntime-dml");
    let opencv_bin = manifest_dir
        .join("thirdparty")
        .join("opencv")
        .join("build")
        .join("x64")
        .join("vc16")
        .join("bin");

    for dir in [&ort_dir, &opencv_bin] {
        println!("cargo:rerun-if-changed={}", dir.display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    let mut copied = 0usize;
    for dll in ONNXRUNTIME_DLLS {
        let src = ort_dir.join(dll);
        if !src.is_file() {
            panic!(
                "missing ONNX Runtime DirectML binary: {} (expected inside {})",
                src.display(),
                ort_dir.display()
            );
        }
        copied += copy_if_changed(&src, &profile_dir);
    }

    if !opencv_bin.is_dir() {
        panic!(
            "missing OpenCV runtime directory: {} (unpack the OpenCV 4.10.0 Windows package into thirdparty/opencv)",
            opencv_bin.display()
        );
    }
    let mut opencv_dlls: Vec<PathBuf> = fs::read_dir(&opencv_bin)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", opencv_bin.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let is_dll = path
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("dll"))
                .unwrap_or(false);
            is_dll.then_some(path)
        })
        .collect();
    opencv_dlls.sort();
    if opencv_dlls.is_empty() {
        panic!("no .dll files found in {}", opencv_bin.display());
    }
    for dll in &opencv_dlls {
        copied += copy_if_changed(dll, &profile_dir);
    }

    println!(
        "cargo:warning=copied {} runtime DLL(s) into {}",
        copied,
        profile_dir.display()
    );
}

/// `OUT_DIR` is `<target-dir>[/<triple>]/<profile>/build/<pkg>-<hash>/out`, so the
/// profile directory (`target/debug`, `target/release`, ...) is three levels up.
fn profile_dir_from_out_dir(out_dir: &Path) -> PathBuf {
    let build_dir = out_dir
        .ancestors()
        .nth(2)
        .unwrap_or_else(|| panic!("unexpected OUT_DIR layout: {}", out_dir.display()));
    assert!(
        build_dir
            .file_name()
            .map(|name| name == OsStr::new("build"))
            .unwrap_or(false),
        "unexpected OUT_DIR layout, expected .../<profile>/build/<pkg>/out, got {}",
        out_dir.display()
    );
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| panic!("unexpected OUT_DIR layout: {}", out_dir.display()))
        .to_path_buf();
    if !profile_dir.is_dir() {
        panic!(
            "resolved profile directory does not exist: {}",
            profile_dir.display()
        );
    }
    profile_dir
}

/// Returns 1 when the file was (re)written, 0 when the destination was already current.
fn copy_if_changed(src: &Path, dst_dir: &Path) -> usize {
    let name = src
        .file_name()
        .unwrap_or_else(|| panic!("source has no file name: {}", src.display()));
    let dst = dst_dir.join(name);

    let src_len = fs::metadata(src)
        .unwrap_or_else(|e| panic!("cannot stat {}: {e}", src.display()))
        .len();
    if let Ok(meta) = fs::metadata(&dst) {
        if meta.len() == src_len {
            return 0;
        }
    }
    fs::copy(src, &dst).unwrap_or_else(|e| {
        panic!(
            "failed to copy {} -> {}: {e}",
            src.display(),
            dst.display()
        )
    });
    1
}
