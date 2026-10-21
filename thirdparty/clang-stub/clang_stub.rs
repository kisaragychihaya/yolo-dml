// Minimal `clang.exe` shim.
//
// `clang-sys` (used by the `opencv` crate's binding generator) runs
// `clang -E -x c++ - -v` only to discover the compiler's default header search
// paths, and needs an executable named `clang` to exist (env `CLANG_PATH`).
//
// This machine has LLVM's `libclang.dll` (from the Python `clang` wheel) but no
// `clang` driver, so this shim answers that single query: it reports either the
// paths listed in `YOLO_CLANG_INCLUDE_PATHS` (comma or semicolon separated) or,
// when that variable is unset, the newest MSVC STL + Windows SDK include
// directories it can find. All real work (parsing OpenCV headers) is still done
// by the genuine libclang through the libclang C API - this program never parses
// anything itself.
//
// Build: rustc -O -o clang.exe clang_stub.rs

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let has_version = args.iter().any(|a| a == "--version");
    let wants_verbose = args.iter().any(|a| a == "-v" || a == "-###");
    let is_preprocess = args.iter().any(|a| a == "-E");

    if has_version {
        // Only used to render a display string; keep it aligned with the libclang
        // that actually performs the parsing.
        println!("clang version 18.1.1 (yolo_dml shim)");
        return;
    }

    if !(wants_verbose && is_preprocess) {
        return;
    }

    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "clang -cc1 version 18.1.1 (yolo_dml shim)");

    let paths = from_env().unwrap_or_else(discover_msvc);
    if paths.is_empty() {
        let _ = writeln!(stderr, "ignoring nonexistent directory \"(none configured)\"");
        return;
    }
    let _ = writeln!(stderr, "#include <...> search starts here:");
    for path in paths {
        let _ = writeln!(stderr, " {}", path.to_string_lossy().replace('\\', "/"));
    }
    let _ = writeln!(stderr, "End of search list.");
}

fn from_env() -> Option<Vec<PathBuf>> {
    let raw = env::var("YOLO_CLANG_INCLUDE_PATHS").ok()?;
    let paths: Vec<PathBuf> = raw
        .split([',', ';'])
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect();
    (!paths.is_empty()).then_some(paths)
}

fn discover_msvc() -> Vec<PathBuf> {
    let mut out = Vec::new();

    for root in [
        "C:/Program Files/Microsoft Visual Studio",
        "C:/Program Files (x86)/Microsoft Visual Studio",
    ] {
        let include = newest_subdir_path(Path::new(root), &["2022", "2019", "2017"])
            .and_then(|year| newest_subdir_path(&year, &[]))
            .map(|edition| edition.join("VC/Tools/MSVC"))
            .and_then(|tools| newest_subdir_path(&tools, &[]))
            .map(|version| version.join("include"))
            .filter(|p| p.is_dir());
        if let Some(dir) = include {
            if !out.contains(&dir) {
                out.push(dir);
            }
        }
    }

    for root in [
        "C:/Program Files (x86)/Windows Kits/10/Include",
        "C:/Program Files/Windows Kits/10/Include",
    ] {
        if let Some(sdk) = newest_subdir_path(Path::new(root), &[]) {
            for part in ["ucrt", "shared", "um"] {
                let dir = sdk.join(part);
                if dir.is_dir() {
                    out.push(dir);
                }
            }
            break;
        }
    }

    out
}

/// Newest (by name) subdirectory of `parent`, optionally only among `preferred`.
fn newest_subdir_path(parent: &Path, preferred: &[&str]) -> Option<PathBuf> {
    let read = std::fs::read_dir(parent).ok()?;
    let mut candidates: Vec<PathBuf> = read
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .filter(|path| {
            preferred.is_empty()
                || path.file_name().and_then(|n| n.to_str()).is_some_and(|name| preferred.contains(&name))
        })
        .collect();
    candidates.sort();
    candidates.pop()
}
