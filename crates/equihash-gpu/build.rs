//! Build script for the CUDA kernel. Only runs when the `cuda` feature is
//! enabled — otherwise we emit no native code and the `cuda` module in `lib.rs`
//! is `cfg`d out.
//!
//! Discovers `nvcc` on PATH, compiles `src/kernel.cu` to a static library,
//! and links against the CUDA runtime (`libcudart`).
//!
//! Compute capability defaults to `sm_61` (Pascal, e.g. GT 1030 / GTX 10-series).
//! Override via `EQUIUM_CUDA_ARCH` env var, e.g. `EQUIUM_CUDA_ARCH=sm_75` for
//! Turing or `sm_86` for Ampere.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=EQUIUM_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        // Feature off: nothing to build, no CUDA dependency.
        return;
    }

    // Resolve nvcc.
    let nvcc = which_nvcc().unwrap_or_else(|| {
        panic!(
            "cuda feature enabled but `nvcc` not found on PATH. \
             Install the CUDA toolkit or set CUDA_PATH."
        );
    });

    let arch = env::var("EQUIUM_CUDA_ARCH").unwrap_or_else(|_| "sm_61".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_path = out_dir.join("equium_kernel.o");
    let lib_path = out_dir.join("libequium_kernel.a");

    // Compile .cu → .o
    let status = Command::new(&nvcc)
        .args([
            "-c",
            "src/kernel.cu",
            "-o",
        ])
        .arg(&obj_path)
        .args([
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-arch",
            &arch,
            "--std=c++14",
        ])
        .status()
        .expect("failed to spawn nvcc");
    if !status.success() {
        panic!("nvcc failed compiling src/kernel.cu");
    }

    // Bundle .o → .a
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("failed to spawn ar");
    if !ar_status.success() {
        panic!("ar failed packaging kernel object");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=equium_kernel");

    // Link against CUDA runtime.
    let cuda_lib_dir = find_cuda_lib_dir(&nvcc);
    if let Some(dir) = cuda_lib_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn which_nvcc() -> Option<PathBuf> {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let p = PathBuf::from(cuda_path).join("bin").join("nvcc");
        if p.exists() {
            return Some(p);
        }
    }
    // Fall back to PATH lookup via `which`-like shell behavior.
    let out = Command::new("sh")
        .arg("-c")
        .arg("command -v nvcc")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn find_cuda_lib_dir(nvcc: &std::path::Path) -> Option<PathBuf> {
    // nvcc typically lives at $CUDA_PATH/bin/nvcc. Lib is at $CUDA_PATH/lib64.
    let parent = nvcc.parent()?.parent()?;
    let lib64 = parent.join("lib64");
    if lib64.exists() {
        return Some(lib64);
    }
    let lib = parent.join("lib");
    if lib.exists() {
        return Some(lib);
    }
    // On Debian/Ubuntu, /usr/bin/nvcc may not have a sibling lib64 — cudart
    // is at /usr/lib/x86_64-linux-gnu instead. The linker's default search
    // should find it; emit no extra hint.
    None
}
