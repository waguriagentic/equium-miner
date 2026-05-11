//! CUDA-backed implementation. Compiled in only when the `cuda` feature is
//! enabled (gated in `lib.rs`).
//!
//! Wraps the C ABI exposed by `src/kernel.cu`:
//!   - `equium_init(batch_size, &mut ctx) -> int`
//!   - `equium_destroy(ctx)`
//!   - `equium_run_batch(ctx, input, nonces, batch, out_valid, out_indices) -> int`
//!
//! Error codes match `cudaError_t` semantics — 0 is success, non-zero is
//! propagated as `GpuError::Cuda(code, "context")`.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uchar, c_uint};

use equihash_core::challenge::I_LEN;

use crate::{GpuCandidate, GpuError};

#[link(name = "equium_kernel", kind = "static")]
extern "C" {
    fn equium_init(batch_size: c_int, ctx_out: *mut *mut c_void) -> c_int;
    fn equium_destroy(ctx: *mut c_void);
    fn equium_run_batch(
        ctx: *mut c_void,
        input: *const c_uchar, // [I_LEN]
        nonces: *const c_uchar, // [batch * 32]
        batch_size: c_int,
        out_valid: *mut c_uchar,  // [batch]
        out_indices: *mut c_uint, // [batch * 32]
    ) -> c_int;
}

pub struct Context {
    handle: *mut c_void,
    batch_size: usize,
}

// The CUDA context is owned by this struct; sending across threads is fine
// as long as we serialize calls (we hold &mut self for run_batch).
unsafe impl Send for Context {}

impl Context {
    pub fn new(batch_size: usize) -> Result<Self, GpuError> {
        if batch_size == 0 {
            return Err(GpuError::UnsupportedParams);
        }
        let mut handle: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { equium_init(batch_size as c_int, &mut handle) };
        if rc != 0 {
            return Err(map_cuda_err(rc, "equium_init"));
        }
        if handle.is_null() {
            return Err(GpuError::OutOfMemory);
        }
        Ok(Self {
            handle,
            batch_size,
        })
    }

    pub fn run_batch(
        &mut self,
        input: &[u8; I_LEN],
        nonces: &[[u8; 32]],
    ) -> Result<Vec<GpuCandidate>, GpuError> {
        if nonces.is_empty() {
            return Ok(Vec::new());
        }
        if nonces.len() > self.batch_size {
            // Caller should chunk; we silently truncate here, but in practice
            // the CLI miner sizes its batches to match.
            return Err(GpuError::UnsupportedParams);
        }
        let batch = nonces.len();
        // Flat layout for FFI.
        let mut nonces_flat = vec![0u8; batch * 32];
        for (i, n) in nonces.iter().enumerate() {
            nonces_flat[i * 32..(i + 1) * 32].copy_from_slice(n);
        }
        let mut out_valid = vec![0u8; batch];
        let mut out_indices = vec![0u32; batch * 32];

        let rc = unsafe {
            equium_run_batch(
                self.handle,
                input.as_ptr(),
                nonces_flat.as_ptr(),
                batch as c_int,
                out_valid.as_mut_ptr(),
                out_indices.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(map_cuda_err(rc, "equium_run_batch"));
        }

        // Compress indices and self-verify per candidate.
        let mut results = Vec::new();
        for i in 0..batch {
            if out_valid[i] == 0 {
                continue;
            }
            let idx_slice = &out_indices[i * 32..(i + 1) * 32];
            let compressed = compress_indices_96_5(idx_slice);
            // Final correctness gate: the upstream verifier. The GPU kernel
            // does its own all-zero-final-hash + distinct check, but we
            // re-verify against the canonical implementation. Cheap insurance
            // against any kernel divergence; never reached in the hot path
            // since most nonces don't yield candidates.
            if equihash::is_valid_solution(96, 5, input, &nonces[i], &compressed).is_ok() {
                results.push(GpuCandidate {
                    nonce: nonces[i],
                    soln_indices: compressed,
                });
            }
        }
        Ok(results)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { equium_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

fn map_cuda_err(rc: c_int, what: &'static str) -> GpuError {
    // Match common cudaError_t values surfaced by the kernel host wrapper.
    // Codes follow `enum cudaError` from cuda_runtime_api.h.
    match rc {
        -1 => GpuError::OutOfMemory,        // host-side malloc failure
        2 => GpuError::OutOfMemory,         // cudaErrorMemoryAllocation
        100 => GpuError::NoDevice,          // cudaErrorNoDevice
        _ => GpuError::Cuda(rc, what),
    }
}

/// Encode the 32 raw u32 indices into the canonical compressed byte form
/// the upstream `equihash::is_valid_solution` expects.
///
/// For (n=96, k=5): bits_per = cbits + 1 = 17. Total bits = 32 × 17 = 544.
/// Total bytes = 68.
fn compress_indices_96_5(indices: &[u32]) -> Vec<u8> {
    const BITS_PER: usize = 17;
    let total_bytes = (BITS_PER * 32).div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mut pos = 0usize;
    for &idx in indices {
        for b in (0..BITS_PER).rev() {
            let bit = (idx >> b) & 1;
            let byte = pos / 8;
            let shift = 7 - (pos % 8);
            out[byte] |= (bit as u8) << shift;
            pos += 1;
        }
    }
    out
}
