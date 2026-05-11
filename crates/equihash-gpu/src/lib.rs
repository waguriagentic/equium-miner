//! CUDA Equihash solver for Equium (n=96, k=5 only).
//!
//! The public API is unconditional — when built without the `cuda` feature,
//! `GpuSolver::new` returns `GpuError::NotCompiled` so callers can present
//! a clean error message without conditional-compilation guards. When `cuda`
//! is enabled, the build script compiles `src/kernel.cu` and links `cudart`.
//!
//! The kernel uses one CUDA thread per nonce — simple, correct, and easy to
//! verify against the CPU solver, but NOT optimized for low-end GPUs. On a
//! GT 1030 you should expect performance roughly comparable to (or worse
//! than) the optimized AVX2 CPU path. A block-cooperative kernel is the
//! natural next step for production-grade GPU throughput.

use equihash_core::challenge::{build_input, I_LEN};

#[derive(Debug)]
pub enum GpuError {
    /// Built without the `cuda` feature. Re-build with `--features gpu` on
    /// the CLI miner (or `--features cuda` on this crate directly).
    NotCompiled,
    /// CUDA returned an error code from a runtime call.
    Cuda(i32, &'static str),
    /// No CUDA device detected at runtime.
    NoDevice,
    /// Out-of-memory when allocating per-nonce workspace.
    OutOfMemory,
    /// (n, k) does not match the supported (96, 5).
    UnsupportedParams,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCompiled => write!(
                f,
                "equihash-gpu built without `cuda` feature — rebuild with `--features gpu`"
            ),
            Self::Cuda(code, what) => write!(f, "CUDA error {} in {}", code, what),
            Self::NoDevice => write!(f, "no CUDA device detected"),
            Self::OutOfMemory => write!(f, "out of GPU memory allocating workspace"),
            Self::UnsupportedParams => write!(f, "GPU solver only supports (n=96, k=5)"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A batch result from `GpuSolver::try_batch`.
#[derive(Debug, Clone)]
pub struct GpuCandidate {
    pub nonce: [u8; 32],
    pub soln_indices: Vec<u8>,
}

/// GPU solver, holding pre-allocated device buffers. Construct once and reuse
/// across nonce batches; teardown happens on `Drop`.
pub struct GpuSolver {
    #[cfg(feature = "cuda")]
    inner: cuda_impl::Context,
}

impl GpuSolver {
    /// Initialize a GPU solver for (n, k) = (96, 5). `batch_size` is the
    /// number of nonces processed per kernel launch. Larger values amortize
    /// launch latency but require more VRAM (~37 MB / nonce).
    pub fn new(n: u32, k: u32, batch_size: usize) -> Result<Self, GpuError> {
        if n != 96 || k != 5 {
            return Err(GpuError::UnsupportedParams);
        }
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                inner: cuda_impl::Context::new(batch_size)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = batch_size;
            Err(GpuError::NotCompiled)
        }
    }

    /// Run a batch of nonces through the GPU kernel. Returns any valid
    /// Equihash solutions (the host re-verifies each against the upstream
    /// `equihash::is_valid_solution` for safety; callers must still check
    /// `hash_under_target` themselves before submitting).
    pub fn try_batch(
        &mut self,
        current_challenge: &[u8; 32],
        miner_pubkey: &[u8; 32],
        block_height: u64,
        nonces: &[[u8; 32]],
    ) -> Result<Vec<GpuCandidate>, GpuError> {
        #[cfg(feature = "cuda")]
        {
            let input = build_input(current_challenge, miner_pubkey, block_height);
            self.inner.run_batch(&input, nonces)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (current_challenge, miner_pubkey, block_height, nonces);
            Err(GpuError::NotCompiled)
        }
    }
}

// Convenience: stub `build_input` so the crate compiles without warnings
// when the cuda feature is off (we still need the symbol referenced).
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn _ensure_imports_used(c: &[u8; 32], m: &[u8; 32], h: u64) -> [u8; I_LEN] {
    build_input(c, m, h)
}

#[cfg(feature = "cuda")]
mod cuda_impl;
