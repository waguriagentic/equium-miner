//! Pure-Rust Equihash solver (Wagner's algorithm).
//!
//! Two paths:
//!   - **Fast path** for the production parameters (n=96, k=5). Uses batched
//!     BLAKE2b leaf generation, radix bucket sort on 16-bit collision keys,
//!     and fixed-size hash buffers — no allocations in the per-collision
//!     hot loop. ~3-5x faster than the generic path on AVX2 hardware.
//!   - **Generic path** for any other (n, k). Kept for tests and the
//!     `gen_bench_solution` example. Allocation-heavy reference implementation.
//!
//! Public API (`BaseState`, `try_nonce`, `solve`) is unchanged.

#![cfg(feature = "solver")]

use blake2b_simd::{Params as Blake2bParams, State as Blake2bState, PERSONALBYTES};

use crate::challenge::I_LEN;

#[derive(Debug, Clone)]
pub struct Solution {
    pub nonce: [u8; 32],
    pub soln_indices: Vec<u8>,
}

#[derive(Debug)]
pub enum SolveError {
    InvalidParams,
    NoSolutionFound,
}

fn hash_output_len(n: u32) -> usize {
    let indices_per_hash = (512 / n) as usize;
    indices_per_hash * (n as usize) / 8
}

fn cbits_of(n: u32, k: u32) -> usize {
    (n / (k + 1)) as usize
}

fn cbytes_of(n: u32, k: u32) -> usize {
    cbits_of(n, k).div_ceil(8)
}

fn init_state(n: u32, k: u32, hash_output: u8) -> Blake2bState {
    let mut p = Vec::with_capacity(PERSONALBYTES);
    p.extend_from_slice(b"ZcashPoW");
    p.extend_from_slice(&n.to_le_bytes());
    p.extend_from_slice(&k.to_le_bytes());
    Blake2bParams::new()
        .hash_length(hash_output as usize)
        .personal(&p)
        .to_state()
}

/// Pre-built base BLAKE2b state for a given (n, k, input). Computed once
/// per round and cloned per-nonce — the per-clone cost is much lower than
/// re-running the personalization + input update each attempt.
pub struct BaseState {
    pub state: Blake2bState,
    pub n: u32,
    pub k: u32,
}

impl BaseState {
    pub fn new(n: u32, k: u32, input: &[u8; I_LEN]) -> Result<Self, SolveError> {
        if n == 0 || k == 0 || n % 8 != 0 || n % (k + 1) != 0 || k >= n {
            return Err(SolveError::InvalidParams);
        }
        let hash_output = hash_output_len(n) as u8;
        let mut state = init_state(n, k, hash_output);
        state.update(input);
        Ok(Self { state, n, k })
    }
}

/// Try a single nonce against a pre-built `BaseState`. Returns the canonical
/// compressed solution-indices byte string if this nonce yields a valid
/// Equihash solution, `None` otherwise.
///
/// Cheaper for repeated calls than `solve()` because the base state is built
/// once and reused. The unit of work is one Wagner search — the caller drives
/// nonce selection and parallelism.
pub fn try_nonce(base: &BaseState, input: &[u8; I_LEN], nonce: &[u8; 32]) -> Option<Vec<u8>> {
    if base.n == 96 && base.k == 5 {
        try_nonce_96_5(base, input, nonce)
    } else {
        try_nonce_generic(base, input, nonce)
    }
}

/// Solve Equihash for the given (n, k) and base input bytes.
///
/// Sequential driver kept for backward compat with single-threaded callers
/// (the WASM solver in the browser miner uses this). Multi-threaded callers
/// should use `BaseState::new` + `try_nonce` and parallelize at the
/// nonce-selection level.
pub fn solve<F>(n: u32, k: u32, input: &[u8; I_LEN], mut next_nonce: F) -> Result<Solution, SolveError>
where
    F: FnMut() -> Option<[u8; 32]>,
{
    let base = BaseState::new(n, k, input)?;

    loop {
        let nonce = match next_nonce() {
            Some(nn) => nn,
            None => return Err(SolveError::NoSolutionFound),
        };
        if let Some(soln_indices) = try_nonce(&base, input, &nonce) {
            return Ok(Solution {
                nonce,
                soln_indices,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Fast path: (n=96, k=5)
// ---------------------------------------------------------------------------
//
// Parameter constants for the locked Equium production params. Each round
// trims `CBYTES` bytes (16 collision bits) off the front of the XORed hash.
// Starting hash is N/8 = 12 bytes; after 5 rounds we're down to 2 bytes,
// which must be all-zero for the candidate to be a valid solution.

const FP_N: usize = 96;
const FP_K: usize = 5;
const FP_CBITS: usize = 16;
const FP_CBYTES: usize = 2;
const FP_HASH_INIT_BYTES: usize = FP_N / 8;            // 12
const FP_HASH_PAD: usize = 16;                          // [u8; 16] for nicer alignment
const FP_INDICES_PER_HASH: usize = 512 / FP_N;          // 5
const FP_HASH_OUT_LEN: usize = FP_INDICES_PER_HASH * FP_HASH_INIT_BYTES; // 60
const FP_N_INIT: usize = 1 << (FP_CBITS + 1);           // 131072
const FP_FINAL_INDICES: usize = 1 << FP_K;              // 32
const FP_N_BUCKETS: usize = 1 << FP_CBITS;              // 65536

/// Try one nonce against a pre-built (n=96, k=5) base state.
fn try_nonce_96_5(base: &BaseState, input: &[u8; I_LEN], nonce: &[u8; 32]) -> Option<Vec<u8>> {
    // Allocate scratch reused across all 5 rounds. The "current" and "next"
    // generation alternate to avoid copying.
    let mut hashes_a: Vec<[u8; FP_HASH_PAD]> = Vec::with_capacity(FP_N_INIT);
    let mut indices_a: Vec<u32> = Vec::with_capacity(FP_N_INIT);
    let mut hashes_b: Vec<[u8; FP_HASH_PAD]> = Vec::with_capacity(FP_N_INIT);
    let mut indices_b: Vec<u32> = Vec::with_capacity(FP_N_INIT);

    generate_leaves_96_5(&base.state, nonce, &mut hashes_a, &mut indices_a);

    // Round 0..k: bucket-sort by leading 16 bits and pair within each bucket.
    let mut hash_bytes = FP_HASH_INIT_BYTES;
    let mut stride = 1usize;
    let (mut cur_h, mut cur_i, mut nxt_h, mut nxt_i) =
        (&mut hashes_a, &mut indices_a, &mut hashes_b, &mut indices_b);

    let mut bucket_count = vec![0u32; FP_N_BUCKETS + 1];
    let mut ordered = Vec::<u32>::with_capacity(FP_N_INIT);

    for _round in 0..FP_K {
        wagner_round_96_5(
            cur_h,
            cur_i,
            stride,
            hash_bytes,
            nxt_h,
            nxt_i,
            &mut bucket_count,
            &mut ordered,
        );
        stride *= 2;
        hash_bytes = hash_bytes.saturating_sub(FP_CBYTES);
        // Swap.
        core::mem::swap(&mut cur_h, &mut nxt_h);
        core::mem::swap(&mut cur_i, &mut nxt_i);
        if cur_h.is_empty() {
            return None;
        }
    }

    // After 5 rounds: stride should be 32, hash_bytes should be 2.
    debug_assert_eq!(stride, FP_FINAL_INDICES);
    debug_assert_eq!(hash_bytes, FP_HASH_INIT_BYTES - FP_K * FP_CBYTES);

    // Find a row whose remaining hash is all zero AND has 32 distinct indices.
    // The bucket-and-pair process already enforces distinct-at-each-merge, but
    // it does NOT guarantee global distinctness across the full tree — that's
    // why the upstream verifier re-checks. We do the same final check here.
    for row_id in 0..cur_h.len() {
        let h = &cur_h[row_id];
        if h[0] != 0 || h[1] != 0 {
            continue;
        }
        let idx_slice = &cur_i[row_id * stride..(row_id + 1) * stride];
        if !globally_distinct(idx_slice) {
            continue;
        }
        let compressed = compress_indices(FP_N as u32, FP_K as u32, idx_slice);
        if equihash::is_valid_solution(FP_N as u32, FP_K as u32, input, nonce, &compressed).is_ok() {
            return Some(compressed);
        }
    }
    None
}

/// Emit all `FP_N_INIT` leaves using batched BLAKE2b — one hash call yields
/// `FP_INDICES_PER_HASH` leaves of `FP_HASH_INIT_BYTES` each.
fn generate_leaves_96_5(
    base: &Blake2bState,
    nonce: &[u8; 32],
    hashes_out: &mut Vec<[u8; FP_HASH_PAD]>,
    indices_out: &mut Vec<u32>,
) {
    hashes_out.clear();
    indices_out.clear();

    let mut state_n = base.clone();
    state_n.update(nonce);

    let n_hashes = FP_N_INIT.div_ceil(FP_INDICES_PER_HASH);

    for hash_idx in 0..n_hashes as u32 {
        let mut s = state_n.clone();
        s.update(&hash_idx.to_le_bytes());
        let h = s.finalize();
        let h_bytes = h.as_bytes();
        debug_assert_eq!(h_bytes.len(), FP_HASH_OUT_LEN);

        for slot in 0..FP_INDICES_PER_HASH {
            let leaf_idx = hash_idx as usize * FP_INDICES_PER_HASH + slot;
            if leaf_idx >= FP_N_INIT {
                break;
            }
            let mut buf = [0u8; FP_HASH_PAD];
            buf[..FP_HASH_INIT_BYTES]
                .copy_from_slice(&h_bytes[slot * FP_HASH_INIT_BYTES..(slot + 1) * FP_HASH_INIT_BYTES]);
            hashes_out.push(buf);
            indices_out.push(leaf_idx as u32);
        }
    }
}

/// One Wagner round, specialized for cbits=16.
///
/// Strategy: radix-sort row IDs by the leading 16 bits of their hash (one
/// pass with histogram + prefix-sum + place), then for each bucket emit all
/// pairs that pass the distinct-indices check. Both passes are O(N).
///
/// `bucket_count` and `ordered` are caller-provided scratch buffers.
#[allow(clippy::too_many_arguments)]
fn wagner_round_96_5(
    in_hashes: &[[u8; FP_HASH_PAD]],
    in_indices: &[u32],
    stride: usize,
    hash_bytes: usize,
    out_hashes: &mut Vec<[u8; FP_HASH_PAD]>,
    out_indices: &mut Vec<u32>,
    bucket_count: &mut Vec<u32>,
    ordered: &mut Vec<u32>,
) {
    out_hashes.clear();
    out_indices.clear();

    let n_rows = in_hashes.len();
    if n_rows < 2 {
        return;
    }

    // Histogram.
    bucket_count.clear();
    bucket_count.resize(FP_N_BUCKETS + 1, 0);
    for hash in in_hashes {
        let key = ((hash[0] as u16) << 8) | (hash[1] as u16);
        bucket_count[key as usize + 1] += 1;
    }
    // Prefix sum (in-place on indices 1..=N).
    for i in 1..=FP_N_BUCKETS {
        bucket_count[i] += bucket_count[i - 1];
    }

    // Place row IDs into bucket positions.
    ordered.clear();
    ordered.resize(n_rows, 0);
    // Cursor per bucket: clone of starts, mutated as we place.
    let mut cursor: Vec<u32> = bucket_count[..FP_N_BUCKETS].to_vec();
    for (row_id, hash) in in_hashes.iter().enumerate() {
        let key = ((hash[0] as u16) << 8) | (hash[1] as u16);
        let pos = cursor[key as usize];
        cursor[key as usize] = pos + 1;
        ordered[pos as usize] = row_id as u32;
    }

    let new_hash_bytes = hash_bytes.saturating_sub(FP_CBYTES);

    // Pair within each bucket. Most buckets have 0-3 entries.
    for bucket_key in 0..FP_N_BUCKETS {
        let start = bucket_count[bucket_key] as usize;
        let end = bucket_count[bucket_key + 1] as usize;
        if end - start < 2 {
            continue;
        }
        for i in start..end {
            let ra = ordered[i] as usize;
            for j in (i + 1)..end {
                let rb = ordered[j] as usize;
                let ia = &in_indices[ra * stride..(ra + 1) * stride];
                let ib = &in_indices[rb * stride..(rb + 1) * stride];
                if !distinct_indices(ia, ib) {
                    continue;
                }

                let ha = &in_hashes[ra];
                let hb = &in_hashes[rb];

                // XOR + trim. The first FP_CBYTES bytes are guaranteed zero
                // (bucket matched on those bits), so we just shift left by 2.
                let mut new_hash = [0u8; FP_HASH_PAD];
                if hash_bytes > FP_CBYTES {
                    for off in 0..new_hash_bytes {
                        new_hash[off] = ha[off + FP_CBYTES] ^ hb[off + FP_CBYTES];
                    }
                }
                out_hashes.push(new_hash);

                // Canonical concat: subtree with smaller min index first.
                if ia[0] < ib[0] {
                    out_indices.extend_from_slice(ia);
                    out_indices.extend_from_slice(ib);
                } else {
                    out_indices.extend_from_slice(ib);
                    out_indices.extend_from_slice(ia);
                }
            }
        }
    }
}

/// Pairwise distinctness check between two index lists. Indices are in
/// canonical tree-concat order (NOT sorted), so this is O(n*m). With short
/// lists (1, 2, 4, 8, 16) the constant factor is small and the early-exit
/// often fires.
fn distinct_indices(a: &[u32], b: &[u32]) -> bool {
    for x in a {
        for y in b {
            if x == y {
                return false;
            }
        }
    }
    true
}

/// Final-row global distinctness: all `FP_FINAL_INDICES` indices must be
/// unique. With 32 indices, a sort + adjacent-scan is fastest.
fn globally_distinct(indices: &[u32]) -> bool {
    let mut sorted: [u32; FP_FINAL_INDICES] = [0; FP_FINAL_INDICES];
    sorted.copy_from_slice(indices);
    sorted.sort_unstable();
    for w in sorted.windows(2) {
        if w[0] == w[1] {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Generic path: any (n, k). Original reference implementation, kept for
// non-locked params used by tests and the bench example.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Row {
    hash: Vec<u8>,
    indices: Vec<u32>,
}

fn generate_leaf_hash(base: &Blake2bState, n: u32, i: u32) -> Vec<u8> {
    let indices_per = (512 / n) as usize;
    let n_bytes = (n / 8) as usize;
    let mut state = base.clone();
    state.update(&(i / indices_per as u32).to_le_bytes());
    let full = state.finalize();
    let off = (i as usize % indices_per) * n_bytes;
    full.as_bytes()[off..off + n_bytes].to_vec()
}

fn first_cbits_eq(a: &[u8], b: &[u8], cbits: usize) -> bool {
    let full_bytes = cbits / 8;
    if a[..full_bytes] != b[..full_bytes] {
        return false;
    }
    let rem_bits = cbits % 8;
    if rem_bits == 0 {
        return true;
    }
    let mask = 0xFFu8 << (8 - rem_bits);
    (a[full_bytes] & mask) == (b[full_bytes] & mask)
}

fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

fn distinct_indices_generic(a: &[u32], b: &[u32]) -> bool {
    for x in a {
        for y in b {
            if x == y {
                return false;
            }
        }
    }
    true
}

fn concat_canonical(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    if a[0] < b[0] {
        out.extend_from_slice(a);
        out.extend_from_slice(b);
    } else {
        out.extend_from_slice(b);
        out.extend_from_slice(a);
    }
    out
}

fn round(rows: Vec<Row>, cbits: usize) -> Vec<Row> {
    let cbytes = cbits.div_ceil(8);
    let mut sorted = rows;
    sorted.sort_by(|a, b| a.hash.cmp(&b.hash));

    let mut out = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && first_cbits_eq(&sorted[i].hash, &sorted[j].hash, cbits) {
            j += 1;
        }
        for ia in i..j {
            for ib in (ia + 1)..j {
                let a = &sorted[ia];
                let b = &sorted[ib];
                if !distinct_indices_generic(&a.indices, &b.indices) {
                    continue;
                }
                let xored = xor(&a.hash, &b.hash);
                let new_hash = if xored.len() <= cbytes {
                    Vec::new()
                } else {
                    xored[cbytes..].to_vec()
                };
                out.push(Row {
                    hash: new_hash,
                    indices: concat_canonical(&a.indices, &b.indices),
                });
            }
        }
        i = j;
    }
    out
}

fn compress_indices(n: u32, k: u32, indices: &[u32]) -> Vec<u8> {
    let bits_per = cbits_of(n, k) + 1;
    let total_bits = bits_per * indices.len();
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mut pos = 0usize;
    for &idx in indices {
        for b in (0..bits_per).rev() {
            let bit = (idx >> b) & 1;
            let byte = pos / 8;
            let shift = 7 - (pos % 8);
            out[byte] |= (bit as u8) << shift;
            pos += 1;
        }
    }
    out
}

fn try_nonce_generic(base: &BaseState, input: &[u8; I_LEN], nonce: &[u8; 32]) -> Option<Vec<u8>> {
    let n = base.n;
    let k = base.k;
    let cbits = cbits_of(n, k);
    let cbytes = cbytes_of(n, k);
    let n_init: u32 = 1u32 << (cbits + 1);

    let mut state_with_nonce = base.state.clone();
    state_with_nonce.update(nonce);

    let mut rows: Vec<Row> = (0..n_init)
        .map(|i| Row {
            hash: generate_leaf_hash(&state_with_nonce, n, i),
            indices: vec![i],
        })
        .collect();

    for _ in 0..k {
        rows = round(rows, cbits);
        if rows.is_empty() {
            return None;
        }
        let _ = cbytes;
    }

    let target_indices_len = 1usize << k;
    for row in rows {
        if row.indices.len() != target_indices_len {
            continue;
        }
        if !row.hash.iter().all(|&b| b == 0) {
            continue;
        }
        let compressed = compress_indices(n, k, &row.indices);
        if equihash::is_valid_solution(n, k, input, nonce, &compressed).is_ok() {
            return Some(compressed);
        }
    }
    None
}
