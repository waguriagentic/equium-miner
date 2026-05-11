// CUDA Equihash (n=96, k=5) solver for Equium.
//
// Architecture: one CUDA thread per nonce. Each thread runs the full Wagner
// search sequentially over its own slab of global memory. This is the
// simplest correct GPU implementation but suffers from severe warp
// divergence (every nonce takes a different path through bucket sort and
// distinct-index checks). On low-end GPUs (e.g. GT 1030) it is typically
// slower than the optimized AVX2 CPU solver in `equihash-core`.
//
// A block-cooperative kernel — where 128-256 threads within a block
// collaborate on a single nonce, parallelizing leaf generation, histogram,
// and pair emission — is the right design for production GPU throughput.
// This kernel is kept simple so future work can swap in the cooperative
// version without changing the FFI surface.
//
// BLAKE2b uses the standard sequential parameters with personalization
// "ZcashPoW" || N_le || K_le. The host pre-computes the initial chain
// values (h0) and pre-buffer (input || nonce) so the kernel only needs
// to absorb the 4-byte hash index and emit one compress call per leaf.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Equihash (96, 5) constants
// ---------------------------------------------------------------------------
#define FP_N                 96
#define FP_K                 5
#define FP_HASH_INIT_BYTES   12
#define FP_HASH_PAD          16
#define FP_INDICES_PER_HASH  5
#define FP_HASH_OUT_LEN      60
#define FP_N_INIT            131072
#define FP_FINAL_INDICES     32
#define FP_N_BUCKETS         65536
#define FP_CBYTES            2
#define MAX_STRIDE           32
#define I_LEN                81

// Per-thread workspace layout (offsets within the per-thread slab).
// hashes_a, hashes_b: 2 × (N_INIT × 16) = 4 MB
// indices_a, indices_b: 2 × (N_INIT × 32 × 4) = 32 MB
// bucket_count: (N_BUCKETS+1) × 4 = ~260 KB
// cursor: N_BUCKETS × 4 = 256 KB
// ordered: N_INIT × 4 = 512 KB
// row counts (a, b): 2 × 4 = 8 bytes
//
// Total: ~37 MB per thread.
#define WS_HASHES_A_OFFSET     ((size_t)0)
#define WS_HASHES_A_SIZE       ((size_t)FP_N_INIT * FP_HASH_PAD)
#define WS_HASHES_B_OFFSET     (WS_HASHES_A_OFFSET + WS_HASHES_A_SIZE)
#define WS_HASHES_B_SIZE       WS_HASHES_A_SIZE
#define WS_INDICES_A_OFFSET    (WS_HASHES_B_OFFSET + WS_HASHES_B_SIZE)
#define WS_INDICES_A_SIZE      ((size_t)FP_N_INIT * MAX_STRIDE * 4)
#define WS_INDICES_B_OFFSET    (WS_INDICES_A_OFFSET + WS_INDICES_A_SIZE)
#define WS_INDICES_B_SIZE      WS_INDICES_A_SIZE
#define WS_BUCKET_COUNT_OFFSET (WS_INDICES_B_OFFSET + WS_INDICES_B_SIZE)
#define WS_BUCKET_COUNT_SIZE   ((size_t)(FP_N_BUCKETS + 1) * 4)
#define WS_CURSOR_OFFSET       (WS_BUCKET_COUNT_OFFSET + WS_BUCKET_COUNT_SIZE)
#define WS_CURSOR_SIZE         ((size_t)FP_N_BUCKETS * 4)
#define WS_ORDERED_OFFSET      (WS_CURSOR_OFFSET + WS_CURSOR_SIZE)
#define WS_ORDERED_SIZE        ((size_t)FP_N_INIT * 4)
#define WS_TOTAL_SIZE          (WS_ORDERED_OFFSET + WS_ORDERED_SIZE)

// ---------------------------------------------------------------------------
// BLAKE2b
// ---------------------------------------------------------------------------
__device__ __constant__ uint64_t blake2b_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

__device__ __constant__ uint8_t blake2b_sigma[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

#define BLAKE_G(a, b, c, d, x, y)                  \
    do {                                            \
        v[a] = v[a] + v[b] + (x);                   \
        v[d] = rotr64(v[d] ^ v[a], 32);             \
        v[c] = v[c] + v[d];                         \
        v[b] = rotr64(v[b] ^ v[c], 24);             \
        v[a] = v[a] + v[b] + (y);                   \
        v[d] = rotr64(v[d] ^ v[a], 16);             \
        v[c] = v[c] + v[d];                         \
        v[b] = rotr64(v[b] ^ v[c], 63);             \
    } while (0)

__device__ void blake2b_compress(uint64_t* h, const uint64_t* m, uint64_t t0, uint64_t t1, int last) {
    uint64_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = h[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[8 + i] = blake2b_iv[i];
    v[12] ^= t0;
    v[13] ^= t1;
    if (last) v[14] = ~v[14];

    for (int r = 0; r < 12; r++) {
        const uint8_t* s = blake2b_sigma[r];
        BLAKE_G(0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        BLAKE_G(1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        BLAKE_G(2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        BLAKE_G(3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        BLAKE_G(0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        BLAKE_G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        BLAKE_G(2, 7,  8, 13, m[s[12]], m[s[13]]);
        BLAKE_G(3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[8 + i];
}

// Finalize one leaf hash. The host has already pre-loaded `h_init` (BLAKE2b
// chain values after personalization, with no compress calls yet) and
// `prefix_113` (the 81-byte I block followed by 32-byte nonce). We append the
// 4-byte hash index, pad to 128 bytes, and run a single compress with the
// final-block flag set. The output is 60 bytes (8 × u64 LE, first 60 bytes).
__device__ void blake2b_leaf_finalize(
    const uint64_t* h_init,
    const uint8_t* prefix_113,
    uint32_t hash_idx,
    uint8_t* out60)
{
    uint64_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) h[i] = h_init[i];

    uint8_t block[128];
    #pragma unroll
    for (int i = 0; i < 113; i++) block[i] = prefix_113[i];
    block[113] = (uint8_t)(hash_idx & 0xFF);
    block[114] = (uint8_t)((hash_idx >> 8) & 0xFF);
    block[115] = (uint8_t)((hash_idx >> 16) & 0xFF);
    block[116] = (uint8_t)((hash_idx >> 24) & 0xFF);
    #pragma unroll
    for (int i = 117; i < 128; i++) block[i] = 0;

    uint64_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint64_t w = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            w |= ((uint64_t)block[i * 8 + j]) << (j * 8);
        }
        m[i] = w;
    }

    // total bytes processed = 117 (input 81 + nonce 32 + hash_idx 4)
    blake2b_compress(h, m, 117, 0, /*last=*/1);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i * 8 + j >= FP_HASH_OUT_LEN) break;
            out60[i * 8 + j] = (uint8_t)(h[i] >> (j * 8));
        }
    }
}

// ---------------------------------------------------------------------------
// Distinct-indices check. Two indices lists in canonical tree order (not
// sorted). O(n*m) but with early exit; on the short slices used here this
// is faster than sorting.
// ---------------------------------------------------------------------------
__device__ int distinct_indices(const uint32_t* a, const uint32_t* b, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t x = a[i];
        for (int j = 0; j < n; j++) {
            if (x == b[j]) return 0;
        }
    }
    return 1;
}

// Final-row global distinctness on the 32 indices of a candidate solution.
__device__ int globally_distinct(const uint32_t* indices) {
    uint32_t sorted[FP_FINAL_INDICES];
    #pragma unroll
    for (int i = 0; i < FP_FINAL_INDICES; i++) sorted[i] = indices[i];
    // Insertion sort — 32 elements, branch-light enough.
    for (int i = 1; i < FP_FINAL_INDICES; i++) {
        uint32_t key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    for (int i = 1; i < FP_FINAL_INDICES; i++) {
        if (sorted[i] == sorted[i - 1]) return 0;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// One Wagner round, specialized for cbits=16. Bucket-sorts by leading 2
// bytes of the hash, then emits all XOR-pairs in each bucket whose
// indices are pairwise distinct.
// ---------------------------------------------------------------------------
__device__ uint32_t wagner_round(
    const uint8_t* in_hashes,        // [n_rows * FP_HASH_PAD]
    const uint32_t* in_indices,      // [n_rows * stride]
    uint32_t n_rows,
    uint32_t stride,
    uint32_t hash_bytes,
    uint8_t* out_hashes,             // [* * FP_HASH_PAD]
    uint32_t* out_indices,           // [* * (stride*2)]
    uint32_t* bucket_count,          // [N_BUCKETS + 1]
    uint32_t* cursor,                // [N_BUCKETS]
    uint32_t* ordered)               // [n_rows]
{
    if (n_rows < 2) return 0;

    // Histogram.
    for (int i = 0; i <= FP_N_BUCKETS; i++) bucket_count[i] = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t key = ((uint32_t)in_hashes[r * FP_HASH_PAD] << 8)
                     | (uint32_t)in_hashes[r * FP_HASH_PAD + 1];
        bucket_count[key + 1]++;
    }
    // Prefix sum.
    for (int i = 1; i <= FP_N_BUCKETS; i++) {
        bucket_count[i] += bucket_count[i - 1];
    }
    // Place.
    for (int i = 0; i < FP_N_BUCKETS; i++) cursor[i] = bucket_count[i];
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t key = ((uint32_t)in_hashes[r * FP_HASH_PAD] << 8)
                     | (uint32_t)in_hashes[r * FP_HASH_PAD + 1];
        ordered[cursor[key]++] = r;
    }

    uint32_t out_n = 0;
    uint32_t new_stride = stride * 2;
    uint32_t new_hash_bytes = (hash_bytes > FP_CBYTES) ? (hash_bytes - FP_CBYTES) : 0;

    for (int bk = 0; bk < FP_N_BUCKETS; bk++) {
        uint32_t start = bucket_count[bk];
        uint32_t end = bucket_count[bk + 1];
        if (end - start < 2) continue;
        for (uint32_t i = start; i < end; i++) {
            uint32_t ra = ordered[i];
            for (uint32_t j = i + 1; j < end; j++) {
                uint32_t rb = ordered[j];
                const uint32_t* ia = &in_indices[(size_t)ra * stride];
                const uint32_t* ib = &in_indices[(size_t)rb * stride];
                if (!distinct_indices(ia, ib, stride)) continue;

                // Emit new row.
                uint8_t* new_h = &out_hashes[(size_t)out_n * FP_HASH_PAD];
                #pragma unroll
                for (int b = 0; b < FP_HASH_PAD; b++) new_h[b] = 0;
                if (hash_bytes > FP_CBYTES) {
                    const uint8_t* ha = &in_hashes[(size_t)ra * FP_HASH_PAD];
                    const uint8_t* hb = &in_hashes[(size_t)rb * FP_HASH_PAD];
                    for (uint32_t off = 0; off < new_hash_bytes; off++) {
                        new_h[off] = ha[off + FP_CBYTES] ^ hb[off + FP_CBYTES];
                    }
                }

                // Canonical concat: subtree with smaller min index first.
                uint32_t* new_i = &out_indices[(size_t)out_n * new_stride];
                if (ia[0] < ib[0]) {
                    for (uint32_t k = 0; k < stride; k++) new_i[k] = ia[k];
                    for (uint32_t k = 0; k < stride; k++) new_i[stride + k] = ib[k];
                } else {
                    for (uint32_t k = 0; k < stride; k++) new_i[k] = ib[k];
                    for (uint32_t k = 0; k < stride; k++) new_i[stride + k] = ia[k];
                }
                out_n++;
                if (out_n >= FP_N_INIT) return out_n; // budget cap
            }
        }
    }
    return out_n;
}

// ---------------------------------------------------------------------------
// Per-thread Wagner search for one nonce.
// ---------------------------------------------------------------------------
__device__ void solve_one_nonce(
    const uint64_t* h_init,         // 8 u64
    const uint8_t* prefix_81,       // 81 bytes = I block
    const uint8_t* nonce,           // 32 bytes
    uint8_t* workspace,             // WS_TOTAL_SIZE bytes
    uint8_t* out_valid,             // 1 byte
    uint32_t* out_indices)          // FP_FINAL_INDICES u32
{
    *out_valid = 0;

    // Build the 113-byte prefix (input || nonce) on stack.
    uint8_t prefix_113[113];
    #pragma unroll
    for (int i = 0; i < 81; i++) prefix_113[i] = prefix_81[i];
    #pragma unroll
    for (int i = 0; i < 32; i++) prefix_113[81 + i] = nonce[i];

    uint8_t* hashes_a  = workspace + WS_HASHES_A_OFFSET;
    uint8_t* hashes_b  = workspace + WS_HASHES_B_OFFSET;
    uint32_t* indices_a = (uint32_t*)(workspace + WS_INDICES_A_OFFSET);
    uint32_t* indices_b = (uint32_t*)(workspace + WS_INDICES_B_OFFSET);
    uint32_t* bucket_count = (uint32_t*)(workspace + WS_BUCKET_COUNT_OFFSET);
    uint32_t* cursor = (uint32_t*)(workspace + WS_CURSOR_OFFSET);
    uint32_t* ordered = (uint32_t*)(workspace + WS_ORDERED_OFFSET);

    // Phase 1: generate FP_N_INIT leaves.
    const uint32_t n_hashes = (FP_N_INIT + FP_INDICES_PER_HASH - 1) / FP_INDICES_PER_HASH;
    uint8_t leaf_out[FP_HASH_OUT_LEN];

    for (uint32_t hi = 0; hi < n_hashes; hi++) {
        blake2b_leaf_finalize(h_init, prefix_113, hi, leaf_out);
        for (int slot = 0; slot < FP_INDICES_PER_HASH; slot++) {
            uint32_t leaf_idx = hi * FP_INDICES_PER_HASH + slot;
            if (leaf_idx >= FP_N_INIT) break;
            uint8_t* dst = &hashes_a[(size_t)leaf_idx * FP_HASH_PAD];
            #pragma unroll
            for (int b = 0; b < FP_HASH_PAD; b++) dst[b] = 0;
            #pragma unroll
            for (int b = 0; b < FP_HASH_INIT_BYTES; b++) {
                dst[b] = leaf_out[slot * FP_HASH_INIT_BYTES + b];
            }
            indices_a[leaf_idx] = leaf_idx;
        }
    }

    // Phase 2: 5 Wagner rounds, ping-pong between (a) and (b).
    uint32_t n_rows = FP_N_INIT;
    uint32_t stride = 1;
    uint32_t hash_bytes = FP_HASH_INIT_BYTES;
    int cur_is_a = 1;

    for (int round = 0; round < FP_K; round++) {
        uint8_t* in_h = cur_is_a ? hashes_a : hashes_b;
        uint32_t* in_i = cur_is_a ? indices_a : indices_b;
        uint8_t* out_h = cur_is_a ? hashes_b : hashes_a;
        uint32_t* out_i = cur_is_a ? indices_b : indices_a;

        n_rows = wagner_round(in_h, in_i, n_rows, stride, hash_bytes,
                              out_h, out_i, bucket_count, cursor, ordered);
        stride *= 2;
        hash_bytes = (hash_bytes > FP_CBYTES) ? (hash_bytes - FP_CBYTES) : 0;
        cur_is_a = !cur_is_a;
        if (n_rows == 0) return;
    }

    // Phase 3: scan for valid solutions (hash all-zero, 32 distinct indices).
    uint8_t* fin_h = cur_is_a ? hashes_a : hashes_b;
    uint32_t* fin_i = cur_is_a ? indices_a : indices_b;

    for (uint32_t r = 0; r < n_rows; r++) {
        uint8_t* h = &fin_h[(size_t)r * FP_HASH_PAD];
        if (h[0] != 0 || h[1] != 0) continue;
        uint32_t* idx = &fin_i[(size_t)r * FP_FINAL_INDICES];
        if (!globally_distinct(idx)) continue;
        // Found one. Write out and stop.
        #pragma unroll
        for (int k = 0; k < FP_FINAL_INDICES; k++) out_indices[k] = idx[k];
        *out_valid = 1;
        return;
    }
}

__global__ void equium_kernel(
    const uint64_t* h_init,
    const uint8_t* prefix_81,
    const uint8_t* nonces,
    uint8_t* workspaces,
    int batch_size,
    uint8_t* out_valid,
    uint32_t* out_indices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    solve_one_nonce(
        h_init,
        prefix_81,
        &nonces[tid * 32],
        &workspaces[(size_t)tid * WS_TOTAL_SIZE],
        &out_valid[tid],
        &out_indices[tid * FP_FINAL_INDICES]);
}

// ---------------------------------------------------------------------------
// Host-side API
// ---------------------------------------------------------------------------

struct EquiumContext {
    int batch_size;
    uint64_t* d_h_init;
    uint8_t* d_prefix;
    uint8_t* d_nonces;
    uint8_t* d_workspaces;
    uint8_t* d_out_valid;
    uint32_t* d_out_indices;
};

// Compute BLAKE2b initial chain values for the personalization
// "ZcashPoW" || N_le || K_le, with digest length 60. No compress calls
// have run yet — this is just IV ^ param_block.
static void compute_h_init_host(uint64_t h[8]) {
    static const uint64_t IV[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    };
    uint8_t param[64] = {0};
    param[0] = FP_HASH_OUT_LEN;  // digest_length = 60
    param[1] = 0;                // key_length
    param[2] = 1;                // fanout
    param[3] = 1;                // depth
    // bytes 4..48 are zero
    static const char personal_prefix[8] = {'Z','c','a','s','h','P','o','W'};
    for (int i = 0; i < 8; i++) param[48 + i] = (uint8_t)personal_prefix[i];
    uint32_t n_le = FP_N;
    uint32_t k_le = FP_K;
    for (int i = 0; i < 4; i++) param[56 + i] = (uint8_t)((n_le >> (8 * i)) & 0xFF);
    for (int i = 0; i < 4; i++) param[60 + i] = (uint8_t)((k_le >> (8 * i)) & 0xFF);

    for (int i = 0; i < 8; i++) {
        uint64_t w = 0;
        for (int j = 0; j < 8; j++) w |= ((uint64_t)param[i * 8 + j]) << (j * 8);
        h[i] = IV[i] ^ w;
    }
}

extern "C" int equium_init(int batch_size, void** ctx_out) {
    if (batch_size <= 0 || !ctx_out) return -1;
    *ctx_out = nullptr;

    EquiumContext* ctx = (EquiumContext*)calloc(1, sizeof(EquiumContext));
    if (!ctx) return -1;
    ctx->batch_size = batch_size;

    cudaError_t err;
    err = cudaMalloc((void**)&ctx->d_h_init, 8 * sizeof(uint64_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&ctx->d_prefix, I_LEN);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&ctx->d_nonces, (size_t)batch_size * 32);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&ctx->d_workspaces, (size_t)batch_size * WS_TOTAL_SIZE);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&ctx->d_out_valid, (size_t)batch_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&ctx->d_out_indices, (size_t)batch_size * FP_FINAL_INDICES * sizeof(uint32_t));
    if (err != cudaSuccess) goto fail;

    // Initialize h_init (constant for the personalization).
    {
        uint64_t h_init[8];
        compute_h_init_host(h_init);
        err = cudaMemcpy(ctx->d_h_init, h_init, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    *ctx_out = ctx;
    return 0;

fail:
    if (ctx->d_h_init)       cudaFree(ctx->d_h_init);
    if (ctx->d_prefix)       cudaFree(ctx->d_prefix);
    if (ctx->d_nonces)       cudaFree(ctx->d_nonces);
    if (ctx->d_workspaces)   cudaFree(ctx->d_workspaces);
    if (ctx->d_out_valid)    cudaFree(ctx->d_out_valid);
    if (ctx->d_out_indices)  cudaFree(ctx->d_out_indices);
    free(ctx);
    return (int)err;
}

extern "C" void equium_destroy(void* ctx_in) {
    EquiumContext* ctx = (EquiumContext*)ctx_in;
    if (!ctx) return;
    if (ctx->d_h_init)      cudaFree(ctx->d_h_init);
    if (ctx->d_prefix)      cudaFree(ctx->d_prefix);
    if (ctx->d_nonces)      cudaFree(ctx->d_nonces);
    if (ctx->d_workspaces)  cudaFree(ctx->d_workspaces);
    if (ctx->d_out_valid)   cudaFree(ctx->d_out_valid);
    if (ctx->d_out_indices) cudaFree(ctx->d_out_indices);
    free(ctx);
}

extern "C" int equium_run_batch(
    void* ctx_in,
    const uint8_t* input,    // I_LEN bytes
    const uint8_t* nonces,   // batch_size * 32
    int batch_size,
    uint8_t* out_valid,      // batch_size
    uint32_t* out_indices)   // batch_size * 32
{
    EquiumContext* ctx = (EquiumContext*)ctx_in;
    if (!ctx || batch_size <= 0 || batch_size > ctx->batch_size) return -1;

    cudaError_t err;
    err = cudaMemcpy(ctx->d_prefix, input, I_LEN, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(ctx->d_nonces, nonces, (size_t)batch_size * 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemset(ctx->d_out_valid, 0, (size_t)batch_size);
    if (err != cudaSuccess) return (int)err;

    // Launch geometry: one thread per nonce. We use 32 threads/block so each
    // warp gets a full set; for small batches we may launch a single block.
    int threads_per_block = 32;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    equium_kernel<<<blocks, threads_per_block>>>(
        ctx->d_h_init,
        ctx->d_prefix,
        ctx->d_nonces,
        ctx->d_workspaces,
        batch_size,
        ctx->d_out_valid,
        ctx->d_out_indices);

    err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;

    err = cudaMemcpy(out_valid, ctx->d_out_valid, (size_t)batch_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(out_indices, ctx->d_out_indices,
                     (size_t)batch_size * FP_FINAL_INDICES * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;

    return 0;
}
