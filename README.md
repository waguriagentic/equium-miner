# Equium Miner ($EQM)

CPU/GPU mineable Equihash PoW on Solana. Forked from [HannaPrints/equium](https://github.com/HannaPrints/equium).

## What is Equium?

Equium is a fair-launch token on Solana using Equihash (96,5) proof-of-work — memory-hard, CPU-friendly, ~50MB RAM per thread. No GPU advantage, no premine.

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your RPC URL and private key
```

### 2. Run

```bash
# Foreground (auto-installs prerequisites + builds on first run)
./run.sh

# Background (tmux)
tmux new-session -d -s eqm "./run.sh"

# Attach to session
tmux attach -t eqm
```

`run.sh` handles everything automatically on first run:
- Detects and installs missing system packages (`python3`, `python3-dev`)
- Installs Rust toolchain via `rustup` if not present
- Installs Python dependencies (`pynacl`, `base58`)
- Builds the miner binary based on `MODE` (cpu/gpu)

To force re-check prerequisites without mining:

```bash
./run.sh --deps
```

### Manual Build (optional)

If you prefer building manually:

```bash
# CPU only
cargo build --release -p equium-cli-miner

# With CUDA GPU support
cargo build --release --features gpu -p equium-cli-miner
```

## Prerequisites

| Dependency | Purpose | Auto-installed? |
|-----------|---------|-----------------|
| Rust toolchain (`cargo`, `rustc`) | Build miner binary | Yes (via `rustup`) |
| Python 3 | Keypair conversion | Yes (via `apt`) |
| `python3-dev` | Build `pynacl` C extension | Yes (via `apt`) |
| `pynacl` | Ed25519 key derivation | Yes (via `pip`) |
| `base58` | Private key decoding | Yes (via `pip`) |
| CUDA Toolkit | GPU mining only | No — install manually |

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `RPC_URL` | — | Solana RPC endpoint (Helius recommended) |
| `PRIVATE_KEY` | — | Private key (base58, hex, or JSON array) |
| `MODE` | `cpu` | Mining mode: `cpu` or `gpu` |
| `THREADS` | `0` | CPU threads. 0 = all cores - 1. Only used in CPU mode |
| `GPU_BATCH` | `16` | Nonces per GPU kernel launch (~37MB VRAM each). Only used in GPU mode |
| `MAX_NONCES_PER_ROUND` | `1000000` | Nonce attempts per round |
| `MAX_BLOCKS` | `0` | Stop after N blocks (0 = infinite) |
| `CU_LIMIT` | `1400000` | Compute units per tx |

## CLI Flags (direct usage)

```bash
./target/release/equium-miner \
    --rpc-url https://your-rpc.com \
    --keypair ~/.config/solana/id.json \
    --threads 12 \
    --max-nonces-per-round 1000000 \
    --max-blocks 0 \
    --cu-limit 1400000

# With GPU
./target/release/equium-miner \
    --rpc-url https://your-rpc.com \
    --keypair ~/.config/solana/id.json \
    --gpu-batch 16 \
    --max-nonces-per-round 1000000
```

## How it Works

1. Fetch on-chain config (challenge, target, block height, reward)
2. Race N CPU/GPU threads to find Equihash solution under target
3. Submit `mine` instruction to Solana program
4. Receive EQM reward → repeat

Real-time hashrate is displayed every 5 seconds during mining.

## GPU Support

The miner has CUDA GPU support. Build with `--features gpu` and set `MODE=gpu` in `.env`.

- One CUDA thread per nonce
- ~37 MB VRAM per nonce batch
- Best on modern GPUs with sufficient VRAM
- Low-end GPUs (GT 1030) may not outperform CPU

## Token

- **Symbol:** EQM
- **Supply:** 21,000,000 (capped)
- **Premine:** 10% (2.1M) for DEX liquidity
- **Mineable:** 90% (18.9M)
- **Halving:** every ~378k blocks (~8.6 months)
- **Block reward:** 25 EQM (current epoch)

## Links

- [Equium Website](https://www.equium.xyz)
- [Mining Dashboard](https://www.equium.xyz/mine)
- [Original Repo](https://github.com/HannaPrints/equium)

## License

Apache-2.0
