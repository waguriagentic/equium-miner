# Equium Miner ($EQM)

CPU/GPU mineable Equihash PoW on Solana. Forked from [HannaPrints/equium](https://github.com/HannaPrints/equium).

## What is Equium?

Equium is a fair-launch token on Solana using Equihash (96,5) proof-of-work — memory-hard, CPU-friendly, ~50MB RAM per thread. No GPU advantage, no premine.

## Quick Start

### 1. Build

```bash
# CPU only
cargo build --release -p equium-cli-miner

# With CUDA GPU support
cargo build --release --features gpu -p equium-cli-miner
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run

```bash
# Foreground
./run.sh

# Background (tmux)
tmux new-session -d -s eqm "./run.sh"

# Attach to session
tmux attach -t eqm
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `RPC_URL` | — | Solana RPC endpoint (Helius recommended) |
| `PRIVATE_KEY` | — | Private key (base58, hex, or JSON array) |
| `MODE` | `cpu` | Mining mode: `cpu` or `gpu` |
| `THREADS` | `0` | CPU threads. 0 = all cores - 1. Only used in CPU mode |
| `GPU_BATCH` | `16` | Nonces per GPU kernel launch (~37MB VRAM each). Only used in GPU mode |
| `MAX_NONCES_PER_ROUND` | `4096` | Nonce attempts per thread per round |
| `MAX_BLOCKS` | `0` | Stop after N blocks (0 = infinite) |
| `CU_LIMIT` | `1400000` | Compute units per tx |

## CLI Flags (direct usage)

```bash
./target/release/equium-miner \
    --rpc-url https://your-rpc.com \
    --keypair ~/.config/solana/id.json \
    --threads 12 \
    --max-nonces-per-round 4096 \
    --max-blocks 0 \
    --cu-limit 1400000

# With GPU
./target/release/equium-miner \
    --rpc-url https://your-rpc.com \
    --keypair ~/.config/solana/id.json \
    --gpu-batch 16 \
    --max-nonces-per-round 4096
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
