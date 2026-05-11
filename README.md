# Equium Miner ($EQM)

CPU-mineable Equihash PoW on Solana. Forked from [HannaPrints/equium](https://github.com/HannaPrints/equium).

## What is Equium?

Equium is a fair-launch token on Solana using Equihash (96,5) proof-of-work — memory-hard, CPU-friendly, ~50MB RAM per thread. No GPU advantage, no premine.

## Build

```bash
# Requires Rust + Solana CLI
cargo build --release -p equium-miner
```

## Usage

```bash
# Generate a Solana keypair if you don't have one
solana-keygen new -o ~/.config/solana/id.json

# Mine on mainnet
./target/release/equium-miner \
    --rpc-url https://api.mainnet-beta.solana.com \
    --keypair ~/.config/solana/id.json

# With options
./target/release/equium-miner \
    --rpc-url https://your-helius-rpc.com \
    --keypair ~/.config/solana/id.json \
    --threads 8 \
    --max-blocks 10
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--rpc-url` | public mainnet | Solana RPC endpoint (use Helius/Triton for production) |
| `--keypair` | — | Path to Solana keypair JSON |
| `--threads` | all cores | Number of solver threads |
| `--max-nonces-per-round` | 4096 | Nonce attempts per thread before refetch |
| `--max-blocks` | 0 (infinite) | Stop after N successful blocks |
| `--cu-limit` | 1,400,000 | Compute units per tx |

## How it Works

1. Fetch on-chain config (challenge, target, block height, reward)
2. Race N CPU threads to find Equihash solution under target
3. Submit `mine` instruction to Solana program
4. Receive EQM reward → repeat

## Token

- **Symbol:** EQM
- **Supply:** 21,000,000 (capped)
- **Premine:** 10% (2.1M) for DEX liquidity
- **Mineable:** 90% (18.9M)
- **Halving:** every ~378k blocks (~8.6 months)

## Links

- [Equium Website](https://www.equium.xyz)
- [Mining Dashboard](https://www.equium.xyz/mine)
- [Original Repo](https://github.com/HannaPrints/equium)

## License

Apache-2.0
