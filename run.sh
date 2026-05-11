#!/bin/bash
# Equium Miner launcher
# Reads config from .env and runs the miner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
MINER="${SCRIPT_DIR}/target/release/equium-miner"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env not found at $ENV_FILE"
    exit 1
fi

if [ ! -f "$MINER" ]; then
    echo "Error: Miner binary not found. Build first:"
    echo "  CPU only:  cargo build --release -p equium-cli-miner"
    echo "  With GPU:  cargo build --release --features gpu -p equium-cli-miner"
    exit 1
fi

# Load .env
set -a
source "$ENV_FILE"
set +a

if [ -z "${PRIVATE_KEY:-}" ]; then
    echo "Error: PRIVATE_KEY not set in .env"
    exit 1
fi

# Mining mode
MODE="${MODE:-cpu}"

# Calculate threads: use all cores minus 1 (reserve 1 for system)
TOTAL_CORES=$(nproc)
if [ "$MODE" = "cpu" ] && [ "${THREADS:-0}" -eq 0 ] 2>/dev/null; then
    THREADS=$((TOTAL_CORES - 1))
    [ "$THREADS" -lt 1 ] && THREADS=1
fi

# Convert private key (base58/hex) to Solana keypair JSON file
TMPKEY=$(mktemp /tmp/eqm-keypair-XXXXXX.json)
trap "rm -f $TMPKEY" EXIT

python3 - "$PRIVATE_KEY" "$TMPKEY" << 'PYEOF'
import sys, json, base58

pk = sys.argv[1].strip()
out = sys.argv[2]

# Detect format: hex or base58
if pk.startswith("0x") or (len(pk) == 128 and all(c in "0123456789abcdefABCDEF" for c in pk)):
    raw = bytes.fromhex(pk.lstrip("0x"))
elif pk.startswith("[") and pk.endswith("]"):
    raw = bytes(json.loads(pk))
else:
    raw = base58.b58decode(pk)

# Solana keypair = 64 bytes (32 private + 32 public)
if len(raw) == 32:
    from nacl.signing import SigningKey
    sk = SigningKey(raw)
    raw = raw + bytes(sk.verify_key)

if len(raw) != 64:
    print(f"Error: keypair must be 64 bytes, got {len(raw)}")
    sys.exit(1)

with open(out, "w") as f:
    json.dump(list(raw), f)

print("Keypair converted OK")
PYEOF

# Build args based on mode
ARGS=(
    --rpc-url "$RPC_URL"
    --keypair "$TMPKEY"
    --max-nonces-per-round "$MAX_NONCES_PER_ROUND"
    --max-blocks "$MAX_BLOCKS"
    --cu-limit "$CU_LIMIT"
)

if [ "$MODE" = "gpu" ]; then
    ARGS+=(--gpu-batch "${GPU_BATCH:-16}")
    echo "Starting Equium Miner (GPU)..."
    echo "  RPC: $RPC_URL"
    echo "  GPU Batch: ${GPU_BATCH:-16} nonces/launch (~$(( ${GPU_BATCH:-16} * 37 ))MB VRAM)"
else
    ARGS+=(--threads "$THREADS")
    echo "Starting Equium Miner (CPU)..."
    echo "  RPC: $RPC_URL"
    echo "  Threads: $THREADS / $TOTAL_CORES cores (1 reserved for system)"
fi

echo "  Max Blocks: $MAX_BLOCKS (0=infinite)"
echo ""

exec "$MINER" "${ARGS[@]}"
