#!/bin/bash
# Equium Miner launcher
# Reads config from .env, installs prerequisites, builds if needed, and runs the miner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
MINER="${SCRIPT_DIR}/target/release/equium-miner"

# ============================================================
# Prerequisites check & install
# ============================================================
install_prerequisites() {
    echo "=== Checking prerequisites ==="
    local MISSING=()

    # Check curl (needed for rustup install)
    if ! command -v curl &>/dev/null; then
        echo "[!] curl not found"
        MISSING+=("curl")
    else
        echo "[ok] curl"
    fi

    # Check Rust toolchain
    if ! command -v cargo &>/dev/null; then
        echo "[!] Rust toolchain not found"
        MISSING+=("rust")
    else
        echo "[ok] Rust $(rustc --version 2>/dev/null | awk '{print $2}')"
    fi

    # Check Python 3
    if ! command -v python3 &>/dev/null; then
        echo "[!] Python 3 not found"
        MISSING+=("python3")
    else
        echo "[ok] Python $(python3 --version 2>/dev/null | awk '{print $2}')"
    fi

    # Check python3-dev (needed for pynacl build)
    if ! dpkg -s python3-dev &>/dev/null 2>&1; then
        MISSING+=("python3-dev")
    else
        echo "[ok] python3-dev"
    fi

    # Check build-essential (C compiler needed for Rust builds)
    if ! dpkg -s build-essential &>/dev/null 2>&1; then
        MISSING+=("build-essential")
    else
        echo "[ok] build-essential"
    fi

    # Install system packages if missing
    local SYS_PKGS=()
    for pkg in "${MISSING[@]}"; do
        case "$pkg" in
            python3|python3-dev|curl|build-essential) SYS_PKGS+=("$pkg") ;;
        esac
    done

    if [ ${#SYS_PKGS[@]} -gt 0 ]; then
        echo ""
        echo "Installing system packages: ${SYS_PKGS[*]}"
        local SUDO="sudo"
        [ "$(id -u)" -eq 0 ] && SUDO=""
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq "${SYS_PKGS[@]}"
    fi

    # Install Rust via rustup if missing
    if [[ " ${MISSING[*]} " =~ " rust " ]]; then
        echo ""
        echo "Installing Rust via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        echo "[ok] Rust $(rustc --version | awk '{print $2}')"
    fi

    # Detect and export CUDA path if GPU mode
    if [ "${MODE:-cpu}" = "gpu" ]; then
        local CUDA_FOUND=0

        # Check if nvcc is already on PATH
        if command -v nvcc &>/dev/null; then
            local NVCC_VER
            NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
            echo "[ok] nvcc $NVCC_VER (on PATH)"
            CUDA_FOUND=1
        fi

        # Try common CUDA install locations
        if [ "$CUDA_FOUND" -eq 0 ]; then
            for cuda_dir in /usr/local/cuda*/bin /opt/cuda*/bin; do
                if [ -x "$cuda_dir/nvcc" ]; then
                    export PATH="$cuda_dir:$PATH"
                    export CUDA_PATH="$(dirname "$cuda_dir")"
                    local NVCC_VER
                    NVCC_VER=$("$cuda_dir/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
                    echo "[ok] nvcc $NVCC_VER found at $cuda_dir (added to PATH)"
                    CUDA_FOUND=1
                    break
                fi
            done
        fi

        # Auto-install CUDA toolkit if nvcc still not found
        if [ "$CUDA_FOUND" -eq 0 ]; then
            echo "[!] nvcc not found — installing CUDA toolkit..."
            # Detect driver-supported CUDA version from nvidia-smi
            local DRIVER_CUDA_VER=""
            if command -v nvidia-smi &>/dev/null; then
                DRIVER_CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
            fi
            if [ -z "$DRIVER_CUDA_VER" ]; then
                # Fallback: check /proc/driver/nvidia/version
                DRIVER_CUDA_VER=$(grep -oP 'CUDA \K[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null || echo "")
            fi

            if [ -n "$DRIVER_CUDA_VER" ]; then
                # Use major.minor for package name (e.g. 12.4 → cuda-nvcc-12-4)
                local CUDA_MAJOR CUDA_MINOR CUDA_PKG_VER
                CUDA_MAJOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f1)
                CUDA_MINOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f2)
                CUDA_PKG_VER="${CUDA_MAJOR}-${CUDA_MINOR}"
                echo "   Driver supports CUDA $DRIVER_CUDA_VER — installing cuda-nvcc-${CUDA_PKG_VER}"
                $SUDO apt-get install -y -qq "cuda-nvcc-${CUDA_PKG_VER}" "cuda-cudart-dev-${CUDA_PKG_VER}" 2>/dev/null \
                    || $SUDO apt-get install -y -qq "cuda-nvcc-${CUDA_PKG_VER}" 2>/dev/null \
                    || { echo "[!] Failed to install cuda-nvcc-${CUDA_PKG_VER}"; exit 1; }

                # Find and export the newly installed CUDA path
                for cuda_dir in /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin /usr/local/cuda-${CUDA_MAJOR}/bin; do
                    if [ -x "$cuda_dir/nvcc" ]; then
                        export PATH="$cuda_dir:$PATH"
                        export CUDA_PATH="$(dirname "$cuda_dir")"
                        echo "[ok] nvcc installed at $cuda_dir"
                        CUDA_FOUND=1
                        break
                    fi
                done
            else
                echo "[!] Cannot detect CUDA version from driver"
                echo "    Install manually: apt-get install -y cuda-nvcc-<MAJOR>-<MINOR>"
                echo "    Then set: export PATH=/usr/local/cuda/bin:\$PATH"
                exit 1
            fi
        fi

        if [ "$CUDA_FOUND" -eq 0 ]; then
            echo "[!] CUDA toolkit installation failed"
            exit 1
        fi

        # Auto-detect GPU architecture for EQUIUM_CUDA_ARCH
        if [ -z "${EQUIUM_CUDA_ARCH:-}" ] && command -v nvidia-smi &>/dev/null; then
            local GPU_NAME
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            if [ -n "$GPU_NAME" ]; then
                # Map GPU name to CUDA compute capability
                case "$GPU_NAME" in
                    *"1030"*|*"1050"*|*"1060"*|*"1070"*|*"1080"*) export EQUIUM_CUDA_ARCH="sm_61" ;;
                    *"1660"*|*"1650"*|*"T600"*|*"T400"*) export EQUIUM_CUDA_ARCH="sm_75" ;;
                    *"2060"*|*"2070"*|*"2080"*|*"T4"*) export EQUIUM_CUDA_ARCH="sm_75" ;;
                    *"3050"*|*"3060"*|*"3070"*|*"3080"*|*"3090"*|*"A40"*|*"A100"*) export EQUIUM_CUDA_ARCH="sm_86" ;;
                    *"4050"*|*"4060"*|*"4070"*|*"4080"*|*"4090"*) export EQUIUM_CUDA_ARCH="sm_89" ;;
                    *"5090"*|*"5080"*|*"5070"*|*"5060"*) export EQUIUM_CUDA_ARCH="sm_100" ;;
                    *) export EQUIUM_CUDA_ARCH="sm_75" ;; # safe default
                esac
                echo "[ok] GPU: $GPU_NAME → arch $EQUIUM_CUDA_ARCH"
            fi
        fi
    fi

    # Check Python modules (pynacl, base58)
    local PIP_PKGS=()
    if ! python3 -c "import nacl" &>/dev/null; then
        echo "[!] pynacl not found"
        PIP_PKGS+=("pynacl")
    else
        echo "[ok] pynacl"
    fi

    if ! python3 -c "import base58" &>/dev/null; then
        echo "[!] base58 not found"
        PIP_PKGS+=("base58")
    else
        echo "[ok] base58"
    fi

    if [ ${#PIP_PKGS[@]} -gt 0 ]; then
        echo ""
        echo "Installing Python packages: ${PIP_PKGS[*]}"
        pip3 install --user --break-system-packages "${PIP_PKGS[@]}" 2>/dev/null \
            || pip3 install --user "${PIP_PKGS[@]}"
    fi

    echo ""
    echo "=== All prerequisites OK ==="
}

# Check for --deps flag to force install
if [[ "${1:-}" == "--deps" ]]; then
    install_prerequisites
    shift
fi

# ============================================================
# Config validation
# ============================================================
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env not found at $ENV_FILE"
    echo "Copy .env.example to .env and fill in your values:"
    echo "  cp .env.example .env"
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

# ============================================================
# Auto-install prerequisites if binary not found
# ============================================================
if [ ! -f "$MINER" ]; then
    echo "Miner binary not found. Installing prerequisites and building..."
    echo ""
    install_prerequisites

    MODE="${MODE:-cpu}"
    echo ""
    echo "Building equium-miner ($MODE mode)..."
    if [ "$MODE" = "gpu" ]; then
        cargo build --release --features gpu -p equium-cli-miner
    else
        cargo build --release -p equium-cli-miner
    fi
    echo ""
fi

# ============================================================
# Mining mode & thread calculation
# ============================================================
MODE="${MODE:-cpu}"
TOTAL_CORES=$(nproc)

if [ "$MODE" = "cpu" ] && [ "${THREADS:-0}" -eq 0 ] 2>/dev/null; then
    THREADS=$((TOTAL_CORES - 1))
    [ "$THREADS" -lt 1 ] && THREADS=1
fi

# ============================================================
# Convert private key to Solana keypair JSON
# ============================================================
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

# ============================================================
# Build args and run
# ============================================================
ARGS=(
    --rpc-url "$RPC_URL"
    --keypair "$TMPKEY"
    --max-nonces-per-round "$MAX_NONCES_PER_ROUND"
    --max-blocks "$MAX_BLOCKS"
    --cu-limit "$CU_LIMIT"
)

if [ "$MODE" = "gpu" ]; then
    ARGS+=(--gpu-batch "${GPU_BATCH:-64}")
    echo "Starting Equium Miner (GPU)..."
    echo "  RPC: $RPC_URL"
    echo "  GPU Batch: ${GPU_BATCH:-64} nonces/launch (~$(( ${GPU_BATCH:-64} * 37 ))MB VRAM)"
else
    ARGS+=(--threads "$THREADS")
    echo "Starting Equium Miner (CPU)..."
    echo "  RPC: $RPC_URL"
    echo "  Threads: $THREADS / $TOTAL_CORES cores (1 reserved for system)"
fi

echo "  Max Blocks: $MAX_BLOCKS (0=infinite)"
echo ""

exec "$MINER" "${ARGS[@]}"
