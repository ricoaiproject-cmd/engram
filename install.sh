#!/bin/sh
# engram installer for macOS / Linux
# Usage: curl -LsSf https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.sh | sh
#        or: ./install.sh
#        or: ./install.sh /path/to/local/engram
set -e

SOURCE="${1:-git+https://github.com/ricoaiproject-cmd/engram.git}"

echo ""
echo "========================================"
echo " engram installer"
echo "========================================"
echo ""

# ----------------------------------------------------------------
# Step 1: uv
# ----------------------------------------------------------------
echo "[1/4] Checking for uv..."
if command -v uv >/dev/null 2>&1; then
    echo "  uv is already installed: $(command -v uv)"
else
    echo "  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv visible in this session
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo ""
        echo "[ERROR] uv is still not on PATH after installation."
        echo "  Restart your terminal and re-run this script, or install manually:"
        echo "  https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "  uv installed: $(command -v uv)"
fi
echo ""

# ----------------------------------------------------------------
# Step 2: git (required to fetch git+ sources)
# ----------------------------------------------------------------
echo "[2/4] Checking for git..."
case "$SOURCE" in
    git+*)
        if command -v git >/dev/null 2>&1; then
            echo "  git is already installed: $(command -v git)"
        else
            echo ""
            echo "[ERROR] git not found."
            echo "  macOS: run 'xcode-select --install' (or 'brew install git'), then re-run this script."
            echo "  Linux: install git with your package manager (e.g. 'sudo apt install git')."
            exit 1
        fi
        ;;
    *)
        echo "  Local source — git not required (skipped)."
        ;;
esac
echo ""

# ----------------------------------------------------------------
# Step 3: install engram
# ----------------------------------------------------------------
# Note: uv-MANAGED Python is required — its SQLite build supports loadable
# extensions (needed by sqlite-vec). System / python.org builds on macOS do
# not, and uv would otherwise prefer them when present, so force the managed
# distribution explicitly.
echo "[3/4] Installing engram..."
echo "  Source: $SOURCE"
echo "  (The first run may download Python 3.12.)"
echo ""

UV_PYTHON_PREFERENCE=only-managed uv tool install --python 3.12 --force "$SOURCE"

export PATH="$HOME/.local/bin:$PATH"
echo ""
echo "  engram installed."
echo ""

# ----------------------------------------------------------------
# Step 4: setup wizard
# ----------------------------------------------------------------
echo "[4/4] Running the setup wizard..."
echo ""

if command -v engram >/dev/null 2>&1; then
    engram setup
else
    echo "[ERROR] engram command not found."
    echo "  Restart your terminal and run 'engram setup'."
    exit 1
fi

echo ""
echo "========================================"
echo " Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Restart your agent (e.g. Claude Code)"
echo "  - Verify with: engram doctor"
echo ""
