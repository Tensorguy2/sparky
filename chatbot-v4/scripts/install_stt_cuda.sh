#!/usr/bin/env bash
# Build CTranslate2 with CUDA on aarch64 (PyPI wheels are CPU-only on ARM).
# Installs to ~/.local/ctranslate2 without sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

PYDEV_ROOT="${PYDEV_ROOT:-$HOME/.local/python3.12-dev}"
CT2_PREFIX="${CT2_PREFIX:-$HOME/.local/ctranslate2}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CT2_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME"
echo "CT2_PREFIX=$CT2_PREFIX"
nvcc --version

# ── Python headers (extract libpython3.12-dev if missing) ───────────────────
if [[ ! -f "$PYDEV_ROOT/usr/include/python3.12/Python.h" ]]; then
  echo "Fetching libpython3.12-dev headers into $PYDEV_ROOT ..."
  mkdir -p /tmp/pydev-dl
  (cd /tmp/pydev-dl && apt-get download libpython3.12-dev)
  rm -rf "$PYDEV_ROOT"
  dpkg-deb -x /tmp/pydev-dl/libpython3.12-dev_*.deb "$PYDEV_ROOT"
fi

export CPLUS_INCLUDE_PATH="${PYDEV_ROOT}/usr/include/python3.12:${PYDEV_ROOT}/usr/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="${PYDEV_ROOT}/usr/lib/aarch64-linux-gnu:${LIBRARY_PATH:-}"
export CPPFLAGS="-I${PYDEV_ROOT}/usr/include/python3.12 -I${PYDEV_ROOT}/usr/include ${CPPFLAGS:-}"
export CXXFLAGS="${CPPFLAGS} ${CXXFLAGS:-}"

# ── CTranslate2 source ───────────────────────────────────────────────────────
CT2_TAG="${CT2_TAG:-v4.7.2}"
CT2_SRC="${CT2_SRC:-/tmp/CTranslate2}"
if [[ ! -d "$CT2_SRC/.git" ]]; then
  git clone --recursive --depth 1 --branch "$CT2_TAG" \
    https://github.com/OpenNMT/CTranslate2.git "$CT2_SRC"
else
  (cd "$CT2_SRC" && git submodule update --init --recursive)
fi

# ── Build C++ library with CUDA ──────────────────────────────────────────────
BUILD_DIR="$CT2_SRC/build-cuda"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_CUDA_ARCH="${CT2_CUDA_ARCH_LIST:-native}"
echo "Configuring CTranslate2 (CUDA arch: $CMAKE_CUDA_ARCH) ..."
cmake .. \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CT2_PREFIX" \
  -DOPENMP_RUNTIME=COMP \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=OFF \
  -DWITH_MKL=OFF \
  -DWITH_RUY=ON \
  -DBUILD_CLI=OFF \
  -DBUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCH"

NPROC="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
echo "Compiling CTranslate2 ($NPROC jobs) ..."
cmake --build . -j"$NPROC"
cmake --install .

export CTRANSLATE2_ROOT="$CT2_PREFIX"
export LD_LIBRARY_PATH="${CT2_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ── Python bindings ──────────────────────────────────────────────────────────
echo "Installing Python bindings ..."
pip install -U pip wheel setuptools pybind11
pip uninstall -y ctranslate2 2>/dev/null || true
pip install --no-build-isolation -r "$CT2_SRC/python/install_requirements.txt" 2>/dev/null || true
pip install --no-build-isolation "$CT2_SRC/python"

python -c "
import ctranslate2 as c
n = c.get_cuda_device_count()
print('ctranslate2', c.__version__, 'CUDA devices:', n)
if n < 1:
    raise SystemExit('Build finished but CUDA is not visible to CTranslate2')
"

echo ""
echo "Done. Add to chatbot/.env (optional):"
echo "  WHISPER_DEVICE=auto"
echo "  WHISPER_COMPUTE_TYPE=auto"
echo ""
echo "Ensure LD_LIBRARY_PATH includes ${CT2_PREFIX}/lib when starting the chatbot."
echo "Restart the chatbot server."
