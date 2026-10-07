#!/usr/bin/env bash
# nomadic installer — downloads a release bundle from GitHub and installs the
# standalone `nomadic` / `nomadic-db` binaries. No Python environment required.
#
#   curl -fsSL https://raw.githubusercontent.com/lancemk/nomadic/main/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --version v0.2.0 --with-mlx
#
# Also works from inside an extracted release bundle (uses the local bin/).
set -euo pipefail

REPO="${NOMADIC_REPO:-lancemk/nomadic}"
VERSION="${NOMADIC_VERSION:-latest}"
INSTALL_DIR="${NOMADIC_INSTALL_DIR:-${HOME}/.local/bin}"
# Mirrors nomadic/paths.py: $NOMADIC_HOME/data, else $XDG_DATA_HOME/nomadic, else ~/.local/share/nomadic.
if [[ -n "${NOMADIC_HOME:-}" ]]; then DATA_DIR="${NOMADIC_HOME}/data"
else DATA_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/nomadic"; fi
WITH_MLX=0
UNINSTALL=0
DRY_RUN=0

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
run()  { if [[ "${DRY_RUN}" == 1 ]]; then printf '[dry-run] %s\n' "$*"; else "$@"; fi; }

usage() {
  cat <<'EOF'
Usage: install.sh [options]

  --version <tag|latest>  Release to install (default: latest; e.g. v0.2.0)
  --prefix <dir>          Install directory for binaries (default: ~/.local/bin)
  --with-mlx              Also create a managed venv with mlx-lm/mlx-vlm for
                          `nomadic serve` (Apple Silicon only)
  --repo <owner/name>     GitHub repository (default: lancemk/nomadic)
  --uninstall             Remove installed binaries (and managed MLX venv)
  --dry-run               Print actions without changing anything
  -h, --help              Show this help

Environment: NOMADIC_VERSION, NOMADIC_INSTALL_DIR, NOMADIC_REPO, NOMADIC_HOME
(NOMADIC_HOME relocates all nomadic config/data/cache/state under one root; run
`nomadic paths` after install to see the resolved locations.)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)   VERSION="${2:?--version needs a value}"; shift 2 ;;
    --prefix)    INSTALL_DIR="${2:?--prefix needs a value}"; shift 2 ;;
    --repo)      REPO="${2:?--repo needs a value}"; shift 2 ;;
    --with-mlx)  WITH_MLX=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           usage >&2; die "unknown option: $1" ;;
  esac
done

if [[ "${UNINSTALL}" == 1 ]]; then
  log "Removing nomadic from ${INSTALL_DIR}"
  run rm -f "${INSTALL_DIR}/nomadic" "${INSTALL_DIR}/nomadic-db"
  run rm -rf "${DATA_DIR}/venv"
  log "Uninstalled. Config in ~/.config/nomadic and session databases were left untouched."
  exit 0
fi

detect_platform() {
  local os arch
  os="$(uname -s)"; arch="$(uname -m)"
  case "${os}:${arch}" in
    Darwin:arm64|Darwin:aarch64) echo "macos-arm64" ;;
    Linux:x86_64|Linux:amd64)    echo "linux-x86_64" ;;
    Linux:arm64|Linux:aarch64)   echo "linux-arm64" ;;
    *) die "unsupported platform ${os}/${arch} (supported: macOS arm64, Linux x86_64/arm64)" ;;
  esac
}

need() { command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

install_binaries() {
  local src="$1"
  [[ -f "${src}/nomadic" ]] || die "bundle has no bin/nomadic (built without PyInstaller?)"
  run mkdir -p "${INSTALL_DIR}"
  run install -m 0755 "${src}/nomadic" "${INSTALL_DIR}/nomadic"
  [[ -f "${src}/nomadic-db" ]] && run install -m 0755 "${src}/nomadic-db" "${INSTALL_DIR}/nomadic-db"
  return 0
}

setup_mlx() {
  [[ "$(detect_platform)" == "macos-arm64" ]] || die "--with-mlx requires Apple Silicon (macOS arm64)"
  log "Creating MLX runtime venv at ${DATA_DIR}/venv"
  run mkdir -p "${DATA_DIR}"
  if command -v uv >/dev/null 2>&1; then
    run uv venv --python 3.14 "${DATA_DIR}/venv"
    run uv pip install --python "${DATA_DIR}/venv/bin/python" mlx-lm mlx-vlm
  else
    need python3
    run python3 -m venv "${DATA_DIR}/venv"
    run "${DATA_DIR}/venv/bin/python" -m pip install --upgrade pip mlx-lm mlx-vlm
  fi
}

PLATFORM="$(detect_platform)"
SCRIPT_DIR=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ -n "${SCRIPT_DIR}" && -f "${SCRIPT_DIR}/bin/nomadic" ]]; then
  # Running from an extracted release bundle.
  log "Installing from local bundle (${PLATFORM})"
  install_binaries "${SCRIPT_DIR}/bin"
else
  need curl; need tar
  if [[ "${VERSION}" == "latest" ]]; then
    log "Resolving latest release of ${REPO}"
    final_url="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest")" \
      || die "could not reach github.com/${REPO}"
    VERSION="${final_url##*/}"
    [[ "${VERSION}" == v* ]] || die "no published release found for ${REPO}"
  fi
  [[ "${VERSION}" == v* ]] || VERSION="v${VERSION}"
  ASSET="nomadic-${VERSION#v}-${PLATFORM}.tar.gz"
  BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

  WORK="$(mktemp -d)"
  trap 'rm -rf "${WORK}"' EXIT

  log "Downloading ${ASSET}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '[dry-run] curl %s/%s (+ SHA256SUMS), verify, extract, install to %s\n' "${BASE_URL}" "${ASSET}" "${INSTALL_DIR}"
  else
    curl -fSL --retry 3 -o "${WORK}/${ASSET}" "${BASE_URL}/${ASSET}" \
      || die "download failed — does ${VERSION} publish a ${PLATFORM} bundle?"
    curl -fsSL --retry 3 -o "${WORK}/SHA256SUMS" "${BASE_URL}/SHA256SUMS" \
      || die "could not fetch SHA256SUMS for ${VERSION}"
    expected="$(awk -v f="${ASSET}" '$2==f || $2=="*"f {print $1}' "${WORK}/SHA256SUMS")"
    [[ -n "${expected}" ]] || die "${ASSET} not listed in SHA256SUMS"
    [[ "$(sha256_of "${WORK}/${ASSET}")" == "${expected}" ]] || die "checksum mismatch for ${ASSET}"
    log "Checksum verified"
    tar -xzf "${WORK}/${ASSET}" -C "${WORK}"
    install_binaries "${WORK}/${ASSET%.tar.gz}/bin"
  fi
fi

[[ "${WITH_MLX}" == 1 ]] && setup_mlx

log "Installed to ${INSTALL_DIR}"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *) warn "${INSTALL_DIR} is not on your PATH. Add: export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac
[[ "${DRY_RUN}" == 1 ]] || "${INSTALL_DIR}/nomadic" --version || true
[[ "${WITH_MLX}" == 1 ]] || log "Tip: re-run with --with-mlx on Apple Silicon to enable \`nomadic serve\`."
