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
CHANNEL="${NOMADIC_CHANNEL:-stable}"
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
  --channel <name>        stable (default) or a pre-release channel such as rc or
                          staging: installs the newest release tagged v*-<name>*
  --prefix <dir>          Install directory for binaries (default: ~/.local/bin)
  --with-mlx              Also create a managed venv with mlx-lm/mlx-vlm for
                          `nomadic serve` (Apple Silicon only)
  --repo <owner/name>     GitHub repository (default: lancemk/nomadic)
  --uninstall             Remove installed binaries (and managed MLX venv)
  --dry-run               Print actions without changing anything
  -h, --help              Show this help

Environment: NOMADIC_RELEASE_BASE_URL (mirror/test override), NOMADIC_VERSION, NOMADIC_CHANNEL, NOMADIC_INSTALL_DIR, NOMADIC_REPO, NOMADIC_HOME
(NOMADIC_HOME relocates all nomadic config/data/cache/state under one root; run
`nomadic paths` after install to see the resolved locations.)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)   VERSION="${2:?--version needs a value}"; shift 2 ;;
    --prefix)    INSTALL_DIR="${2:?--prefix needs a value}"; shift 2 ;;
    --channel)   CHANNEL="${2:?--channel needs a value}"; shift 2 ;;
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
  run rm -rf "${DATA_DIR}/venv" "${DATA_DIR}/app"
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

# Bundles ship a PyInstaller *onedir* app (fast start: nothing is unpacked per launch) plus
# bin/ symlinks. Older bundles shipped single-file binaries, still handled below.
install_binaries() {
  local root="$1"
  run mkdir -p "${INSTALL_DIR}"
  if [[ -x "${root}/app/nomadic" ]]; then
    log "Installing app to ${DATA_DIR}/app"
    run mkdir -p "${DATA_DIR}"
    run rm -rf "${DATA_DIR}/app.new"
    run cp -R "${root}/app" "${DATA_DIR}/app.new"
    run rm -rf "${DATA_DIR}/app"
    run mv "${DATA_DIR}/app.new" "${DATA_DIR}/app"
    run ln -sfn "${DATA_DIR}/app/nomadic" "${INSTALL_DIR}/nomadic"
    run ln -sfn "${DATA_DIR}/app/nomadic" "${INSTALL_DIR}/nomadic-db"
  elif [[ -f "${root}/bin/nomadic" ]]; then
    run rm -rf "${DATA_DIR}/app"
    run rm -f "${INSTALL_DIR}/nomadic" "${INSTALL_DIR}/nomadic-db"
    run install -m 0755 "${root}/bin/nomadic" "${INSTALL_DIR}/nomadic"
    [[ -f "${root}/bin/nomadic-db" ]] && run install -m 0755 "${root}/bin/nomadic-db" "${INSTALL_DIR}/nomadic-db"
  else
    die "bundle has no app/ or bin/nomadic (built without PyInstaller?)"
  fi
  return 0
}

# Newest release tag for a pre-release channel (e.g. rc -> v0.4.0-rc.2). The API lists newest first.
resolve_channel_tag() {
  local channel="$1" auth=()
  [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
  curl -fsSL "${auth[@]}" "https://api.github.com/repos/${REPO}/releases?per_page=50" \
    | grep -o '"tag_name": *"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/' \
    | grep -E -- "-${channel}" | head -n 1
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

if [[ -n "${SCRIPT_DIR}" && ( -x "${SCRIPT_DIR}/app/nomadic" || -f "${SCRIPT_DIR}/bin/nomadic" ) ]]; then
  # Running from an extracted release bundle.
  log "Installing from local bundle (${PLATFORM})"
  install_binaries "${SCRIPT_DIR}"
else
  need curl; need tar
  if [[ "${VERSION}" == "latest" && "${CHANNEL}" != "stable" ]]; then
    log "Resolving newest '${CHANNEL}' release of ${REPO}"
    VERSION="$(resolve_channel_tag "${CHANNEL}")" || true
    [[ -n "${VERSION}" ]] || die "no '${CHANNEL}' release found for ${REPO}"
  elif [[ "${VERSION}" == "latest" ]]; then
    log "Resolving latest release of ${REPO}"
    final_url="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest")" \
      || die "could not reach github.com/${REPO}"
    VERSION="${final_url##*/}"
    [[ "${VERSION}" == v* ]] || die "no published release found for ${REPO}"
  fi
  [[ "${VERSION}" == v* ]] || VERSION="v${VERSION}"
  BASE_URL="${NOMADIC_RELEASE_BASE_URL:-https://github.com/${REPO}/releases/download}/${VERSION}"

  WORK="$(mktemp -d)"
  trap 'rm -rf "${WORK}"' EXIT

  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '[dry-run] curl %s/SHA256SUMS + nomadic-*-%s.tar.gz, verify, extract, install to %s\n' "${BASE_URL}" "${PLATFORM}" "${INSTALL_DIR}"
  else
    curl -fsSL --retry 3 -o "${WORK}/SHA256SUMS" "${BASE_URL}/SHA256SUMS" \
      || die "could not fetch SHA256SUMS for ${VERSION}"
    # The asset name carries the PEP 440 version (0.4.0rc1), which can differ from the tag
    # spelling (v0.4.0-rc.1), so read it from the checksum list instead of deriving it.
    ASSET="$(awk -v p="-${PLATFORM}.tar.gz" 'index($2, p) {sub(/^\*/, "", $2); print $2; exit}' "${WORK}/SHA256SUMS")"
    [[ -n "${ASSET}" ]] || die "${VERSION} publishes no ${PLATFORM} bundle"
    log "Downloading ${ASSET}"
    curl -fSL --retry 3 -o "${WORK}/${ASSET}" "${BASE_URL}/${ASSET}" || die "download failed"
    expected="$(awk -v f="${ASSET}" '$2==f || $2=="*"f {print $1}' "${WORK}/SHA256SUMS")"
    [[ "$(sha256_of "${WORK}/${ASSET}")" == "${expected}" ]] || die "checksum mismatch for ${ASSET}"
    log "Checksum verified"
    tar -xzf "${WORK}/${ASSET}" -C "${WORK}"
    install_binaries "${WORK}/${ASSET%.tar.gz}"
  fi
fi

[[ "${WITH_MLX}" == 1 ]] && setup_mlx

log "Installed to ${INSTALL_DIR}"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *) warn "${INSTALL_DIR} is not on your PATH. Add: export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac
[[ "${DRY_RUN}" == 1 ]] || "${INSTALL_DIR}/nomadic" --version || true
if [[ "${DRY_RUN}" != 1 ]]; then
  # nomadic finds mlx-lm / mlx-vlm in a `uv tool` environment (or the managed venv) on its own.
  if ! "${INSTALL_DIR}/nomadic" runtime >/dev/null 2>&1; then
    log "MLX runtime not found (needed for \`nomadic serve\`). Install it with one of:"
    log "  uv tool install mlx-lm            # add: --with mlx-vlm   for vision models"
    log "  install.sh --with-mlx             # managed venv"
    log "Then check with: nomadic runtime"
  else
    "${INSTALL_DIR}/nomadic" runtime || true
  fi
fi
