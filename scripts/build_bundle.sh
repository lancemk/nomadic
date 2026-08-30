#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/release_env.sh"

REPO_ROOT="$(nomadic_repo_root)"
VERSION="$(nomadic_version "${REPO_ROOT}")"
PLATFORM_ID="$(nomadic_platform_id)"
BUNDLE_STEM="$(nomadic_bundle_stem "${VERSION}" "${PLATFORM_ID}")"

DIST_DIR="${NOMADIC_DIST_DIR:-${REPO_ROOT}/dist}"
ARTIFACT_DIR="${NOMADIC_ARTIFACT_DIR:-${REPO_ROOT}/artifacts}"
TMP_ROOT="${NOMADIC_BUNDLE_TMP_DIR:-${REPO_ROOT}/.bundle-tmp}"
SKIP_PYTHON_BUILD="${NOMADIC_BUILD_SKIP_PYTHON_BUILD:-0}"

mkdir -p "${DIST_DIR}" "${ARTIFACT_DIR}" "${TMP_ROOT}"

if [[ "${SKIP_PYTHON_BUILD}" != "1" ]]; then
  if ! python3 -m build --version >/dev/null 2>&1; then
    printf 'python -m build is required. Install it with: python3 -m pip install build\n' >&2
    exit 1
  fi

  rm -f "${DIST_DIR}"/nomadic-*
  (
    cd "${REPO_ROOT}"
    python3 -m build --sdist --wheel --outdir "${DIST_DIR}"
  )
fi

if ! find "${DIST_DIR}" -maxdepth 1 -type f \( -name 'nomadic-*.whl' -o -name 'nomadic-*.tar.gz' \) | grep -q .; then
  printf 'No build outputs were found in %s\n' "${DIST_DIR}" >&2
  exit 1
fi

BUNDLE_ROOT="${TMP_ROOT}/${BUNDLE_STEM}"
ARCHIVE_PATH="${ARTIFACT_DIR}/${BUNDLE_STEM}.tar.gz"
rm -rf "${BUNDLE_ROOT}" "${ARCHIVE_PATH}"
mkdir -p "${BUNDLE_ROOT}/dist" "${BUNDLE_ROOT}/sql" "${BUNDLE_ROOT}/docs"

cp "${DIST_DIR}"/nomadic-* "${BUNDLE_ROOT}/dist/"
cp "${REPO_ROOT}/README.md" "${BUNDLE_ROOT}/"
cp "${REPO_ROOT}/CHANGELOG.md" "${BUNDLE_ROOT}/"
cp "${REPO_ROOT}/LICENSE.md" "${BUNDLE_ROOT}/"
cp "${REPO_ROOT}/GUIDE.md" "${BUNDLE_ROOT}/docs/"
cp "${REPO_ROOT}/nomadic/sql/"*.sql "${BUNDLE_ROOT}/sql/"

if python3 -c "import PyInstaller" >/dev/null 2>&1; then
  printf 'Building standalone CLI executables via PyInstaller...\n'
  mkdir -p "${BUNDLE_ROOT}/bin"
  python3 -m PyInstaller --onefile --name nomadic "${REPO_ROOT}/nomadic_bin.py" --add-data "${REPO_ROOT}/nomadic/sql:nomadic/sql" --clean --workpath "${TMP_ROOT}/build" --distpath "${BUNDLE_ROOT}/bin" --specpath "${TMP_ROOT}"
  python3 -m PyInstaller --onefile --name nomadic-db "${REPO_ROOT}/db.py" --add-data "${REPO_ROOT}/nomadic/sql:nomadic/sql" --clean --workpath "${TMP_ROOT}/build" --distpath "${BUNDLE_ROOT}/bin" --specpath "${TMP_ROOT}"
  printf 'Standalone CLI executables built successfully.\n'
else
  printf 'PyInstaller not found. Skipping standalone binary compilation.\n'
fi

cat > "${BUNDLE_ROOT}/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "${SCRIPT_DIR}/bin" ] && [ -f "${SCRIPT_DIR}/bin/nomadic" ]; then
  INSTALL_DIR="${1:-/usr/local/bin}"
  printf "Found standalone binaries. Installing to %s...\n" "${INSTALL_DIR}"
  if [ -w "${INSTALL_DIR}" ]; then
    cp "${SCRIPT_DIR}/bin/nomadic" "${INSTALL_DIR}/nomadic"
    if [ -f "${SCRIPT_DIR}/bin/nomadic-db" ]; then
      cp "${SCRIPT_DIR}/bin/nomadic-db" "${INSTALL_DIR}/nomadic-db"
    fi
  else
    printf "Write permission denied for %s. Retrying with sudo...\n" "${INSTALL_DIR}"
    sudo cp "${SCRIPT_DIR}/bin/nomadic" "${INSTALL_DIR}/nomadic"
    if [ -f "${SCRIPT_DIR}/bin/nomadic-db" ]; then
      sudo cp "${SCRIPT_DIR}/bin/nomadic-db" "${INSTALL_DIR}/nomadic-db"
    fi
  fi
  printf 'Successfully installed nomadic standalone CLI.\n'
else
  python3 -m pip install "${SCRIPT_DIR}"/dist/nomadic-*.whl
  printf 'Installed nomadic from bundled wheel.\n'
fi
EOF
chmod +x "${BUNDLE_ROOT}/install.sh"

cat > "${BUNDLE_ROOT}/BUILD_INFO.txt" <<EOF
Project: nomadic
Version: ${VERSION}
Platform: ${PLATFORM_ID}
Generated-At: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cat > "${BUNDLE_ROOT}/release-metadata.json" <<EOF
{"version":"${VERSION}","platform":"${PLATFORM_ID}","bundle":"${BUNDLE_STEM}.tar.gz"}
EOF

(
  cd "${TMP_ROOT}"
  tar -czf "${ARCHIVE_PATH}" "${BUNDLE_STEM}"
)

printf '%s\n' "${ARCHIVE_PATH}"

