#!/usr/bin/env bash

set -euo pipefail

nomadic_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

nomadic_version() {
  local repo_root
  repo_root="${1:-$(nomadic_repo_root)}"
  python3 -c 'import pathlib, tomllib; root = pathlib.Path("'"${repo_root}"'"); print(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])'
}

nomadic_branch_class() {
  local branch_name
  branch_name="${1:-}"
  if [[ "${branch_name}" == "main" ]]; then
    printf 'main\n'
  elif [[ "${branch_name}" == develop/* ]]; then
    printf 'develop\n'
  else
    printf 'other\n'
  fi
}

nomadic_is_changelog_branch() {
  local branch_name
  branch_name="${1:-}"
  [[ "${branch_name}" == "main" || "${branch_name}" == develop/* ]]
}

nomadic_platform_id() {
  if [[ -n "${NOMADIC_PLATFORM_ID_OVERRIDE:-}" ]]; then
    printf '%s\n' "${NOMADIC_PLATFORM_ID_OVERRIDE}"
    return
  fi

  local os_name arch_name
  os_name="$(uname -s)"
  arch_name="$(uname -m)"

  case "${os_name}:${arch_name}" in
    Darwin:arm64|Darwin:aarch64)
      printf 'macos-arm64\n'
      ;;
    Darwin:x86_64)
      printf 'macos-x86_64\n'
      ;;
    Linux:x86_64)
      printf 'linux-x86_64\n'
      ;;
    Linux:arm64|Linux:aarch64)
      printf 'linux-arm64\n'
      ;;
    *)
      printf '%s-%s\n' "$(printf '%s' "${os_name}" | tr '[:upper:]' '[:lower:]')" "${arch_name}"
      ;;
  esac
}

nomadic_bundle_stem() {
  local version platform_id
  version="${1:?version required}"
  platform_id="${2:?platform id required}"
  printf 'nomadic-%s-%s\n' "${version}" "${platform_id}"
}
