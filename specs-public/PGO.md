# PGO: Profile-Guided Optimization for Python CLI artifacts

A practical guide for shipping Python command-line tools as installable artifacts that start fast. It starts with the incident that prompted it in this repository (slow first load after `install.sh`), then generalises to a repeatable measure → pinpoint → fix → gate loop.

"PGO" in the strict sense is a compiler technique: build an instrumented binary, run a representative workload, and let the compiler optimise for what actually ran. This guide uses the term for the whole loop, because for a CLI the same principle decides where the wins are: **profile first, optimise what the profile shows, and make the profile a CI gate.** Section 5 covers PGO proper (the interpreter) and explains why it is the smallest lever for a CLI.

Measured numbers in this document come from one Linux x86_64 host (4 vCPU Xeon @ 2.8 GHz, CPython 3.13, PyInstaller 6.22.3), medians of 7–9 warm runs unless stated. **macOS was not measured**; statements about macOS are marked as such.

---

## 1. Incident: slow load after `curl … | bash`

### Symptom
Installing a `staging` / `rc` (and stable) release via `install.sh` and then running `nomadic` felt slow. The installer itself ends by running `nomadic --version`, so the first thing a user saw was a multi-second pause.

### Method
1. Reproduce with the shipped artifact, not the source tree (`pyinstaller` with the exact flags from `scripts/build_bundle.sh`).
2. Separate **bootloader/packaging cost** from **Python import cost** by building the same entry point both ways (onefile vs onedir) and comparing with source (`python -m nomadic`).
3. Attribute Python-side cost with `python -X importtime`.
4. Attribute packaging-side cost with `strace -f -c` on the onefile binary.

### Findings (ranked by cost)

| # | Cause | Evidence | Share of `--version` |
|---|-------|----------|----------------------|
| 1 | **PyInstaller `--onefile` re-extracts the whole app on every launch.** The bootloader unpacks ~114 MB to `/tmp/_MEIxxxxxx`, runs, then deletes it. | One `--version` run performs ~30 000 `write` calls, creates 69 directories and unlinks ~300 files (`strace -f -c`). Same code as onedir: 0.72 s vs 0.10 s → **~0.62 s is pure extraction**. | ~0.6–1.0 s |
| 2 | **`--version` imported the entire application.** `nomadic/__init__.py` imported `nomadic.cli`, which imported `rich`, `prompt_toolkit`, `requests` (and transitively `markdown_it`, `urllib3`, …) at module level. | `-X importtime`: `nomadic.cli` = 421 ms cumulative (`prompt_toolkit` 141, `requests` 93, `rich.markdown` 75). | ~0.3–0.4 s |
| 3 | **Frozen imports are slower than source imports** (unpack from the PYZ archive, more files to stat). | Source `python -m nomadic --version` 0.37 s vs onedir 0.60 s, same code. | ~0.2 s |
| 4 | Install-time smoke test pays all of the above once, at the worst moment. | `install.sh` ran `nomadic --version` as its last step. | n/a |
| 5 | *macOS only, not measured here:* unsigned files freshly written by the onefile bootloader are candidates for Gatekeeper/XProtect checks on **every** launch, since the extraction directory is new each time. Expect this to make cause 1 worse than on Linux. Treat as a hypothesis until measured on macOS runners. | – | – |

Root cause in one sentence: *the artifact paid an unpack-and-import tax on every launch, including launches that only print a version.*

### Fixes shipped

| Fix | Where | Effect |
|-----|-------|--------|
| `--onedir` instead of `--onefile`; installer copies the app to `~/.local/share/nomadic/app` and symlinks `nomadic` / `nomadic-db` into `~/.local/bin` | `scripts/build_bundle.sh`, `install.sh` | Removes cause 1 entirely. One app serves both commands (`nomadic.entry` dispatches on `argv[0]`), so the bundle also carries one copy of the runtime instead of two. |
| `--version` fast path that never imports the CLI | `nomadic/entry.py`, `nomadic/version.py` | `--version` becomes a few imports from the stdlib. Fixes causes 2 and 4. |
| Lazy package exports (PEP 562 `__getattr__`) | `nomadic/__init__.py` | `import nomadic.paths` no longer drags in the CLI. |
| Defer `requests`, `prompt_toolkit`, `rich.markdown`, `rich.syntax` until first use | `nomadic/cli.py` | `import nomadic.cli` 421 ms → ~150 ms. `psycopg` was already lazy. |
| Exclude unused stdlib/dev packages from the frozen app | `scripts/build_bundle.sh` | Smaller bundle; no measurable speed change (see "what did not matter"). |
| Startup budget gate | `scripts/check_startup.py`, run by the build script and by CI `binary-smoke` | A regression to per-launch extraction or an eager heavy import fails the build. |

### Results

| Artifact (same commit, `--version`) | Median | `--help` median | Size on disk | Download (tar.gz) |
|-------------------------------------|--------|-----------------|--------------|-------------------|
| Before: onefile, eager imports | **1.29 s** | 1.42 s | 48 MB (×2: `nomadic` and `nomadic-db`) | ~2 × 50 MB |
| onefile + lazy imports | 0.75 s | 0.94 s | 43 MB | 44 MB |
| **After: onedir + lazy imports + fast path** | **0.09–0.10 s** | **0.25–0.32 s** | 114 MB | **44 MB** |
| Source tree, for reference | 0.04 s | 0.19 s | – | – |

`--version` is ~13× faster and `--help` ~5× faster; download size is unchanged for one binary and roughly halved overall, because the second copy is gone. Disk use grows (onedir is uncompressed) — a trade worth making for something executed hundreds of times.

What did **not** matter (measured): `--optimize 2` plus `--strip` changed startup by <10 ms and saved ~3 MB of download. They were left out because `-OO` removes docstrings and asserts that third-party code can depend on, for no startup benefit.

### Reproduce

```bash
# 1. Build both layouts from the same entry point
python3 -m PyInstaller --onefile --name nomadic-onefile --collect-all psycopg --collect-all psycopg_binary nomadic_bin.py
python3 -m PyInstaller --onedir  --name nomadic-onedir  --collect-all psycopg --collect-all psycopg_binary nomadic_bin.py

# 2. Warm-run medians (any harness works; hyperfine is ideal)
hyperfine --warmup 2 -N 'dist/nomadic-onefile --version' 'dist/nomadic-onedir/nomadic-onedir --version'

# 3. Where does Python time go?
python3 -X importtime -c 'import nomadic.cli' 2> imports.txt
sort -t'|' -k2 -n -r imports.txt | head -20
PYTHONPROFILEIMPORTTIME=1 dist/nomadic-onedir/nomadic-onedir --help   # same, inside a frozen build

# 4. Is the packaging unpacking per launch?
strace -f -c -e trace=write,openat,unlink,mkdir dist/nomadic-onefile --version   # Linux
sudo fs_usage -w -f filesys $(pgrep -n nomadic)                                  # macOS
```

---

## 2. The loop: measure → pinpoint → fix → gate

1. **Define the user-visible latency you care about** and split it: *cold* (first run after install; empty page cache, possibly first-run OS checks) and *warm* (typical). For CLIs the commands that matter are usually `--version`, `--help`, one cheap subcommand, and the real entry (REPL/first request).
2. **Measure the shipped artifact**, never only the source tree. Packaging is frequently the dominant term.
3. **Decompose** into: process start + packaging unpack → interpreter init → imports → your code → first I/O. Each has a different tool (section 3).
4. **Fix the biggest term**, re-measure, repeat until the next term is not worth it. Stop when the number is below perception (≈100 ms for "instant", ≈1 s for "no waiting"), not at zero.
5. **Freeze the result as a budget** in CI (section 7) so it cannot silently regress.

Rule of thumb that the incident confirmed: for short-lived CLIs, **imports and packaging dominate; your own code and interpreter speed barely register.**

---

## 3. Profiling toolbox

| Question | Tool | Notes |
|----------|------|-------|
| Total wall time, with warm-up and statistics | `hyperfine`, or a loop with `time.perf_counter()` (`scripts/check_startup.py`) | Report min **and** median; discard the first run. |
| Which imports are slow? | `python -X importtime`, `PYTHONPROFILEIMPORTTIME=1`; visualise with `tuna` or `importtime-waterfall` | Cumulative column includes children; sort on it. Run it on a fresh interpreter, `.pyc` already written. |
| Interpreter floor | `python -X importtime -c pass`, `python -S -c pass`, `python -I -c pass` | Debian/Ubuntu `python3 -c pass` ≈ 20 ms here. Anything above is yours. |
| What does my code do at runtime? | `cProfile` + `snakeviz`; `py-spy record` / `py-spy top` (sampling, attach to a running process, no code change); `pyinstrument` | Use only after startup is understood. |
| Syscall-level truth (packaging, file churn, network at startup) | `strace -f -c`, `strace -f -ttt -e trace=execve` (Linux); `fs_usage`, `dtruss`, Instruments (macOS) | `execve` timestamps show onefile bootloaders re-exec themselves after extracting. |
| Memory / allocations at startup | `tracemalloc`, `python -X tracemalloc` | Rarely the issue for CLIs. |
| Cold-cache behaviour | Linux: `sync; echo 3 > /proc/sys/vm/drop_caches`; macOS: `sudo purge` | Needed to reproduce "first run after install". |
| Hot loops in C (interpreter PGO analysis) | `perf stat`, `perf record` | Only if you build your own interpreter (section 5). |

---

## 4. Application-level optimisation (the large lever)

### 4.1 Import discipline
- **Fast-path the trivial commands.** `--version`, `--help` of a trivial wrapper, `paths`, shell completion: decide before importing the application. Pattern used here: a tiny `entry.py` that checks `argv` and imports `version.py` only.
- **Make packages lazy.** An `__init__.py` that re-exports from a heavy submodule makes every `import pkg.anything` pay for it. Use PEP 562:
  ```python
  def __getattr__(name):
      if name in _EXPORTS:
          from . import cli
          return getattr(cli, name)
      raise AttributeError(name)
  ```
- **Defer heavy third-party imports into the functions that need them**, or into a one-shot loader for globals:
  ```python
  def _load_prompt_toolkit():
      if "Application" in globals(): return
      from prompt_toolkit.application import Application  # ...
      globals().update(Application=Application, ...)
  ```
  For a module-style dependency, a 10-line `_LazyModule` proxy keeps call sites (`requests.get(...)`) unchanged.
- **Check availability without importing:** `importlib.util.find_spec("pkg") is not None` replaces `try: import pkg`.
- **Type-only imports go under `if TYPE_CHECKING:`** (with `from __future__ import annotations`).
- **No work at import time:** no config reads, network, subprocess calls, regex compiles of large tables, or `sysctl`/`git` calls in module scope. Compute lazily and cache.
- **Know the usual offenders:** `requests` (~90 ms, pulls `urllib3`, `chardet`/`charset_normalizer`), `rich.markdown`/`rich.syntax` (`markdown_it`, `pygments`), `prompt_toolkit` (~140 ms), `pandas`/`numpy`/`torch`, `pydantic` v1 model building, `typing_extensions`-heavy stacks, `email`/`http.client` pulled in by accident, `argparse` subparsers constructing every command's parser.
- **Guard it with a test**, not a code comment. `tests/test_startup.py` asserts in a subprocess that `--version` and `import nomadic` load none of `requests`, `prompt_toolkit`, `rich`, `psycopg`, `pygments`, `markdown_it`.

### 4.2 CLI structure
- Build `argparse` parsers lazily per subcommand, or use a framework with lazy command loading (`click` lazy groups). Constructing 40 subparsers just to run one costs milliseconds each.
- Avoid a REPL-sized dependency tree for non-interactive commands; split optional heavy features (browser automation, ML runtime) behind extras and lazy imports — this repo already keeps MLX out of the binary entirely.
- Print progress/spinners only after the first slow step begins; do not initialise a console library to print one line.

### 4.3 Runtime hot paths (after startup is fine)
Profile with `cProfile`/`py-spy`, fix algorithms and I/O first (batching, streaming, caching, avoiding re-parsing), then consider `__slots__`, `functools.cache`, compiled helpers. None of this moves a 100 ms startup.

---

## 5. Interpreter-level PGO (PGO proper)

CPython can be compiled with profile-guided optimisation and link-time optimisation:

```bash
./configure --enable-optimizations --with-lto   # PGO + LTO
make -j"$(nproc)"                               # builds, runs the training suite, rebuilds
```
`--enable-optimizations` makes the build run an instrumented interpreter over a training task (`PROFILE_TASK`, by default a subset of the regression tests), then recompiles using the collected profile. CPython's own documentation and benchmarks put the generic gain in the **~10 % range** for interpreter-bound code; consult the "Optimizations" notes of your target release for current figures.

### What it does and does not do for a CLI
- It speeds up the **bytecode interpreter loop**. Startup of a CLI is dominated by filesystem work, unpacking and imports (which are mostly the interpreter executing module bodies — a few percent faster, not 10×).
- Expected effect on the incident above: low single-digit percent of the remaining ~100 ms. **Do it last, or get it for free** by using a distribution that already ships a PGO/LTO interpreter.

### Getting a PGO/LTO interpreter without building one
- Official python.org macOS/Windows installers and the `python-build-standalone` builds used by `uv` are built with PGO and LTO for most targets (confirm for the exact release you pin).
- **Check, don't assume:**
  ```bash
  python -c "import sysconfig; print(sysconfig.get_config_var('CONFIG_ARGS'))" | tr ' ' '\n' | grep -E 'optimizations|lto'
  ```
  On the Debian system used for the measurements above this printed nothing, i.e. the distro flags were not visible there — which is exactly why you check the interpreter your release job actually freezes.
- Pin the interpreter in CI (`actions/setup-python` with an exact `python-version`; or `uv python install 3.14`) and record `sys.version`, `sysconfig` build flags in the release metadata so a build-environment change shows up as a diff.

### Training on your own workload (advanced)
If you do build CPython yourself (for example a vendored runtime shipped inside the artifact), replace the training task with your tool's real commands so hot paths are the ones your users hit:
```bash
make profile-opt PROFILE_TASK='-m nomadic --help'   # illustrative: any script/module the built python can run
```
Keep the training workload deterministic, offline and short; refresh it when commands change; and validate with section 7's budget, because a bad profile can make things slower. Do not do this unless measurement shows the interpreter loop (not startup) is your bottleneck — e.g. a long-running agent loop.

### Other interpreter/runtime levers
- `PYTHONDONTWRITEBYTECODE` hurts (no `.pyc`); the opposite — **ship precompiled bytecode** (`python -m compileall -q --invalidation-mode unchecked-hash`) when installing from wheels/zipapps into read-only locations. PyInstaller already does this inside the PYZ archive.
- Frozen stdlib modules (3.11+) already remove the stdlib-import stat cost in standard builds.
- Experimental JIT / free-threading builds: measure, do not assume wins for CLIs.
- Explicit lazy imports (PEP 810, aimed at a future CPython) will make section 4.1 a language feature; until your minimum Python has it, keep the manual patterns.

---

## 6. Packaging choices

| Format | Cold start | Warm start | Install size | Needs Python on target | Notes |
|--------|-----------|------------|--------------|------------------------|-------|
| PyInstaller **onefile** | Slowest | **Slow: re-extracts every launch** (~0.6 s Linux, worse on macOS) | Small download | No | Fine for rarely-run tools. Wrong for a command people run all day. |
| PyInstaller **onedir** (+ symlinks) | Good | **Fast** (0.1 s here) | Larger on disk (uncompressed) | No | Chosen here. Ship as `app/` + symlinks in `bin/`. Resolves symlinks to find `_internal/`. |
| Nuitka (`--standalone`, optional `--onefile`) | Good | Good; compiled modules import faster | Medium | No | Real compilation (can use C-compiler PGO via `--pgo-c`). Longer builds, more per-platform tuning. Worth a pilot if import time remains dominant. |
| `uv tool install` / `pipx` | Venv creation at install | Fast after install; interpreter startup + imports | Wheel size + deps | Managed (uv fetches Python) | Great for developers. Installer needs network. Uses a PGO/LTO standalone Python (verify). |
| zipapp / `shiv` / `pex` | Unzip-on-first-run cache (shiv/pex) | Fast after warm cache | Small | Yes | Good for ops tooling where Python exists. |
| Wheels only | Pip install | Fast | Small | Yes | The baseline for libraries. |

PyInstaller specifics worth knowing:
- **Prefer onedir** for anything interactive; the cost of onefile is per launch, not per install.
- `--collect-all pkg` pulls every submodule/data/binary of `pkg`. It costs size, not startup, in onedir; trim it (`--collect-submodules`, `--collect-binaries`, `--collect-data`) when download size matters.
- `--exclude-module` for stdlib and dev packages you never import (`tkinter`, `unittest`, `pydoc`, `setuptools`, `pip`, `distutils`). Verify with a smoke test: exclusions fail at runtime, not build time.
- `--noupx`: UPX-compressed executables decompress on every launch and trip macOS signing; do not use UPX for CLIs.
- `--strip`/`--optimize 2`: tiny gains in testing; `-OO` semantics can break dependencies. Opt in only with a test suite that runs against the frozen build.
- One frozen app can expose several commands: install symlinks and dispatch on `os.path.basename(sys.argv[0])` (see `nomadic/entry.py`).
- Freeze with the **same Python you test and ship**; PyInstaller bundles the interpreter it runs under.
- Platform matrix: build on each target (`ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-14`); PyInstaller does not cross-compile.

### macOS distribution
- Apple Silicon requires at least an ad-hoc signature; PyInstaller applies one. For distribution you want a Developer ID signature **and notarisation**, applied to the whole app directory (`codesign --deep --options runtime`, then `notarytool submit` + `stapler staple`).
- Files downloaded by `curl` are not quarantined, so Gatekeeper's first-launch dialog does not apply to `install.sh` users, but XProtect/endpoint scanners still look at new executables. A **static onedir app is scanned once**; a onefile extraction directory is new on every launch (hypothesis, to be measured on macOS runners).
- Keep the installed path stable (`~/.local/share/nomadic/app`) so security tooling can cache verdicts.

---

## 7. Release engineering for fast artifacts

### 7.1 Bundle layout and install flow (current)
```
nomadic-<ver>-<platform>.tar.gz
└── nomadic-<ver>-<platform>/
    ├── app/                 # PyInstaller onedir: nomadic + _internal/
    ├── bin/nomadic    -> ../app/nomadic
    ├── bin/nomadic-db -> ../app/nomadic
    ├── sql/  docs/  dist/   # schemas, guide, wheel + sdist
    ├── install.sh  README.md  CHANGELOG.md  LICENSE.md
    └── BUILD_INFO.txt  release-metadata.json
```
`install.sh`: resolve release → fetch `SHA256SUMS` → pick the platform asset from it → download → verify checksum → extract → copy `app/` to `~/.local/share/nomadic/app.new` → atomically swap into `app/` → `ln -sfn` the two commands into the prefix → run `nomadic --version` (now instant). Legacy single-file bundles are still installable.

### 7.2 Channels: stable / rc / staging
- **Tags:** `v1.4.0` (stable), `v1.4.0-rc.1`, `v1.4.0-staging.3` (pre-release). The `Release` workflow normalises tag and `pyproject.toml` with PEP 440 (`v1.4.0-rc.1` ≡ `1.4.0rc1`) before comparing, and creates GitHub **pre-releases** for any tag containing `-`, so `releases/latest` (stable) never points at an rc.
- **Install:** `install.sh --channel rc` (or `staging`) installs the newest release whose tag matches `-rc*`; `--version v1.4.0-rc.1` pins one. The asset file name uses the PEP 440 version (`nomadic-1.4.0rc1-…`), so the installer reads it from `SHA256SUMS` instead of deriving it from the tag.
- The channel query uses the unauthenticated GitHub API (60 requests/hour/IP); set `GITHUB_TOKEN` in CI or behind a shared NAT.
- `NOMADIC_RELEASE_BASE_URL` redirects downloads to a mirror or a local test server; the end-to-end install test in this repo's history used `python -m http.server`.

### 7.3 Performance gates
- `scripts/check_startup.py <exe>`: warm-up, then median of N runs of `--version` and `--help` against budgets (defaults 0.75 s and 1.5 s; override via `NOMADIC_STARTUP_VERSION_BUDGET` / `NOMADIC_STARTUP_HELP_BUDGET`). It runs inside `build_bundle.sh` (a binary that is slow or cannot report its version never ships) and again in CI `binary-smoke` against the **installed** copy.
- Budgets are ~7× the measured value on purpose: shared CI runners vary 2–3×. Tighten them per platform once you have a week of data; never loosen them to make a failing build pass without understanding why.
- `tests/test_startup.py` guards the cause, not the symptom: forbidden imports on the fast paths.
- Publish the numbers: append `check_startup.py` output to `$GITHUB_STEP_SUMMARY` and keep the series (artifact or a branch) to see drift. A relative gate (fail on >25 % regression vs the last release) catches slow creep that an absolute budget misses.

### 7.4 Artifact hygiene
- Build once, promote: the `rc` artifact that was tested is the one re-tagged stable (or at least byte-identical inputs: locked deps, pinned Python, `SOURCE_DATE_EPOCH`).
- Lock dependency versions for the frozen build (`uv pip compile` / `pip-tools`); unpinned transitive updates are a common source of "release got slower".
- Compress for download with `gzip -9` or `zstd -19` (smaller, faster to unpack; needs `zstd` on the target, so keep gzip for the one-line installer). Trim `--collect-all` before reaching for exotic compressors.
- Record `BUILD_INFO` (Python version/flags, PyInstaller version, git SHA) next to the checksums.
- Sign checksums (minisign/cosign/GitHub artifact attestations) so `install.sh` can verify provenance, not just integrity.

---

## 8. Triage runbook: "the CLI feels slow after install"

1. `time nomadic --version` ×5. **> 0.3 s?** Go to 2. Otherwise the fast path is intact; measure the real command.
2. Is the executable a onefile build? `ls /tmp | grep _MEI` appears during a run, or `strace -f -c` shows tens of thousands of writes → **switch to onedir.**
3. `PYTHONPROFILEIMPORTTIME=1 nomadic --help 2>&1 | sort -t'|' -k2 -nr | head`. Any heavy package in the top 5 that a fast-path command doesn't need → defer it; add it to `tests/test_startup.py`.
4. Compare frozen vs source (`python -m <pkg> --version`). Frozen ≫ source by more than ~2× → check for `--collect-all` of huge packages, missing `--exclude-module`, or hooks importing at startup.
5. Slow only on the *first* run after install → cold cache or OS scanning (macOS: signing/notarisation, endpoint tools). Measure with `purge`/`drop_caches`; check `codesign -dv` and `spctl --assess`.
6. Slow only in CI → noisy runner; compare medians over ≥5 runs and use relative budgets.
7. Everything above is clean but the interpreter is still the bottleneck in long-running work → profile with `py-spy`, then consider a PGO/LTO interpreter (section 5) or Nuitka.

---

## 9. Checklist for a new Python CLI

- [ ] `--version` answered without importing the application.
- [ ] Package `__init__` is lazy; heavy deps imported inside functions; `TYPE_CHECKING` for types.
- [ ] Import-time budget test in the suite (forbidden modules on the fast paths).
- [ ] Frozen as **onedir** (or Nuitka standalone); one app, multiple command names via symlinks.
- [ ] Installer: checksum verified, atomic swap, symlinks, instant post-install smoke test.
- [ ] CI builds on every target OS/arch and enforces `check_startup.py` budgets on the installed artifact.
- [ ] Pinned interpreter and locked dependencies; build metadata recorded.
- [ ] Pre-release channels (`rc`, `staging`) flagged as GitHub pre-releases and installable by `--channel`.
- [ ] macOS: Developer ID signing + notarisation of the app directory (open item).
- [ ] Interpreter PGO/LTO verified (`sysconfig`), not assumed.

## 10. Open items for this repository
- Measure on `macos-14` runners (cold and warm) and add the numbers here; confirm or refute the per-launch scanning hypothesis for onefile.
- Developer ID signing and notarisation of `app/`.
- Pilot Nuitka standalone with `--pgo-c` for comparison against PyInstaller onedir.
- Publish `check_startup.py` results to the job summary and track a relative regression gate.
- Defer the remaining eager imports in `nomadic.cli` for non-REPL commands (`rich.console`, `nomadic.agent` → `search.metadata` → `urllib.request`) to get `--help` below 0.2 s frozen.
- Split `nomadic/cli.py` so subcommands load only their own dependencies.
