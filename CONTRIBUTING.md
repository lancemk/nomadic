# Contributing to nomadic

Thanks for your interest in improving `nomadic`. This project welcomes focused, well-explained contributions that fit the repository's stated scope and quality bar.

## Before You Start

- Read [README.md](./README.md) for the current project goals and v1 boundaries.
- Open or join a GitHub Issue before starting non-trivial features, behavior changes, architecture changes, or large refactors.

Small typo fixes, docs clarifications, and tightly scoped bug fixes can usually go straight to a pull request.

## Contribution Flow

1. Confirm the work fits the current repository direction.
2. Fork the repository and create a focused branch.
3. Make the smallest complete change that solves one problem well.
4. Add or update tests for behavior changes.
5. Add or update docs for user-facing changes.
6. Run the relevant local checks before opening a PR.
7. Open a pull request with clear motivation, verification details, and any relevant issue links.
8. Respond to review feedback and keep the branch up to date until merge or close.

## Pull Request Expectations

- Keep each PR focused on one concern.
- Link the related issue or provide equivalent context.
- Explain why the change is needed, not only what changed.
- Include reproducible verification steps.
- Update tests when behavior changes.
- Update docs when the public surface, install flow, config shape, or usage changes.
- Avoid drive-by refactors unrelated to the stated goal.
- Be responsive to maintainer review and ready to reduce scope if requested.

## Contribute Practices to Abide

- Open an issue first for large features, behavior changes, or architecture changes.
- Keep one PR per concern so review stays fast and reliable.
- Explain the problem, constraints, and intended outcome in plain language.
- Include exact commands, inputs, or scenarios used to verify the change.
- Stay engaged during review and answer follow-up questions directly.
- Accept that maintainers may decline or narrow changes that do not fit current project priorities.

## Branch and Commit Hygiene

- Use short-lived branches with descriptive names.
- Prefer small, reviewable commits over large mixed changesets.
- Write commit messages that describe the user or maintainer impact clearly.
- Rebase or otherwise tidy your branch before merge if maintainers ask for it.


## Release/CD Setup

For maintainers operating the release pipeline:

- `ci.yml` validates every PR/push (pytest, shellcheck, binary + `install.sh` smoke test).
- `post_merge.yml` bumps the version and changelog on merges to `main`/`develop/**`; on `main` it also creates the `vX.Y.Z` tag and dispatches `release.yml`.
- `release.yml` builds `macos-arm64`, `linux-x86_64`, `linux-arm64` bundles and publishes them, with `SHA256SUMS` and `install.sh`, to a GitHub Release.
- Repository settings: Actions workflow permissions must be "Read and write", and branch protection on `main` must let `github-actions[bot]` push the version commit and tag.
