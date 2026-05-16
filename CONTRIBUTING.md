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

For maintainers operating release pipelines, configure repository and account-level settings as follows:

- GitHub Pages publishing source: **GitHub Actions**
- Canonical release workflow: `.github/workflows/release-cd.yml` (tag-driven)
- Compatibility workflows (`release-github.yml`, `release-pypi.yml`) are manual wrappers that delegate to the canonical workflow.
- Required environments:
  - `pypi` (for trusted publishing)
  - `github-pages` (for docs deployment)
- Required workflow permissions by workflow/job:
  - `contents: write` for GitHub Release asset upload
  - `id-token: write` for PyPI trusted publishing
  - `pages: write` and `id-token: write` for docs deploy workflow
- PyPI Trusted Publisher binding should match this repository and release workflow details:
  - owner/repository
  - workflow file path
  - environment name `pypi`
  - tag-based trigger (`vX.Y.Z.B`)

GitHub Release publishing in Actions uses `GITHUB_TOKEN`; no Personal Access Token (PAT) is required for release uploads in the workflow context.

## AI-Assisted Contributions

AI tools may be used to help draft code, docs, tests, or analysis, but they do not replace contributor responsibility.

- A human contributor must understand the full change before submission.
- A human contributor must run or meaningfully review verification steps.
- A human contributor must stand behind the final PR and address review comments.
- Purely autonomous or end-to-end agent-submitted PRs are not acceptable for merge.
- Disclose meaningful AI assistance in the PR template.

See [AI_POLICY.md](./AI_POLICY.md) for the full repository policy.

## Community Standards

By participating in this project, you agree to follow the expectations in [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## Security

Do not file public issues for security vulnerabilities. Follow [SECURITY.md](./SECURITY.md) instead.
