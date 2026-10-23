# TODO

Digested from README (roadmap, "What Nomadic is not"), CHANGELOG, and the SRS references in `tests/test_srs_ext.py` (SRS-ext-v1.1: FC-1 Agent Instance, FC-2 Orchestrator, FC-3 HITL, FC-4 Task Management).

## Roadmap (open items from README)

- [ ] Tests for orchestrator, HITL, and task registry on par with the kernel loop
- [ ] Hosted/OpenAI-compatible remote provider as a first-class backend
- [ ] Pluggable tool registries and remote execution adapters
- [ ] Retrieval and memory integrations (vector/index, source-aware answers)
- [ ] File-centric workflows beyond `SKILL.md`
- [ ] Signed/notarized macOS binaries; Intel macOS and Windows feasibility
- [ ] Public docs, issue templates, API stability policy

## Known limitations (README "What Nomadic is not")

- [ ] MCP client / extension API beyond tool packs and prompt presets
- [ ] `nomadic/cli.py` is a single ~3.5k-line module; split it up

## Gaps vs. `specs/`

`specs/` is listed in `.gitignore` and is not in the cloned repo, so the specs could not be read. Gaps have not been derived yet.

- [ ] Provide `specs/` (SRS-ext-v1.1 and any others), then diff each requirement (FC-1..FC-4 and beyond) against `nomadic/agent.py`, `orchestrator.py`, `hitl.py`, `tasks.py`, and list every unimplemented or partial requirement here.
