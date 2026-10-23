"""Context resolver (FC-12): turn ``@path`` references into injected file context."""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass, field

# ``@path`` or ``@"path with spaces"`` at the start of a word (so emails never match).
FILE_REF_RE = re.compile(r'(?<!\S)@(?:"([^"]+)"|(\S+))')
_TRAILING_PUNCT = ".,;:!?)]}'\""
CHARS_PER_TOKEN = 4
TRUNCATION_MARKER = "\n[... truncated ...]"


@dataclass(frozen=True)
class ResolvedFile:
    ref: str
    path: pathlib.Path
    content: str
    truncated: bool = False

    def render(self) -> str:
        return f'<file path="{self.ref}"><content>\n{self.content}\n</content></file>'


@dataclass
class ContextResolution:
    files: list[ResolvedFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n".join(f.render() for f in self.files)

    def append_to(self, text: str) -> str:
        """Return *text* with the attached file blocks appended to the prompt."""
        if not self.files:
            return text
        return f"{text}\n\n[Attached Context:]\n{self.render()}"


def extract_refs(text: str) -> list[str]:
    """Return unique ``@`` references in order of first appearance."""
    refs: list[str] = []
    for quoted, bare in FILE_REF_RE.findall(text):
        ref = quoted or bare
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _candidates(ref: str) -> list[str]:
    stripped = ref.rstrip(_TRAILING_PUNCT)
    return [ref] if stripped == ref or not stripped else [ref, stripped]


def _is_binary(path: pathlib.Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" in handle.read(4096)


def resolve_file_references(
    text: str,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    max_file_chars: int = 12_000,
    max_context_tokens: int | None = None,
) -> ContextResolution:
    """Resolve ``@path`` references in *text* relative to *base_dir* (default: PWD).

    Each file is truncated to ``max_file_chars``; all attachments together are
    limited to half of ``max_context_tokens`` (approximated at 4 chars/token) so
    references can never crowd out the conversation history.
    """
    base = pathlib.Path(base_dir) if base_dir is not None else pathlib.Path.cwd()
    result = ContextResolution()
    budget = None if max_context_tokens is None else max(0, max_context_tokens) * CHARS_PER_TOKEN // 2
    seen: set[pathlib.Path] = set()

    for ref in extract_refs(text):
        path: pathlib.Path | None = None
        for candidate in _candidates(ref):
            p = pathlib.Path(os.path.expanduser(candidate))
            p = p if p.is_absolute() else base / p
            if p.exists():
                path, ref = p, candidate
                break
        if path is None:
            continue  # not a file reference (e.g. a handle); leave the text alone
        if path.is_dir():
            result.warnings.append(f"@{ref} is a directory; reference a file or use list_dir")
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            if _is_binary(path):
                result.warnings.append(f"@{ref} looks binary; skipped")
                continue
            limit = max_file_chars if budget is None else min(max_file_chars, budget)
            if limit <= 0:
                result.warnings.append(f"@{ref} skipped: context budget exhausted")
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                content = handle.read(limit + 1)
        except OSError as exc:
            result.warnings.append(f"Failed to attach @{ref}: {exc}")
            continue
        truncated = len(content) > limit
        if truncated:
            content = content[:limit] + TRUNCATION_MARKER
            result.warnings.append(f"@{ref} truncated to {limit} chars")
        if budget is not None:
            budget -= min(len(content), limit)
        result.files.append(ResolvedFile(ref=ref, path=path, content=content, truncated=truncated))
    return result
