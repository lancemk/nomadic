"""Public package surface for Nomadic.

Names are resolved lazily (PEP 562) so ``import nomadic`` stays cheap: the CLI
pulls in rich, prompt_toolkit and friends, which commands like ``--version``
never need.
"""

from __future__ import annotations

from typing import Any

_CLI_EXPORTS = (
    "KernelInputSubmission",
    "KernelInterruptController",
    "parse_kernel_input",
    "parse_next_action_options",
    "render_next_action_menu",
    "resolve_next_action_selection",
)

__all__ = list(_CLI_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _CLI_EXPORTS:
        from . import cli

        return getattr(cli, name)
    raise AttributeError(f"module 'nomadic' has no attribute {name!r}")
