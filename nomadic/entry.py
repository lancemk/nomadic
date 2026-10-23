"""Console entry point shared by ``nomadic`` and the frozen binaries.

``nomadic --version`` is answered before the CLI (rich, prompt_toolkit, ...) is
imported, so install-time smoke tests and shell prompts stay instant.  When the
executable is invoked as ``nomadic-db`` (a symlink to the same frozen app) it
dispatches to the database helper instead.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    prog = os.path.basename(sys.argv[0])
    if prog.startswith("nomadic-db"):
        from .storage.db import main as db_main

        db_main()
        return 0
    if args == ["--version"]:
        from .version import get_version

        print(f"nomadic {get_version()}")
        return 0
    from .cli import main as cli_main

    return cli_main() or 0


if __name__ == "__main__":
    sys.exit(main())
