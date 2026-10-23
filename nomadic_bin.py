"""PyInstaller entry point (see scripts/build_bundle.sh)."""
import sys

from nomadic.entry import main

if __name__ == "__main__":
    sys.exit(main())
