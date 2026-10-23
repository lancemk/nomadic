#!/usr/bin/env python3
"""Compatibility wrapper for `nomadic.storage.db`."""

from nomadic.storage.db import *  # noqa: F403
from nomadic.storage.db import main


if __name__ == "__main__":
    main()
