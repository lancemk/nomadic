#!/usr/bin/env python3
"""Compatibility wrapper for `nomadic.db`."""

from nomadic.db import *  # noqa: F403
from nomadic.db import main


if __name__ == "__main__":
    main()
