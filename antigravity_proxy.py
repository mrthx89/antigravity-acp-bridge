#!/usr/bin/env python3
"""Backwards-compatible launcher for the Antigravity OpenAI-compatible proxy."""

import sys

from antigravity_proxy.cli import main

if __name__ == "__main__":
    sys.exit(main())
