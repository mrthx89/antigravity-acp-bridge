"""Allow running the proxy via ``python -m antigravity_proxy``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
