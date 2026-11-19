"""Allow running the bridge via ``python -m antigravity_acp_bridge``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
