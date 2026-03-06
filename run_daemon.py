#!/usr/bin/env python
"""Runner script for the EMMS ConsciousnessDaemon.

Run directly::

    python run_daemon.py

Or as a macOS LaunchAgent (auto-start on login):
    See ~/Library/LaunchAgents/com.emms.consciousness.plist
"""

import sys
import os

# Ensure the src package is importable when running as a standalone script
_sdk_root = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_sdk_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from emms.daemon.consciousness_daemon import main

if __name__ == "__main__":
    main()
