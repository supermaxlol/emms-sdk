#!/usr/bin/env python
"""Runner script for the EMMS ConsciousnessDaemon.

Run directly::

    python run_daemon.py          # v1: cron-based MemoryScheduler (26 jobs)
    python run_daemon.py --v2     # v2: event-driven CognitiveLoop + APScheduler

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

if __name__ == "__main__":
    if "--v2" in sys.argv:
        from emms.daemon.consciousness_daemon_v2 import main
    else:
        from emms.daemon.consciousness_daemon import main
    main()
