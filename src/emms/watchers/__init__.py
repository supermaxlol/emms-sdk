"""EMMS Watchers — read-only observers that ingest external data streams.

Watchers are passive: they tail log files, read state JSONs, or poll APIs.
They NEVER modify the source system. One-way data flow: source → EMMS.
"""
