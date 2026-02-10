# EMMS — Enhanced Memory Management System

Persistent hierarchical memory for AI agents. Give your agent human-like memory
with working, short-term, long-term, and semantic tiers, cross-modal binding,
episode detection, and persistent identity.

```python
from emms import EMMS, Experience

agent = EMMS()
agent.store(Experience(content="The market rose 2% today", domain="finance"))
results = agent.retrieve("market trends")
```

## Install

```bash
pip install -e ".[all,dev]"
```

## Test

```bash
pytest
```
