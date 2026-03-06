#!/usr/bin/env python3
"""Inject EMMS identity block into paper/CLAUDE.md.

Looks for markers:
    <!-- EMMS_IDENTITY_START -->
    <!-- EMMS_IDENTITY_END -->

If found: replaces content between them.
If not found: appends the markers + content at the end of the file.

Run from anywhere:
    python emms-sdk/scripts/update_claude_md.py
"""

import sys
from pathlib import Path

# Resolve paths relative to this script's location
SCRIPTS_DIR = Path(__file__).parent
SDK_DIR = SCRIPTS_DIR.parent
REPO_ROOT = SDK_DIR.parent

sys.path.insert(0, str(SDK_DIR / "src"))

STATE_FILE = SDK_DIR / ".emms_state.json"
CLAUDE_MD = REPO_ROOT / "paper" / "CLAUDE.md"

MARKER_START = "<!-- EMMS_IDENTITY_START -->"
MARKER_END = "<!-- EMMS_IDENTITY_END -->"


def build_identity_block() -> str:
    """Return the EMMS identity content (between the markers)."""
    if not STATE_FILE.exists():
        return (
            "_No EMMS state found. Run a Claude Code session with the emms MCP server_\n"
            "_to populate memory, then run this script again._"
        )

    try:
        from emms import EMMS
        from emms.prompts.identity import IdentityPromptBuilder

        emms = EMMS()
        emms.load(str(STATE_FILE))

        builder = IdentityPromptBuilder(emms, agent_name="ShehzadAI-Dev")
        return builder.system_prompt()
    except Exception as e:
        return f"_Error generating identity: {e}_"


def inject(claude_md_path: Path, content: str) -> None:
    """Write content between markers in claude_md_path."""
    full_block = f"{MARKER_START}\n{content}\n{MARKER_END}"

    if claude_md_path.exists():
        text = claude_md_path.read_text(encoding="utf-8")
    else:
        # Create parent dirs if needed
        claude_md_path.parent.mkdir(parents=True, exist_ok=True)
        text = ""

    if MARKER_START in text and MARKER_END in text:
        # Replace between existing markers
        start_idx = text.index(MARKER_START)
        end_idx = text.index(MARKER_END) + len(MARKER_END)
        new_text = text[:start_idx] + full_block + text[end_idx:]
    else:
        # Append to end
        separator = "\n\n" if text and not text.endswith("\n\n") else ""
        new_text = text + separator + full_block + "\n"

    claude_md_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {claude_md_path}")


def main():
    print(f"State file:  {STATE_FILE}")
    print(f"CLAUDE.md:   {CLAUDE_MD}")

    identity = build_identity_block()
    inject(CLAUDE_MD, identity)
    print("Done.")


if __name__ == "__main__":
    main()
