"""Regenerate src/claude_swap/_node_options.py from the local `node`.

Run it after a node upgrade, when tests/test_process_detection.py reports that
the shipped tables no longer match this machine's interpreter:

    uv run python tools/regen_node_options.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claude_swap.node_help import parse_node_help, parse_v8_options  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "src" / "claude_swap" / "_node_options.py"


def _run(*argv: str) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.stdout + proc.stderr


def main() -> int:
    version = _run("node", "--version").strip()
    help_value, help_bool = parse_node_help(_run("node", "--help"))
    v8_value, v8_bool = parse_v8_options(_run("node", "--v8-options"))
    value = help_value | v8_value
    boolean = (help_bool | v8_bool) - value

    def block(name: str, names) -> str:
        body = "".join(f'    "{n}",\n' for n in sorted(names))
        return f"{name} = frozenset({{\n{body}}})\n"

    OUT.write_text(
        '"""Node option arity tables, generated from this machine\'s node.\n\n'
        "Do not edit by hand. Regenerate with `uv run python\n"
        "tools/regen_node_options.py`; tests/test_process_detection.py fails when\n"
        "these tables and the local `node --help` disagree.\n"
        '"""\n\nfrom __future__ import annotations\n\n'
        f'GENERATED_FROM = "{version}"\n\n'
        + block("NODE_VALUE_FLAGS", value)
        + "\n"
        + block("NODE_BOOLEAN_FLAGS", boolean)
    )
    print(f"{OUT}: {len(value)} value-taking, {len(boolean)} boolean, from {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
