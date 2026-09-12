"""Parse `node --help` and `node --v8-options` into option arity tables.

The process probe has to know which interpreter options consume the token
after them, because the token it consumes may otherwise look like the script
path. Getting that wrong in either direction misreads a live `claude` main:
skipping an option that took a value leaves the value looking like the script,
and consuming one that took none swallows the script itself.

Hand-maintaining the list failed. `--inspect-port` and `--diagnostic-dir` were
missing, and `--experimental-test-isolation` was filed as taking no value when
it takes one. So the tables are derived from the interpreter's own help output
instead, by these two functions, and a test diffs the shipped tables against
the `node` on the machine running it.

Both functions are pure: they take help text and return
``(value_taking, boolean)``. Running `node` is the caller's business, which is
what lets the generator and the drift test share one parser.
"""

from __future__ import annotations

import re

# One `node --help` entry: the option spec, then an optional argument
# placeholder. The help wraps its descriptions at a narrow width, so a long
# option name can be followed by its description after a SINGLE space; the
# spec is therefore read as a run of option-shaped tokens rather than as
# "everything before the description".
_HELP_ENTRY = re.compile(
    r"^  (-[^\s,]+(?:\s*,\s*-[^\s,]+)*)(\s+\[\.\.\.\]|\s+<[^>]*>)?(?:\s|$)"
)
_OPTION_NAME = re.compile(r"-{1,2}[A-Za-z0-9][A-Za-z0-9.-]*")

# `node --v8-options` prints the name and description on one line and the type
# on the next. Anything not `type: bool` takes a value.
_V8_ENTRY = re.compile(r"^  (--[A-Za-z0-9][A-Za-z0-9.-]*)\s")
_V8_TYPE = re.compile(r"^\s+type:\s+(\S+)")


def parse_node_help(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """`node --help` text into ``(value_taking, boolean)`` option names.

    Three spellings have to be told apart. ``--opt=...`` takes a value, and
    node accepts it detached as ``--opt value`` too, so it consumes a token.
    ``--opt[=x]`` takes one only when attached, so it never consumes a token
    and counts as boolean. Aliases share a line and share an arity, so
    ``--loader, --experimental-loader=...`` makes BOTH value-taking — reading
    the arity off each token alone filed `--loader` as boolean, which is how
    an option that swallows the next token came to look like one that does
    not.

    `-` and `--` are dropped: they are positions rather than options, and the
    classifier handles them itself.
    """
    value: set[str] = set()
    boolean: set[str] = set()
    for line in text.splitlines():
        entry = _HELP_ENTRY.match(line)
        if entry is None:
            continue
        spec, placeholder = entry.group(1), entry.group(2)
        names: list[str] = []
        takes_value = bool(placeholder)
        for token in re.split(r"\s*,\s*", spec):
            if token in ("-", "--"):
                continue
            if "[=" in token:
                name = token.split("[=", 1)[0]
            elif "=" in token:
                name = token.split("=", 1)[0]
                takes_value = True
            else:
                name = token
            if _OPTION_NAME.fullmatch(name):
                names.append(name)
        (value if takes_value else boolean).update(names)
    return frozenset(value), frozenset(boolean - value)


def parse_v8_options(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """`node --v8-options` text into ``(value_taking, boolean)`` option names.

    V8's options reach node too, and they are where `--max-old-space-size`
    lives, so leaving them out left a common real command line unresolvable.
    V8 states each option's type on the line below its name; every type other
    than ``bool`` takes a value, and V8 accepts both `--opt=v` and `--opt v`.
    """
    value: set[str] = set()
    boolean: set[str] = set()
    pending: str | None = None
    for line in text.splitlines():
        entry = _V8_ENTRY.match(line)
        if entry is not None:
            pending = entry.group(1)
            continue
        kind = _V8_TYPE.match(line)
        if kind is not None and pending is not None:
            (boolean if kind.group(1) == "bool" else value).add(pending)
            pending = None
    return frozenset(value), frozenset(boolean - value)
