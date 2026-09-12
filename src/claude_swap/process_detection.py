"""Detect running Claude Code instances.

Reads session PID files (~/.claude/sessions/{pid}.json) and IDE lockfiles
(~/.claude/ide/{port}.lock) to determine which Claude Code instances are
currently running. Uses the same mechanism Claude Code itself uses internally.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from claude_swap.paths import get_claude_config_home

logger = logging.getLogger(__name__)


@dataclass
class ClaudeSession:
    """A running Claude Code session from ~/.claude/sessions/{pid}.json."""

    pid: int
    session_id: str
    cwd: str
    started_at: int  # epoch milliseconds
    kind: str  # "interactive", "bg", "daemon", "daemon-worker"
    entrypoint: str  # "cli", "claude-vscode", "claude-desktop", "sdk-cli", "mcp"
    status: str | None = None  # "busy", "idle", "waiting"


@dataclass
class IdeInstance:
    """A running IDE instance from ~/.claude/ide/{port}.lock."""

    port: int  # from filename
    pid: int
    ide_name: str  # "Visual Studio Code", "Cursor", "Windsurf"
    workspace_folders: list[str] = field(default_factory=list)


def get_claude_dir() -> Path:
    """Return the Claude config directory, respecting CLAUDE_CONFIG_DIR."""
    return get_claude_config_home()


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is running.

    Cross-platform:
    - macOS/Linux/WSL: os.kill(pid, 0)
    - Windows: ctypes OpenProcess
    """
    if pid <= 1:
        return False

    if sys.platform == "win32":
        return _is_pid_alive_windows(pid)

    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # EPERM means the process exists but we lack permission
        return True
    except OSError:
        return False


def _is_pid_alive_windows(pid: int) -> bool:
    """Windows-specific PID liveness check using ctypes."""
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def scan_sessions(claude_dir: Path | None = None) -> tuple[list[ClaudeSession], int]:
    """Live sessions, and how many records could NOT be read.

    Two kinds of caller read this directory and they need opposite things from
    an unparseable record:

    - A SCAN (a listing, a status display) wants it skipped. One bad file must
      not take out the whole listing.
    - A GUARD wants to know. ``0 live`` and ``0 readable`` are the same list,
      and only the first is safe to act on -- the callers gate ``_bootstrap``
      (which deletes a profile's Keychain entry and overwrites
      ``.credentials.json``) and account removal, so reading "could not tell"
      as "nobody there" runs them underneath a live instance.

    So the count is returned rather than swallowed, and ``list_sessions``
    below is the scan-shaped view that drops it.
    """
    sessions_dir = (claude_dir or get_claude_dir()) / "sessions"
    if not sessions_dir.is_dir():
        return [], 0

    sessions = []
    unreadable = 0
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = data["pid"]
            if not is_pid_alive(pid):
                continue
            sessions.append(ClaudeSession(
                pid=pid,
                session_id=data.get("sessionId", ""),
                cwd=data.get("cwd", ""),
                started_at=data.get("startedAt", 0),
                kind=data.get("kind", ""),
                entrypoint=data.get("entrypoint", ""),
                status=data.get("status"),
            ))
        except (
            json.JSONDecodeError,   # malformed JSON
            KeyError,               # required field missing
            TypeError,              # field has the wrong type (e.g. pid not an int)
            AttributeError,         # valid JSON that is not an object: a
                                    # top-level array reaches `.get` as a list.
                                    # Also how a too-deep nesting lands where
                                    # the parser's recursion limit is high
                                    # enough not to raise -- it differs per
                                    # machine, so BOTH outcomes must be inert.
            ValueError,             # includes UnicodeDecodeError from read_text
            OverflowError,          # pid too large for os.kill's C long (is_pid_alive)
            RecursionError,         # pathologically nested JSON in json.loads
            OSError,
        ) as exc:
            unreadable += 1
            logger.debug("Skipping session file %s: %s", path, exc)
    return sessions, unreadable


def list_sessions(claude_dir: Path | None = None) -> list[ClaudeSession]:
    """Live sessions. A record that cannot be read is SKIPPED.

    SCAN USE ONLY. The returned list cannot distinguish "no live sessions"
    from "no readable records", so anything gating a destructive step must
    call :func:`scan_sessions` and treat a non-zero count as live.
    """
    return scan_sessions(claude_dir)[0]


# --- independent liveness: the OS process table ------------------------------
#
# Everything above reads Claude Code's OWN session registry, and a live
# instance can simply be absent from it. Observed on a machine running seven
# `claude` mains against one profile: only five had records, and both missing
# ones were `claude --resume`. That gap is invisible to `scan_sessions`,
# because a record that was never written is not an unreadable record -- it is
# nothing at all, and "0 live, 0 unreadable" is precisely the reading that says
# "safe to overwrite". The cost is a mid-session logout: `_bootstrap` deletes
# the profile's Keychain entry and rewrites .credentials.json under a running
# instance, resetting it to a generation whose refresh token was already spent.
#
# So the guard needs one signal we own rather than inherit. CLAUDE_CONFIG_DIR
# is it: a running `claude` whose environment names this profile IS using this
# profile's credentials, whether or not it ever announced itself.

_ENV_PROBE_ARGV = ("ps", "ewwx", "-o", "pid=,command=")
# `comm` is the executable path with no argv after it, so it is the LAST
# column and survives `split(None, 1)` intact. `command=` cannot: argv follows
# the path, so a binary under `.../Application Support/...` splits mid-path and
# its basename becomes "Application". That reads as "not claude", which reads
# as "profile idle", which is the exact false negative this module exists to
# prevent -- a credential rewrite under a live session.
_COMM_PROBE_ARGV = ("ps", "-Awwo", "pid=,comm=")
# argv WITHOUT the environment (no `e` flag). Subtracting this from the `ewwx`
# line yields the environment exactly, with no guess about where argv ended.
# `ww` is load-bearing, not cosmetic: without it `ps` truncates to the
# detected terminal width, so plain_argv arrives as a PREFIX while the `ewwx`
# probe is unbounded. _split_argv_env would then hand the truncated tail of
# argv to the environment matcher — the false binding that subtraction exists
# to eliminate.
_ARGV_PROBE_ARGV = ("ps", "-Awwo", "pid=,command=")

# Just the pids. Used to enumerate candidates where there is no /proc to list.
_PID_PROBE_ARGV = ("ps", "-Ao", "pid=")

# `ps ewwx` prints `PID argv... VAR=VAL VAR=VAL`, so argv ends at the first
# env-shaped token. argv can hold its own `=` (`--model=x`, or a prompt saying
# `FOO=bar`), which would cut it early -- harmless, because that only shortens
# argv, and the binary tested below is its first token either way.
_ENV_PAIR = re.compile(r"\s+[A-Za-z_][A-Za-z0-9_]*=")

# The same assignment shape, but capturing the NAME and allowed to match at the
# very start of the string, so the environment can be parsed into pairs rather
# than substring-searched. Substring matching got both ends of a variable
# wrong: with no boundary before the name, `OTHER_CLAUDE_CONFIG_DIR=/p` satisfied
# a search for `CLAUDE_CONFIG_DIR=/p`, and with the value ended at whitespace,
# a profile at `/p` matched a process whose profile was `/p old`.
_ENV_ASSIGN = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=")

_WHITESPACE = re.compile(r"\s")


def _bound_text(target: str) -> re.Pattern[str]:
    """`CLAUDE_CONFIG_DIR=<target>` as a complete assignment in flattened output.

    Bounded at both ends: a sibling profile `<target>-old` does not satisfy the
    right side, and `OTHER_CLAUDE_CONFIG_DIR=<target>` does not satisfy the
    left.
    """
    return re.compile(
        r"(?:^|\s)CLAUDE_CONFIG_DIR=" + re.escape(target) + r"(?=\s|$)"
    )

# Runtimes a non-native install can front the CLI with. Matched with a trailing
# version, because `node22` and `python3.13` are the same programs.
_INTERPRETER = re.compile(r"^(node|bun|deno|python)[\d.]*$")

# The script argument names claude: either a bare `claude` or any path ending
# `/claude`, in both cases at a token boundary so `claude-helper` misses.
_SCRIPT_TAIL = re.compile(r"(?:^|[\s/])claude(?=\s|$)")

# Endings that mark a token as a complete script name in its own right.
_SCRIPT_SUFFIXES = (".js", ".mjs", ".cjs", ".ts", ".py")

# The npm distribution runs as `node .../@anthropic-ai/claude-code/cli.js`, so
# the script's BASENAME is cli.js and the suffix rule above would file a real
# main as a generic worker — missing it entirely and letting its credentials be
# rewritten underneath it. The package directory is what identifies it. Matched
# as a whole path segment so an unrelated `/home/claude-code-notes/x.js` misses.
_CLAUDE_PACKAGE = re.compile(r"/claude-code/")

# Interpreter options, split by what they do to the token after them. The
# lists are transcribed from `node --help` (v26) and `python --help`, and they
# are deliberately CLOSED: an option in neither list is not assumed harmless,
# because assuming wrongly reads a real main's script argument as an option's
# value and reports a live profile as idle. `--inspect-port 0 /opt/bin/claude`
# did exactly that. Anything unlisted makes the argv unknown, which fails
# closed — a deferred re-seed rather than a logout.
#
# A flag written `--opt=value` is self-contained whatever it means, so it never
# reaches the lists and never makes anything unknown.
_NODE_VALUE_FLAGS = frozenset({
    "-C", "-r",
    "--allow-fs-read", "--allow-fs-write", "--build-sea",
    "--build-snapshot-config", "--conditions", "--cpu-prof-dir",
    "--cpu-prof-interval", "--cpu-prof-name", "--debug-port",
    "--diagnostic-dir", "--disable-proto", "--disable-warning",
    "--dns-result-order", "--env-file", "--env-file-if-exists",
    "--experimental-config-file", "--experimental-loader",
    "--experimental-sea-config", "--heap-prof-dir", "--heap-prof-interval",
    "--heap-prof-name", "--heapsnapshot-near-heap-limit",
    "--heapsnapshot-signal", "--icu-data-dir", "--import", "--input-type",
    "--inspect-port", "--inspect-publish-uid", "--loader",
    "--localstorage-file", "--max-http-header-size", "--max-old-space-size",
    "--max-old-space-size-percentage",
    "--network-family-autoselection-attempt-timeout", "--openssl-config",
    "--redirect-warnings", "--report-dir", "--report-directory",
    "--report-filename", "--report-signal", "--require", "--run",
    "--secure-heap", "--secure-heap-min", "--snapshot-blob",
    "--test-concurrency", "--test-coverage-branches",
    "--test-coverage-exclude", "--test-coverage-functions",
    "--test-coverage-include", "--test-coverage-lines",
    "--test-global-setup", "--test-isolation", "--test-name-pattern",
    "--test-reporter", "--test-reporter-destination",
    "--test-rerun-failures", "--test-shard", "--test-skip-pattern",
    "--test-timeout", "--title", "--tls-cipher-list", "--tls-keylog",
    "--trace-event-categories", "--trace-event-file-pattern",
    "--trace-require-module", "--unhandled-rejections", "--use-largepages",
    "--v8-pool-size", "--watch-kill-signal", "--watch-path",
})

_NODE_BOOL_FLAGS = frozenset({
    "-c", "-h", "-i", "-v",
    "--abort-on-uncaught-exception", "--allow-addons", "--allow-child-process",
    "--allow-inspector", "--allow-net", "--allow-wasi", "--allow-worker",
    "--build-snapshot", "--check", "--completion-bash", "--cpu-prof",
    "--disable-sigusr1", "--disable-wasm-trap-handler",
    "--disallow-code-generation-from-strings", "--enable-etw-stack-walking",
    "--enable-fips", "--enable-network-family-autoselection",
    "--enable-source-maps", "--entry-url", "--expose-gc",
    "--force-context-aware", "--force-fips",
    "--force-node-api-uncaught-exceptions-policy", "--frozen-intrinsics",
    "--heap-prof", "--help", "--insecure-http-parser", "--inspect",
    "--inspect-brk", "--inspect-wait", "--interactive",
    "--interpreted-frames-native-stack", "--jitless", "--node-memory-debug",
    "--openssl-legacy-provider", "--openssl-shared-config",
    "--pending-deprecation", "--permission", "--permission-audit",
    "--preserve-symlinks", "--preserve-symlinks-main", "--prof",
    "--prof-process", "--report-compact", "--report-exclude-env",
    "--report-exclude-network", "--report-on-fatalerror", "--report-on-signal",
    "--report-uncaught-exception", "--require-module", "--test",
    "--test-force-exit", "--test-only", "--test-update-snapshots",
    "--throw-deprecation", "--tls-max-v1", "--tls-min-v1",
    "--trace-deprecation", "--trace-env", "--trace-env-js-stack",
    "--trace-env-native-stack", "--trace-exit", "--trace-promises",
    "--trace-sigint", "--trace-sync-io", "--trace-tls", "--trace-uncaught",
    "--trace-warnings", "--track-heap-objects", "--use-bundled-ca",
    "--use-env-proxy", "--use-openssl-ca", "--use-system-ca", "--v8-options",
    "--version", "--watch", "--watch-preserve-output", "--webstorage",
    "--zero-fill-buffers",
} | {f for f in (
    # Every `--no-` negation node accepts is a boolean by construction.
    "--no-addons", "--no-async-context-frame", "--no-deprecation",
    "--no-experimental-detect-module", "--no-experimental-global-navigator",
    "--no-experimental-repl-await", "--no-experimental-require-module",
    "--no-experimental-sqlite", "--no-experimental-websocket",
    "--no-experimental-webstorage", "--no-extra-info-on-fatal-exception",
    "--no-force-async-hooks-checks", "--no-global-search-paths",
    "--no-network-family-autoselection", "--no-require-module",
    "--no-strip-types", "--no-warnings",
)} | {f for f in (
    # Experimental gates, all boolean.
    "--experimental-addon-modules", "--experimental-default-config-file",
    "--experimental-eventsource", "--experimental-import-meta-resolve",
    "--experimental-inspector-network-resource",
    "--experimental-network-inspection", "--experimental-print-required-tla",
    "--experimental-storage-inspection", "--experimental-stream-iter",
    "--experimental-strip-types", "--experimental-test-coverage",
    "--experimental-test-isolation", "--experimental-test-module-mocks",
    "--experimental-vm-modules", "--experimental-worker-inspection",
)})

# Options whose value IS the program. There is no script argument after one of
# these, so every remaining token is argv for the inline code: `node --eval
# 'setInterval(...)' /opt/bin/claude` holds a path, it does not run claude.
_NODE_EVAL_FLAGS = frozenset({"--eval", "--print"})

# Short options combine into one cluster, so the set of literal spellings is
# open-ended: `-pe`, `-ep`, `-ipe` all mean eval. Matching the letters instead
# of the spellings closes it.
_NODE_EVAL_CLUSTER = re.compile(r"^-[a-zA-Z]*[ep][a-zA-Z]*$")

_PYTHON_VALUE_FLAGS = frozenset({"-W", "-X", "--check-hash-based-pycs"})
_PYTHON_BOOL_FLAGS = frozenset({
    "-b", "-bb", "-B", "-d", "-E", "-i", "-I", "-O", "-OO", "-P", "-q", "-s",
    "-S", "-u", "-v", "-vv", "-x", "-h", "--help", "-V", "--version",
})
# `-c CODE` and `-m MODULE` both replace the script with something that is not
# a path, so neither can name claude.
_PYTHON_EVAL_FLAGS = frozenset({"-c", "-m"})


class _FlagTable(NamedTuple):
    value: frozenset[str]
    boolean: frozenset[str]
    inline: frozenset[str]
    cluster: re.Pattern[str] | None


_FLAG_TABLES = {
    "node": _FlagTable(_NODE_VALUE_FLAGS, _NODE_BOOL_FLAGS,
                       _NODE_EVAL_FLAGS, _NODE_EVAL_CLUSTER),
    "python": _FlagTable(_PYTHON_VALUE_FLAGS, _PYTHON_BOOL_FLAGS,
                         _PYTHON_EVAL_FLAGS, None),
    # bun and deno front the CLI the same way but take their own options, and
    # transcribing two more closed lists buys nothing: a bare `bun /path/claude`
    # has no options to place, and one that does is unknown, which fails closed.
    "bun": _FlagTable(frozenset(), frozenset(), frozenset(), None),
    "deno": _FlagTable(frozenset(), frozenset(), frozenset(), None),
}

# What a classifier says about one process's argv.
ARGV_CLAUDE = "claude"
ARGV_OTHER = "other"
ARGV_UNKNOWN = "unknown"


def _interpreter_family(argv0: str) -> str | None:
    """Which flag table applies to this program, or None if it hosts nothing."""
    match = _INTERPRETER.match(os.path.basename(argv0))
    return match.group(1) if match else None


def _may_host_claude(executable: str) -> bool:
    """Whether a main could be running under this executable.

    A `claude` main is either the binary itself or one an interpreter is
    hosting, so an executable that is neither rules the process out without
    reading anything else. Used as the cheap prefilter before the expensive
    per-process read.
    """
    base = os.path.basename(executable)
    return base == "claude" or _INTERPRETER.match(base) is not None


def _pid_map(argv: tuple[str, ...], label: str) -> dict[int, str] | None:
    """pid -> trailing field, for a `ps` format whose last column is one value.

    The value is taken with ``partition`` rather than ``split``, so a path
    containing spaces survives intact — which is the whole reason both callers
    exist rather than reusing the combined probe.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=10
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # ValueError covers UnicodeDecodeError, which is NOT an OSError or a
        # SubprocessError. `errors="replace"` above should stop it arising at
        # all; this is the structural belt, because an escape here would leave
        # profile_is_quiescent raising instead of answering, and take `cswap
        # run` down with it.
        logger.warning("%s failed (%s); continuing without it", label, exc)
        return None
    if proc.returncode != 0:
        logger.warning("%s exited %s; continuing without it",
                       label, proc.returncode)
        return None
    out: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        head, _, value = line.strip().partition(" ")
        try:
            out[int(head)] = value.strip()
        except ValueError:
            continue
    return out


def _env_pairs(env: str) -> tuple[dict[str, str], bool]:
    """Parse a flattened `ps` environment into ``({name: value}, ambiguous)``.

    Each value runs to the start of the next assignment rather than to the
    next space, so a value containing spaces survives and cannot be confused
    with a shorter one that is a prefix of it.

    The parse is a guess, because flattened output carries no quoting: a value
    holding an assignment-shaped word (``NOTE=a B=c``) splits at that word, and
    if the word reuses a name already seen the later fragment overwrites the
    real value. That is not a cosmetic loss — a value containing the text
    ``CLAUDE_CONFIG_DIR=/elsewhere`` would report this profile as UNBOUND and
    let its credentials be rewritten under a live session.

    So repetition is reported rather than resolved. ``ambiguous`` is True when
    any name appears twice, and the caller must treat that pid's binding as
    unknown. Callers that read a structured source instead — ``/proc`` or
    ``KERN_PROCARGS2``, both NUL-separated — never reach this function.
    """
    marks = list(_ENV_ASSIGN.finditer(env))
    pairs: dict[str, str] = {}
    ambiguous = False
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(env)
        name = m.group(1)
        if name in pairs:
            ambiguous = True
        pairs[name] = env[m.end():end]
    return pairs, ambiguous


def _split_argv_env(rest: str, plain_argv: str | None) -> tuple[str, str | None]:
    """Split a `ps ewwx` line body into (argv, env); env None when not visible.

    `ps` appends the environment AFTER argv, and nothing in the combined string
    marks the boundary. Guessing it from the first assignment-shaped token is
    wrong in both directions: a prompt like
    ``claude -p "set CLAUDE_CONFIG_DIR=/elsewhere"`` puts argv text into the
    environment (binding a profile this process never had, pinning it
    non-quiescent), while an env VALUE containing spaces puts environment text
    into argv.

    So the boundary is taken rather than guessed: the same `ps` without the
    `e` flag gives argv alone, and the environment is what the combined line
    has beyond it. The heuristic remains only as a fallback for a pid the
    plain probe did not report — a process that started between the two calls.
    """
    if plain_argv is not None and rest.startswith(plain_argv):
        return plain_argv, rest[len(plain_argv):]
    split = _ENV_PAIR.search(rest)
    if split is None:
        return rest, None
    return rest[: split.start()], rest[split.start():]


def _classify_argv(argv: str) -> str:
    """Classify a FLATTENED argv line: claude, other, or unknown.

    `ps` joins arguments with spaces and quotes nothing, so this cannot always
    tell one argument from two. Where it cannot, it says so. A Bash-tool shell
    inherits CLAUDE_CONFIG_DIR from the claude that spawned it and can outlive
    it as an orphan (64 such strays against one profile here, versus 7 real
    mains), so binding on the variable alone would hold a profile
    un-quiescent forever. The program is the honest test.
    """
    tokens = argv.split()
    if not tokens:
        return ARGV_OTHER
    if os.path.basename(tokens[0]) == "claude":
        return ARGV_CLAUDE
    # Non-native installs reach the CLI through an interpreter, so the binary
    # is not argv[0]: `node /path/to/claude`. Only the SCRIPT position counts.
    # Scanning every argument instead would read `vim /usr/local/bin/claude`
    # or `tail -f .../claude` as a live main and hold the profile
    # un-quiescent for as long as that editor stayed open — and those are
    # exactly the commands run from inside a claude session, which inherit
    # CLAUDE_CONFIG_DIR and so reach this test in the first place. The same
    # reasoning bars matching the npm package directory anywhere in argv:
    # `node worker.js .../@anthropic-ai/claude-code/cli.js` is a worker that
    # was handed the entrypoint's path, not the entrypoint running.
    family = _interpreter_family(tokens[0])
    if family is None:
        return ARGV_OTHER
    table = _FLAG_TABLES[family]
    rest = argv.split(None, 1)[1] if len(tokens) > 1 else ""
    guessed = False
    while rest.startswith("-"):
        parts = rest.split(None, 1)
        flag, remainder = parts[0], (parts[1] if len(parts) > 1 else "")
        base = flag.split("=", 1)[0]
        if base in table.inline or (
                table.cluster is not None and table.cluster.match(base)):
            return ARGV_OTHER       # inline code, so no script argument at all
        if "=" in flag:
            pass                    # self-contained, consumes nothing
        elif base in table.value:
            # The value may itself have held a space, in which case only its
            # first word is being dropped here and the rest still looks like
            # argv. Remember that the walk is no longer exact.
            after = remainder.split(None, 1)
            remainder = after[1] if len(after) > 1 else ""
            guessed = True
        elif base not in table.boolean:
            # An option this module has never seen. Assuming it takes no value
            # reads a real main's script argument as an option's value;
            # assuming it takes one swallows the script. Neither is a guess
            # worth making under a credential rewrite.
            return ARGV_UNKNOWN
        rest = remainder
    script = rest
    if not script:
        return ARGV_OTHER
    first = script.split(None, 1)[0]
    if os.path.basename(first) == "claude" or _CLAUDE_PACKAGE.search(first):
        return ARGV_CLAUDE
    # A first token that already looks like a complete script name ENDS the
    # script argument, so everything after it is claude's own argv rather than
    # more of the path: `node worker.js /tmp/claude` is a worker holding a
    # path, not a main. Without this, such a child — which inherits
    # CLAUDE_CONFIG_DIR and can outlive its parent — would pin the profile
    # non-quiescent forever.
    complete = first.endswith(_SCRIPT_SUFFIXES)
    # The npm entrypoint under a path with a space in it arrives split, so its
    # first token is a bare directory and the package segment is further
    # along. That only applies while the first token is NOT a script name of
    # its own — otherwise `node worker.js .../claude-code/cli.js` reads as the
    # entrypoint when it is a worker holding the entrypoint's path.
    if not complete and _CLAUDE_PACKAGE.search(script):
        return ARGV_CLAUDE
    if guessed:
        # A token here may be the tail of a consumed value rather than the
        # script, so a negative verdict would be a guess.
        return ARGV_UNKNOWN
    if complete:
        return ARGV_OTHER
    # Otherwise the script path may simply contain spaces (`node
    # "/Users/me/Application Support/claude"`), where splitting yields
    # `/Users/me/Application` and rejects a live main. Only the path's tail is
    # needed to recognise it, and the tail has no space in it.
    return (ARGV_CLAUDE if _SCRIPT_TAIL.search(script)
            else ARGV_OTHER)


class ProcArgs(NamedTuple):
    """A process's argv and environment, as the kernel holds them."""

    argv: list[str]
    env: dict[str, str] | None


def _classify_argv_tokens(argv: list[str]) -> str:
    """Classify an argv that arrived already separated: claude, other, unknown.

    The exact counterpart of :func:`_classify_argv`, which has to re-split a
    flattened line. Nothing here needs the suffix or tail heuristics: the
    script argument is a single token, so
    ``/Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js`` is
    recognised whole instead of being cut at the space and read as a worker.
    An option in neither flag list is still unknown — separated arguments say
    where a token ends, not what an option means.
    """
    if not argv:
        return ARGV_OTHER
    if os.path.basename(argv[0]) == "claude":
        return ARGV_CLAUDE
    family = _interpreter_family(argv[0])
    if family is None:
        return ARGV_OTHER
    table = _FLAG_TABLES[family]
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        flag = argv[i]
        base = flag.split("=", 1)[0]
        if base in table.inline or (
                table.cluster is not None and table.cluster.match(base)):
            return ARGV_OTHER
        if "=" in flag:
            pass
        elif base in table.value:
            i += 1
        elif base not in table.boolean:
            return ARGV_UNKNOWN
        i += 1
    if i >= len(argv):
        return ARGV_OTHER
    script = argv[i]
    if (os.path.basename(script) == "claude"
            or _CLAUDE_PACKAGE.search(script)):
        return ARGV_CLAUDE
    return ARGV_OTHER


def _split_nul_env(records: list[bytes]) -> dict[str, str]:
    """NUL-separated ``NAME=VALUE`` records into a dict.

    The separator is the kernel's, not a guess, so a value containing spaces,
    newlines, or an ``=`` sign survives whole. That is the whole reason this
    path exists beside the `ps` one.

    An empty record TERMINATES the environment rather than being skipped. The
    kernel writes the area as one NUL-terminated run, so the first empty slot
    is the end of it; anything past that is whatever the region held before,
    and reading it back would attribute a stale variable to a live process.
    """
    env: dict[str, str] = {}
    for chunk in records:
        if not chunk:
            break
        name, sep, value = chunk.partition(b"=")
        if sep:
            env.setdefault(name.decode("utf-8", "replace"),
                           value.decode("utf-8", "replace"))
    return env


def _proc_args_linux(pid: int) -> ProcArgs | None:
    """argv and environment from ``/proc/<pid>``, or None when unreadable."""
    base = Path("/proc") / str(pid)
    try:
        raw_argv = (base / "cmdline").read_bytes()
    except OSError:
        return None
    argv = [c.decode("utf-8", "replace") for c in raw_argv.split(b"\0") if c]
    try:
        raw_env = (base / "environ").read_bytes()
    except OSError:
        # Another user's process: the kernel shows argv and withholds environ.
        return ProcArgs(argv, None)
    return ProcArgs(argv, _split_nul_env(raw_env.split(b"\0")))


_CTL_KERN = 1
_KERN_ARGMAX = 8
_KERN_PROCARGS2 = 49
_PROC_PIDPATHINFO_MAXSIZE = 4 * 1024
_libc: ctypes.CDLL | None = None
_libc_tried = False
_argmax: int | None = None


def _load_libc() -> ctypes.CDLL | None:
    """libc with ``sysctl`` and ``proc_pidpath`` bound, or None if unreachable."""
    global _libc, _libc_tried
    if _libc_tried:
        return _libc
    _libc_tried = True
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t,
        ]
        libc.sysctl.restype = ctypes.c_int
    except (OSError, AttributeError, TypeError) as exc:
        logger.debug("libc sysctl unavailable: %r", exc)
        return None
    _libc = libc
    return _libc


def _sysctl(mib_values: list[int], size: int) -> bytes:
    """One ``sysctl`` read into a fixed buffer. Raises OSError on refusal."""
    libc = _load_libc()
    if libc is None:
        raise OSError(0, "libc unavailable")
    mib = (ctypes.c_int * len(mib_values))(*mib_values)
    buf = ctypes.create_string_buffer(size)
    length = ctypes.c_size_t(size)
    if libc.sysctl(mib, len(mib_values), buf, ctypes.byref(length),
                   None, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return buf.raw[:length.value]


def _macos_argmax() -> int | None:
    """The kernel's argument-area size, read once. None when unavailable."""
    global _argmax
    if _argmax is None:
        try:
            _argmax = int.from_bytes(
                _sysctl([_CTL_KERN, _KERN_ARGMAX], 4), sys.byteorder
            )
        except OSError as exc:
            logger.debug("KERN_ARGMAX unavailable: %r", exc)
            return None
    return _argmax


def _exec_path_macos(pid: int) -> str | None:
    """The executable behind a pid, from ``proc_pidpath``. None when refused.

    Two orders of magnitude cheaper than reading the whole argument area, and
    enough to rule out every process that is neither `claude` nor an
    interpreter — which is nearly all of them.
    """
    libc = _load_libc()
    if libc is None or not hasattr(libc, "proc_pidpath"):
        return None
    buf = ctypes.create_string_buffer(_PROC_PIDPATHINFO_MAXSIZE)
    written = libc.proc_pidpath(pid, buf, _PROC_PIDPATHINFO_MAXSIZE)
    if written <= 0:
        return None
    return buf.raw[:written].decode("utf-8", "replace")


def _proc_args_macos(pid: int) -> ProcArgs | None:
    """argv and environment from ``KERN_PROCARGS2``, or None when refused.

    The kernel hands back ``argc``, the exec path, then argv and the
    environment as NUL-separated records — the same bytes ``execve`` was
    given. It refuses with EPERM for another user's process and for the
    platform binaries macOS protects, and with EINVAL for a pid that has
    already exited; all three are "unknown for this pid", never "unbound".

    A buffer holding fewer than ``argc`` records is refused too. Short output
    means the copy was truncated, and reading the environment out of what
    followed would take argv fragments for variables.
    """
    size = _macos_argmax()
    if size is None:
        return None
    try:
        raw = _sysctl([_CTL_KERN, _KERN_PROCARGS2, pid], size)
    except OSError:
        return None
    if len(raw) < 5:
        return None
    argc = int.from_bytes(raw[:4], sys.byteorder)
    rest = raw[4:]
    end = rest.find(b"\0")
    if end < 0:
        return None
    # The exec path is followed by alignment NULs before argv[0] starts.
    i = end
    while i < len(rest) and rest[i:i + 1] == b"\0":
        i += 1
    records = rest[i:].split(b"\0")
    if len(records) < argc:
        return None
    argv = [c.decode("utf-8", "replace") for c in records[:argc]]
    return ProcArgs(argv, _split_nul_env(records[argc:]))


def _cannot_host_claude(pid: int) -> bool:
    """Whether a cheap, POSITIVE read rules this pid out as a claude main.

    False whenever the read failed, so "we could not look" never shortens the
    work. On macOS this is ``proc_pidpath``; on Linux it is the world-readable
    ``cmdline``, which also means the environment is never opened for a
    process that could not be a main anyway.
    """
    if sys.platform == "darwin":
        executable = _exec_path_macos(pid)
        return executable is not None and not _may_host_claude(executable)
    if sys.platform.startswith("linux"):
        try:
            raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError:
            return False
        argv = [c.decode("utf-8", "replace") for c in raw.split(b"\0") if c]
        return _classify_argv_tokens(argv) == ARGV_OTHER
    return False


def _read_proc_args(pid: int) -> ProcArgs | None:
    """Structured argv and environment for one pid, or None when unreadable.

    Structured means NUL-separated, straight from the kernel: no whitespace
    splitting, so nothing here can confuse an argument with an environment
    variable or a value with the value beside it. `ps` remains only as the
    last resort for pids and platforms this cannot reach.
    """
    if sys.platform == "darwin":
        return _proc_args_macos(pid)
    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        return _proc_args_linux(pid)
    return None


def _candidate_pids() -> list[int] | None:
    """Every pid on the machine, or None when they could not be listed."""
    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        try:
            return [int(n) for n in os.listdir("/proc") if n.isdigit()]
        except OSError as exc:
            logger.warning("Could not list /proc (%s); profile liveness is "
                           "unknown", exc)
            return None
    listing = _pid_map(_PID_PROBE_ARGV, "pid probe")
    if listing is None:
        return None
    return list(listing)


# One pid's relationship to a profile.
_BOUND = "bound"
_UNBOUND = "unbound"
_UNKNOWN = "unknown"
_UNREADABLE = "unreadable"      # nothing structured covered it; try `ps`
_GONE = "gone"                  # exited between the listing and the read


class _PsFallback:
    """Last-resort view of the process table, for pids no kernel source covers.

    Everything `ps` prints is flattened and space-separated, so argv and the
    environment run together and a value with a space in it is indistinguishable
    from two values. That ambiguity is reported, never resolved.

    Each of the three probes can fail on its own, and a failed probe is an
    ABSENT answer, not a negative one. A pid this class has no trusted reading
    of is unknown — never unbound — because "we could not look" is the one
    thing that must not read as "nobody is there".
    """

    def __init__(self) -> None:
        combined = _pid_map(_ENV_PROBE_ARGV, "CLAUDE_CONFIG_DIR probe")
        argvs = _pid_map(_ARGV_PROBE_ARGV, "argv probe")
        executables = _pid_map(_COMM_PROBE_ARGV, "executable probe")
        self.combined = combined
        self.argvs = argvs
        self.executables = executables

    def _argv_text(self, pid: int) -> str | None:
        """This pid's argv from whichever probe reported it, or None."""
        if self.argvs is not None and pid in self.argvs:
            return self.argvs[pid]
        if self.combined is not None and pid in self.combined:
            # The combined line's argv/env boundary is a guess, but argv is a
            # PREFIX either way, so the program in argv[0] is intact.
            return _split_argv_env(self.combined[pid],
                                   None if self.argvs is None
                                   else self.argvs.get(pid))[0]
        return None

    def verdict(self, pid: int, target: str) -> str:
        """``bound``, ``unbound``, or ``unknown`` for one pid."""
        executable = (None if self.executables is None
                      else self.executables.get(pid))
        argv = self._argv_text(pid)
        if executable is not None and not _may_host_claude(executable):
            # Positively read, and a main runs as either the binary or an
            # interpreter, so this one cannot be a main. Most of the process
            # table lands here; failing closed on all of it would wedge every
            # profile on a machine with other users.
            return _UNBOUND
        if argv is None:
            return _UNKNOWN         # no trusted reading of this pid at all
        verdict = _classify_argv(argv)
        if verdict == ARGV_OTHER:
            return _UNBOUND
        if verdict == ARGV_UNKNOWN:
            return _UNKNOWN
        if self.combined is None or pid not in self.combined:
            return _UNKNOWN         # a main, and no environment to read
        _, env = _split_argv_env(
            self.combined[pid],
            None if self.argvs is None else self.argvs.get(pid),
        )
        if not env:
            # argv with no environment after it: macOS withholds the
            # environment of another user's process and of the platform
            # binaries it protects.
            return _UNKNOWN
        pairs, ambiguous = _env_pairs(env)
        value = pairs.get("CLAUDE_CONFIG_DIR")
        if ambiguous:
            return _UNKNOWN
        if value is not None and _WHITESPACE.search(value):
            # A value that reaches a space may have ended there or may not;
            # the flattened form does not say which.
            return _UNKNOWN
        if value != target and _bound_text(target).search(env):
            # The binding is there as a complete value, yet the parse read
            # something else — a later assignment-shaped word overwrote it.
            return _UNKNOWN
        return _BOUND if value == target else _UNBOUND


def _scan_pid(pid: int, target: str) -> str:
    """One pid's relationship to ``target``, from the structured source."""
    if _cannot_host_claude(pid):
        return _UNBOUND
    info = _read_proc_args(pid)
    if info is None:
        return _GONE if not is_pid_alive(pid) else _UNREADABLE
    verdict = _classify_argv_tokens(info.argv)
    if verdict == ARGV_OTHER:
        return _UNBOUND
    if verdict == ARGV_UNKNOWN:
        return _UNKNOWN
    if info.env is None:
        # Linux shows every process's cmdline and withholds other users'
        # environ. A main whose environment is hidden is unknown.
        logger.debug("pid %s looks like a claude main but its environment "
                     "is not readable", pid)
        return _UNKNOWN
    return _BOUND if info.env.get("CLAUDE_CONFIG_DIR") == target else _UNBOUND


def scan_env_bound_claude(session_dir: Path) -> tuple[list[int], bool]:
    """Live `claude` PIDs bound to ``session_dir``, and whether the probe ran.

    Returns ``(pids, probed)``. ``probed`` is False when some pid that might be
    a claude main could not be resolved. The caller must treat that as
    "unknown", not as "nobody is there" — the step behind
    :func:`profile_is_quiescent` rewrites credentials, and doing that under a
    session this probe failed to see is the mid-session logout the probe was
    added to prevent. An empty list with ``probed`` True is the only answer
    that means the profile is idle.

    A pid is reported unbound only when its program and its environment were
    both read from a source that says where each argument ends, and neither
    named claude. Everything else is unknown.

    Each pid is read from ``/proc`` on Linux or ``KERN_PROCARGS2`` on macOS.
    Those give argv and the environment as the kernel holds them, so a path
    with a space in it stays one argument and an environment value cannot be
    confused with the argument beside it. `ps` is consulted only for pids those
    sources refuse, and an answer it cannot give unambiguously counts as
    unknown rather than as a guess.

    The pid list is a SUPPLEMENT to the session registry and may only ever ADD
    liveness it missed. It stops at the first main it finds, because every
    caller asks whether the profile is busy rather than by whom, so a non-empty
    list is not exhaustive and ``probed`` only means anything when it is empty.

    Windows is the one exception, reported as ``(…, True)``. There is no cheap
    environment read there, so this profile keeps the registry-only behavior
    that shipped before this probe existed: an unregistered `claude --resume`
    on Windows can still have its credentials rewritten underneath it. Failing
    closed instead would leave every Windows profile permanently
    un-re-seedable, which is the worse of the two.
    """
    if sys.platform == "win32":
        return [], True

    pids = _candidate_pids()
    if pids is None:
        logger.warning("CLAUDE_CONFIG_DIR probe could not list processes; "
                       "profile liveness is unknown")
        return [], False

    target = str(session_dir)
    probed = True
    fallback: _PsFallback | None = None
    for pid in pids:
        answer = _scan_pid(pid, target)
        if answer == _UNREADABLE:
            if fallback is None:
                fallback = _PsFallback()
            answer = fallback.verdict(pid, target)
        if answer == _BOUND:
            return [pid], probed
        if answer == _UNKNOWN:
            logger.debug("pid %s could not be resolved for or against %s",
                         pid, target)
            probed = False
    if not probed:
        logger.warning("CLAUDE_CONFIG_DIR probe could not resolve every "
                       "process; profile liveness is unknown")
    return [], probed


def list_ide_instances(claude_dir: Path | None = None) -> list[IdeInstance]:
    """Read IDE lockfiles and return only those with alive processes."""
    ide_dir = (claude_dir or get_claude_dir()) / "ide"
    if not ide_dir.is_dir():
        return []

    instances = []
    for path in ide_dir.glob("*.lock"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = data.get("pid")
            if pid is None or not is_pid_alive(pid):
                continue
            port = int(path.stem)
            instances.append(IdeInstance(
                port=port,
                pid=pid,
                ide_name=data.get("ideName", "Unknown IDE"),
                workspace_folders=data.get("workspaceFolders", []),
            ))
        except (
            json.JSONDecodeError,   # malformed JSON
            KeyError,               # required field missing
            TypeError,              # field has the wrong type (e.g. pid not an int)
            AttributeError,         # valid JSON that is not an object: a
                                    # top-level array reaches `.get` as a list.
                                    # Also how a too-deep nesting lands where
                                    # the parser's recursion limit is high
                                    # enough not to raise -- it differs per
                                    # machine, so BOTH outcomes must be inert.
            ValueError,             # includes UnicodeDecodeError from read_text
            OverflowError,          # pid too large for os.kill's C long (is_pid_alive)
            RecursionError,         # pathologically nested JSON in json.loads
            OSError,
        ) as exc:
            logger.debug("Skipping IDE lockfile %s: %s", path, exc)
    return instances


def get_running_instances(
    claude_dir: Path | None = None,
) -> tuple[list[ClaudeSession], list[IdeInstance]]:
    """Return all running Claude Code sessions and IDE instances."""
    resolved = claude_dir or get_claude_dir()
    return list_sessions(resolved), list_ide_instances(resolved)
