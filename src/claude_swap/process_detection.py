"""Detect running Claude Code instances.

Reads session PID files (~/.claude/sessions/{pid}.json) and IDE lockfiles
(~/.claude/ide/{port}.lock) to determine which Claude Code instances are
currently running. Uses the same mechanism Claude Code itself uses internally.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

# Runtimes a non-native install can front the CLI with. Only these get their
# script argument inspected; for anything else argv[0] is the whole test.
_SCRIPT_INTERPRETERS = frozenset({"node", "bun", "deno", "python", "python3"})

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

# Interpreter options that consume the FOLLOWING token. Skipping only the
# option leaves its value looking like the script: `node --require setup.js
# .../claude-code/cli.js` would be filed as a worker called setup.js, missing a
# live main. `--opt=value` forms need no entry here; they are self-contained.
_VALUE_FLAGS = frozenset({
    "-r", "--require", "--import", "--loader", "--experimental-loader",
    "--env-file", "-C", "--conditions",
})

# Options whose value IS the program. There is no script argument after one of
# these, so every remaining token is argv for the inline code: `node --eval
# 'setInterval(...)' /opt/bin/claude` holds a path, it does not run claude.
# Kept apart from _VALUE_FLAGS, which merely consume a token and carry on.
_EVAL_FLAGS = frozenset({"-e", "--eval", "-p", "--print"})


def _pid_map(argv: tuple[str, ...], label: str) -> dict[int, str]:
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
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        head, _, value = line.strip().partition(" ")
        try:
            out[int(head)] = value.strip()
        except ValueError:
            continue
    return out


def _executables_by_pid() -> dict[int, str]:
    """pid -> executable path. Empty when the probe cannot run."""
    return _pid_map(_COMM_PROBE_ARGV, "executable probe")


def _argv_by_pid() -> dict[int, str]:
    """pid -> argv, with no environment appended. Empty when unavailable."""
    return _pid_map(_ARGV_PROBE_ARGV, "argv probe")


def _env_pairs(env: str) -> dict[str, str]:
    """Parse a flattened `ps` environment into ``{name: value}``.

    Each value runs to the start of the next assignment rather than to the
    next space, so a value containing spaces survives and cannot be confused
    with a shorter one that is a prefix of it.

    A value that itself contains an assignment-shaped word (``NOTE=a B=c``)
    still splits at that word — flattened output carries no quoting, so the
    ambiguity is in the data, not the parse. It costs a spurious extra name,
    never a wrong value for the name this module reads.
    """
    marks = list(_ENV_ASSIGN.finditer(env))
    pairs: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(env)
        pairs[m.group(1)] = env[m.end():end]
    return pairs


def _is_claude_executable(executable: str | None) -> bool:
    """Whether the resolved executable path IS the claude binary."""
    return bool(executable) and os.path.basename(executable) == "claude"


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


def _is_claude_argv(argv: str) -> bool:
    """True when argv is a `claude` main, not something that inherited its env.

    A Bash-tool shell inherits CLAUDE_CONFIG_DIR from the claude that spawned
    it and can outlive it as an orphan, so binding on the variable alone would
    hold a profile un-quiescent forever (64 such strays against one profile
    here, versus 7 real mains). The binary is the honest test.
    """
    tokens = argv.split()
    if not tokens:
        return False
    if os.path.basename(tokens[0]) == "claude":
        return True
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
    if os.path.basename(tokens[0]) not in _SCRIPT_INTERPRETERS:
        return False
    rest = argv.split(None, 1)[1] if len(tokens) > 1 else ""
    script = _interpreter_script(rest)
    if script is None:
        return False
    first = script.split(None, 1)[0]
    if os.path.basename(first) == "claude":
        return True
    if _CLAUDE_PACKAGE.search(first):
        return True  # npm layout: .../@anthropic-ai/claude-code/cli.js
    # A first token that already looks like a complete script name ENDS the
    # script argument, so everything after it is claude's own argv rather than
    # more of the path: `node worker.js /tmp/claude` is a worker holding a
    # path, not a main. Without this, such a child — which inherits
    # CLAUDE_CONFIG_DIR and can outlive its parent — would pin the profile
    # non-quiescent forever.
    if first.endswith(_SCRIPT_SUFFIXES):
        return False
    # Otherwise the script path may simply contain spaces (`node
    # "/Users/me/Application Support/claude"`), where splitting yields
    # `/Users/me/Application` and rejects a live main. Only the path's tail is
    # needed to recognise it, and the tail has no space in it.
    return _SCRIPT_TAIL.search(script) is not None


def _interpreter_script(rest: str) -> str | None:
    """Everything from the script argument on, or None when there is no script.

    ``rest`` is argv after the interpreter. Interpreter options precede the
    script path, and two kinds have to be told apart:

    * an option that CONSUMES the next token (``--require setup.js``) — drop
      both, or the value is mistaken for the script and a live main is missed;
    * an option that REPLACES the script with inline code (``--eval``,
      ``--print``) — there is no script at all, so every token after the code
      is argv for that inline program. ``node --eval '…' /opt/bin/claude`` is
      a one-liner that was handed a path, and reading it as a main pins the
      profile non-quiescent for as long as it runs.

    The value is returned rather than the bare first token because a script
    path may contain spaces; the caller needs the tail to recognise it.
    """
    while rest.startswith("-"):
        parts = rest.split(None, 1)
        flag, remainder = parts[0], (parts[1] if len(parts) > 1 else "")
        base = flag.split("=", 1)[0]
        if base in _EVAL_FLAGS:
            # Inline code replaces the script. `--eval=CODE` carries its own
            # value; the separate form consumes the next token. Either way
            # nothing after this point is a script path.
            return None
        if "=" not in flag and base in _VALUE_FLAGS:
            after = remainder.split(None, 1)
            remainder = after[1] if len(after) > 1 else ""
        rest = remainder
    return rest or None


def scan_env_bound_claude(session_dir: Path) -> tuple[list[int], bool]:
    """Live `claude` PIDs bound to ``session_dir``, and whether the probe ran.

    Returns ``(pids, probed)``. ``probed`` is False when the OS could not be
    asked at all: `ps` is absent, raised, timed out, or exited nonzero. The
    caller must treat that as "unknown", not as "nobody is there" — the step
    behind :func:`profile_is_quiescent` rewrites credentials, and doing that
    under a session this probe simply failed to see is the mid-session logout
    the probe was added to prevent. An empty list with ``probed`` True is the
    only answer that means the profile is idle.

    The pid list is still a SUPPLEMENT to the session registry and may only
    ever ADD liveness it missed; the flag is what stops a broken probe from
    silently subtracting the signal instead.

    Windows is the one exception, reported as ``(…, True)``: there is no cheap
    environment read there, so the probe is not broken but absent. Failing
    closed on it would leave every Windows profile permanently un-re-seedable,
    which is a worse bargain than the registry alone.
    """
    if sys.platform == "win32":
        return [], True
    if shutil.which("ps") is None:
        logger.warning("CLAUDE_CONFIG_DIR probe unavailable (no `ps`); "
                       "profile liveness is unknown")
        return [], False

    try:
        proc = subprocess.run(
            _ENV_PROBE_ARGV, capture_output=True, text=True,
            errors="replace", timeout=10
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # See _executables_by_pid: ValueError is here for UnicodeDecodeError,
        # which neither of the other two clauses covers. Raising out of the
        # probe would break `cswap run` outright, so the failure is reported
        # rather than raised.
        logger.warning("CLAUDE_CONFIG_DIR probe failed (%s); profile liveness "
                       "is unknown", exc)
        return [], False
    if proc.returncode != 0:
        logger.warning("CLAUDE_CONFIG_DIR probe exited %s; profile liveness "
                       "is unknown", proc.returncode)
        return [], False

    target = str(session_dir)
    executables = _executables_by_pid()
    argvs = _argv_by_pid()
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        head, _, rest = line.strip().partition(" ")
        try:
            pid_candidate = int(head)
        except ValueError:
            continue
        argv, env = _split_argv_env(rest, argvs.get(pid_candidate))
        if env is None:
            continue  # no environment visible (another user's process)
        # Exact name, exact value. A sibling profile `<dir>-old` differs in the
        # value and a variable merely ENDING in the name differs in the key,
        # so neither can bind this directory.
        if _env_pairs(env).get("CLAUDE_CONFIG_DIR") != target:
            continue
        pid = pid_candidate
        # Executable first: it is space-safe. argv is the fallback that still
        # catches an interpreter-hosted install, where the binary is `node` and
        # only the script argument names claude.
        if not (_is_claude_executable(executables.get(pid))
                or _is_claude_argv(argv)):
            continue
        pids.append(pid)
    return pids, True


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
