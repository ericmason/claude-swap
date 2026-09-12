"""Detect running Claude Code instances.

Reads session PID files (~/.claude/sessions/{pid}.json) and IDE lockfiles
(~/.claude/ide/{port}.lock) to determine which Claude Code instances are
currently running. Uses the same mechanism Claude Code itself uses internally.
"""

from __future__ import annotations

import calendar
import ctypes
import ctypes.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from claude_swap._node_options import NODE_BOOLEAN_FLAGS, NODE_VALUE_FLAGS
from claude_swap.paths import get_claude_config_home

logger = logging.getLogger(__name__)

# A session record names a pid, and the OS recycles pids: once the claude
# that wrote the record is gone, the number can belong to anything. The
# record also carries claude's reading of its own start (``procStart``): on
# Linux the ``/proc/<pid>/stat`` start time in clock ticks since boot, fixed
# for the process's lifetime, and elsewhere ``ps -o lstart``, a wall-clock
# time. Only the latter needs slack, for the small clock steps that move a
# ``ps`` start time on some platforms.
PID_REUSE_SLACK_S = 120

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


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


def _ps(pid: int, *columns: str) -> str | None:
    """``ps -o`` ``columns`` for ``pid``, or None when unknowable.

    POSIX only, under ``LC_ALL=C TZ=UTC`` like claude's own reading. Windows
    and every failure answer None: not knowing must never be read as "not
    the recorded process".
    """
    if sys.platform == "win32":
        return None
    try:
        proc = subprocess.run(
            ["ps", "-o", ",".join(f"{c}=" for c in columns), "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        return None
    return text


def process_started_at(pid: int) -> int | None:
    """Epoch seconds at which ``pid`` started, or None when unknowable.

    Read the way claude stamps ``procStart`` into its record, ``ps -o
    lstart=`` under ``LC_ALL=C TZ=UTC``, so the two agree to the second for
    the same process.
    """
    text = _ps(pid, "lstart")
    if text is None:
        return None
    try:
        return _lstart_seconds(text)
    except ValueError:
        return None


def _lstart_seconds(text: str) -> int:
    """``Wed Sep  2 20:35:59 2026``, the ``ps -o lstart`` format under
    ``LC_ALL=C TZ=UTC``, as epoch seconds. Parsed by hand because ``strptime``
    reads month names in the process locale."""
    parts = text.split()
    if len(parts) != 5 or parts[1] not in _MONTHS:
        raise ValueError(text)
    _, month, day, clock, year = parts
    hours, minutes, seconds = (int(p) for p in clock.split(":"))
    return calendar.timegm(
        (int(year), _MONTHS.index(month) + 1, int(day), hours, minutes, seconds, 0, 0, 0)
    )


def process_start_ticks(pid: int) -> str | None:
    """``/proc/<pid>/stat``'s start time, in clock ticks since boot, or None
    when unreadable. Linux by construction: the file exists nowhere else."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    return _stat_start_ticks(text)


def _stat_start_ticks(text: str) -> str | None:
    """Field 22 of a ``/proc/<pid>/stat`` line, as the string claude stamps
    into ``procStart``. The command name sits in parentheses before the
    numeric fields and may itself contain spaces or parentheses, so the
    fields are counted from the last closing one."""
    _, _, rest = text.rpartition(")")
    fields = rest.split()
    if len(fields) <= 19 or not fields[19].isdigit():
        return None
    return fields[19]


def process_is_claude(pid: int) -> bool | None:
    """Does the process at ``pid`` look like a claude, or None when unknowable.

    Judged from ``ps -o comm=,args=``: the native binary and the symlink to
    it are named ``claude``, and an npm install runs ``cli.js`` out of a
    ``claude-code`` package directory.
    """
    text = _ps(pid, "comm", "args")
    if text is None:
        return None
    return "claude" in text.lower()


def pid_matches_record(pid: int, proc_start: str | None) -> bool:
    """Is the live process at ``pid`` the one that wrote a record stamped
    ``proc_start``, claude's reading of its own start?

    On Linux the stamp is the ``/proc/<pid>/stat`` start time in clock ticks
    since boot, which a process keeps for life and no two processes at one
    pid share, so equality is the whole test and no clock domain is
    involved. Elsewhere it is ``ps -o lstart``, a wall-clock time: a
    recycled pid belongs to a process that started after the recorded claude
    did, and only that direction disqualifies, and only a stranger. Some
    ``ps`` builds derive every start time from a boot time that moves with
    each wall-clock step (a WSL2 resume re-syncing the clock steps it by the
    whole sleep), so a live session can read as younger than its own record;
    a claude at the pid is kept either way, and a pid genuinely recycled by
    another claude lingers only for that process's lifetime, which is what
    happened before this check. Everything unknowable (Windows, whose stamp
    is a FILETIME with no ``/proc`` to check it against, ``ps`` unavailable,
    an unstamped or unparseable record) passes, because "cannot tell" must
    never turn a live session into "nobody there".
    """
    if not proc_start:
        return True
    if proc_start.isdigit():
        ticks = process_start_ticks(pid)
        return ticks is None or ticks == proc_start
    try:
        recorded = _lstart_seconds(proc_start)
    except ValueError:
        return True
    started = process_started_at(pid)
    if started is None or started <= recorded + PID_REUSE_SLACK_S:
        return True
    return process_is_claude(pid) is not False


def scan_sessions(claude_dir: Path | None = None) -> tuple[list[ClaudeSession], int]:
    """Live sessions, and how many records could NOT be read.

    A record counts as live only when its pid is alive AND still belongs to
    the claude that wrote it (see ``pid_matches_record``): a crashed claude
    leaves its record behind, and the OS can hand the number to something
    else.

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
            if not pid_matches_record(pid, data.get("procStart")):
                logger.debug(
                    "Skipping session file %s: pid %s was recycled", path, pid
                )
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

# Who owns each process. A `claude` bound to a profile this tool manages runs
# as the user running this tool, so a process owned by someone else is not the
# session at risk — and its environment is unreadable here anyway, which would
# otherwise leave every root daemon on the machine permanently unknown.
_UID_PROBE_ARGV = ("ps", "-Ao", "pid=,uid=")

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
# version, because `node22` and `python3.13` are the same programs. Debian
# ships node as `nodejs`, and a framework build runs as `Python` with a
# capital P from inside `Python.app/Contents/MacOS`, so neither the spelling
# nor the case may decide whether a program can host the CLI.
_INTERPRETER = re.compile(r"^(nodejs|node|bun|deno|python)[\d.]*$",
                          re.IGNORECASE)

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

# Interpreter options, split by what they do to the token after them. The node
# tables are GENERATED from `node --help` and `node --v8-options` by
# tools/regen_node_options.py, never transcribed: hand-maintaining them left
# `--inspect-port` and `--diagnostic-dir` out and filed
# `--experimental-test-isolation` as taking no value when it takes one, and
# each of those misreads a live main. A test diffs the shipped tables against
# the `node` on the machine running it, so a node upgrade shows up as a test
# failure rather than as a wrong answer.
#
# The tables are also CLOSED: an option in neither list is not assumed
# harmless, because assuming wrongly reads a real main's script argument as an
# option's value and reports a live profile as idle. Anything unlisted makes
# the argv unknown, which fails closed — a deferred re-seed rather than a
# logout.
#
# A flag written `--opt=value` is self-contained whatever it means, so it never
# reaches the lists and never makes anything unknown.
#
# The short aliases are added here because `node --help` prints them beside
# their long spellings rather than as entries of their own.
_NODE_VALUE_FLAGS = NODE_VALUE_FLAGS | frozenset({"-C", "-r"})
_NODE_BOOL_FLAGS = NODE_BOOLEAN_FLAGS | frozenset({"-c", "-h", "-i", "-v"})

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
    if match is None:
        return None
    name = match.group(1).lower()
    return "node" if name == "nodejs" else name


# The native installer runs the CLI from a VERSION-NAMED file — on this
# machine `~/.local/share/claude/versions/2.1.269`, reached through a
# `~/.local/bin/claude` symlink — so the running executable's basename is a
# version number and nothing about it says "claude". Recognising only the
# literal basename ruled out every real session on this Mac.
# Apple's `ps` and `pgrep` print the kernel's short process name wrapped in
# parentheses when they cannot read the argument area. It is a read failure
# wearing the shape of an answer, in the executable column and in argv alike.
_UNREADABLE_FIELD = re.compile(r"^\(.*\)$")

# `bun run x.js` and `deno run x.js` put a subcommand where node puts the
# script. Reading the subcommand as the script filed every such main as a
# program called `run`.
_INTERPRETER_SUBCOMMANDS = frozenset({"run", "exec", "x"})

_CLAUDE_PATH_PART = re.compile(r"(?:^|/)claude(?:/|$)")
_CLAUDE_VERSIONS = re.compile(r"(?:^|/)claude/versions/")
_VERSION_NAME = re.compile(r"^[0-9][0-9A-Za-z._+-]*$")

# The resolved binary and the fact that it was resolved are ONE value, a
# single-element tuple, so publishing them is one assignment and a reader can
# never see "already looked" without the answer that look produced. Two
# separate variables let a second thread arrive between the flag and the
# value, read None, and classify a live native main as not-the-CLI: the
# menubar refresh thread and the CLI run this concurrently, and the window is
# a `shutil.which` PATH walk wide.
_claude_binary_cache: tuple[str | None] | None = None
_claude_binary_lock = threading.Lock()


def _native_claude_binary() -> str | None:
    """The real file behind `claude` on PATH, or None when it cannot be found.

    `which claude` may land on a wrapper script rather than the CLI — the
    CodeToGo shim on this machine is one — so this is a hint that ADDS
    recognition, never one the prefilter depends on.

    Resolved once and cached. The lock keeps concurrent callers from each
    walking PATH; the single-assignment publish above is what keeps them from
    reading a half-initialised answer.
    """
    global _claude_binary_cache
    cached = _claude_binary_cache
    if cached is not None:
        return cached[0]
    with _claude_binary_lock:
        cached = _claude_binary_cache
        if cached is not None:
            return cached[0]
        found = shutil.which("claude")
        resolved = None if found is None else os.path.realpath(found)
        _claude_binary_cache = (resolved,)
        return resolved


def _is_claude_binary(executable: str) -> bool:
    """Whether this executable IS the CLI, rather than something hosting it.

    Stronger than :func:`_may_host_claude`: an interpreter may host a main and
    may host anything else, but this answers yes only for the CLI itself, so a
    yes settles the process without reading argv at all.
    """
    return (os.path.basename(executable) == "claude"
            or executable == _native_claude_binary()
            or _CLAUDE_VERSIONS.search(executable) is not None)


# Directories macOS SEALS. Their contents are cryptographically verified and
# replaced only by a system update, so nobody can install the CLI into one and
# no process running from one can be a Claude main. This is the only location
# rule strong enough to settle a pid without reading its argv.
_SEALED_DIRS = (
    "/System/", "/usr/libexec/", "/usr/sbin/", "/sbin/", "/usr/bin/", "/bin/",
    "/usr/share/", "/Library/Apple/",
)

# Locations where a Claude main is merely UNLIKELY: third-party software under
# `/Library`, and the executable inside an application or framework bundle.
# Anyone can write to these, so `/Library/Acme/assistant` running
# `claude --resume` is a real shape. A process here is ruled out only once its
# argv has been read and says it is not a main.
_THIRD_PARTY_DIRS = ("/Library/",)
_BUNDLE_EXECUTABLE = re.compile(
    r"/[^/]+\.(?:app|framework|bundle|xpc|systemextension)/Contents/")

# Programs a claude session starts, or that start beside one, and that are
# positively not it. They inherit CLAUDE_CONFIG_DIR and can outlive the
# session as orphans, so recognising them is what keeps a profile re-seedable
# — an unrecognised stray with the variable set defers forever.
_NOT_CLAUDE_PROGRAMS = frozenset({
    # shells
    "sh", "bash", "zsh", "dash", "ksh", "tcsh", "csh", "fish", "login",
    # the tool subprocesses a session runs
    "rg", "grep", "egrep", "fgrep", "ag", "ack", "find", "fd", "xargs",
    "sed", "awk", "cat", "head", "tail", "sort", "uniq", "cut", "tr", "wc",
    "ls", "cp", "mv", "rm", "mkdir", "touch", "chmod", "chown", "ln", "df",
    "du", "stat", "file", "which", "env", "echo", "sleep", "true", "false",
    "date", "basename", "dirname", "realpath", "readlink", "tee", "diff",
    "patch", "tar", "gzip", "gunzip", "zip", "unzip", "xz", "zstd", "jq",
    "yq", "curl", "wget", "ssh", "scp", "rsync", "ping", "dig", "nc",
    "git", "gh", "hub", "svn", "hg", "make", "cmake", "ninja", "gcc", "g++",
    "clang", "clang++", "cc", "c++", "ld", "ar", "nm", "strip", "install",
    "docker", "podman", "kubectl", "helm", "terraform", "aws", "gcloud", "az",
    "npm", "npx", "pnpm", "yarn", "corepack", "uv", "uvx", "pip", "pip3",
    "poetry", "pipx", "cargo", "rustc", "go", "gofmt", "java", "javac",
    "ruby", "gem", "bundle", "rake", "perl", "php", "swift", "swiftc",
    "less", "more", "vi", "vim", "nvim", "emacs", "nano", "pager", "man",
    "tmux", "screen", "watch", "top", "htop", "ps", "kill", "pkill", "pgrep",
    "open", "say", "osascript", "pbcopy", "pbpaste", "defaults", "codesign",
})


def _known_not_claude(executable: str) -> bool:
    """Whether this executable is POSITIVELY some program other than the CLI.

    The one question the cheap prefilter is allowed to answer. It is a
    denylist on purpose: an executable this function has never seen returns
    False and is settled by the full argv and environment read, because "a
    name we do not recognise" is not "not claude". Recognising only a fixed
    set of claude-shaped names instead — the inverse of this — hid six of the
    eight real sessions running on this machine behind a version-named binary,
    and would hide any install under a name nobody thought of.

    A `claude` basename, a version-named file, a path with a `claude`
    component, and an interpreter that could be hosting the CLI all answer
    False however sealed the directory, so nothing below can rule out a real
    main by location alone.
    """
    if _could_be_claude(executable):
        return False
    base = os.path.basename(executable)
    if base in _NOT_CLAUDE_PROGRAMS:
        return True
    if base == "<defunct>":
        return True             # a zombie is not running anything
    return executable.startswith(_SEALED_DIRS)


def _could_be_claude(executable: str) -> bool:
    """Whether this executable's NAME leaves it able to be a Claude main.

    True for the CLI itself, a version-named file, anything under a path with
    a `claude` component, and any interpreter, so none of the location rules
    below can rule out a real main by where it happens to live.
    """
    base = os.path.basename(executable)
    return (_is_claude_binary(executable)
            or _INTERPRETER.match(base) is not None
            or _VERSION_NAME.match(base) is not None
            or _CLAUDE_PATH_PART.search(os.path.dirname(executable)) is not None)


def _unlikely_location(executable: str) -> bool:
    """Whether this executable sits where a Claude main is improbable.

    Weaker than :func:`_known_not_claude` and never enough on its own. Both
    `/Library` and the inside of an app bundle are writable by whoever
    installed the software there, so a main can be running from one. The
    caller must have read argv and found it not claude-shaped before acting
    on this.
    """
    if _could_be_claude(executable):
        return False
    return (executable.startswith(_THIRD_PARTY_DIRS)
            or _BUNDLE_EXECUTABLE.search(executable) is not None)


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


def _walk_flattened(argv: str, resolve_cwd=None) -> tuple[str, bool]:
    """One pass over a flattened argv line: ``(verdict, guessed)``.

    ``guessed`` records that the walk consumed an option's value by taking one
    whitespace-separated word. A value holding a space leaves its tail behind,
    so from that point on every token's position is a guess and no NEGATIVE
    verdict from this walk can be trusted. The caller acts on that; keeping it
    out of here is what stops a new early exit from forgetting to.
    """
    guessed = False
    tokens = argv.split()
    if not tokens:
        return ARGV_OTHER, guessed
    if _UNREADABLE_FIELD.match(tokens[0]):
        return ARGV_UNKNOWN, guessed    # `(claude)`: nothing was read
    if _is_claude_binary(tokens[0]):
        return ARGV_CLAUDE, guessed
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
        return ARGV_OTHER, guessed
    table = _FLAG_TABLES[family]
    after = _skip_subcommand(tokens, 1, family)
    rest = argv.split(None, after)[after] if len(tokens) > after else ""
    while rest.startswith("-"):
        parts = rest.split(None, 1)
        flag, remainder = parts[0], (parts[1] if len(parts) > 1 else "")
        if flag == "-":
            # The program comes from stdin, so there is no script path and no
            # claude. Every interpreter here spells it the same way, and it is
            # not an option, so the unknown-option rule must not catch it.
            return ARGV_OTHER, guessed
        if flag == "--":
            rest = remainder
            break                   # end of options; the script is next
        base = flag.split("=", 1)[0]
        if base in table.inline or (
                table.cluster is not None and table.cluster.match(base)):
            # Inline code, so there is no script argument at all.
            return ARGV_OTHER, guessed
        if "=" in flag:
            pass                    # self-contained, consumes nothing
        elif base in table.value:
            after = remainder.split(None, 1)
            remainder = after[1] if len(after) > 1 else ""
            guessed = True
        elif base not in table.boolean:
            # An option this module has never seen. Assuming it takes no value
            # reads a real main's script argument as an option's value;
            # assuming it takes one swallows the script. Neither is a guess
            # worth making under a credential rewrite.
            return ARGV_UNKNOWN, guessed
        rest = remainder
    script = rest
    if not script:
        return ARGV_OTHER, guessed
    first = script.split(None, 1)[0]
    resolved = _resolve_script(first, resolve_cwd)
    if resolved is None:
        # Relative, and the working directory is hidden, so this token names
        # either the npm entrypoint or an unrelated worker.
        return ARGV_UNKNOWN, guessed
    if _is_claude_binary(resolved) or _CLAUDE_PACKAGE.search(resolved):
        return ARGV_CLAUDE, guessed
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
        return ARGV_CLAUDE, guessed
    if complete:
        return ARGV_OTHER, guessed
    # Otherwise the script path may simply contain spaces (`node
    # "/Users/me/Application Support/claude"`), where splitting yields
    # `/Users/me/Application` and rejects a live main. Only the path's tail is
    # needed to recognise it, and the tail has no space in it.
    return ((ARGV_CLAUDE if _SCRIPT_TAIL.search(script) else ARGV_OTHER),
            guessed)


def _classify_argv(argv: str, resolve_cwd=None) -> str:
    """Classify a FLATTENED argv line: claude, other, or unknown.

    `ps` joins arguments with spaces and quotes nothing, so this cannot always
    tell one argument from two. Where it cannot, it says so. A Bash-tool shell
    inherits CLAUDE_CONFIG_DIR from the claude that spawned it and can outlive
    it as an orphan (64 such strays against one profile here, versus 7 real
    mains), so binding on the variable alone would hold a profile
    un-quiescent forever. The program is the honest test.

    Every negative verdict passes through the one check below. Once the walk
    has consumed an option value it cannot place the tokens after it, so a
    "not claude" from that point is a guess and becomes unknown — including
    from the walk's early exits, which used to answer "not claude" outright
    and so let `node -r "hook -e" /opt/bin/claude` read as inline code.
    """
    verdict, guessed = _walk_flattened(argv, resolve_cwd)
    return ARGV_UNKNOWN if guessed and verdict == ARGV_OTHER else verdict


class ProcArgs(NamedTuple):
    """A process's argv and environment, as the kernel holds them."""

    argv: list[str]
    env: dict[str, str] | None


def _resolve_script(script: str, resolve_cwd) -> str | None:
    """An absolute path for a script argument, or None when it cannot be had.

    `node cli.js` is the npm entrypoint or an unrelated worker depending
    entirely on where it is running, and the argument alone does not say. The
    working directory does, so a relative script is resolved against it; a
    working directory that cannot be read leaves the argument unresolvable
    rather than assumed harmless.
    """
    if os.path.isabs(script):
        return script
    cwd = resolve_cwd() if resolve_cwd is not None else None
    return None if cwd is None else os.path.normpath(os.path.join(cwd, script))


def _skip_subcommand(argv: list[str], i: int, family: str) -> int:
    """Step past `run`/`exec`/`x`, which bun and deno take before the script."""
    if (family in ("bun", "deno") and i < len(argv)
            and argv[i] in _INTERPRETER_SUBCOMMANDS):
        return i + 1
    return i


def _classify_argv_tokens(argv: list[str], resolve_cwd=None) -> str:
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
    if _UNREADABLE_FIELD.match(argv[0]):
        return ARGV_UNKNOWN     # `(claude)`: the argument area was not read
    if _is_claude_binary(argv[0]):
        return ARGV_CLAUDE
    family = _interpreter_family(argv[0])
    if family is None:
        return ARGV_OTHER
    table = _FLAG_TABLES[family]
    i = _skip_subcommand(argv, 1, family)
    while i < len(argv) and argv[i].startswith("-"):
        flag = argv[i]
        if flag == "-":
            return ARGV_OTHER       # the program comes from stdin
        if flag == "--":
            i += 1
            break                   # end of options; the script is next
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
    script = _resolve_script(argv[i], resolve_cwd)
    if script is None:
        # Relative, and the working directory is hidden, so this token names
        # either the npm entrypoint or an unrelated worker. `node cli.js` run
        # from the package directory is exactly how a main looks.
        return ARGV_UNKNOWN
    if _is_claude_binary(script) or _CLAUDE_PACKAGE.search(script):
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

    False whenever the read failed or the program is one this module does not
    recognise, so neither "we could not look" nor "we have never seen this
    name" shortens the work. On macOS the read is ``proc_pidpath``; on Linux
    it is the world-readable ``cmdline``, which also keeps the environment
    from being opened for a process already settled.
    """
    if sys.platform == "darwin":
        executable = _exec_path_macos(pid)
        if executable is None:
            return False
        if _known_not_claude(executable):
            return True
        if not _unlikely_location(executable):
            return False
        # Third-party software under `/Library`, or a bundle executable.
        # Anyone can install there, so the location alone cannot settle it:
        # `/Library/Acme/assistant` running `claude --resume` is a real main
        # and ruling it out unread reported its live profile as idle.
        info = _read_proc_args(pid)
        if info is None or not info.argv:
            return False
        return _classify_argv_tokens(
            info.argv, lambda: _proc_cwd(pid)) == ARGV_OTHER
    if sys.platform.startswith("linux"):
        try:
            raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError:
            return False
        argv = [c.decode("utf-8", "replace") for c in raw.split(b"\0") if c]
        if not argv:
            return False
        if _known_not_claude(argv[0]):
            return True
        if not _unlikely_location(argv[0]):
            return False
        return _classify_argv_tokens(
            argv, lambda: _proc_cwd(pid)) == ARGV_OTHER
    return False


_PROC_PIDVNODEPATHINFO = 9
# `struct proc_vnodepathinfo` is two `vnode_info_path` records, current
# directory first. The path sits after the 152-byte `vnode_info` header and
# runs to MAXPATHLEN.
_VNODEPATHINFO_SIZE = 2352
_VNODEPATHINFO_CWD = 152
_MAXPATHLEN = 1024


def _proc_uid(pid: int) -> int | None:
    """The EFFECTIVE uid owning a process, or None when it cannot be read.

    Raises ``ProcessLookupError`` when the pid is gone. A vanished process and
    an unreadable one are different answers: the first is settled, the second
    is not, and folding them together reported a pid that exited mid-scan as
    something this probe failed to read.

    macOS has no equivalent read here, so this answers None there and the
    ownership rule falls to the `ps` probe, which reports the effective uid
    for every pid in one snapshot.
    """
    if sys.platform.startswith("linux"):
        try:
            return os.stat(f"/proc/{pid}").st_uid
        except (FileNotFoundError, ProcessLookupError):
            raise ProcessLookupError(pid) from None
        except OSError:
            return None
    return None


def _proc_cwd(pid: int) -> str | None:
    """A process's working directory, or None when it cannot be read.

    Needed only to resolve a RELATIVE script argument: `node cli.js` names a
    real entrypoint or a worker depending entirely on where it is running, and
    without this the two are indistinguishable.
    """
    if sys.platform.startswith("linux"):
        try:
            return os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            return None
    if sys.platform != "darwin":
        return None
    libc = _load_libc()
    if libc is None or not hasattr(libc, "proc_pidinfo"):
        return None
    buf = ctypes.create_string_buffer(_VNODEPATHINFO_SIZE)
    written = libc.proc_pidinfo(pid, _PROC_PIDVNODEPATHINFO, ctypes.c_uint64(0),
                                buf, _VNODEPATHINFO_SIZE)
    if written < _VNODEPATHINFO_SIZE:
        return None
    path = buf.raw[_VNODEPATHINFO_CWD:_VNODEPATHINFO_CWD + _MAXPATHLEN]
    cwd = path.split(b"\0", 1)[0].decode("utf-8", "replace")
    return cwd or None


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
        self.combined = _pid_map(_ENV_PROBE_ARGV, "CLAUDE_CONFIG_DIR probe")
        self.argvs = _pid_map(_ARGV_PROBE_ARGV, "argv probe")
        self.executables = _pid_map(_COMM_PROBE_ARGV, "executable probe")
        self.uids = _pid_map(_UID_PROBE_ARGV, "owner probe")

    def _is_ours(self, pid: int) -> bool | None:
        """Whether this process is owned by the user running this tool."""
        if self.uids is None:
            return None
        owner = self.uids.get(pid)
        try:
            # macOS `ps uid=` prints the EFFECTIVE uid, so compare against the
            # same thing rather than the real one.
            return int(owner) == os.geteuid()
        except (TypeError, ValueError):
            return None

    def _executable(self, pid: int) -> str | None:
        """This pid's executable path, or None when `ps` could not read it.

        Apple's `ps` prints the kernel's short process name in PARENTHESES
        when it cannot read the full argument area — `(claude)`, `(node)` —
        and that is a read failure wearing the shape of an answer. Taking
        `(claude)` for a program name ruled the process out as "not claude",
        which is the exact inversion of what the marker means.
        """
        if self.executables is None:
            return None
        value = self.executables.get(pid)
        if value is None or (value.startswith("(") and value.endswith(")")):
            return None
        return value

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

    def _unread(self, pid: int, verdict: str) -> str:
        """The answer for a pid whose ENVIRONMENT could not be read.

        Ownership is consulted only at this point, never before. A reading
        that succeeded always outranks it: a process whose argv and
        environment were read and name a main is a main whoever owns it, and
        one that was read and names something else is unbound on that reading
        alone. Asking about the owner first rejected a readable, matching
        claude argv on the strength of its uid.

        Where nothing could be read, ownership is the last reading left. A
        `claude` bound to a profile this tool manages runs as this user, and
        another user's environment is precisely what `ps` will not show, so
        without this the machine's root-owned daemons deferred forever. A
        claude-shaped argv is never ruled out this way, and an owner that
        could not be read rules out nothing.
        """
        if verdict != ARGV_CLAUDE and self._is_ours(pid) is False:
            return _UNBOUND
        return _UNKNOWN

    def verdict(self, pid: int, target: str) -> str:
        """``bound``, ``unbound``, or ``unknown`` for one pid."""
        executable = self._executable(pid)
        argv = self._argv_text(pid)
        recognised = ((executable is not None and _known_not_claude(executable))
                      or (argv is not None and bool(argv.split())
                          and _known_not_claude(argv.split()[0])))
        if executable is not None and _known_not_claude(executable):
            # Positively read, and recognised as some other program. Most of
            # the process table lands here; failing closed on all of it would
            # wedge every profile on a machine with other users.
            return _UNBOUND
        if argv is None:
            if not is_pid_alive(pid):
                # It exited between the pid listing and the `ps` snapshot. A
                # process that no longer exists is not a live claude, and
                # calling it unknown would wedge every profile on this machine
                # for as long as short-lived processes keep starting.
                return _UNBOUND
            # Nothing was read. Ownership is the only reading left, and it is
            # exactly here that it belongs: see `_not_ours` below.
            return _UNBOUND if self._is_ours(pid) is False else _UNKNOWN
        verdict = _classify_argv(argv, lambda: _proc_cwd(pid))
        if verdict == ARGV_OTHER and executable is not None:
            # `comm` reports the executable as a single field, so it survives
            # a path with a space in it that the flattened argv does not. Its
            # reading outranks argv's: naming the CLI itself settles the pid as
            # a main, and naming anything else unrecognised leaves argv's "no"
            # too weak to settle anything. That is how a native main under
            # `/Users/me/Application Support/claude` came to read as idle.
            verdict = (ARGV_CLAUDE if _is_claude_binary(executable)
                       else ARGV_UNKNOWN)
        if verdict == ARGV_OTHER and recognised:
            return _UNBOUND
        if verdict == ARGV_OTHER:
            # argv named a program nobody listed, so it may be a main under a
            # name this module has never seen. Only the environment can settle
            # it, and only against this profile.
            verdict = ARGV_UNKNOWN
        if self.combined is None or pid not in self.combined:
            # Either a main, or a pid argv could not place; no environment to
            # settle it against either way.
            return self._unread(pid, verdict)
        _, env = _split_argv_env(
            self.combined[pid],
            None if self.argvs is None else self.argvs.get(pid),
        )
        if not env:
            # argv with no environment after it: macOS withholds the
            # environment of another user's process and of the platform
            # binaries it protects.
            return self._unread(pid, verdict)
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
        if value != target:
            # The environment settles it whatever the program turned out to
            # be: a process bound to some other profile, or to none, is not
            # bound to this one.
            return _UNBOUND
        # Bound to this profile. Report it as a main only if argv actually
        # placed it as one; otherwise it may be a stray that merely inherited
        # the variable, which is non-quiescent all the same but does not
        # belong in a list of mains.
        return _BOUND if verdict == ARGV_CLAUDE else _UNKNOWN


def _scan_pid(pid: int, target: str) -> str:
    """One pid's relationship to ``target``, from the structured source.

    Three readings combine. The program settles a pid only when it is
    RECOGNISED: the CLI, or a program this module positively knows is not it.
    An unrecognised program could be a main under a name nobody listed, so it
    cannot answer. The environment settles the rest, and it is conclusive in
    one direction whatever the program turned out to be — a process bound to
    another profile, or to none, is not bound to this one. What is left is an
    unrecognised program bound to THIS profile, which is genuinely unknown: it
    may be a main, or a tool subprocess that inherited the variable.
    """
    if _cannot_host_claude(pid):
        return _UNBOUND
    info = _read_proc_args(pid)
    if info is None:
        return _GONE if not is_pid_alive(pid) else _UNREADABLE
    verdict = _classify_argv_tokens(info.argv, lambda: _proc_cwd(pid))
    recognised = bool(info.argv) and _known_not_claude(info.argv[0])
    if info.env is None:
        # Linux shows every process's cmdline and withholds other users'
        # environ, so argv is the only reading available.
        if verdict == ARGV_OTHER and recognised:
            return _UNBOUND
        if verdict != ARGV_CLAUDE:
            # Not claude-shaped, and the environment that would settle it is
            # withheld. Ownership is the last reading left, and only here: a
            # `claude` bound to a profile this tool manages runs as this user,
            # so someone else's process is not the session about to have its
            # credentials rewritten. Without this the machine's root-owned
            # daemons, whose environment nothing can read, deferred forever
            # and no profile was ever quiescent.
            try:
                owner = _proc_uid(pid)
            except ProcessLookupError:
                return _GONE
            if owner is not None and owner != os.geteuid():
                return _UNBOUND
        logger.debug("pid %s could not be settled by argv alone and its "
                     "environment is not readable", pid)
        return _UNKNOWN
    if info.env.get("CLAUDE_CONFIG_DIR") != target:
        return _UNBOUND
    if verdict == ARGV_CLAUDE:
        return _BOUND
    if verdict == ARGV_OTHER and recognised:
        # A shell or tool that inherited the variable from the session that
        # started it. These outnumber real mains roughly nine to one and
        # outlive them as orphans, so binding on the variable alone would hold
        # the profile un-quiescent forever.
        return _UNBOUND
    return _UNKNOWN


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
