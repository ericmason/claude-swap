"""Tests for Claude Code process detection."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from claude_swap.process_detection import (
    PID_REUSE_SLACK_S,
    ClaudeSession,
    ProcArgs,
    ARGV_CLAUDE,
    ARGV_OTHER,
    ARGV_UNKNOWN,
    _classify_argv,
    _classify_argv_tokens,
    scan_env_bound_claude,
    IdeInstance,
    _lstart_seconds,
    _stat_start_ticks,
    get_claude_dir,
    get_running_instances,
    is_pid_alive,
    list_ide_instances,
    list_sessions,
    pid_matches_record,
    process_is_claude,
    process_start_ticks,
    process_started_at,
)
from claude_swap.session import profile_is_quiescent
from claude_swap.printer import abbreviate_path, entrypoint_label, format_age


# --- get_claude_dir ---


class TestGetClaudeDir:
    def test_default_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
            result = get_claude_dir()
            assert result == Path.home() / ".claude"

    def test_respects_env_var(self, tmp_path):
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(tmp_path)}):
            assert get_claude_dir() == tmp_path


# --- is_pid_alive ---


class TestIsPidAlive:
    # The os.kill(pid, 0) semantics below are the POSIX branch; on Windows
    # is_pid_alive() dispatches to _is_pid_alive_windows() and never calls
    # os.kill. Pin the platform so these exercise the intended path on any host.
    def test_alive_pid(self):
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("os.kill") as mock_kill:
            mock_kill.return_value = None
            assert is_pid_alive(12345) is True
            mock_kill.assert_called_once_with(12345, 0)

    def test_dead_pid(self):
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("os.kill", side_effect=OSError("No such process")):
            assert is_pid_alive(12345) is False

    def test_permission_error_means_alive(self):
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("os.kill", side_effect=PermissionError("Operation not permitted")):
            assert is_pid_alive(12345) is True

    def test_windows_dispatches_to_ctypes_impl(self):
        """On win32, is_pid_alive() delegates to the ctypes-based helper."""
        with patch("claude_swap.process_detection.sys.platform", "win32"), \
             patch(
                 "claude_swap.process_detection._is_pid_alive_windows",
                 return_value=True,
             ) as mock_win:
            assert is_pid_alive(12345) is True
            mock_win.assert_called_once_with(12345)

    def test_current_process_is_alive(self):
        """Smoke test on the real platform branch (win32 or POSIX)."""
        assert is_pid_alive(os.getpid()) is True

    def test_invalid_pid_zero(self):
        assert is_pid_alive(0) is False

    def test_invalid_pid_one(self):
        assert is_pid_alive(1) is False

    def test_negative_pid(self):
        assert is_pid_alive(-1) is False


# --- process start time / pid reuse ---


LSTART = "Wed Sep  2 20:35:59 2026"
LSTART_S = 1788381359
TICKS = "11485"
# /proc/<pid>/stat: pid, (comm), then the numeric fields; start time is 22nd.
STAT = (
    "4242 (claude) S 1 4242 4242 0 -1 4194560 1234 0 0 0 10 5 0 0 20 0 8 0 "
    f"{TICKS} 123456789 4321 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 0 "
    "17 3 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
)


class TestLstartSeconds:
    @pytest.mark.parametrize(
        "text, expected",
        [
            (LSTART, LSTART_S),
            ("Thu Jan  1 00:00:00 1970", 0),
            ("Tue Nov 14 22:13:20 2023", 1_700_000_000),
        ],
    )
    def test_parses_ps_lstart(self, text, expected):
        assert _lstart_seconds(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["", "abc", "Wed Sep 2 20:35 2026", "Wed Foo  2 20:35:59 2026",
         "Wed Sep  2 20:35:59", "Wed Sep  2 20:35:xx 2026"],
    )
    def test_rejects_garbage(self, text):
        with pytest.raises(ValueError):
            _lstart_seconds(text)


class TestStatStartTicks:
    def test_reads_field_22(self):
        assert _stat_start_ticks(STAT) == TICKS

    def test_counts_from_the_last_parenthesis(self):
        """The command name may contain spaces and parentheses of its own."""
        assert _stat_start_ticks(STAT.replace("(claude)", "(my (odd) name)")) == TICKS

    @pytest.mark.parametrize(
        "text", ["", "4242 (claude) S 1 2 3", STAT.replace(TICKS, "x")]
    )
    def test_rejects_garbage(self, text):
        assert _stat_start_ticks(text) is None


class TestProcessStartTicks:
    def test_unreadable_is_unknowable(self):
        with patch.object(Path, "read_text", side_effect=OSError("no /proc")):
            assert process_start_ticks(4242) is None

    def test_reads_proc_stat(self):
        with patch.object(Path, "read_text", autospec=True, return_value=STAT) as read:
            assert process_start_ticks(4242) == TICKS
        assert read.call_args[0][0] == Path("/proc/4242/stat")

    @pytest.mark.skipif(sys.platform != "linux", reason="/proc is Linux")
    def test_own_process_matches_proc_self(self):
        expected = _stat_start_ticks(Path("/proc/self/stat").read_text())
        assert process_start_ticks(os.getpid()) == expected


class TestProcessStartedAt:
    def test_windows_is_unknowable(self):
        with patch("claude_swap.process_detection.sys.platform", "win32"), \
             patch("claude_swap.process_detection.subprocess.run") as run:
            assert process_started_at(1234) is None
        run.assert_not_called()

    def test_ps_failure_is_unknowable(self):
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("claude_swap.process_detection.subprocess.run",
                   side_effect=OSError("no ps")):
            assert process_started_at(1234) is None

    def test_unknown_pid_is_unknowable(self):
        proc = subprocess_result(returncode=1, stdout="")
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("claude_swap.process_detection.subprocess.run", return_value=proc):
            assert process_started_at(1234) is None

    def test_garbage_is_unknowable(self):
        proc = subprocess_result(returncode=0, stdout="??\n")
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("claude_swap.process_detection.subprocess.run", return_value=proc):
            assert process_started_at(1234) is None

    def test_reads_lstart_the_way_claude_does(self):
        """The record's ``procStart`` is claude's own ``LC_ALL=C TZ=UTC ps -o
        lstart=`` output; the live reading must match it to the second."""
        proc = subprocess_result(returncode=0, stdout=f"{LSTART}    \n")
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("claude_swap.process_detection.subprocess.run",
                   return_value=proc) as run:
            assert process_started_at(1234) == LSTART_S
        args, kwargs = run.call_args
        assert args[0] == ["ps", "-o", "lstart=", "-p", "1234"]
        assert kwargs["env"]["LC_ALL"] == "C"
        assert kwargs["env"]["TZ"] == "UTC"

    @pytest.mark.skipif(os.name != "posix", reason="ps is POSIX")
    def test_own_process_started_in_the_past(self):
        started = process_started_at(os.getpid())
        assert started is not None
        assert started <= time.time()


def subprocess_result(returncode: int, stdout: str):
    from subprocess import CompletedProcess

    return CompletedProcess(["ps"], returncode, stdout=stdout, stderr="")


class TestProcessIsClaude:
    def test_ps_failure_is_unknowable(self):
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("claude_swap.process_detection.subprocess.run",
                   side_effect=OSError("no ps")):
            assert process_is_claude(1234) is None

    @pytest.mark.parametrize(
        "line, expected",
        [
            ("claude           claude --resume 2d6cbe5d", True),
            ("node             node /usr/lib/node_modules/@anthropic-ai/claude-code/cli.js", True),
            ("2.1.258          /home/u/.local/share/claude/versions/2.1.258", True),
            ("vim              vim notes.md", False),
        ],
    )
    def test_judges_comm_and_args(self, line, expected):
        proc = subprocess_result(returncode=0, stdout=f"{line}\n")
        with patch("claude_swap.process_detection.sys.platform", "linux"), \
             patch("claude_swap.process_detection.subprocess.run",
                   return_value=proc) as run:
            assert process_is_claude(1234) is expected
        assert run.call_args[0][0] == ["ps", "-o", "comm=,args=", "-p", "1234"]


class TestPidMatchesRecord:
    @pytest.mark.parametrize("proc_start", [None, "", "garbage"])
    def test_unstamped_or_garbage_record_passes(self, proc_start):
        with patch("claude_swap.process_detection.process_started_at") as started:
            assert pid_matches_record(1234, proc_start) is True
        started.assert_not_called()

    def test_unknowable_start_passes(self):
        with patch("claude_swap.process_detection.process_started_at",
                   return_value=None), \
             patch("claude_swap.process_detection.process_is_claude") as is_claude:
            assert pid_matches_record(1234, LSTART) is True
        is_claude.assert_not_called()

    def test_same_start_ticks_is_the_recorded_process(self):
        """A Linux record stamps the /proc start time in ticks since boot,
        which the process keeps for life: equality is the whole test."""
        with patch("claude_swap.process_detection.process_start_ticks",
                   return_value=TICKS), \
             patch("claude_swap.process_detection.process_started_at") as started:
            assert pid_matches_record(1234, TICKS) is True
        started.assert_not_called()

    def test_other_start_ticks_is_a_recycled_pid(self):
        with patch("claude_swap.process_detection.process_start_ticks",
                   return_value="998877"), \
             patch("claude_swap.process_detection.process_is_claude") as is_claude:
            assert pid_matches_record(1234, TICKS) is False
        is_claude.assert_not_called()

    def test_unreadable_ticks_pass(self):
        """A FILETIME on Windows, or /proc hidden: no comparison is possible."""
        with patch("claude_swap.process_detection.process_start_ticks",
                   return_value=None):
            assert pid_matches_record(1234, "134332352612628209") is True

    @pytest.mark.parametrize(
        "started", [LSTART_S, LSTART_S - 3600, LSTART_S + PID_REUSE_SLACK_S // 2]
    )
    def test_process_not_younger_than_record_passes(self, started):
        with patch("claude_swap.process_detection.process_started_at",
                   return_value=started), \
             patch("claude_swap.process_detection.process_is_claude") as is_claude:
            assert pid_matches_record(1234, LSTART) is True
        is_claude.assert_not_called()

    def test_stranger_younger_than_record_is_a_recycled_pid(self):
        with patch("claude_swap.process_detection.process_started_at",
                   return_value=LSTART_S + 86400), \
             patch("claude_swap.process_detection.process_is_claude",
                   return_value=False):
            assert pid_matches_record(1234, LSTART) is False

    def test_claude_younger_than_record_is_kept(self):
        """Linux ps derives start times from a boot time that moves with the
        wall clock, so after a clock step a live session reads as younger
        than its own record. A claude at the pid is never a recycled pid."""
        with patch("claude_swap.process_detection.process_started_at",
                   return_value=LSTART_S + 86400), \
             patch("claude_swap.process_detection.process_is_claude",
                   return_value=True):
            assert pid_matches_record(1234, LSTART) is True

    def test_unknowable_identity_is_kept(self):
        with patch("claude_swap.process_detection.process_started_at",
                   return_value=LSTART_S + 86400), \
             patch("claude_swap.process_detection.process_is_claude",
                   return_value=None):
            assert pid_matches_record(1234, LSTART) is True


# --- list_sessions ---


def _write_session(sessions_dir: Path, pid: int, **overrides) -> Path:
    """Write a session PID file with sensible defaults."""
    data = {
        "pid": pid,
        "sessionId": f"session-{pid}",
        "cwd": "/home/user/project",
        "startedAt": int(time.time() * 1000),
        "kind": "interactive",
        "entrypoint": "cli",
    }
    data.update(overrides)
    path = sessions_dir / f"{pid}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestListSessions:
    def test_reads_valid_sessions(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001, entrypoint="cli", cwd="/home/user/app")
        _write_session(sessions_dir, 1002, entrypoint="claude-vscode", cwd="/home/user/web")

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_sessions(tmp_path)

        assert len(result) == 2
        pids = {s.pid for s in result}
        assert pids == {1001, 1002}

    def test_filters_dead_pids(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001)
        _write_session(sessions_dir, 1002)

        def alive(pid):
            return pid == 1001

        with patch("claude_swap.process_detection.is_pid_alive", side_effect=alive):
            result = list_sessions(tmp_path)

        assert len(result) == 1
        assert result[0].pid == 1001

    def test_filters_recycled_pids(self, tmp_path):
        """A crashed claude's record survives it; the OS may hand its pid to
        something else. Only a process that started when the record says its
        claude did is the recorded one."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001, procStart=LSTART)
        _write_session(sessions_dir, 1002, procStart=LSTART)

        def started(pid):
            return LSTART_S if pid == 1002 else LSTART_S + 86400

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True), \
             patch("claude_swap.process_detection.process_started_at",
                   side_effect=started), \
             patch("claude_swap.process_detection.process_is_claude",
                   return_value=False):
            result = list_sessions(tmp_path)

        assert [s.pid for s in result] == [1002]

    def test_filters_recycled_pids_by_start_ticks(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001, procStart=TICKS)
        _write_session(sessions_dir, 1002, procStart=TICKS)

        def ticks(pid):
            return TICKS if pid == 1002 else "998877"

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True), \
             patch("claude_swap.process_detection.process_start_ticks",
                   side_effect=ticks):
            result = list_sessions(tmp_path)

        assert [s.pid for s in result] == [1002]

    def test_unknowable_start_time_keeps_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001, procStart=LSTART)

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True), \
             patch("claude_swap.process_detection.process_started_at",
                   return_value=None):
            result = list_sessions(tmp_path)

        assert [s.pid for s in result] == [1001]

    def test_missing_sessions_dir(self, tmp_path):
        assert list_sessions(tmp_path) == []

    @pytest.mark.parametrize(
        "write_bad_file",
        [
            lambda p: p.write_text("not json{{{", encoding="utf-8"),
            lambda p: p.write_bytes(b"\xff\xfe{\"pid\": 1}"),
            lambda p: p.write_text(json.dumps({"pid": 2**31}), encoding="utf-8"),
            lambda p: p.write_text("[" * 2000 + "]" * 2000, encoding="utf-8"),
            lambda p: p.write_text("[]", encoding="utf-8"),
        ],
        ids=["invalid_json", "invalid_utf8", "pid_overflows_c_long",
             "json_nested_too_deep", "json_is_an_array"],
    )
    def test_corrupt_session_file_is_skipped(self, tmp_path, write_bad_file):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        write_bad_file(sessions_dir / "9999.json")

        result = list_sessions(tmp_path)
        assert result == []

    def test_missing_pid_field(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "9999.json").write_text(
            json.dumps({"sessionId": "abc", "cwd": "/tmp"}), encoding="utf-8"
        )

        result = list_sessions(tmp_path)
        assert result == []

    def test_optional_status_field(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001, status="busy")

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_sessions(tmp_path)

        assert result[0].status == "busy"

    def test_status_absent(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001)

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_sessions(tmp_path)

        assert result[0].status is None

    def test_session_fields(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(
            sessions_dir, 5000,
            sessionId="sess-abc",
            cwd="/projects/foo",
            startedAt=1700000000000,
            kind="bg",
            entrypoint="claude-desktop",
        )

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_sessions(tmp_path)

        s = result[0]
        assert s.pid == 5000
        assert s.session_id == "sess-abc"
        assert s.cwd == "/projects/foo"
        assert s.started_at == 1700000000000
        assert s.kind == "bg"
        assert s.entrypoint == "claude-desktop"


# --- list_ide_instances ---


def _write_ide_lock(ide_dir: Path, port: int, **overrides) -> Path:
    """Write an IDE lockfile with sensible defaults."""
    data = {
        "pid": port + 1000,
        "workspaceFolders": ["/home/user/project"],
        "ideName": "Visual Studio Code",
        "transport": "ws",
    }
    data.update(overrides)
    path = ide_dir / f"{port}.lock"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestListIdeInstances:
    def test_reads_valid_lockfiles(self, tmp_path):
        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        _write_ide_lock(ide_dir, 45000, ideName="Visual Studio Code")
        _write_ide_lock(ide_dir, 45001, ideName="Cursor")

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_ide_instances(tmp_path)

        assert len(result) == 2
        names = {i.ide_name for i in result}
        assert names == {"Visual Studio Code", "Cursor"}

    def test_filters_dead_pids(self, tmp_path):
        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        _write_ide_lock(ide_dir, 45000, pid=2001)
        _write_ide_lock(ide_dir, 45001, pid=2002)

        with patch("claude_swap.process_detection.is_pid_alive", side_effect=lambda p: p == 2001):
            result = list_ide_instances(tmp_path)

        assert len(result) == 1
        assert result[0].pid == 2001

    def test_missing_ide_dir(self, tmp_path):
        assert list_ide_instances(tmp_path) == []

    @pytest.mark.parametrize(
        "write_bad_file",
        [
            lambda p: p.write_text("broken", encoding="utf-8"),
            lambda p: p.write_text(json.dumps({"pid": 2**31}), encoding="utf-8"),
            lambda p: p.write_text("[" * 2000 + "]" * 2000, encoding="utf-8"),
            lambda p: p.write_text("[]", encoding="utf-8"),
        ],
        ids=["invalid_json", "pid_overflows_c_long", "json_nested_too_deep",
             "json_is_an_array"],
    )
    def test_corrupt_json(self, tmp_path, write_bad_file):
        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        write_bad_file(ide_dir / "9999.lock")

        assert list_ide_instances(tmp_path) == []

    def test_missing_pid_field(self, tmp_path):
        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        (ide_dir / "9999.lock").write_text(
            json.dumps({"ideName": "VS Code"}), encoding="utf-8"
        )

        assert list_ide_instances(tmp_path) == []

    def test_port_from_filename(self, tmp_path):
        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        _write_ide_lock(ide_dir, 12345)

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_ide_instances(tmp_path)

        assert result[0].port == 12345

    def test_workspace_folders(self, tmp_path):
        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        _write_ide_lock(ide_dir, 45000, workspaceFolders=["/a", "/b"])

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            result = list_ide_instances(tmp_path)

        assert result[0].workspace_folders == ["/a", "/b"]


# --- get_running_instances ---


class TestGetRunningInstances:
    def test_returns_both(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, 1001)

        ide_dir = tmp_path / "ide"
        ide_dir.mkdir()
        _write_ide_lock(ide_dir, 45000)

        with patch("claude_swap.process_detection.is_pid_alive", return_value=True):
            sessions, ides = get_running_instances(tmp_path)

        assert len(sessions) == 1
        assert len(ides) == 1

    def test_empty_when_no_dirs(self, tmp_path):
        sessions, ides = get_running_instances(tmp_path)
        assert sessions == []
        assert ides == []


# --- Display helpers (in switcher.py) ---


class TestEntrypointLabel:
    @pytest.mark.parametrize(
        "entrypoint,expected",
        [
            ("cli", "CLI"),
            ("claude-vscode", "VS Code"),
            ("claude-desktop", "Desktop"),
            ("sdk-cli", "SDK"),
            ("mcp", "MCP"),
            ("unknown-thing", "unknown-thing"),
        ],
    )
    def test_known_and_unknown(self, entrypoint, expected):
        assert entrypoint_label(entrypoint) == expected


class TestAbbreviatePath:
    def test_replaces_home(self):
        home = str(Path.home())
        assert abbreviate_path(f"{home}/projects/foo") == "~/projects/foo"

    def test_non_home_path_unchanged(self):
        assert abbreviate_path("/opt/data/bar") == "/opt/data/bar"

    def test_home_root(self):
        home = str(Path.home())
        assert abbreviate_path(home) == "~"


class TestFormatAge:
    def test_just_now(self):
        now_ms = int(time.time() * 1000)
        assert format_age(now_ms) == "just now"

    def test_minutes(self):
        ms = int((time.time() - 300) * 1000)  # 5 minutes ago
        assert format_age(ms) == "5m ago"

    def test_hours(self):
        ms = int((time.time() - 7200) * 1000)  # 2 hours ago
        assert format_age(ms) == "2h ago"

    def test_days(self):
        ms = int((time.time() - 172800) * 1000)  # 2 days ago
        assert format_age(ms) == "2d ago"


# --- scan_env_bound_claude: liveness the session registry cannot see ---


@contextlib.contextmanager
def fallback_only(*pids):
    """Force the `ps` path for ``pids``: no structured source answers for them.

    The scan reads /proc or KERN_PROCARGS2 first and only consults `ps` for
    pids those refuse, so a test that fabricates `ps` output has to say which
    pids exist and that nothing structured covers them.
    """
    with patch("claude_swap.process_detection._cannot_host_claude",
               return_value=False), \
         patch("claude_swap.process_detection._candidate_pids",
               return_value=list(pids)), \
         patch("claude_swap.process_detection._read_proc_args",
               return_value=None), \
         patch("claude_swap.process_detection.is_pid_alive", return_value=True):
        yield


@contextlib.contextmanager
def structured(**by_pid):
    """Answer the structured read for each pid, as the kernel would.

    ``by_pid`` maps a pid (as a keyword, so ``p4242=...``) to a ProcArgs or to
    None for a pid the kernel refuses.
    """
    table = {int(k.lstrip("p")): v for k, v in by_pid.items()}
    with patch("claude_swap.process_detection._cannot_host_claude",
               return_value=False), \
         patch("claude_swap.process_detection._candidate_pids",
               return_value=list(table)), \
         patch("claude_swap.process_detection._read_proc_args",
               side_effect=lambda pid: table[pid]), \
         patch("claude_swap.process_detection.is_pid_alive", return_value=True):
        yield


class TestIsClaudeArgv:
    """Only a claude MAIN counts; processes that merely inherited its env don't."""

    @pytest.mark.parametrize("argv", [
        "/Users/e/.local/bin/claude",
        "/Users/e/.local/bin/claude --resume abc",
        "node /opt/claude/claude --foo",
    ])
    def test_mains_are_claude(self, argv):
        assert _classify_argv(argv) == ARGV_CLAUDE

    @pytest.mark.parametrize("argv", [
        "node /usr/lib/node_modules/@anthropic-ai/claude-code/cli.js",
        "node --enable-source-maps /opt/nm/@anthropic-ai/claude-code/cli.js",
    ])
    def test_the_npm_distribution_is_a_main(self, argv):
        """Its basename is cli.js, so the complete-script-name rule would file
        a real main as a worker and let its credentials be rewritten."""
        assert _classify_argv(argv) == ARGV_CLAUDE

    @pytest.mark.parametrize("argv", [
        "node --require setup.js /opt/nm/@anthropic-ai/claude-code/cli.js",
        "node -r hook.js /opt/nm/@anthropic-ai/claude-code/cli.js",
        "node --require setup.js /opt/bin/claude",
    ])
    def test_an_option_that_consumes_a_value_does_not_hide_the_script(self, argv):
        """Skipping only the option leaves its value looking like the script,
        so a live main reads as a worker called setup.js."""
        assert _classify_argv(argv) == ARGV_CLAUDE

    def test_a_consumed_value_is_not_mistaken_for_claude(self):
        """Once a value has been consumed, the next token may be the tail of
        that value rather than the script, so the flattened line cannot say
        this is not a main. Answering "other" here let a live main whose
        `--require` value held a space have its credentials rewritten."""
        assert _classify_argv("node --require setup.js worker.js") == ARGV_UNKNOWN

    def test_a_lookalike_package_directory_is_not(self):
        assert _classify_argv("node /home/claude-code-notes/build.js") == ARGV_OTHER

    @pytest.mark.parametrize("argv", [
        "node /Users/me/Application Support/claude",
        "node --enable-source-maps /Users/me/Application Support/claude",
    ])
    def test_an_interpreter_script_path_may_contain_spaces(self, argv):
        """`comm` says `node`, so the executable probe cannot rescue this one;
        rejecting it would call a live profile idle."""
        assert _classify_argv(argv) == ARGV_CLAUDE

    def test_a_lookalike_script_is_not_claude(self):
        assert _classify_argv("node /opt/claude-helper") == ARGV_OTHER

    def test_a_worker_merely_passed_a_claude_path_is_not_a_main(self):
        """A complete script name ends the script argument, so what follows is
        claude's own argv. Such a child inherits CLAUDE_CONFIG_DIR and can
        outlive its parent — reading it as a main pins the profile forever."""
        argv = "node /srv/app/worker.mjs /tmp/claude"
        assert _classify_argv(argv) == ARGV_OTHER

    def test_a_relative_worker_is_read_against_its_working_directory(self):
        """`node worker.js` says nothing on its own: the same line names a
        different file from every directory. With the process's cwd it resolves
        and reads as not-a-main; without one it stays unknown."""
        argv = "node worker.js /tmp/claude"
        assert _classify_argv(argv, lambda: "/srv/app") == ARGV_OTHER
        assert _classify_argv(argv) == ARGV_UNKNOWN

    def test_an_interpreter_flag_does_not_hide_the_script(self):
        assert _classify_argv("node --enable-source-maps /opt/app/server.js") == ARGV_OTHER
        assert _classify_argv("node --enable-source-maps /opt/claude/claude") == ARGV_CLAUDE

    @pytest.mark.parametrize("argv", [
        "/bin/zsh -c source /tmp/snapshot-zsh",   # Bash-tool shell, can outlive its parent
        "/usr/bin/rg --files",
        "",
    ])
    def test_inherited_strays_are_not(self, argv):
        assert _classify_argv(argv) == ARGV_OTHER

    @pytest.mark.parametrize("argv", [
        "node --eval setInterval(()=>{},1e3) /opt/bin/claude",
        "node -e require('./x') /opt/nm/@anthropic-ai/claude-code/cli.js",
        "node --eval=setInterval(()=>{},1e3) /opt/bin/claude",
        "node --print process.version /opt/bin/claude",
    ])
    def test_inline_code_replaces_the_script_so_nothing_after_it_is_one(self, argv):
        """`--eval` and `--print` ARE the program, so every later token is argv
        for that one-liner. A daemon handed claude's path is not claude, and
        reading it as one pins the profile non-quiescent while it runs."""
        assert _classify_argv(argv) == ARGV_OTHER

    def test_inline_code_does_not_hide_a_script_that_comes_after_a_value_flag(self):
        """The two option shapes must stay apart: --require consumes a token
        and carries on, --eval ends the search."""
        assert _classify_argv(
            "node --require setup.js --enable-source-maps /opt/bin/claude") == ARGV_CLAUDE

    @pytest.mark.parametrize("argv", [
        "node -pe \"/opt/claude-code/cli.js\";setInterval(()=>{},60000)",
        "node -ep /opt/bin/claude",
        "node -ipe /opt/bin/claude",
        "node -e=setInterval(()=>{},1e3) /opt/bin/claude",
    ])
    def test_short_options_combine_into_one_cluster(self, argv):
        """Node takes `-pe` as one flag meaning print-and-eval. Matching the
        spellings `-e` and `-p` alone missed every cluster, so a daemon holding
        claude's path in its inline code read as a live main."""
        assert _classify_argv(argv) == ARGV_OTHER

    def test_a_cluster_with_neither_letter_still_consumes_its_value(self):
        assert _classify_argv("node -r hook.js /opt/bin/claude") == ARGV_CLAUDE

    @pytest.mark.parametrize("argv", [
        "node /Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js",
        "node --enable-source-maps "
        "/Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js",
    ])
    def test_the_npm_entrypoint_under_a_spacey_path_is_a_main(self, argv):
        """Splitting at the space made the script `/Users/me/Application`,
        which is not claude — so a live main read as an idle profile and its
        credentials were rewritten underneath it."""
        assert _classify_argv(argv) == ARGV_CLAUDE

    def test_the_package_directory_only_counts_at_the_entrypoint(self):
        """Matching `/claude-code/` anywhere in argv read a worker that was
        handed the entrypoint's path as the entrypoint itself."""
        assert _classify_argv(
            "node /srv/worker.js /opt/nm/@anthropic-ai/claude-code/cli.js") == ARGV_OTHER

    @pytest.mark.parametrize("argv", [
        "vim /usr/local/bin/claude",
        "tail -f /Users/e/.local/bin/claude",
        "cp /usr/local/bin/claude /tmp/claude",
    ])
    def test_a_command_merely_naming_the_binary_is_not_a_main(self, argv):
        """These run FROM a claude session, so they inherit CLAUDE_CONFIG_DIR
        and reach this test; treating them as mains would hold the profile
        un-quiescent for as long as the editor stayed open."""
        assert _classify_argv(argv) == ARGV_OTHER


class TestArgvThatArrivesAlreadySeparated:
    """The kernel hands over argv as records, so nothing has to be re-split."""

    @pytest.mark.parametrize("argv", [
        ["/usr/local/bin/claude", "--resume", "x"],
        ["node", "/Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js"],
        ["node", "--enable-source-maps", "/opt/my apps/bin/claude"],
        ["node", "--require", "setup.js", "/opt/bin/claude"],
    ])
    def test_a_main_is_recognised_however_spacey_its_path(self, argv):
        assert _classify_argv_tokens(argv) == ARGV_CLAUDE

    @pytest.mark.parametrize("argv", [
        ["node", "-pe", "require('/opt/claude-code/cli.js')"],
        ["node", "/srv/worker.js", "/opt/nm/@anthropic-ai/claude-code/cli.js"],
        ["/bin/zsh", "-c", "ls"],
        ["vim", "/usr/local/bin/claude"],
        [],
    ])
    def test_a_lookalike_is_still_not_one(self, argv):
        assert _classify_argv_tokens(argv) == ARGV_OTHER

    def test_an_argument_holding_an_assignment_cannot_move_the_binding(
        self, tmp_path,
    ):
        """`claude -p "set CLAUDE_CONFIG_DIR=/elsewhere"` is argv, and argv and
        the environment arrive as separate records here, so no amount of
        assignment-shaped text in one can be read as the other."""
        d = tmp_path / "1-acct"
        argv = ["/usr/bin/claude", "-p", f"set CLAUDE_CONFIG_DIR=/elsewhere"]
        with structured(p4242=ProcArgs(argv, {"CLAUDE_CONFIG_DIR": str(d)})):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_a_value_holding_an_assignment_is_still_that_value(self, tmp_path):
        d = tmp_path / "1-acct"
        env = {"CLAUDE_CONFIG_DIR": str(d), "NOTE": "CLAUDE_CONFIG_DIR=/elsewhere"}
        with structured(p4242=ProcArgs(["/usr/bin/claude"], env)):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_a_pid_the_kernel_shows_without_its_environment(self, tmp_path):
        """Linux shows every process's cmdline and withholds other users'
        environ. argv settles the ones that cannot be claude; a main whose
        environment is hidden is unknown."""
        d = tmp_path / "1-acct"
        with structured(p10=ProcArgs(["/usr/bin/rg", "--files"], None)):
            assert scan_env_bound_claude(d) == ([], True)
        with structured(p10=ProcArgs(["/usr/local/bin/claude"], None)):
            assert scan_env_bound_claude(d) == ([], False)


class TestTheKernelIsTheSource:
    """The structured readers, exercised against this very process."""

    def test_nul_records_keep_spaces_and_equals_signs(self):
        from claude_swap.process_detection import _split_nul_env

        env = _split_nul_env([b"LS_COLORS=a b c", b"EQ=x=y"])
        assert env == {"LS_COLORS": "a b c", "EQ": "x=y"}

    def test_an_empty_record_ends_the_environment(self):
        """The kernel writes the environment as one NUL-terminated run, so the
        first empty slot is its end. Skipping past it read whatever the region
        held before and attributed a stale binding to a live process."""
        from claude_swap.process_detection import _split_nul_env

        env = _split_nul_env([b"A=1", b"", b"CLAUDE_CONFIG_DIR=/stale"])
        assert env == {"A": "1"}

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS")
    def test_a_truncated_argument_area_is_unreadable_not_empty(self):
        """A buffer holding fewer records than argc means the copy was cut
        short. Taking the environment from what followed read argv fragments
        as variables, so the pid looked unbound instead of unreadable."""
        from claude_swap import process_detection as pd

        short = (5).to_bytes(4, sys.byteorder) + b"/bin/x\0\0argv0\0argv1\0"
        with patch.object(pd, "_sysctl", return_value=short), \
             patch.object(pd, "_macos_argmax", return_value=4096):
            assert pd._proc_args_macos(os.getpid()) is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS")
    def test_macos_reads_this_process_back(self):
        from claude_swap.process_detection import _proc_args_macos

        info = _proc_args_macos(os.getpid())
        assert info is not None, "our own process is always readable"
        assert info.argv and info.argv[0]
        assert info.env is not None and "PATH" in info.env

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS")
    def test_macos_refuses_a_pid_that_does_not_exist(self):
        from claude_swap.process_detection import _proc_args_macos

        assert _proc_args_macos(2 ** 30) is None

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc is Linux")
    def test_linux_reads_this_process_back(self):
        from claude_swap.process_detection import _proc_args_linux

        info = _proc_args_linux(os.getpid())
        assert info is not None
        assert info.env is not None and "PATH" in info.env

    def test_the_live_probe_answers_for_this_machine(self, tmp_path):
        """No mocks: whatever is running right now, an unused profile path is
        bound to nothing and the probe must be able to say so."""
        pids, probed = scan_env_bound_claude(tmp_path / "never-used")
        assert pids == []
        assert probed, "a machine this probe cannot read would wedge every profile"


class TestScanEnvBoundClaude:
    def _ps(self, stdout, returncode=0, argv_out="", comm_out="", pids=(),
            env_returncode=None, uid_out=None):
        """Fake the three `ps` probes separately; they ask different questions.

        Leaving argv_out empty exercises the heuristic fallback, which is what
        a pid the plain probe missed goes through.
        """
        def fake(argv, **kwargs):
            code = returncode
            if "ewwx" in argv:
                out = stdout
                code = returncode if env_returncode is None else env_returncode
            elif argv[-1].endswith("comm="):
                out = comm_out
            elif argv[-1].endswith("uid="):
                out = ("" if uid_out is None else uid_out)
            else:
                out = argv_out
            return SimpleNamespace(stdout=out, returncode=code)

        stack = contextlib.ExitStack()
        stack.enter_context(fallback_only(*pids))
        stack.enter_context(patch(
            "claude_swap.process_detection.subprocess.run", side_effect=fake))
        return stack

    def test_finds_a_claude_bound_to_this_profile(self, tmp_path):
        d = tmp_path / "1-acct"
        out = f"  4242 /usr/local/bin/claude --resume x CLAUDE_CONFIG_DIR={d} TERM=xterm\n"
        with self._ps(out, argv_out="  4242 /usr/local/bin/claude --resume x\n",
                      pids=(4242,)):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_ignores_a_process_that_only_inherited_the_var(self, tmp_path):
        d = tmp_path / "1-acct"
        out = f"  99 /bin/zsh -c ls CLAUDE_CONFIG_DIR={d}\n"
        with self._ps(out, argv_out="  99 /bin/zsh -c ls\n", pids=(99,)):
            assert scan_env_bound_claude(d) == ([], True)

    def test_a_longer_sibling_dir_does_not_match(self, tmp_path):
        """`.../1-eric` must not match `.../1-eric-old` — the match is right-anchored."""
        d = tmp_path / "1-eric"
        out = f"  7 /usr/bin/claude CLAUDE_CONFIG_DIR={d}-old\n"
        with self._ps(out, argv_out="  7 /usr/bin/claude\n", pids=(7,)):
            assert scan_env_bound_claude(d) == ([], True)

    def test_a_claude_with_no_visible_env_is_unknown_not_absent(self, tmp_path):
        """`ps` withholds the environment of another user's process. A main we
        cannot read is exactly the case that must not read as an idle
        profile."""
        d = tmp_path / "1-acct"
        with self._ps("  5 /usr/bin/claude\n",
                      argv_out="  5 /usr/bin/claude\n", pids=(5,)):
            assert scan_env_bound_claude(d) == ([], False)

    def test_a_broken_probe_reports_that_it_could_not_look(
        self, tmp_path, caplog,
    ):
        """An empty list alone reads as "nobody is there", and the caller
        rewrites credentials on it. A probe that raised saw nothing, so it
        must say so and let the caller fail closed."""
        with patch("claude_swap.process_detection._candidate_pids",
                   return_value=None):
            assert scan_env_bound_claude(tmp_path) == ([], False)
        assert "could not list processes" in caplog.text, \
            "a silent fallback would hide it"

    def test_a_claude_the_fallback_cannot_resolve_is_unknown(self, tmp_path):
        """The structured read refused AND `ps` exited nonzero, so nothing
        settled whether this main holds the profile."""
        with self._ps("", env_returncode=1, comm_out="  5 /usr/bin/claude\n",
                      pids=(5,)):
            assert scan_env_bound_claude(tmp_path) == ([], False)

    def test_a_process_that_cannot_be_claude_needs_no_environment(self, tmp_path):
        """Failing closed on every unreadable pid would wedge the profile: most
        of the process table belongs to other users. argv settles those."""
        with self._ps("", env_returncode=1, comm_out="  5 /usr/bin/rg\n",
                      argv_out="  5 /usr/bin/rg --files\n", pids=(5,)):
            assert scan_env_bound_claude(tmp_path) == ([], True)

    def test_windows_is_the_one_platform_that_still_fails_open(self, tmp_path):
        """There is no cheap environment read on Windows, so the probe is
        absent rather than broken. Failing closed there would leave every
        Windows profile permanently un-re-seedable."""
        with patch("claude_swap.process_detection.sys.platform", "win32"):
            assert scan_env_bound_claude(tmp_path) == ([], True)


class TestProfileIsQuiescentUsesBothSignals:
    """The regression: Claude Code's registry can miss a live instance."""

    def test_unregistered_live_claude_blocks_a_rewrite(self, tmp_path):
        d = tmp_path / "profile"
        (d / "sessions").mkdir(parents=True)          # registry: empty
        with structured(p4242=ProcArgs(
                ["/usr/local/bin/claude", "--resume", "x"],
                {"CLAUDE_CONFIG_DIR": str(d)})):
            assert profile_is_quiescent(d) is False

    def test_a_genuinely_idle_profile_is_still_quiescent(self, tmp_path):
        d = tmp_path / "profile"
        (d / "sessions").mkdir(parents=True)
        with structured(p4242=ProcArgs(
                ["/usr/local/bin/claude"], {"CLAUDE_CONFIG_DIR": "/elsewhere"})):
            assert profile_is_quiescent(d) is True

    def test_a_probe_that_could_not_run_is_not_quiescence(self, tmp_path):
        """The registry is exactly what misses an unregistered `claude
        --resume`, so a dead probe leaves liveness unknown. Calling that idle
        rewrites credentials under a live session: the bug this module
        exists to stop, reached by a different route."""
        d = tmp_path / "profile"
        (d / "sessions").mkdir(parents=True)          # registry: empty
        with patch("claude_swap.process_detection._candidate_pids",
                   return_value=None):
            assert profile_is_quiescent(d) is False


class TestUndecodableProcessTable:
    """`ps` output is other processes' bytes; none of it is ours to trust."""

    def test_one_undecodable_process_does_not_blind_the_whole_probe(self, tmp_path):
        """errors="replace" keeps the rest of the table readable. Dropping it
        would hide every live claude because some unrelated process had a
        non-UTF-8 byte in its argv."""
        d = tmp_path / "1-acct"
        out = f"  7 /usr/bin/claude CLAUDE_CONFIG_DIR={d}\n"
        with fallback_only(7), patch(
                "claude_swap.process_detection.subprocess.run",
                side_effect=lambda argv, **k: SimpleNamespace(
                    stdout=out if "ewwx" in argv else "", returncode=0)) as run:
            assert scan_env_bound_claude(d) == ([7], True)
        assert all(c.kwargs.get("errors") == "replace" for c in run.call_args_list)

    def test_a_decode_error_is_answered_not_raised(self, tmp_path):
        """UnicodeDecodeError is a ValueError, so neither OSError nor
        SubprocessError catches it; escaping would break `cswap run`."""
        boom = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
        with patch("claude_swap.process_detection.subprocess.run",
                   side_effect=boom):
            assert scan_env_bound_claude(tmp_path) == ([], False)
        # The structured readers decode the same way, per byte string rather
        # than per table, so one bad process cannot blind them either.


class TestEnvIsTakenNotGuessed:
    """`ps` marks no boundary between argv and the environment it appends."""

    def _probes(self, env_line, argv_line, pid=8):
        def fake(argv, **kwargs):
            if "ewwx" in argv:
                out = env_line
            elif argv[-1].endswith("comm="):
                out = ""
            else:
                out = argv_line
            return SimpleNamespace(stdout=out, returncode=0)

        stack = contextlib.ExitStack()
        stack.enter_context(fallback_only(pid))
        stack.enter_context(patch(
            "claude_swap.process_detection.subprocess.run", side_effect=fake))
        return stack

    def test_an_assignment_in_a_prompt_is_not_an_environment_binding(self, tmp_path):
        """`claude -p "set CLAUDE_CONFIG_DIR=<profile>"` names the profile in
        its ARGV. Reading that as a binding pins the profile non-quiescent
        until an unrelated process exits."""
        d = tmp_path / "1-acct"
        argv = f'/usr/bin/claude -p set CLAUDE_CONFIG_DIR={d}'
        with self._probes(f"  8 {argv} PATH=/usr/bin\n", f"  8 {argv}\n"):
            assert scan_env_bound_claude(d) == ([], True)

    def test_the_real_binding_is_still_found_beside_such_an_argument(self, tmp_path):
        d = tmp_path / "1-acct"
        argv = f'/usr/bin/claude -p set CLAUDE_CONFIG_DIR={d}'
        with self._probes(f"  8 {argv} CLAUDE_CONFIG_DIR={d}\n", f"  8 {argv}\n"):
            assert scan_env_bound_claude(d) == ([8], True)

    def test_an_env_value_with_spaces_does_not_swallow_the_binding(self, tmp_path):
        """Guessing the boundary from the first assignment also fails the other
        way: a spacey env value would push the real binding into argv."""
        d = tmp_path / "1-acct"
        argv = "/usr/bin/claude"
        env = f"LS_COLORS=a b c CLAUDE_CONFIG_DIR={d}"
        with self._probes(f"  9 {argv} {env}\n", f"  9 {argv}\n", pid=9):
            assert scan_env_bound_claude(d) == ([9], True)


class TestTheEnvironmentIsParsedNotSubstringSearched:
    """Both ends of a variable have to be pinned, or a neighbour reads as a hit."""

    def _probes(self, env, argv_line):
        def fake(argv, **kwargs):
            if "ewwx" in argv:
                out = f"  11 {argv_line} {env}\n"
            elif argv[-1].endswith("comm="):
                out = ""
            else:
                out = f"  11 {argv_line}\n"
            return SimpleNamespace(stdout=out, returncode=0)

        stack = contextlib.ExitStack()
        stack.enter_context(fallback_only(11))
        stack.enter_context(patch(
            "claude_swap.process_detection.subprocess.run", side_effect=fake))
        return stack

    def test_a_variable_whose_name_merely_ends_in_the_key_is_not_it(self, tmp_path):
        """With no boundary before the name, `OTHER_CLAUDE_CONFIG_DIR=<profile>`
        satisfied a search for `CLAUDE_CONFIG_DIR=<profile>`, so a process
        bound to somebody else's profile held this one non-quiescent."""
        d = tmp_path / "1-acct"
        env = f"OTHER_CLAUDE_CONFIG_DIR={d} CLAUDE_CONFIG_DIR=/other"
        with self._probes(env, "/usr/bin/claude"):
            assert scan_env_bound_claude(d) == ([], True)

    def test_a_value_that_reaches_a_space_is_unknown_not_unbound(self, tmp_path):
        """`<profile> old` used to match `<profile>`, because the value ended
        at whitespace. The flattened form cannot say which of the two it is,
        so the fallback declines rather than guessing either way."""
        d = tmp_path / "1-acct"
        with self._probes(f"CLAUDE_CONFIG_DIR={d} old", "/usr/bin/claude"):
            assert scan_env_bound_claude(d) == ([], False)

    def test_a_binding_that_did_not_survive_the_parse_is_unknown(self, tmp_path):
        """An argv or env value holding the text `CLAUDE_CONFIG_DIR=/elsewhere`
        starts a second assignment under this name, and the later fragment
        overwrote the real value — reporting a live profile as UNBOUND and
        letting its credentials be rewritten. The repetition is reported."""
        d = tmp_path / "1-acct"
        env = f"CLAUDE_CONFIG_DIR={d} NOTE=see CLAUDE_CONFIG_DIR=/elsewhere"
        with self._probes(env, "/usr/bin/claude"):
            assert scan_env_bound_claude(d) == ([], False)

    def test_a_spacey_profile_is_resolved_by_the_structured_read(self, tmp_path):
        """What the fallback declines, the kernel answers exactly: /proc and
        KERN_PROCARGS2 are NUL-separated, so a space is just a character."""
        d = tmp_path / "1 acct"
        with structured(p11=ProcArgs(["/usr/bin/claude"],
                                     {"CLAUDE_CONFIG_DIR": str(d)})):
            assert scan_env_bound_claude(d) == ([11], True)


class TestProbesMustNotBeTruncated:
    def test_every_probe_forces_wide_output(self):
        """Without `ww`, ps truncates to terminal width: plain_argv arrives a
        prefix while the ewwx probe is unbounded, and _split_argv_env hands the
        truncated tail of argv to the environment matcher."""
        from claude_swap import process_detection as pd
        for probe in (pd._ENV_PROBE_ARGV, pd._ARGV_PROBE_ARGV, pd._COMM_PROBE_ARGV):
            assert any("ww" in token for token in probe), probe


class TestNothingWeCouldNotReadIsUnbound:
    """"We could not look" must never answer "nobody is there"."""

    def test_a_pid_no_probe_reported_is_unknown(self, tmp_path):
        """Every `ps` probe failing left empty dicts behind, and a pid missing
        from all three read as unbound — so a machine where the process table
        cannot be read at all reported every profile idle and let a live
        session's credentials be rewritten."""
        from claude_swap import process_detection as pd

        d = tmp_path / "1-acct"
        with fallback_only(4242), \
             patch.object(pd, "_pid_map", return_value=None):
            assert scan_env_bound_claude(d) == ([], False)

    def test_a_pid_that_exited_mid_scan_is_not_unknown(self, tmp_path):
        """`ps` is snapshotted once, after the pid listing, so a short-lived
        process can be listed and then gone. A pid that no longer exists is
        not a live claude, and deferring on it wedged every profile on a busy
        machine."""
        from claude_swap import process_detection as pd

        d = tmp_path / "1-acct"
        with patch.object(pd, "_cannot_host_claude", return_value=False), \
             patch.object(pd, "_candidate_pids", return_value=[4242]), \
             patch.object(pd, "_read_proc_args", return_value=None), \
             patch.object(pd, "_pid_map", return_value={}), \
             patch.object(pd, "is_pid_alive", side_effect=[True, False]):
            assert scan_env_bound_claude(d) == ([], True)

    def test_a_pid_with_argv_but_no_environment_probe_is_unknown(self, tmp_path):
        """The executable and argv probes place it as a main; without the
        combined line there is no environment to place it against."""
        from claude_swap import process_detection as pd

        d = tmp_path / "1-acct"

        def only_argv(argv, label):
            return {4242: "/usr/local/bin/claude"} if "ewwx" not in argv else None

        with fallback_only(4242), patch.object(pd, "_pid_map", side_effect=only_argv):
            assert scan_env_bound_claude(d) == ([], False)

    def test_an_unrecognised_option_is_unknown_not_a_guess(self, tmp_path):
        """`--inspect-port 0 /opt/bin/claude` took `0` for the script. Rather
        than chase the flag list, an option in neither table stops the walk."""
        assert _classify_argv("node --inspect-port 0 /opt/bin/claude") == ARGV_CLAUDE
        assert _classify_argv("node --frobnicate 0 /opt/bin/claude") == ARGV_UNKNOWN
        assert _classify_argv_tokens(
            ["node", "--frobnicate", "0", "/opt/bin/claude"]) == ARGV_UNKNOWN

    def test_stdin_and_end_of_options_are_not_unknown_options(self):
        """`python -` reads the program from stdin, so there is no script and
        no claude. The unknown-option rule caught the bare `-` and left every
        such process unresolved, which held every profile non-quiescent for as
        long as one was running."""
        assert _classify_argv("/usr/bin/python3 -") == ARGV_OTHER
        assert _classify_argv_tokens(["/usr/bin/python3", "-"]) == ARGV_OTHER
        assert _classify_argv("node -- /opt/bin/claude") == ARGV_CLAUDE
        assert _classify_argv_tokens(
            ["node", "--", "/opt/bin/claude"]) == ARGV_CLAUDE

    def test_a_flattened_line_with_an_unplaceable_token_is_unknown(self, tmp_path):
        """`-r "/tmp/hook setup.js" /opt/bin/claude` flattens to four words, so
        the walk drops `/tmp/hook` and reads `setup.js` as the script. It
        cannot tell that from a real worker, so it must say so."""
        d = tmp_path / "1-acct"
        out = f"  4242 node -r /tmp/hook setup.js /opt/bin/claude CLAUDE_CONFIG_DIR={d}\n"
        with TestScanEnvBoundClaude()._ps(
                out, argv_out="  4242 node -r /tmp/hook setup.js /opt/bin/claude\n",
                pids=(4242,)):
            assert scan_env_bound_claude(d) == ([], False)

    def test_the_kernel_answer_is_exact_for_the_same_argv(self, tmp_path):
        """Separated records place every token, so the same command is settled
        rather than deferred."""
        d = tmp_path / "1-acct"
        argv = ["node", "-r", "/tmp/hook setup.js", "/opt/bin/claude"]
        with structured(p4242=ProcArgs(argv, {"CLAUDE_CONFIG_DIR": str(d)})):
            assert scan_env_bound_claude(d) == ([4242], True)


class TestTheCheapCheckComesFirst:
    """Reading a megabyte of arguments for every pid on the box is the cost."""

    def test_a_pid_ruled_out_cheaply_is_never_read_in_full(self, tmp_path):
        from claude_swap import process_detection as pd

        d = tmp_path / "1-acct"
        with patch.object(pd, "_candidate_pids", return_value=[10, 11]), \
             patch.object(pd, "_cannot_host_claude", return_value=True), \
             patch.object(pd, "_read_proc_args") as full:
            assert scan_env_bound_claude(d) == ([], True)
        full.assert_not_called()

    def test_a_read_that_failed_never_rules_a_pid_out(self, tmp_path):
        """`_cannot_host_claude` answers True only from a POSITIVE read, so a
        refused one falls through to the full read rather than shortcutting."""
        from claude_swap import process_detection as pd

        if sys.platform == "darwin":
            with patch.object(pd, "_exec_path_macos", return_value=None):
                assert pd._cannot_host_claude(os.getpid()) is False
        else:
            assert pd._cannot_host_claude(2 ** 30) is False

    def test_the_scan_stops_at_the_first_main_it_finds(self, tmp_path):
        """Every caller asks whether the profile is busy, not by whom."""
        from claude_swap import process_detection as pd

        d = tmp_path / "1-acct"
        seen = []

        def read(pid):
            seen.append(pid)
            return ProcArgs(["/usr/bin/claude"], {"CLAUDE_CONFIG_DIR": str(d)})

        with patch.object(pd, "_candidate_pids", return_value=[10, 11, 12]), \
             patch.object(pd, "_cannot_host_claude", return_value=False), \
             patch.object(pd, "_read_proc_args", side_effect=read):
            assert scan_env_bound_claude(d) == ([10], True)
        assert seen == [10]


class TestThePrefilterMayOnlyRuleOut:
    """It runs before the real read, so it must be a denylist, not an allowlist."""

    @pytest.mark.parametrize("executable", [
        "/Users/eric/.local/share/claude/versions/2.1.269",
        "/opt/claude/versions/9.9.9",
        "/opt/homebrew/Cellar/node/26.0.0/bin/node",
        "/Users/me/Application Support/claude",
        "/usr/local/bin/claude",
        "/opt/tools/assistant",
        "/Users/e/bin/my-renamed-cli",
    ])
    def test_a_name_it_does_not_recognise_falls_through(self, executable):
        """Recognising only claude-shaped names hid six of the eight real
        sessions on this machine behind a version-named binary, and would hide
        any install under a name nobody thought of. The prefilter may answer
        only about programs it positively knows are something else."""
        from claude_swap.process_detection import _known_not_claude

        assert _known_not_claude(executable) is False

    @pytest.mark.parametrize("executable", [
        "/usr/bin/ssh",
        "/usr/libexec/logd",
        "/System/Library/PrivateFrameworks/X.framework/Support/mediaremoted",
        "/bin/zsh",
        "/opt/homebrew/bin/rg",
    ])
    def test_a_name_it_does_recognise_is_still_ruled_out(self, executable):
        """The prefilter has to keep paying for itself, and these are the
        programs a session starts and leaves behind as orphans."""
        from claude_swap.process_detection import _known_not_claude

        assert _known_not_claude(executable) is True

    @pytest.mark.parametrize("executable", [
        "/Library/Acme/assistant",
        "/Applications/Safari.app/Contents/MacOS/Safari",
        "/Library/SystemExtensions/x.systemextension/Contents/MacOS/x",
    ])
    def test_a_writable_location_is_not_enough_on_its_own(self, executable):
        """`/Library` and the inside of an app bundle are writable by whoever
        installed the software there, so a main really can run from one. These
        are ruled out only after argv has been read and says otherwise."""
        from claude_swap.process_detection import (
            _known_not_claude, _unlikely_location,
        )

        assert _known_not_claude(executable) is False
        assert _unlikely_location(executable) is True

    @pytest.mark.parametrize("executable", [
        "/Library/Acme/claude",
        "/Applications/X.app/Contents/MacOS/node",
        "/Library/Frameworks/Python.framework/Versions/3.13/Resources/"
        "Python.app/Contents/MacOS/Python",
        "/usr/bin/nodejs",
    ])
    def test_an_interpreter_or_claude_shape_beats_the_location(self, executable):
        """A framework Python runs as `Python` with a capital P, and Debian
        spells node `nodejs`. Either could be hosting the CLI."""
        from claude_swap.process_detection import (
            _known_not_claude, _unlikely_location,
        )

        assert _known_not_claude(executable) is False
        assert _unlikely_location(executable) is False

    @pytest.mark.parametrize("executable", [
        "/usr/bin/claude",
        "/System/x/claude/versions/1.2.3",
        "/bin/node",
    ])
    def test_a_sealed_location_cannot_rule_out_a_claude_shape(self, executable):
        """Location is the weaker signal; a claude-shaped name wins over it."""
        from claude_swap.process_detection import _known_not_claude

        assert _known_not_claude(executable) is False

    def test_the_native_binary_is_recognised_as_the_cli_itself(self):
        from claude_swap.process_detection import _is_claude_binary

        assert _is_claude_binary("/Users/e/.local/share/claude/versions/2.1.269")
        assert _is_claude_binary("/usr/local/bin/claude")
        assert not _is_claude_binary("/opt/homebrew/bin/node")

    def test_a_version_named_claude_main_is_found_end_to_end(self, tmp_path):
        """The whole point: the prefilter must not hide it from the scan."""
        from claude_swap import process_detection as pd

        d = tmp_path / "1-acct"
        native = "/Users/e/.local/share/claude/versions/2.1.269"
        with patch.object(pd, "_candidate_pids", return_value=[4242]), \
             patch.object(pd, "_exec_path_macos", return_value=native), \
             patch.object(pd, "sys") as fake_sys, \
             patch.object(pd, "_read_proc_args", return_value=ProcArgs(
                 ["/Users/e/.local/bin/claude", "--resume", "x"],
                 {"CLAUDE_CONFIG_DIR": str(d)})):
            fake_sys.platform = "darwin"
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_an_unrecognised_program_bound_to_the_profile_is_unknown(self, tmp_path):
        """It may be a main under a name nobody listed, or a tool subprocess
        that inherited the variable. Answering unbound rewrote credentials
        under whichever it was."""
        d = tmp_path / "1-acct"
        env = {"CLAUDE_CONFIG_DIR": str(d)}
        with structured(p4242=ProcArgs(["/opt/tools/assistant", "--serve"], env)):
            assert scan_env_bound_claude(d) == ([], False)

    def test_a_recognised_program_bound_to_the_profile_is_unbound(self, tmp_path):
        """A zsh that inherited the variable is the common case by an order of
        magnitude, and it outlives the session. Deferring on these would hold
        every profile un-quiescent forever."""
        d = tmp_path / "1-acct"
        env = {"CLAUDE_CONFIG_DIR": str(d)}
        with structured(p4242=ProcArgs(["/bin/zsh", "-c", "sleep 90"], env)):
            assert scan_env_bound_claude(d) == ([], True)

    def test_an_unrecognised_program_bound_elsewhere_is_still_unbound(self, tmp_path):
        """The environment is conclusive in that direction whatever the
        program is, which is what keeps the machine from deferring wholesale."""
        d = tmp_path / "1-acct"
        env = {"CLAUDE_CONFIG_DIR": "/somewhere/else"}
        with structured(p4242=ProcArgs(["/opt/tools/assistant"], env)):
            assert scan_env_bound_claude(d) == ([], True)


class TestApplePsMarksUnreadableProcessesWithParentheses:
    """`(claude)` is a read failure wearing the shape of an answer."""

    def test_a_parenthesised_comm_is_not_a_program_name(self, tmp_path):
        """Apple's `ps` prints the kernel's short name in parentheses when it
        cannot read the argument area. Reading `(claude)` as a program name
        called it "not claude" and reported the profile idle."""
        d = tmp_path / "1-acct"
        with TestScanEnvBoundClaude()._ps(
                "", argv_out="", comm_out="  4242 (claude)\n", pids=(4242,)):
            assert scan_env_bound_claude(d) == ([], False)

    def test_an_unparenthesised_comm_still_settles_the_pid(self, tmp_path):
        d = tmp_path / "1-acct"
        with TestScanEnvBoundClaude()._ps(
                "", argv_out="", comm_out="  4242 /usr/libexec/logd\n",
                pids=(4242,)):
            assert scan_env_bound_claude(d) == ([], True)


class TestCommOutranksAFlattenedArgv:
    """`comm` is one field, so a path with a space in it survives it intact."""

    def test_a_native_main_under_a_spacey_path_is_not_called_idle(self, tmp_path):
        """Flattened argv splits `/Users/me/Application Support/claude` into
        two words and reads the first as the program, so argv says "not
        claude" about a live main. `comm` kept the path whole and has to win."""
        d = tmp_path / "1-acct"
        comm = "  4242 /Users/me/Application Support/claude\n"
        argv = "  4242 /Users/me/Application Support/claude --resume x\n"
        with TestScanEnvBoundClaude()._ps(
                f"{argv.rstrip()} CLAUDE_CONFIG_DIR={d}\n",
                argv_out=argv, comm_out=comm, pids=(4242,)):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_an_interpreter_leaves_the_environment_to_settle_it(self, tmp_path):
        """`comm` says node, which hosts anything, so argv's negative is too
        weak to settle the pid — but an environment naming another profile is
        conclusive whatever the program is, and that is what keeps every node
        process on the machine from deferring."""
        d = tmp_path / "1-acct"
        comm = "  4242 /opt/homebrew/bin/node\n"
        argv = "  4242 node /srv/worker.js\n"
        with TestScanEnvBoundClaude()._ps(
                f"  4242 node /srv/worker.js CLAUDE_CONFIG_DIR=/elsewhere\n",
                argv_out=argv, comm_out=comm, pids=(4242,)):
            assert scan_env_bound_claude(d) == ([], True)


class TestEveryNegativeExitChecksTheGuess:
    """Once a value has been consumed, no "not claude" from the walk is exact."""

    @pytest.mark.parametrize("argv", [
        "node -r hook -e /opt/bin/claude".replace(" ", " "),
        "node -r hook --eval /opt/bin/claude",
        "node -r /opt/bin/claude",
        "node -r hook -",
    ])
    def test_an_early_exit_after_a_consumed_value_is_unknown(self, argv):
        """`node -r "hook -e" /opt/bin/claude` is a real main whose hook path
        held a space. The inline-code exit answered "not claude" outright and
        never looked at the guess, so the main read as a one-liner."""
        assert _classify_argv(argv) == ARGV_UNKNOWN

    @pytest.mark.parametrize("argv", [
        "node --eval code /opt/bin/claude",
        "node -",
        "node",
    ])
    def test_the_same_exits_stay_exact_without_a_guess(self, argv):
        assert _classify_argv(argv) == ARGV_OTHER


class TestTheNodeTablesAreDerivedNotTranscribed:
    """Hand-maintained arity tables drifted, and drift here misreads a main."""

    HELP = """\
Usage: node [options] [ script.js ] [arguments]

Options:
  -                           script read from stdin
  --                          indicate the end of node options
  -c, --check                 syntax check script without executing
  --loader, --experimental-loader=...
                              use the specified module as a loader
  --experimental-test-isolation, --test-isolation=...
                              configures test isolation
  --inspect[=[host:]port]     activate inspector on host:port
  -p, --print [...]           evaluate script and print result
  --disable-wasm-trap-handler Disable trap-handler-based WebAssembly
  --tls-min-v1.2              set default TLS minimum to TLSv1.2
"""

    V8 = """\
Options:
  --max-old-space-size (max size of the old generation in MBytes)
        type: size_t  default: 0
  --harmony (enable all completed harmony features)
        type: bool  default: --no-harmony
  --enable-armv7 (enable ARMv7 instructions)
        type: maybe_bool  default: unset
"""

    def test_aliases_on_one_line_share_one_arity(self):
        """`--loader, --experimental-loader=...` marks the value on the LAST
        spelling only. Reading each token alone filed `--loader` as taking no
        value, so it swallowed nothing and its value read as the script."""
        from claude_swap.node_help import parse_node_help

        value, boolean = parse_node_help(self.HELP)
        assert "--loader" in value and "--experimental-loader" in value
        assert "--experimental-test-isolation" in value
        assert "--test-isolation" in value

    def test_an_attached_only_value_consumes_no_token(self):
        """`--inspect[=port]` takes a value only when attached, so it must not
        be filed with the options that swallow the token after them."""
        from claude_swap.node_help import parse_node_help

        value, boolean = parse_node_help(self.HELP)
        assert "--inspect" in boolean and "--inspect" not in value
        assert "--print" in value          # `-p, --print [...]` is detached

    def test_a_description_one_space_away_is_not_read_as_an_option(self):
        from claude_swap.node_help import parse_node_help

        value, boolean = parse_node_help(self.HELP)
        assert "--disable-wasm-trap-handler" in boolean
        assert "--tls-min-v1.2" in boolean
        assert not any(n.startswith("--Disable") for n in value | boolean)
        assert "-" not in value | boolean and "--" not in value | boolean

    def test_v8_types_decide_arity(self):
        from claude_swap.node_help import parse_v8_options

        value, boolean = parse_v8_options(self.V8)
        assert value == {"--max-old-space-size"}
        assert boolean == {"--harmony", "--enable-armv7"}

    def test_a_maybe_bool_option_swallows_nothing(self):
        """V8's `maybe_bool` takes a value only when one is attached with `=`,
        so filing it with the options that swallow the next token made
        `node --enable-armv7 /opt/bin/claude` read as a main with no script."""
        from claude_swap.node_help import parse_v8_options

        value, boolean = parse_v8_options(self.V8)
        assert "--enable-armv7" not in value
        assert _classify_argv("node --enable-armv7 /opt/bin/claude") == ARGV_CLAUDE

    def test_the_shipped_tables_match_this_machines_node(self):
        """The tables are CLOSED, so a node upgrade that adds or re-types an
        option has to surface as a failure here rather than as a wrong answer
        about a live session. Regenerate with
        `uv run python tools/regen_node_options.py`."""
        import shutil as _shutil
        import subprocess as _subprocess

        if _shutil.which("node") is None:
            pytest.skip("no node on this machine")

        from claude_swap._node_options import (
            GENERATED_FROM, NODE_BOOLEAN_FLAGS, NODE_VALUE_FLAGS,
        )
        from claude_swap.node_help import parse_node_help, parse_v8_options

        def run(*argv):
            p = _subprocess.run(argv, capture_output=True, text=True)
            return p.stdout + p.stderr

        version = run("node", "--version").strip()
        hv, hb = parse_node_help(run("node", "--help"))
        vv, vb = parse_v8_options(run("node", "--v8-options"))
        value, boolean = hv | vv, (hb | vb) - (hv | vv)

        misfiled = sorted((value & NODE_BOOLEAN_FLAGS) | (boolean & NODE_VALUE_FLAGS))
        missing = sorted((value | boolean) - NODE_VALUE_FLAGS - NODE_BOOLEAN_FLAGS)
        stale = sorted((NODE_VALUE_FLAGS | NODE_BOOLEAN_FLAGS) - value - boolean
                       - {"-C", "-r", "-c", "-h", "-i", "-v"})
        assert not misfiled, (
            f"node {version} disagrees with the shipped tables (generated from "
            f"{GENERATED_FROM}) about these options: {misfiled}. An option on "
            "the wrong side either swallows a main's script argument or leaves "
            "its value looking like one. Run tools/regen_node_options.py."
        )
        assert not missing and not stale, (
            f"node {version} and the shipped tables (generated from "
            f"{GENERATED_FROM}) list different options; missing={missing} "
            f"stale={stale}. Run tools/regen_node_options.py."
        )


class TestAnInterpreterSubcommandIsNotTheScript:
    """`bun run x.js` and `deno run x.js` put a word before the script."""

    @pytest.mark.parametrize("argv", [
        "bun run /opt/nm/@anthropic-ai/claude-code/cli.js",
        "deno run /opt/nm/@anthropic-ai/claude-code/cli.js",
        "bun x /opt/nm/@anthropic-ai/claude-code/cli.js",
        "deno exec /opt/nm/@anthropic-ai/claude-code/cli.js",
    ])
    def test_a_subcommand_does_not_hide_the_script(self, argv):
        """Reading `run` as the script filed these as not-a-main, so a live
        session started through bun or deno had its credentials rewritten."""
        assert _classify_argv(argv) == ARGV_CLAUDE

    def test_node_has_no_subcommands(self):
        """`node run` really does mean the file named `run`, so the path after
        it is an argument to that script and not the script."""
        argv = ["node", "run", "/opt/bin/claude"]
        assert _classify_argv_tokens(argv, lambda: "/srv") == ARGV_OTHER


class TestARelativeScriptIsReadAgainstTheProcessCwd:
    """`node cli.js` names a different file from every directory."""

    def test_a_relative_claude_entrypoint_resolves_to_a_main(self):
        argv = ["node", "cli.js"]
        cwd = "/opt/nm/@anthropic-ai/claude-code"
        assert _classify_argv_tokens(argv, lambda: cwd) == ARGV_CLAUDE

    def test_a_relative_script_with_no_cwd_is_unknown(self):
        """Without the working directory there is no reading at all, and
        "no reading" is never "not a main"."""
        assert _classify_argv_tokens(["node", "cli.js"], lambda: None) == ARGV_UNKNOWN
        assert _classify_argv_tokens(["node", "cli.js"]) == ARGV_UNKNOWN

    def test_a_relative_script_bound_to_the_profile_defers(self, tmp_path):
        """The environment names the profile and the script cannot be
        resolved, so the pid has to defer rather than be called idle."""
        d = tmp_path / "1-acct"
        with structured(**{"4242": ProcArgs(
                argv=["node", "cli.js"],
                env={"CLAUDE_CONFIG_DIR": str(d)})}):
            with patch("claude_swap.process_detection._proc_cwd",
                       return_value=None):
                assert scan_env_bound_claude(d) == ([], False)

    def test_the_cwd_settles_that_same_pid(self, tmp_path):
        d = tmp_path / "1-acct"
        with structured(**{"4242": ProcArgs(
                argv=["node", "cli.js"],
                env={"CLAUDE_CONFIG_DIR": str(d)})}):
            with patch("claude_swap.process_detection._proc_cwd",
                       return_value="/opt/nm/@anthropic-ai/claude-code"):
                assert scan_env_bound_claude(d) == ([4242], True)


class TestAParenthesisedArgvIsAReadFailure:
    """`ps` wraps the name in parentheses in argv too, not only in comm."""

    def test_a_parenthesised_argv_is_not_a_program_name(self):
        """`(claude)` in argv means the argument area was unreadable. Reading
        it as a program name called a live main "not claude"."""
        assert _classify_argv("(claude)") == ARGV_UNKNOWN
        assert _classify_argv("(node) --enable-source-maps") == ARGV_UNKNOWN

    def test_a_parenthesised_argv_bound_to_the_profile_defers(self, tmp_path):
        d = tmp_path / "1-acct"
        argv = "  4242 (claude)\n"
        with TestScanEnvBoundClaude()._ps(
                f"  4242 (claude) CLAUDE_CONFIG_DIR={d}\n",
                argv_out=argv, comm_out="", pids=(4242,)):
            assert scan_env_bound_claude(d) == ([], False)

    def test_parentheses_inside_an_argument_are_left_alone(self):
        """Only argv[0] is the program name; a shell one-liner may contain
        anything."""
        assert _classify_argv("/bin/zsh -c (echo hi)") == ARGV_OTHER


class TestAnotherUsersProcessIsNotThisProfilesSession:
    """A `claude` bound to a profile this tool manages runs as this user."""

    def test_a_root_daemon_does_not_hold_the_profile_open(self, tmp_path):
        """`ps` cannot show another user's environment, so every root daemon
        on the machine answered "unknown" for good and no profile was ever
        quiescent. Such a process is not the session about to be rewritten."""
        d = tmp_path / "1-acct"
        with TestScanEnvBoundClaude()._ps(
                "", argv_out="  4242 /opt/telegraf/bin/telegraf\n",
                comm_out="", pids=(4242,), uid_out="  4242 0\n"):
            assert scan_env_bound_claude(d) == ([], True)

    def test_this_users_unreadable_process_still_defers(self, tmp_path):
        """Same shape, same missing environment — but it could be the user's
        own session, so it has to defer."""
        d = tmp_path / "1-acct"
        with TestScanEnvBoundClaude()._ps(
                "", argv_out="  4242 /opt/tools/assistant\n",
                comm_out="", pids=(4242,),
                uid_out=f"  4242 {os.getuid()}\n"):
            assert scan_env_bound_claude(d) == ([], False)

    def test_an_unreadable_owner_defers(self, tmp_path):
        """No owner probe, no ruling out."""
        d = tmp_path / "1-acct"
        with TestScanEnvBoundClaude()._ps(
                "", argv_out="  4242 /opt/tools/assistant\n",
                comm_out="", pids=(4242,), uid_out=""):
            assert scan_env_bound_claude(d) == ([], False)


class TestOwnershipIsTheLastReadingNotTheFirst:
    """A reading that succeeded outranks the uid; only nothing does not."""

    def test_a_readable_matching_main_is_found_whoever_owns_it(self, tmp_path):
        """argv and the environment were both read and both name a main, so
        the answer is settled before ownership is ever a question. Asking
        about the uid first rejected exactly this process."""
        d = tmp_path / "1-acct"
        argv = "  4242 /usr/local/bin/claude --resume x\n"
        with TestScanEnvBoundClaude()._ps(
                f"  4242 /usr/local/bin/claude --resume x CLAUDE_CONFIG_DIR={d}\n",
                argv_out=argv, comm_out="  4242 /usr/local/bin/claude\n",
                pids=(4242,), uid_out="  4242 0\n"):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_a_readable_non_claude_is_unbound_on_that_reading(self, tmp_path):
        d = tmp_path / "1-acct"
        argv = "  4242 /bin/zsh -c sleep\n"
        with TestScanEnvBoundClaude()._ps(
                f"  4242 /bin/zsh -c sleep CLAUDE_CONFIG_DIR={d}\n",
                argv_out=argv, comm_out="", pids=(4242,), uid_out="  4242 0\n"):
            assert scan_env_bound_claude(d) == ([], True)

    def test_an_unreadable_claude_shape_is_never_ruled_out_by_uid(self, tmp_path):
        """argv says main and the environment is withheld. That is not the
        daemon case the ownership rule exists for, so it has to defer."""
        d = tmp_path / "1-acct"
        argv = "  4242 /usr/local/bin/claude --resume x\n"
        with TestScanEnvBoundClaude()._ps(
                "", argv_out=argv, comm_out="", pids=(4242,),
                uid_out="  4242 0\n"):
            assert scan_env_bound_claude(d) == ([], False)

    def test_the_structured_path_defers_on_a_claude_shape_too(self, tmp_path):
        d = tmp_path / "1-acct"
        with structured(p4242=ProcArgs(["/usr/local/bin/claude", "--resume"], None)):
            with patch("claude_swap.process_detection._proc_uid", return_value=0):
                assert scan_env_bound_claude(d) == ([], False)

    def test_the_structured_path_rules_out_another_users_daemon(self, tmp_path):
        d = tmp_path / "1-acct"
        with structured(p4242=ProcArgs(["/opt/telegraf/bin/telegraf"], None)):
            with patch("claude_swap.process_detection._proc_uid", return_value=0):
                assert scan_env_bound_claude(d) == ([], True)

    def test_a_pid_that_vanished_is_gone_not_unknown(self, tmp_path):
        """`/proc/<pid>` disappearing between the argv read and the owner read
        means the process exited, which is settled. Folding that in with an
        unreadable owner reported a dead pid as something we failed to read."""
        d = tmp_path / "1-acct"
        with structured(p4242=ProcArgs(["/opt/tools/assistant"], None)):
            with patch("claude_swap.process_detection._proc_uid",
                       side_effect=ProcessLookupError(4242)):
                assert scan_env_bound_claude(d) == ([], True)

    def test_a_vanished_proc_entry_raises_rather_than_answering_none(self):
        """`/proc/<pid>` is gone, which says the process exited. Returning
        None there made a settled answer look like an unreadable one."""
        from claude_swap import process_detection as pd

        with patch.object(pd.sys, "platform", "linux"), \
             patch.object(pd.os, "stat", side_effect=FileNotFoundError):
            with pytest.raises(ProcessLookupError):
                pd._proc_uid(4242)

    def test_an_unreadable_proc_entry_still_answers_none(self):
        from claude_swap import process_detection as pd

        with patch.object(pd.sys, "platform", "linux"), \
             patch.object(pd.os, "stat", side_effect=PermissionError):
            assert pd._proc_uid(4242) is None

    def test_the_comparison_uses_the_effective_uid(self):
        """macOS `ps uid=` prints the effective uid, so the real one is the
        wrong thing to compare against."""
        import inspect
        from claude_swap import process_detection as pd

        source = inspect.getsource(pd)
        assert "os.getuid()" not in source
        assert "os.geteuid()" in source


class TestAWritableLocationIsRuledOutOnlyOnArgv:
    """`/Library/Acme/assistant` running `claude --resume` is a real main."""

    def _macos(self, executable, argv, env):
        from claude_swap import process_detection as pd

        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(pd, "_candidate_pids",
                                         return_value=[4242]))
        stack.enter_context(patch.object(pd, "_exec_path_macos",
                                         return_value=executable))
        stack.enter_context(patch.object(pd, "_read_proc_args",
                                         return_value=ProcArgs(argv, env)))
        stack.enter_context(patch.object(pd, "is_pid_alive", return_value=True))
        fake_sys = stack.enter_context(patch.object(pd, "sys"))
        fake_sys.platform = "darwin"
        return stack

    def test_a_claude_argv_from_a_library_path_is_a_main(self, tmp_path):
        """The prefilter short-circuited to unbound on the path alone and
        never read the argv that says this is a live session."""
        d = tmp_path / "1-acct"
        with self._macos("/Library/Acme/assistant",
                         ["claude", "--resume", "x"],
                         {"CLAUDE_CONFIG_DIR": str(d)}):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_a_bundle_executable_hosting_a_main_is_found(self, tmp_path):
        d = tmp_path / "1-acct"
        with self._macos("/Applications/Term.app/Contents/MacOS/Term",
                         ["/usr/local/bin/claude"],
                         {"CLAUDE_CONFIG_DIR": str(d)}):
            assert scan_env_bound_claude(d) == ([4242], True)

    def test_a_library_path_with_an_ordinary_argv_is_still_ruled_out(self, tmp_path):
        """The location rule still has to pay for itself once argv agrees."""
        d = tmp_path / "1-acct"
        with self._macos("/Library/Acme/assistant",
                         ["/Library/Acme/assistant", "--serve"],
                         {"CLAUDE_CONFIG_DIR": str(d)}):
            assert scan_env_bound_claude(d) == ([], True)


class TestTheResolvedBinaryIsPublishedWhole:
    """Two threads reach this cache: the CLI and the menubar refresh."""

    @staticmethod
    def _reset():
        from claude_swap import process_detection as pd

        pd._claude_binary_cache = None

    def test_a_concurrent_caller_never_sees_a_half_built_cache(self):
        """Setting the "already looked" flag before the answer let a second
        thread arrive in between, read None, and classify a live native main
        as not the CLI. The window is a `shutil.which` PATH walk wide, and the
        consequence is a running session's credentials rewritten."""
        import threading
        from claude_swap import process_detection as pd

        native = "/Users/e/.local/bin/claude-real"
        started = threading.Barrier(9)

        def slow_which(_name):
            time.sleep(0.05)        # the PATH walk this cache exists to avoid
            return native

        answers = []

        def ask():
            started.wait()
            answers.append(pd._is_claude_binary(native))

        self._reset()
        try:
            with patch.object(pd.shutil, "which", side_effect=slow_which), \
                 patch.object(pd.os.path, "realpath", side_effect=lambda p: p):
                threads = [threading.Thread(target=ask) for _ in range(8)]
                for t in threads:
                    t.start()
                started.wait()
                for t in threads:
                    t.join()
        finally:
            self._reset()

        assert len(answers) == 8
        assert all(answers), (
            "a caller read the cache after it was marked resolved but before "
            f"the value was stored: {answers}"
        )

    def test_path_is_walked_once_however_many_callers_arrive(self):
        import threading
        from claude_swap import process_detection as pd

        calls = []

        def counting_which(name):
            calls.append(name)
            time.sleep(0.02)
            return "/usr/local/bin/claude"

        self._reset()
        try:
            with patch.object(pd.shutil, "which", side_effect=counting_which), \
                 patch.object(pd.os.path, "realpath", side_effect=lambda p: p):
                threads = [threading.Thread(target=pd._native_claude_binary)
                           for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            self._reset()

        assert calls == ["claude"]

    def test_a_missing_claude_on_path_is_cached_as_such(self):
        from claude_swap import process_detection as pd

        self._reset()
        try:
            with patch.object(pd.shutil, "which", return_value=None) as which:
                assert pd._native_claude_binary() is None
                assert pd._native_claude_binary() is None
            assert which.call_count == 1
        finally:
            self._reset()
