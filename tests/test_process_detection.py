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
    ClaudeSession,
    ProcArgs,
    _is_claude_argv,
    _is_claude_argv_tokens,
    scan_env_bound_claude,
    IdeInstance,
    get_claude_dir,
    get_running_instances,
    is_pid_alive,
    list_ide_instances,
    list_sessions,
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
    with patch("claude_swap.process_detection._candidate_pids",
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
    with patch("claude_swap.process_detection._candidate_pids",
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
        assert _is_claude_argv(argv) is True

    @pytest.mark.parametrize("argv", [
        "node /usr/lib/node_modules/@anthropic-ai/claude-code/cli.js",
        "node --enable-source-maps /opt/nm/@anthropic-ai/claude-code/cli.js",
    ])
    def test_the_npm_distribution_is_a_main(self, argv):
        """Its basename is cli.js, so the complete-script-name rule would file
        a real main as a worker and let its credentials be rewritten."""
        assert _is_claude_argv(argv) is True

    @pytest.mark.parametrize("argv", [
        "node --require setup.js /opt/nm/@anthropic-ai/claude-code/cli.js",
        "node -r hook.js /opt/nm/@anthropic-ai/claude-code/cli.js",
        "node --require setup.js /opt/bin/claude",
    ])
    def test_an_option_that_consumes_a_value_does_not_hide_the_script(self, argv):
        """Skipping only the option leaves its value looking like the script,
        so a live main reads as a worker called setup.js."""
        assert _is_claude_argv(argv) is True

    def test_a_consumed_value_is_not_mistaken_for_claude(self):
        assert _is_claude_argv("node --require setup.js worker.js") is False

    def test_a_lookalike_package_directory_is_not(self):
        assert _is_claude_argv("node /home/claude-code-notes/build.js") is False

    @pytest.mark.parametrize("argv", [
        "node /Users/me/Application Support/claude",
        "node --enable-source-maps /Users/me/Application Support/claude",
    ])
    def test_an_interpreter_script_path_may_contain_spaces(self, argv):
        """`comm` says `node`, so the executable probe cannot rescue this one;
        rejecting it would call a live profile idle."""
        assert _is_claude_argv(argv) is True

    def test_a_lookalike_script_is_not_claude(self):
        assert _is_claude_argv("node /opt/claude-helper") is False

    @pytest.mark.parametrize("argv", [
        "node worker.js /tmp/claude",
        "node /srv/app/worker.mjs /tmp/claude",
    ])
    def test_a_worker_merely_passed_a_claude_path_is_not_a_main(self, argv):
        """A complete script name ends the script argument, so what follows is
        claude's own argv. Such a child inherits CLAUDE_CONFIG_DIR and can
        outlive its parent — reading it as a main pins the profile forever."""
        assert _is_claude_argv(argv) is False

    def test_an_interpreter_flag_does_not_hide_the_script(self):
        assert _is_claude_argv("node --enable-source-maps /opt/app/server.js") is False
        assert _is_claude_argv("node --enable-source-maps /opt/claude/claude") is True

    @pytest.mark.parametrize("argv", [
        "/bin/zsh -c source /tmp/snapshot-zsh",   # Bash-tool shell, can outlive its parent
        "/usr/bin/rg --files",
        "",
    ])
    def test_inherited_strays_are_not(self, argv):
        assert _is_claude_argv(argv) is False

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
        assert _is_claude_argv(argv) is False

    def test_inline_code_does_not_hide_a_script_that_comes_after_a_value_flag(self):
        """The two option shapes must stay apart: --require consumes a token
        and carries on, --eval ends the search."""
        assert _is_claude_argv(
            "node --require setup.js --enable-source-maps /opt/bin/claude") is True

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
        assert _is_claude_argv(argv) is False

    def test_a_cluster_with_neither_letter_still_consumes_its_value(self):
        assert _is_claude_argv("node -r hook.js /opt/bin/claude") is True

    @pytest.mark.parametrize("argv", [
        "node /Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js",
        "node --enable-source-maps "
        "/Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js",
    ])
    def test_the_npm_entrypoint_under_a_spacey_path_is_a_main(self, argv):
        """Splitting at the space made the script `/Users/me/Application`,
        which is not claude — so a live main read as an idle profile and its
        credentials were rewritten underneath it."""
        assert _is_claude_argv(argv) is True

    def test_the_package_directory_only_counts_at_the_entrypoint(self):
        """Matching `/claude-code/` anywhere in argv read a worker that was
        handed the entrypoint's path as the entrypoint itself."""
        assert _is_claude_argv(
            "node /srv/worker.js /opt/nm/@anthropic-ai/claude-code/cli.js") is False

    @pytest.mark.parametrize("argv", [
        "vim /usr/local/bin/claude",
        "tail -f /Users/e/.local/bin/claude",
        "cp /usr/local/bin/claude /tmp/claude",
    ])
    def test_a_command_merely_naming_the_binary_is_not_a_main(self, argv):
        """These run FROM a claude session, so they inherit CLAUDE_CONFIG_DIR
        and reach this test; treating them as mains would hold the profile
        un-quiescent for as long as the editor stayed open."""
        assert _is_claude_argv(argv) is False


class TestArgvThatArrivesAlreadySeparated:
    """The kernel hands over argv as records, so nothing has to be re-split."""

    @pytest.mark.parametrize("argv", [
        ["/usr/local/bin/claude", "--resume", "x"],
        ["node", "/Users/me/Application Support/nm/@anthropic-ai/claude-code/cli.js"],
        ["node", "--enable-source-maps", "/opt/my apps/bin/claude"],
        ["node", "--require", "setup.js", "/opt/bin/claude"],
    ])
    def test_a_main_is_recognised_however_spacey_its_path(self, argv):
        assert _is_claude_argv_tokens(argv) is True

    @pytest.mark.parametrize("argv", [
        ["node", "-pe", "require('/opt/claude-code/cli.js')"],
        ["node", "worker.js", "/opt/nm/@anthropic-ai/claude-code/cli.js"],
        ["/bin/zsh", "-c", "ls"],
        ["vim", "/usr/local/bin/claude"],
        [],
    ])
    def test_a_lookalike_is_still_not_one(self, argv):
        assert _is_claude_argv_tokens(argv) is False

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

        env = _split_nul_env([b"LS_COLORS=a b c", b"EQ=x=y", b"", b"T=1"])
        assert env == {"LS_COLORS": "a b c", "EQ": "x=y", "T": "1"}

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
            env_returncode=None):
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
