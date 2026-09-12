# Local patch: `fix/wrong-account-and-resume-logout`

This checkout carries a local fix that is not upstream. `cswap` on this machine is installed from this branch, not from PyPI.

## What the patch fixes

Two ways `cswap` acted on the wrong account.

**Mid-session logout.** `profile_is_quiescent` in `src/claude_swap/session.py` trusted only Claude Code's own session registry. A live instance can be missing from that registry — `claude --resume` mains with no record are the observed case — so a busy profile read as idle and `_bootstrap` rewrote its keychain credentials underneath a running session. That session was logged out mid-conversation.

The fix adds `scan_env_bound_claude(session_dir)` in `src/claude_swap/process_detection.py`, which asks the OS which processes are bound to that profile through `CLAUDE_CONFIG_DIR`. Each process is read from a structured source: `/proc/<pid>/cmdline` and `/proc/<pid>/environ` on Linux, `KERN_PROCARGS2` through `sysctl` on macOS. Both hand back argv and the environment as NUL-separated records, exactly as `execve` received them, so a path with a space in it stays one argument and no amount of assignment-shaped text in an argument can be mistaken for an environment variable.

`_classify_argv_tokens` decides what counts: a `claude` main, a `main --resume`, and a node-hosted main from the npm layout all count; a shell that merely inherited the variable, a tool subprocess, a worker handed the entrypoint's path, and a `node --eval` or `node -pe` one-liner do not. Bash-tool shells outlive their parent and outnumber real mains roughly nine to one, so binding on the variable alone would pin a profile un-quiescent forever.

Interpreter options are read from a closed table, one per interpreter family, listing which options take a value and which do not. An option in neither list stops the walk and reports unknown. Guessing was the failure the table replaces: `node --inspect-port 0 /opt/bin/claude` read `0` as the script and filed a live main as idle, and no flag list stays complete on its own.

`ps` remains only as the last resort, for pids the kernel refuses and for platforms neither source covers. Everything `ps` prints is flattened and space-separated, so it cannot always tell an argument from a variable or one value from the next. Those cases report unknown rather than guessing: a repeated variable name, a value that reaches a space, a binding present in the text that did not survive the parse, or an argument the walk could not place because a consumed option value held a space. Each of the three `ps` probes can fail on its own, and a failed probe is an absent answer rather than a negative one, so a pid none of them reported is unknown too.

That is the rule the whole probe runs on: a pid is unbound only when its program and its environment were both read from a source that says where each argument ends, and neither named claude. Anything that could not be read is unknown, never unbound.

The probe fails closed. It returns `(pids, probed)`, and `profile_is_quiescent` reports idle only when `probed` is true and the list is empty. A process that could be a `claude` main and could not be read leaves this profile's liveness unknown, and re-seeding credentials on an unknown is the same mid-session logout reached by a different route. The same guard runs before backup-credential invalidation deletes a profile's credential material, and an unreadable session-registry record defers there too.

Every caller asks whether the profile is busy rather than by whom, so the scan stops at the first main it finds and the returned list is not exhaustive. Before reading a pid's full argument area, a cheap check rules it out: `proc_pidpath` on macOS, the world-readable `cmdline` on Linux. A main runs as either the `claude` binary or an interpreter hosting it, so an executable that is neither settles the pid without a megabyte read, and on Linux the environment is never opened for a process that could not be a main. That check answers "ruled out" only from a successful read, so a refused one falls through to the full read. On this machine it settles 1379 of 1490 pids and a full scan that matches nothing takes 0.33s, against 0.41s with every pid read in full.

Windows is the one carve-out. There is no cheap environment read there, so `scan_env_bound_claude` reports "probed, nothing bound" and Windows keeps the registry-only behavior that shipped before this probe existed: an unregistered `claude --resume` on Windows can still have its credentials rewritten underneath it. Failing closed instead would leave every Windows profile permanently un-re-seedable, which is worse. This fork runs on macOS.

**Stale config, wrong active slot.** The active account came from `~/.claude.json`, which goes stale the moment you log in with Claude Code directly — and that is the documented recovery for a slot whose refresh token died. An identity oracle in `src/claude_swap/switcher.py` (`_resolve_active_slot`, `_active_slot_and_identity`, `_resolve_live_identity`, `_prefetch_live_identity`, `_slot_holding_identity`) resolves the active slot from the live credential instead. The cache is keyed on the credential fingerprint, invalidated when account metadata changes, and the verdict is bound to the credential that produced it, so a cached answer cannot outlive its input. Only an identity complete enough to place a slot is cached: a uuid-only response cannot place slots that store no uuid, and remembering it would replay that unusable answer for the life of the credential instead of asking again.

## Branch and provenance

Branch `fix/wrong-account-and-resume-logout` on the fork `git@github.com:ericmason/claude-swap.git`, on top of tag `v0.26.0`. The first commit is the recovered patch: six files, 1426 insertions, 16 deletions. Three later commits apply what three rounds of Codex review found: the second replaced the `ps` parsing with the structured kernel sources, and the third replaced every remaining place where an unreadable process answered "unbound".

The patch was originally written on August 29, 2026 against claude-swap 0.25.0 plus upstream commit `2213700`, went through eleven review passes, and was never pushed. `/tmp` was wiped and `cswap upgrade` to 0.26.0 overwrote the installed copy, so the only surviving record was the Claude Code session transcript.

It was recovered on September 11, 2026 by replaying every file-mutating tool call from that transcript against a fresh checkout of `2213700`, then cherry-picking the result onto `v0.26.0`. The replayed diffstat matched the lost commit's recorded diffstat exactly, and the cherry-pick auto-merged with no conflicts. Transcript id: `98a02f90-cc2d-422a-a34a-8192e5db1ca8` (workspace `/Users/eric/src_eric/codetogo`), readable with `aii show cc/98a02f90 --from N --to M`.

## Install

```
uv tool install --force --from git+https://github.com/ericmason/claude-swap.git@fix/wrong-account-and-resume-logout claude-swap
```

This puts `cswap` and `claude-swap` on PATH at `~/.local/bin/`.

## Upgrade caveat

`cswap upgrade` runs `uv tool upgrade claude-swap`, and uv resolves that against whatever the install receipt records. After the command above, `~/.local/share/uv/tools/claude-swap/uv-receipt.toml` records the git branch rather than the PyPI package, so `cswap upgrade` cannot pull a PyPI release over this patch. It reports "Nothing to upgrade" and leaves the patched code in place.

The cost is that `cswap upgrade` can no longer bring in upstream releases at all. The update notifier still checks PyPI, so once upstream publishes a version above 0.26.0, `cswap` will say a newer version is available and `cswap upgrade` will do nothing about it. To take a new upstream release, rebase this branch onto the new tag, push it, and re-run the install command above.

To go back to stock and lose the patch, run `uv tool install --force claude-swap`.

## Tests

```
uv run pytest -q
```

2287 passed, 4 skipped. Pristine `v0.26.0` is 2166 passed, 3 skipped, so the patch adds 121 passing tests and breaks nothing. The fourth skip is a `/proc` test that only runs on Linux. The new coverage is `tests/test_active_slot_identity_oracle.py` (35 tests, including `TestCachedVerdictsCannotGoStale` and `TestAnIdentityTooPartialToPlaceIsNotCached`) plus additions to `tests/test_process_detection.py` and `tests/test_session.py`.
