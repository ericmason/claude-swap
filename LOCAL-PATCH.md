# Local patch: `fix/wrong-account-and-resume-logout`

This checkout carries a local fix that is not upstream. `cswap` on this machine is installed from this branch, not from PyPI.

## What the patch fixes

Two ways `cswap` acted on the wrong account.

**Mid-session logout.** `profile_is_quiescent` in `src/claude_swap/session.py` trusted only Claude Code's own session registry. A live instance can be missing from that registry — `claude --resume` mains with no record are the observed case — so a busy profile read as idle and `_bootstrap` rewrote its keychain credentials underneath a running session. That session was logged out mid-conversation.

The fix adds `scan_env_bound_claude(session_dir)` in `src/claude_swap/process_detection.py`, which asks the OS which processes are bound to that profile through `CLAUDE_CONFIG_DIR`. `_is_claude_argv` decides what counts: a `claude` main, a `main --resume`, and a node-hosted main from the npm layout all count; a shell that merely inherited the variable, a tool subprocess, a worker handed the entrypoint's path, a `node --eval` one-liner, and an empty argv do not. Bash-tool shells outlive their parent and outnumber real mains roughly nine to one, so binding on the variable alone would pin a profile un-quiescent forever. The flattened environment is parsed into name and value pairs rather than substring-searched, so neither `OTHER_CLAUDE_CONFIG_DIR` nor a sibling profile named `<dir>-old` nor a value of `<dir> something-else` can satisfy a probe for `<dir>`.

The probe fails closed. It returns `(pids, probed)`, and `profile_is_quiescent` reports idle only when `probed` is true and the list is empty. A `ps` that is missing, raised, timed out, or exited nonzero leaves this profile's liveness unknown, and re-seeding credentials on an unknown is the same mid-session logout reached by a different route. Windows is the one carve-out: there is no cheap environment read there, so the probe is absent rather than broken, and failing closed would leave every Windows profile permanently un-re-seedable.

**Stale config, wrong active slot.** The active account came from `~/.claude.json`, which goes stale the moment you log in with Claude Code directly — and that is the documented recovery for a slot whose refresh token died. An identity oracle in `src/claude_swap/switcher.py` (`_resolve_active_slot`, `_active_slot_and_identity`, `_resolve_live_identity`, `_prefetch_live_identity`, `_slot_holding_identity`) resolves the active slot from the live credential instead. The cache is keyed on the credential fingerprint, invalidated when account metadata changes, and the verdict is bound to the credential that produced it, so a cached answer cannot outlive its input. Only an identity complete enough to place a slot is cached: a uuid-only response cannot place slots that store no uuid, and remembering it would replay that unusable answer for the life of the credential instead of asking again.

## Branch and provenance

Branch `fix/wrong-account-and-resume-logout` on the fork `git@github.com:ericmason/claude-swap.git`, on top of tag `v0.26.0`. The first commit is the recovered patch: six files, 1426 insertions, 16 deletions. A second commit applies the four fixes a Codex review of it found.

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

2248 passed, 3 skipped. Pristine `v0.26.0` is 2166 passed, 3 skipped, so the patch adds 82 passing tests and breaks nothing. The new coverage is `tests/test_active_slot_identity_oracle.py` (34 tests, including `TestCachedVerdictsCannotGoStale` and `TestAnIdentityTooPartialToPlaceIsNotCached`) plus additions to `tests/test_process_detection.py` and `tests/test_session.py`.
