"""The live credential, not ~/.claude.json, decides which slot is active.

Logging in with Claude Code directly — the documented recovery for a slot
whose refresh token died — moves the credential without moving the config.
Afterwards the config still names the previous account, and everything that
keys off it attributes the live bytes to the wrong slot.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from claude_swap.switcher import ClaudeAccountSwitcher

CREDS_LIVE = json.dumps({"claudeAiOauth": {
    "accessToken": "at-live", "refreshToken": "rt-live", "expiresAt": 9e12}})
CREDS_SLOT2 = json.dumps({"claudeAiOauth": {
    "accessToken": "at-two", "refreshToken": "rt-two", "expiresAt": 9e12}})

# The config names account 2; the live credential really belongs to account 1.
SEQUENCE = {
    "activeAccountNumber": 2,
    "sequence": [1, 2],
    "accounts": {
        "1": {"email": "one@example.com", "uuid": "uuid-1", "organizationUuid": "org-1"},
        "2": {"email": "two@example.com", "uuid": "uuid-2", "organizationUuid": "org-2"},
    },
}
def _active(value, degraded=False):
    """An ActiveCredentials-shaped double: the oracle reads .value/.degraded."""
    return SimpleNamespace(value=value, degraded=degraded)


PROFILE_ACCOUNT_1 = {"uuid": "uuid-1", "email": "one@example.com",
                     "organizationUuid": "org-1"}


def _drifted(temp_home, mock_claude_config):
    """A switcher whose config says slot 2 while the live credential is slot 1's."""
    s = ClaudeAccountSwitcher()
    s._setup_directories()
    s._init_sequence_file()
    s._write_json(s.sequence_file, SEQUENCE)
    s._get_claude_config_path().write_text(json.dumps({"oauthAccount": {
        "emailAddress": "two@example.com", "organizationUuid": "org-2",
        "accountUuid": "uuid-2"}}), encoding="utf-8")
    return s


class TestOracleResolvesTheSlot:
    def test_resolves_the_slot_the_credential_belongs_to(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            assert s._live_slot_from_identity_oracle(SEQUENCE) == "1"

    def test_unresolvable_returns_none_so_the_config_still_wins(
        self, temp_home, mock_claude_config,
    ):
        """Offline is not evidence: the oracle is advisory, so it abstains."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile", return_value=None):
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None

    def test_no_live_credential_resolves_to_none(self, temp_home, mock_claude_config):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials", return_value=_active("")):
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None

    def test_matching_backup_needs_no_network_at_all(
        self, temp_home, mock_claude_config,
    ):
        """Config and credential agree — the healthy case stays offline."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile") as fetch:
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None
        fetch.assert_not_called()

    def test_result_is_cached_per_credential(self, temp_home, mock_claude_config):
        """_build_accounts_info runs several times a command; resolve once."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1) as fetch:
            for _ in range(3):
                assert s._live_slot_from_identity_oracle(SEQUENCE) == "1"
        assert fetch.call_count == 1

    def test_a_partial_identity_does_not_affirm_a_slot(
        self, temp_home, mock_claude_config,
    ):
        """_resolved_matches_slot_identity's None means unverifiable, not match."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value={"uuid": "uuid-nobody"}), \
             patch.object(s, "_resolved_matches_slot_identity", return_value=None):
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None


class TestBuildAccountsInfoPrefersTheOracle:
    def test_active_slot_follows_the_credential_not_the_stale_config(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            info = s._build_accounts_info()
        active = [num for num, _e, _o, _ou, is_active, _c, _al in info if is_active]
        assert active == [1], "the config said 2; the credential says 1"

    def test_without_the_oracle_the_config_still_decides(
        self, temp_home, mock_claude_config,
    ):
        """No drift, no override — the ordinary path is untouched."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile") as fetch:
            info = s._build_accounts_info()
        active = [num for num, _e, _o, _ou, is_active, _c, _al in info if is_active]
        assert active == [2]
        fetch.assert_not_called()


class TestEveryCallerAgreesOnTheActiveSlot:
    """list, status and the JSON payload each used to answer this from the
    config alone, so one drift showed up three different ways."""

    def _drifted_with_oracle(self, temp_home, mock_claude_config):
        s = _drifted(temp_home, mock_claude_config)
        return s, [
            patch.object(s, "_read_active_credentials",
                         return_value=_active(CREDS_LIVE)),
            patch.object(s, "_read_credentials", return_value=CREDS_LIVE),
            patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2),
            patch("claude_swap.oauth.fetch_oauth_profile",
                  return_value=PROFILE_ACCOUNT_1),
        ]

    def test_resolver_prefers_the_credential_over_the_config(
        self, temp_home, mock_claude_config,
    ):
        s, patches = self._drifted_with_oracle(temp_home, mock_claude_config)
        with patches[0], patches[1], patches[2], patches[3]:
            slot = s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2"))
        assert slot == "1"

    def test_resolver_keeps_the_config_when_the_oracle_abstains(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile", return_value=None):
            slot = s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2"))
        assert slot == "2"

    def test_status_payload_reports_the_credentials_account(
        self, temp_home, mock_claude_config,
    ):
        """The email must follow the slot, or the payload names slot 1 while
        printing slot 2's address."""
        s, patches = self._drifted_with_oracle(temp_home, mock_claude_config)
        with patches[0], patches[1], patches[2], patches[3]:
            payload = s._build_status_payload()
        assert payload["active"]["email"] == "one@example.com"

    def test_unmanaged_login_is_still_unmanaged(
        self, temp_home, mock_claude_config,
    ):
        """No slot owns the live credential — the oracle must not invent one."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value={"uuid": "uuid-stranger",
                                 "email": "nobody@example.com",
                                 "organizationUuid": "org-x"}):
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None


class TestCachedVerdictsCannotGoStale:
    """The cache key is the credential, which outlives the things it maps to."""

    def test_a_removed_slot_is_not_returned_from_cache(
        self, temp_home, mock_claude_config,
    ):
        """A long-lived switcher (the menubar refresh loop) caches slot 1, then
        slot 1 is removed. status() indexes data["accounts"][num] on this
        answer, so a phantom slot is a KeyError, not a mislabel."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            assert s._live_slot_from_identity_oracle(SEQUENCE) == "1"

            without_1 = {"activeAccountNumber": 2, "sequence": [2],
                         "accounts": {"2": SEQUENCE["accounts"]["2"]}}
            assert s._live_slot_from_identity_oracle(without_1) is None

    def test_status_survives_a_slot_removed_under_a_cached_verdict(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            s._live_slot_from_identity_oracle(SEQUENCE)          # warms the cache
            without_1 = {"activeAccountNumber": 2, "sequence": [2],
                         "accounts": {"2": SEQUENCE["accounts"]["2"]}}
            s._write_json(s.sequence_file, without_1)
            s.status()                                            # must not KeyError

    def test_the_verdict_is_filed_under_the_credential_it_resolved(
        self, temp_home, mock_claude_config,
    ):
        """A switch landing between the two credential reads must not file the
        new bytes' identity under the old bytes' fingerprint."""
        from claude_swap import oauth as oauth_mod

        s = _drifted(temp_home, mock_claude_config)
        moved = json.dumps({"claudeAiOauth": {
            "accessToken": "at-moved", "refreshToken": "rt-moved", "expiresAt": 9e12}})
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_prefetch_live_identity",
                          return_value={"live": moved,
                                        "resolved": PROFILE_ACCOUNT_1}):
            # probed != the live bytes, so this IS the moved-mid-lookup case:
            # abstain now, but still file the verdict under what it describes.
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None
        cached_fp, cached_identity = s._oracle_identity_cache
        assert cached_identity == PROFILE_ACCOUNT_1
        assert cached_fp == oauth_mod.credential_fingerprint(moved), \
            "filed under the bytes the identity actually describes"
        assert cached_fp != oauth_mod.credential_fingerprint(CREDS_LIVE)


class TestTheCacheHoldsIdentityNotSlot:
    """A slot NUMBER goes wrong without the credential moving at all."""

    def _warm(self, s):
        return (patch.object(s, "_read_active_credentials",
                             return_value=_active(CREDS_LIVE)),
                patch.object(s, "_read_credentials", return_value=CREDS_LIVE),
                patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2),
                patch("claude_swap.oauth.fetch_oauth_profile",
                      return_value=PROFILE_ACCOUNT_1))

    def test_reassigning_a_slot_moves_the_answer_with_it(
        self, temp_home, mock_claude_config,
    ):
        """`cswap swap 1 2` reassigns which account holds slot 1; a cached slot
        number would attribute the live credential to whoever moved in."""
        s = _drifted(temp_home, mock_claude_config)
        a, b, c, d = self._warm(s)
        with a, b, c, d:
            assert s._live_slot_from_identity_oracle(SEQUENCE) == "1"
            swapped = {
                "activeAccountNumber": 2, "sequence": [1, 2],
                "accounts": {
                    "1": SEQUENCE["accounts"]["2"],   # accounts exchanged slots
                    "2": SEQUENCE["accounts"]["1"],
                },
            }
            # account_identity() reads the sequence FILE, so the swap has to
            # land there and not just in the dict passed to the oracle.
            s._write_json(s.sequence_file, swapped)
            assert s._live_slot_from_identity_oracle(swapped) == "2"

    def test_an_unresolved_lookup_is_retried_not_remembered(
        self, temp_home, mock_claude_config,
    ):
        """Offline once must not pin a long-lived TUI to the stale config until
        the credential itself rotates."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   side_effect=[None, PROFILE_ACCOUNT_1]) as fetch:
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None
            assert s._live_slot_from_identity_oracle(SEQUENCE) == "1"
        assert fetch.call_count == 2, "the miss must not have been cached"


class TestTheOracleKnowsWhenNotToSpeak:
    def test_a_degraded_read_is_never_resolved(self, temp_home, mock_claude_config):
        """The plaintext fallback after an unreadable Keychain can be an older
        generation of a DIFFERENT account; resolving it would mark the wrong
        slot active everywhere at once."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE, degraded=True)), \
             patch("claude_swap.oauth.fetch_oauth_profile") as fetch:
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None
        fetch.assert_not_called()


class TestConfigNamingNoManagedSlot:
    """The case that needs the API most: nothing local to compare against."""

    def _unmanaged_config(self, s):
        s._get_claude_config_path().write_text(json.dumps({"oauthAccount": {
            "emailAddress": "gone@example.com", "organizationUuid": "org-gone",
            "accountUuid": "uuid-gone"}}), encoding="utf-8")

    def test_a_managed_login_is_found_behind_an_unmanaged_config(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        self._unmanaged_config(s)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            assert s._live_slot_from_identity_oracle(SEQUENCE) == "1"

    def test_it_still_abstains_when_the_direct_lookup_fails(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        self._unmanaged_config(s)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile", return_value=None):
            assert s._live_slot_from_identity_oracle(SEQUENCE) is None


PROFILE_STRANGER = {"uuid": "uuid-stranger", "email": "nobody@example.com",
                    "organizationUuid": "org-x"}


class TestResolvedButUnmanagedIsNotAFailedProbe:
    """Both return "no slot", but only one may fall back to the config."""

    def test_an_identified_unmanaged_login_does_not_borrow_a_managed_slot(
        self, temp_home, mock_claude_config,
    ):
        """The config names slot 2. The credential is positively identified as
        nobody cswap manages. Reporting slot 2 would run its active-usage path
        with a stranger's credential."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_STRANGER):
            assert s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2")) is None

    def test_an_unresolvable_probe_still_falls_back_to_the_config(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile", return_value=None):
            assert s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2")) == "2"


class TestAutomationAndDisplayAgree:
    def test_current_account_number_matches_the_usage_pass(
        self, temp_home, mock_claude_config,
    ):
        """AutoSwitchEngine reads current_account_number(); the usage pass reads
        _build_accounts_info(). Disagreement makes automation switch off the
        wrong account."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            info = s._build_accounts_info()
            active_from_display = [n for n, _e, _o, _ou, a, _c, _al in info if a]
            assert s.current_account_number() == "1"
            assert active_from_display == [1]


class TestAMissingConfigLoginStaysInactive:
    def test_no_oauth_account_means_no_slot(self, temp_home, mock_claude_config):
        """A logged-out or half-removed config must not be reported active just
        because the credential store still holds a managed blob — autoswitch
        would switch a configuration nobody is logged into."""
        s = _drifted(temp_home, mock_claude_config)
        s._get_claude_config_path().write_text(json.dumps({}), encoding="utf-8")
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1) as fetch:
            assert s.current_account_number() is None
        fetch.assert_not_called()

    def test_status_names_the_credentials_owner_when_unmanaged(
        self, temp_home, mock_claude_config,
    ):
        """Config names managed two@; credential is a stranger. Printing
        two@ '(not managed)' would call a managed address unmanaged."""
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_STRANGER):
            payload = s._build_status_payload()
        assert payload["active"]["email"] == "nobody@example.com"
        assert payload["active"]["managed"] is False


class TestUnverifiableIsNotUnmanaged:
    def test_a_partial_identity_keeps_the_config_slot(
        self, temp_home, mock_claude_config,
    ):
        """Slots without stored uuids make every comparison unverifiable.
        Reading that as "unmanaged" would stop auto-switch on a managed login."""
        s = _drifted(temp_home, mock_claude_config)
        no_uuids = {
            "activeAccountNumber": 2, "sequence": [1, 2],
            "accounts": {
                "1": {"email": "one@example.com", "organizationUuid": "org-1"},
                "2": {"email": "two@example.com", "organizationUuid": "org-2"},
            },
        }
        s._write_json(s.sequence_file, no_uuids)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value={"uuid": "uuid-1"}), \
             patch.object(s, "_resolved_matches_slot_identity", return_value=None):
            assert s._resolve_active_slot(
                no_uuids, ("two@example.com", "org-2")) == "2"

    def test_a_conclusive_no_match_still_reports_unmanaged(
        self, temp_home, mock_claude_config,
    ):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_STRANGER), \
             patch.object(s, "_resolved_matches_slot_identity", return_value=False):
            assert s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2")) is None


class TestAResolverWithNoConfiguredLogin:
    def test_build_accounts_info_marks_nothing_active(
        self, temp_home, mock_claude_config,
    ):
        """_build_accounts_info calls the resolver directly, so the guard has to
        live there — not only in current_account_number()."""
        s = _drifted(temp_home, mock_claude_config)
        s._get_claude_config_path().write_text(json.dumps({}), encoding="utf-8")
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1) as fetch:
            info = s._build_accounts_info()
        assert [n for n, _e, _o, _ou, a, _c, _al in info if a] == []
        fetch.assert_not_called()


class TestIdentityComparisonUsesTheCallersSnapshot:
    def test_a_concurrent_swap_does_not_leak_into_this_pass(
        self, temp_home, mock_claude_config,
    ):
        """The dict on disk moves under us; the answer must stay coherent with
        the snapshot the caller is iterating and indexing."""
        s = _drifted(temp_home, mock_claude_config)
        swapped_on_disk = {
            "activeAccountNumber": 2, "sequence": [1, 2],
            "accounts": {"1": SEQUENCE["accounts"]["2"],
                         "2": SEQUENCE["accounts"]["1"]},
        }
        s._write_json(s.sequence_file, swapped_on_disk)
        # caller still holds the ORIGINAL layout
        assert s._resolved_matches_slot_identity("1", PROFILE_ACCOUNT_1,
                                                 SEQUENCE) is True
        assert s._resolved_matches_slot_identity("1", PROFILE_ACCOUNT_1) is False


class TestTheCredentialMayMoveDuringTheLookup:
    def test_a_switch_mid_lookup_abstains_rather_than_naming_the_old_slot(
        self, temp_home, mock_claude_config,
    ):
        """The profile request takes time; a concurrent `cswap switch` during it
        makes the answer describe bytes that are no longer active."""
        s = _drifted(temp_home, mock_claude_config)
        moved = json.dumps({"claudeAiOauth": {
            "accessToken": "at-new", "refreshToken": "rt-new", "expiresAt": 9e12}})
        # first read (pre-lookup) sees the old bytes, the re-read sees the new
        with patch.object(s, "_read_active_credentials",
                          side_effect=[_active(CREDS_LIVE), _active(moved)]), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            assert s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2")) == "2", "keeps the config"

    def test_a_stable_credential_still_resolves(self, temp_home, mock_claude_config):
        s = _drifted(temp_home, mock_claude_config)
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch.object(s, "_read_account_credentials", return_value=CREDS_SLOT2), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=PROFILE_ACCOUNT_1):
            assert s._resolve_active_slot(
                SEQUENCE, ("two@example.com", "org-2")) == "1"


class TestStatusResolvesOnlyOnce:
    def test_an_unavailable_endpoint_is_not_asked_twice(
        self, temp_home, mock_claude_config,
    ):
        """Failed resolutions are deliberately uncached, so asking twice costs a
        second network wait — up to ~10s on one status call."""
        s = _drifted(temp_home, mock_claude_config)
        s._get_claude_config_path().write_text(json.dumps({"oauthAccount": {
            "emailAddress": "gone@example.com", "organizationUuid": "org-gone",
            "accountUuid": "uuid-gone"}}), encoding="utf-8")
        with patch.object(s, "_read_active_credentials",
                          return_value=_active(CREDS_LIVE)), \
             patch.object(s, "_read_credentials", return_value=CREDS_LIVE), \
             patch("claude_swap.oauth.fetch_oauth_profile",
                   return_value=None) as fetch:
            s._build_status_payload()
        assert fetch.call_count == 1, "status resolved the identity twice"
