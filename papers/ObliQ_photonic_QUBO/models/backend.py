"""Quandela credential resolution for the photonic solvers.

Single source of truth for the API token. Resolution order, first hit wins:

1. an explicit ``token`` passed by the caller (config ``solver_options.token``),
2. the ``QUANDELA_API_KEY`` / ``QUANDELA_TOKEN`` environment variables,
3. the ``configuration/_QUANDELA_API_KEY`` file (git-ignored).

Returns an empty string when nothing is configured, which is what
``pcvl.RemoteProcessor`` expects when the token comes from Perceval's own saved
persistent configuration instead.
"""

from __future__ import annotations

import os

#: Git-ignored file holding the token for local development.
API_KEY_PATH = os.path.join("configuration", "_QUANDELA_API_KEY")

#: Environment variables consulted before the file.
API_KEY_ENV_VARS = ("QUANDELA_API_KEY", "QUANDELA_TOKEN")


def read_quandela_api_key(token: str | None = None) -> str:
    """Resolve the Quandela API token.

    Args:
        token: explicit token from a config/caller; short-circuits the lookup.

    Returns:
        The token, or an empty string when none is configured.
    """
    if token:
        return str(token).strip()

    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value.strip()

    try:
        with open(API_KEY_PATH) as handle:
            return handle.read().strip()
    except FileNotFoundError:
        return ""
