"""Optional OpenReview account authentication.

OpenReview started intermittently challenge-walling anonymous API requests
around 2026-06-30 (Cloudflare bot check). Authenticated requests use a
different trust tier and are not subject to the same anonymous-client
challenge. If `OPENREVIEW_EMAIL` / `OPENREVIEW_PASSWORD` are set, callers
get a bearer token to attach to every request; if unset, callers fall back
to the previous anonymous behavior unchanged.

Getting a token is a login call, not a bypass of any protection — this is
OpenReview's own documented auth flow (the same one `openreview-py` uses).
"""

from __future__ import annotations

import os

import httpx

LOGIN_URL = "https://api2.openreview.net/login"

_cached_token: str | None = None


async def openreview_auth_headers(client: httpx.AsyncClient) -> dict[str, str]:
    """Return `{"Authorization": "Bearer <token>"}` if OPENREVIEW_EMAIL and
    OPENREVIEW_PASSWORD are set in the environment, else `{}` (anonymous,
    same behavior as before this module existed). The token is cached for
    the process lifetime — one login per run, not per request."""
    global _cached_token
    if _cached_token is not None:
        return {"Authorization": f"Bearer {_cached_token}"}

    email = os.environ.get("OPENREVIEW_EMAIL")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not email or not password:
        return {}

    resp = await client.post(LOGIN_URL, json={"id": email, "password": password})
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("openreview login succeeded but response had no 'token' field")
    _cached_token = token
    return {"Authorization": f"Bearer {_cached_token}"}
