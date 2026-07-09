import httpx
import pytest
import respx
from httpx import Response

import sources._openreview_auth as auth_module
from sources._openreview_auth import openreview_auth_headers


@pytest.fixture(autouse=True)
def _reset_cached_token():
    auth_module._cached_token = None
    yield
    auth_module._cached_token = None


@pytest.mark.asyncio
async def test_returns_empty_headers_when_credentials_unset(monkeypatch):
    monkeypatch.delenv("OPENREVIEW_EMAIL", raising=False)
    monkeypatch.delenv("OPENREVIEW_PASSWORD", raising=False)
    async with httpx.AsyncClient() as client:
        headers = await openreview_auth_headers(client)
    assert headers == {}


@respx.mock
@pytest.mark.asyncio
async def test_logs_in_and_returns_bearer_token_when_credentials_set(monkeypatch):
    monkeypatch.setenv("OPENREVIEW_EMAIL", "user@example.com")
    monkeypatch.setenv("OPENREVIEW_PASSWORD", "secret")
    route = respx.post("https://api2.openreview.net/login").mock(
        return_value=Response(200, json={"token": "tok-123"})
    )
    async with httpx.AsyncClient() as client:
        headers = await openreview_auth_headers(client)
    assert headers == {"Authorization": "Bearer tok-123"}
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_caches_token_across_calls(monkeypatch):
    monkeypatch.setenv("OPENREVIEW_EMAIL", "user@example.com")
    monkeypatch.setenv("OPENREVIEW_PASSWORD", "secret")
    route = respx.post("https://api2.openreview.net/login").mock(
        return_value=Response(200, json={"token": "tok-123"})
    )
    async with httpx.AsyncClient() as client:
        await openreview_auth_headers(client)
        headers = await openreview_auth_headers(client)
    assert headers == {"Authorization": "Bearer tok-123"}
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_raises_when_login_response_has_no_token(monkeypatch):
    monkeypatch.setenv("OPENREVIEW_EMAIL", "user@example.com")
    monkeypatch.setenv("OPENREVIEW_PASSWORD", "secret")
    respx.post("https://api2.openreview.net/login").mock(return_value=Response(200, json={}))
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError):
            await openreview_auth_headers(client)
