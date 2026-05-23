from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from typing import Any

import httpx


class AiProxyService:
    """Routes chat requests through backend, never exposing API keys to the client."""

    PROVIDER_OPENROUTER = "openrouter"
    PROVIDER_GROK = "grok"
    PROVIDER_DEEPSEEK = "deepseek"

    _provider_configs: dict[str, dict[str, str]] = {}
    _rate_limits: dict[str, list[float]] = defaultdict(list)
    _last_provider_warning: float = 0

    MAX_RETRIES = 2
    RATE_LIMIT_PER_USER = 30  # requests per minute
    RATE_LIMIT_WINDOW = 60  # seconds

    def __init__(self) -> None:
        self._init_providers()
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _init_providers(self) -> None:
        configs: dict[str, dict[str, str]] = {}

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if openrouter_key:
            configs[self.PROVIDER_OPENROUTER] = {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": openrouter_key,
                "model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
            }

        grok_key = os.environ.get("GROK_API_KEY", "")
        if grok_key:
            configs[self.PROVIDER_GROK] = {
                "url": "https://api.x.ai/v1/chat/completions",
                "key": grok_key,
                "model": os.environ.get("GROK_MODEL", "grok-2-latest"),
            }

        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            configs[self.PROVIDER_DEEPSEEK] = {
                "url": "https://api.deepseek.com/v1/chat/completions",
                "key": deepseek_key,
                "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            }

        self._provider_configs = configs

    async def chat(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | None:
        if not self._check_rate_limit(user_id):
            return {
                "error": "rate_limited",
                "message": "Too many requests. Please wait before sending another request.",
            }

        if not self._provider_configs:
            return {
                "error": "no_providers",
                "message": "No AI providers configured on the server.",
            }

        provider_order = [self.PROVIDER_OPENROUTER, self.PROVIDER_GROK, self.PROVIDER_DEEPSEEK]
        last_error: str | None = None

        for provider_name in provider_order:
            config = self._provider_configs.get(provider_name)
            if config is None:
                continue

            try:
                body: dict[str, Any] = {
                    "model": config["model"],
                    "messages": messages,
                    "stream": stream,
                }
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                if temperature is not None:
                    body["temperature"] = temperature

                headers = {
                    "Authorization": f"Bearer {config['key']}",
                    "Content-Type": "application/json",
                }
                if provider_name == self.PROVIDER_OPENROUTER:
                    headers["HTTP-Referer"] = os.environ.get(
                        "SITE_URL", "https://axon.app"
                    )
                    headers["X-Title"] = "Axon"

                response = await self._http.post(
                    config["url"],
                    json=body,
                    headers=headers,
                )

                if response.status_code == 429:
                    last_error = f"{provider_name}: rate limited"
                    continue

                if response.status_code >= 500:
                    last_error = f"{provider_name}: server error {response.status_code}"
                    continue

                response.raise_for_status()
                data = response.json()
                return {
                    "provider": provider_name,
                    "model": config["model"],
                    "choices": data.get("choices", []),
                    "usage": data.get("usage", {}),
                }

            except httpx.TimeoutException:
                last_error = f"{provider_name}: timeout"
                continue
            except Exception as exc:
                last_error = f"{provider_name}: {exc}"
                continue

        return {
            "error": "all_providers_failed",
            "message": f"All AI providers failed. Last error: {last_error}",
        }

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        if not self._check_rate_limit(user_id):
            yield f"data: {json.dumps({'error': 'rate_limited', 'message': 'Too many requests'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if not self._provider_configs:
            yield f"data: {json.dumps({'error': 'no_providers', 'message': 'No AI providers configured'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        provider_order = [self.PROVIDER_OPENROUTER, self.PROVIDER_GROK, self.PROVIDER_DEEPSEEK]

        for provider_name in provider_order:
            config = self._provider_configs.get(provider_name)
            if config is None:
                continue

            try:
                body: dict[str, Any] = {
                    "model": config["model"],
                    "messages": messages,
                    "stream": True,
                }
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                if temperature is not None:
                    body["temperature"] = temperature

                headers = {
                    "Authorization": f"Bearer {config['key']}",
                    "Content-Type": "application/json",
                }
                if provider_name == self.PROVIDER_OPENROUTER:
                    headers["HTTP-Referer"] = os.environ.get(
                        "SITE_URL", "https://axon.app"
                    )
                    headers["X-Title"] = "Axon"

                async with self._http.stream(
                    "POST", config["url"], json=body, headers=headers
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"data: {json.dumps({'error': f'{provider_name}: HTTP {response.status_code}', 'detail': error_text.decode(errors='replace')[:500]})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield f"{line}\n"
                            if line.strip() == "data: [DONE]":
                                return

                return

            except (httpx.TimeoutException, httpx.RequestError) as exc:
                yield f"data: {json.dumps({'warning': f'{provider_name} failed, trying next', 'detail': str(exc)})}\n\n"
                continue

        yield f"data: {json.dumps({'error': 'all_providers_failed'})}\n\n"
        yield "data: [DONE]\n\n"

    def _check_rate_limit(self, user_id: str) -> bool:
        now = time.time()
        window_start = now - self.RATE_LIMIT_WINDOW
        self._rate_limits[user_id] = [
            t for t in self._rate_limits[user_id] if t > window_start
        ]
        if len(self._rate_limits[user_id]) >= self.RATE_LIMIT_PER_USER:
            return False
        self._rate_limits[user_id].append(now)
        return True

    def get_status(self) -> dict[str, Any]:
        return {
            "providers": list(self._provider_configs.keys()),
            "provider_count": len(self._provider_configs),
            "rate_limited_users": len(
                [u for u, t in self._rate_limits.items() if len(t) > 0]
            ),
        }

    async def close(self) -> None:
        await self._http.aclose()
