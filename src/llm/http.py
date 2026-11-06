"""Small HTTP helper shared by API-backed model providers."""

import requests

from src.llm.base import LLMError


def post_json(
    url: str, payload: dict, headers: dict[str, str] | None = None, timeout: int = 90
) -> dict:
    """POST JSON with friendly failures and without leaking API credentials."""
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.Timeout as exc:
        raise LLMError("Model request timed out. Try again.") from exc
    except requests.RequestException as exc:
        raise LLMError("Cannot connect to the model service. Check your connection.") from exc

    if response.status_code == 429:
        raise LLMError("Free-tier rate limit reached. Wait and retry.")
    if response.status_code in (401, 403):
        raise LLMError("Authentication failed. Check your API key or account permissions.")
    if response.status_code >= 400:
        raise LLMError(f"Model service returned HTTP {response.status_code}.")
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError("Model service returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise LLMError("Model service returned an unexpected response.")
    return data
