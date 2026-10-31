"""
tests/test_llm_client.py
------------------------
Tests for LLM client implementations and the provider factory.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.llm.client import MockClient, OllamaClient, get_llm_client


def _install_fake_ollama(monkeypatch, response=None, error=None):
    """Install a fake `ollama` module and return (module, client_instance)."""
    fake_client = MagicMock()
    if error is not None:
        fake_client.chat.side_effect = error
    else:
        fake_client.chat.return_value = response or {"message": {"content": "hello"}}
    fake_module = SimpleNamespace(Client=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    return fake_module, fake_client


class TestOllamaClient:
    def test_uses_configured_base_url(self, monkeypatch) -> None:
        module, _ = _install_fake_ollama(monkeypatch)
        OllamaClient(model="m", base_url="http://remote:11434").complete("hi")
        module.Client.assert_called_once_with(host="http://remote:11434")

    def test_base_url_read_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://envhost:1234")
        module, _ = _install_fake_ollama(monkeypatch)
        OllamaClient().complete("hi")
        module.Client.assert_called_once_with(host="http://envhost:1234")

    def test_returns_message_content(self, monkeypatch) -> None:
        _install_fake_ollama(monkeypatch, response={"message": {"content": "answer"}})
        assert OllamaClient(model="m").complete("q") == "answer"

    def test_wraps_errors_in_runtime_error(self, monkeypatch) -> None:
        _install_fake_ollama(monkeypatch, error=ConnectionError("refused"))
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            OllamaClient(model="m").complete("q")


class TestFactory:
    def test_default_provider_is_ollama(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert isinstance(get_llm_client(), OllamaClient)

    def test_mock_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert isinstance(get_llm_client(), MockClient)

    def test_unknown_provider_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "nope")
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            get_llm_client()
