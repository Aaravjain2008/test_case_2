"""
app/llm/client.py
-----------------
Modular LLM client abstraction.

Supports:
  - Ollama  (local, no API key, LLM_PROVIDER=ollama)

Architecture:
  BaseLLMClient defines the interface.
  Concrete implementations are selected by LLM_PROVIDER env var.
  Adding a new provider requires only a new subclass.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from app import get_logger

logger = get_logger(__name__)

_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    All providers must implement ``complete(prompt)``.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response text.

        Args:
            prompt: The full prompt string (system instructions + context + question).

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If the LLM API call fails.
        """


# ── Ollama implementation ──────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """LLM client backed by a local Ollama server.

    Requires Ollama to be running: https://ollama.ai
    Pull the model first: ``ollama pull llama3.2``

    Attributes:
        model:    Ollama model tag (e.g. "llama3.2", "mistral").
        base_url: Ollama server URL.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialise the Ollama client.

        Args:
            model:    Model tag.  Defaults to ``OLLAMA_MODEL`` env var.
            base_url: Server URL.  Defaults to ``OLLAMA_BASE_URL`` env var.
        """
        self.model: str = model or os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        self.base_url: str = base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)
        logger.info(
            "OllamaClient initialised: model='%s', url='%s'.",
            self.model,
            self.base_url,
        )

    def complete(self, prompt: str) -> str:
        """Send *prompt* to Ollama and return the response.

        Args:
            prompt: Full grounded prompt.

        Returns:
            Model's text response.

        Raises:
            RuntimeError: If Ollama is not reachable or returns an error.
            ImportError:  If the ``ollama`` package is not installed.
        """
        try:
            import ollama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is not installed. Run: pip install ollama"
            ) from exc

        logger.info("Sending request to Ollama (model='%s') …", self.model)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},  # deterministic for grounded answers
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed for model '{self.model}' at '{self.base_url}'.\n"
                f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {self.model}`).\n"
                f"Error: {exc}"
            ) from exc

        text: str = response["message"]["content"]
        logger.info("Ollama response received (%d characters).", len(text))
        return text


# ── Mock implementation ────────────────────────────────────────────────────────

class MockClient(BaseLLMClient):
    """LLM client that returns a mock grounded response based on the prompt's context.
    
    Used for testing the pipeline flow without a running Ollama server.
    """

    def complete(self, prompt: str) -> str:
        logger.info("MockClient: parsing context to generate a grounded answer...")
        
        # Simple extraction of key facts from prompt to make mock answer feel real
        context_lower = prompt.lower()
        
        if "forgot password" in context_lower or "reset my password" in context_lower:
            answer = (
                "To reset your password:\n"
                "1. Click 'Forgot Password' on the login page.\n"
                "2. Enter your registered email address.\n"
                "3. Click the password reset link sent to your email within 5 minutes.\n"
                "4. Choose a new password of at least 8 characters with at least one number."
            )
        elif "upgrade" in context_lower or "subscription" in context_lower:
            answer = (
                "To upgrade your subscription plan:\n"
                "1. Go to Settings → Subscription.\n"
                "2. Choose the plan you want (Pro at ₹499/mo or Team at ₹1499/mo).\n"
                "3. Enter payment details and confirm the upgrade."
            )
        elif "api access" in context_lower or "token" in context_lower:
            answer = (
                "To generate an API token:\n"
                "1. Go to Settings → Developer Settings.\n"
                "2. Click 'Generate New Token'.\n"
                "3. Copy the token immediately; it will not be shown again."
            )
        else:
            answer = (
                "The requested topic was found in the retrieved documents, but "
                "the Mock LLM has no pre-configured answer template for it."
            )
            
        return f"{answer}\n\n[Note: This is a grounded simulation response generated by MockClient]"


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client() -> BaseLLMClient:
    """Factory function that returns the configured LLM client.

    The provider is selected by the ``LLM_PROVIDER`` environment variable:
      - ``"ollama"`` → OllamaClient
      - ``"mock"``   → MockClient (for running without local server)

    Returns:
        A configured BaseLLMClient subclass instance.

    Raises:
        ValueError: If LLM_PROVIDER is set to an unsupported value.
        RuntimeError: If required configuration is missing.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

    if provider == "ollama":
        return OllamaClient()
    if provider == "mock":
        return MockClient()

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{provider}'.\n"
        f"Supported providers: ollama, mock\n"
        f"Check your .env file."
    )

