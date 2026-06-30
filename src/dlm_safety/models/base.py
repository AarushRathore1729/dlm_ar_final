"""
Abstract base class for language models.

All model wrappers (AR and Diffusion) implement this interface.
It provides the two operations needed by the evaluation pipeline:
  1. Generate responses given prompts
  2. Compute log P(response | prompt)
"""

from abc import ABC, abstractmethod
from typing import Tuple


class LanguageModel(ABC):
    """Base class that both AR and Diffusion LM wrappers must implement."""

    @abstractmethod
    def get_responses(
        self,
        prompts: list[str],
        batched: bool = False,
        **kwargs,
    ) -> Tuple[list[list[int]], list[str]]:
        """Generate responses for a list of prompts.

        Args:
            prompts: List of input prompt strings.
            batched: Whether to process prompts as a batch.
            **kwargs: Generation config (max_new_tokens, etc.).

        Returns:
            Tuple of:
                - response_tokens: List of token ID lists, one per prompt.
                - responses: List of decoded response strings.
        """
        ...

    @abstractmethod
    def compute_log_probabilities(
        self,
        prompts: list[str],
        responses: list[list[int]],
        batched: bool = False,
    ) -> list[float]:
        """Compute log P(response | prompt) for each (prompt, response) pair.

        Args:
            prompts: List of prompt strings.
            responses: List of token ID lists (one per prompt).
            batched: Whether to process in batch.

        Returns:
            List of log-probability floats, one per (prompt, response) pair.
        """
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this model is on (e.g. 'cpu', 'cuda:0')."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable model name for logging."""
        ...
