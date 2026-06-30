"""Model wrappers for AR and Diffusion language models.

Keep package imports lightweight so scripts can import a specific model
wrapper without pulling in optional backends at import time.
"""

from .base import LanguageModel

__all__ = ["LanguageModel"]
