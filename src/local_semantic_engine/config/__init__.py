"""Typed runtime configuration."""

from local_semantic_engine.config.loader import load_settings
from local_semantic_engine.config.models import AppSettings

__all__ = ["AppSettings", "load_settings"]
