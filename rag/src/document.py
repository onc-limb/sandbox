"""Document dataclass module.

Provides a lightweight Document class to replace LangChain's Document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document with text content and metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
