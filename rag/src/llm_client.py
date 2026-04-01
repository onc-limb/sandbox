"""LLM client module.

Provides chat function using LiteLLM for Gemini API calls.
"""

from __future__ import annotations

import litellm

from src.config import Config, get_config


def chat(
    messages: list[dict[str, str]],
    config: Config | None = None,
) -> str:
    """Send messages to the LLM and return the response text.

    Args:
        messages: List of message dicts with "role" and "content" keys.
            Example: [{"role": "user", "content": "Hello"}]
        config: Application config. Uses default if None.

    Returns:
        The assistant's response text.
    """
    if config is None:
        config = get_config()

    response = litellm.completion(
        model=config.llm_model_name,
        messages=messages,
        api_key=config.gemini_api_key,
    )
    return response.choices[0].message.content
