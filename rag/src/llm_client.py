from __future__ import annotations

import litellm

from src.config import Config


class LlmClient:
    def __init__(self, config: Config) -> None:
        self._config = config

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = litellm.completion(
            model=self._config.llm_model_name,
            messages=messages,
            api_key=self._config.gemini_api_key,
        )
        return response.choices[0].message.content
