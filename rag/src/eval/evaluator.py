"""Evaluator using LLM-as-Judge approach.

Scores RAG answers on configurable criteria defined in eval/criteria.py.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from src.config import Config
from src.retrieval.llm_client import LlmClient
from src.eval.criteria import Criterion, DEFAULT_CRITERIA, build_eval_prompt

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Handles both raw JSON and ```json fenced code blocks.
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text.strip())


def _validate_eval_result(data: dict, criteria: list[Criterion]) -> dict:
    """Validate and normalize the evaluation result structure."""
    result = {}
    for criterion in criteria:
        if criterion.name not in data:
            raise ValueError(f"Missing criterion: {criterion.name}")
        entry = data[criterion.name]
        if not isinstance(entry, dict) or "score" not in entry or "reason" not in entry:
            raise ValueError(
                f"Criterion '{criterion.name}' must have 'score' and 'reason' keys"
            )
        score = int(entry["score"])
        if not 1 <= score <= 5:
            raise ValueError(
                f"Score for '{criterion.name}' must be between 1 and 5, got {score}"
            )
        result[criterion.name] = {"score": score, "reason": str(entry["reason"])}
    return result


class Evaluator:
    """Evaluates RAG answers using an LLM-as-Judge approach.

    Args:
        config: Application config.
        eval_model: Override the LLM model used for evaluation.
        criteria: Evaluation criteria to use. Defaults to DEFAULT_CRITERIA.
                  Pass a custom list to change what dimensions are scored.
    """

    def __init__(
        self,
        config: Config,
        eval_model: str | None = None,
        criteria: list[Criterion] | None = None,
    ) -> None:
        if eval_model is not None:
            eval_config = replace(config, llm_model_name=eval_model)
        else:
            eval_config = config
        self._client = LlmClient(eval_config)
        self._criteria = criteria if criteria is not None else DEFAULT_CRITERIA

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        """Evaluate an answer against the given question and context.

        Returns a dict with scores and reasons for each criterion:
        {"relevance": {"score": 4, "reason": "..."}, ...}
        """
        prompt = build_eval_prompt(self._criteria, question, context, answer)
        logger.debug("Sending evaluation request to LLM")
        response = self._client.chat([{"role": "user", "content": prompt}])

        try:
            raw = _parse_json_response(response)
            result = _validate_eval_result(raw, self._criteria)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse evaluation response: %s", e)
            logger.error("Raw response was: %s", response)
            raise ValueError(f"Failed to parse LLM evaluation response: {e}") from e

        logger.debug("Evaluation complete: %s", result)
        return result
