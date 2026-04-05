"""Evaluation module using LLM-as-Judge approach.

Scores RAG answers on four criteria: relevance, faithfulness,
completeness, and conciseness.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from src.config import Config
from src.llm_client import LlmClient

logger = logging.getLogger(__name__)

_EVAL_PROMPT = """\
You are an impartial judge evaluating a RAG (Retrieval-Augmented Generation) system's answer.

## Inputs
- **Question**: The user's question.
- **Context**: The retrieved source documents provided to the system.
- **Answer**: The system's generated answer.

## Evaluation Criteria
Score each criterion from 1 (worst) to 5 (best):

1. **relevance**: How well does the answer address the question?
2. **faithfulness**: Is the answer faithful to the source context? (No hallucination)
3. **completeness**: Does the answer cover all important aspects of the question?
4. **conciseness**: Is the answer concise without unnecessary information?

## Output Format
Return ONLY a JSON object (no other text) with this exact structure:
```json
{{
  "relevance": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "faithfulness": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "completeness": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "conciseness": {{"score": <1-5>, "reason": "<brief explanation>"}}
}}
```

## Inputs to Evaluate

**Question:**
{question}

**Context:**
{context}

**Answer:**
{answer}
"""

_CRITERIA = ("relevance", "faithfulness", "completeness", "conciseness")


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Handles both raw JSON and ```json fenced code blocks.
    """
    # Try to extract from ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    return json.loads(text.strip())


def _validate_eval_result(data: dict) -> dict:
    """Validate and normalize the evaluation result structure."""
    result = {}
    for criterion in _CRITERIA:
        if criterion not in data:
            raise ValueError(f"Missing criterion: {criterion}")
        entry = data[criterion]
        if not isinstance(entry, dict) or "score" not in entry or "reason" not in entry:
            raise ValueError(
                f"Criterion '{criterion}' must have 'score' and 'reason' keys"
            )
        score = int(entry["score"])
        if not 1 <= score <= 5:
            raise ValueError(
                f"Score for '{criterion}' must be between 1 and 5, got {score}"
            )
        result[criterion] = {"score": score, "reason": str(entry["reason"])}
    return result


class Evaluator:
    """Evaluates RAG answers using an LLM-as-Judge approach."""

    def __init__(self, config: Config, eval_model: str | None = None) -> None:
        if eval_model is not None:
            eval_config = replace(config, llm_model_name=eval_model)
        else:
            eval_config = config
        self._client = LlmClient(eval_config)

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        """Evaluate an answer against the given question and context.

        Returns a dict with scores and reasons for each criterion:
        {"relevance": {"score": 4, "reason": "..."}, ...}
        """
        prompt = _EVAL_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
        )
        logger.debug("Sending evaluation request to LLM")
        response = self._client.chat([{"role": "user", "content": prompt}])

        try:
            raw = _parse_json_response(response)
            result = _validate_eval_result(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse evaluation response: %s", e)
            logger.error("Raw response was: %s", response)
            raise ValueError(f"Failed to parse LLM evaluation response: {e}") from e

        logger.debug("Evaluation complete: %s", result)
        return result
