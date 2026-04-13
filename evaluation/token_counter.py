"""
Token estimation utilities.
Uses tiktoken for accurate counting where available, with a character-based
fallback so the app works even without an API key.
"""

from __future__ import annotations

import re

from backend.logger import logger

try:
    import tiktoken

    _ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}

    def _get_encoder(model_id: str) -> tiktoken.Encoding:
        if model_id not in _ENCODER_CACHE:
            try:
                _ENCODER_CACHE[model_id] = tiktoken.encoding_for_model(model_id)
            except KeyError:
                _ENCODER_CACHE[model_id] = tiktoken.get_encoding("cl100k_base")
        return _ENCODER_CACHE[model_id]

    def count_tokens(text: str, model_id: str = "gpt-4o") -> int:
        enc = _get_encoder(model_id)
        return len(enc.encode(text))

    TIKTOKEN_AVAILABLE = True

except ImportError:
    logger.warning("tiktoken not installed — using character-based token estimation")
    TIKTOKEN_AVAILABLE = False

    def count_tokens(text: str, model_id: str = "gpt-4o") -> int:  # type: ignore[misc]
        # ~4 chars per token is a reasonable heuristic
        return max(1, len(text) // 4)


def estimate_eval_tokens(
    system_prompt: str,
    data_text: str,
    question: str,
    expected_answer_words: int = 150,
) -> tuple[int, int]:
    """
    Estimate (input_tokens, output_tokens) for a single evaluation call.

    Returns:
        (input_tokens, output_tokens)
    """
    full_input = f"{system_prompt}\n\n{data_text}\n\nQuestion: {question}"
    input_tokens = count_tokens(full_input)
    # Rough output estimate: ~1.3 tokens per word
    output_tokens = int(expected_answer_words * 1.3)
    return input_tokens, output_tokens


def estimate_batch_tokens(
    system_prompt: str,
    data_text: str,
    questions: list[str],
    expected_answer_words: int = 150,
) -> tuple[int, int]:
    """Sum up token estimates for a batch of questions."""
    total_input = 0
    total_output = 0
    for q in questions:
        inp, out = estimate_eval_tokens(system_prompt, data_text, q, expected_answer_words)
        total_input += inp
        total_output += out
    return total_input, total_output
