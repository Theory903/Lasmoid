"""
Lasmoid — reward.py
===================
Reasoning self-evolution reward function and text normalisation helpers.
"""

import re
from typing import Optional


def _extract_answer_text(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>", 1)[-1].strip()
    if "</Summary>" in response:
        return response.split("</Summary>", 1)[-1].strip()
    return response.strip()


def _normalise_answer(text: str) -> str:
    text = _extract_answer_text(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.+\\-/% ]", "", text)
    return text


def reasoning_self_evolution_reward(
    response: str, gt_answer: Optional[str] = None
) -> float:
    """
    Reward function for reasoning self-evolution — scores structure,
    reasoning depth, scientific markers, and answer accuracy.

    Raises
    ------
    TypeError
        If ``response`` is not a string (callers should use ``safe_reward`` to
        catch malformed input rather than calling this directly from untrusted
        completions).
    """
    if not isinstance(response, str):
        raise TypeError(
            f"reasoning_self_evolution_reward expects a str, got {type(response).__name__}"
        )

    reward = 0.0

    has_think = "<think>" in response and "</think>" in response
    has_parallel = "<Parallel>" in response and "</Parallel>" in response
    answer = _extract_answer_text(response)

    if has_think:
        reward += 1.0
        trace = response.split("<think>", 1)[-1].split("</think>", 1)[0]
    elif has_parallel:
        reward += 0.8
        trace = response.split("<Parallel>", 1)[-1].split("</Parallel>", 1)[0]
    else:
        trace = response

    trace_len = len(trace.strip())
    if 150 <= trace_len <= 2000:
        reward += 0.8
    elif trace_len > 0:
        reward += 0.3

    lower_trace = trace.lower()
    logic_markers = [
        "verify",
        "hypothesis",
        "empirical",
        "deduction",
        "axiom",
        "topology",
        "manifold",
        "synthesis",
    ]
    scientific_markers = [
        "sequence",
        "fold",
        "energy",
        "minimum",
        "gradient",
        "stochastic",
        "converge",
        "optimization",
    ]

    found_logic = sum(1 for m in logic_markers if m in lower_trace)
    found_sci = sum(1 for m in scientific_markers if m in lower_trace)

    reward += min(0.6, found_logic * 0.15)
    reward += min(0.6, found_sci * 0.15)

    if any(
        x in lower_trace
        for x in ["collaborate", "consensus", "delegate", "orchestrate"]
    ):
        reward += 0.4

    if answer:
        reward += 0.5
    if gt_answer:
        gt = _normalise_answer(gt_answer)
        pred = _normalise_answer(answer if answer else response)
        if gt and (gt in pred or pred in gt):
            reward += 3.0

    words = re.findall(r"\b\w+\b", response.lower())
    if len(words) > 30:
        repeats = sum(1 for a, b, c in zip(words, words[1:], words[2:]) if a == b == c)
        reward -= min(2.0, repeats * 0.4)

    if len(response) > 8000:
        reward -= 1.0

    return float(reward)
