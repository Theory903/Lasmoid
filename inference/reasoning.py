"""
Lasmoid — reasoning.py
======================================================================
Tool-augmented propose→verify→refine reasoning + a *soft, non-deterministic*
scientific-domain prior for the DomainCortexRouter, plus a Socratic
self-questioning loop for human-like concept learning.

Design philosophy
-----------------
The domain prior here is deliberately a **soft, low-strength, optionally
stochastic hint** — never a hard override. The cortex router (noisy top-k
gating) is what actually *learns* which experts to call from data; this module
only nudges. So the number of "domains" is not fixed in spirit: keyword cues are
a weak Bayesian prior that washes out as the learned router specialises.

Performance
-----------
* Per-domain keyword matchers are compiled **once** at import (word-boundary
  regex — fixes false-substring hits like "ion" in "function").
* Tool-call extraction caches the optional DSML parser import and pre-compiles
  its regex.
* The controller **caches tool dispatches** (same call never runs twice) and
  supports parallel best-of-N proposals — quality without redundant compute.

All public APIs and behaviours are preserved; every addition is opt-in.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - torch always present here
    torch = None  # type: ignore

try:
    from .tools import ToolRegistry, ToolResult
except ImportError:
    from tools import ToolRegistry, ToolResult


# ══════════════════════════════════════════════════════════════════════
# DOMAIN PRIOR  (soft, non-deterministic — a hint, not a decision)
# ══════════════════════════════════════════════════════════════════════

# Keyword cues per default domain column (see ModelArgs.domain_names).
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "general": [],
    "mathematics": ["theorem", "integral", "matrix", "probability", "equation",
                    "proof", "derivative", "algebra", "geometry", "topology"],
    "physics": ["quantum", "relativity", "momentum", "thermodynamic", "voltage",
                "particle", "energy", "force", "wavelength", "entropy"],
    "chemistry": ["molecule", "reaction", "compound", "acid", "bond", "catalyst",
                  "stoichiometry", "organic", "ion", "ph"],
    "biology_medical": ["protein", "gene", "cell", "patient", "diagnosis", "enzyme",
                        "dna", "clinical", "dosage", "pathway", "disease"],
    "astronomy": ["galaxy", "orbit", "redshift", "telescope", "exoplanet", "stellar",
                  "cosmic", "luminosity", "nebula", "spectra"],
    "computer_science": ["algorithm", "complexity", "compiler", "neural", "gradient",
                         "dataset", "function", "runtime", "tensor", "graph"],
    "data_analysis": ["regression", "correlation", "variance", "cluster", "p-value",
                      "histogram", "distribution", "outlier", "pca", "dataframe"],
}

# Compile one word-boundary matcher per domain *once* (avoids substring false
# positives such as "ion" matching "function", and is far faster than per-call
# str.count loops).
_DOMAIN_MATCHERS: Dict[str, "re.Pattern[str]"] = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b")
    for name, kws in _DOMAIN_KEYWORDS.items()
    if kws
}


def detect_domain_scores(text: str, domain_names: List[str]) -> Dict[str, float]:
    """Keyword-evidence score per domain, normalised to [0, 1] (max-scaled)."""
    low = text.lower()
    raw = {
        name: float(len(_DOMAIN_MATCHERS[name].findall(low))) if name in _DOMAIN_MATCHERS else 0.0
        for name in domain_names
    }
    mx = max(raw.values()) if raw else 0.0
    if mx > 0:
        return {k: v / mx for k, v in raw.items()}
    return raw


def build_domain_steer(
    text: str,
    domain_names: List[str],
    strength: float = 2.0,
    temperature: float = 1.0,
    stochastic: bool = False,
    device: Any = None,
) -> "torch.Tensor":
    """Soft additive prior over domain columns for the DomainCortexRouter.

    This is intentionally *weak* — the learned router dominates. ``strength``
    scales the nudge; ``stochastic=True`` adds Gumbel noise so the prior is
    non-deterministic (different reasonable columns can win run-to-run, letting
    the model explore connections rather than locking to fixed keyword domains).
    'general' keeps a small floor so it always stays reachable.
    """
    assert torch is not None
    scores = detect_domain_scores(text, domain_names)
    vec = torch.tensor([scores[name] for name in domain_names], dtype=torch.float32)
    steer = (vec / max(temperature, 1e-6)) * strength
    if stochastic:
        # Gumbel(0,1) noise → non-deterministic soft selection.
        g = -torch.log(-torch.log(torch.rand_like(steer).clamp_min(1e-9)).clamp_min(1e-9))
        steer = steer + g
    if "general" in domain_names:
        gi = domain_names.index("general")
        steer[gi] = max(steer[gi].item(), 0.5)  # keep general reachable
    if device is not None:
        steer = steer.to(device)
    return steer


# ══════════════════════════════════════════════════════════════════════
# VERIFIER
# ══════════════════════════════════════════════════════════════════════

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"\w+")
_REASON_MARKERS = re.compile(
    r"\b(because|therefore|thus|hence|since|first|then|so that|implies|follows)\b"
)


def extract_final_answer(text: str) -> Optional[str]:
    """Extract a \\boxed{...} answer, else the last number, else None."""
    m = list(_BOXED_RE.finditer(text))
    if m:
        return m[-1].group(1).strip()
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def _repetition_penalty(text: str) -> float:
    """Fraction of repeated bigrams — a cheap degeneration detector in [0, 1)."""
    toks = _WORD_RE.findall(text.lower())
    if len(toks) < 4:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    uniq = len(set(bigrams))
    return 1.0 - uniq / len(bigrams)


def heuristic_score(answer: str, brevity_bonus: float = 500.0) -> float:
    """Reward a final answer + reasoning structure; penalise degeneration.

    Base term adapted from reasoning-from-scratch ch05.heuristic_score, extended
    with a small, bounded reasoning-quality term and a repetition penalty so the
    verifier prefers *well-reasoned* answers, not just short ones.
    """
    score = 0.0
    if _BOXED_RE.search(answer):
        score += 2.0
    elif _NUM_RE.search(answer):
        score += 1.0
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    # Bounded reasoning-structure reward (caps at +0.3).
    score += min(0.3, 0.1 * len(_REASON_MARKERS.findall(answer.lower())))
    # Degeneration penalty (repeated bigrams).
    score -= 0.5 * _repetition_penalty(answer)
    return score


class Verifier:
    """Scores candidate answers; prefers data-grounded (tool) evidence."""

    def __init__(self, score_fn: Optional[Callable[[str], float]] = None):
        self.score_fn = score_fn or heuristic_score

    def score(self, answer: str, tool_results: Optional[List[ToolResult]] = None) -> float:
        s = self.score_fn(answer)
        if tool_results:
            ok = sum(1 for r in tool_results if r.status == "success")
            s += 0.5 * ok  # reward grounding in successful tool evidence
        return s


# ══════════════════════════════════════════════════════════════════════
# TOOL-CALL EXTRACTION
# ══════════════════════════════════════════════════════════════════════

_FENCED_JSON_RE = re.compile(r"```(?:tool|json)?\s*(\{.*?\})\s*```", re.DOTALL)


@lru_cache(maxsize=1)
def _dsml_parser():
    """Locate the optional DSML tool-call parser once (cached)."""
    try:
        try:
            from ..encoding.encoding_lasmoid import parse_tool_calls  # type: ignore
        except Exception:
            from encoding.encoding_lasmoid import parse_tool_calls  # type: ignore
        return parse_tool_calls
    except Exception:
        return None


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Best-effort tool-call extraction.

    Prefers the Lasmoid DSML parser; falls back to fenced ```tool {json}```
    blocks of the form {"name": ..., "arguments": {...}}.
    """
    if "tool_calls" in text:
        parse = _dsml_parser()
        if parse is not None:
            try:
                _, _, dsml_calls = parse(0, text)
                if dsml_calls:
                    return dsml_calls
            except Exception:
                pass
    calls: List[Dict[str, Any]] = []
    for block in _FENCED_JSON_RE.findall(text):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "name" in obj:
            calls.append(
                {"function": {"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))}}
            )
    return calls


# ══════════════════════════════════════════════════════════════════════
# STRUCTURED REASONING PARSING (Req 12.1, 12.2, 12.3)
# ══════════════════════════════════════════════════════════════════════

# Common reasoning markup delimiters (supports <think>...</think>, <reasoning>...</reasoning>,
# and markdown-style ### Reasoning / ### Answer blocks).
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_REASONING_TAG_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_MD_REASONING_RE = re.compile(
    r"(?:^|\n)#+\s*(?:Reasoning|Thinking|Analysis)\s*\n(.*?)(?=\n#+\s*(?:Answer|Final|Result|Conclusion)|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_MD_ANSWER_RE = re.compile(
    r"(?:^|\n)#+\s*(?:Answer|Final|Result|Conclusion)\s*\n(.*)",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedReasoning:
    """Structured result of reasoning completion parsing.

    Fields:
        reasoning_segments: list of extracted reasoning/thinking text blocks.
        final_answer: the final answer text, or None if not found.
        tool_calls: list of parsed tool calls, each with 'name' (str) and
                    'arguments' (dict) as separate, already-parsed fields.
        parse_error: None on success; a string describing the parse failure
                     on malformed input (never raises).
    """
    reasoning_segments: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    parse_error: Optional[str] = None


def parse_completion(text: str) -> ParsedReasoning:
    """Parse a model completion into reasoning segments, final answer, and tool calls.

    Extracts reasoning segments and final answer as SEPARATE fields (Req 12.1).
    Parses tool-call payloads into name + argument map (Req 12.2).
    Returns a parse-error indicator on malformed input instead of raising (Req 12.3).

    Supports multiple markup formats:
      - XML-style: <think>...</think>, <reasoning>...</reasoning>, <answer>...</answer>
      - Markdown-style: ### Reasoning / ### Answer sections
      - \\boxed{...} final answers
      - Fenced JSON tool calls: ```tool {"name": ..., "arguments": {...}}```
    """
    if not isinstance(text, str):
        return ParsedReasoning(parse_error=f"expected str input, got {type(text).__name__}")

    reasoning_segments: List[str] = []
    final_answer: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    parse_error: Optional[str] = None

    try:
        # ── Extract reasoning segments ──
        # Try XML-style <think>...</think> blocks
        think_matches = _THINK_RE.findall(text)
        if think_matches:
            reasoning_segments.extend(s.strip() for s in think_matches if s.strip())

        # Try <reasoning>...</reasoning> blocks
        reason_matches = _REASONING_TAG_RE.findall(text)
        if reason_matches:
            reasoning_segments.extend(s.strip() for s in reason_matches if s.strip())

        # Try markdown-style reasoning sections
        if not reasoning_segments:
            md_reason = _MD_REASONING_RE.findall(text)
            if md_reason:
                reasoning_segments.extend(s.strip() for s in md_reason if s.strip())

        # ── Extract final answer ──
        # Try <answer>...</answer> first
        answer_match = _ANSWER_TAG_RE.findall(text)
        if answer_match:
            final_answer = answer_match[-1].strip()
        else:
            # Try markdown answer section
            md_ans = _MD_ANSWER_RE.search(text)
            if md_ans:
                final_answer = md_ans.group(1).strip()
            else:
                # Fall back to \boxed{} or last number
                final_answer = extract_final_answer(text)

        # If we have reasoning segments but no explicit answer, derive the answer
        # as the text remaining after all reasoning blocks are removed.
        if reasoning_segments and final_answer is None:
            remainder = text
            for seg in reasoning_segments:
                remainder = remainder.replace(seg, "", 1)
            # Strip markup tags
            remainder = re.sub(r"</?(?:think|reasoning)>", "", remainder).strip()
            if remainder:
                final_answer = remainder

        # ── Parse tool-call payloads into name + argument map (Req 12.2) ──
        tool_calls = parse_tool_payload(text)

    except Exception as exc:
        parse_error = f"parse_error: {type(exc).__name__}: {exc}"

    return ParsedReasoning(
        reasoning_segments=reasoning_segments,
        final_answer=final_answer,
        tool_calls=tool_calls,
        parse_error=parse_error,
    )


def parse_tool_payload(text: str) -> List[Dict[str, Any]]:
    """Parse tool-call payloads into structured name + argument map (Req 12.2).

    Returns a list of dicts each with:
        - "name": str — the tool name
        - "arguments": dict — the parsed argument map (NOT a JSON string)

    On malformed JSON within a tool block, that block is skipped (returns a
    partial list of what could be parsed). Never raises.
    """
    if not isinstance(text, str):
        return []

    results: List[Dict[str, Any]] = []

    # Try DSML parser first (the native format)
    if "tool_calls" in text or "invoke" in text:
        parse = _dsml_parser()
        if parse is not None:
            try:
                _, _, dsml_calls = parse(0, text)
                if dsml_calls:
                    for call in dsml_calls:
                        name = call.get("name", call.get("function", {}).get("name", ""))
                        args_raw = call.get("arguments", call.get("function", {}).get("arguments", {}))
                        if isinstance(args_raw, str):
                            try:
                                args_raw = json.loads(args_raw)
                            except (json.JSONDecodeError, TypeError):
                                args_raw = {}
                        if not isinstance(args_raw, dict):
                            args_raw = {}
                        results.append({"name": name, "arguments": args_raw})
                    return results
            except Exception:
                pass

    # Fallback: fenced ```tool/json {...}``` blocks
    for block in _FENCED_JSON_RE.findall(text):
        try:
            obj = json.loads(block)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            results.append({"name": obj["name"], "arguments": args})

    return results


# ══════════════════════════════════════════════════════════════════════
# CONTROLLER
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ReasoningStep:
    kind: str  # "propose" | "refine"
    text: str = ""
    score: float = 0.0
    tool_results: List[ToolResult] = field(default_factory=list)
    accepted: bool = False


@dataclass
class Transcript:
    best_answer: str
    best_score: float
    steps: List[ReasoningStep] = field(default_factory=list)
    converged: bool = False


def _call_key(call: Dict[str, Any]) -> str:
    fn = call.get("function", call)
    return f"{fn.get('name', '')}::{fn.get('arguments', '')}"


class ReasoningController:
    """Propose → (cached tool dispatch) → verify → refine, monotonic acceptance.

    Quality/perf features (all opt-in, defaults preserve behaviour):
      • ``n_candidates`` > 1  → parallel best-of-N proposals (pick highest score).
      • tool-dispatch cache  → identical tool calls never execute twice per run
                               ("best tool call per usage").
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        verifier: Optional[Verifier] = None,
        max_steps: int = 4,
        convergence_score: float = float("inf"),
        n_candidates: int = 1,
    ):
        self.registry = registry
        self.verifier = verifier or Verifier()
        self.max_steps = max_steps
        self.convergence_score = convergence_score
        self.n_candidates = max(1, n_candidates)
        self._tool_cache: Dict[str, ToolResult] = {}

    def _run_tools(self, text: str) -> List[ToolResult]:
        if self.registry is None:
            return []
        calls = extract_tool_calls(text)
        if not calls:
            return []
        results: List[ToolResult] = []
        for call in calls:
            key = _call_key(call)
            cached = self._tool_cache.get(key)
            if cached is None:
                cached = self.registry.dispatch_call(call)
                self._tool_cache[key] = cached  # best tool call per usage: run once
            results.append(cached)
        return results

    def _propose_best(self, propose: Callable[[], str]) -> Tuple[str, float, List[ToolResult]]:
        """Parallel best-of-N proposal: sample n_candidates, keep highest score."""
        best_text, best_score, best_tools = "", -math.inf, []
        for _ in range(self.n_candidates):
            text = propose()
            tools = self._run_tools(text)
            sc = self.verifier.score(text, tools)
            if sc > best_score:
                best_text, best_score, best_tools = text, sc, tools
        return best_text, best_score, best_tools

    def run(
        self,
        propose: Callable[[], str],
        refine: Optional[Callable[[str, str], str]] = None,
        critique: Optional[Callable[[str], str]] = None,
    ) -> Transcript:
        """Drive the reasoning loop. See class docstring for the opt-in features."""
        self._tool_cache.clear()
        draft, score, tools = self._propose_best(propose)
        steps = [ReasoningStep("propose", draft, score, tools, accepted=True)]
        best, best_score = draft, score

        for _ in range(self.max_steps):
            if best_score >= self.convergence_score:
                return Transcript(best, best_score, steps, converged=True)
            if refine is None:
                break
            crit = critique(best) if critique else ""
            cand = refine(best, crit)
            cand_tools = self._run_tools(cand)
            cand_score = self.verifier.score(cand, cand_tools)
            accepted = cand_score >= best_score  # monotonic non-worsening
            steps.append(ReasoningStep("refine", cand, cand_score, cand_tools, accepted))
            if accepted:
                best, best_score = cand, cand_score
        return Transcript(best, best_score, steps, converged=best_score >= self.convergence_score)


# ══════════════════════════════════════════════════════════════════════
# LIVE-MODEL ADAPTER
# ══════════════════════════════════════════════════════════════════════


def _default_critique_prompt(question: str, draft: str) -> str:
    return (
        "You are a meticulous reviewer. Identify logical errors, missing steps, "
        "or arithmetic mistakes. Then give a short fix plan.\n\n"
        f"Question:\n{question}\n\nDraft answer:\n{draft}\n\nCritique:"
    )


def _default_refine_prompt(question: str, draft: str, critique: str) -> str:
    return (
        "Revise the answer using the critique. Be concise and end with a final "
        "boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{question}\n\nPrevious answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\nRevised answer:"
    )


class LasmoidReasoner:
    """Binds the ReasoningController to a live text generator.

    Provide a ``generate_fn(prompt, domain_steer=None) -> str`` or a
    ``(model, tokenizer)`` pair via :meth:`from_model`. ``answer`` derives a
    *soft, optionally stochastic* domain prior from the question (the learned
    cortex router does the real work) and drives propose→verify→refine with
    optional parallel best-of-N and tool dispatch.
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        *,
        registry: Optional[ToolRegistry] = None,
        verifier: Optional[Verifier] = None,
        domain_names: Optional[List[str]] = None,
        max_steps: int = 3,
        steer_strength: float = 2.0,
        auto_domain_steer: bool = True,
        stochastic_steer: bool = True,
        n_candidates: int = 1,
    ):
        self.generate_fn = generate_fn
        self.registry = registry
        self.verifier = verifier or Verifier()
        self.domain_names = domain_names
        self.max_steps = max_steps
        self.steer_strength = steer_strength
        self.auto_domain_steer = auto_domain_steer
        self.stochastic_steer = stochastic_steer
        self.n_candidates = n_candidates

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        registry: Optional[ToolRegistry] = None,
        verifier: Optional[Verifier] = None,
        max_steps: int = 3,
        temperature: float = 0.7,
        top_k: int = 0,
        max_new_tokens: int = 256,
        steer_strength: float = 2.0,
        auto_domain_steer: bool = True,
        stochastic_steer: bool = True,
        n_candidates: int = 1,
    ) -> "LasmoidReasoner":
        device = next(model.parameters()).device
        domain_names = list(getattr(model.args, "domain_names", []))

        def generate_fn(prompt: str, domain_steer=None) -> str:
            ids = tokenizer.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                domain_steer=domain_steer,
            )
            new_ids = out[0, idx.shape[1]:].tolist()
            return tokenizer.decode(new_ids)

        return cls(
            generate_fn,
            registry=registry,
            verifier=verifier,
            domain_names=domain_names,
            max_steps=max_steps,
            steer_strength=steer_strength,
            auto_domain_steer=auto_domain_steer,
            stochastic_steer=stochastic_steer,
            n_candidates=n_candidates,
        )

    def _steer_for(self, question: str):
        if not self.auto_domain_steer or not self.domain_names or torch is None:
            return None
        return build_domain_steer(
            question,
            self.domain_names,
            strength=self.steer_strength,
            stochastic=self.stochastic_steer,
        )

    def answer(
        self,
        question: str,
        *,
        critique_prompt: Callable[[str, str], str] = _default_critique_prompt,
        refine_prompt: Callable[[str, str, str], str] = _default_refine_prompt,
    ) -> Transcript:
        steer = self._steer_for(question)

        def propose() -> str:
            return self.generate_fn(question, domain_steer=steer)

        def critique(draft: str) -> str:
            return self.generate_fn(critique_prompt(question, draft), domain_steer=steer)

        def refine(draft: str, crit: str) -> str:
            return self.generate_fn(refine_prompt(question, draft, crit), domain_steer=steer)

        controller = ReasoningController(
            registry=self.registry,
            verifier=self.verifier,
            max_steps=self.max_steps,
            n_candidates=self.n_candidates,
        )
        return controller.run(propose=propose, refine=refine, critique=critique)


# ══════════════════════════════════════════════════════════════════════
# SOCRATIC SELF-QUESTIONING LOOP
# ══════════════════════════════════════════════════════════════════════

_Q_SPLIT = re.compile(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•Q]:?)\s*", re.MULTILINE)


def parse_questions(text: str, k: int = 3) -> List[str]:
    """Extract up to k de-duplicated questions (numbered/bulleted/lines)."""
    parts = [p.strip() for p in _Q_SPLIT.split(text) if p.strip()]
    qs = [p for p in parts if "?" in p] or parts
    cleaned: List[str] = []
    seen = set()
    for q in qs:
        q = (q.split("?")[0].strip() + "?") if "?" in q else q.strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            cleaned.append(q)
    return cleaned[:k]


def make_bridging_prompt(topic: str, known: str, k: int) -> str:
    known_str = known or "the fundamentals"
    return (
        f"I already understand {known_str}. I am now learning: {topic}.\n"
        f"Ask the {k} most useful questions that connect this new topic to what I "
        f"already know — questions that reveal why it is needed and how it differs.\n"
        f"List them as 1., 2., 3.\nQuestions:"
    )


@dataclass
class SocraticTrace:
    topic: str
    known: str
    questions: List[str]
    answers: List[str]
    synthesis: str
    curriculum: List[Dict[str, str]] = field(default_factory=list)


class SocraticReasoner:
    """Human-like concept learning via self-questioning.

    Generate the top-k *bridging* questions for a new topic given prior
    knowledge, answer each, then synthesise an integrated explanation. The
    (question, answer) pairs form a *self-curriculum* — automating concept
    acquisition so the model learns the *reasoning behind* data, not just data,
    and can surface novel connections.
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        k: int = 3,
        registry: Optional[ToolRegistry] = None,
        verifier: Optional[Verifier] = None,
    ):
        self.generate_fn = generate_fn
        self.k = k
        self.registry = registry
        self.verifier = verifier or Verifier()

    def ask(self, topic: str, known: str = "") -> List[str]:
        return parse_questions(self.generate_fn(make_bridging_prompt(topic, known, self.k)), self.k)

    def learn(self, topic: str, known: str = "") -> SocraticTrace:
        questions = self.ask(topic, known)
        answers = [
            self.generate_fn(
                f"Answer concisely, connecting to {known or 'prior knowledge'}.\n"
                f"Question: {q}\nAnswer:"
            )
            for q in questions
        ]
        qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
        synthesis = self.generate_fn(
            f"Using these question-answer pairs, write a short integrated "
            f"explanation of '{topic}':\n{qa}\nIntegrated understanding:"
        )
        curriculum = [{"prompt": q, "completion": a} for q, a in zip(questions, answers)]
        return SocraticTrace(topic, known, questions, answers, synthesis, curriculum)

    def build_curriculum(self, traces: List[SocraticTrace]) -> List[Dict[str, str]]:
        """Flatten Socratic traces into self-training (prompt, completion) pairs,
        keeping only answers the Verifier scores above a small floor."""
        examples: List[Dict[str, str]] = []
        for tr in traces:
            examples.extend(ex for ex in tr.curriculum if self.verifier.score(ex["completion"]) > 0.0)
            if tr.synthesis.strip():
                examples.append({"prompt": f"Explain {tr.topic}.", "completion": tr.synthesis})
        return examples


__all__ = [
    "detect_domain_scores",
    "build_domain_steer",
    "extract_final_answer",
    "heuristic_score",
    "Verifier",
    "extract_tool_calls",
    "ParsedReasoning",
    "parse_completion",
    "parse_tool_payload",
    "ReasoningStep",
    "Transcript",
    "ReasoningController",
    "LasmoidReasoner",
    "parse_questions",
    "make_bridging_prompt",
    "SocraticTrace",
    "SocraticReasoner",
]
