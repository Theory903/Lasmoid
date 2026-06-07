"""
Lasmoid — tools.py
======================================================================
Typed tool registry + ML/EDA toolset so the model can call tools and derive
information from raw data (exploratory data analysis across scientific domains).

Two layers:
  • ToolRegistry — register / validate / dispatch tools with JSON-schema params,
    integrating with the DSML tool-call format used by encoding_lasmoid.py.
  • EDA toolset — dependency-light (torch-only) primitives that turn raw data
    into bounded, text-serialisable findings: describe, correlate, fit_model,
    hypothesis_test, reduce_dim (PCA), cluster (k-means), plot_to_text.

Design goals: deterministic, safe (schema-validated), bounded output (never
overflow the context window), and zero hard dependency on scipy/sklearn.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


# ══════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ToolResult:
    """Result envelope compatible with encoding_lasmoid tool messages."""

    tool: str
    status: str  # "success" | "error"
    content: Any
    tool_use_id: str = ""
    confidence: float = 1.0

    def to_envelope(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool": self.tool,
            "tool_use_id": self.tool_use_id,
            "status": self.status,
            "confidence": self.confidence,
            "content": self.content
            if isinstance(self.content, str)
            else json.dumps(self.content, default=_json_default),
        }


@dataclass
class ToolSpec:
    name: str
    parameters: Dict[str, Any]  # JSON-schema-ish: {type:object, properties, required}
    handler: Callable[..., Any]
    description: str = ""

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


class ToolRegistry:
    """Register, validate, and dispatch typed tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ── registration ──────────────────────────────────────────────────
    def register(
        self,
        name: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        description: str = "",
    ) -> None:
        self._tools[name] = ToolSpec(name, parameters, handler, description)

    def register_spec(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-format tool schemas for the system prompt."""
        return [self._tools[n].to_openai() for n in self.names()]

    # ── validation ────────────────────────────────────────────────────
    def validate(self, name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Return an error string if arguments are invalid, else None."""
        if name not in self._tools:
            return f"unknown tool: {name}"
        schema = self._tools[name].parameters or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for req in required:
            if req not in arguments:
                return f"missing required parameter: {req}"
        for key, val in arguments.items():
            if key in props and "type" in props[key]:
                expected = props[key]["type"]
                py = _JSON_TYPES.get(expected)
                # bool is a subclass of int — guard integer/number checks
                if py is not None:
                    if expected in ("number", "integer") and isinstance(val, bool):
                        return f"parameter {key} must be {expected}, got boolean"
                    if not isinstance(val, py):
                        return f"parameter {key} must be {expected}"
        return None

    # ── dispatch ──────────────────────────────────────────────────────
    def dispatch(
        self, name: str, arguments: Dict[str, Any], tool_use_id: str = ""
    ) -> ToolResult:
        err = self.validate(name, arguments)
        if err is not None:
            return ToolResult(name, "error", {"error": err}, tool_use_id, 0.0)
        try:
            out = self._tools[name].handler(**arguments)
            return ToolResult(name, "success", out, tool_use_id, 1.0)
        except Exception as exc:  # noqa: BLE001 — tools must degrade gracefully
            return ToolResult(
                name, "error", {"error": f"{type(exc).__name__}: {exc}"}, tool_use_id, 0.0
            )

    def dispatch_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Dispatch a parsed DSML/OpenAI tool call dict."""
        fn = tool_call.get("function", tool_call)
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult(name, "error", {"error": f"bad JSON args: {exc}"}, "", 0.0)
        else:
            arguments = raw_args or {}
        tuid = tool_call.get("id", tool_call.get("tool_use_id", ""))
        return self.dispatch(name, arguments, tuid)

    def dispatch_many(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        return [self.dispatch_call(tc) for tc in tool_calls]


# ══════════════════════════════════════════════════════════════════════
# EDA TOOLSET  (torch-only, bounded outputs)
# ══════════════════════════════════════════════════════════════════════

_MAX_COLS = 64
_ROUND = 6


def _as_2d(data: Any) -> torch.Tensor:
    t = torch.as_tensor(data, dtype=torch.float64)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    if t.ndim != 2:
        raise ValueError("data must be 1D or 2D (rows × columns)")
    if t.shape[1] > _MAX_COLS:
        raise ValueError(f"too many columns ({t.shape[1]} > {_MAX_COLS})")
    return t


def _r(x: float) -> float:
    if not math.isfinite(x):
        return x
    return round(float(x), _ROUND)


def eda_describe(data: Any) -> Dict[str, Any]:
    """Per-column summary statistics."""
    t = _as_2d(data)
    n, d = t.shape
    q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    cols = []
    for c in range(d):
        col = t[:, c]
        quart = torch.quantile(col, q).tolist() if n > 0 else [float("nan")] * 3
        cols.append(
            {
                "count": int(n),
                "mean": _r(col.mean().item()),
                "std": _r(col.std(unbiased=True).item()) if n > 1 else 0.0,
                "min": _r(col.min().item()),
                "q25": _r(quart[0]),
                "median": _r(quart[1]),
                "q75": _r(quart[2]),
                "max": _r(col.max().item()),
            }
        )
    return {"n_rows": int(n), "n_cols": int(d), "columns": cols}


def eda_correlate(data: Any) -> Dict[str, Any]:
    """Pearson correlation matrix across columns."""
    t = _as_2d(data)
    if t.shape[0] < 2:
        raise ValueError("need at least 2 rows to correlate")
    tc = t - t.mean(dim=0, keepdim=True)
    std = tc.std(dim=0, unbiased=True).clamp_min(1e-12)
    corr = (tc / std).T @ (tc / std) / (t.shape[0] - 1)
    corr = corr.clamp(-1.0, 1.0)
    return {"matrix": [[_r(v) for v in row] for row in corr.tolist()]}


def eda_fit_model(x: Any, y: Any, kind: str = "linear", degree: int = 1) -> Dict[str, Any]:
    """Least-squares fit (linear or polynomial); returns coefficients + R²."""
    xt = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    yt = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    if xt.numel() != yt.numel() or xt.numel() < 2:
        raise ValueError("x and y must be equal-length vectors with ≥2 points")
    deg = max(1, int(degree)) if kind == "poly" else 1
    # Design matrix [1, x, x², ...]
    cols = [torch.ones_like(xt)] + [xt**p for p in range(1, deg + 1)]
    X = torch.stack(cols, dim=1)
    sol = torch.linalg.lstsq(X, yt.unsqueeze(1)).solution.squeeze(1)
    pred = X @ sol
    ss_res = ((yt - pred) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = 1.0 - (ss_res / ss_tot).item()
    return {
        "kind": "poly" if kind == "poly" else "linear",
        "degree": deg,
        "coefficients": [_r(v) for v in sol.tolist()],  # [intercept, x¹, x², ...]
        "r2": _r(r2),
    }


def _norm_sf(z: float) -> float:
    """Survival function of standard normal via erfc (scipy-free p-value approx)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def eda_hypothesis_test(sample_a: Any, sample_b: Any, test: str = "welch_t") -> Dict[str, Any]:
    """Two-sample Welch's t-test. p-value uses a normal approximation."""
    a = torch.as_tensor(sample_a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(sample_b, dtype=torch.float64).reshape(-1)
    if a.numel() < 2 or b.numel() < 2:
        raise ValueError("each sample needs ≥2 observations")
    na, nb = a.numel(), b.numel()
    va, vb = a.var(unbiased=True), b.var(unbiased=True)
    se = torch.sqrt(va / na + vb / nb).clamp_min(1e-12)
    t = ((a.mean() - b.mean()) / se).item()
    # Welch–Satterthwaite dof
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    dof = (num / den.clamp_min(1e-12)).item()
    p = 2.0 * _norm_sf(abs(t))
    return {
        "test": "welch_t",
        "t_statistic": _r(t),
        "dof": _r(dof),
        "p_value_normal_approx": _r(p),
        "mean_a": _r(a.mean().item()),
        "mean_b": _r(b.mean().item()),
        "note": "p-value is a normal approximation (scipy-free); accurate for large dof",
    }


def eda_reduce_dim(data: Any, n_components: int = 2) -> Dict[str, Any]:
    """PCA via SVD; returns explained-variance ratios and top components."""
    t = _as_2d(data)
    if t.shape[0] < 2:
        raise ValueError("need at least 2 rows for PCA")
    k = max(1, min(int(n_components), t.shape[1]))
    tc = t - t.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(tc, full_matrices=False)
    var = (S**2) / (t.shape[0] - 1)
    ratio = (var / var.sum().clamp_min(1e-12))[:k]
    comps = Vh[:k]
    return {
        "n_components": k,
        "explained_variance_ratio": [_r(v) for v in ratio.tolist()],
        "components": [[_r(v) for v in row] for row in comps.tolist()],
    }


def eda_cluster(data: Any, k: int = 3, iters: int = 25, seed: int = 0) -> Dict[str, Any]:
    """K-means (Lloyd's algorithm, torch); returns labels, centroids, inertia."""
    t = _as_2d(data)
    n = t.shape[0]
    k = max(1, min(int(k), n))
    g = torch.Generator().manual_seed(int(seed))
    centroids = t[torch.randperm(n, generator=g)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(int(iters)):
        d2 = torch.cdist(t, centroids) ** 2  # [n, k]
        new_labels = d2.argmin(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = t[mask].mean(dim=0)
    inertia = (torch.cdist(t, centroids) ** 2).gather(1, labels.unsqueeze(1)).sum().item()
    return {
        "k": k,
        "labels": labels.tolist(),
        "centroids": [[_r(v) for v in row] for row in centroids.tolist()],
        "inertia": _r(inertia),
    }


def eda_plot_to_text(data: Any, bins: int = 10, width: int = 20) -> Dict[str, Any]:
    """Text histogram (unicode bars) of a 1D series — bounded, context-safe."""
    t = torch.as_tensor(data, dtype=torch.float64).reshape(-1)
    if t.numel() < 1:
        raise ValueError("empty data")
    bins = max(1, int(bins))
    lo, hi = t.min().item(), t.max().item()
    if hi <= lo:
        hi = lo + 1.0
    edges = torch.linspace(lo, hi, bins + 1, dtype=torch.float64)
    counts = torch.histc(t, bins=bins, min=lo, max=hi)
    cmax = counts.max().clamp_min(1).item()
    lines = []
    for i in range(bins):
        bar = "█" * int(round((counts[i].item() / cmax) * width))
        lines.append(f"[{_r(edges[i].item()):>10}, {_r(edges[i+1].item()):>10}) | {bar} {int(counts[i].item())}")
    return {"min": _r(lo), "max": _r(hi), "bins": bins, "histogram": "\n".join(lines)}


# JSON schemas for the EDA tools
def _arr2d_schema(desc: str) -> Dict[str, Any]:
    return {"type": "array", "description": desc}


def make_eda_registry() -> ToolRegistry:
    """A ToolRegistry pre-loaded with the EDA toolset."""
    reg = ToolRegistry()
    reg.register(
        "describe",
        {"type": "object", "properties": {"data": _arr2d_schema("rows × cols")}, "required": ["data"]},
        eda_describe,
        "Per-column summary statistics of a dataset.",
    )
    reg.register(
        "correlate",
        {"type": "object", "properties": {"data": _arr2d_schema("rows × cols")}, "required": ["data"]},
        eda_correlate,
        "Pearson correlation matrix across columns.",
    )
    reg.register(
        "fit_model",
        {
            "type": "object",
            "properties": {
                "x": _arr2d_schema("predictor vector"),
                "y": _arr2d_schema("response vector"),
                "kind": {"type": "string"},
                "degree": {"type": "integer"},
            },
            "required": ["x", "y"],
        },
        eda_fit_model,
        "Least-squares linear/polynomial regression with R².",
    )
    reg.register(
        "hypothesis_test",
        {
            "type": "object",
            "properties": {
                "sample_a": _arr2d_schema("first sample"),
                "sample_b": _arr2d_schema("second sample"),
                "test": {"type": "string"},
            },
            "required": ["sample_a", "sample_b"],
        },
        eda_hypothesis_test,
        "Two-sample Welch's t-test.",
    )
    reg.register(
        "reduce_dim",
        {
            "type": "object",
            "properties": {"data": _arr2d_schema("rows × cols"), "n_components": {"type": "integer"}},
            "required": ["data"],
        },
        eda_reduce_dim,
        "PCA dimensionality reduction (explained variance + components).",
    )
    reg.register(
        "cluster",
        {
            "type": "object",
            "properties": {"data": _arr2d_schema("rows × cols"), "k": {"type": "integer"}},
            "required": ["data"],
        },
        eda_cluster,
        "K-means clustering (labels, centroids, inertia).",
    )
    reg.register(
        "plot_to_text",
        {
            "type": "object",
            "properties": {"data": _arr2d_schema("1D series"), "bins": {"type": "integer"}},
            "required": ["data"],
        },
        eda_plot_to_text,
        "Text histogram of a 1D series.",
    )
    return reg


__all__ = [
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "make_eda_registry",
    "eda_describe",
    "eda_correlate",
    "eda_fit_model",
    "eda_hypothesis_test",
    "eda_reduce_dim",
    "eda_cluster",
    "eda_plot_to_text",
]
