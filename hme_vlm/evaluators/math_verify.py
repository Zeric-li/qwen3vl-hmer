from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import inspect
import time
from typing import Any

from hme_vlm.normalization import normalize_crohme_latex, wrap_latex_in_math_mode


def collect_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in [
        "math-verify",
        "sympy",
        "latex2sympy2_extended",
        "transformers",
        "datasets",
        "torch",
    ]:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


@dataclass(frozen=True)
class MathVerifyConfig:
    parsing_timeout_seconds: int | None = 2
    verify_timeout_seconds: int | None = 2
    float_rounding: int = 6
    numeric_precision: int = 15
    strict: bool = True
    allow_set_relation_comp: bool = False
    raise_on_error: bool = False
    pred_use_expr_fallback: bool = True


@dataclass
class MathVerifyResult:
    matched: bool
    gold_parse_ok: bool
    pred_parse_ok: bool
    verify_attempted: bool
    verify_timeout: bool
    error_type: str
    error_message: str
    gold_parse_count: int
    pred_parse_count: int
    parse_latency_ms: float
    verify_latency_ms: float

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "math_verify_match": self.matched,
            "math_verify_gold_parse_ok": self.gold_parse_ok,
            "math_verify_pred_parse_ok": self.pred_parse_ok,
            "math_verify_verify_attempted": self.verify_attempted,
            "math_verify_timeout": self.verify_timeout,
            "math_verify_error_type": self.error_type,
            "math_verify_error_message": self.error_message,
            "math_verify_gold_parse_count": self.gold_parse_count,
            "math_verify_pred_parse_count": self.pred_parse_count,
            "math_verify_parse_latency_ms": self.parse_latency_ms,
            "math_verify_verify_latency_ms": self.verify_latency_ms,
        }


class MathVerifier:
    def __init__(self, config: MathVerifyConfig | None = None):
        self.config = config or MathVerifyConfig()
        self.runtime_versions = collect_runtime_versions()
        self._parse_cache: dict[tuple[str, str], tuple[list[Any], bool, str, str, float]] = {}
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            math_verify = importlib.import_module("math_verify")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "math_verify is required for evaluation. Install math-verify in the active environment."
            ) from exc

        self._parse_fn = getattr(math_verify, "parse")
        self._verify_fn = getattr(math_verify, "verify")
        self._parse_signature = inspect.signature(self._parse_fn)
        self._verify_signature = inspect.signature(self._verify_fn)
        self._latex_config_cls = getattr(math_verify, "LatexExtractionConfig", None)
        self._expr_config_cls = getattr(math_verify, "ExprExtractionConfig", None)

        self._gold_extraction_config = []
        if self._latex_config_cls is not None:
            self._gold_extraction_config.append(self._latex_config_cls())

        self._pred_extraction_config = []
        if self._latex_config_cls is not None:
            self._pred_extraction_config.append(self._latex_config_cls())
        if self.config.pred_use_expr_fallback and self._expr_config_cls is not None:
            self._pred_extraction_config.append(self._expr_config_cls())

    def _supports_kwarg(self, signature: inspect.Signature, name: str) -> bool:
        return name in signature.parameters

    def _prepare_expression(self, text: str) -> str:
        return wrap_latex_in_math_mode(normalize_crohme_latex(text))

    def _parse_expression(self, expression: str, role: str) -> tuple[list[Any], bool, str, str, float]:
        prepared = self._prepare_expression(expression)
        cache_key = (role, prepared)
        if cache_key in self._parse_cache:
            return self._parse_cache[cache_key]

        kwargs: dict[str, Any] = {}
        extraction_config = self._gold_extraction_config if role == "gold" else self._pred_extraction_config
        if extraction_config and self._supports_kwarg(self._parse_signature, "extraction_config"):
            kwargs["extraction_config"] = extraction_config
        if self._supports_kwarg(self._parse_signature, "parsing_timeout"):
            kwargs["parsing_timeout"] = self.config.parsing_timeout_seconds
        elif self._supports_kwarg(self._parse_signature, "timeout_seconds"):
            kwargs["timeout_seconds"] = self.config.parsing_timeout_seconds
        if self._supports_kwarg(self._parse_signature, "raise_on_error"):
            kwargs["raise_on_error"] = self.config.raise_on_error

        start = time.perf_counter()
        try:
            parsed = self._parse_fn(prepared, **kwargs)
            result = (parsed or [], True, "", "", (time.perf_counter() - start) * 1000.0)
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            error_type = "parse_timeout" if "timeout" in exc_name else "parse_error"
            result = ([], False, error_type, str(exc), (time.perf_counter() - start) * 1000.0)

        self._parse_cache[cache_key] = result
        return result

    def preparse_gold(self, expressions: list[str]) -> None:
        for expression in expressions:
            self._parse_expression(expression, role="gold")

    def match(self, gold_latex: str, pred_latex: str) -> MathVerifyResult:
        gold_parsed, gold_ok, gold_error_type, gold_error_message, gold_latency_ms = self._parse_expression(
            gold_latex,
            role="gold",
        )
        pred_parsed, pred_ok, pred_error_type, pred_error_message, pred_latency_ms = self._parse_expression(
            pred_latex,
            role="pred",
        )

        if not gold_ok or not pred_ok or not gold_parsed or not pred_parsed:
            return MathVerifyResult(
                matched=False,
                gold_parse_ok=gold_ok and bool(gold_parsed),
                pred_parse_ok=pred_ok and bool(pred_parsed),
                verify_attempted=False,
                verify_timeout=False,
                error_type=gold_error_type or pred_error_type,
                error_message=gold_error_message or pred_error_message,
                gold_parse_count=len(gold_parsed),
                pred_parse_count=len(pred_parsed),
                parse_latency_ms=gold_latency_ms + pred_latency_ms,
                verify_latency_ms=0.0,
            )

        verify_kwargs: dict[str, Any] = {
            "float_rounding": self.config.float_rounding,
            "numeric_precision": self.config.numeric_precision,
            "strict": self.config.strict,
            "allow_set_relation_comp": self.config.allow_set_relation_comp,
        }
        if self._supports_kwarg(self._verify_signature, "timeout_seconds"):
            verify_kwargs["timeout_seconds"] = self.config.verify_timeout_seconds
        if self._supports_kwarg(self._verify_signature, "raise_on_error"):
            verify_kwargs["raise_on_error"] = self.config.raise_on_error

        start = time.perf_counter()
        try:
            matched = bool(self._verify_fn(gold_parsed, pred_parsed, **verify_kwargs))
            verify_latency_ms = (time.perf_counter() - start) * 1000.0
            return MathVerifyResult(
                matched=matched,
                gold_parse_ok=True,
                pred_parse_ok=True,
                verify_attempted=True,
                verify_timeout=False,
                error_type="",
                error_message="",
                gold_parse_count=len(gold_parsed),
                pred_parse_count=len(pred_parsed),
                parse_latency_ms=gold_latency_ms + pred_latency_ms,
                verify_latency_ms=verify_latency_ms,
            )
        except Exception as exc:
            verify_latency_ms = (time.perf_counter() - start) * 1000.0
            exc_name = type(exc).__name__.lower()
            error_type = "verify_timeout" if "timeout" in exc_name else "verify_error"
            return MathVerifyResult(
                matched=False,
                gold_parse_ok=True,
                pred_parse_ok=True,
                verify_attempted=True,
                verify_timeout=error_type == "verify_timeout",
                error_type=error_type,
                error_message=str(exc),
                gold_parse_count=len(gold_parsed),
                pred_parse_count=len(pred_parsed),
                parse_latency_ms=gold_latency_ms + pred_latency_ms,
                verify_latency_ms=verify_latency_ms,
            )
