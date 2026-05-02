#!/usr/bin/env python3
"""
LLM Metrics Logger — Decorator-pattern instrumentation for multi-model benchmarking.

Usage (automatic via decorator — no manual calls needed in most code):
    from llm_metrics_logger import llm_call_metrics, LLMMetricsLogger

    class MyAnalyzer:
        @llm_call_metrics()
        def analyze_threats(self, system_data):
            ...

RL-impact update (call from coordinator after episode reward is known):
    LLMMetricsLogger.instance().update_rl_impact(
        call_id   = self.llm_analyzer._last_call_id,
        reward_before = prev_reward,
        reward_after  = current_reward,
        task_success  = True,
        rl_plan_accepted = True,
    )

Output files (written to ./llm_metrics/ by default):
    llm_calls_<timestamp>_<run_id>.csv   — append-only per-call rows
    llm_calls_<timestamp>_<run_id>.jsonl — full records (rewritten on RL updates)

Comparative analysis (offline, in pandas / notebook):
    import pandas as pd
    df = pd.read_csv("llm_metrics/llm_calls_*.csv")

    # Efficiency Score
    df["efficiency"] = df["quality_score_rule"] / (df["cost_usd"].clip(lower=1e-9) * df["total_time_s"])

    # Robustness Index  (per model)
    g = df.groupby("model_name")["quality_score_rule"]
    df_rob = (g.mean() * (1 - g.std())).rename("robustness_index")

    # RL Integration Gain
    df["rl_reward_delta"].groupby(df["model_name"]).mean()

    # Latency-Quality Correlation
    df[["ttft_s","quality_score_rule"]].dropna().corr(method="pearson")

    # Pareto Frontier (3D: cost, quality, latency)
    # plot df[["cost_usd","quality_score_rule","total_time_s"]] coloured by model_name
"""

import csv
import json
import re
import time
import uuid
import functools
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any


# ── Model Pricing Table ───────────────────────────────────────────────────────
# USD per 1 000 tokens  (snapshot 2025-04; free models = $0.00)
# Add new models here; unknown models default to $0 so runs never crash.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # ── Google Gemini (native SDK) ────────────────────────────────────────────
    "models/gemini-2.5-flash":          {"input_k": 0.000150, "output_k": 0.000600},
    "models/gemini-2.5-pro":            {"input_k": 0.001250, "output_k": 0.010000},
    "models/gemini-1.5-flash":          {"input_k": 0.000075, "output_k": 0.000300},
    "models/gemini-1.5-pro":            {"input_k": 0.003500, "output_k": 0.010500},
    # ── Google Gemini via OpenRouter ──────────────────────────────────────────
    "google/gemini-2.5-flash":          {"input_k": 0.000150, "output_k": 0.000600},
    "google/gemini-2.5-pro":            {"input_k": 0.001250, "output_k": 0.010000},
    "google/gemini-1.5-flash":          {"input_k": 0.000075, "output_k": 0.000300},
    "google/gemini-1.5-pro":            {"input_k": 0.003500, "output_k": 0.010500},
    "google/gemini-2.0-flash-exp:free": {"input_k": 0.000000, "output_k": 0.000000},
    # ── OpenAI via OpenRouter ─────────────────────────────────────────────────
    "openai/gpt-4o":                    {"input_k": 0.002500, "output_k": 0.010000},
    "openai/gpt-4o-mini":               {"input_k": 0.000150, "output_k": 0.000600},
    "openai/gpt-4-turbo":               {"input_k": 0.010000, "output_k": 0.030000},
    "openai/gpt-4.1":                   {"input_k": 0.002000, "output_k": 0.008000},
    "openai/gpt-4.1-mini":              {"input_k": 0.000400, "output_k": 0.001600},
    "openai/gpt-oss-120b:free":         {"input_k": 0.000000, "output_k": 0.000000},
    # ── Anthropic via OpenRouter ──────────────────────────────────────────────
    "anthropic/claude-3.5-sonnet":      {"input_k": 0.003000, "output_k": 0.015000},
    "anthropic/claude-3.5-haiku":       {"input_k": 0.000800, "output_k": 0.004000},
    "anthropic/claude-sonnet-4.5":      {"input_k": 0.003000, "output_k": 0.015000},
    # ── DeepSeek via OpenRouter ───────────────────────────────────────────────
    "deepseek/deepseek-v3.2":           {"input_k": 0.000200, "output_k": 0.000600},
    "deepseek/deepseek-chat-v3.1:free":             {"input_k": 0.0, "output_k": 0.0},
    "deepseek/deepseek-r1:free":                     {"input_k": 0.0, "output_k": 0.0},
    # ── Free / open-source via OpenRouter ────────────────────────────────────
    "meta-llama/llama-3.3-70b-instruct:free":        {"input_k": 0.0, "output_k": 0.0},
    "meta-llama/llama-3.3-70b-instruct":             {"input_k": 0.0, "output_k": 0.0},
    "mistralai/mistral-small-3.1-24b-instruct:free": {"input_k": 0.0, "output_k": 0.0},
    "minimax/minimax-m2.5:free":                     {"input_k": 0.0, "output_k": 0.0},
    "nvidia/nemotron-3-super-120b-a12b:free":        {"input_k": 0.0, "output_k": 0.0},
    "openrouter/elephant-alpha":                     {"input_k": 0.0, "output_k": 0.0},
}

# ── Rule-based quality helpers ────────────────────────────────────────────────
# MITRE ATT&CK technique IDs:  T1234  or  T1234.001
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

# STRIDE threat categories (all six)
_STRIDE_KEYWORDS = {
    "spoofing",
    "tampering",
    "repudiation",
    "information disclosure",
    "denial of service",
    "elevation of privilege",
}

# Required top-level keys for schema validation (any one of these = valid schema)
_SCHEMA_REQUIRED_KEYS = {
    # analyze_threats / general threat analysis responses
    "threats", "deployments", "vulnerabilities", "recommendations",
    "attack_strategy", "analysis",
    # RL attack-ordering responses  (generate_rl_attack_plan)
    "ordered_attacks", "attack_plan", "attacks",
    # RL adaptation responses       (adapt_rl_strategy / generate_adaptation)
    "adaptation_recommendation", "adaptation_decision",
    "continue_current_strategy", "should_continue", "strategy",
}


def _compute_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for this call based on the pricing table."""
    pricing = MODEL_PRICING.get(model_name.lower(), {"input_k": 0.0, "output_k": 0.0})
    return (prompt_tokens * pricing["input_k"] + completion_tokens * pricing["output_k"]) / 1000.0


def _quality_score_rule_based(text: str) -> Dict[str, Any]:
    """
    Rule-based quality scorer over the raw LLM response text.

    Returns:
        mitre_id_count      — number of distinct MITRE technique IDs found
        stride_category_count — STRIDE categories mentioned (0–6)
        quality_score_rule  — composite score in [0, 1]
                              60% MITRE coverage + 40% STRIDE coverage
    """
    if not text:
        return {"mitre_id_count": 0, "stride_category_count": 0, "quality_score_rule": 0.0}

    lower = text.lower()
    mitre_ids = set(_MITRE_RE.findall(text))           # unique IDs
    stride_hits = sum(1 for kw in _STRIDE_KEYWORDS if kw in lower)

    # Normalise: ≥5 distinct MITRE IDs = full MITRE score; 6 STRIDE = full STRIDE score
    mitre_norm  = min(len(mitre_ids) / 5.0, 1.0)
    stride_norm = stride_hits / 6.0
    quality     = round(0.6 * mitre_norm + 0.4 * stride_norm, 4)

    return {
        "mitre_id_count":       len(mitre_ids),
        "stride_category_count": stride_hits,
        "quality_score_rule":    quality,
    }


# ── Data record ───────────────────────────────────────────────────────────────

@dataclass
class LLMCallRecord:
    """One row in the metrics CSV / JSONL — one record per decorated LLM method call."""

    # ── Identity ──────────────────────────────────────────────────────────────
    run_id:           str = ""
    call_id:          str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provider:         str = ""          # "gemini" | "openrouter"
    model_name:       str = ""
    method_name:      str = ""          # e.g. "analyze_threats"

    # ── Latency ───────────────────────────────────────────────────────────────
    total_time_s:     float         = 0.0
    # Time-to-first-token: only populated for streaming calls.
    # For batch (non-streaming) calls this stays None.
    ttft_s:           Optional[float] = None

    # ── Token counts & estimated cost ────────────────────────────────────────
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0

    # ── Format / Schema validation ───────────────────────────────────────────
    is_valid_json:    bool = False
    schema_valid:     bool = False      # response has at least one expected top-level key
    json_parse_error: str  = ""

    # ── Rule-based quality ────────────────────────────────────────────────────
    mitre_id_count:        int   = 0
    stride_category_count: int   = 0
    quality_score_rule:    float = 0.0

    # ── Raw artefacts for offline / model-based quality ───────────────────────
    prompt_length_chars:   int = 0
    response_length_chars: int = 0
    response_snippet:      str = ""     # first 500 chars of LLM text output

    # ── Task success (filled retroactively by downstream consumer) ───────────
    task_success:      Optional[bool] = None   # did downstream accept the plan?
    rl_plan_accepted:  Optional[bool] = None   # specifically: RL used this plan?

    # ── RL impact (filled retroactively by episode logger) ───────────────────
    rl_reward_before:  Optional[float] = None
    rl_reward_after:   Optional[float] = None
    rl_reward_delta:   Optional[float] = None  # reward_after - reward_before

    # ── Simulation context ───────────────────────────────────────────────────
    grid_id:  Optional[int] = None
    episode:  Optional[int] = None
    step:     Optional[int] = None

    # ── Error tracking ───────────────────────────────────────────────────────
    is_fallback:   bool = False
    error_message: str  = ""


# ── Singleton logger ──────────────────────────────────────────────────────────

class LLMMetricsLogger:
    """
    Singleton that persists LLMCallRecord objects to:
      • CSV  — one row per call; rewritten when RL-impact data arrives
      • JSONL — one JSON object per line; also rewritten on updates

    Both files live in  output_dir/  and share the same timestamp + run_id stem.
    """

    _instance: Optional["LLMMetricsLogger"] = None

    def __init__(self, output_dir: str = "llm_metrics", run_id: str = None):
        self.run_id     = run_id or str(uuid.uuid4())[:8]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"llm_calls_{ts}_{self.run_id}"
        self.csv_path   = self.output_dir / f"{stem}.csv"
        self.jsonl_path = self.output_dir / f"{stem}.jsonl"

        # In-memory store keyed by call_id for retroactive updates
        self._records:         Dict[str, LLMCallRecord] = {}
        self._csv_initialized: bool = False

        print(f"📊 LLMMetricsLogger  →  {self.csv_path.name}")

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    def instance(cls, **kwargs) -> "LLMMetricsLogger":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force creation of a fresh logger on next .instance() call (useful in tests)."""
        cls._instance = None

    # ── Write path ────────────────────────────────────────────────────────────

    def log_call(self, record: LLMCallRecord) -> None:
        """Append a new call record to CSV and JSONL."""
        record.run_id = self.run_id
        self._records[record.call_id] = record
        self._append_csv(record)
        self._append_jsonl(record)

    # ── Retroactive RL-impact update ─────────────────────────────────────────

    def update_rl_impact(
        self,
        call_id: str,
        *,
        reward_before:    float,
        reward_after:     float,
        task_success:     Optional[bool] = None,
        rl_plan_accepted: Optional[bool] = None,
    ) -> None:
        """
        Fill RL-reward columns for a previously logged call and rewrite both files.

        Call this from the coordinator immediately after the outer-episode reward
        is available, e.g.:

            LLMMetricsLogger.instance().update_rl_impact(
                call_id           = self.llm_analyzer._last_call_id,
                reward_before     = prev_episode_reward,
                reward_after      = current_episode_reward,
                task_success      = success_flag,
                rl_plan_accepted  = (source == 'gemini'),
            )
        """
        rec = self._records.get(call_id)
        if rec is None:
            return  # unknown call_id — silently ignore

        rec.rl_reward_before = float(reward_before)
        rec.rl_reward_after  = float(reward_after)
        rec.rl_reward_delta  = round(float(reward_after) - float(reward_before), 6)
        if task_success is not None:
            rec.task_success = task_success
        if rl_plan_accepted is not None:
            rec.rl_plan_accepted = rl_plan_accepted

        # Rewrite both files so the stored data always reflects latest state
        self._rewrite_csv()
        self._rewrite_jsonl()

    # ── Low-level I/O ─────────────────────────────────────────────────────────

    def _field_names(self) -> list:
        import dataclasses
        return [f.name for f in dataclasses.fields(LLMCallRecord)]

    def _append_csv(self, record: LLMCallRecord) -> None:
        row = asdict(record)
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writeheader()
            self._csv_initialized = True
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

    def _append_jsonl(self, record: LLMCallRecord) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")

    def _rewrite_csv(self) -> None:
        if not self._records:
            return
        rows = list(self._records.values())
        fields = list(asdict(rows[0]).keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in rows:
                writer.writerow(asdict(rec))

    def _rewrite_jsonl(self) -> None:
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for rec in self._records.values():
                f.write(json.dumps(asdict(rec), ensure_ascii=False, default=str) + "\n")

    # ── Live summary helpers ──────────────────────────────────────────────────

    def summary_report(self, top_n: int = 10) -> str:
        """
        Return a human-readable summary of all calls logged so far.

        Covers:
          • Total calls, fallback rate, error rate
          • Token & cost totals
          • Quality score distribution (rule-based)
          • RL reward-delta where available

        Useful for a quick mid-run health check or final printout.
        """
        if not self._records:
            return "No LLM calls logged yet."

        records = list(self._records.values())
        n = len(records)
        fallbacks  = sum(1 for r in records if r.is_fallback)
        errors     = sum(1 for r in records if r.error_message)
        total_toks = sum(r.total_tokens for r in records)
        total_cost = sum(r.cost_usd    for r in records)
        avg_qual   = sum(r.quality_score_rule for r in records) / n
        avg_time   = sum(r.total_time_s for r in records) / n

        rl_deltas = [r.rl_reward_delta for r in records if r.rl_reward_delta is not None]
        rl_str    = (
            f"mean RL Δreward = {sum(rl_deltas)/len(rl_deltas):+.4f}  "
            f"(n={len(rl_deltas)})"
            if rl_deltas else "no RL-impact data yet"
        )

        lines = [
            "═" * 60,
            f"  LLM Metrics Summary  |  run_id={self.run_id}",
            "═" * 60,
            f"  Total calls   : {n}",
            f"  Fallback rate : {fallbacks}/{n}  ({100*fallbacks/n:.1f}%)",
            f"  Error rate    : {errors}/{n}  ({100*errors/n:.1f}%)",
            f"  Tokens used   : {total_toks:,}",
            f"  Est. cost     : ${total_cost:.6f} USD",
            f"  Avg quality   : {avg_qual:.3f}  (rule-based 0-1)",
            f"  Avg latency   : {avg_time:.2f} s",
            f"  RL impact     : {rl_str}",
            "─" * 60,
        ]

        # Last N call IDs for traceability
        recent = records[-min(top_n, n):]
        lines.append(f"  Last {len(recent)} calls:")
        for r in recent:
            rl_tag = (
                f"  RL_delta={r.rl_reward_delta:+.3f}"
                if r.rl_reward_delta is not None else ""
            )
            lines.append(
                f"    [{r.method_name:30s}]  "
                f"qual={r.quality_score_rule:.2f}  "
                f"t={r.total_time_s:.2f}s  "
                f"cost=${r.cost_usd:.6f}"
                f"{rl_tag}"
            )
        lines.append("═" * 60)
        return "\n".join(lines)

    def model_comparison_table(self) -> str:
        """
        Return a per-model aggregated comparison table (ASCII).

        Columns: model | calls | avg_quality | avg_latency_s | total_cost_usd
                       | fallback_rate | avg_rl_delta

        Useful for multi-model benchmarking: just swap OPENROUTER_MODEL_NAME
        between runs and load all CSVs into pandas for the full Pareto analysis.
        """
        from collections import defaultdict
        if not self._records:
            return "No LLM calls logged yet."

        buckets: Dict[str, list] = defaultdict(list)
        for r in self._records.values():
            buckets[r.model_name].append(r)

        header = (
            f"{'Model':<45} {'Calls':>5} {'AvgQual':>8} "
            f"{'AvgLat':>8} {'TotCost':>12} {'FBRate':>7} {'AvgRLΔ':>9}"
        )
        sep = "─" * len(header)
        rows = [sep, header, sep]

        for model, recs in sorted(buckets.items()):
            n        = len(recs)
            avg_q    = sum(r.quality_score_rule for r in recs) / n
            avg_lat  = sum(r.total_time_s       for r in recs) / n
            tot_cost = sum(r.cost_usd           for r in recs)
            fb_rate  = sum(1 for r in recs if r.is_fallback) / n
            deltas   = [r.rl_reward_delta for r in recs if r.rl_reward_delta is not None]
            avg_rl   = sum(deltas) / len(deltas) if deltas else float("nan")
            rows.append(
                f"{model:<45} {n:>5} {avg_q:>8.3f} "
                f"{avg_lat:>8.2f} {tot_cost:>12.6f} {fb_rate:>7.1%} "
                f"{avg_rl:>+9.4f}"
            )

        rows.append(sep)
        return "\n".join(rows)


# ── Decorator ─────────────────────────────────────────────────────────────────

def llm_call_metrics(method_name: str = None):
    """
    Class-method decorator for GeminiLLMThreatAnalyzer.

    Responsibilities:
      1. Times the full decorated method call (total_time_s).
      2. Reads self._last_raw_response (set by _timed_generate) to extract
         token counts from both Gemini and OpenRouter raw response objects.
      3. Reads self._last_prompt_length (also set by _timed_generate) for the
         accurate prompt character count.
      4. Computes rule-based quality (MITRE + STRIDE) from the returned text.
      5. Validates JSON format and schema of the response.
      6. Calculates estimated cost from the pricing table.
      7. Logs an LLMCallRecord via LLMMetricsLogger.instance().
      8. Exposes self._last_call_id so downstream callers can attach RL data.

    The decorated class MUST implement _timed_generate(prompt) which sets
    self._last_raw_response and self._last_prompt_length before returning.
    """

    def decorator(func):
        _mname = method_name or func.__name__

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # ── Reset per-call state ───────────────────────────────────────
            self._last_raw_response  = None
            self._last_prompt_length = 0
            self._last_call_id       = None

            error_msg   = ""
            is_fallback = False
            result      = {}

            t0 = time.perf_counter()

            try:
                result = func(self, *args, **kwargs)
            except Exception as exc:
                error_msg = str(exc)
                raise
            finally:
                elapsed = time.perf_counter() - t0

                # ── Extract token counts from raw response ─────────────────
                prompt_tokens     = 0
                completion_tokens = 0
                raw = getattr(self, "_last_raw_response", None)

                if raw is not None:
                    # Gemini native SDK: response.usage_metadata
                    um = getattr(raw, "usage_metadata", None)
                    if um is not None:
                        prompt_tokens     = int(getattr(um, "prompt_token_count",      0) or 0)
                        completion_tokens = int(getattr(um, "candidates_token_count",  0) or 0)

                    # OpenRouter (_OpenRouterResponse): raw.raw is the OpenAI CompletionObject
                    or_raw = getattr(raw, "raw", None)
                    if or_raw is not None:
                        usage = getattr(or_raw, "usage", None)
                        if usage is not None:
                            prompt_tokens     = int(getattr(usage, "prompt_tokens",     0) or 0)
                            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

                # ── Derive response text for quality scoring ───────────────
                resp_text = ""
                if isinstance(result, dict):
                    resp_text   = result.get("llm_response", "") or ""
                    is_fallback = (result.get("analysis_type", "") == "Fallback_Analysis")
                if not resp_text and raw is not None:
                    resp_text = getattr(raw, "text", "") or ""

                # ── JSON validity ──────────────────────────────────────────
                # Gemini often wraps JSON in markdown code fences (```json ... ```).
                # Strip those before attempting to parse so is_valid_json is correct.
                is_valid_json    = False
                json_parse_error = ""
                stripped = resp_text.strip()
                # Remove markdown code fence if present: ```json\n...\n``` or ```\n...\n```
                import re as _re
                _fence_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', stripped, _re.DOTALL)
                if _fence_match:
                    stripped = _fence_match.group(1).strip()
                if stripped.startswith(("{", "[")):
                    try:
                        json.loads(stripped)
                        is_valid_json = True
                    except json.JSONDecodeError as je:
                        json_parse_error = str(je)[:200]

                # ── Schema validity ────────────────────────────────────────
                schema_valid = False
                if is_valid_json:
                    try:
                        parsed = json.loads(stripped)
                        if isinstance(parsed, dict):
                            schema_valid = bool(_SCHEMA_REQUIRED_KEYS & set(parsed.keys()))
                    except Exception:
                        pass

                # ── Rule-based quality ─────────────────────────────────────
                qual = _quality_score_rule_based(resp_text)

                # ── Cost ───────────────────────────────────────────────────
                cost = _compute_cost_usd(
                    getattr(self, "model_name", ""),
                    prompt_tokens,
                    completion_tokens,
                )

                # ── Build and log record ───────────────────────────────────
                record = LLMCallRecord(
                    provider         = getattr(self, "provider",    "unknown"),
                    model_name       = getattr(self, "model_name",  "unknown"),
                    method_name      = _mname,
                    total_time_s     = round(elapsed, 4),
                    prompt_tokens    = prompt_tokens,
                    completion_tokens= completion_tokens,
                    total_tokens     = prompt_tokens + completion_tokens,
                    cost_usd         = round(cost, 8),
                    is_valid_json    = is_valid_json,
                    schema_valid     = schema_valid,
                    json_parse_error = json_parse_error,
                    mitre_id_count        = qual["mitre_id_count"],
                    stride_category_count = qual["stride_category_count"],
                    quality_score_rule    = qual["quality_score_rule"],
                    prompt_length_chars   = getattr(self, "_last_prompt_length", 0),
                    response_length_chars = len(resp_text),
                    response_snippet      = resp_text[:500],
                    is_fallback           = is_fallback,
                    error_message         = error_msg,
                )

                self._last_call_id = record.call_id
                LLMMetricsLogger.instance().log_call(record)

                # Accumulate call_ids for per-episode RL-impact update.
                # _episode_call_ids is initialised to [] by the system
                # at the start of each episode; we just append here.
                if hasattr(self, '_episode_call_ids') and isinstance(self._episode_call_ids, list):
                    self._episode_call_ids.append(record.call_id)

            return result
        return wrapper
    return decorator
