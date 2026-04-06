import time
from typing import Dict, Any, List
from src.telemetry.logger import logger

# Pricing per 1K tokens (approximate, USD)
_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-3-flash-preview": {"input": 0.000075, "output": 0.0003},
    "gemini-3.1-flash-lite-preview": {"input": 0.0000375, "output": 0.00015},
}

class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs.
    """
    def __init__(self):
        self.session_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_estimate": self._calculate_cost(model, usage),
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculate estimated cost based on model pricing.
        Falls back to a generic rate if model is unknown.
        """
        model_lower = model.lower()
        if model_lower in _PRICING:
            pricing = _PRICING[model_lower]
            input_cost = (usage.get("prompt_tokens", 0) / 1000) * pricing["input"]
            output_cost = (usage.get("completion_tokens", 0) / 1000) * pricing["output"]
            return round(input_cost + output_cost, 6)
        # Fallback generic rate
        return round((usage.get("total_tokens", 0) / 1000) * 0.01, 6)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Returns an aggregated summary of all tracked metrics in the session.
        """
        if not self.session_metrics:
            return {"total_requests": 0}

        total_tokens = sum(m["total_tokens"] for m in self.session_metrics)
        total_latency = sum(m["latency_ms"] for m in self.session_metrics)
        total_cost = sum(m["cost_estimate"] for m in self.session_metrics)
        latencies = [m["latency_ms"] for m in self.session_metrics]
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        summary = {
            "total_requests": len(self.session_metrics),
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "avg_latency_ms": round(total_latency / len(self.session_metrics), 2),
            "p50_latency_ms": p50,
            "p99_latency_ms": p99,
            "total_cost_estimate": round(total_cost, 6),
        }
        logger.log_event("SESSION_SUMMARY", summary)
        return summary

# Global tracker instance
tracker = PerformanceTracker()
