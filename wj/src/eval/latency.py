"""모델 효율성 측정 도구."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch


@dataclass
class LatencyConfig:
    warmup_iters: int = 5
    measure_iters: int = 20
    repeat: int = 3


@contextlib.contextmanager
def _cuda_nvml_context():
    """NVML 또는 torch.cuda API를 이용한 VRAM 측정."""
    start_memory = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
    yield
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        end = torch.cuda.memory_allocated()
        return {
            "memory_start": start_memory,
            "memory_peak": peak,
            "memory_end": end,
        }
    return None


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(
    model: torch.nn.Module,
    inputs: Iterable[Dict[str, Any]],
    *,
    warmup_iters: int = 5,
    measure_iters: int = 20,
) -> Dict[str, float]:
    """Latency(ms) 및 FPS 측정."""
    timings = []
    iterator = iter(inputs)

    # Warm-up
    for _ in range(warmup_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        _run_inference(model, batch)

    for _ in range(measure_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)

        start = time.perf_counter()
        _run_inference(model, batch)
        _synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0  # ms
        timings.append(elapsed)

    timings_tensor = torch.tensor(timings)
    mean = float(timings_tensor.mean())
    std = float(timings_tensor.std(unbiased=False))
    fps = 1000.0 / mean if mean > 0 else float("inf")
    return {"latency_ms_mean": mean, "latency_ms_std": std, "fps": fps}


def _run_inference(model: torch.nn.Module, batch: Dict[str, Any]) -> None:
    model.eval()
    with torch.no_grad():
        model.predict(batch)


def measure_params(model: torch.nn.Module) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": float(total_params),
        "params_trainable": float(trainable_params),
    }


def estimate_flops(model: torch.nn.Module, example_batch: Dict[str, Any]) -> Optional[float]:
    """FLOPs 추정 (torch.profiler 사용) - TODO: TinyCLIP 예시에 맞게 구현."""
    # Placeholder: torch.profiler, fvcore, ptflops 등과 연동 필요
    _ = model, example_batch  # 미사용 방지
    # TODO: ptflops 또는 fvcore를 활용한 FLOPs 측정 구현
    return None


def profile_model(
    model: torch.nn.Module,
    inputs: Iterable[Dict[str, Any]],
    *,
    latency_cfg: Optional[LatencyConfig] = None,
) -> Dict[str, float]:
    """Latency / FPS / VRAM / Params / FLOPs 종합 측정."""
    cfg = latency_cfg or LatencyConfig()
    results: Dict[str, float] = {}

    with _cuda_nvml_context():
        latency_stats = measure_latency(
            model,
            inputs,
            warmup_iters=cfg.warmup_iters,
            measure_iters=cfg.measure_iters,
        )

    results.update(latency_stats)
    results.update(measure_params(model))

    # FLOPs는 외부 유틸 연동 시 구현 예정
    results["flops"] = float("nan")

    if torch.cuda.is_available():
        results["vram_peak_bytes"] = float(torch.cuda.max_memory_allocated())

    return results


__all__ = [
    "LatencyConfig",
    "measure_latency",
    "measure_params",
    "estimate_flops",
    "profile_model",
]

