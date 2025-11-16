"""모델 효율성 측정 도구."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch

from methods.base import ForwardResult


@dataclass
class LatencyConfig:
    warmup_iters: int = 1000
    measure_iters: int = 1000
    repeat: int = 3


LOGGER = logging.getLogger(__name__)


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


def measure_throughput(
    model: torch.nn.Module,
    inputs: Iterable[Dict[str, Any]],
    *,
    num_passes: int = 1000,
) -> Dict[str, float]:
    """
    Throughput 측정 (논문 기준).
    
    Args:
        model: 추론할 모델
        inputs: 입력 데이터 iterator (batch size 16)
        num_passes: 측정할 forward pass 횟수 (기본값: 1000)
    
    Returns:
        throughput: Throughput (samples/sec). Throughput = (batch_size * num_passes) / total_time
    """
    iterator = iter(inputs)
    model.eval()
    
    # Warm-up은 latency 측정과 달리 논문에서 명시되지 않았지만, 안정적인 측정을 위해 수행
    # 100번 정도 warm-up
    for _ in range(100):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        _run_inference(model, batch)
    
    _synchronize()
    
    # 1000 pass의 총 시간 측정
    start_time = time.perf_counter()
    total_samples = 0
    
    for _ in range(num_passes):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        
        # 배치 크기 확인 (논문 기준: batch size 16)
        if isinstance(batch, dict) and "image" in batch:
            batch_size = batch["image"].shape[0] if hasattr(batch["image"], "shape") else 1
        else:
            batch_size = 1
        
        _run_inference(model, batch)
        total_samples += batch_size
    
    _synchronize()
    total_time = time.perf_counter() - start_time  # seconds
    
    # Throughput = total_samples / total_time
    throughput = float(total_samples / total_time) if total_time > 0 else 0.0
    
    return {
        "throughput_samples_per_sec": throughput,
        "throughput_total_samples": float(total_samples),
        "throughput_total_time_sec": total_time,
    }


def measure_gpu_memory(
    model: torch.nn.Module,
    inputs: Iterable[Dict[str, Any]],
    *,
    num_iters: int = 1000,
) -> Dict[str, float]:
    """
    GPU Memory 측정 (논문 기준: PyTorch profiler로 peak reserved GPU memory).
    
    Args:
        model: 측정할 모델
        inputs: 입력 데이터 iterator (batch size 1)
        num_iters: 측정 횟수 (기본값: 1000)
    
    Returns:
        peak_reserved_bytes: Peak reserved GPU memory (bytes), 1000회 평균
    """
    if not torch.cuda.is_available():
        return {"gpu_peak_reserved_bytes": float("nan")}
    
    iterator = iter(inputs)
    model.eval()
    
    # Warm-up (100회)
    for _ in range(100):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        _run_inference(model, batch)
    
    _synchronize()
    torch.cuda.empty_cache()
    
    # 1000회 측정
    memory_readings = []
    
    for _ in range(num_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        
        torch.cuda.reset_peak_memory_stats()
        _run_inference(model, batch)
        _synchronize()
        
        # Peak reserved memory 측정
        peak_reserved = torch.cuda.max_memory_reserved()
        memory_readings.append(float(peak_reserved))
    
    # 평균 계산
    if memory_readings:
        memory_tensor = torch.tensor(memory_readings)
        avg_peak_reserved = float(memory_tensor.mean())
        std_peak_reserved = float(memory_tensor.std(unbiased=False))
    else:
        avg_peak_reserved = float("nan")
        std_peak_reserved = float("nan")
    
    return {
        "gpu_peak_reserved_bytes": avg_peak_reserved,
        "gpu_peak_reserved_bytes_std": std_peak_reserved,
    }


def measure_params(model: torch.nn.Module) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": float(total_params),
        "params_trainable": float(trainable_params),
    }


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        device_attr = getattr(model, "device")
        if isinstance(device_attr, torch.device):
            return device_attr
        if isinstance(device_attr, str):
            return torch.device(device_attr)

    for param in model.parameters():
        return param.device

    for buffer in model.buffers():
        return buffer.device

    return torch.device("cpu")


def _to_device(sample: Any, device: torch.device) -> Any:
    if isinstance(sample, torch.Tensor):
        return sample.to(device=device, non_blocking=True)
    if isinstance(sample, dict):
        return {key: _to_device(value, device) for key, value in sample.items()}
    if isinstance(sample, tuple):
        return tuple(_to_device(item, device) for item in sample)
    if isinstance(sample, list):
        return [_to_device(item, device) for item in sample]
    return sample


def _flatten_forward_output(output: Any) -> Any:
    if isinstance(output, ForwardResult):
        tensors = []
        for attr in ("image_scores", "pixel_scores", "embeddings"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        if not tensors:
            raise ValueError("ForwardResult에 텐서가 없어 FLOPs 계산에 사용할 수 없습니다.")
        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)
    return output


class _FlopsForwardWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, batch: Dict[str, Any]) -> Any:  # type: ignore[override]
        output = self.module.forward(batch)
        return _flatten_forward_output(output)


def estimate_flops(model: torch.nn.Module, example_batch: Dict[str, Any]) -> Optional[float]:
    """FLOPs 추정 (torch.profiler 사용) - TODO: TinyCLIP 예시에 맞게 구현."""
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except ImportError:
        LOGGER.warning("fvcore를 import 할 수 없어 FLOPs를 계산하지 않습니다.")
        return None

    if not example_batch:
        LOGGER.warning("FLOPs 계산을 위한 배치가 비어 있습니다.")
        return None

    device = _infer_model_device(model)
    batch_on_device = _to_device(example_batch, device)

    was_training = model.training
    if was_training:
        model.eval()

    try:
        with torch.no_grad():
            wrapped = _FlopsForwardWrapper(model)
            wrapped.eval()
            analysis = FlopCountAnalysis(wrapped, (batch_on_device,))
            flops = float(analysis.total())
    except RuntimeError as exc:
        LOGGER.warning("FLOPs 계산 중 RuntimeError 발생: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - 방어적 처리
        LOGGER.warning("FLOPs 계산 실패: %s", exc)
        return None
    finally:
        if was_training:
            model.train()

    return flops


def profile_model(
    model: torch.nn.Module,
    inputs: Iterable[Dict[str, Any]],
    *,
    latency_cfg: Optional[LatencyConfig] = None,
    throughput_inputs: Optional[Iterable[Dict[str, Any]]] = None,
    enable_gpu_memory_measurement: bool = True,
    throughput_num_passes: int = 1000,
    gpu_memory_num_iters: int = 1000,
) -> Dict[str, float]:
    """
    Latency / Throughput / FPS / VRAM / Params / FLOPs 종합 측정.
    
    Args:
        model: 측정할 모델
        inputs: Latency 측정용 입력 (batch size 1)
        latency_cfg: Latency 측정 설정
        throughput_inputs: Throughput 측정용 입력 (batch size 16). None이면 throughput 측정 안함.
        enable_gpu_memory_measurement: GPU 메모리 측정 여부 (논문 기준: batch size 1, 1000회 평균)
        throughput_num_passes: Throughput 측정 횟수 (기본값: 1000)
        gpu_memory_num_iters: GPU 메모리 측정 횟수 (기본값: 1000)
    """
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
    
    # Throughput 측정 (논문 기준: batch size 16, 1000 pass)
    if throughput_inputs is not None:
        throughput_stats = measure_throughput(
            model,
            throughput_inputs,
            num_passes=throughput_num_passes,
        )
        results.update(throughput_stats)
    
    # GPU Memory 측정 (논문 기준: batch size 1, 1000회 평균, peak reserved memory)
    if enable_gpu_memory_measurement and torch.cuda.is_available():
        # inputs를 재사용하기 위해 iterator 생성
        memory_inputs = inputs
        gpu_memory_stats = measure_gpu_memory(
            model,
            memory_inputs,
            num_iters=gpu_memory_num_iters,
        )
        results.update(gpu_memory_stats)
    
    results.update(measure_params(model))

    results["flops"] = float("nan")

    # 기존 allocated memory도 유지 (하위 호환성)
    if torch.cuda.is_available():
        results["vram_peak_bytes"] = float(torch.cuda.max_memory_allocated())

    return results


__all__ = [
    "LatencyConfig",
    "measure_latency",
    "measure_throughput",
    "measure_gpu_memory",
    "measure_params",
    "estimate_flops",
    "profile_model",
]

