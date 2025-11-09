"""YAML 기반 실험 설정 로더 및 머지 유틸리티."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _ensure_project_path(path: Union[str, Path]) -> Path:
    """프로젝트 루트를 기준으로 상대 경로를 절대 경로로 변환."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _deep_update(base: MutableMapping[str, Any], overrides: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """중첩 딕셔너리를 재귀적으로 병합 (override 우선)."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config 파일은 dict 형태여야 합니다: {path}")
    return data


def _collect_inherited_configs(paths: Iterable[Union[str, Path]]) -> Dict[str, Any]:
    """inherit_from에 명시된 설정 파일을 순서대로 로드 후 병합."""
    merged: Dict[str, Any] = {}
    for relative_path in paths:
        cfg_path = _ensure_project_path(relative_path)
        data = _load_yaml(cfg_path)
        parent_refs = data.pop("inherit_from", None)
        if parent_refs:
            parent_iter = parent_refs if isinstance(parent_refs, Iterable) and not isinstance(parent_refs, (str, bytes)) else [parent_refs]
            parent_cfg = _collect_inherited_configs(parent_iter)
            merged = _deep_update(merged, parent_cfg)
        merged = _deep_update(merged, data)
    return merged


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[MutableMapping[str, Any]] = None,
    *,
    resolve_paths: bool = True,
) -> Dict[str, Any]:
    """YAML 설정을 로드하고 상속 및 추가 override를 적용."""
    cfg_path = _ensure_project_path(config_path)
    config_data = _load_yaml(cfg_path)

    parent = config_data.pop("inherit_from", None)
    if parent:
        parent_refs = parent if isinstance(parent, Iterable) and not isinstance(parent, (str, bytes)) else [parent]
        merged = _collect_inherited_configs(parent_refs)
    else:
        merged = {}

    merged = _deep_update(merged, config_data)

    if overrides:
        merged = _deep_update(merged, deepcopy(overrides))

    if resolve_paths:
        _resolve_relative_paths(merged)

    return merged


def _resolve_relative_paths(node: MutableMapping[str, Any]) -> None:
    """딕셔너리 내 `*_root`, `*_path` 키에 대해 상대 경로를 절대 경로로 변환."""
    for key, value in list(node.items()):
        if isinstance(value, MutableMapping):
            _resolve_relative_paths(value)
        elif isinstance(value, str) and key.lower().endswith(("path", "root", "dir")):
            node[key] = str(_ensure_project_path(value))


def save_config_copy(config: Dict[str, Any], target_dir: Union[str, Path], filename: str = "config.yaml") -> Path:
    """현재 실험 설정을 target_dir에 저장."""
    target_path = _ensure_project_path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    output_path = target_path / filename
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return output_path


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_ROOT",
    "load_config",
    "save_config_copy",
]

