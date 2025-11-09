"""프롬프트 템플릿 관리."""

from __future__ import annotations

from typing import Dict, List


PROMPT_LIBRARY: Dict[str, List[str]] = {
    "winclip_default": [
        "a photo of a {} object.",
        "a close-up photo of a {} object.",
        "a cropped photo of a {} object.",
        "a bright photo of a {} object.",
        "a dark photo of a {} object.",
    ],
    "anomaly": [
        "a damaged {}.",
        "a defective {}.",
        "a broken {}.",
        "a {} with anomalies.",
    ],
}


def get_prompt_templates(name: str) -> List[str]:
    if name not in PROMPT_LIBRARY:
        raise KeyError(f"등록되지 않은 프롬프트 템플릿입니다: {name}")
    return PROMPT_LIBRARY[name]


def register_prompt(name: str, templates: List[str]) -> None:
    PROMPT_LIBRARY[name] = templates


__all__ = ["get_prompt_templates", "register_prompt", "PROMPT_LIBRARY"]

