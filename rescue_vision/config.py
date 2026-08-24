from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    validate_config(data)
    return data


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    validate_config(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(target)


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("classes"), dict) or not config["classes"]:
        raise ValueError("配置必须包含非空classes对象")
    for name, profile in config["classes"].items():
        if not isinstance(profile.get("references", []), list):
            raise ValueError(f"{name}.references必须是数组")
        variants = [("基础参考", profile)] + [
            (str(item.get("reference_name", f"参考{index}")), item)
            for index, item in enumerate(profile.get("references", []), start=1)
        ]
        for reference_name, variant in variants:
            for space in ("hsv", "lab"):
                values = variant.get(space)
                if not isinstance(values, list) or len(values) != 6:
                    raise ValueError(f"{name}.{reference_name}.{space}必须是6个整数")
            morph = variant.get("morphology", {})
            for key in ("open", "close"):
                value = int(morph.get(key, 0))
                if value < 0 or value > 31:
                    raise ValueError(f"{name}.{reference_name}.morphology.{key}超出0..31")


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)
