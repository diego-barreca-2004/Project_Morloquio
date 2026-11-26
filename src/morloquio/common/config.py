"""Parametri degli stadi della pipeline (``configs/pipeline/<nome>.yaml``)."""

from __future__ import annotations

import yaml

from morloquio.common.paths import repo_root


def pipeline_config(name: str) -> dict:
    return yaml.safe_load((repo_root() / "configs" / "pipeline" / f"{name}.yaml").read_text(encoding="utf-8"))
