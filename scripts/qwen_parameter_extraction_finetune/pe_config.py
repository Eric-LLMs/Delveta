#!/usr/bin/env python3
"""Shared configuration access for the PE fine-tune stages.

The ONE place that knows how :mod:`config.yaml` resolves paths, so the builder,
the trainer, the evaluator and the exporter cannot disagree:

* ``source.*``            -> relative to the REPOSITORY ROOT (it points at the
                             repo's ``data/`` and ``logs/`` trees, which live
                             outside this self-contained script directory);
* every other section      -> relative to THIS SCRIPT DIRECTORY (``output.*`` /
  (``output`` / ``train`` /     ``train`` / ``eval`` / ``export``), so the whole
   ``eval`` / ``export``)      project stays under
                             ``scripts/qwen_parameter_extraction_finetune/``.

Stdlib + PyYAML only — never imports torch / transformers / peft — so a config
check runs in any environment and can gate a real training launch.
"""
from __future__ import annotations

from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"

# The only section whose paths are repo-root-relative.
SOURCE_SECTION = "source"


class ConfigError(Exception):
    """A missing / malformed config, or a path that escapes its stated root.
    Fail loud — never resolve a training/eval path by accident."""


def load_config(path: str | Path | None = None) -> dict:
    """Read and parse ``config.yaml`` (default: the one beside this module)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_file():
        raise ConfigError(f"missing config file: {p}")
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # malformed YAML is fatal, not defaulted
        raise ConfigError(f"config {p} is not valid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"config {p} did not parse to a mapping")
    return cfg


def source_path(cfg: dict, key: str) -> Path:
    """Resolve ``source.<key>`` against the repository root."""
    section = cfg.get(SOURCE_SECTION)
    if not isinstance(section, dict) or key not in section:
        raise ConfigError(f"config is missing source.{key}")
    return REPO_ROOT / str(section[key])


def local_path(cfg: dict, section: str, key: str) -> Path:
    """Resolve ``<section>.<key>`` against this script directory."""
    if section == SOURCE_SECTION:
        raise ConfigError("use source_path() for source.* entries")
    sub = cfg.get(section)
    if not isinstance(sub, dict) or key not in sub:
        raise ConfigError(f"config is missing {section}.{key}")
    return SCRIPT_DIR / str(sub[key])
