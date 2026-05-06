"""설정 파일과 .env 로딩."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    data_dir: Path
    patent_csv: Path
    ipc_ksic_xlsx: Path
    ksic_goods_xlsx: Path
    cache_dir: Path
    output_dir: Path


@dataclass
class EmbeddingConfig:
    model: str
    dimensions: int
    batch_size: int
    max_retries: int
    request_timeout: int


@dataclass
class SearchConfig:
    top_k: int


@dataclass
class Config:
    paths: Paths
    embedding: EmbeddingConfig
    search: SearchConfig
    openai_api_keys: list[str] = field(default_factory=list)


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


_KEY_NAME_RE = re.compile(r"^OPENAI_API_KEY(\d*)$")


def _collect_api_keys() -> list[str]:
    """.env에서 OPENAI_API_KEY 모두 수집.
    숫자 접미사 오름차순으로 정렬
    """
    found: list[tuple[int, str]] = []
    for name, val in os.environ.items():
        m = _KEY_NAME_RE.match(name)
        if not m:
            continue
        v = (val or "").strip()
        if not v or v.startswith("sk-REPLACE"):
            continue
        idx = int(m.group(1)) if m.group(1) else 0
        found.append((idx, v))
    found.sort(key=lambda x: x[0])
    seen: set[str] = set()
    keys: list[str] = []
    for _, k in found:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def load_config(config_path: Path | str | None = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    api_keys = _collect_api_keys()
    if not api_keys:
        raise RuntimeError(
            "OPENAI_API_KEY 가 설정되지 않았습니다. "
            f"{PROJECT_ROOT / '.env'} 파일에 OPENAI_API_KEY1=... 형태로 키를 입력하세요."
        )

    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = _resolve(PROJECT_ROOT, raw["paths"]["data_dir"])
    paths = Paths(
        data_dir=data_dir,
        patent_csv=_resolve(data_dir, raw["paths"]["patent_csv"]),
        ipc_ksic_xlsx=_resolve(data_dir, raw["paths"]["ipc_ksic_xlsx"]),
        ksic_goods_xlsx=_resolve(data_dir, raw["paths"]["ksic_goods_xlsx"]),
        cache_dir=_resolve(PROJECT_ROOT, raw["paths"]["cache_dir"]),
        output_dir=_resolve(PROJECT_ROOT, raw["paths"]["output_dir"]),
    )
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        paths=paths,
        embedding=EmbeddingConfig(**raw["embedding"]),
        search=SearchConfig(**raw["search"]),
        openai_api_keys=api_keys,
    )
