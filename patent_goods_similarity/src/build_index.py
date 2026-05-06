"""1회 실행하여 cache/ 에 임베딩과 인덱스를 만드는 py입니다.
중간 산출물들은 캐싱하여 재사용할 수 있도록 했습니다.

산출물 (cache_dir):
    patents.parquet         : 번호, 명칭, ipc_list, ksic_prefixes
    goods_unique.parquet    : 지정상품 고유 텍스트 + unique id
    goods_index.pkl         : ksic5 → set[unique text id]  (dict)
    patent_emb.npy          : (N_patent, dim) float32 (L2-normalized)
    goods_emb.npy           : (N_goods,  dim) float32 (L2-normalized)
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from .candidate import (
    GoodsIndex,
    build_goods_index,
    build_ipc_prefix_index,
    patent_ipc_to_ksic_prefixes,
)
from .config import Config, load_config
from .data_loader import load_goods, load_ipc_ksic_linkage, load_patents
from .embedder import OpenAIEmbedder


def _save_goods_index(goods_idx: GoodsIndex, cache_dir: Path) -> None:
    pd.DataFrame({"text": goods_idx.texts}).to_parquet(
        cache_dir / "goods_unique.parquet", index=True
    )
    with open(cache_dir / "goods_index.pkl", "wb") as f:
        pickle.dump(
            {
                "ksic5_to_text_ids": goods_idx.ksic5_to_text_ids,
                "all_ksic5": goods_idx.all_ksic5,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def build(cfg: Config | None = None, force: bool = False) -> None:
    cfg = cfg or load_config()
    cache = cfg.paths.cache_dir
    t0 = time.time()

    print("[1/5] 통합_CSV.csv 로딩...")
    patents = load_patents(cfg.paths)
    print(f"      특허 수: {len(patents):,}")

    print("[2/5] KSIC-IPC 연계표 로딩 + IPC prefix 인덱스 생성...")
    linkage = load_ipc_ksic_linkage(cfg.paths)
    ipc_index = build_ipc_prefix_index(linkage)
    print(f"      연계 규칙 수: {len(ipc_index):,}")

    print("[3/5] 각 특허의 KSIC prefix 집합 계산...")
    ksic_lists: list[list[str]] = []
    empty_cnt = 0
    for ipc_list in patents["ipc_list"]:
        prefixes = patent_ipc_to_ksic_prefixes(ipc_list, ipc_index)
        if not prefixes:
            empty_cnt += 1
        ksic_lists.append(sorted(prefixes))
    patents["ksic_prefixes"] = ksic_lists
    print(f"      KSIC 매칭 0건인 특허: {empty_cnt:,} (전체의 {empty_cnt/len(patents)*100:.1f}%)")
    patents.to_parquet(cache / "patents.parquet", index=False)

    print("[4/5] KSIC-지정상품 연계표 로딩 + 고유 지정상품 인덱스 생성...")
    goods_df = load_goods(cfg.paths)
    print(f"      (ksic5, 지정상품) 쌍: {len(goods_df):,}")
    goods_idx = build_goods_index(goods_df)
    print(f"      고유 지정상품 수: {len(goods_idx.texts):,}")
    print(f"      고유 5-digit KSIC 수: {len(goods_idx.all_ksic5):,}")
    _save_goods_index(goods_idx, cache)

    print(f"[5/5] 임베딩 (OpenAI text-embedding-3-small, API 키 {len(cfg.openai_api_keys)}개 병렬)...")
    embedder = OpenAIEmbedder(cfg)

    if force:
        for fn in ("goods_emb.npy", "patent_emb.npy"):
            (cache / fn).unlink(missing_ok=True)

    print(f"      → 지정상품 {len(goods_idx.texts):,}건 임베딩")
    embedder.embed_texts(
        goods_idx.texts,
        cache / "goods_emb.npy",
        desc="goods",
    )
    print(f"      → 특허 명칭 {len(patents):,}건 임베딩")
    embedder.embed_texts(
        patents["명칭"].tolist(),
        cache / "patent_emb.npy",
        desc="patents",
    )

    print(f"\n완료. 총 {time.time()-t0:.1f}초")
    print(f"산출물 위치: {cache}")
