"""특허 → 후보 지정상품 인덱스 매핑 빌더 py입니다. 

특허 IPC
→ IPC-KSIC 연계표
→ KSIC prefix
→ 지정상품 후보군

구체적으로, 
   특허 IPC 리스트 (정규화된 'G06F 5/00' 형식)
   → 연계표 prefix 매칭 → KSIC 셀 집합
   → expand_ksic_cell 결과를 union → KSIC prefix 집합 (2~4 자리)
   → 미리 만들어둔 '5자리 ksic → unique_goods_ids' 인덱스 위에서
       prefix.startswith 매칭 → unique_goods_ids 합집합 = 후보군
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GoodsIndex:
    """임베딩과 후보 매칭을 위한 인덱스 구조.

    - texts:               unique 지정상품 문자열 (임베딩 1행과 1:1 대응)
    - ksic5_to_text_ids:   '00000' 5자리 KSIC → set[unique text id]
    - all_ksic5:           모든 5자리 코드 (prefix 매칭 시 스캔 대상)
    """
    texts: list[str]
    ksic5_to_text_ids: dict[str, set[int]]
    all_ksic5: list[str]


def build_goods_index(goods_df: pd.DataFrame) -> GoodsIndex:
    """load_goods() 결과 → GoodsIndex.
    같은 지정상품 텍스트가 여러 KSIC 코드에 묶여 있으면 모두 보존합니다.
    KSIC와 지정상품을 연결하는 로직임다.
    """
    unique_texts = pd.unique(goods_df["지정상품"])
    text_to_id = {t: i for i, t in enumerate(unique_texts)}
    ksic5_to_text_ids: dict[str, set[int]] = defaultdict(set)
    for ksic5, text in zip(goods_df["ksic5"].values, goods_df["지정상품"].values):
        ksic5_to_text_ids[ksic5].add(text_to_id[text])
    return GoodsIndex(
        texts=list(unique_texts),
        ksic5_to_text_ids=dict(ksic5_to_text_ids),
        all_ksic5=sorted(ksic5_to_text_ids.keys()),
    )


def build_ipc_prefix_index(linkage_df: pd.DataFrame) -> list[tuple[str, set[str]]]:
    """연계표 행을 [(ipc_prefix, ksic_set), ...] 리스트로 (긴 prefix 우선).
    IPC ↔ KSIC 연결 인덱스 생성 로직입니다.
    요런 식
    [
        ("G06F", {"620","631"}),
        ("G06F 5/", {"6201"}),
        ("H05K", {"262"})
    ]
    """
    rows = list(zip(linkage_df["ipc_prefix"].values, linkage_df["ksic_set"].values))
    rows.sort(key=lambda x: -len(x[0]))
    return rows


def patent_ipc_to_ksic_prefixes(
    patent_ipc_list: list[str],
    ipc_index: list[tuple[str, set[str]]],
) -> set[str]:
    """특허 IPC 리스트 → KSIC prefix 집합 (2~5자리)."""
    out: set[str] = set()
    for n in patent_ipc_list:
        for prefix, ksic_set in ipc_index:
            if n.startswith(prefix):
                out |= ksic_set
    return out


def candidate_text_ids(
    ksic_prefixes: set[str],
    goods_index: GoodsIndex,
) -> np.ndarray:
    """KSIC prefix(5자리) 집합 → 후보 unique 지정상품 인덱스 배열 (정렬된 거 기준)."""
    if not ksic_prefixes:
        return np.array([], dtype=np.int64)
    matched: set[int] = set()
    for code5, ids in goods_index.ksic5_to_text_ids.items():
        for p in ksic_prefixes:
            if code5.startswith(p):
                matched |= ids
                break
    return np.array(sorted(matched), dtype=np.int64)
