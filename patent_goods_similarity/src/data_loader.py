"""원본 CSV/XLSX 로딩."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Paths
from .ipc_norm import linkage_pattern_to_prefix, parse_patent_ipc_list
from .ksic_expand import expand_ksic_cell, ksic_int_to_5digit
from .preprocess import clean_title


PATENT_USECOLS = ["번호", "명칭(원문)", "국제특허분류"]


def load_patents(paths: Paths) -> pd.DataFrame:
    """통합_CSV.csv → DataFrame[번호, 명칭, ipc_list].
    1~4행은 메타데이터, 5행이 헤더이므로 skiprows=4.
    <- 추후 초록이라든지, 청구항이라든지 다른 컬럼이 필요해지면 usecols 조정하면 됩니다!~
    """
    df = pd.read_csv(
        paths.patent_csv,
        skiprows=4,
        encoding="utf-8-sig",
        usecols=PATENT_USECOLS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    df = df.rename(columns={"명칭(원문)": "명칭"})
    df["명칭"] = df["명칭"].map(clean_title)
    df["ipc_list"] = df["국제특허분류"].map(parse_patent_ipc_list)
    df = df.drop(columns=["국제특허분류"])
    df = df[df["번호"].astype(str).str.len() > 0].reset_index(drop=True)
    if df["번호"].duplicated().any():
        df = df.drop_duplicates(subset=["번호"], keep="first").reset_index(drop=True)
    return df


def load_ipc_ksic_linkage(paths: Paths) -> pd.DataFrame:
    """KSIC-특허IPC 연계표 → DataFrame[ipc_prefix, ksic_set].
    각 행은 IPC 패턴 한 개 → KSIC(제11차) 셀 한 개. 셀의 콤마/범위는 그대로 보존하고,
    prefix 컬럼은 '시작 일치' 비교에 쓸 형태로 만들어둡니다. 
    """
    df = pd.read_excel(
        paths.ipc_ksic_xlsx,
        sheet_name=0,
        usecols=["KSIC(제11차)", "특허코드(IPC)"],
    )
    df = df.dropna(subset=["KSIC(제11차)", "특허코드(IPC)"])
    df["ipc_prefix"] = df["특허코드(IPC)"].astype(str).map(linkage_pattern_to_prefix)
    df = df[df["ipc_prefix"].str.len() > 0].reset_index(drop=True)
    df["ksic_set"] = df["KSIC(제11차)"].map(expand_ksic_cell)
    return df[["ipc_prefix", "ksic_set"]]


def load_goods(paths: Paths) -> pd.DataFrame:
    """KSIC-지정상품 연계표 → DataFrame[ksic5, 지정상품].
    중복 제거는 호출자에서 처리. 여기서는 (ksic5, 텍스트) 페어 그대로 반환하도록 했습니다.
    """
    df = pd.read_excel(
        paths.ksic_goods_xlsx,
        sheet_name=0,
        usecols=["ksic 세세분류(11차)", "지정상품(국문)"],
    )
    df = df.dropna(subset=["ksic 세세분류(11차)", "지정상품(국문)"])
    df["ksic5"] = df["ksic 세세분류(11차)"].astype(int).map(ksic_int_to_5digit)
    df["지정상품"] = df["지정상품(국문)"].astype(str).map(clean_title)
    df = df[df["지정상품"].str.len() > 0]
    return df[["ksic5", "지정상품"]].reset_index(drop=True)
