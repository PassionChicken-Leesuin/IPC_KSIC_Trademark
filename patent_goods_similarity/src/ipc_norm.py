"""IPC 코드 정규화 및 매칭 py 입니다.

통합 CSV의 국제특허분류는 'G06F-005/00' 처럼 dash와 zero-pad가 들어간 형식인데,
KSIC-IPC 연계표의 IPC 패턴은 'G06F%' 또는 'H05K 11/%' 처럼 % 이 포함되어 있었습니다.

비교 시 양측을 표준 IPC 형식 'G06F 5/00' 으로 정규화한 뒤
'patent_norm'.startswith('linkage_prefix') 로 매칭하도록 했습니다. 
"""
from __future__ import annotations

import re
from typing import Iterable

# 'G06F-005/00', 'G06F 5/00', 'G06F005/00' 등 여러 형태를 같게 인식하는 정규식
_PATENT_IPC_RE = re.compile(
    r"^\s*([A-Z])\s*(\d{1,2})\s*([A-Z])\s*[-\s]?\s*(\d+)\s*/\s*(\d+)\s*$"
)
# 'G06F%' 처럼 subclass 만 있는 경우
_PATENT_IPC_SUBCLASS_RE = re.compile(r"^\s*([A-Z])\s*(\d{1,2})\s*([A-Z])\s*$")


def normalize_patent_ipc(code: str) -> str | None:
    """KIPRIS 형식의 IPC 코드를 표준형 'G06F 5/00' 으로 변환.

    파싱 실패 시 None.
    """
    if not isinstance(code, str):
        return None
    s = code.strip()
    if not s:
        return None
    m = _PATENT_IPC_RE.match(s)
    if m:
        section, cls, subcls, main, sub = m.groups()
        return f"{section}{cls.zfill(2)}{subcls} {int(main)}/{sub}"
    m2 = _PATENT_IPC_SUBCLASS_RE.match(s)
    if m2:
        section, cls, subcls = m2.groups()
        return f"{section}{cls.zfill(2)}{subcls}"
    return None


def parse_patent_ipc_list(raw: object) -> list[str]:
    """국제특허분류 셀 → 정규화된 IPC 코드 리스트."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in s.split(","):
        n = normalize_patent_ipc(part)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def linkage_pattern_to_prefix(pattern: str) -> str:
    """KSIC-IPC 연계표의 IPC 패턴 → '시작 일치'에 쓸 prefix 문자열.

    'A01D%'   -> 'A01D'
    'H05K 11/%' -> 'H05K 11/'
    """
    if not isinstance(pattern, str):
        return ""
    return pattern.strip().rstrip("%").rstrip()  # rstrip 후 trailing whitespace 제거 (slash 보존)


def match_patent_ipc_to_prefixes(patent_ipc: str, prefixes: Iterable[str]) -> list[str]:
    """정규화된 patent IPC 한 건과 매칭되는 prefix 들."""
    return [p for p in prefixes if p and patent_ipc.startswith(p)]
