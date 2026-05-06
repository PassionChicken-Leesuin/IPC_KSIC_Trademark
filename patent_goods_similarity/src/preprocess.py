"""
특허 명칭 전처리 py 입니다. 

text-embedding-3-small은 다국어를 모두 처리하므로
'한글{ENGLISH}' 혼합 형식을 그대로 입력으로 사용하도록 했습니다.
공백 정리와 NaN 방어만 수행하는 코드입니다. 
"""
from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def clean_title(text: object) -> str:
    if text is None:
        return ""
    s = str(text)
    if s.strip().lower() in {"", "nan", "none"}:
        return ""
    s = s.replace("　", " ").replace("\xa0", " ")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s
