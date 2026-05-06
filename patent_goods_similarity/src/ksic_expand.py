"""KSIC 코드 표기 확장 py 입니다.

연계표의 'KSIC(제11차)' 셀은 다음 형태를 가질 수 있더군요.
아래처럼 처리하게끔 했습니다.

  - 단일값:        '10', '2041', '211'
  - 범위:          '01~03'   → {'01','02','03'}
  - 콤마 다중값:   '211, 212' → {'211','212'}

지정상품 표의 'ksic 세세분류(11차)' (int)는 5자리 zero-pad 후 '시작 일치'로 매칭합니다.
  - 예: 2041 → '02041' → '02041'
"""
from __future__ import annotations


def expand_ksic_cell(raw: object) -> set[str]:
    """연계표 KSIC 셀 값 → 'startswith' 매칭에 쓸 prefix 문자열 집합."""
    """'01~03' → {'01', '02', '03'}"""
    if raw is None:
        return set()
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return set()
    out: set[str] = set()
    for part in s.split(","):
        token = part.strip()
        if not token:
            continue
        if "~" in token:
            a, b = [t.strip() for t in token.split("~", 1)]
            if a.isdigit() and b.isdigit():
                width = max(len(a), len(b))
                for i in range(int(a), int(b) + 1):
                    out.add(str(i).zfill(width))
            else:
                out.add(token)
        else:
            out.add(token)
    return out


def ksic_int_to_5digit(code: int | str) -> str:
    """세세분류(11차)의 int를 5자리 zero-pad 문자열로."""
    return str(int(code)).zfill(5)
