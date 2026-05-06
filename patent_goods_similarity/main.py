"""특허 명칭 ↔ 지정상품 유사도 시스템 CLI.

사용법:
    # 1. 인덱스/임베딩 빌드 (최초 1회 또는 데이터 갱신 시)
    python main.py build
    python main.py build --force          # 임베딩까지 재생성

    # 2. 검색은 search.ipynb 노트북에서 수행

    # 3. 키 검증
    python main.py check
"""
from __future__ import annotations

import argparse
import sys

from src.build_index import build
from src.config import load_config


def cmd_build(args: argparse.Namespace) -> int:
    cfg = load_config()
    print(f"OpenAI API 키 {len(cfg.openai_api_keys)}개 감지 → 병렬 임베딩 활성화")
    build(cfg, force=args.force)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config()
    print(f"OpenAI API 키 {len(cfg.openai_api_keys)}개 로드됨")
    print(f"  - 모델     : {cfg.embedding.model}")
    print(f"  - 차원     : {cfg.embedding.dimensions}")
    print(f"  - 배치 크기: {cfg.embedding.batch_size}")
    print(f"  - top_k    : {cfg.search.top_k}")
    print(f"  - 데이터 디렉터리: {cfg.paths.data_dir}")
    for label, p in [
        ("통합_CSV.csv", cfg.paths.patent_csv),
        ("KSIC-IPC 연계표", cfg.paths.ipc_ksic_xlsx),
        ("KSIC-지정상품 연계표", cfg.paths.ksic_goods_xlsx),
    ]:
        print(f"  - {label}: {'OK' if p.exists() else '없음'} ({p})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patent-goods-similarity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="인덱스 + 임베딩 생성")
    p_build.add_argument("--force", action="store_true", help="기존 임베딩 무시하고 재생성")
    p_build.set_defaults(func=cmd_build)

    p_check = sub.add_parser("check", help="설정/경로/API키 확인")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
