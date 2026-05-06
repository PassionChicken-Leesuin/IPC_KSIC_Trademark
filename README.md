# IPC ↔ KSIC ↔ 상표권 지정상품 유사도 시스템

특허의 **IPC(국제특허분류)** 와 상표권 **지정상품**을 KSIC(한국표준산업분류) 연계표를 통해 매칭하고, OpenAI 임베딩으로 의미 유사도까지 결합해 특허 명칭과 가장 유사한 지정상품 후보를 검색하는 파이프라인입니다.

서울대학교 지재연(지식재산연구) 과제로 진행되었습니다.

## 파이프라인

```
특허 IPC
   └─ KSIC-IPC 연계표 ──→ KSIC prefix 집합
                              └─ KSIC-지정상품 연계표 ──→ 후보 지정상품군
                                                              └─ OpenAI 임베딩 코사인 유사도 ──→ Top-K
```

1. 특허 IPC 리스트를 정규화(`G06F 5/00` 형식)
2. **KSIC-특허IPC 연계표**의 prefix 매칭으로 KSIC 코드 셀 집합 생성
3. KSIC 코드를 2~5자리 prefix로 확장
4. **KSIC-지정상품 연계표**에서 prefix.startswith 매칭으로 후보 지정상품군 추출
5. 특허 명칭과 후보 지정상품을 `text-embedding-3-small`로 임베딩하여 코사인 유사도 Top-K 반환

## 디렉터리 구조

```
.
├── data/                                # (gitignore) 원본 데이터
│   ├── 통합_CSV.csv                      # 특허 데이터 (~546MB)
│   ├── KSIC-특허IPC 연계표.xlsx
│   └── KSIC-지정상품 연계표.xlsx
└── patent_goods_similarity/
    ├── main.py                          # CLI 진입점
    ├── config.yaml                      # 경로/임베딩/검색 설정
    ├── requirements.txt
    ├── search.ipynb                     # 검색 실행 노트북
    ├── src/
    │   ├── config.py                    # 설정 + .env 로딩
    │   ├── data_loader.py               # CSV/XLSX 로딩
    │   ├── preprocess.py                # 텍스트 정리
    │   ├── ipc_norm.py                  # IPC 코드 정규화
    │   ├── ksic_expand.py               # KSIC 셀 확장
    │   ├── candidate.py                 # IPC→KSIC→지정상품 후보 인덱스
    │   ├── embedder.py                  # OpenAI 임베딩 (다중 키 병렬)
    │   └── build_index.py               # 빌드 오케스트레이션
    ├── cache/                           # (gitignore) 임베딩/인덱스 캐시
    └── output/                          # (gitignore) 검색 결과 CSV
```

## 설치

```bash
cd patent_goods_similarity
pip install -r requirements.txt
```

## 데이터 준비

`data/` 폴더에 다음 3개 파일을 배치합니다 (저장소에 포함되지 않음).

| 파일 | 설명 |
|---|---|
| `통합_CSV.csv` | 특허 원본 데이터 (`번호`, `명칭(원문)`, `국제특허분류` 컬럼 사용; 1~4행 메타, 5행 헤더) |
| `KSIC-특허IPC 연계표.xlsx` | IPC 패턴 ↔ KSIC(제11차) 매핑 |
| `KSIC-지정상품 연계표.xlsx` | KSIC 세세분류(11차) ↔ 지정상품(국문) |

## OpenAI API 키

`patent_goods_similarity/.env` 파일에 다음 형식으로 입력합니다. 다중 키를 등록하면 임베딩 호출이 키별로 병렬 처리됩니다.

```env
OPENAI_API_KEY1=sk-...
OPENAI_API_KEY2=sk-...
# 필요한 만큼 추가 (OPENAI_API_KEY 단독도 가능)
```

## 사용

### 1. 설정 점검

```bash
python main.py check
```

API 키 개수, 임베딩 모델/차원, 데이터 경로 존재 여부를 확인합니다.

### 2. 인덱스 + 임베딩 빌드 (최초 1회)

```bash
python main.py build           # cache/에 산출물이 있으면 재사용
python main.py build --force   # 임베딩까지 재생성
```

산출물 (`patent_goods_similarity/cache/`):

- `patents.parquet` — 번호, 명칭, ipc_list, ksic_prefixes
- `goods_unique.parquet` — 고유 지정상품 텍스트
- `goods_index.pkl` — KSIC5 → 지정상품 id 매핑
- `patent_emb.npy`, `goods_emb.npy` — L2 정규화된 임베딩 행렬

### 3. 검색

`patent_goods_similarity/search.ipynb` 노트북에서 특정 특허번호를 입력하여 Top-K 지정상품을 조회합니다. 결과는 `output/top{K}_{특허번호}.csv`로 저장됩니다.

## 설정 (`config.yaml`)

```yaml
embedding:
  model: "text-embedding-3-small"
  dimensions: 1536          # 256/512/1024 등으로 축소 가능
  batch_size: 256
  max_retries: 6
  request_timeout: 60

search:
  top_k: 10
```
