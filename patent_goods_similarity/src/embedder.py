"""OpenAI text-embedding-3-small 임베딩 py 입니다.

처리 속도를 위해, 5개의 API 키를 사용했기 때문에.
키마다 별도 클라이언트를 만들고
ThreadPoolExecutor 로 배치를 동시 처리하도록 했습니다.
중간 중단 시 .partial.npy 에서 이어서 재개.

이 코드는 임베딩 모델이 달라질 경우 새로 작성해야 합니다~!
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from .config import Config


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


class OpenAIEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.clients = [
            OpenAI(api_key=k, timeout=cfg.embedding.request_timeout)
            for k in cfg.openai_api_keys
        ]
        self.n_workers = len(self.clients)
        self._print_lock = threading.Lock()

    def _embed_batch(self, batch: list[str], client_idx: int) -> np.ndarray:
        client = self.clients[client_idx]
        last_err: Exception | None = None
        for attempt in range(self.cfg.embedding.max_retries):
            try:
                resp = client.embeddings.create(
                    model=self.cfg.embedding.model,
                    input=batch,
                    dimensions=self.cfg.embedding.dimensions,
                )
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(60.0, (2 ** attempt) + 0.5)
                with self._print_lock:
                    tqdm.write(
                        f"[embed][key#{client_idx}] 재시도 {attempt + 1}/"
                        f"{self.cfg.embedding.max_retries} in {wait:.1f}s: "
                        f"{type(e).__name__}: {e}"
                    )
                time.sleep(wait)
        raise RuntimeError(f"임베딩 실패 (key#{client_idx}): {last_err}")

    def embed_texts(
        self,
        texts: list[str],
        save_path: Path,
        desc: str = "embed",
    ) -> np.ndarray:
        save_path = Path(save_path)
        if save_path.exists():
            existing = np.load(save_path)
            if existing.shape[0] == len(texts):
                return existing
            tqdm.write(
                f"[embed] {save_path.name} 행 수 불일치"
                f"({existing.shape[0]} vs {len(texts)}) → 재생성"
            )

        partial_path = save_path.with_suffix(".partial.npy")
        meta_path = save_path.with_suffix(".partial.meta")
        bs = self.cfg.embedding.batch_size
        total = len(texts)

        # 부분 결과 재개
        chunks: list[np.ndarray] = []
        start_idx = 0
        if partial_path.exists() and meta_path.exists():
            try:
                saved = np.load(partial_path)
                saved_n = int(meta_path.read_text().strip())
                if saved.shape[0] == saved_n and saved_n <= total:
                    chunks.append(saved)
                    start_idx = saved_n
                    tqdm.write(f"[embed] {desc}: 부분 결과 발견 → {start_idx}건부터 재개")
            except Exception:  # noqa: BLE001
                tqdm.write("[embed] 부분 결과 손상 → 처음부터 재시작")
                start_idx = 0
                chunks.clear()

        # 한 사이클당 동시에 보낼 배치 수 = 키 수
        chunk_texts = bs * self.n_workers

        pbar = tqdm(total=total, initial=start_idx, desc=desc, unit="text")
        try:
            i = start_idx
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                while i < total:
                    cycle_end = min(i + chunk_texts, total)
                    batches: list[tuple[int, list[str]]] = []
                    for j in range(i, cycle_end, bs):
                        batches.append((j, texts[j : j + bs]))

                    futures = {}
                    for k, (start, batch_texts) in enumerate(batches):
                        client_idx = k % self.n_workers
                        fut = ex.submit(self._embed_batch, batch_texts, client_idx)
                        futures[fut] = start

                    # 결과를 시작 인덱스 기준으로 모아서 순서대로 합치기
                    results: dict[int, np.ndarray] = {}
                    for fut in as_completed(futures):
                        start = futures[fut]
                        results[start] = fut.result()
                        pbar.update(len(results[start]))

                    for start, _ in batches:
                        chunks.append(results[start])

                    i = cycle_end
                    # 부분 저장 (사이클 종료마다)
                    merged = np.concatenate(chunks, axis=0)
                    np.save(partial_path, merged)
                    meta_path.write_text(str(merged.shape[0]))
        finally:
            pbar.close()

        embeddings = np.concatenate(chunks, axis=0)
        embeddings = _l2_normalize(embeddings)
        np.save(save_path, embeddings)
        partial_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return embeddings
