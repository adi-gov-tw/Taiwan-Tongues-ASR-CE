"""ASR 後處理：使用 zhpr（https://pypi.org/project/zhpr/）為逐字稿加入中文標點。

模型：``p208p2002/zh-wiki-punctuation-restore`` —— 以 bert-base-chinese 為基底的
token-classification 模型，可預測 6 種標點：``，、。？！；``，模型約 100MB，
CPU/GPU 皆能順跑。

設計重點
--------
- **逐句處理**：每次只送一段 Whisper segment，避免長文本造成跨句語意污染。
- **延遲載入**：第一次呼叫 punctuate 時才載入模型；載入失敗自動降級為 no-op。
- **失敗回退**：任何例外都以原文回傳，不會讓上層任務 fail。
- **裝置自動**：偵測到 CUDA 就用 GPU；否則 CPU 推論（模型小、CPU 也很快）。

對外介面
--------
``PunctuationProcessor.punctuate_segments(texts)`` 取一組 segment 字串，回傳
等長的、加上標點的字串列表；任一段失敗則該段以原文回傳。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, List, Optional

logger = logging.getLogger("asr_api")

DEFAULT_MODEL_ID = os.getenv(
    "ASR_API_PUNCTUATION_MODEL", "p208p2002/zh-wiki-punctuation-restore"
)
# zhpr 預設 window=384/step=307；縮小 window 對短逐字稿更省記憶體，且 200 step
# 仍有 56 token overlap 給 merge_stride 平滑邊界。
DEFAULT_WINDOW_SIZE = int(os.getenv("ASR_API_PUNCTUATION_WINDOW_SIZE", "256"))
DEFAULT_STRIDE_STEP = int(os.getenv("ASR_API_PUNCTUATION_STRIDE_STEP", "200"))
DEFAULT_BATCH_SIZE = int(os.getenv("ASR_API_PUNCTUATION_BATCH_SIZE", "8"))


def is_enabled() -> bool:
    """env: ASR_API_ENABLE_PUNCTUATION=0 可關閉，預設啟用。"""
    return os.getenv("ASR_API_ENABLE_PUNCTUATION", "1").strip() not in (
        "0",
        "false",
        "False",
        "",
    )


class PunctuationProcessor:
    """單例式 zhpr 包裝；多執行緒共享一份模型，generate 加鎖串行。"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride_step: int = DEFAULT_STRIDE_STEP,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.window_size = window_size
        self.stride_step = stride_step
        self.batch_size = max(1, batch_size)

        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False
        self._load_failed = False
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                import torch
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                )

                cuda_ok = torch.cuda.is_available()
                self._device = torch.device("cuda" if cuda_ok else "cpu")

                logger.info(
                    f"標點模型載入中：{self.model_id} "
                    f"(device={self._device}, batch={self.batch_size})"
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForTokenClassification.from_pretrained(
                    self.model_id
                )
                self._model.to(self._device)
                self._model.eval()
                self._loaded = True
                logger.info("標點模型載入完成")
                return True
            except Exception as e:
                logger.error(f"標點模型載入失敗，後續將跳過標點：{e}")
                self._load_failed = True
                return False

    # -------------------------------------------------------------- inference
    def _predict_one(self, text: str) -> str:
        """跑單一段文字，回傳加上標點後的字串；失敗即原文回傳。"""
        if not text:
            return text

        try:
            import torch
            from torch.utils.data import DataLoader
            from zhpr.predict import DocumentDataset, merge_stride, decode_pred

            dataset = DocumentDataset(
                text, window_size=self.window_size, step=self.stride_step
            )
            if len(dataset) == 0:
                return text
            loader = DataLoader(
                dataset, shuffle=False, batch_size=self.batch_size
            )

            model_pred_out: list = []
            with torch.inference_mode():
                for batch in loader:
                    batch = batch.to(self._device)
                    output = self._model(input_ids=batch)
                    pred_ids = output["logits"].argmax(-1)
                    for predicted_token_class_ids, input_ids in zip(pred_ids, batch):
                        ids_list = input_ids.tolist()
                        try:
                            pad_start = ids_list.index(self._tokenizer.pad_token_id)
                        except ValueError:
                            pad_start = len(ids_list)
                        tokens = self._tokenizer.convert_ids_to_tokens(ids_list)[
                            :pad_start
                        ]
                        classes = [
                            self._model.config.id2label[t.item()]
                            for t in predicted_token_class_ids
                        ][:pad_start]
                        model_pred_out.append(list(zip(tokens, classes)))

            merged = merge_stride(model_pred_out, step=self.stride_step)
            decoded = decode_pred(merged)
            result = "".join(decoded)

            # zhpr 會把 `[UNK]` 等 BERT special token 直接吐出；若出現代表原句有
            # 模型不認的字元，保留原文比較不會錯改。
            if "[UNK]" in result or "[CLS]" in result or "[SEP]" in result:
                return text
            return result
        except Exception as e:
            logger.warning(f"單段標點推論失敗，回退原文：{e}")
            return text

    def punctuate_segments(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """逐段加上標點；第一次呼叫時懶載入模型。"""
        if not texts:
            return []
        if not self.load():
            return list(texts)
        out: List[str] = []
        total = len(texts)
        with self._gen_lock:
            for idx, t in enumerate(texts, start=1):
                out.append(self._predict_one(t))
                if progress_callback is not None:
                    try:
                        progress_callback(idx, total)
                    except Exception:
                        pass
        return out


_processor_lock = threading.Lock()
_processor: Optional[PunctuationProcessor] = None


def get_processor() -> PunctuationProcessor:
    """取得（或初次建立）行程內共享的單例 processor。"""
    global _processor
    if _processor is not None:
        return _processor
    with _processor_lock:
        if _processor is None:
            _processor = PunctuationProcessor()
    return _processor
