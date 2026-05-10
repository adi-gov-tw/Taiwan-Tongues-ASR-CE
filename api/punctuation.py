"""ASR 後處理：以 google/gemma-3-1b-it 為逐字稿加入標點符號。

設計重點
--------
- **逐句處理**：每次只送一段 Whisper segment（必要時再以 CHUNK_SIZE 切段），
  避免長文本造成 LLM 幻覺、改字、或自行刪減內容。
- **字元保真**：模型輸出去掉標點後，必須與輸入一字不差；不一致即回退原文。
- **延遲載入**：第一次呼叫 punctuate 時才載入模型；載入失敗自動降級成 no-op。
- **失敗回退**：任何例外（OOM、斷網、HF 授權未通過）都不會讓上層任務失敗，
  最差情況等同於沒做標點。

對外介面
--------
``PunctuationProcessor.punctuate_segment(text)`` 回傳加上標點的字串；失敗則
回傳輸入原文。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, List, Optional

logger = logging.getLogger("asr_api")

DEFAULT_MODEL_ID = os.getenv("ASR_API_PUNCTUATION_MODEL", "google/gemma-3-1b-it")
DEFAULT_CHUNK_SIZE = int(os.getenv("ASR_API_PUNCTUATION_CHUNK_SIZE", "150"))
# 256 對 chunk_size=150 字（≈150 token）已綽綽有餘。greedy 會在 EOS 早停，
# 但批次推論時所有列要等批內最慢一列；上限太大會在生成壞例時拖很久。
MAX_NEW_TOKENS = int(os.getenv("ASR_API_PUNCTUATION_MAX_NEW_TOKENS", "256"))
# 批次大小：把多段（攤平後的 chunk）打包成一次 forward，攤平 prefill 成本。
# 1B bf16 + chunk_size=150：batch=8 約 1GB KV cache，8GB 卡很安全。
DEFAULT_BATCH_SIZE = int(os.getenv("ASR_API_PUNCTUATION_BATCH_SIZE", "8"))


def is_enabled() -> bool:
    """env: ASR_API_ENABLE_PUNCTUATION=0 可關閉，預設啟用。"""
    return os.getenv("ASR_API_ENABLE_PUNCTUATION", "1").strip() not in ("0", "false", "False", "")


SYSTEM_PROMPT = (
    "你是一個專業的中文標點符號標註助手。"
    "請僅為輸入的繁體中文逐字稿加入適當的標點符號。\n"
    "\n"
    "嚴格規則：\n"
    "1. 不得修改、增加或刪除任何原始文字。\n"
    "2. 不得加入任何說明、前後綴、解釋或標題。\n"
    "3. 直接輸出加上標點後的結果。\n"
    "\n"
    "標點選擇原則：\n"
    "- 「。」結束完整句子（每個獨立想法後都應該有一個）。\n"
    "- 「，」用於句中停頓或分隔子句。\n"
    "- 「、」僅用於並列名詞或短詞組之間（例：蘋果、香蕉、橘子）；\n"
    "  不要在子句之間或一般停頓位置使用頓號。\n"
    "- 「？」用於疑問句結束。\n"
    "  含「嗎、呢、吧」等疑問語氣助詞，或「為什麼、怎麼、是不是、有沒有、好不好」\n"
    "  等疑問詞時，該句應以「？」結尾，不要用「。」。\n"
    "- 「！」用於感嘆句、命令句或表達強烈情緒的句子結束。\n"
    "  含「太好了、好厲害、糟糕、加油、小心、注意、不可以」或「哇、啊、唉、哎呀」\n"
    "  等感嘆詞時，該句應以「！」結尾。\n"
)

FEW_SHOT_EXAMPLES = [
    (
        "今天天氣很好我想下午去公園走走順便買杯咖啡",
        "今天天氣很好，我想下午去公園走走，順便買杯咖啡。",
    ),
    (
        "我喜歡吃蘋果香蕉芒果這些水果尤其是芒果最甜",
        "我喜歡吃蘋果、香蕉、芒果這些水果，尤其是芒果最甜。",
    ),
    (
        "會議將於下週一上午十點舉行請各位準時出席如果有事無法參加請提前告知主管",
        "會議將於下週一上午十點舉行，請各位準時出席。如果有事無法參加，請提前告知主管。",
    ),
    (
        "你下週是不是要出差呢出發前要不要先一起吃個飯討論一下行程",
        "你下週是不是要出差呢？出發前要不要先一起吃個飯，討論一下行程？",
    ),
    (
        "哇這個產品真的太厲害了大家加油繼續努力小心不要出錯",
        "哇，這個產品真的太厲害了！大家加油，繼續努力！小心不要出錯！",
    ),
]

# 含中英常用標點與空白；用於字元保真比對
_PUNCT_CHARS = set(
    "，。！？、；：「」『』（）《》〈〉…—·"
    ",.!?;:\"'()<>[]{}…—-—"
    " \t\n\r"
)


def _strip_punct(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT_CHARS)


def _chunk_text(text: str, size: int) -> List[str]:
    """字元數均勻切段。沒有標點的逐字稿無法用語義切點，只能定長切。"""
    if len(text) <= size:
        return [text]
    n_chunks = (len(text) + size - 1) // size
    chunk_len = (len(text) + n_chunks - 1) // n_chunks
    return [text[i:i + chunk_len] for i in range(0, len(text), chunk_len)]


class PunctuationProcessor:
    """單例式 LLM 包裝；多執行緒共享一份模型，generate 加鎖串行。"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self._model = None
        self._tokenizer = None
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
                # 必須在 import transformers 前設定，避免探測到壞掉的 TF/JAX
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("USE_FLAX", "0")
                os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
                if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                cuda_ok = torch.cuda.is_available()
                if cuda_ok:
                    # 1B bf16 ~2GB，直接固定到 GPU 0，避免 accelerate 試圖分流到 CPU
                    device_map = {"": 0}
                    dtype = torch.bfloat16
                else:
                    device_map = {"": "cpu"}
                    dtype = torch.float32

                logger.info(
                    f"標點模型載入中：{self.model_id} (cuda={cuda_ok}, dtype={dtype}, batch={self.batch_size})"
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                # 批次推論需要 left padding（Causal LM 由右側續寫）
                self._tokenizer.padding_side = "left"
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    device_map=device_map,
                )
                self._model.eval()
                self._loaded = True
                logger.info("標點模型載入完成")
                return True
            except Exception as e:
                logger.error(f"標點模型載入失敗，後續將跳過標點：{e}")
                self._load_failed = True
                return False

    # -------------------------------------------------------------- inference
    def _build_input_ids(self, text: str) -> List[int]:
        """以 chat template 把 system / few-shot / 使用者輸入組成 token ids。"""
        messages: list[dict] = []
        first_in, first_out = FEW_SHOT_EXAMPLES[0]
        messages.append(
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n請為下列逐字稿加上標點：\n{first_in}",
            }
        )
        messages.append({"role": "assistant", "content": first_out})
        for inp, out in FEW_SHOT_EXAMPLES[1:]:
            messages.append({"role": "user", "content": f"請為下列逐字稿加上標點：\n{inp}"})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"請為下列逐字稿加上標點：\n{text}"})
        ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        return list(ids)

    def _generate_batch(self, texts: List[str]) -> List[str]:
        """同時處理多段；left-padding + 一次 forward，攤平 prompt prefill 成本。"""
        if not texts:
            return []
        import torch

        all_ids = [self._build_input_ids(t) for t in texts]
        max_len = max(len(ids) for ids in all_ids)
        pad_id = self._tokenizer.pad_token_id

        padded_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in all_ids:
            pad_count = max_len - len(ids)
            padded_ids.append([pad_id] * pad_count + ids)
            attention_mask.append([0] * pad_count + [1] * len(ids))

        device = self._model.device
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attn_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.0,
                use_cache=True,
                pad_token_id=pad_id,
            )

        # left-padding 下所有 row 的 prompt 結束點都在 max_len，response 從那裡切
        results: List[str] = []
        for row in outputs:
            response_ids = row[max_len:]
            decoded = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            results.append(decoded)
        return results

    # ----------------------------------------------------------------- public
    def punctuate_segments(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """為多段逐字稿加標點（**逐句送 LLM**，但多段批次同時推論加速）。

        - 仍維持單段獨立性（每段是 batch 中的一列，不互相影響），不會增加幻覺風險。
        - 任何單段字元保真不過 / 例外，僅該段回退原文，不影響其他段。
        - progress_callback(done_chunks, total_chunks) 在每批次完成後呼叫一次。
        """
        n = len(texts)
        results: List[str] = list(texts)
        if n == 0:
            return results
        if not self.load():
            return results

        # 攤平：每段 strip 後再以 CHUNK_SIZE 切。flat 元素 = (orig_idx, chunk_idx, chunk_text)
        per_segment_chunks: dict[int, List[Optional[str]]] = {}
        flat: List[tuple[int, int, str]] = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            cleaned = _strip_punct(text)
            if not cleaned:
                continue
            chunks = _chunk_text(cleaned, self.chunk_size)
            per_segment_chunks[i] = [None] * len(chunks)
            for ci, ch in enumerate(chunks):
                flat.append((i, ci, ch))

        if not flat:
            return results

        total = len(flat)
        done = 0

        with self._gen_lock:
            for batch_start in range(0, total, self.batch_size):
                batch = flat[batch_start:batch_start + self.batch_size]
                batch_inputs = [item[2] for item in batch]
                try:
                    outputs = self._generate_batch(batch_inputs)
                except Exception as e:
                    logger.warning(f"批次標點推論失敗，整批回退原文：{e}")
                    outputs = batch_inputs  # 等同沒加標點

                for (orig_idx, chunk_idx, chunk_text), out in zip(batch, outputs):
                    if _strip_punct(out) == chunk_text:
                        per_segment_chunks[orig_idx][chunk_idx] = out
                    else:
                        logger.warning(
                            "標點輸出與原文字元不一致，回退原文"
                            f"（原長={len(chunk_text)}, 輸出去標點長={len(_strip_punct(out))}）"
                        )
                        per_segment_chunks[orig_idx][chunk_idx] = chunk_text

                done += len(batch)
                if progress_callback:
                    try:
                        progress_callback(done, total)
                    except Exception:
                        pass

        # 把每段的 chunks 拼回去；任一 chunk 缺失則整段回退原文
        for orig_idx, chunk_list in per_segment_chunks.items():
            if all(c is not None for c in chunk_list):
                results[orig_idx] = "".join(chunk_list)
        return results

    def punctuate_segment(self, text: str) -> str:
        """單段便利介面（內部仍走批次邏輯，僅 batch 大小=1）。"""
        return self.punctuate_segments([text])[0]


# 全域單例（lazy load）
_processor: Optional[PunctuationProcessor] = None
_processor_lock = threading.Lock()


def get_processor() -> PunctuationProcessor:
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = PunctuationProcessor()
    return _processor
