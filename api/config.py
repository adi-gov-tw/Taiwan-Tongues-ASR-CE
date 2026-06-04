"""
ASR 系統配置檔案（精簡版）

目前專案僅使用以下兩個參數由即時串流 ASR 載入：
- MODEL_DEVICE：'cpu'、'cuda' 或 'auto'（依硬體自動選擇）
- MODEL_COMPUTE_TYPE：'float16'、'int8' 或 'auto'（cuda → float16、cpu → int8）

若未來需要將 VAD、音訊或日誌參數外部化，請在實際用到的程式檔讀取本檔配置後再行新增。
"""

# 模型設置（供 streaming ASR 讀取）
# 'auto' 會在 faster_whisper_asr.py 內依 torch.cuda.is_available() 自動選擇
MODEL_DEVICE = "auto"
MODEL_COMPUTE_TYPE = "auto"
