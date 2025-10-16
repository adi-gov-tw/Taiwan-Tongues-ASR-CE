"""
ASR 系統配置檔案（精簡版）

目前專案僅使用以下兩個參數由即時串流 ASR 載入：
- MODEL_DEVICE：'cpu' 或 'cuda'
- MODEL_COMPUTE_TYPE：如 'float16'、'int8' 等

若未來需要將 VAD、音訊或日誌參數外部化，請在實際用到的程式檔讀取本檔配置後再行新增。
"""

# 模型設置（供 streaming ASR 讀取）
MODEL_DEVICE = "cuda"  # 'cpu' 或 'cuda'
MODEL_COMPUTE_TYPE = "float16"  # 如 'float16'、'int8' 等
