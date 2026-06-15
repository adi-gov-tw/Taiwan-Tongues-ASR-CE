# Taiwan Tongues ASR CE專案

本專案提供一套自動語音辨識（ASR, Automatic Speech Recognition）模型訓練流程，並附有已訓練好的國語、台語、客語、英語、印尼語模型。你可以根據自己的語音資料進行微調（fine-tune），或直接使用現有模型進行語音辨識。

## 目錄結構

```
.
├── api/                              # API 服務（詳見 api/README.md）
│   ├── README.md                     # API 服務文件（安裝/規格/測試）
│   ├── app.py                        # 整合啟動入口（Port 5000）
│   ├── file_asr.py                   # 離線辨識（任務式 HTTP API）
│   ├── streaming_asr.py              # 即時辨識（WebSocket）
│   ├── auth_api.py                   # JWT 認證 API
│   ├── auth_shared.py                # 認證共用工具
│   ├── config.py                     # 模型裝置設定
│   ├── requirements.txt
│   ├── setup_api_env.sh / .bat       # API venv 一鍵建置（asr_api）
│   ├── start_app.sh / .bat           # 服務啟動腳本（兩平台）
│   ├── .env.example                  # 機敏設定範本（請複製為 .env）
│   ├── build.py
│   ├── static/                       # 測試頁（index/test_files/test_realtime.html）
│   ├── audio_files/                  # 任務暫存目錄（執行後產生）
│   └── stt_streaming/                # 即時串流模組（ASR / VAD / Buffering）
├── sample_corpus/                    # 含 zh / en / id 三語最小範例可直接驗證流程
│   ├── train_ds_01/                  # 中文（zh）
│   │   ├── train.tsv
│   │   ├── test.tsv
│   │   ├── validated.tsv
│   │   └── clips/
│   ├── train_ds_02/                  # 英文（en）
│   │   ├── train.tsv
│   │   ├── test.tsv
│   │   ├── validated.tsv
│   │   └── clips/
│   └── train_ds_id/                  # 印尼文（id）
│       ├── train.tsv
│       ├── test.tsv
│       ├── validated.tsv
│       └── clips/
├── models/                           # 推論用模型（需另行下載）
├── model_for_finetune/               # 微調基底模型（需另行下載）
├── train_asr.py                      # 訓練腳本（支援多語混訓 `ds:lang` 寫法）
├── evaluate_asr.py                   # 訓練後批次測試工具（轉錄+CER 評估）
├── cer.py                            # CER 計算工具
├── train.sh / .bat                   # 執行訓練（自動啟用 train_env）
├── setup_train_env.sh / .bat         # 訓練 venv 一鍵建置（train_env）
├── requirements.txt                  # 訓練/推論依賴
├── CITATION.cff
├── LICENSE
└── README.md
```


### 資料夾說明

- **sample_corpus/**  
  存放語音資料與標註檔案，每個子資料夾（如 `train_ds_01`、`train_ds_02`、`train_ds_id`）代表一個資料集。每個資料集包含：
  - `train.tsv`、`test.tsv`、`validated.tsv`：標註檔案，以Tab分隔，包含語音檔案路徑與對應轉寫文字。
  - `clips/`：存放實際語音檔案，支援多層子目錄。
  
  附帶的 `train_ds_01`（中文）、`train_ds_02`（英文）、`train_ds_id`（印尼文）為三語最小範例，可直接驗證訓練流程。

- **models/**  
  推論用模型（CTranslate2 格式，含國語、台語、客語、英語、印尼語）。請至 [adi-gov-tw on Hugging Face](https://huggingface.co/adi-gov-tw) 下載，解壓後放入專案根目錄的 `models/`。

- **model_for_finetune/**  
  HuggingFace 檢查點格式，供 `train_asr.py --model_name_or_path model_for_finetune` 載入微調。請至 [adi-gov-tw on Hugging Face](https://huggingface.co/adi-gov-tw) 下載，解壓後放入專案根目錄的 `model_for_finetune/`。

- **api/**  
  FastAPI 服務（File ASR + Auth + Streaming），單一埠 5000。安裝、啟動、API 規格詳見 [`api/README.md`](api/README.md)。

- **train_asr.py / train.sh / train.bat**  
  訓練腳本與啟動器；`train.{sh,bat}` 會自動啟用 `train_env`。

## 預訓練模型與開源語料下載

本專案使用的預訓練模型與開源語料皆釋出於 Hugging Face：

**[https://huggingface.co/adi-gov-tw](https://huggingface.co/adi-gov-tw)**

該組織內提供：

- **預訓練模型**：涵蓋 **國語、英語、台語、客語、印尼語** 五個語種
  - 推論用模型（CTranslate2 格式）→ 放入專案根目錄 `models/`
  - 微調基底模型（HuggingFace 檢查點格式）→ 放入專案根目錄 `model_for_finetune/`
- **開源語料**：涵蓋 **國語、英語、台語、客語、印尼語** 五個語種，可作為訓練、微調與評估之用。語料格式請對照下方「語料格式說明」放入 `sample_corpus/<dataset>/`。

下載方式擇一：

```bash
# 方法 1：使用 huggingface_hub（已隨 requirements.txt 安裝）
pip install -U huggingface_hub
huggingface-cli download adi-gov-tw/<repo-name> --local-dir ./models

# 方法 2：git clone（需安裝 git-lfs）
git lfs install
git clone https://huggingface.co/adi-gov-tw/<repo-name> models
```

> 實際的 `<repo-name>` 請至 [adi-gov-tw](https://huggingface.co/adi-gov-tw) 頁面查看（例如各語種模型、各語種語料分別為獨立 repo）。

## 建議硬體規格
| 任務 | 方案 | 建議硬體規格 (含廠牌與數量) | 預估時間 |
| :--- | :--- | :--- | :--- |
| **推論**<br>(Inference) | **CPU** | **CPU：** 高階多核 CPU、核心/執行緒 16+、基礎時脈 3.0 GHz+<br>*(建議廠牌：Intel / AMD / MAC M1 以上)*<br>**RAM：** 32 GB 或更高 *(廠牌皆可)* | 以 5 分鐘音檔估算，約需 **7 分鐘** |
| **推論**<br>(Inference) | **GPU** | **Host CPU：** 高階多核 CPU、核心/執行緒 8+、基礎時脈 2.0 GHz+<br>*(建議廠牌：Intel / AMD)*<br>**GPU：** VRAM 10 GB 以上<br>*(建議同 NVIDIA RTX 3080 / 4070 Ti 規格或更高)*<br>**RAM：** 16 GB - 32 GB *(廠牌皆可)* | 以 5 分鐘音檔估算，約需 **30 秒** |
| **訓練/微調**<br>(Training) | **CPU** | **CPU：** 伺服器級多核心 CPU、核心/執行緒 32+、基礎時脈 2.5 GHz+<br>*(建議廠牌：Intel / AMD / MAC M1 以上)*<br>**RAM：** 128 GB 或更高 *(廠牌皆可)* | 建議僅用於資料預處理<br>訓練微調需數月以上，效率極低 |
| **訓練/微調**<br>(Training) | **GPU** | **Host CPU：** 高階多核 CPU、核心/執行緒 8+、基礎時脈 2.0 GHz+<br>*(建議廠牌：Intel / AMD)*<br>**GPU：** VRAM 40 GB 以上<br>*(建議同 NVIDIA 6000PRO 規格或更高)*<br>**RAM：** 128 GB 或更高 *(廠牌皆可)* | 以 200 小時音訊評估，約需 **72 小時** |

- 以上建議硬體規格僅供參考，係基於市場主流品牌之架構提供其推論與訓練需求所整理之建議值；實際規格資訊將依市場可取得之硬體品牌、型號與規格，並以實測效能為準，進行相關調整。

## 語料格式說明

- 語音資料與標註檔案需放在 `sample_corpus` 目錄下，每個子資料夾（如 `train_ds_01`、`train_ds_02`）代表一個資料集。
- 每個資料集需包含：
  - `train.tsv`、`test.tsv`、`validated.tsv`：標註檔案，格式如下（以Tab分隔）：
    ```
    path    sentence
    audio_train_01_1.wav    這是一段語音
    audio_train_01_2.wav    另一段語音
    ```
    - `path` 欄位為語音檔案的相對路徑。
    - `sentence` 欄位為對應的語音轉寫文字。
  - `clips/`：實際語音檔案存放處，支援多層子目錄。

## 訓練方法

1. **安裝依賴套件**

   請先安裝 **Python 3.10 以上版本**（本專案最低要求 3.10）。提供一鍵安裝腳本，自動建立 `train_env/` 並安裝 `requirements.txt`。腳本會優先採用 `py -3.10`（Windows）或 `python3.10`（Linux/macOS）；找不到時則回退到任一 `>=3.10` 的直譯器。會以 `nvidia-smi` 偵測 NVIDIA GPU：偵測到時安裝 CUDA 12.4 版 PyTorch 與 cuDNN 9 / cuBLAS（faster-whisper 推論需要）；否則安裝 CPU 版。

   > Windows 取得 Python 3.10：執行 `py install 3.10` 或前往 [python.org 下載](https://www.python.org/downloads/release/python-31011/)。
   > Linux：透過 `apt` / `pyenv` 安裝 `python3.10` 或更新版。
   > macOS：`brew install python@3.10`（或更新版）。

   - Linux / macOS：
     ```bash
     bash setup_train_env.sh
     ```
   - Windows：
     ```cmd
     setup_train_env.bat
     ```

   或手動建立（請以 Python 3.10 以上版本建立 venv）：
   ```bash
   # Linux / macOS
   python -m venv train_env   # 或 python3.11 / python3.12 ...
   source train_env/bin/activate
   # Windows (cmd)
   # py -3.10 -m venv train_env   # 或 py -3.11 / py -3.12 ...
   # train_env\Scripts\activate.bat
   # Windows (PowerShell)
   # py -3.10 -m venv train_env
   # train_env\Scripts\Activate.ps1

   pip install --upgrade pip

   # 依硬體擇一：有 NVIDIA GPU
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"
   # 無 GPU
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   pip install -r requirements.txt
   ```
   > 事後加裝 GPU：在 venv 內 `pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`，再 `pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"`。

2. **準備語料**  
   依照上述格式放置語音資料與標註檔案；`sample_corpus/` 內已附最小範例可直接驗證流程。完整的中、英、台、客、印尼開源語料可至 [adi-gov-tw on Hugging Face](https://huggingface.co/adi-gov-tw) 下載。

3. **下載微調基底模型**  
   至 [adi-gov-tw on Hugging Face](https://huggingface.co/adi-gov-tw) 下載 HuggingFace 檢查點，並放置於專案根目錄的 `model_for_finetune/`。

4. **執行訓練腳本**

   訓練腳本會自動偵測並啟用 `train_env/`，無需手動 activate。

   - Linux / macOS / Windows（Git Bash / WSL）：
     ```bash
     bash train.sh
     ```
   - Windows（cmd / PowerShell）：
     ```cmd
     train.bat
     ```

   主要參數說明：
   - `--model_name_or_path`：本地檢查點路徑（預設為 `model_for_finetune`）或 HuggingFace 模型 ID（如 `openai/whisper-large-v3`）。
   - `--corpus_data_dir`：語料資料夾（如 `sample_corpus`）。
   - `--dataset_config_name`：資料集組合，以 `+` 串接。支援兩種寫法：
     - 單語：`train_ds_01+train_ds_02` — 所有資料集共用 `--language`（向後相容）。
     - **多語混訓**：`train_ds_01:zh+train_ds_02:en+train_ds_id:id` — 每份資料集帶自己的 Whisper 語系代碼，`prepare_dataset` 會逐筆切換 prefix token，可同時混訓多語（例如中/英/印尼）。
   - `--language`：預設語系代碼（如 `zh`、`en`、`id`、`nan`、`hak`）。當 `--dataset_config_name` 未帶 `:lang` 時作為 fallback。
   - 其他參數可參考 `train.sh` / `train.bat` 及 `train_asr.py`。

   **多語混訓範例**（透過環境變數覆寫 `train.sh` / `train.bat` 內建預設）：
   ```bash
   # Linux / macOS / Git Bash
   DATASET_CONFIG_NAME="train_ds_01:zh+train_ds_02:en+train_ds_id:id" bash train.sh
   ```
   ```cmd
   :: Windows (cmd)
   set DATASET_CONFIG_NAME=train_ds_01:zh+train_ds_02:en+train_ds_id:id
   train.bat
   ```

5. **訓練結果**  
   訓練完成後，模型與相關設定會儲存在 `output/` 目錄。

## 推論/辨識語音

### 使用 evaluate_asr.py 進行批次轉錄與 CER 評估

訓練完成後可用 `evaluate_asr.py` 對指定資料夾做批次轉錄與品質評估：

- **批次轉錄**：處理 `.wav` / `.mp3` / `.flac` / `.m4a` / `.aac`，每檔輸出 `{原檔名}_asr.txt`
- **後處理**（依序執行）：
  - phrase 替換（如「百分之十五」→「15%」、「零八零零零九五九八」→「080009598」），可在 `replace_words()` 內擴充
  - 簡 → 繁（OpenCC `s2tw`，台灣用字）
  - 中文數字 → 阿拉伯數字（`cn2an.transform`，例：「九百三十一」→「931」、「九十九元」→「99元」；解析失敗時保留原文）
  - 全形 → 半形（NFKC 正規化；限 Latin / 數字 / 標點，中文字不會被動到）
  - 移除中英常見標點（`，"'。：；「」（）⋯` 等）
  - 全文小寫
- **CER 自動評估**（選填）：若同目錄有同名 `.txt` / `_transcript.txt` / `_original.txt` / `_reference.txt` / `_ground_truth.txt` 作為 ground truth，會自動跟 ASR 結果比對並輸出 JSON 報告

需要 `models/` 內的推論模型檔。

```bash
python evaluate_asr.py <音檔資料夾路徑>
# 例：python evaluate_asr.py test_dataset
```

## 離線辨識與即時辨識 API

詳見 [`api/README.md`](api/README.md)。

## 引用

(*此處需列出專案主要貢獻者、發起人*)

If you use this project, please cite it as follows:

```yaml
cff-version: 1.2.0
title: "Automatic Speech Recognition (ASR) Project"
authors:
  - family-names: "Hsieh"
    given-names: "Archer"
    affiliation: "Taiwan Mobile Co., Ltd"
date-released: "2025-07-14"
version: "1.0.0"
abstract: |
  This project provides a comprehensive framework for Automatic Speech Recognition (ASR), supporting multilingual speech processing and fine-tuning capabilities. It includes pre-trained models for Mandarin, Taiwanese, Hakka, English, and Indonesian, and tools for speech-to-text conversion and spoken language identification.

keywords:
  - ASR
  - Automatic Speech Recognition
  - Multilingual Speech Processing
  - Speech-to-Text
  - Open Source

repository-code: "https://github.com/your-repo/asr-project"
license: "MIT"
```

---

如需更多協助，請於 Issues 留言或聯絡專案維護者。 