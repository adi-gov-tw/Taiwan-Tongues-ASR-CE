# Taiwan Tongues ASR CE — API 服務

本文件說明 `api/` 目錄下提供的 **離線檔案辨識（File ASR, HTTP）** 與 **即時串流辨識（Streaming ASR, WebSocket）** 兩項服務的安裝、啟動、API 規格與測試方式。

關於模型訓練 / 微調流程，請回到專案根目錄的 [README.md](../README.md)。

---

## 目錄

- [服務內容](#服務內容)
- [安裝與啟動](#安裝與啟動)
- [File ASR（HTTP, 5000）API 規格（任務式）](#file-asrhttp-5000api-規格任務式)
- [認證 API（HTTP, 5000）規格](#認證-apihttp-5000規格)
- [Streaming ASR（WebSocket）API 規格](#streaming-asrwebsocketapi-規格)
- [音訊規格建議](#音訊規格建議)
- [手動驗證](#手動驗證)
- [引用](#引用)

---

## 服務內容

本專案在 `api/` 目錄提供兩種服務：檔案轉錄（HTTP）與即時串流（WebSocket）。模型請放在專案根目錄 `models/`。

- `app.py`：整合啟動入口（單一埠 5000），同時掛載 File ASR 與 Streaming ASR。
- `file_asr.py`：任務式檔案轉錄 API（HTTP, 5000）。端點：
  - `GET /api/health`
  - `POST /api/v1/subtitle/tasks`（建立任務，立即回傳任務 id；背景處理）
  - `POST /api/v1/subtitle/tasks/{id}`（查任務狀態/進度）
  - `GET /api/v1/subtitle/tasks/{id}/subtitle-link?type=TXT|SRT|DIA`（取得下載連結）
  - `GET /api/v1/subtitle/tasks/{id}/subtitle?type=TXT|SRT|DIA`（直接下載字幕檔）
  - 測試頁：`/test_files.html`（顯示任務進度與 TXT/SRT 下載）
- `streaming_asr.py`：即時串流 ASR（WebSocket）。端點包含：
  - `GET /stream/health`、`GET /stream/test`、測試頁（建議透過主服務 `/test_realtime.html` 載入）
  - WS `/ws/v1/transcript`（Query: `token`；合併模式直掛於 5000 埠）
- `auth_api.py` / `auth_shared.py`：JWT 認證模組（admin/user 角色管理、SQLite 儲存）。
- `stt_streaming/`：即時串流的模組與策略實作（ASR、VAD、Buffering 等）。
- `punctuation.py`：File ASR 完成後對逐字稿加標點符號的後處理模組
  （以 [zhpr](https://pypi.org/project/zhpr/) + `p208p2002/zh-wiki-punctuation-restore`
  約 100MB，CPU/GPU 皆能跑；任何例外回退原文）。

---

## 安裝與啟動

### 一鍵安裝（建議）

> **環境需求**：本專案最低要求 **Python 3.10**（3.10 以上皆可）。Windows 可透過 `py install 3.10` 安裝；Linux/macOS 請使用 `python3.10` 或更新版（`apt` / `pyenv` / `brew install python@3.10`）。

於專案根目錄執行對應腳本，會自動以符合需求的 Python 建立 `asr_api/` 虛擬環境並完成 `api/requirements.txt` 安裝。腳本會優先選用 `py -3.10`（Windows）或 `python3.10`（Linux/macOS），找不到時回退到任一 `>=3.10` 的直譯器；接著以 `nvidia-smi` 偵測 NVIDIA GPU：偵測到時安裝 CUDA 12.4 版 PyTorch 與 cuBLAS / cuDNN 9，否則安裝 CPU 版。

> 主要套件版本：Python 3.10+、PyTorch 2.x、transformers 4.x（受 zhpr 限制 `<5`，僅 API 端）、faster-whisper、zhpr、pytorch-lightning。詳見 `api/requirements.txt`。

- Linux / macOS：
  ```bash
  bash api/setup_api_env.sh
  ```
- Windows：
  ```cmd
  api\setup_api_env.bat
  ```

### 手動安裝（跨平台）

```bash
# 在專案根目錄執行（請以 Python 3.10 以上版本建立 venv）

# Linux / macOS
python -m venv asr_api          # 或 python3.11 / python3.12 ...
source asr_api/bin/activate

# Windows (cmd)
# py -3.10 -m venv asr_api          # 或 py -3.11 / py -3.12 ...
# asr_api\Scripts\activate.bat

# Windows (PowerShell)
# py -3.10 -m venv asr_api
# asr_api\Scripts\Activate.ps1

pip install --upgrade pip

# 依硬體擇一：有 NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 無 GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r api/requirements.txt
```

> `api/config.py` 預設 `MODEL_DEVICE=auto`，依 `torch.cuda.is_available()` 自動選 GPU/CPU。事後加裝 GPU：`pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`，再 `pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"`。

### 設定機敏資訊（.env）

API 服務的所有機敏設定（JWT 密鑰、預設管理員密碼等）皆透過 `api/.env` 載入。請務必在啟動前完成設定：

```bash
# 在專案根目錄執行
cp api/.env.example api/.env
# Windows (cmd):  copy api\.env.example api\.env
```

接著編輯 `api/.env`，將下列欄位改為自己的安全值：

| 欄位 | 說明 |
| :--- | :--- |
| `ASR_API_JWT_SECRET` | JWT 簽章密鑰；建議 32+ bytes 亂數字串（`python -c "import secrets; print(secrets.token_urlsafe(48))"`） |
| `ASR_API_BOOTSTRAP_ADMIN_PASSWORD` | 預設管理員密碼 |
| `ASR_API_BOOTSTRAP_ADMIN_USERNAME` | 預設管理員帳號（預設 `admin`） |
| `ASR_API_RESET_ADMIN_ON_STARTUP` | `1`=每次啟動重設 admin 密碼；`0`=保留手動修改值 |

> - `api/.env` 已加入 `.gitignore`。
> - 未建立 `.env` 時 fallback 為不安全預設值（`JWT_SECRET=CHANGE_ME_SECRET`），僅本機 demo 用。
> - 系統環境變數優先序高於 `.env`。

### 啟動服務（單一埠 5000）

兩個啟動腳本都會自動偵測 `asr_api/`、`train_env/`、`.venv/`、`venv/` 中可用的虛擬環境；若無 `api/.env` 會提示。

- Linux / macOS：
  ```bash
  bash api/start_app.sh
  ```
- Windows：
  ```cmd
  api\start_app.bat
  ```

---

## File ASR（HTTP, 5000）API 規格（任務式）

基底 URL：`http://127.0.0.1:5000`

- GET `/api/health`：健康檢查
  ```json
  {"status":"healthy","model_loaded":true,"timestamp":"2025-01-01T12:00:00"}
  ```

- 建立任務：`POST /api/v1/subtitle/tasks`
  - multipart/form-data：
    - `audio`（必填）：.wav/.mp3/.flac/.m4a/.aac
    - `reference_text`（選填）
  - 立即回傳（背景處理）：`{"code":200,"message":"created","id":<task_id>}`
  - 範例（使用內附的 `stt_streaming/warm_up.wav` 測試，或替換成自己的音檔）：
    ```bash
    curl -H "Authorization: Bearer <TOKEN>" \
      -F "audio=@api/stt_streaming/warm_up.wav" \
      http://127.0.0.1:5000/api/v1/subtitle/tasks
    ```

### 建立任務：擴充 / 進階 Body 欄位

以下欄使用前請先與作者聯繫以啟用、確認服務版本與有效值；未啟用時服務可能忽略或拒絕這些欄位。

- Purpose：建立離線辨識任務
- HTTP Method：POST
- API End-point：`/api/v1/subtitle/tasks`
- Parameters：N/A（Query 無附加參數）
- Body（`multipart/form-data`）攜帶：
  - `sourceType`：音檔來源（int；1：透過 YouTube 連結下載，2：直接上傳音檔）。
  - `sourceWebLink`：指定之 YouTube 連結（string；當 `sourceType=1` 時需要）。
  - `title`：任務標題（string）。
  - `description`：任務描述（string）。
  - `audioChannel`：音檔音軌設定（int；0：不指定，1：只使用左聲道，2：只使用右聲道）（optional）。
  - `modelName`：指定之模型代號（string；需為模型代號，非模型顯示名稱）（optional）。
  - `modelVersion`：指定之模型版本號（string）（optional）。
  - `taskPriority`：優先權（int；預設值為 1，值越大表示優先權越高）（optional）。
  - `speakerNum`：音檔之語者人數（int；系統有支援語者標記功能時使用；預設值為 0，表示自動偵測）（optional）。
  - `dspMode`：音檔優化模式（int；預設值為 1，表示開啟；0 表示關閉優化功能）（optional）。
  - `promptWords`：音檔內容之關鍵字（string；設定音檔內出現之人名或專有名詞等；詞與詞之間使用半形逗號隔開，如 `keyword-1,keyword-2,keyword-3`）（optional）。
  - `textTrim`：文字潤飾模式（int；系統有支援文字潤飾功能時使用；有效值為 0=disable，1=enable；預設值為 0）（optional）。

- 查任務狀態：`POST /api/v1/subtitle/tasks/{id}`
  - 回傳：`{"code":200,"data":[{"status":<狀態碼>,"progress":<0-100>}]}`
  - 狀態碼：
    - 0 等待確認檔案；3 成功；4 失敗；5 已取消
    - 10 上傳中；11 等待處理逐字稿；12 檔案下載中；13 逐字稿處理中
    - 20 音檔等待處理；21 音檔處理中；22 音檔處理完成
    - 30 串流進行中；31 串流成功；32 串流失敗；33 串流無內容

- 取得下載連結：`GET /api/v1/subtitle/tasks/{id}/subtitle-link?type=TXT|SRT|DIA`
  - 回傳：`{"code":200,"data":[{"id":<id>,"type":"TXT|SRT|DIA","url":"/api/v1/subtitle/tasks/<id>/subtitle?type=..."}]}`

- 直接下載字幕：`GET /api/v1/subtitle/tasks/{id}/subtitle?type=TXT|SRT|DIA`
  - 下載 TXT（text/plain）或 SRT（application/x-subrip）。

- 取得可用字幕格式：`GET /api/v1/subtitle/tasks/{id}/subtitle-types`
  - 回傳（`id` 為任務 id；`types` 依產出順序提供 TXT、SRT、DIA）：
    ```json
    {
      "code": 200,
      "data": [
        { "id": <task_id>, "types": ["TXT", "SRT", "DIA"] }
      ]
    }
    ```
  - 任務不存在或請求有誤時回傳 404/400。

- GET `/test_files.html`：測試頁（健康檢查 / 單一音檔轉錄）

### 標點符號後處理

當任務轉錄完成後（status 進入 22 → 3），服務會將每個 Whisper segment 逐句送入
[zhpr](https://pypi.org/project/zhpr/)（以 `p208p2002/zh-wiki-punctuation-restore`
為底）加上 6 種中文標點：`，、。？！；`，再寫入 TXT / SRT 檔。
- 模型於首個任務觸發時延遲載入；本體約 100MB，CPU 也能順跑（GPU 更快）。
- 載入或推論失敗會自動回退原文，不會讓任務 fail。
- 相關環境變數：`ASR_API_ENABLE_PUNCTUATION`（0=停用，預設 1）、
  `ASR_API_PUNCTUATION_MODEL`、`ASR_API_PUNCTUATION_WINDOW_SIZE`、
  `ASR_API_PUNCTUATION_STRIDE_STEP`、`ASR_API_PUNCTUATION_BATCH_SIZE`。
  詳見 `api/.env.example`。

> 注意：zhpr 0.1.3 在套件 metadata 把 `transformers` 釘在 `<5`，故
> `api/requirements.txt` 把 transformers 跟著降到 `>=4.24,<5`；訓練端
> （root `requirements.txt`）獨立 venv 仍保持 `transformers>=5,<6`。

---

## 認證 API（HTTP, 5000）規格

基底 URL：`http://127.0.0.1:5000`

- GET `/api/v1/health`：認證服務健康檢查
  ```json
  {"status":"ok"}
  ```

- POST `/api/v1/login`：使用者登入
  - 請求體：
    ```json
    {
      "username": "<your-admin-username>",
      "password": "<your-admin-password>",
      "rememberMe": 0
    }
    ```
  - 回應：
    ```json
    {
      "code": 200,
      "token": "...",
      "expiration": 86400,
      "pwdExpired": 0
    }
    ```

- POST `/api/v1/logout`：使用者登出（需要 Bearer Token）
  - 標頭：`Authorization: Bearer <token>`
  - 回應：
    ```json
    {
      "code": 200,
      "username": "admin",
      "message": "logged out"
    }
    ```

- POST `/api/v1/user`：建立新使用者（僅管理員，需要 Bearer Token）
  - 標頭：`Authorization: Bearer <token>`
  - 請求體：
    ```json
    {
      "username": "newuser",
      "nickname": "新使用者",
      "role": "user",
      "comment": "測試帳號",
      "password": "password123",
      "expiredTime": "2025-12-31T23:59:59Z",
      "status": 1
    }
    ```

- PUT `/api/v1/user/password`：更新使用者密碼（需要 Bearer Token）
  - 標頭：`Authorization: Bearer <token>`
  - 查詢參數：`username`、`newPassword`

### 環境變數

完整清單與說明請見 `api/.env.example`。常用：

**認證 / 安全**
- `ASR_API_AUTH_DB`：認證資料庫路徑（預設 `api/auth.db`）
- `ASR_API_JWT_SECRET`：JWT 密鑰
- `ASR_API_JWT_ALGORITHM`：JWT 演算法（預設：HS256）
- `ASR_API_BOOTSTRAP_ADMIN_USERNAME` / `ASR_API_BOOTSTRAP_ADMIN_PASSWORD` / `ASR_API_BOOTSTRAP_ADMIN_NICKNAME`：預設管理員帳號
- `ASR_API_RESET_ADMIN_ON_STARTUP`：啟動時是否重設管理員密碼

**FastAPI / Streaming 執行設定**
- `FASTAPI_HOST` / `FASTAPI_PORT`：監聽位址 / 埠（預設 `0.0.0.0` / `5000`）
- `FASTAPI_SKIP_INIT`：1 = 略過 VAD/ASR 初始化（測試 WS 連線用）
- `FASTAPI_WARMUP`：1 = 啟用模型預熱（降低首次推論延遲）
- `FASTAPI_ASR_MODEL_SIZE`：faster-whisper 模型大小或本地資料夾（預設 `models`）
- `BUFFERING_CHUNK_LENGTH_SECONDS` / `BUFFERING_CHUNK_OFFSET_SECONDS`：串流緩衝參數

**標點符號後處理**
- `ASR_API_ENABLE_PUNCTUATION`：1 = 啟用（預設）；0 = 停用
- `ASR_API_PUNCTUATION_MODEL`：模型 ID（預設 `p208p2002/zh-wiki-punctuation-restore`）
- `ASR_API_PUNCTUATION_WINDOW_SIZE` / `ASR_API_PUNCTUATION_STRIDE_STEP`：zhpr 滑窗推論的視窗大小與步長
- `ASR_API_PUNCTUATION_BATCH_SIZE`：一次 forward 處理多少視窗（預設 8）

---

## Streaming ASR（WebSocket）API 規格

- Path：
  - 基底 URL：`http://127.0.0.1:5000/stream`
  - 測試頁：`http://127.0.0.1:5000/test_realtime.html`

- GET `/stream/health`：健康檢查（`asr_device` 與 `asr_compute_type` 依執行環境自動偵測，GPU 環境為 `cuda`/`float16`，純 CPU 環境為 `cpu`/`int8`）
  ```json
  {"status":"healthy","connected_clients":0,"vad_pipeline":"ready","asr_pipeline":"ready","asr_device":"cuda","asr_compute_type":"float16","asr_model_size":"models"}
  ```

- GET `/test_realtime.html`：即時辨識測試頁（可直接於瀏覽器使用麥克風測試）

- GET `/stream/test`：服務端內建的最簡 HTML 測試頁（streaming app 子掛載於 `/stream`）

- WebSocket `/ws/v1/transcript`：即時串流端點
  - Query 參數：`token`（必填，簡單驗證用）
  - 上行：
    - 二進位音訊：Int16 PCM, mono, 16kHz，分片送入
  - 下行（JSON 範例）：
    ```json
    {
      "id": "7191c96a-b3db-4bda-a614-434c300d6f4f",
      "code": 200,
      "message": "轉譯成功",
      "result": [
        {
          "segment": 0,
          "transcript": "測試123123",
          "final": 1,
          "startTime": 2.976,
          "endTime": 5.356
        }
      ]
    }
    ```

### 擴充 / 進階 Query 參數

以下欄使用前請先與作者聯繫以啟用、確認服務版本與有效值；未啟用時服務可能忽略或拒絕這些欄位。

- Purpose：建立即時辨識

- Parameters：
  - `ticket`：透過 `/api/v1/streaming/transcript/access-info` 取得之認證資訊（必要）；送出前需先做 URL encoding。
  - `type`：語音資料型態（必要）。有效型態請參考內部規格表。
  - `rate`：語音取樣頻率（必要）。有效值為 8000 或 16000；若 `type=file`，此設定將被忽略。
  - `channel`：語音通道數量（選填）。有效值為 1（mono）或 2（stereo）；未指定預設為 1；若 `type=file`，此設定無效。
  - `modelName`：指定模型名稱（選填）；值對應 API `/api/v1/models` 回應中模型之 `name` 欄位。
  - `title`：本次連線標題（選填），上限 128 字；內容可用於日後搜尋。
  - `saveResult`：是否儲存本次連線之內容（選填）。有效值：1（儲存）、0（不儲存）；預設 0。
  - `audioFilename`：指定儲存之檔名（含副檔名）（選填），僅允許英數與 `-`、`_`、`.`，上限 64 字；若 `type=file`，必填。
  - `enableTransient`：是否要收到暫時性（final=0）轉譯結果（選填）。有效值：1：是，0：否（預設）。
  - `charactersToNumbers`：是否開啟國字轉阿拉伯數字（選填）。有效值：1/0（on/off），預設 1。
  - `minSilenceDurMs`：完成單句辨識之停頓毫秒數（選填）。例如 1000 代表連續 1000ms 無人聲即輸出單句結果。
  - `maxPacketLossDurSec`：封包遺失判斷秒數（選填）。例如 2 代表超過 2 秒未收到聲音封包即判定遺失並中斷連線。
  - `noSpeechTimeout`：無辨識內容之逾時秒數（選填）。例如 5 代表超過 5 秒仍未偵測到人聲即逾時並中斷連線。

- Partial result：設定參數 `enableTransient=1`，回應內容的 `final`: True

---

## 音訊規格建議

- 取樣率：16 kHz
- 聲道：mono（單聲道）
- 位寬：16-bit（二進位請送 Int16 PCM）
- 建議開始錄音前約 1 秒保持安靜，利於噪音底噪校準與說話偵測。

---

## 手動驗證

服務啟動後，可用以下流程逐項驗證主要功能（皆可用 `curl`、Postman 或測試頁完成）：

```bash
# 1) 健康檢查
curl http://127.0.0.1:5000/api/health
curl http://127.0.0.1:5000/api/v1/health
curl http://127.0.0.1:5000/stream/health

# 2) 登入取得 JWT
curl -X POST http://127.0.0.1:5000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username":"<your-admin-username>","password":"<your-admin-password>","rememberMe":0}'

# 3) 建立檔案辨識任務（將 <TOKEN> 換成上一步取得之 token）
curl -H "Authorization: Bearer <TOKEN>" \
  -F "audio=@api/stt_streaming/warm_up.wav" \
  http://127.0.0.1:5000/api/v1/subtitle/tasks
# 輪詢任務狀態（直到 status=3 表示成功）
curl -X POST -H "Authorization: Bearer <TOKEN>" \
  http://127.0.0.1:5000/api/v1/subtitle/tasks/<ID>
# 下載字幕
curl -H "Authorization: Bearer <TOKEN>" \
  "http://127.0.0.1:5000/api/v1/subtitle/tasks/<ID>/subtitle?type=SRT" -o out.srt

# 4) 即時辨識測試頁（瀏覽器開啟，按麥克風即可）
#    http://127.0.0.1:5000/test_realtime.html
```

> 目前專案尚未附上自動化測試套件；`api/requirements.txt` 中保留 `pytest` / `httpx`
> 以便日後加入。

---

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
  This project provides a comprehensive framework for Automatic Speech Recognition (ASR), supporting multilingual speech processing and fine-tuning capabilities. It includes pre-trained models for Mandarin, Taiwanese, Hakka, and English, and tools for speech-to-text conversion and spoken language identification.

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
