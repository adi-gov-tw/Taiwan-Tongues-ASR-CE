# Taiwan Tongues ASR CE專案

本專案提供一套自動語音辨識（ASR, Automatic Speech Recognition）模型訓練流程，並附有已訓練好的國語、台語、客語、英語模型。你可以根據自己的語音資料進行微調（fine-tune），或直接使用現有模型進行語音辨識。

## 目錄結構

```
.
├── api/
│   ├── app.py                     # API服務
│   ├── file_asr.py                # 離線辨識(檔案辨識)
│   ├── streaming_asr.py           # 即時辨識
│   ├── test_files.html            # 健康檢查/單檔轉錄 測試頁
│   ├── test_realtime.html         # 即時辨識測試頁（裝置選擇/音量條/說話偵測）
│   ├── requirements.txt
│   ├── config.py
│   ├── start_app.bat              # 服務啟動（Port 5000）
│   ├── start_file_asr.bat
│   ├── start_streaming_asr.bat
│   ├── build.py
│   ├── logs/
│   ├── audio_files/
│   └── stt_streaming/
│       ├── README.md
│       ├── requirements.txt
│       └── src/
│           ├── asr/
│           │   └── faster_whisper_asr.py
│           ├── buffering_strategy/
│           ├── client.py
│           ├── server.py
│           └── ...
├── sample_corpus/
│   ├── train_ds_01/
│   │   ├── train.tsv
│   │   ├── test.tsv
│   │   ├── validated.tsv
│   │   └── clips/
│   │       └── ...
│   └── train_ds_02/
│       ├── train.tsv
│       ├── test.tsv
│       ├── validated.tsv
│       └── clips/
│           └── ...
├── models/               # 需至額外下載
├── model_for_finetune/   # 需至額外下載
├── train_asr.py
├── asr_core.py
├── cer.py
├── run.sh
├── requirements.txt
└── README.md
```

### 資料夾說明

- **sample_corpus/**  
  存放語音資料與標註檔案，每個子資料夾（如 `train_ds_01`、`train_ds_02`）代表一個資料集。每個資料集包含：
  - `train.tsv`、`test.tsv`、`validated.tsv`：標註檔案，以Tab分隔，包含語音檔案路徑與對應轉寫文字。
  - `clips/`：存放實際語音檔案，支援多層子目錄。

- **models/**  
  存放已訓練好的國語、台語、客語、英語模型，包含：
  - `model.bin`：模型權重檔案。
  - `config.json`、`preprocessor_config.json`、`tokenizer.json`、`vocabulary.json`：模型設定與詞彙表。
  - 注意：預設此資料夾不隨專案提供。請下載模型檔案，並將主要權重與設定檔放入專案根目錄的 `models/` 目錄。

- **model_for_finetune/**  
  存放可供 HuggingFace Transformers 訓練/微調使用的模型檢查點（checkpoint）。典型包含：
  - `pytorch_model.bin`：PyTorch 權重檔（或等價的 safetensors 檔）。
  - `config.json`、`preprocessor_config.json`、`tokenizer.json`：模型與處理設定檔。
  - 用途：可直接被 `train_asr.py` 以 `--model_name_or_path model_for_finetune` 載入進行微調，或作為基底模型繼續訓練。
  - 與 `models/` 差異：`models/` 偏向推論部署（如 faster-whisper/CTranslate2 風格），`model_for_finetune/` 為可反向傳播的 HF 檢查點格式。
  - 注意：預設此資料夾不隨專案提供。請下載模型檔案，並將主要權重與設定檔放入專案根目錄的 `model_for_finetune/` 目錄。

- **api/**  
  提供以 FastAPI 實作的 API 與頁面：
  - `file_asr.py`：任務式檔案轉錄 API（HTTP, 5000）。端點：
    - `GET /api/health`
    - `POST /api/v1/subtitle/tasks`（建立任務，立即回傳任務 id；背景處理）
    - `POST /api/v1/subtitle/tasks/{id}`（查任務狀態/進度）
    - `GET /api/v1/subtitle/tasks/{id}/subtitle-link?type=TXT|SRT|DIA`（取得下載連結）
    - `GET /api/v1/subtitle/tasks/{id}/subtitle?type=TXT|SRT|DIA`（直接下載字幕檔）
    - 測試頁：`/test_files.html`（顯示任務進度與 TXT/SRT 下載）
  - `streaming_asr.py`：即時串流 ASR（WebSocket）。端點包含：
    - `GET /health`、`GET /test`、測試頁（建議透過主服務 `/test_realtime.html` 載入）
    - WS `/ws/v1/transcript`（Query: `token`；合併模式直掛於 5000 埠）
  - `stt_streaming/`：即時串流的模組與策略實作（ASR、VAD、Buffering 等）。
  - 其餘檔案：啟動腳本（.bat）、需求檔、記錄檔目錄等。

- **train_asr.py**  
  訓練腳本，用於微調模型。

- **run.sh**  
  執行訓練的腳本，可根據需求修改參數。

- **README.md**  
  專案說明文件，提供使用指南與參考資訊。

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
   請先安裝 Python 3.8+ 及以下套件（建議使用虛擬環境）：
   ```
   pip install torch transformers datasets evaluate
   ```

2. **準備語料**  
   依照上述格式放置語音資料與標註檔案。

3. **執行訓練腳本**  
   可直接執行 `run.sh`，或根據需求修改參數：
   ```bash
   bash run.sh
   ```
   主要參數說明：
   - `--model_name_or_path`：預訓練模型名稱（如 openai/whisper-large-v3）
   - `--corpus_data_dir`：語料資料夾（如 sample_corpus）
   - `--dataset_config_name`：資料集組合（如 train_ds_01+train_ds_02）
   - `--language`：語言代碼（如 zh、en、nan、hak）
   - 其他參數可參考 `run.sh` 及 `train_asr.py`。

4. **訓練結果**  
   訓練完成後，模型與相關設定會儲存在 `output/` 目錄。

## 已訓練模型

- 已訓練好的國語、台語、客語、英語模型存放於 `models/` 目錄，包含：
  - `model.bin`：模型權重
  - `config.json`、`preprocessor_config.json`、`tokenizer.json`、`vocabulary.json`：模型設定與詞彙表

## 推論/辨識語音

可利用 HuggingFace Transformers 或自行撰寫推論腳本，載入 `models/` 內的模型進行語音辨識。

### 使用 asr_core.py 進行語音轉錄

`asr_core.py` 是一個專門用於語音轉錄的工具，可以處理指定資料夾中的所有音檔並進行自動語音辨識。

#### 功能特色

- **多格式支援**：支援 `.wav`、`.mp3`、`.flac`、`.m4a`、`.aac` 等音檔格式
- **批次處理**：可一次處理資料夾中的所有音檔
- **中文優化**：針對中文語音進行優化，包含簡繁轉換、數字轉換等後處理
- **錯誤處理**：即使個別檔案處理失敗，仍會繼續處理其他檔案
- **自動輸出**：每個音檔會產生對應的 `{檔名}_asr.txt` 轉錄結果檔案

#### 使用方法

```bash
python asr_core.py <音檔資料夾路徑>
```

**參數說明：**
- `folder`：音檔資料夾路徑（必需）
- `--output`：輸出檔案名稱（可選，預設為 `transcription_results.txt`，已棄用）

**使用範例：**
```bash
# 處理 test_dataset 資料夾中的所有音檔
python asr_core.py test_dataset

# 處理指定路徑的音檔
python asr_core.py /path/to/audio/files
```

#### 輸出格式

每個音檔會產生一個對應的轉錄結果檔案，檔名格式為 `{原檔名}_asr.txt`，內容包含：
- 轉錄的文字內容（經過後處理）
- 如果處理失敗，會記錄錯誤訊息

#### 後處理功能

轉錄結果會經過以下後處理：
- **簡繁轉換**：將簡體中文轉換為繁體中文
- **數字轉換**：將中文數字轉換為阿拉伯數字
- **特殊字符移除**：移除標點符號和特殊字符
- **全形轉半形**：將全形字符轉換為半形字符
- **文字正規化**：統一文字格式

#### 系統需求

- Python 3.8+
- CUDA 12.4+ 支援的 GPU（建議）
- 依賴套件： 請參考 `requirements.txt`

#### 注意事項

- 確保 `models/` 資料夾中有正確的模型檔案
- 建議使用 GPU 以加速處理速度
- 處理大量檔案時請確保有足夠的磁碟空間存放結果

## 測試

本專案提供完整的 Python 測試框架，確保 API 功能的正確性。

### 測試環境設定

1. **安裝測試依賴**
   ```bash
   pip install pytest pytest-asyncio httpx
   ```

2. **執行測試**
   ```bash
   # 執行所有測試
   pytest api/tests/
   
   # 執行特定測試檔案
   pytest api/tests/test_file_asr.py
   
   # 顯示詳細輸出
   pytest -v api/tests/
   ```

### 測試內容

#### `test_file_asr.py`

測試檔案 ASR API 的各項功能：

- **健康檢查測試**：驗證 `/api/health` 端點正常運作
- **認證測試**：測試登入、登出功能
- **權限測試**：驗證未授權存取會被拒絕
- **轉錄測試**：測試音檔上傳和轉錄功能
- **CER 計算測試**：驗證字元錯誤率計算

#### 測試特色

- **模擬模型**：使用假模型避免載入實際 Whisper 模型，加速測試執行
- **獨立資料庫**：每次測試使用獨立的 SQLite 資料庫
- **環境隔離**：透過環境變數設定測試專用的配置
- **自動清理**：測試完成後自動清理測試資料

#### 測試資料

測試會自動生成測試音檔：
- 格式：WAV
- 取樣率：16 kHz
- 聲道：單聲道
- 內容：440 Hz 正弦波（0.2 秒）

### 測試配置

測試使用以下環境變數：
- `ASR_API_AUTH_DB`：測試用認證資料庫路徑
- `ASR_API_JWT_SECRET`：測試用 JWT 密鑰
- `ASR_API_JWT_ALGORITHM`：測試用 JWT 演算法
- `ASR_API_BOOTSTRAP_ADMIN_USERNAME`：測試用管理員帳號
- `ASR_API_BOOTSTRAP_ADMIN_PASSWORD`：測試用管理員密碼

## 參考

- 若需自訂資料集或語言，請參考 `train_asr.py` 內部註解與參數說明。


## 離線辨識與即時辨識API

本專案在 `api/` 目錄提供兩種服務：檔案轉錄（HTTP）與即時串流（WebSocket）。模型請放在專案根目錄 `models/`。

### 安裝與啟動

1) 安裝依賴

```bash
pip install -r api/requirements.txt
```

2) 啟動服務（單一埠 5000，HTTP 開發模式）

```bash
api/start_app.bat
```

> 注意：如需對外提供 HTTPS，建議以 Nginx/Caddy 作為反向代理終結 TLS。

### File ASR（HTTP, 5000）API 規格（任務式）

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
  - 範例：
    ```bash
    curl -H "Authorization: Bearer <TOKEN>" \
      -F "audio=@api/test.wav" \
      http://127.0.0.1:5000/api/v1/subtitle/tasks
    ```

#### 建立任務：擴充 / 進階 Body 欄位

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
  - 回傳：
    ```json
    {
      "code": 200,
      "data": [
        { "id": 3, "types": ["SRT", "TXT", "DIA"] }
      ]
    }
    ```
  - 任務不存在或請求有誤時回傳 404/400。

- GET `/test_files.html`：測試頁（健康檢查 / 單一音檔轉錄）

### 認證 API（HTTP, 5000）規格

基底 URL：`http://127.0.0.1:5000`

- GET `/api/v1/health`：認證服務健康檢查
  ```json
  {"status":"ok"}
  ```

- POST `/api/v1/login`：使用者登入
  - 請求體：
    ```json
    {
      "username": "admin",
      "password": "admin@0935",
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

#### 認證系統特色

- **JWT Token 認證**：使用 JWT 進行無狀態認證
- **角色權限管理**：支援 `admin` 和 `user` 兩種角色
- **自動管理員建立**：首次啟動時自動建立預設管理員帳號
- **密碼過期管理**：支援使用者帳號過期時間設定
- **SQLite 資料庫**：輕量級使用者資料儲存
- **環境變數配置**：可透過環境變數自訂預設值

#### 預設管理員帳號

- 使用者名稱：`admin`
- 密碼：`admin@0935`
- 角色：`admin`
- 過期時間：`2099-12-31`

#### 環境變數

- `ASR_API_AUTH_DB`：認證資料庫路徑
- `ASR_API_JWT_SECRET`：JWT 密鑰
- `ASR_API_JWT_ALGORITHM`：JWT 演算法（預設：HS256）
- `ASR_API_BOOTSTRAP_ADMIN_USERNAME`：預設管理員使用者名稱
- `ASR_API_BOOTSTRAP_ADMIN_PASSWORD`：預設管理員密碼
- `ASR_API_BOOTSTRAP_ADMIN_NICKNAME`：預設管理員暱稱
- `ASR_API_RESET_ADMIN_ON_STARTUP`：啟動時是否重設管理員密碼

### Streaming ASR（WebSocket）API 規格

- Path：
  - 基底 URL：`http://127.0.0.1:5000/stream`
  - 測試頁：`http://127.0.0.1:5000/test_realtime.html`

- GET `/stream/health`：健康檢查
  ```json
  {"status":"healthy","connected_clients":0,"vad_pipeline":"ready","asr_pipeline":"ready","asr_device":"cuda","asr_compute_type":"float16","asr_model_size":"models"}
  ```

- GET `/test_realtime.html`：即時辨識測試頁

- GET `/test`：簡易測試頁

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

### 音訊規格建議
- 取樣率：16 kHz
- 聲道：mono（單聲道）
- 位寬：16-bit（二進位請送 Int16 PCM）
- 建議開始錄音前約 1 秒保持安靜，利於噪音底噪校準與說話偵測。

#### 擴充 / 進階 Query 參數

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

### 引用

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