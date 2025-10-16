import os
import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_env(tmp_path_factory):
    # 設定獨立的測試 DB 與 JWT 參數
    db_dir = tmp_path_factory.mktemp("authdb")
    db_path = db_dir / "auth_test.db"
    os.environ["ASR_API_AUTH_DB"] = str(db_path)
    os.environ["ASR_API_JWT_SECRET"] = "TEST_SECRET"
    os.environ["ASR_API_JWT_ALGORITHM"] = "HS256"
    os.environ["ASR_API_BOOTSTRAP_ADMIN_USERNAME"] = "admin"
    os.environ["ASR_API_BOOTSTRAP_ADMIN_PASSWORD"] = "admin@0935"
    os.environ["ASR_API_BOOTSTRAP_ADMIN_NICKNAME"] = "ADMIN"
    os.environ["ASR_API_RESET_ADMIN_ON_STARTUP"] = "1"
    return {"db_path": str(db_path)}


@pytest.fixture()
def client(test_env, monkeypatch):
    # 匯入 app 與模組以便 monkeypatch（同目錄下的 file_asr）
    import sys, os

    api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    import file_asr as fasr

    # 假模型：避免實際載入 Whisper
    class DummySegment:
        def __init__(self, text):
            self.text = text

    class DummyModel:
        def transcribe(
            self,
            audio,
            language="zh",
            word_timestamps=False,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=True,
            initial_prompt="",
        ):
            segments = [DummySegment("這是單元測試")]  # 簡單回傳固定文本
            info = {}
            return segments, info

    def _mock_load_model() -> bool:
        fasr.whisper_model = DummyModel()
        return True

    # 替換載入模型邏輯
    monkeypatch.setattr(fasr, "load_model", _mock_load_model)

    # 建立 TestClient（使用 context manager 觸發 lifespan -> auth_startup 建立/重設 admin）
    with TestClient(fasr.app) as test_client:
        yield test_client


def _login_and_get_token(client: TestClient) -> str:
    resp = client.post(
        "/api/v1/login",
        json={
            "username": "admin",
            "password": "admin@0935",
            "rememberMe": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("code") == 200
    token = data.get("token")
    assert isinstance(token, str) and len(token) > 10
    return token


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "timestamp" in data


def test_login_and_logout(client: TestClient):
    token = _login_and_get_token(client)
    # logout
    resp = client.post("/api/v1/logout", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("code") == 200
    assert data.get("username") == "admin"


def test_transcribe_unauthorized(client: TestClient):
    # 未帶 token 應 401
    wav = _make_wav_bytes()
    files = {"audio": ("test.wav", wav, "audio/wav")}
    resp = client.post("/api/transcribe", files=files)
    assert resp.status_code == 401


def test_transcribe_success(client: TestClient):
    token = _login_and_get_token(client)
    wav = _make_wav_bytes()
    files = {"audio": ("test.wav", wav, "audio/wav")}
    data = {"reference_text": "這是單元測試"}
    resp = client.post(
        "/api/transcribe",
        files=files,
        data=data,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload.get("success") is True
    assert isinstance(payload.get("asr_result"), str)
    # 有提供 reference_text 時，應回傳 cer_result 欄位（若 compare_texts 正常）
    assert "cer_result" in payload


def test_test_files_html(client: TestClient):
    # 檔案可能不存在 -> 200 或 404 都接受
    resp = client.get("/test_files.html")
    assert resp.status_code in (200, 404)


def _make_wav_bytes(duration_sec: float = 0.2, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(
        file=buf, data=audio, samplerate=sample_rate, format="WAV", subtype="PCM_16"
    )
    buf.seek(0)
    return buf.read()
