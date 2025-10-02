import os
import sys
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, Header, status, Query, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))
from auth_shared import (
    generate_jwt_token,
    verify_jwt_token,
)

DB_PATH = os.getenv(
    "ASR_API_AUTH_DB", os.path.join(os.path.dirname(__file__), "auth.db")
)


def _ensure_db_schema() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                nickname TEXT,
                role TEXT NOT NULL,
                comment TEXT,
                password_hash TEXT NOT NULL,
                status INTEGER NOT NULL,
                expired_time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


# 密碼雜湊（改用 pbkdf2_sha256，避免外部 bcrypt 相依問題）
try:
    from passlib.hash import pbkdf2_sha256
except Exception as e:  # pragma: no cover
    pbkdf2_sha256 = None


def _hash_password(password: str) -> str:
    if not pbkdf2_sha256:
        raise RuntimeError("passlib 未安裝")
    return pbkdf2_sha256.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    if not pbkdf2_sha256:
        raise RuntimeError("passlib 未安裝")
    try:
        return pbkdf2_sha256.verify(password, password_hash)
    except Exception:
        return False


class LoginRequest(BaseModel):
    username: str
    password: str
    rememberMe: int = Field(default=0)


class CreateUserRequest(BaseModel):
    username: str
    nickname: str
    role: str = Field(pattern=r"^(admin|user)$")
    comment: Optional[str] = ""
    password: str
    expiredTime: datetime  # ISO8601，Swagger 會顯示為 date-time
    status: int = Field(default=1)


def _parse_iso8601(dt_str: str) -> datetime:
    try:
        # 允許結尾 Z
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(dt_str)
        # 若無時區資訊，視為 UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        raise HTTPException(status_code=400, detail="invalid expiredTime format")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _require_token_payload(credentials: Optional[HTTPAuthorizationCredentials]) -> Dict:
    token = credentials.credentials if credentials else None
    payload = verify_jwt_token(token)
    return payload


def _require_admin(payload: Dict) -> None:
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin only")


router = APIRouter(prefix="/api/v1", tags=["auth"])

# Swagger/OpenAPI：宣告 Bearer 安全方案（供 /logout、/user、/user/password 於文件中顯示鎖頭並自動帶 Authorization）
bearer_scheme = HTTPBearer(auto_error=False)


def auth_startup() -> None:
    _ensure_db_schema()
    # 若資料庫尚無任何使用者，建立預設管理員（可用環境變數覆蓋）
    bootstrap_username = os.getenv("ASR_API_BOOTSTRAP_ADMIN_USERNAME", "admin")
    bootstrap_password = os.getenv("ASR_API_BOOTSTRAP_ADMIN_PASSWORD", "admin@0935")
    bootstrap_nickname = os.getenv("ASR_API_BOOTSTRAP_ADMIN_NICKNAME", "ADMIN")
    with get_db_conn() as conn:
        # 直接檢查是否存在該管理員帳號；不存在才建立（避免覆蓋既有密碼）
        cur = conn.execute(
            "SELECT username FROM users WHERE username=?", (bootstrap_username,)
        )
        exists = cur.fetchone() is not None
        if not exists:
            now_iso = _now_utc().isoformat()
            expired_iso = datetime(
                2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc
            ).isoformat()
            conn.execute(
                """
                INSERT INTO users (username, nickname, role, comment, password_hash, status, expired_time, created_at, updated_at)
                VALUES (?, ?, 'admin', '', ?, 1, ?, ?, ?)
                """,
                (
                    bootstrap_username,
                    bootstrap_nickname,
                    _hash_password(bootstrap_password),
                    expired_iso,
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
        else:
            # 可選：啟動時重設 admin 密碼為 bootstrap_password（預設開啟，可用環境變數關閉）
            if os.getenv("ASR_API_RESET_ADMIN_ON_STARTUP", "1") in (
                "1",
                "true",
                "True",
            ):
                now_iso = _now_utc().isoformat()
                expired_iso = datetime(
                    2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc
                ).isoformat()
                conn.execute(
                    "UPDATE users SET password_hash=?, status=1, expired_time=?, updated_at=? WHERE username=?",
                    (
                        _hash_password(bootstrap_password),
                        expired_iso,
                        now_iso,
                        bootstrap_username,
                    ),
                )
                conn.commit()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/login")
def login(req: LoginRequest):
    with get_db_conn() as conn:
        cur = conn.execute(
            "SELECT username, nickname, role, password_hash, status, expired_time FROM users WHERE username=?",
            (req.username,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="invalid credentials")
        username, nickname, role, password_hash, status_flag, expired_time_str = row

        if status_flag != 1:
            raise HTTPException(status_code=403, detail="user disabled")
        try:
            expired_time = _parse_iso8601(expired_time_str)
        except HTTPException:
            # 若資料格式損壞，視同過期
            raise HTTPException(status_code=403, detail="user expired")
        if expired_time <= _now_utc():
            return {"code": 200, "pwdExpired": 1}

        if not _verify_password(req.password, password_hash):
            raise HTTPException(status_code=401, detail="invalid credentials")

        expiration = 34560000 if int(req.rememberMe or 0) else 86400
        token = generate_jwt_token(
            {
                "sub": username,
                "role": role,
                "nickname": nickname,
                "loginType": "default",
                "expiration": expiration,
            },
            expires_in_seconds=expiration,
        )
        return {
            "code": 200,
            "token": token,
            "expiration": expiration,
            "pwdExpired": 0,
        }


@router.post("/logout")
def logout(
    __credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    payload = _require_token_payload(__credentials)
    username = payload.get("sub") or payload.get("username") or ""
    return {"code": 200, "username": username, "message": "logged out"}


@router.post("/user")
def create_user(
    req: CreateUserRequest,
    __credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    payload = _require_token_payload(__credentials)
    _require_admin(payload)

    # Pydantic 已轉為 datetime
    expired_dt = req.expiredTime
    now_iso = _now_utc().isoformat()
    pwd_hash = _hash_password(req.password)

    with get_db_conn() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (username, nickname, role, comment, password_hash, status, expired_time, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req.username,
                    req.nickname,
                    req.role,
                    req.comment or "",
                    pwd_hash,
                    int(req.status),
                    expired_dt.isoformat(),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="username exists")

    return {"code": 200, "username": req.username, "message": "added"}


@router.put("/user/password")
def update_password(
    username: str = Query(...),
    newPassword: str = Query(...),
    __credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    payload = _require_token_payload(__credentials)
    is_admin = payload.get("role") == "admin"
    requester = payload.get("sub")

    if not is_admin and requester != username:
        raise HTTPException(status_code=403, detail="forbidden")

    with get_db_conn() as conn:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username=?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user not found")
        old_hash = row[0]

        # 本人或管理員皆可直接變更密碼，不再要求 currentPassword

        new_hash = _hash_password(newPassword)
        conn.execute(
            "UPDATE users SET password_hash=?, updated_at=? WHERE username=?",
            (new_hash, _now_utc().isoformat(), username),
        )
        conn.commit()

        # 驗證更新是否生效
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username=?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user not found after update")
        if not _verify_password(newPassword, row[0]):
            raise HTTPException(
                status_code=500, detail="password update verification failed"
            )

    return {"code": 200, "username": username, "message": "password updated"}


__all__ = [
    "router",
    "auth_startup",
]
