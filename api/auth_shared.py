import os
import time
from typing import Optional, Tuple, Dict

import jwt
from fastapi import HTTPException, status


# 讀取 JWT 設定（可用環境變數覆蓋）
JWT_SECRET = os.getenv("ASR_API_JWT_SECRET", "CHANGE_ME_SECRET")
JWT_ALGORITHM = os.getenv("ASR_API_JWT_ALGORITHM", "HS256")


def generate_jwt_token(claims: Dict, expires_in_seconds: int) -> str:
    """產生 JWT。

    claims: 自定義 payload 欄位
    expires_in_seconds: token 期限（秒）
    """
    now_ts = int(time.time())
    payload = {
        **claims,
        "iat": now_ts,
        "exp": now_ts + int(expires_in_seconds),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    # PyJWT v2 會回傳 str
    return token


def verify_jwt_token(token: str) -> Dict:
    """驗證 JWT 並回傳 payload，失敗則拋出 401。"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
        )


def get_bearer_token_from_authorization_header(authorization: Optional[str]) -> str:
    """從 Authorization 標頭擷取 Bearer token。"""
    if not authorization:
        raise HTTPException(status_code=401, detail="authorization header required")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="invalid authorization header")
    return parts[1].strip()
