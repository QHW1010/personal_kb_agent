"""认证原语：bcrypt 密码哈希 + JWT 签发/校验。

不依赖 passlib（其 1.7.4 与 bcrypt>=4.1 存在兼容性问题），直接用 bcrypt。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings

# bcrypt 只处理前 72 字节，超长密码先截断，避免静默抛错
_BCRYPT_MAX_BYTES = 72


def _normalize(password: str) -> bytes:
    raw = password.encode("utf-8")
    return raw[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_normalize(password), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_normalize(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    cfg = get_settings()
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=cfg.access_token_expire_minutes)).timestamp()),
        "typ": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, cfg.secret_key, algorithm=cfg.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """解码失败抛 jwt.PyJWTError，由上层转 401。"""
    cfg = get_settings()
    return jwt.decode(token, cfg.secret_key, algorithms=[cfg.jwt_algorithm])
