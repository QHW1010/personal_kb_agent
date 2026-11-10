"""FastAPI 依赖：当前用户解析、JWT 校验。"""

from __future__ import annotations

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AuthError
from app.core.logging import get_logger
from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_session

logger = get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False, description="在 Authorization 头携带 Bearer <token>")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthError("缺少 Bearer Token")
    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token 已过期，请重新登录") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Token 无效") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise AuthError("Token 缺少主体信息")

    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthError("用户不存在或已停用")
    return user
