"""认证路由：注册 / 登录 / 当前用户。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.errors import AuthError, ConflictError
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.db.session import get_session
from app.schemas.auth import Token, UserCreate, UserLogin, UserRead

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _issue(user: User) -> Token:
    cfg = get_settings()
    token = create_access_token(user.id, extra={"email": user.email})
    return Token(
        access_token=token,
        expires_in=cfg.access_token_expire_minutes * 60,
        user=UserRead.model_validate(user),
    )


@router.post(
    "/register",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="注册并直接签发 Token",
)
async def register(payload: UserCreate, session: AsyncSession = Depends(get_session)) -> Token:
    exists = await session.scalar(select(User).where(User.email == payload.email))
    if exists is not None:
        raise ConflictError("该邮箱已注册", details={"email": payload.email})

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    session.add(user)
    await session.flush()
    await session.refresh(user)
    logger.info("user_registered id=%s", user.id)
    return _issue(user)


@router.post("/login", response_model=Token, summary="邮箱密码登录")
async def login(payload: UserLogin, session: AsyncSession = Depends(get_session)) -> Token:
    user = await session.scalar(select(User).where(User.email == payload.email))
    if user is None or not verify_password(payload.password, user.hashed_password):
        # 不区分"用户不存在"与"密码错误"，避免账号枚举
        raise AuthError("邮箱或密码不正确")
    if not user.is_active:
        raise AuthError("账号已停用")
    return _issue(user)


@router.get("/me", response_model=UserRead, summary="当前登录用户")
async def me(current: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current)
