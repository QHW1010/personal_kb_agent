"""认证链路：注册 / 登录 / Token 校验。"""

from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_roundtrip() -> None:
    hashed = hash_password("correct horse battery")
    assert hashed != "correct horse battery"
    assert verify_password("correct horse battery", hashed)
    assert not verify_password("wrong password", hashed)


def test_password_longer_than_bcrypt_limit() -> None:
    """bcrypt 只处理 72 字节，超长密码不应崩溃。"""
    long_pw = "x" * 200
    hashed = hash_password(long_pw)
    assert verify_password(long_pw, hashed)


def test_jwt_roundtrip() -> None:
    token = create_access_token("user-123")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["typ"] == "access"


async def test_register_returns_token(client: AsyncClient) -> None:
    email = f"a{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post("/api/v1/auth/register", json={"email": email, "password": "sup3rsecret!"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == email
    assert "hashed_password" not in body["user"]


async def test_duplicate_email_conflicts(client: AsyncClient) -> None:
    email = f"b{uuid.uuid4().hex[:8]}@example.com"
    payload = {"email": email, "password": "sup3rsecret!"}
    assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 201
    dup = await client.post("/api/v1/auth/register", json=payload)
    assert dup.status_code == 409
    assert dup.json()["error"]["code"] == "conflict"


async def test_short_password_rejected(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": f"c{uuid.uuid4().hex[:8]}@example.com", "password": "short"},
    )
    assert resp.status_code == 422


async def test_login_and_me(client: AsyncClient) -> None:
    email = f"d{uuid.uuid4().hex[:8]}@example.com"
    password = "sup3rsecret!"
    await client.post("/api/v1/auth/register", json={"email": email, "password": password})

    login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email


async def test_wrong_password_is_401(client: AsyncClient) -> None:
    email = f"e{uuid.uuid4().hex[:8]}@example.com"
    await client.post("/api/v1/auth/register", json={"email": email, "password": "sup3rsecret!"})
    resp = await client.post("/api/v1/auth/login", json={"email": email, "password": "wrong-password"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_chat_without_token_is_401(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/chat", json={"message": "你好", "stream": False})
    assert resp.status_code == 401


async def test_garbage_token_is_401(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401
