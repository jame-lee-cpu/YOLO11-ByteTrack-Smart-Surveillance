"""认证模块：注册、登录、密码哈希与角色校验。"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from config import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME, DEFAULT_DB_PATH
from user_db import UserDB

ADMIN_USERNAME = DEFAULT_ADMIN_USERNAME
ADMIN_PASSWORD = DEFAULT_ADMIN_PASSWORD
ADMIN_ROLE = "admin"


@dataclass
class LoginUser:
    id: int
    username: str
    role: str
    created_at: str


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """PBKDF2 哈希，返回格式：pbkdf2_sha256$iters$salt_hex$hash_hex。"""
    if salt is None:
        salt = os.urandom(16)
    iterations = 120000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters_s, salt_hex, hash_hex = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(got, expected)


class AuthService:
    """认证服务：提供注册/登录/管理员能力。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db = UserDB(db_path=db_path)
        self._ensure_default_admin()

    def _ensure_default_admin(self) -> None:
        admin = self.db.get_user_by_username(ADMIN_USERNAME)
        if admin is None:
            self.db.create_user(
                username=ADMIN_USERNAME,
                password_hash=_hash_password(ADMIN_PASSWORD),
                role=ADMIN_ROLE,
            )

    def register(self, username: str, password: str, confirm_password: str) -> Tuple[bool, str]:
        username = (username or "").strip()
        if not username:
            return False, "用户名不能为空。"
        if self.db.get_user_by_username(username) is not None:
            return False, "用户名已存在，请更换。"
        if len(password or "") < 6:
            return False, "密码长度至少 6 位。"
        if password != confirm_password:
            return False, "两次输入的密码不一致。"

        self.db.create_user(username=username, password_hash=_hash_password(password), role="user")
        return True, "注册成功。"

    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[LoginUser]]:
        username = (username or "").strip()
        if not username:
            return False, "用户名不能为空。", None
        row = self.db.get_user_by_username(username)
        if row is None:
            return False, "用户不存在。", None
        if not _verify_password(password or "", str(row["password_hash"])):
            return False, "密码错误。", None
        user = LoginUser(
            id=int(row["id"]),
            username=str(row["username"]),
            role=str(row["role"]),
            created_at=str(row["created_at"]),
        )
        return True, "登录成功。", user

    def list_users(self):
        return self.db.list_users()

    def delete_user(self, operator_role: str, operator_username: str, username_to_delete: str) -> Tuple[bool, str]:
        target = (username_to_delete or "").strip()
        if operator_role != "admin":
            return False, "权限不足：仅管理员可删除用户。"
        if not target:
            return False, "请输入要删除的用户名。"
        if target == ADMIN_USERNAME:
            return False, "默认管理员账号不可删除。"
        if target == operator_username:
            return False, "不能删除当前登录账号。"

        user = self.db.get_user_by_username(target)
        if user is None:
            return False, "目标用户不存在。"
        if str(user["role"]) == "admin":
            return False, "不能删除管理员用户。"
        ok = self.db.delete_user(target)
        if not ok:
            return False, "删除失败，请重试。"
        return True, f"已删除用户：{target}"
