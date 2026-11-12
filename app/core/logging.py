"""日志配置：单行结构化，方便 docker logs 与后续接采集。"""

from __future__ import annotations

import logging
import sys

from app.core.config import get_settings

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    cfg = get_settings()
    # 日志走 stderr，stdout 留给结构化输出。评估脚本的 --json 要能被 jq / CI 直接消费，
    # 日志混进 stdout 会把整份 JSON 变成非法输入（每行前面多一条时间戳）。
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(cfg.log_level.upper())
    # 降噪
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
