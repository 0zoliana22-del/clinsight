"""LLM 适配层：OpenAI 兼容接口 + 严格失败降级。

三条原则：
1. **失败一律降级**，不抛异常打断整条流水线——医院内网断网是常态；
2. **确定性优先**：temperature=0，能复现；
3. **模型只做语言理解，不做责任判断**：抽取可以交给模型，排序和合规不行。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class LLMError(RuntimeError):
    """模型调用失败。调用方必须捕获并降级。"""


def config() -> dict[str, str]:
    return {
        "key": os.environ.get("CLINSIGHT_LLM_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        "base": os.environ.get("CLINSIGHT_LLM_BASE", DEFAULT_BASE).rstrip("/"),
        "model": os.environ.get("CLINSIGHT_LLM_MODEL", DEFAULT_MODEL),
    }


def available() -> bool:
    return bool(config()["key"])


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
    retries: int = 2,
) -> str:
    cfg = config()
    if not cfg["key"]:
        raise LLMError("未配置 CLINSIGHT_LLM_KEY")

    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last: str = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{cfg['base']}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['key']}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:300]}"
        except Exception as exc:  # 网络 / 超时 / 结构异常
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(last or "unknown error")


def chat_json(messages: list[dict[str, str]], **kwargs) -> dict:
    """要求模型返回 JSON 对象；解析失败视为调用失败，交给上层降级。"""
    raw = chat(messages, **kwargs)
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"返回不是合法 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("返回的 JSON 顶层不是对象")
    return parsed


SYSTEM_PROMPT = (
    "你是医疗 AI 产品经理。把给定的需求陈述改写得像人话：保留全部事实、数字与限定条件，"
    "不要新增能力承诺，不要删掉「人确认」「不替代诊断」这类边界。只输出改写后的一句话，不要解释。"
)


def rewrite(text: str, timeout: float = 12.0) -> str | None:
    """失败返回 None，调用方保持原文——不因为润色失败而丢结果。"""
    if not available():
        return None
    try:
        out = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=256,
            timeout=timeout,
            retries=0,
        ).strip()
    except LLMError:
        return None
    return out if out and len(out) <= 400 else None
