"""可选的 LLM 适配层（默认关闭，零依赖时自动降级）。

一个重要取舍：**模型不参与抽取、排序和合规判断**，只参与"把话说顺"。
原因：
- 排序要能被医生当面质疑并当场复算，模型给不出稳定的中间量；
- 合规判断要可审计，prompt 的随机性不能进入责任链；
- 抽取环节换成模型后，必须先在金标集上打赢规则基线，否则不值得引入成本和延迟。

所以 v0.1 的定位是：确定性内核 + 可选的语言润色。这也意味着断网、无 key、
超预算时产品依然完整可用——这在医院内网环境里不是加分项，是准入项。
"""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "你是医疗 AI 产品经理。把给定的需求陈述改写得像人话：保留全部事实、数字与限定条件，"
    "不要新增能力承诺，不要删掉「人确认」「不替代诊断」这类边界。只输出改写后的一句话，不要解释。"
)


def available() -> bool:
    return bool(os.environ.get("CLINSIGHT_LLM_KEY") or os.environ.get("OPENAI_API_KEY"))


def _config() -> tuple[str, str, str]:
    key = os.environ.get("CLINSIGHT_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("CLINSIGHT_LLM_BASE", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("CLINSIGHT_LLM_MODEL", DEFAULT_MODEL)
    return key, base, model


def rewrite(text: str, timeout: float = 12.0) -> str | None:
    """失败一律返回 None，调用方保持原文——不因为润色失败而丢结果。"""
    if not available():
        return None
    key, base, model = _config()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        out = body["choices"][0]["message"]["content"].strip()
        if not out or len(out) > 400:
            return None
        return out
    except Exception:  # 网络 / 配额 / 格式问题都走降级
        return None
