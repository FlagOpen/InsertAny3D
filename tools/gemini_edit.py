#!/usr/bin/env python3
"""Call an APIYi/Gemini image-edit endpoint from the command line.

The API key is deliberately read from GEMINI_API_KEY (or --api-key) and is
never included in output files or status messages.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import requests


DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_IMAGE_URL = (
    "https://api.apiyi.com/v1beta/models/"
    "gemini-3.1-flash-image-preview:generateContent"
)
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _mime_type(path: Path) -> str:
    value, _ = mimetypes.guess_type(path.name)
    return value or "image/png"


def _endpoint_from_args(args: argparse.Namespace) -> str:
    endpoint = args.endpoint or os.environ.get("GEMINI_IMAGE_URL")
    if not endpoint:
        # APIYi exposes the native Gemini route beside its OpenAI-compatible
        # /v1 base.  Derive /v1beta from the configured base when callers do
        # not provide the more specific GEMINI_IMAGE_URL.
        parts = urlsplit(args.base_url or DEFAULT_BASE_URL)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        root = urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
        endpoint = f"{root}/v1beta/models/{args.model}:generateContent"
    # APIYi deployments can expose a model-specific image route while using a
    # different model name for metadata.  Preserve an explicitly configured
    # image URL exactly; callers who need another route can pass --endpoint.
    return endpoint


def _redact(value: str, secret: str | None) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _decode_data_uri(value: str) -> tuple[bytes, str] | None:
    match = re.match(r"^data:([^;,]+)(?:;base64)?,(.*)$", value, flags=re.S)
    if not match:
        return None
    mime = match.group(1)
    encoded = match.group(2)
    try:
        return base64.b64decode(encoded), mime
    except (ValueError, base64.binascii.Error):
        return None


def _extract_image(response: Any, session: requests.Session, timeout: float) -> tuple[bytes, str] | None:
    """Accept native Gemini inlineData and common OpenAI-compatible variants."""
    for item in _iter_dicts(response):
        for key in ("inlineData", "inline_data"):
            candidate = item.get(key)
            if isinstance(candidate, dict) and candidate.get("data"):
                try:
                    return base64.b64decode(candidate["data"]), candidate.get("mimeType", "image/png")
                except (ValueError, base64.binascii.Error):
                    pass
        if item.get("b64_json"):
            try:
                return base64.b64decode(item["b64_json"]), item.get("mime_type", "image/png")
            except (ValueError, base64.binascii.Error):
                pass
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str):
            decoded = _decode_data_uri(image_url)
            if decoded:
                return decoded
            if image_url.startswith(("http://", "https://")):
                downloaded = session.get(image_url, timeout=timeout)
                downloaded.raise_for_status()
                return downloaded.content, downloaded.headers.get("Content-Type", "image/png").split(";", 1)[0]
    return None


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for item in _iter_dicts(response):
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(dict.fromkeys(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 APIYi/Gemini 根据提示词编辑图片")
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--endpoint", help="完整 generateContent URL，默认使用 GEMINI_IMAGE_URL")
    parser.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=None, help="不建议命令行传递；默认读取 GEMINI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--response-json", type=Path, help="可选：保存不含请求头的原始 JSON 响应")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入和请求配置，不发送请求")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_image.is_file():
        raise SystemExit(f"输入图片不存在: {args.input_image}")
    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise SystemExit("提示词为空")

    endpoint = _endpoint_from_args(args)
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if args.dry_run:
        print("GEMINI_EDIT_CONFIG", endpoint, args.base_url, args.model, args.input_image, args.output_image)
        return 0
    if not api_key:
        raise SystemExit("未找到 GEMINI_API_KEY；请通过环境变量提供 API key")

    image_bytes = args.input_image.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inlineData": {"mimeType": _mime_type(args.input_image), "data": encoded}},
                ],
            }
        ],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    session = requests.Session()
    response = None
    last_error = None
    for attempt in range(max(0, args.retries) + 1):
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=args.timeout)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= args.retries:
                detail = ""
                if response is not None:
                    detail = _redact(response.text[:2000], api_key)
                raise SystemExit(f"Gemini 请求失败: {_redact(str(exc), api_key)}\n{detail}") from exc
    if response is None:
        raise SystemExit(f"Gemini 请求失败: {last_error}")

    try:
        response_json = response.json()
    except ValueError as exc:
        raise SystemExit(f"Gemini 返回不是 JSON: {_redact(response.text[:1000], api_key)}") from exc
    if args.response_json:
        args.response_json.parent.mkdir(parents=True, exist_ok=True)
        args.response_json.write_text(json.dumps(response_json, ensure_ascii=False, indent=2), encoding="utf-8")

    image = _extract_image(response_json, session, args.timeout)
    if image is None:
        text = _response_text(response_json)
        raise SystemExit("Gemini 响应中没有找到图片数据" + (f"；文本响应: {text[:500]}" if text else ""))
    image_bytes, mime = image
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_image.write_bytes(image_bytes)
    print("GEMINI_EDIT_READY", args.output_image, mime)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("已取消")
