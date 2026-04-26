"""
Test script for NIM endpoint connectivity and rate limit status.
Uses the OpenAI SDK format (same as graphrag/litellm uses) to test
qwen/qwen3.5-397b-a17b via NVIDIA NIM free endpoint.
"""
import sys
import json
import time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def load_nim_config():
    """Load NIM config from .env file."""
    config = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("NIM_"):
                key, _, value = line.partition("=")
                config[key] = value.strip()
    return config.get("NIM_API_KEY", ""), config.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


def test_nim_completion(client: OpenAI, model: str, prompt: str, max_tokens: int = 100):
    """Send a single test request via OpenAI SDK client."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=30,
        )
        return {
            "status": "success",
            "model": response.model,
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)
        if hasattr(e, "status_code"):
            return {"status": "error", "code": e.status_code, "error": {"type": err_type, "message": err_msg}}
        return {"status": "error", "code": None, "error": {"type": err_type, "message": err_msg}}


def check_rate_limit(client: OpenAI, model: str):
    """Send a minimal request to check if rate limit is currently active."""
    return test_nim_completion(client, model, "hi", max_tokens=5)


def main():
    api_key, base_url = load_nim_config()
    if not api_key:
        print("ERROR: NIM_API_KEY not found in .env file")
        sys.exit(1)

    # Build OpenAI client pointing at NIM endpoint (same way litellm/graphrag does)
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = "qwen/qwen3.5-397b-a17b"

    print(f"Endpoint: {base_url}")
    print(f"Model: {model}")
    print(f"API key: {api_key[:12]}...{api_key[-4:]}")
    print()

    # Quick rate limit check
    print("=== Rate Limit Check ===")
    result = check_rate_limit(client, model)
    if result["status"] == "error":
        code = result.get("code")
        if code == 429:
            print("RESULT: 429 Too Many Requests — rate limit is ACTIVE")
            print("Wait ~60 seconds before retrying")
            sys.exit(1)
        else:
            print(f"RESULT: HTTP {code} — {result['error']}")
    else:
        print("RESULT: OK — no rate limit active")
    print()

    # Full test with Chinese prompt (same format graphrag uses)
    print("=== Full API Test (OpenAI SDK format) ===")
    result = test_nim_completion(client, model, "請用一句話自我介紹", max_tokens=100)
    if result["status"] == "success":
        print(f"Model: {result['model']}")
        print(f"Response: {result['response']}")
        print(f"Usage: {result['usage']}")
    else:
        print(f"ERROR [{result.get('code')}]: {result['error']}")
        if result.get("code") == 429:
            print("Rate limit hit — wait and retry")


if __name__ == "__main__":
    main()
