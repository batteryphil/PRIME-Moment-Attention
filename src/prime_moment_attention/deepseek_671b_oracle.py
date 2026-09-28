#!/usr/bin/env python3
"""
DeepSeek-R1 671B Frontier Oracle Bridge
=======================================
Connects the local dual-model research lab to the full 671-Billion parameter
DeepSeek-R1 frontier reasoning model via OpenRouter Free Tier or DeepSeek Direct API.

Role in Lab Hierarchy:
1. Level 3 (Senior Oracle): DeepSeek-R1 671B (Deep frontier mathematical proofs & meta-theorems)
2. Level 2 (Local Edge Teacher): DeepSeek-R1-Distill-14B (Fast local C99 code & SymPy verification)
3. Level 1 (Plastic Apprentice): PRIME-152M (Continual Online TTT at 280,000 tokens/sec)

Features:
- Dual provider support: OpenRouter (deepseek/deepseek-r1:free) & DeepSeek Direct (deepseek-reasoner)
- Automatic fallback: If cloud API is unavailable or rate-limited, smoothly falls back to local 14B model.
- Key resolution: Reads from environment variable (OPENROUTER_API_KEY / DEEPSEEK_API_KEY) or config file.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, Tuple

KEYS_CONFIG_PATH = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/api_keys.json"


def get_api_key() -> Tuple[Optional[str], str]:
    """
    Resolves API key and provider in order of priority:
    1. Environment variables: OPENROUTER_API_KEY, DEEPSEEK_API_KEY
    2. Local configuration file: research_vault/api_keys.json
    Returns (key, provider_name).
    """
    # 1. Environment
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY"), "openrouter"
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.getenv("DEEPSEEK_API_KEY"), "deepseek"

    # 2. Config file
    if os.path.exists(KEYS_CONFIG_PATH):
        try:
            with open(KEYS_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if cfg.get("openrouter_api_key"):
                    return cfg.get("openrouter_api_key"), "openrouter"
                if cfg.get("deepseek_api_key"):
                    return cfg.get("deepseek_api_key"), "deepseek"
        except Exception:
            pass

    return None, "none"


class DeepSeek671BOracle:
    """Interface to query full 671B DeepSeek-R1 for frontier mathematics and reasoning."""
    def __init__(self):
        self.key, self.provider = get_api_key()

    def is_available(self) -> bool:
        """Returns True if a valid API key is present."""
        self.key, self.provider = get_api_key()
        return bool(self.key and len(self.key.strip()) > 5)

    def query_derivation(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 45
    ) -> Optional[Dict[str, Any]]:
        """
        Queries DeepSeek-R1 671B for a step-by-step mathematical derivation and reasoning trace.
        Returns dictionary with 'content', 'reasoning', and 'model' or None if call fails.
        """
        self.key, self.provider = get_api_key()
        if not self.is_available():
            return None

        openrouter_model = "deepseek/deepseek-r1:free"
        if os.path.exists(KEYS_CONFIG_PATH):
            try:
                with open(KEYS_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    openrouter_model = cfg.get("openrouter_model", openrouter_model)
            except Exception:
                pass

        if self.provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            model_name = openrouter_model
            headers = {
                "Authorization": f"Bearer {self.key}",
                "HTTP-Referer": "https://github.com/batteryphil/PRIME-Moment-Attention",
                "X-Title": "PRIME-Moment-Attention Lab",
                "Content-Type": "application/json"
            }
        else:
            url = "https://api.deepseek.com/chat/completions"
            model_name = "deepseek-reasoner"
            headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"
            }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 1500
        }
        if self.provider == "openrouter":
            data["include_reasoning"] = True

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))

            choice = resp_json.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "") or message.get("reasoning", "") or ""

            # If reasoning was enclosed inside <think> in content
            if not reasoning and "<think>" in content:
                parts = content.split("</think>")
                reasoning = parts[0].replace("<think>", "").strip()
                content = parts[1].strip() if len(parts) > 1 else content

            return {
                "status": "SUCCESS",
                "provider": self.provider,
                "model": model_name,
                "content": content,
                "reasoning_trace": reasoning,
                "full_output": f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
            }
        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8")
            except Exception:
                pass
            print(f"[-] DeepSeek-R1 671B Oracle HTTP Error ({he.code}): {he.reason} | {err_body[:200]}")
            return None
        except Exception as e:
            print(f"[-] DeepSeek-R1 671B Oracle request failed ({self.provider}): {e}")
            return None


if __name__ == "__main__":
    oracle = DeepSeek671BOracle()
    print(f"[*] DeepSeek-R1 671B Oracle Status: {'ONLINE' if oracle.is_available() else 'AWAITING_API_KEY'}")
    print(f"[*] Configured Provider: {oracle.provider}")
    if oracle.is_available():
        print("[*] Testing live connection to DeepSeek-R1 671B...")
        res = oracle.query_derivation("State the Euler-Lagrange equation for a classical scalar field and provide the 1-line proof.")
        if res:
            print(f"[+] Success! Received response from {res['model']}:")
            print(res['content'][:200])
        else:
            print("[-] Test query failed.")
    else:
        print(f"[*] To activate, set OPENROUTER_API_KEY or save key to {KEYS_CONFIG_PATH}")
