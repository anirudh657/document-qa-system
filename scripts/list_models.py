"""Print the models this API key can reach, and what each one costs to call.

Why this exists: the project was calling `gemini-flash-latest`, an alias that follows
whatever Google currently points it at. That turned out to carry a 20-request daily
free-tier cap — enough to make a 34-question cache-warming pass impossible, and enough
to invalidate any recorded baseline, since the model underneath a `-latest` alias can
change between two runs that claim to compare the same thing.

Uses the models.list endpoint, which returns metadata and does not consume the
generateContent quota. Safe to run when that quota is exhausted.

Run from the repo root:  python scripts/list_models.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"


def api_key() -> str:
    """Read the key from the environment, falling back to .env.

    Read, never printed. The whole point of keeping it in a gitignored .env is that it
    does not end up in terminal output that gets pasted somewhere.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    env = Path(".env")
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY":
                return value.strip().strip("'\"")

    sys.exit("No GEMINI_API_KEY found in the environment or .env. Run from the repo root.")


def fetch() -> list[dict[str, Any]]:
    request = urllib.request.Request(ENDPOINT, headers={"x-goog-api-key": api_key()})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            models: list[dict[str, Any]] = json.load(response).get("models", [])
            return models
    except urllib.error.HTTPError as error:
        sys.exit(f"HTTP {error.code} listing models: {error.read().decode()[:300]}")
    except urllib.error.URLError as error:
        sys.exit(f"Could not reach the API: {error.reason}")


def main() -> None:
    models = [
        model
        for model in fetch()
        if "generateContent" in (model.get("supportedGenerationMethods") or [])
    ]
    if not models:
        sys.exit("The key reached the API but no model supports generateContent.")

    # Aliases first: they are the ones whose identity is not fixed, so they are what
    # needs looking at before anything records a number against them.
    names = [model["name"].removeprefix("models/") for model in models]
    aliases = sorted(name for name in names if "latest" in name)
    pinned = sorted(name for name in names if "latest" not in name)

    print(f"{len(models)} model(s) support generateContent with this key.\n")

    print("ALIASES — avoid for anything measured; the model underneath can change:")
    for name in aliases:
        print(f"  {name}")

    print("\nPINNED versions — safe to record a baseline against:")
    for name in pinned:
        print(f"  {name}")

    print(
        "\nThe daily quota is per project and per model, and the API does not report it "
        "here.\nRead the live numbers at https://aistudio.google.com/rate-limit, then set\n"
        "LLM_MODEL in .env to a pinned name above."
    )


if __name__ == "__main__":
    main()