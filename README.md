# Safe Qwen OpenRouter Gateway

A Python CLI batch-processing gateway that calls a Qwen model through OpenRouter only after input safety checks. It also applies output safety checks before returning model text.

## Safety gates

- Pre-filter blocks the required jailbreak phrase and US SSN-shaped PII.
- Post-filter blocks `[CONFIDENTIAL]` text and `sudo rm -rf`.
- Every final decision is recorded in `audit.jsonl`.
- Final batch responses are written to `results.json`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` before running the pipeline.

## Planned execution

```powershell
python gateway.py
```

## Architecture

```text
redteam.json
  -> gateway.py
  -> pre-filter
  -> OpenRouter/Qwen (only when allowed)
  -> post-filter
  -> results.json and audit.jsonl
```