#!/bin/bash
# Experiment G on the Gemini free tier (no cost). ~300 calls, ~35 min (rate-limited). Results are cached:
# if the daily quota ends, just run the same command again tomorrow and it continues.
cd "$(dirname "$0")"
python3 run_hidden.py --provider gemini --runs 10 2>&1 | tail -45
