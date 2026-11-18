#!/bin/bash
# Experiment G: hidden consequence (G4) with a real autonomous agent (Claude Haiku 5.5), 3 scenarios x 10 runs.
# Budget cap: USD 3 (stops by itself). Expected cost: well under USD 1.
cd "$(dirname "$0")"
if [ -z "$ANTHROPIC_API_KEY" ]; then read -s -p "Cole a chave da Anthropic: " K; echo; export ANTHROPIC_API_KEY="$K"; fi
python3 run_hidden.py --model claude-haiku-5-5 --runs 10 --budget 3 2>&1 | tail -45
