#!/bin/bash
# Experiment G with Claude. check = 10% (3 of 30 sessions per model, then stop); rest = complete the remaining sessions.
cd "$(dirname "$0")"
command -v deactivate >/dev/null 2>&1 && deactivate
PY=/usr/bin/python3; [ -x .venv-tau2/bin/python ] && PY=.venv-tau2/bin/python
export PATH="$HOME/.local/bin:$PATH"; $PY -c "import anthropic" 2>/dev/null || uv pip install -q --python $PY anthropic
if [ -z "$ANTHROPIC_API_KEY" ]; then read -s -p "Cole a chave da Anthropic: " K; echo; export ANTHROPIC_API_KEY="$K"; fi
N=3; [ "$1" = "rest" ] && N=0
for M in claude-haiku-5-5 claude-sonnet-5-5; do
  echo "===== $M ====="
  $PY run_hidden.py --model $M --runs 10 --budget 3 --max-sessions $N > logs_G_$M.log 2>&1
  grep -E "API OK|PARTIAL|sessions_with_unsafe|n_unsafe_actions|full_supervisor|trusting_supervisor|\"critic\"|Done" logs_G_$M.log \
    || { echo "[G] failed - last lines:"; tail -8 logs_G_$M.log; }
done
