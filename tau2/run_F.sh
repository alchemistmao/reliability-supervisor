#!/bin/bash
# Experiment F: tau2-bench airline (public benchmark), clean and stale-data faults,
# vanilla vs supervisor vs LLM critic. Agent and user simulator: Claude Haiku 5.5.
set -e
cd "$(dirname "$0")"
mkdir -p logs_F
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [ ! -d tau2-bench ]; then
  git clone -q https://github.com/sierra-research/tau2-bench.git
  (cd tau2-bench && git checkout -q c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4)
fi
[ -d .venv-tau2 ] || uv venv -q .venv-tau2 --python 3.12
source .venv-tau2/bin/activate
uv pip install -q -e tau2-bench
export TAU2_DATA_DIR="$PWD/tau2-bench/data"

echo "[F] smoke test (no API calls)..."
ANTHROPIC_API_KEY=mock python run_tau2.py --mode supervisor --faults stale --tasks 0 --trials 1 --mock > logs_F/smoke.log 2>&1 \
  && echo "[F] smoke test OK" || { echo "[F] smoke test FAILED - send me logs_F/smoke.log"; exit 1; }

if [ -z "$ANTHROPIC_API_KEY" ]; then read -s -p "Cole a chave da Anthropic: " K; echo; export ANTHROPIC_API_KEY="$K"; fi

AGENT=${AGENT:-anthropic/claude-haiku-5-5}
echo "[F] live check: 1 task, 1 trial..."
python run_tau2.py --mode supervisor --faults none --agent $AGENT --tasks 1 --trials 1 --suffix _check > logs_F/check.log 2>&1 || true
grep -q "\[F\] sims=1" logs_F/check.log && ! grep "\[F\] terminations" logs_F/check.log | grep -qi "error" && echo "[F] live check OK: $(grep '\[F\] \(sims\|termin\)' logs_F/check.log | tr '\n' ' ')" || { echo "[F] live check FAILED - send me logs_F/check.log"; exit 1; }
TRIALS=${TRIALS:-3}
echo "[F] running 6 conditions x 50 tasks x $TRIALS trials with $AGENT (about 1-2 h)..."
for F in none stale; do
  for M in vanilla supervisor critic; do
    python run_tau2.py --mode $M --faults $F --agent $AGENT --trials $TRIALS --concurrency 4 > logs_F/${F}_${M}.log 2>&1 &
  done
done
wait
SHORT=${AGENT##*/}
python analyze_tau2.py tau2-bench/data/simulations/F_none_*_$SHORT tau2-bench/data/simulations/F_stale_*_$SHORT > tau2_summary_$SHORT.txt
grep -h "\[F\] done" logs_F/*.log
grep PAIRED tau2_summary_$SHORT.txt
echo "[F] finished: send me tau2_summary_$SHORT.txt"
