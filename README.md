# Reliability Supervisor — artifact

Artifact for the paper *Decoupling Reliability from Intelligence: A Four-Gate Supervisor for Enterprise AI Agents in Long, Complex Sessions* (AGENT'27 @ ICSE 2027).

The agent proposes an Intent Packet; a separate supervisor holds execution authority and re-derives every safety-relevant fact from systems of record through four gates (G1 Grounding, G2 Rules, G3 Feasibility, G4 Consequence).

## Layout

| Folder | Content | Paper |
|---|---|---|
| `supervisor/` | Reference implementation of the four gates, the ground-truth oracle and the simulated enterprise | Sec. IV, VI |
| `benchmark/` | Seeded benchmark generator (12 templates, 12,000 packets) | Sec. VI |
| `experiments/` | Real-agent experiments A–E and G (one-turn traps, long sessions with compaction, implicit rules, autonomous sessions, hidden consequence) | Sec. VII |
| `tau2/` | Gates for the τ²-bench airline domain, fault injection, runner and analysis (experiment F); replay of third-party trajectories (experiment H) | Sec. VII |
| `results/` | Summaries per experiment, raw model outputs (`raw/`) and τ²-bench simulations (`tau2/`) | — |

## Reproduce

Python 3.12. API keys are read from the environment (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`); none are stored here.

- Seeded benchmark: `python benchmark/benchmark.py`
- Real-agent experiments: see the scripts in `experiments/` (`--max-sessions` runs a partial check first).
- Experiment F: install [τ²-bench](https://github.com/sierra-research/tau2-bench), then `bash tau2/run_F.sh` and `python tau2/analyze_tau2.py`.
- Experiment H (no API calls): `python tau2/replay_third_party.py` replays the trajectories published in the τ²-bench repository through the frozen gates.

## License

MIT for code; result files CC BY 4.0. The Orca AI Incident Archive used in Study 1 is CC BY 4.0 by its contributors.
