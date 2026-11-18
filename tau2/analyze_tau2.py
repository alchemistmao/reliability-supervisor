"""Analyse Experiment F results (tau2-bench airline). Usage: python analyze_tau2.py <sim_dir> ..."""
import json, random, sys, re
from collections import defaultdict, Counter
from pathlib import Path
from math import comb

WRITES = {"book_reservation", "cancel_reservation", "update_reservation_flights",
          "update_reservation_baggages", "update_reservation_passengers", "send_certificate"}


def norm(x):
    return json.dumps(x, sort_keys=True)


def load(d):
    d = Path(d)
    r = json.loads((d / "results.json").read_text())
    gold = {t["id"]: [(a["name"], norm(a.get("arguments", {}))) for a in t["evaluation_criteria"]["actions"]]
            for t in r["tasks"]}
    return r, gold


def walk(sim):
    """Yield (name, args, content, error) for each assistant tool call."""
    msgs = sim["messages"]
    calls = {}
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                calls[tc["id"]] = tc
        if m["role"] == "tool":
            tc = calls.get(m.get("id"))
            if tc:
                yield tc["name"], tc.get("arguments", {}), m.get("content") or "", bool(m.get("error"))


def pass_hat_k(n, c, k):
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def summarize(d):
    r, gold = load(d)
    sims = r["simulations"]
    by_task = defaultdict(list)
    out = Counter()
    gates = Counter(); false_blocks = 0; blocks = 0
    wrong_writes = 0; silent = 0; fails = 0; cost = 0.0
    for s in sims:
        rew = s["reward_info"]["reward"] if s.get("reward_info") else 0.0
        by_task[s["task_id"]].append(rew)
        cost += (s.get("agent_cost") or 0) + (s.get("user_cost") or 0)
        g = gold.get(s["task_id"], [])
        gw = {(n, a) for n, a in g if n in WRITES}
        any_err = False; fired = False
        for name, a, content, err in walk(s):
            if "SUPERVISOR BLOCK" in content or "CRITIC" in content[:40]:
                fired = True; blocks += 1
                m = re.search(r"\[([^\]]+)\]", content)
                gates[m.group(1) if m else "?"] += 1
                if (name, norm(a)) in gw:
                    false_blocks += 1
                continue
            if err:
                any_err = True
            if name in WRITES and not err and (name, norm(a)) not in gw:
                wrong_writes += 1
        out["fired_sims"] += fired
        if rew < 1:
            fails += 1
            db = (s["reward_info"] or {}).get("db_check") or {}
            if not any_err and db.get("db_match") is False:
                silent += 1
    n = len(sims)
    k_max = min(len(v) for v in by_task.values())
    passk = {k: sum(pass_hat_k(len(v), int(sum(v)), k) for v in by_task.values()) / len(by_task)
             for k in range(1, k_max + 1)}
    return {
        "dir": str(d), "sims": n, "tasks": len(by_task), "pass1": sum(sum(v) for v in by_task.values()) / n,
        "pass_k": passk, "failures": fails, "silent_wrong_state_failures": silent,
        "executed_writes_not_in_gold": wrong_writes, "blocks": blocks, "false_blocks_on_gold_action": false_blocks,
        "blocks_by_gate": dict(gates), "sims_with_block": out["fired_sims"], "cost_usd": round(cost, 2),
        "_by_task": {t: sum(v) / len(v) for t, v in by_task.items()},
    }


def paired(a, b, iters=20000, seed=7):
    ts = sorted(set(a["_by_task"]) & set(b["_by_task"]))
    diffs = [b["_by_task"][t] - a["_by_task"][t] for t in ts]
    rnd = random.Random(seed)
    boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    mean = sum(diffs) / len(diffs)
    p = sum(1 for x in boots if x <= 0) / iters
    return {"delta_pp": round(100 * mean, 1), "ci95_pp": [round(100 * boots[int(.025 * iters)], 1),
            round(100 * boots[int(.975 * iters)], 1)], "p_one_sided": p, "tasks": len(ts)}


if __name__ == "__main__":
    res = [summarize(d) for d in sys.argv[1:]]
    for x in res:
        print(json.dumps({k: v for k, v in x.items() if k != "_by_task"}, indent=1))
    def key(x):
        p = Path(x["dir"]).name.split("_", 3)  # F, faults, mode, agent
        return p[1], p[2], p[3]
    for b in res:
        fb, mb, ab = key(b)
        if mb != "vanilla":
            continue
        for x in res:
            fx, mx, ax = key(x)
            if x is not b and fx == fb and ax == ab:
                print("PAIRED", Path(b["dir"]).name, "->", Path(x["dir"]).name, json.dumps(paired(b, x)))
