import json, sys, glob, collections
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TAU2 = os.environ.get('TAU2_BENCH', 'tau2-bench')
PUB = os.environ.get('PUBLISHED_TRAJ', 'published_trajectories')
from tau2_supervisor import check, Block, WRITE_TOOLS
from tau2.domains.airline.environment import get_environment, get_tasks

tasks = {t.id: t for t in get_tasks()}
def gold_writes(t):
    acts = (t.evaluation_criteria.actions or []) if t.evaluation_criteria else []
    return [(a.name, json.dumps(a.arguments, sort_keys=True)) for a in acts if a.name in WRITE_TOOLS]

out = {}
for f in sorted(glob.glob(os.path.join(TAU2,'data/tau2/results/final/*_airline_*.json'))+glob.glob(os.path.join(PUB,'*.json'))):
    model = f.split('/')[-1].split('_airline')[0].replace('.json','')
    d = json.load(open(f))
    st = collections.Counter(); ex = []
    for s in d['simulations']:
        t = tasks.get(s['task_id']);
        if t is None: st['no_task'] += 1; continue
        env = get_environment()
        if t.initial_state and t.initial_state.initialization_data: pass
        gold = set(gold_writes(t)); ok = (s.get('reward_info') or {}).get('reward', 0) >= 1
        st['sims'] += 1; st['solved'] += ok
        for m in s['messages']:
            for tc in (m.get('tool_calls') or []):
                name, args = tc['name'], tc['arguments']
                if name in WRITE_TOOLS:
                    key = (name, json.dumps(args, sort_keys=True)); in_gold = key in gold
                    st['writes'] += 1; st['gold_writes'] += in_gold
                    try:
                        check(env.tools.db, name, args); blocked = None
                    except Block as b: blocked = b.gate; reason=b.reason
                    except Exception: blocked = None
                    if blocked:
                        st['blocks'] += 1; st['block_'+blocked] += 1
                        if in_gold: st['false_blocks'] += 1
                        elif not ok: st['blocks_in_failed_sims'] += 1
                        ex.append((s['task_id'], name, blocked, in_gold, ok, reason[:90]))
                    if not in_gold and not ok: st['wrong_writes_failed'] += 1; st['wrong_caught'] += bool(blocked); st['unc_'+name] += (not blocked)
                try: env.tools.use_tool(name, **args)
                except Exception: pass
    out[model] = (dict(st), ex)
    print(model, dict(st)); 
json.dump(out, open('replay_H_13models.json','w'), indent=1, default=str)
