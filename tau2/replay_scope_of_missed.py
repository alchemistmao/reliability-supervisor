import json, sys, glob, collections
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TAU2 = os.environ.get('TAU2_BENCH', 'tau2-bench')
PUB = os.environ.get('PUBLISHED_TRAJ', 'published_trajectories')
from tau2_supervisor import check, Block, WRITE_TOOLS, NOW
from tau2.domains.airline.environment import get_environment, get_tasks
from datetime import datetime
tasks = {t.id: t for t in get_tasks()}
old={'claude-3-7-sonnet-20250219','gpt-4.1-2025-04-14','gpt-4.1-mini-2025-04-14','o4-mini-2025-04-16'}
C = collections.Counter()
files = sorted(glob.glob(os.path.join(TAU2,'data/tau2/results/final/*_airline_*.json'))+glob.glob(os.path.join(PUB,'*.json')))
for f in files:
    model = f.split('/')[-1].split('_airline')[0].replace('.json',''); grp = 'old' if model in old else 'new'
    for s in json.load(open(f))['simulations']:
        t = tasks[s['task_id']]; env = get_environment()
        acts = t.evaluation_criteria.actions or []
        gold = {(a.name, json.dumps(a.arguments, sort_keys=True)) for a in acts if a.name in WRITE_TOOLS}
        ok = (s.get('reward_info') or {}).get('reward', 0) >= 1
        for m in s['messages']:
            for tc in (m.get('tool_calls') or []):
                n, a = tc['name'], tc['arguments']
                if n in WRITE_TOOLS and not ok and (n, json.dumps(a, sort_keys=True)) not in gold:
                    try: check(env.tools.db, n, a); blocked = False
                    except Block: blocked = True
                    except Exception: blocked = False
                    if not blocked:
                        cat = 'intent' if n in ('update_reservation_flights','book_reservation','update_reservation_baggages') else n
                        if n == 'cancel_reservation':
                            r = env.tools.db.reservations.get(a.get('reservation_id'))
                            if r is None: cat = 'cancel_nonexistent'
                            else:
                                created = datetime.fromisoformat(r.created_at)
                                if (NOW - created).total_seconds() <= 86400: cat = 'cancel_ok_24h'
                                elif r.cabin == 'business': cat = 'cancel_ok_business'
                                elif r.insurance == 'yes': cat = 'cancel_insured_reason_unverifiable'
                                else: cat = 'cancel_ok_other(airline-cancelled flight)'
                        C[(grp, cat)] += 1
                try: env.tools.use_tool(n, **a)
                except Exception: pass
for k in sorted(C): print(k, C[k])
