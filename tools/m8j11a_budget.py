"""Process-safe append-only reservation ledger; failed forwards consume budget."""
import fcntl
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def reserve(path, run_id, condition, index, maximum=243, primary_limit=None):
    path=Path(path)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.seek(0); rows=[json.loads(l) for l in f if l.strip()]
        if len(rows)>=maximum:
            raise RuntimeError('ForwardBudgetExceeded')
        if primary_limit is not None and sum(r['condition']==condition for r in rows)>=primary_limit:
            raise RuntimeError('ConditionBudgetExceeded')
        row=dict(reservation=len(rows)+1,run_id=run_id,condition=condition,index=index,
                 host_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                 semantics='Reserved immediately before full simple_test; failures remain charged')
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        return row
