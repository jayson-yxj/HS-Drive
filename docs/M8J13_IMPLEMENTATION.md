# J13 opt-in tools and bounded sensor candidate

J13-A remains Partial. No live model actuation loop is delivered or authorized. The native SparseDriveV2 PID and protocol remain unchanged. `tools.m8j13_supervisor --phase campaign` rejects the incomplete gate and unimplemented driving path.

The clean public branch exports code from the privately archived implementation without merging that history. Detailed images, callback/snapshot identities, source-to-export commit mapping and lifecycle logs stay private. See [the consolidated sensor report](../reports/m8j13_sensor_sync_followup.md) for historical failures, official fix, candidate results and limits.

The 0.9.16 candidate uses an existing isolated server/client. `configs/m8j13_candidate.json` preserves two-start/900-tick limits and points to a private append-only ledger; both starts are consumed. Do not change its output root or budget to repeat experiments without a new bounded task. Historical 0.9.15 diagnostics retain their independent three-start budget.

```bash
# Tests only; no server, model loading or driving.
/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/.venv-gpu/bin/python -m pytest -q tests/m8j12 tests/m8j13
```

Candidate commands, run requests, actual environment paths/hashes and raw evidence are preserved privately. Sensor callbacks copy native metadata and BGRA bytes at entry; hashing and disk writes occur elsewhere. Rendered markers are diagnostic-only and never enter driving data. `DeadlineAndProtection` records rejection and a numeric stop output without applying it to a vehicle. This does not substitute for the remaining moving safety tests.
