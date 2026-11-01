# J13 opt-in tools and bounded sensor candidate

J13-A remains Partial. No live model actuation loop is delivered or authorized. The native SparseDriveV2 PID and protocol remain unchanged. `tools.m8j13_supervisor --phase campaign` rejects the incomplete gate and unimplemented driving path.

The clean public branch exports code from the privately archived implementation without merging that history. Detailed images, callback/snapshot identities, source-to-export commit mapping and lifecycle logs stay private. See [the consolidated sensor report](../reports/m8j13_sensor_sync_followup.md) for historical failures, official fix, candidate results and limits.

The 0.9.16 candidate uses an existing isolated server/client. `configs/m8j13_candidate.json` preserves two-start/900-tick limits and points to a private append-only ledger; both starts are consumed. Do not change its output root or budget to repeat experiments without a new bounded task. Historical 0.9.15 diagnostics retain their independent three-start budget.

```bash
# Tests only; no server, model loading or driving.
/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/.venv-gpu/bin/python -m pytest -q tests/m8j12 tests/m8j13
```

Candidate commands, run requests, actual environment paths/hashes and raw evidence are preserved privately. Sensor callbacks copy native metadata and BGRA bytes at entry; hashing and disk writes occur elsewhere. Rendered markers are diagnostic-only and never enter driving data. `DeadlineAndProtection` records rejection and a numeric stop output without applying it to a vehicle. This does not substitute for the remaining moving safety tests.

The subsequent explicit 0.9.16 compatibility task is recorded in the same report and `reports/m8j13_sources/compatibility/`. Its two starts are also consumed. `configs/m8j13_compatibility.json` and `tools.m8j13_compatibility` keep a separate private ledger: no repeat is authorized by unused tick or forward capacity. GPS route import is incompatible with the original inverse projection, so shadow inference did not run. One physical test-control round moved the vehicle but did not reach its fixed fault-injection speed within 15 seconds; no moving fault stop was demonstrated. The tool still needs a bounded fallback brake/observation interval when that prerequisite fails. Do not treat it as a completed safety executor.

`tools.m8j13_shadow_service` is a dry-run process boundary with no actuator/control response IPC; it retains the original frozen model environment. `tools.m8j13_stop_planner_stub` provides synthetic test heartbeats, not production planner health or owner-actuation acknowledgement. None of these tools enables model driving or changes the default CARLA version. Navigation, physical stop, deadline and code-delivery statuses remain separate.
