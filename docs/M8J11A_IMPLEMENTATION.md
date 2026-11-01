# Frozen StreamPETR temporal sensitivity

One branch: `feat/streampetr-temporal-sensitivity`, based on integrated main `1d7472dea3941132a0e5946dc7f80feddc26b4f6`. Default model, official private runtime source, historical runner, decoder and input loader are unchanged.

`datasets/m8j11a_temporal.py` is an opt-in runner adapter, default `normal`. It observes the official head's `pre_update_memory` and calls the original implementation unchanged. `cold` resets every observation; `midpoint` resets at the declared scene midpoint. Reset sets detector `prev_scene_token=None` and calls official `head.reset_memory()`; real metadata is untouched. `simple_test_pts` then initializes `prev_exists=0` via its original scene-entry path. `test_flag` is false in this direct simple_test path; `single_test` is a configuration field, not a manipulated scene identity.

Five memory tensors are cleared together. Cold pseudo-reference initialization is legal. The audit tracks prior observation blocks (256 proposals per frame; up to1024 read), checking scene, causal timestamps and actual age values, then records current writes. Observed entries and padded entries are distinct. Official relative age is current absolute seconds plus stored negative source seconds. The read-only audit does not modify features, poses, outputs or timestamps.

`tools/m8j11a_gpu.py` reuses J11's exact `load`, input allowlist, GT/raw-payload guard, configuration, seed, precision, attention choice, strict state load and decoder checks. No optimizer or training hooks are constructed, although the frozen source config retains unused training sections. Actual floating compute dtypes are checked separately from inherited `fp16_enabled` attribute names. Every frame checks model state against an independently cloned initial snapshot. Normal code is not patched; wrappers are installed only for this experiment instance.

`tools/m8j11a_budget.py` uses a process lock and fsync append before `simple_test`. Reservations consume the 243 cap even if a call fails. Per-condition primary caps81/81/41 prevent accidental primary restarts; extra diagnostics would require a separately declared run/condition using the same ledger and remaining40 total allowance. None were needed. No warmup or hidden loader queue exists: each invocation reads one explicit cached six-camera input. T3 never opens skipped inputs for a forward or writes them into memory.

## Frozen files and selection

`configs/m8j11a_protocol.json` was created before any forward and contains all81 ordered identifiers, scene-local indices, resets, T3 selection, real intervals, T0 cache SHA and77 fixed-asset hashes. Source/weight/config/backend identity is rechecked before each run and at evaluation. T0 is only J11 StreamPETR, never A/B/C. The official private source commit is95f6470; optional import patch is the same J11 patch. No new network access, package install or asset download.

Host execution timestamps are retained without correction. Client date and the recorded execution host UTC date are both2026-11-01. Exact timestamps are in the phase records.

## Reproduction commands

Commands below describe the recorded execution and use exclusive-create outputs. Existing outputs will fail rather than overwrite. For an independently authorized repeat, use a new private root/run_id in a copied protocol and preserve the original schedule, inputs, hashes and numeric settings. Do not reset the existing budget ledger to obtain more forwards.

```bash
# Metadata/hash-only freeze; creates config and private root, refuses existing outputs:
.venv/bin/python -m tools.m8j11a_prepare
# Relevant synthetic tests; no full model forward:
python3 -m tools.m8j11a_phase final_contract_tests env CUDA_VISIBLE_DEVICES= \
  bash tools/m8j11_env.sh -m pytest -q -ra --confcutdir=tests/m8j11a \
  tests/m8j11a tests/m8j11
# Each is a new process; no residual cross-experiment memory:
python3 -m tools.m8j11a_phase T2_run bash tools/m8j11_env.sh -m tools.m8j11a_gpu --condition T2
python3 -m tools.m8j11a_phase T1_run bash tools/m8j11_env.sh -m tools.m8j11a_gpu --condition T1
python3 -m tools.m8j11a_phase T3_run bash tools/m8j11_env.sh -m tools.m8j11a_gpu --condition T3
# CPU evaluation/export in unchanged baseline environment:
python3 -m tools.m8j11a_phase evaluation bash tools/m8j11_baseline_env.sh -m tools.m8j11a_evaluate
python3 -m tools.m8j11a_phase visuals bash tools/m8j11_baseline_env.sh -m tools.m8j11a_visualize
python3 -m tools.m8j11a_phase report bash tools/m8j11_baseline_env.sh -m tools.m8j11a_report
```

Phase supervisor has600s limit, isolated child process group, append-only private stdout and public command/exit/source hashes. Model runs at6e36963; aggregate evaluation at2ac39c2. Later report/test changes do not retroactively alter executed inference. Official summary/details remain byte-preserved under private `evaluation/T1/official` and `evaluation/T2/official`; hashes and transformed-export checks are public. T0 exact official metrics are reused, project T0 is recomputed from its cache and compared with the historical metrics. T3 uses exactly its retained41frames and no official split changes.

GT is joined only in the offline evaluator; only an explicit boxes/scores/labels dict reaches the exporter. Matching/threshold/range rules and normalization are inherited from J9/J8/J7. No PR threshold search added. `paired` fields A/B mean T0/intervention, not PETR-A/B. Full frame × object pair records and memory histories stay private, while per-scene/per-class aggregate counts and reset-local metrics are public. No independent object count inferred from repeated frame observations.

## Limits and branch completion

Intervention correctness, experiment completion and model sensitivity are distinct states. Performance degradation is a result, not a merge blocker. Default-off behavior is tested against an unwrapped Mock and the40real normal pre-reset frames match T0 exactly. Existing CUDA attention unit test is skipped in the CPU test command; this is not represented as a rerun pass. Actual203GPU forwards are separate evidence.

Two reused mini_val development scenes do not support universal safety/generalization or stable recovery guarantees. T1 is temporal weights without history, not a trained single-frame baseline. T3 changes interval and effective temporal span together. Instrumented cached-input timing includes synchronous memory checks; not comparable to raw-image J11 timing and not realtime qualification.

Merge and branch cleanup require the user's single final confirmation, using the final exact tip after the report commit. No default weight promotion, tag changes, new training or SparseDriveV2 implementation is bundled into this branch.
