# M8-J11 frozen StreamPETR candidate

Base `68ee3e8`; independent branch `feat/m8j11-streampetr-r50-frozen-candidate`.
Protocol: `configs/m8j11_protocol.json`. Exactly one official R50 ImageNet 90-epoch checkpoint. No optimizer, loss/backward, CARLA or dataset download.

## Isolation and provenance

Runtime `/root/gpufree-data/hs-drive-experiments/m8j11-streampetr` is an independent copy of the environment, with its own MMDet/MMDet3D/MMCV packages. Baseline environment is unchanged. Official StreamPETR commit `95f64702306ccdb7a78889578b2a55b5deb35b2a`; MMDet3D rc6 commit `47285b3f1e9dba358e98fcd12e523cfd0769c876`. Direct official clone explicitly bypasses the host's preexisting third-party Git URL rewrite. The earlier mirrored copy is quarantined and unused. Initial direct download failures are retained. After the user reported restoring VPN, one bounded official HTTP206 resume completed.

Weight SHA256 `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`, size154852451. Official expected digest is unavailable; the local hash fixes identity, not independent authenticity. Official URL and GitHub asset metadata are retained. Restricted `torch.load(weights_only=True)` runs in a network-isolated child environment. Private assets are outside Git.

The official optional Flash path is replaced by the same repository's `PETRMultiheadAttention(fp16=False)` implementation, preserving QKV/output weight keys. A minimal private source patch skips optional Flash/EVA imports; no geometry, decoder or memory rule changes. The original official tree is preserved. Strict state loading and observed dtypes must pass before FP32 claims. This is a backend difference, not proof of numerical equivalence to Flash attention.

## Input and output contracts

Official camera order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT. Official fixed resize0.44, crop(0,140,704,396), RGB mean/std, final six×3×256×704. Pose references LIDAR_TOP at its own timestamp solely as a coordinate system; no point-cloud payload. Per-camera real timestamp/calibration/ego pose determines camera-to-reference transform. Sample/pose/calibration tables are read without annotation joins. GT never enters memory or perception.

Official MMDet3D rc6 boxes use bottomXYZ, fullLWH, CCW yaw. The legacy J9 exporter expects bottomXYZ, WLH, SECOND yaw. `to_legacy_boxes` swaps dimensions3/4 and sets yaw=-yaw_native-pi/2; positions/height/velocity remain unchanged. Export tests independently check the resulting rotation, global position and velocity. Native output remains private for provenance. Original Top300 native truncation and range happen before public score0.2/strictXY51.2 filter; no removed candidates are reconstructed. Memory writes remain inside official head before public filtering.

## Execution

```bash
# After private environment/assets are provisioned:
PYTHONPATH=. /root/gpufree-data/hs-drive-experiments/m8j11-streampetr/env/bin/python -m pytest -q --confcutdir=tests/m8j11 tests/m8j11
bash tools/m8j11_env.sh -m tools.m8j11_prepare
bash tools/m8j11_env.sh -m tools.m8j11_gpu --phase probe
bash tools/m8j11_baseline_env.sh -m tools.m8j11_evaluate --phase probe
# Only after short export passes:
bash tools/m8j11_env.sh -m tools.m8j11_gpu --phase full
bash tools/m8j11_baseline_env.sh -m tools.m8j11_evaluate --phase full
bash tools/m8j11_baseline_env.sh -m tools.m8j11_a_timing
```

All output directories are exclusive-create; reruns require a new private run directory/protocol, never overwrite old runs. Wrappers use `unshare --net`. The initial default workspace Python lacks pyquaternion and the root pytest conftest uses Python3.10 syntax; isolated tests use their own confcutdir in the existing Python3.8 candidate environment. Those initial collection failures are retained.

Short test: first3frames scene1 then first2scene2, repeat from reset, then fresh scene2 alone. Formal run starts a new process and scene memory; all81frames including cold frames retained. `state_dict` parameters/buffers must remain exact; temporal memory is separately logged. Timestamps are FP64 while image/attention computations must be FP32. No AMP wrapper/scaler/hook constructed.

Speed boundary explicitly covers cached-input read, transfer, compute and CPU predictions; it is not camera capture/JPEG preprocessing, control cycle or safety timing. GPU events and wallclock are separate. Same boundary applies to A. Official/project metrics reuse J9/J8; GT3393 versus3559 stay separate. The set is previously used development data, not a new blind test. Public pretrained provenance is not an independently verified per-sample training ledger.

## Actual completion and final timing

Short sequence and full81 run executed at `b74f243`; raw-input timing at `e18974f`; per-phase JSON records exact source hashes. Final official repeat and history/cache verification passed. Eleven isolated tests passed. Initial preparation float64 tolerance failure is retained under `prepared_failed_float64_tolerance`; fixed checks use FP32 scale-aware rounding, and all648 numeric arrays are identical between preparations.

Primary resource results use full raw-image processing, not prepared-cache timing. Both paths were compared against every fixed cache prediction; RGB preprocessed arrays were also exactly equal to prepared inputs. Run separately with no competing GPU process:

```bash
bash tools/m8j11_env.sh -m tools.m8j11_raw_timing --model StreamPETR
bash tools/m8j11_baseline_env.sh -m tools.m8j11_raw_timing --model A
bash tools/m8j11_baseline_env.sh -m tools.m8j11_verify
bash tools/m8j11_baseline_env.sh -m tools.m8j11_visualize
bash tools/m8j11_baseline_env.sh -m tools.m8j11_report
```

These exclusive-create commands describe the recorded run; to repeat, first choose a new private run path/protocol and output names, preserving existing evidence. `tools/m8j11_phase.py NAME bash ...` provides bounded600s phase supervision, command, PID, exit status, source SHA and log SHA. No training process exists.

Environment provisioning used copied torch2.0.0+cu118/Python3.8 and independently installed mmdet2.28.2/mmseg0.30/einops0.6.1/MMDet3Drc6. MMCV1.7.0 was compiled from the officially hash-verified release with CUDA11.8/GCC11/sm89; the initial serial build timed out. During the one bounded continuation, PATH was corrected to find ninja and MAX_JOBS=4. All128objects linked before timeout during wheel ZIP packaging; `python -m wheel pack` completed packaging of the existing staging tree without recompiling. Built wheel hash `a2c9b54decf18d5b01cc3012ca0ef1179b4f316b068f01239041c2ef041e0ed3`. The incomplete476byte wheel was not installed. All failures and their resolution are recorded.

The candidate uses public train-set weights and two previously used development scenes. Broad generalization, temporal causal benefit, online deadlines and driving safety remain unvalidated. No automatic follow-on experiment.
