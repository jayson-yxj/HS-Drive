# Official source/backend review (2026-10-29 UTC)

Only StreamPETR-R50 ImageNet90epoch900query was selected, before evaluation. Official commit/config/checkpoint identities are in the frozen protocol. The README public scores are **not** CARLA/mini_val measurements and are not used as local comparison results.

- Official setup explicitly says Flash attention is optional and corresponding imports can be omitted. Official repository provides `PETRMultiheadAttention(fp16=False)` and FP32 speed paths. Our private import patch skips optional Flash and unrelated EVA imports only.
- FlashMHA and torch MultiheadAttention use identical packed QKV and output projection state keys. Strict full591tensor loading is required. A synthetic standard-attention/QKV calculation tests the supported FP32 backend; it does not certify bitwise equivalence to Flash execution.
- FP32 module input/output checks, no autocast/wrap_fp16/scaler, and `state_dict` equality are required. Timestamp memory is deliberately float64, as official formatting requires. `fp16_enabled=True` exists on the wrapper but its FP32 forward has no auto_fp16 decorator; actual dtypes are measured.
- Original native900queries =644learned +256propagated. Memory1024,256internal top proposals. Memory is updated from native output before public Top300/score/range filtering. It is not GT memory.
- Official inference skips the auxiliary2D ROI head when aux_2d_only=True; those weights remain loaded, not repurposed. The used visual path is ResNet50 stage3/4 -> CPFPN -> selected stride16 feature -> temporal head.
- Original GridMask returns input unchanged in eval; its probability draw may consume RNG, but no pixels are masked. Scene repeats use eval and deterministic cuDNN; nonfinite values before official nan_to_num must fail.
- Camera source transforms use each sensor timestamp and pose. The reference named ego_pose in this implementation is lidar2global, not literal ego2global. Reading reference calibration/poses is not point-cloud inference.
- MMDet3Drc6 has new LiDAR box conventions. We use the native converter's documented LWH/CCW geometry, then a tested prediction-only bridge to the unchanged J9 exporter. No GT-derived fields are inserted.
- Public output remains nativeTop300 followed by score>=0.2 and strictabsXY<51.2, class-specific export radius and original official evaluator. No NMS. Attribute rules remain deterministic J9 speed/class defaults. The J9 use_external=False JSONflag is kept for comparability, while ImageNet pretraining is explicitly disclosed; that metadata flag does not alter metrics.

Sources: official local repository files with hashes in configs/m8j11_protocol.json; https://github.com/exiawsh/StreamPETR/tree/95f64702306ccdb7a78889578b2a55b5deb35b2a ; MMDet3D https://github.com/open-mmlab/mmdetection3d/tree/47285b3f1e9dba358e98fcd12e523cfd0769c876 .
