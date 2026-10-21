# M5-K 实施与复现

基线 `db03730`，分支 `feat/m5k-target-depth-geometry`。代码仅增加显式离线研究路径，原模型、标签、Runtime、Controller 和 Physics 未改动。每个输出工具拒绝覆盖已有结果；复现必须使用新的输出目录/工作副本，不能直接删除历史结果再运行。

## 环境

实际 RTX 4090 24GB，Python 3.10，`.venv-gpu` 为 PyTorch 2.8/CUDA12.8；`.venv` 含 CARLA 0.9.15 SDK。CARLA Epic offscreen，由非 root 用户运行：

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

无视觉控制。采集复用严格同步六摄 RGB/Depth/Instance 与每个配置独立的 ego-removed 自车参考。几何协议固定位置、天气和标定；跨 split 的位置及所有变体整体隔离。新测试 RGB 在科学配置冻结后采集。

## 首项对齐诊断

```bash
.venv-gpu/bin/python -m tools.m5k_prepare
.venv-gpu/bin/python -m tools.m5k_alignment --split historical_val
.venv-gpu/bin/python -m pytest -q tests/test_m5k_geometry.py tests/test_m5k_input.py tests/test_m5k_adapter.py
```

原生 S16 概率在 FP32 中双线性重采样，再逐射线归一化。所有 P8 组沿相同整数 UV `4::8`，深度指标统一严格 Mask 且 GT 在 BEV 内。原生 S16 单列。正式 Mobile+LearnedDepth 投影不接受 GT Mask；Oracle 干预明确单列。score=`1-exp(-sum(q*p))`，阈值精确 0.2。

## 新开发数据与门禁

```bash
.venv/bin/python -m tools.m5k_capture_protocol
.venv/bin/python -m tools.m5f_capture --protocol configs/m5k_capture_dev.json --output /root/gpufree-data/hs-drive-data/raw/m5k_dev
.venv-gpu/bin/python -m tools.m5k_dev_pipeline
```

Pipeline 等待全部采集 manifest 完成，再依次运行 V2 sidecar、像素标签、Train/Val 支持审查、对齐实验、训练决策和采样诊断。训练决策输出 `reports/m5k_training_decision.json`。不得依据测试结果补调 Loss 或预算。

条件通过时才执行：

```bash
.venv-gpu/bin/python -m tools.m5k_train
.venv-gpu/bin/python -m tools.m5k_residual_eval --split val
```

B/C 残差共享 113 输入通道、32 隐通道、48 输出；无 Mobile 组将最后一通道置零。最后卷积零初始化，初始 logits 精确等于原模型。两种子，各 600 步，Val loss 在 0/200/400/600 选择。原 Backbone/Mobile/Depth 不更新。

Loss 为四个有效区域的等权深度 CE（近/远 Actor 与近/远背景）+0.25 Actor 同层 1m 概率 NLL +0.5 原 M4 近场 Replay CE +0.1 近场原分布 KL。空间正样本要求至少一个 48-bin 中心实际落入邻域，无法表示的射线不造假空间正例，但仍保留可靠深度 CE。GT 仅出现在这些 loss 参数，不进入残差 forward。该轮不宣称独立验证空间 Loss 的贡献。

## 采样与输入通路

```bash
.venv-gpu/bin/python -m tools.m5k_sampling --split val
.venv/bin/python -m tools.m5k_native_buffer
.venv-gpu/bin/python -m tools.m5k_input_pipeline
```

P8 中心与 S4 中心不重叠。Adaptive extras 为 Mobile q≥0.9 的 S4 像素；Uniform 逐帧逐种子选相同个数、均匀覆盖 camera-major 索引。所有目标仍在召回分母。Learned S4 使用 P8 概率插值，仅诊断。当前 dense 计算后掩掉未选射线，实测耗时不代表稀疏 kernel 加速。

BGRA buffer 拷贝到自有 pinned uint8 内存，异步 H2D 后转 FP32。对齐六摄 frame/timestamp，拒绝缺相机、乱序、重复、未来和过期包。完整延迟包含输入验证、Mobile、原 Depth、Lift 与支持计数；直接逐次测墙钟，非相加分位数。实际传感器 buffer 回放不包含在线 callback/IPC/控制调度延迟。

## 封存验证与交付

```bash
.venv-gpu/bin/python -m tools.m5k_freeze
.venv-gpu/bin/python -m tools.m5k_test_pipeline
.venv-gpu/bin/python -m tools.m5k_details --split val
.venv-gpu/bin/python -m tools.m5k_details --split test
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_imaging_rig.py tests/test_snapshot.py tests/test_session.py tests/test_geometry_extended.py
.venv-gpu/bin/python -m tools.m5k_integrity
```

`runs/m5k_logs/` 保留完整日志及失败尝试。逐帧概率、统计及完整 SHA 索引留在服务器，Git 只提交代码、精简报告、引用 SHA 和必要图片。数据参考来自 CARLA 可见首表面及保守 Actor 关联，不是独立精确 Mesh 真值。Unknown 不充当安全负样本；Free-Ray 不表示整格可通行。
