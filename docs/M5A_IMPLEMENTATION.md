# M5-A 实施与复现

基线 `ffb466df3471ad93a6bb11c51891dec7f63fb4fb`；独立分支 `feat/m5a-visual-evidence-generalization`。原 M4–M4.8 文件、默认控制器、Depth Bin、Splat、物理配置不变。没有合并 main。

## 数据与监督合同

`configs/m5a_protocol.json` 是采集前固定协议。12 个位置组 × 9 种布局 = 108 帧；每地图 2/1/1 个组分入 train/val/test，即 54/27/27 帧。六摄 RGB 640×384，18 个同步 RGB/Depth/Instance sensor。自车沉降后冻结，摄像头固定世界位姿。每种布局另采匹配的去自车参考，只用于标签可见性过滤。布局是静态摆放，不代表运动物体监督；无随机跨帧拆分。Town01/02/03 是历史地图，只称 M5-A Head 未见位置组。

保留原 `datasets/m48_evidence.py`，使用原可靠射线接口，native surface 像素去除 `(u+2v)%4==1`，free stride16，细高度 .25m 后聚合至 `[4,160,120]`。监督入口 `datasets/m5a_training.py`，输入沿用原 allowlist；关闭监督时不会读标签 sidecar。可靠 Mask 去除 ego、未解几何冲突、语义冲突。

- `observability = surface OR free_ray` 是测量事件并集，不能解释成完整观察物理体素。
- Surface/Free 的负例表示本配方未记录事件。二者独立 BCE，混合格可同时为正；Unknown 无损失。
- Obstacle 通道仅在可靠可见 Ground/Obstacle 表面判别；Ground 可提供非障碍表面负例，不能据此证明整格没有障碍。Other/不明表面不作 obstacle 负例。
- 没有把 Free-Ray 输出接入 Planner；也没有把 Actor GT 输入视觉推理。
- Other 在本轮无训练正例，不能宣称其识别能力已验证。

`m5a_spatial_contract_v2.json` 的参考是同渲染器交错留出像素，不是独立 Mesh 真值。初版 `m5a_spatial_contract.json` 仅包含静态障碍语义 code2，保留为诊断记录；v2 包含 code2+3。没有据留出结果修改标签。

## 学习对照与评测

复用 M4 expanded joint Encoder、M4.8 EvidenceHead/Loss。A 是 384,000 参数位置先验；B 是 112,020 参数卷积 Head。两者种子 51/52、同一 batch2 索引计划、600 steps；A lr .05，B .003。容量和优化器步长不同，不能视作架构单变量消融。B 缓存冻结 RGB-BEV feature；几何 FP32，视觉及 Head 在 BF16 autocast 下运行（归一化等输出仍可能 FP32）。真实训练峰值和耗时见 training JSON，独立 Encoder 前向见 GPU JSON。

每100步仅在 val 按场景均值 `(Surface IoU + Obstacle IoU)/2`、阈值 .5 选模型。条件 C 只看 val：B 未对两项均取得至少 .02 增益才允许 Context-only 200步。本轮两个种子均超过该条件，因此**没有运行 C**。不能在看到 test 后重定义条件；下一研究需新的协议与独立测试组。

阈值在 val 的 .1–.9 中选全局 IoU 最大值；同时报告固定 .5。每任务在 val 拟合 scalar temperature+bias，test 报 Brier/ECE；概率仅针对事件标签，绝不解释成物理占据/安全阈值。PR-AUC 用 ties 分组的非插值 average precision。`None` 表示没有对应正例/定义域。

测试含场景、距离、高度层统计和局部直行走廊 x∈[0,30], |y|<2；该走廊不是实际导航转弯路线。平均图像、跨位置错误配对、语义 actor 像素遮蔽、同位姿配置交换均为诊断输入；遮蔽 Mask 不进入正常推理。新增 changed-cell 诊断比较同位置空场景与障碍 RGB，在双方有效且事件变化的交集上统计，样本少且不代表完整障碍物体积。

## 真异步运行时

`simulator/m5a_async.py` 使用 spawn 进程与有界队列。Worker 只有离线 OpenDRIVE Map、route、深拷贝状态/障碍 snapshot，**没有 CARLA Client/World/tick**。Planner+Safety 保持60×.1s完整六秒；精确坐标地图查询缓存上限100,000，未启用近似道路快速接受证书。

唯一主进程每 .05s tick、每 .2s提交规划。记录 Scene/Frame、输入状态仿真时间、输入/开始/完成墙钟时间、控制应用时间。校验结果格式、非有限值、乱序、未来帧、原始 generation time 和双时钟 .35s有效期，控制应用前再次检查。重采样不重置原始 age；可用剩余时域随 age 缩短，但原始检查六秒完整。

超时/无轨迹/错误输入/Worker崩溃均制动。主线程每tick检查当前道路 footprint，并对提供的 Actor 做 .35s 常速度矩形碰撞 guard。仍是 Oracle 状态/地图/Actor 验证，**不是视觉闭环**。矩形 SAT 在恒定朝向/速度假设下连续，路面角点采样不是精确 swept mesh；未验证密集动态交通。

`m5a_async_runtime.json` 保存首个完整9场景运行；当时有离线分析/测试并行负载。`m5a_async_isolated_runtime.json` 为清空此类负载后的最终运行，并加固控制应用前有效期检查。全部日志保留。静态障碍首次 spawn 碰撞失败见 `async_drive.txt`，按道路高度沉降修复后另存，未覆盖。

墙钟平均Hz、控制应用间隔、tick完成 deadline、Controller计算、Planner服务时间分开报告。cached expired plan 的 age 保留用于诊断；`m5a_closedloop_safety.json` 另算实际正常控制所使用的 age。严格50ms实时性独立于“平均20Hz”。

## 服务器命令

在仓库根目录执行。工具拒绝已有输出目录；复现应使用新 checkout/工作目录和新输出路径，不能覆盖历史资产。现有完整数据可直接读取。

```bash
# 两套既有环境：CARLA采集/运行 与 Torch/CUDA分析
.venv/bin/python -c "import carla; c=carla.Client('127.0.0.1',2100); print(c.get_server_version())"
.venv-gpu/bin/python -c "import torch; print(torch.__version__,torch.cuda.get_device_name())"
# 全新采集示例；协议内已固定组/划分，勿改后冒充原实验
.venv/bin/python -m tools.m5a_capture --output /root/gpufree-data/hs-drive-data/raw/m5a_reproduction
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5a_reproduction --output /root/gpufree-data/hs-drive-data/processed/m5a_reproduction --report reports/m5a_reproduction_quality.json
# 原实验训练命令（已有runs时会拒绝；使用新的 --run/--output）
.venv-gpu/bin/python -m tools.m5a_learning --index /root/gpufree-data/hs-drive-data/processed/m5a_index --run runs/m5a_learning --output reports/m5a_training.json
.venv-gpu/bin/python -m tools.m5a_evaluate
.venv-gpu/bin/python -m tools.m5a_visualize
.venv/bin/python -m tools.m5a_quality_audit
.venv-gpu/bin/python -m tools.m5a_reachability
.venv/bin/python -m tools.m5a_async_drive --output reports/m5a_async_isolated_runtime.json --artifacts reports/m5a_async_isolated_artifacts
.venv-gpu/bin/python -m tools.m5a_runtime_analysis
.venv/bin/python -m tools.m5a_provenance
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_geometry_extended.py tests/test_imaging_rig.py tests/test_session.py tests/test_snapshot.py -q
```

原始试采集/full采集、各阶段命令输出见 `reports/m5a_execution_logs/`。训练精确源码快照在 `runs/m5a_learning/source`，采集精确快照 `reports/m5a_capture_source.py`，异步各次运行有独立 `source/`。`m5a_dataset_manifest.json` 保存划分/样本引用，`m5a_asset_verification.json` 保存外部原始数据/标签/checkpoint SHA。外部数据和checkpoint保留服务器，不将约900MB图像复制进Git。

复现训练不保证 CUDA scatter 原子操作逐bit相同；使用两种子衡量重复性，不把种子间差异隐藏。数据重采集也会有渲染和帧号差异，数值精确复核应使用已哈希的数据与checkpoint。

评测边界统一使用严格 `probability > threshold`，与选模 `logit > 0`一致。原 `>=` 版保存为 `m5a_initial_*`；权威必交JSON已换成修正版，`m5a_corrected_*`保留修正过程。初版停车检测误计启动静止也已修正；权威安全JSON记录最终1s稳定。

实际依赖版本见 `reports/m5a_gpu_requirements.txt` 与 `reports/m5a_carla_requirements.txt`（GPU环境含matplotlib，CARLA环境含carla0.9.15）。最终源码用Black24.10.0格式化，执行时未格式化源码快照保留用于哈希溯源。Worker崩溃注入在主线程执行terminate/join，可能计入该压力场景的控制尾延迟；不是生产正常控制路径。
