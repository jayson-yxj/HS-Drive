# M4.8 — Spatial Evidence Representation & Control Feasibility

基线 `d8393bc34147d2a0d2ca15b9feb8ebc004a8a241`；独立分支 `feat/m48-spatial-evidence-control`。旧文件、默认网络、Depth Bin、Splat、车辆 Physics Control 均保留。M5-Pilot 只写入独立 `runs/m48_pilot`。不批准自动扩大训练。

## Spatial Evidence Label V2 数学合同

Schema `hsd-m48-spatial-evidence-v2`。输出 `[Z,X,Y]=[4,160,120]`，HS X前/Y左/Z上；X=[-20,60)、Y=[-30,30)、XY=0.5m、Z=[-2,-.5,1,2.5,4]。与旧 BEV 完全一致。

定义固定测量配方 R、可靠首表面集合 P、体素 C：

- S(C)=1：P 中至少一个表面端点位于 C。
- F(C)=1：R 中至少一条通过可见性审核的自由线段与 C 存在正长度交集。
- O(C)=S(C)∨F(C)：至少有一种观测事件；不等同体积全部被观察。
- S、F 可同时为1；模型用独立 sigmoid，不使用互斥 Softmax。
- 在 O 内 S=0/F=0 的负样本表示**固定配方未产生对应事件**，不是“物理表面不存在”或“完全没有自由空间”。O=0 不计算任何损失。改变采样配方会改变标签分布，必须版本化。
- `free_ray_evidence` 永远不得直接用作 Planner 的可通行空间。

字段：S/F 对应 `surface_evidence`/`free_ray_evidence`；`surface_valid=O`；`free_evidence_valid=O & ~evidence_conflict_mask`。`observability=O`；`mixed_cell_mask` 标注 S∧F、细格射线/表面交会或语义歧义。不是通过扩大 Unknown 来抹除混合。

`surface_semantic` 为 `[3,4,160,120]` multi-hot，通道 Ground/Obstacle/Other。静态、可运动表面均合并 Obstacle；Ground 不是 Obstacle。不同细高度子格中 Ground 与 Obstacle 可同时存在；同一个细格语义矛盾或未知语义则屏蔽相应语义损失。Obstacle 有独立 `obstacle_valid`，不把 Other/Unknown 当障碍负类。

内部 XY 仍0.5m，Z细分0.25m；一个1.5m粗高度层含6个子格。端点提供细格表面；自由射线精确遍历网格，距端点回退0.1m，遇到已知细格表面即截断，记录歧义。聚合使用逻辑 OR；`fine_*_fraction` 只表示有事件的子格比例，不是自由体积比例或置信概率。少量子格的自由证据不推广成整格空闲。

复用 M46 的 ego-present/ego-removed 可见性审核，原深度边缘和自车过滤不放宽。自车相交粗格无监督。标签生成与实际推理隔离：`EvidenceDataset(...,supervision=False)` 不读 sidecar，输入仍旧 LearnedInputs 白名单；Actor、Depth、Instance、反事实参考都不能进入 Encoder。

## 固定配方与公平比较

V1：旧 sparse16 + dense2 + 原边界 Unknown，历史生成器不修改。
V2 matched：同 sparse16/dense2，比较表征变化。
V2 dense：free仍stride16，表面使用原生像素但排除 `(u+2v)%4==1`；排除像素只评价。这些交错像素共享渲染器，不是独立真值。

新 Town01 spawn20 Audi A2、Town03 spawn180 Tesla 在生成器和协议 SHA 固定后采集评价；规则未再调参。这些是本轮未见位姿，不声称项目从未见过对应地图。旧 Town02 只叫开发/历史评测。碰撞查询是不同表示的补充证据，不能视为渲染 Mesh 或真实体积；可见性分桶仅分析，不用于训练筛选。

104格局部重放主动使用审计端点，因此是合同验证，不能叫盲测。正式生成仍留出像素，104格中未恢复的表面必须如实报告。

## Pilot

仅训练 road_a、near_a 各首帧（同位姿、不同障碍配置），选择写入 `configs/m48_protocol.json`，不读验证集做选择。完全冻结 `m4_expanded_joint`，RGB推理一次缓存BEV，训练共享两层64通道Conv + 三输出头400步；位置先验是每格可训练logit、同样400步。这个先验可记住固定空间，比坐标MLP更强，不能把过拟合当视觉优势。

图像/Head BF16 autocast；原 Splat 的几何计算禁用 autocast 并保持 FP32，脚本hook实际检查dtype。Loss按正/负事件分别平均后平衡；仅在 Mask 内计算，几何/语义歧义从 Pilot 监督移除。两帧没有 Other 正样本，因此其正类可学习性未验证。Mask屏蔽后的事件学习成功不批准正式 Occupancy 泛化训练。

## 控制和道路候选

`m48_motion.RouteController` 复用旧纵向控制器，仅修改横向lookahead；路线速度目标2.5m/s，但实测仍有超调，不宣称速度控制已修复。M47完整6秒Safety、真实车型尺寸和裕量保留。当前道路判定不再使用厘米量化cache，而用精确坐标，避免边界错误接受。

`m48_road.CertifiedRoad` 只为 OpenDRIVE 常宽、零laneOffset、非路口直线构造内缩矩形。弯道/路口/边界回退完整地图。第一版坐标符号审计失败，修正版4地图抽样零假接受；尚未证明高度重叠、孔洞拓扑和方向合法性，所以闭环**不启用**证书。所有实际结果仍走完整地图。

`m48_junction_short` 的2秒检查只是延迟诊断，不是安全验收替代。`m48_schedule.TimedPlan` 提供5Hz/20Hz采样接口，世界坐标重采样、原生成时间有效期0.25s、每帧当前Footprint/短时障碍检查；6秒完整Safety在规划帧运行。当前执行器仍串行、会阻塞控制，不能称实际实时20Hz。重采样后的跟踪误差不再错误解释为原规划不满足运动学，但实际速度/跟踪偏差须继续验收。紧急/超时/非法/无轨迹强制制动保留。

## 实际环境与复现

沿用 `.venv` CARLA0.9.15、`.venv-gpu` PyTorch2.8 CUDA环境，RTX4090；不新增运行依赖。CARLA端口2100；同一时刻只能有一个tick owner。所有输出拒绝覆盖，复跑需新路径。以下列出本轮入口；报告和原始逐帧JSONL保留配置/源码SHA，失败目录也保留。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
git switch feat/m48-spatial-evidence-control
nvidia-smi
.venv/bin/python -c 'import carla; c=carla.Client("127.0.0.1",2100); c.set_timeout(30); print(c.get_server_version())'

.venv/bin/python -m tools.m48_spatial \
  --raw /root/gpufree-data/hs-drive-data/raw/m46_occupancy \
        /root/gpufree-data/hs-drive-data/raw/m47_development \
        /root/gpufree-data/hs-drive-data/raw/m47_heldout_town02 \
        /root/gpufree-data/hs-drive-data/raw/m47_heldout_town02_tesla \
  --output reports/m48_spatial_rerun.json \
  --labels /root/gpufree-data/hs-drive-data/processed/m48_labels_rerun \
  --artifacts reports/m48_spatial_rerun_artifacts

.venv/bin/python -m tools.m48_capture --map Town01 --spawn 20 --vehicle vehicle.audi.a2 --role heldout --output /root/gpufree-data/hs-drive-data/raw/m48_town01_rerun
.venv/bin/python -m tools.m48_capture --map Town03 --spawn 180 --vehicle vehicle.tesla.model3 --role heldout --output /root/gpufree-data/hs-drive-data/raw/m48_town03_rerun
.venv/bin/python -m tools.m48_spatial --raw /root/gpufree-data/hs-drive-data/raw/m48_town01_rerun /root/gpufree-data/hs-drive-data/raw/m48_town03_rerun --output reports/m48_heldout_rerun.json --labels /root/gpufree-data/hs-drive-data/processed/m48_heldout_rerun --artifacts reports/m48_heldout_rerun_artifacts

.venv-gpu/bin/python -m tools.m48_pilot --steps 400 --run runs/m48_pilot_rerun --output reports/m48_pilot_rerun.json --artifacts reports/m48_pilot_rerun_artifacts
.venv/bin/python -m tools.m48_dynamics --output reports/m48_dynamics_rerun.json --artifacts reports/m48_dynamics_rerun_artifacts
.venv/bin/python -m tools.m48_junction --scenarios straight mild_curve junction_turn --versions m48 --replay-routes reports/m45_oracle_driving_artifacts --green-lights --output reports/m48_routes_rerun.json --artifacts reports/m48_routes_rerun_artifacts
.venv/bin/python -m tools.m48_multirate --scenarios straight mild_curve junction_turn emergency timeout invalid missing --versions m48 --replay-routes reports/m45_oracle_driving_artifacts --green-lights --output reports/m48_multirate_rerun.json --artifacts reports/m48_multirate_rerun_artifacts

.venv/bin/python -m pytest -q -ra --junitxml=reports/m48_sim_rerun.xml
.venv-gpu/bin/python -m pytest -q -ra --junitxml=reports/m48_gpu_rerun.xml
```

Pilot需要另指定 `--artifacts` 新目录（见脚本help）；其标签报告固定为本轮冻结输入，训练SHA仍可复核。仅固定输出路径的分析工具（`m48_extra_contract`、`m48_visibility_analysis`、`m48_road_audit`、`m48_road_diagnosis`、`m48_low_speed_followup`、`m48_hold_feasibility_analysis`、`m48_pilot_metrics`、`m48_junction_metrics`、`m48_visualize`）应在独立工作树中复现；不要删除旧产物来复跑。低速补充、道路图和图像原始数据位于独立 `m48_*` 数据目录，不写旧索引。

## 独立状态

工程接口通过、Pilot有限拟合成功、泛化质量和安全闭环是四个独立门禁。实时性、舒适性未过；正式M5扩大训练未获批准。结果见 `reports/m48_summary.md`、机器门禁 `reports/m48_acceptance.json`。

最终交付补充：`m48_schedule`保存轨迹/状态深拷贝，原时间戳不受外部修改；`m48_multirate_faults_final.json`是该版本实际故障复测。源码改动与最终测试见`m48_delivery.json`，原运行源码快照保留。
