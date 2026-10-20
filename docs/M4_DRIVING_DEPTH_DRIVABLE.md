# M4 增量：独立驾驶、深度与静态 Drivable

本文记录继小样本工程基线之后的实现。整体 ResNet18、48-bin LID、四高度层、128 通道 BEV 和 RGB-only forward 接口不变。结果见 `reports/m4_expansion.md`（已实际完成）；旧基线报告和数据不覆盖。

## 数据与划分

`configs/capture_expanded.yaml` 定义六条纳入训练/验证/测试的 16 秒、161 帧 Traffic Manager 低速驾驶片段；另保留一条运动比例不足的停车诊断片段及一条传感器缺帧片段，不纳入本轮训练。四条训练路线来自 Town01/02，验证路线来自 Town04，新的留出测试路线来自 Town05。不同出生点不自动等于独立路线：`tools.audit_driving_data` 记录累计路程、5m 格子覆盖、运动比例、航向变化、同地图轨迹距离和重叠。相同地图的新路线采用相同 group，绝不跨划分。Town03 原验证场景作固定回归集；Town10 已在前次工作中评测过，视为历史测试，不再称全新盲测。

复用 CARLA 0.9.15 官方 Traffic Manager。采集进程必须独占世界 tick 和空闲 TM 端口 8050，使用同步模式、固定随机种子、不自动变道、5m/s 目标速度，保持交通灯规则。清理时关闭自车 autopilot / 自有 TM 同步并恢复世界设置。未加入 NPC，不能作为交互或学习控制闭环通过证据。地图 OpenDRIVE 导出仅作监督来源，不导出为 Learned 输入。

官方依据：[Traffic Manager](https://carla.readthedocs.io/en/0.9.15/adv_traffic_manager/)、[Map/Waypoint API](https://carla.readthedocs.io/en/0.9.15/python_api/#carlamap)。

## Drivable 的当前语义

`datasets/drivable.py` 定义 **static lane support**，属于技术基线中 Drivable 的静态道路组成部分：

- `drivable_static`：uint8 `[160,120]`，1 为 Driving 车道内部；0 为明确映射的 Sidewalk/Shoulder/Biking/Border/Median 内部；255 为 Unknown。
- `drivable_valid`：bool，同形状，只允许 0/1 参与损失和指标。车道边缘 0.5m、路口、无地图、未支持车道种类、距局部 Ego 平面高度差 >0.75m 均屏蔽。斜坡远处可能被保守屏蔽，不能把缺失区域判成负样本。
- `drivable_rule_valid`：当前全 false。尚无交通灯、行驶方向、导航路径许可监督；不能将 static lane support 直接当作当前合法通行图。

静态支持与障碍占用/可见性独立，障碍物覆盖的道路仍是道路；不可见处可以有特权地图监督，但不因此获得 Free 标签。多层道路、高坡、路口与当前规则仍是后续工作。

标签在独立 sidecar 目录，原始录制不可变。manifest 固定地图、原帧、BEV 配置、代码哈希；index 构建验证逐帧身份与数组类型/掩码。旧录制使用同版本 CARLA 包内 OpenDRIVE，并明确标记非同期地图导出。DataLoader 仅在 `supervision=True` 打开标签，旧场景缺标签时返回全 Unknown。RGB inference 不读取 NPZ、map.xodr、Actor 或 target JSON。

## 深度对照与训练

三份配置：`m4_expanded_ce.yaml`、`m4_expanded_depth.yaml`、`m4_expanded_joint.yaml`。使用相同新增数据、seed、48 帧/场景采样上限、1200 更新步、batch=1/BF16。CE 对照保持原深度损失；depth 组额外对预测分布期望深度的 log 值加入 SmoothL1（beta=0.2，权重1）；joint 组再加静态 Drivable 1×1 head 和 masked balanced BCE+Dice（权重0.2）。范围均衡损失接口已实现但本次预设关闭，避免同时引入更多变化。

每 200 步只以验证深度 MAE 选择 checkpoint，测试不能用于选步/选损失。全 split 输出分距离、分场景深度指标和训练子集拟合的像素位置均值先验。新旧数据和训练步数均发生改变，历史基线不是严格数据单变量消融；本次 CE/depth 两组是同预算损失对照。Drivable 仅在有效区统计 IoU，须同时报告有效覆盖与全正类别先验，避免大面积 Unknown 或类别不均衡误导。另用 `tools.eval_drivable_prior` 仅从实际训练子集拟合逐 BEV 格子道路频率先验（阈值0.5，未见格回退训练全局频率），在完全相同评价 mask 比较，识别固定自车视角带来的位置捷径。

## 可执行命令

在项目根目录运行：

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2000 Epic
# 另开终端，完整场景可验证后跳过，已有未完成场景需改 run_id 保留失败证据：
.venv/bin/python -m tools.capture_campaign
.venv/bin/python -m tools.audit_driving_data --output reports/m4_driving_audit.json
.venv/bin/python -m tools.prepare_expanded_dataset --registry configs/dataset_v2_accepted.yaml --output /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted
# 采集完成后停止本次 CARLA 进程，避免与训练争用 GPU。
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_expanded_ce.yaml --output runs/m4_expanded_ce
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_expanded_depth.yaml --output runs/m4_expanded_depth
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_expanded_joint.yaml --output runs/m4_expanded_joint
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_expanded_joint --split val --output reports/m4_expanded_joint_val
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_expanded_joint --split test --output reports/m4_expanded_joint_test
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q
```

所有训练、评测和索引输出目录要求全新；重跑请改输出路径。`prepare_expanded_dataset` 使用 CPU 并可等待同一 campaign 的正在采集场景，再生成 sidecar/索引。已有输出不覆盖。正式指标与是否通过，必须以实际完成后的报告为准。

运动筛选：Town02 spawn0 完成161帧但 >1m/s 的帧占比仅37.9%，低于预设50%，保留全部原始文件。用 Town02 spawn20 补充；候选 registry 保留；候选索引在 Town04 完整性校验失败后未提交，最终只提交 accepted 索引，模型仅使用后者。筛选只依据运动与完整性，不查看测试深度误差。

Town04 首轮记录在区间内出现1个不完整传感器帧，索引门禁拒绝。保持拒绝门槛，原始记录/诊断/失败日志保留；使用同地图/出生点/seed重录，wall delay增加至0.7s，仿真步长和10Hz采样不变。


本轮最终结果：正式索引1461帧，train/val/test=1017/222/222；训练实际均匀采样299帧。CE/log-depth/joint三组完成。MAE选择的主要深度组为CE，joint为预设次要实验，测试后不改选择。joint完整验证/测试深度MAE=4.151/3.240m；Drivable IoU=0.511/0.608，验证仍未超过位置/全正先验，完整M4开放。

完整对照和位置先验可复现（更改prefix和输出目录以避免覆盖历史）：

```bash
.venv-gpu/bin/python -m tools.run_depth_comparison --prefix m4_expanded_repeat
.venv/bin/python -m tools.eval_drivable_prior --run runs/m4_expanded_joint --split val --model-metrics reports/m4_expanded_joint_val/metrics.json --output reports/m4_drivable_prior_val_repeat.json
.venv/bin/python -m tools.eval_drivable_prior --run runs/m4_expanded_joint --split test --model-metrics reports/m4_expanded_joint_test/metrics.json --output reports/m4_drivable_prior_test_repeat.json
```

`capture_campaign` 中 `complete` 仅表示采满请求帧，不能替代独立完整性验收；本次Town04首录就是完整性失败示例。`audit_driving_data` 检查运动；`prepare_expanded_dataset` 的 index 构建检查实际完整性，两者均通过才纳入训练。

产物一致性审计：`.venv/bin/python -m tools.validate_m4_expansion --output reports/m4_expansion_acceptance_repeat.json`。本轮 `reports/m4_expansion_acceptance.json` 工程检查全部通过，质量诊断单列，`full_m4_closed=false`。该审计不是事后放宽完整M4阈值。
