# M4 增量：远距离深度、跨地图道路与规则监督

延续既有ResNet18、48-bin LID、四高度层BEV和RGB-only输入；Temporal/Sparse/Planner尚不启用。本轮三组训练、全量验证和冻结选择后的回归已实际完成，结果见[实测报告](../reports/m4_far_rules.md)。完整M4质量门禁未关闭。

## 预设实验

1. `m4_far_balanced`：同一v2数据、299个训练采样、seed7、1200步，四距离段CE/log-depth损失分别平均后等权相加。按验证四段macro AbsRel选择，避免近场数量决定checkpoint；同时报告近场退化、全体MAE以及远场误差。历史m4_expanded_joint为参考，但checkpoint选择准则也变了，不能称严格单变量消融。
2. `m4_road_point` / `m4_road_context`：相同数据、road v2标签、距离均衡、Drivable权重0.5、相同步序和1200步。仅Drivable head不同：原1×1与64通道dilation=1/2/4/8卷积堆叠。共享BEV仍为128×160×120；context head获得更大局部上下文，单独配置启用，旧checkpoint兼容。两组都只按验证macro AbsRel选步，然后按验证逐场景平均Drivable IoU选road配置，冻结后测试。

旧Town05/Town10已被观察过，本轮仅称历史回归测试，不再称未使用过的盲测。Town04从未进入训练，但多轮用于验证，仍需未来增加全新地图盲测和多seed。

本轮配对实验完成后，场景平均IoU按预设协议选择point头（0.650 vs context 0.617）。context保留为可复现实验，不启用为默认配置。两种头的checkpoint均按联合任务的macro-depth AbsRel选择，因此这不是针对道路单任务搜索得到的各自最佳性能，也不能据此断言扩大上下文没有价值。

## 静态道路v2

`datasets/road_supervision.py`从CARLA Driving waypoint及其next连接构造1m间隔车道条带（半宽减0.25m），包括路口内部。不使用路口包围框直接填满道路。映射至既有BEV，非道路负样本继承已验证v1；地图外保留Unknown。

新增：`junction_support`，以及HS ego坐标下`lane_direction[2,H,W]`和有效掩码。多方向交叉处方向Unknown，几何道路仍可为正；多层重叠/相对Ego高度超0.75m保守处理。`drivable_rule_valid`仍全false：路口道路支持不意味着车辆有通行权。lane_direction是有向车道监督，不直接推断本车未来路线。

新标签为独立overlay，固定原index、原帧、OpenDRIVE、BEV配置哈希；不修改历史标签/索引。DataLoader仅在supervision=True读取overlay；Inference不打开overlay manifest。新旧IoU有效mask不同，不能直接比较数值；point/context两组用同一v2mask、全正先验和训练位置先验。

`HSDriveDataset(index, split, road_overlay=...)`保持原RGB输入合同。新增的单帧target如下，序列/批处理在前面添加对应维度：

| 字段 | 类型/形状 | 语义 |
|---|---|---|
| `lane_direction` | float32 `[2,160,120]` | HS ego中平面单位方向；无效处为0 |
| `lane_direction_valid` | bool `[160,120]` | 方向冲突、多层重叠、非道路无效 |
| `junction_support` | bool `[160,120]` | 路口内Driving条带的有效正格 |
| `signal_stop_required` | int64标量 | 1停车义务，0仅该信号无需停车，-1未知 |
| `signal_stop_valid` | bool标量 | 仅关联唯一、状态稳定的红/绿灯为真 |
| `signal_state` | int64标量 | 当前ego有效关联信号：0红、2绿、-1未知；完整五态保留在原始规则JSON |

本轮道路损失使用v2 `drivable_static`和有效掩码，方向与信号字段暂未加入模型损失。没有新监督的历史场景返回零有效掩码及未知值，不制造负样本。

## 信号与停止位置监督

`simulator/traffic_rules.py`用CARLA0.9.15官方`get_stop_waypoints`和`get_state`；官方说明前者来自trigger box，后者来自last tick缓存。因此标为“trigger-derived stop position”，不假称精确路面停止线。每次world.tick后，在读取前后确认相同frame，和该帧snapshot/图像一起缓存。状态转换两tick内保留Unknown，处理渲染延迟风险。

仅当停止位置匹配当前road/section/lane、前向一致且距离在0–60m内时建立本车关联。稳定Red→stop_required=1；稳定Green→0（仅信号停车义务解除）；Yellow/Off/Unknown/关联冲突→无效。`full_traffic_permission_valid`始终false：未覆盖路线许可、让行、优先权、行人及障碍净空。

`signal_stop_valid`表示规则关联与灯态稳定，不表示该灯在RGB中可见。目前未提供逐灯可见性/遮挡掩码、图像ROI或视觉识别头；后续训练信号识别时必须补齐对应观测条件，不能把这个规则有效掩码直接当作可见性标注。

`--signal-probe`为静止、固定红→黄→绿→红实验，0.9.15的freeze为全局操作，因此先保存全部信号灯态与全局冻结状态，仅控制实验组灯态，退出恢复全部灯态及全局冻结状态；不作为闭环成功证据。三地图Town01/04/02分别用于规则接口train/val/test，所有灯态在同一录制组内，严禁把同位置不同灯态随机跨split。真实信号识别网络尚未训练，本轮先验收监督合同。

依据：[CARLA 0.9.15 TrafficLight API](https://carla.readthedocs.io/en/0.9.15/python_api/#carlatrafficlight)、[CARLA Map/Waypoint](https://carla.readthedocs.io/en/0.9.15/core_map/)。

## 可执行命令

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_far_balanced.yaml --output runs/m4_far_balanced_repeat
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_far_balanced_repeat --split val --output reports/m4_far_balanced_repeat_val
.venv/bin/python -m tools.build_road_overlay --index /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted --output /root/gpufree-data/hs-drive-data/processed/road-overlay-v2-repeat
.venv/bin/python -m tools.audit_road_overlay --index /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted --overlay /root/gpufree-data/hs-drive-data/processed/road-overlay-v2-exact --output reports/m4_road_overlay_audit_repeat.json
.venv/bin/python -m tools.visualize_road_supervision --index /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted --overlay /root/gpufree-data/hs-drive-data/processed/road-overlay-v2-exact --scene m4drive_town01_0 --output reports/road_supervision_repeat
# 已有专用CARLA服务、无训练时运行；新prefix避免覆盖原始录制：
.venv/bin/python -m tools.capture_signal_probes --prefix signals_repeat
.venv/bin/python -m tools.prepare_signal_dataset --prefix signals_repeat --registry configs/dataset_signals_repeat.yaml --audit-prefix reports/signals_repeat --output /root/gpufree-data/hs-drive-data/processed/hsd-signals-repeat
# 原始本轮记录的独立验收：
.venv/bin/python -m tools.validate_signal_capture --scene /root/gpufree-data/hs-drive-data/raw/m4signal_v2_town01 --output reports/signal_town01_repeat
.venv-gpu/bin/python -m tools.audit_signal_dataset --index /root/gpufree-data/hs-drive-data/processed/hsd-signal-probes-v1 --output reports/signal_loader_repeat.json
# 停止本次专用CARLA后，按配对协议训练+验证+冻结选择+历史回归测试：
.venv-gpu/bin/python -m tools.run_road_comparison --prefix m4_road_repeat
# 原始本轮实验的先验/旧mask诊断；重训请相应替换run与model-metrics：
.venv/bin/python -m tools.eval_drivable_prior --run runs/m4_road_point --split val --model-metrics reports/m4_road_point_val/metrics.json --output reports/road_point_prior_repeat.json
.venv-gpu/bin/python -m tools.eval_legacy_road_mask --run runs/m4_road_point --split val --output reports/road_point_legacy_repeat.json
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q
```

所有原始录制、overlay、训练和评价目录拒绝覆盖；重复命令请改输出名。静态信号实验的41帧是接口/状态切换验收，不满足M1原先30秒录制门槛，也不是独立驾驶数据扩容。

相机采集请使用现有[服务器启动与清理流程](M0_M2_RUNBOOK.md)。训练与CARLA均使用同一张4090，本轮按先采集、停止专用服务、再训练的顺序执行。运行报告中的`peak_allocated_gib`只包含PyTorch分配器；不等于整个GPU进程或CARLA的总显存。

实施修正：首次灯态录制因NumPy布尔值JSON序列化失败，0帧提交，保留失败目录；修复为原生bool后使用`m4signal_v2_*`。Town07、Town06均不在服务器安装地图列表，保留失败日志；规则诊断测试采用Town02，独立于Town01/04规则录制，但Town02已用于道路训练，绝不称新地图视觉盲测。

绘图栅格化会覆盖仅与条带边缘相触的格子；第二版overlay采用凸多边形内部测试，要求BEV格中心在连续条带内。模型使用`road-overlay-v2-exact`，旧候选overlay不覆盖。与显式非道路标签冲突时保留Unknown，不强行覆盖为道路。新增格的地图查询一致性及改标统计见`reports/m4_road_overlay_exact_audit.json`。

地图抽查覆盖11场景、每场景3帧，新增正格的抽样中心查询一致性全部100%；两种方法仍共享OpenDRIVE，不算独立语义标注验收。Town05抽查中有1180个旧有效格因高度歧义变为Unknown，其余抽查场景的旧有效标签一致。这个差异由单独的旧v1掩码重评隔离。
