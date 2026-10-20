# M4.6 实施与复现

基线提交 `129a4131567ac02edf88b19b9199cad979ce3d72`；开发分支 `feat/m46-occupancy-motion-quality`。结论以 [最终报告](../reports/m46_summary.md) 和 [验收 JSON](../reports/m46_acceptance.json) 为准。旧模型、Depth Bin、默认 Splat、旧报告和旧数据均未修改。没有启动网络训练，也没有合并 main。

## 接口和标签合同

- 新生成器 `datasets/occupancy_quality.py` 复用旧 `build_observed` 的网格遍历，但把整射线 AABB 拒绝替换为逐像素可见性认证；旧生成器保留作为对照。
- 六摄固定世界位姿，采集同步 RGB/Depth/Instance 三种流，再移除自车采集同位姿参考。以车体范围内消失的像素确定渲染实例，拒绝自车轮廓及 2px 外缘。**渲染实例编号不假设等于 actor.id**。穿过粗自车盒的射线必须满足深度差≤0.05m、相同语义/实例、3×3 稳定支持；否则 Unknown。
- 车体盒相交的粗体素仍为 Unknown。恢复射线只在实际 ego-present 深度首个命中前提供 Free 证据，不把移除自车后的可见地面用于穿透自车雕刻。自车遮挡射线自身不贡献 Free；多摄像头看到同一全局体素的独立有效证据允许保留。
- `[Z,X,Y]=[4,160,120]`、HS X前/Y左/Z上，范围及高度层与 M4.5 完全相同。state=0 Unknown/1 Free/2 Surface Occupied。Ground 是物理表面，不能当障碍；Obstacle 正类为静态/可运动表面，Unknown/Other/语义冲突不参与负类监督。Surface Count 是命中计数。
- Dense 版本只补 stride2 的可靠表面，不添加 Free 射线；表面周围同高度一格 XY 的 Free 改 Unknown，保留不确定性。stride1 交错像素作为额外审核，不参与 dense endpoint 生成。这不是独立几何真值，且粗体素整体空闲未获证明。
- `OccupancyDataset` 保持 LearnedInputs 白名单；`supervision=False` 不读取 NPZ 标签。新索引保持同路线/位姿所有变体在同 split：spawn45 共9帧 val，其他开发位置27帧 train；3帧脚本动态只做诊断。不是盲测集。
- `observed_occupancy_loss` 接受 `[B,2,Z,X,Y]` logits，只训练 Free/Surface；Unknown 忽略。当前只跑零 logits 梯度检查，没有构建 Occupancy 网络或 optimizer。
- `MotionPlanner` 返回原 `Trajectory` 合同，dt=0.1s、60点。速度积分产生位置，规划加速≤1.5、减速≤1.8m/s²、Jerk≤2.5m/s³；曲线 chord 与弧长有0.025m一致性容差。停止目标为路线弧长 bumper gap 减3m，终点和障碍使用同一 remaining_m 传给 Controller。独立 Safety 检查实际时间轨迹和同尺寸 footprint。
- Safety margin=纵向0.2m/侧向0.15m；3m是规划站停间距，不是碰撞膨胀半径。离散检查≤0.1m/0.05s，非连续碰撞检测。policy中的 emergency_decel 目前是保留字段，不能解释为车辆制动能力保证。
- `align_visible_history` 使用每高度独立可见性和可运动表面 mask；当前诊断 mask 来自 Oracle。没有把 GT mask 接入学习推理，也没有训练时序网络。

## 已有服务器环境

Python3.10；`.venv` 含 CARLA0.9.15、numpy、networkx；`.venv-gpu` 含 PyTorch2.8.0+cu128、matplotlib。依赖沿用 `requirements-sim.lock.txt`、`requirements-train.lock.txt`、`requirements-m45-sim.txt`、`requirements-diagnostics.txt`。未改变现有虚拟环境包。格式化使用独立 `/tmp/m46_format_tools`，不是运行依赖。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
git branch --show-current
nvidia-smi
.venv/bin/python -c 'import carla; c=carla.Client("127.0.0.1",2100); c.set_timeout(30); print(c.get_client_version(),c.get_server_version(),c.get_world().get_map().name)'
```

本轮已实际使用2100端口的 Town03/Epic/offscreen 服务。仅在该端口没有服务时另开终端启动；不要启动第二个 tick owner：

```bash
runuser -u ubuntu -- bash /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

## 重用现有数据复现

以下命令采用新路径，脚本遇到已有输出会拒绝覆盖。再次执行请更换 rerun 后缀。原始数据205MB、dense标签约1.5MB，位于数据盘；代表 NPZ/图片、生成报告与数据 SHA 保存在仓库。

```bash
.venv/bin/python -m tools.build_m46_occupancy \
  --raw /root/gpufree-data/hs-drive-data/raw/m46_occupancy \
  --output /root/gpufree-data/hs-drive-data/processed/m46_sparse_rerun \
  --report reports/m46_sparse_rerun.json
.venv/bin/python -m tools.refine_m46_occupancy \
  --source reports/m46_sparse_rerun.json --surface-stride 2 \
  --output /root/gpufree-data/hs-drive-data/processed/m46_dense_rerun \
  --report reports/m46_dense_rerun.json
.venv/bin/python -m tools.validate_m46_labels \
  --source reports/m46_sparse_rerun.json --dense reports/m46_dense_rerun.json \
  --output reports/m46_labels_rerun.json
.venv/bin/python -m tools.index_m46_occupancy \
  --source reports/m46_sparse_rerun.json --labels reports/m46_dense_rerun.json \
  --output /root/gpufree-data/hs-drive-data/processed/m46_index_rerun \
  --report reports/m46_index_rerun.json
.venv-gpu/bin/python -m tools.validate_m46_loader \
  --index /root/gpufree-data/hs-drive-data/processed/m46_index_rerun \
  --output reports/m46_loader_rerun.json
.venv-gpu/bin/python -m tools.validate_m46_temporal \
  --source reports/m46_sparse_rerun.json --output reports/m46_temporal_rerun.json
.venv/bin/python -m tools.validate_m46_safety_sampling --output reports/m46_sampling_rerun.json
```

需要重新采集而非重建标签时，先运行：

```bash
.venv/bin/python -u -m tools.capture_m46_occupancy \
  --output /root/gpufree-data/hs-drive-data/raw/m46_capture_rerun
```

再把上述生成命令的 `--raw` 改成新目录。13场景×3帧，加13组ego-removed参考。动态 Actor 是每帧离散移动2m，不能称真实物理动态采集。

## 实际 CARLA 闭环复现

两个命令顺序执行；路线测试固定绿灯并恢复原灯态，用于排除交通灯周期对轨迹回归的干扰，不验证红灯/让行。

```bash
.venv/bin/python -u -m tools.run_m46_closedloop \
  --scenarios static_8_v2 static_15_v4 static_30_v5 static_45_v4 lead_braking multi_actor emergency timeout invalid missing \
  --output reports/m46_braking_rerun.json --artifacts reports/m46_braking_rerun_artifacts
.venv/bin/python -u -m tools.run_m46_closedloop \
  --scenarios straight mild_curve junction_turn \
  --replay-routes reports/m45_oracle_driving_artifacts --green-lights \
  --output reports/m46_routes_rerun.json --artifacts reports/m46_routes_rerun_artifacts
```

两版本相同路线、车型、暖机速度条件和 actor 放置，整个 Planner/Safety/Controller 栈不同；不是单变量消融。记录每帧轨迹、速度、控制命令、实际上一帧控制、障碍状态和事件，启动时保存源码副本与 SHA。前车前4s使用脚本恒速2.5m/s，之后解除恒速并物理制动；不是普通自然跟车泛化。静态障碍关闭物理，碰撞传感器仍启用。速度目标2/4/5m/s对应实测起速约2.015/3.837/4.756m/s。

## 测试、可视化和验收汇总

```bash
.venv-gpu/bin/python -m pytest -q -ra --junitxml=reports/m46_gpu_rerun.xml
.venv/bin/python -m pytest -q -ra --junitxml=reports/m46_sim_rerun.xml
.venv-gpu/bin/python -m pytest -q tests/test_m46_motion.py tests/test_m46_occupancy.py tests/test_m46_temporal_visibility.py tests/test_m46_loader.py --junitxml=reports/m46_module_rerun.xml
.venv-gpu/bin/python -m tools.visualize_m46_visibility --output reports/m46_visibility_rerun
.venv-gpu/bin/python -m tools.summarize_m46 \
  --driving reports/m46_closedloop_verified.json \
  --figures reports/m46_figures_rerun --output reports/m46_metrics_rerun.json
.venv/bin/python -m tools.audit_m46_delivery \
  --output reports/m46_acceptance_rerun.json --supplement reports/m46_supplement_rerun.json
```

`audit_m46_delivery` 汇总已经交付的 verified 输入，重算运动阶段 Jerk 和日志 SHA；不会把新跑但未指定的结果自动升级为正式证据。全部实际 stdout/JUnit 位于 `reports/m46_execution_logs/` 和对应 XML。早期 pilot/探针和路线失败结果完整保留，不能和最终 verified 结果混用。
