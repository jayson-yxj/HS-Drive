# M4.7 实施与复现

基于 `e9e2e02097a77123138c3b802909c9111faf11b4`，分支 `feat/m47-label-assurance-dynamics`。全部新逻辑独立放在 `m47_*.py`，旧默认行为不替换。质量结论：[报告](../reports/m47_summary.md)、[门禁](../reports/m47_acceptance.json)。

## 模块和合同

| 模块 | 用途与范围 |
|---|---|
| datasets/m47_assurance.py | 仅将有冲突证据的Free格置Unknown，不膨胀周围Unknown |
| tools/m47_false_free_audit.py | 复现104个逐帧冲突，重建原stride16雕刻路径、首surface截断、入口/出口和留出像素 |
| tools/m47_collision_audit.py / m47_attribute.py | CARLA保存位姿重建及cast_ray补充几何证据；与渲染记录联结 |
| tools/m47_capture.py / m47_geometry_queries.py | 六摄RGB/Depth/Instance、ego-removed参考、固定随机种子的碰撞射线和垂直查询 |
| tools/m47_quality.py / m47_local_quality.py | 评估冻结生成器，再单独保存冲突隔离版本；距离/高度/局部边界统计 |
| tools/m47_vehicle_dynamics.py / m47_fit_dynamics.py | 46组开环系统辨识、物理配置与有符号速度、标定表及高速度回归 |
| simulator/m47_motion.py | 独立标定控制器、TRACK/COAST/HOLD/RELEASE/EMERGENCY、连续平移SAT及保守旋转padding |
| simulator/m47_road.py | 未采用的道路走廊预计算候选，有边界回退；不是精确道路Mesh |
| tools/m47_stopping.py | 同物理配置、路线、起速和障碍的M46/M47闭环对照 |
| tools/m47_profile.py / m47_road_profile.py | 保存状态的组件profile、连续碰撞压力测试、走廊假接受审计 |
| tools/m47_gpu_contract.py | 旧RGB模型CUDA前向、一次临时Head backward、Unknown/语义监督梯度验证；无optimizer |
| tools/m47_acceptance.py / m47_visualize.py | 分离门禁、原始测量统计和104张三维冲突图 |

BEV保持 `[4,160,120]`、XY=0.5m、Z边界[-2,-0.5,1,2.5,4]，HS X前/Y左/Z上。Free仅是射线观测证据，有限射线不能证明整格空闲。Ground是Surface语义；Obstacle与Unknown的mask独立。模型只接原LearnedInputs，侧车数据只供监督/诊断；不修改Depth Bin或默认Splat。

Controller中的沿程零油门距离近似来自测得减速度趋势（0.2+0.4v），高速度前馈由实测线性拟合得到，**低速仍明显不符合这个模型**。最终版本为保护停车间距，在剩余目标距离≤0.4m且速度<1.2m/s时允许进入HOLD；这违反预设0.12m/s舒适Hold门禁，明确作为失败条件记录，不能称自然停车。源码对这一有限策略保留完整版本/失败探针，不作为默认控制器。

连续SAT仅对分段线性中心运动、固定朝向矩形精确；旋转用角点位移上界padding保守覆盖。没有精确旋转Mesh swept volume、复杂意图或完整路权保证。道路仍使用原角点map检查；路线走廊候选未被启用。未采用未经验证的多频率控制。

## 环境

沿用 `.venv`（CARLA0.9.15）和 `.venv-gpu`（PyTorch2.8+cu128），没有改依赖、驱动或车型物理参数。相应requirements同M46。GPU矩阵逆需要已有 `training.runtime.prepare_cuda()` 的绝对库路径加载；已接入新GPU检查脚本，早期失败保留。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
git branch --show-current
nvidia-smi
.venv/bin/python -c 'import carla; c=carla.Client("127.0.0.1",2100); c.set_timeout(30); print(c.get_server_version(),c.get_world().get_map().name)'
```

本轮实际使用端口2100的CARLA/Epic/offscreen。只有端口空闲时另终端启动，不同时运行两个tick owner：

```bash
runuser -u ubuntu -- bash /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

## 已执行命令与不可覆盖规则

下列是本轮实际入口与产物路径。默认路径已经存在，脚本会拒绝覆盖。复跑使用相应 `--output`/`--artifacts`/`--labels` 新路径；不提供参数的分析汇总入口应在独立工作树中按同一路径复现。CARLA采集和闭环顺序执行；离线CPU分析可以独立运行。

```bash
.venv/bin/python -m tools.m47_false_free_audit
.venv/bin/python -m tools.m47_collision_audit
.venv/bin/python -m tools.m47_attribute
.venv/bin/python -m tools.m47_vehicle_dynamics --artifacts reports/m47_dynamics_v3_artifacts
.venv/bin/python -m tools.m47_fit_dynamics
.venv/bin/python -m tools.m47_capture --output /root/gpufree-data/hs-drive-data/raw/m47_development --map Town03 --spawn 100 --vehicle vehicle.tesla.model3 --role development
.venv/bin/python -m tools.m47_capture --output /root/gpufree-data/hs-drive-data/raw/m47_heldout_town02 --map Town02 --spawn 10 --vehicle vehicle.audi.a2 --role heldout
.venv/bin/python -m tools.m47_capture --output /root/gpufree-data/hs-drive-data/raw/m47_heldout_town02_tesla --map Town02 --spawn 35 --vehicle vehicle.tesla.model3 --role heldout
.venv/bin/python -m tools.m47_quality --raw /root/gpufree-data/hs-drive-data/raw/m47_development /root/gpufree-data/hs-drive-data/raw/m47_heldout_town02 /root/gpufree-data/hs-drive-data/raw/m47_heldout_town02_tesla
.venv/bin/python -m tools.m47_local_quality
.venv/bin/python -m tools.m47_stopping --scenarios static_8_v2 static_15_v4 static_30_v5 static_45_v4 lead_braking multi_actor emergency timeout invalid missing --output reports/m47_stopping_verified.json --artifacts reports/m47_stopping_verified_artifacts
.venv/bin/python -m tools.m47_stopping --scenarios straight mild_curve junction_turn --replay-routes reports/m45_oracle_driving_artifacts --green-lights --output reports/m47_routes_verified.json --artifacts reports/m47_routes_verified_artifacts
.venv/bin/python -m tools.m47_profile --output reports/m47_latency_verified.json
.venv/bin/python -m tools.m47_road_profile
.venv-gpu/bin/python -m tools.m47_gpu_contract
.venv-gpu/bin/python -m tools.m47_visualize
.venv/bin/python -m tools.m47_hold_replay
.venv-gpu/bin/python -m tools.m47_visualize --stopping-only --output reports/m47_figures_reviewed
.venv/bin/python -m tools.m47_acceptance
```

无需重新采集即可复现的命令示例（新输出）：

```bash
.venv/bin/python -m tools.m47_false_free_audit \
  --output reports/m47_false_free_rerun.json \
  --artifacts reports/m47_conflict_rerun_artifacts \
  --labels /root/gpufree-data/hs-drive-data/processed/m47_quarantine_rerun
.venv/bin/python -m tools.m47_vehicle_dynamics \
  --output reports/m47_dynamics_rerun.json --artifacts reports/m47_dynamics_rerun_artifacts
.venv/bin/python -m tools.m47_stopping \
  --scenarios static_15_v4 static_45_v4 emergency \
  --output reports/m47_stopping_rerun.json --artifacts reports/m47_stopping_rerun_artifacts
.venv-gpu/bin/python -m tools.m47_gpu_contract --output reports/m47_gpu_rerun.json
```

原始数据在 `/root/gpufree-data/hs-drive-data/raw/m47_*`；修正104格后的标签位于 `processed/m47_quarantined`，新场景独立评测后隔离的标签位于 `processed/m47_reserved_labels`。它们是诊断数据，不能绕过门禁成为正式训练集。Town02留出场景只评估冻结M46规则，没有根据结果调规则；后续若用这些样本开发，就必须另留新的未见场景。

## 测试与门禁复现

```bash
.venv/bin/python -m pytest -q -ra --junitxml=reports/m47_sim_rerun.xml
.venv-gpu/bin/python -m pytest -q -ra --junitxml=reports/m47_gpu_rerun.xml
.venv-gpu/bin/python -m pytest -q tests/test_m47_assurance_motion.py --junitxml=reports/m47_module_rerun.xml
.venv/bin/python -m tools.m47_acceptance --output reports/m47_acceptance_rerun.json --metrics reports/m47_metrics_rerun.json
.venv/bin/python -m tools.m47_delivery_integrity --output reports/m47_delivery_integrity_rerun.json
```

门禁阈值在 `configs/m47_acceptance.yaml`，于诊断pilot之后、最终验证之前固定。舒适性、间距、Hold速度、反向滚动、低速时间、实时性分别判断。Emergency不受舒适门禁限制。汇总使用verified结果；早期 `m47_stopping_validation_first.json`、pilot及动力学暖机失败保留，不参与最佳结果挑选。

主要实验和测试的stdout在 `reports/m47_execution_logs`，闭环和动力学逐帧JSONL包括真实仿真时间、速度、加速度、位姿、命令与Physics配置。最终闭环在启动时保存源码副本和SHA并校验运行期间未变。原始warmup试验失败、低速滑行不停车、路口失败都保留。动力学交付源码只在最后修正了暖机注释，运行源码仍以 `m47_dynamics_v3_artifacts/source` 快照为准。

## 未执行项

标签质量未过，因此不执行以此为前提的1–4帧overfit/位置先验训练对比；CUDA一次backward不等于可学习性验证。没有大型联合训练、没有修改Depth Bin、没有默认Gated Splat。未通过的控制器、道路走廊和实时性不自动替代旧默认。所有质量风险记录为Partial。

最终图复核发现M46没有记录`mode`，原始停车图的HOLD面板不能使用。`m47_figures_reviewed`使用`m47_hold_replay`从冻结控制器和已记录输入重建M46内部状态，图注明为离线重放；M47状态直接读取日志。原图保留，动力学指标不受此绘图问题影响。交付源码与运行快照的对应关系见`reports/m47_delivery_integrity.json`。

前车制动对照首帧存在0.125m间距和前车初速度差异，虽然自车起速及物理一致，该组只能描述结果、不能视为严格受控改善证据。所有静态/多Actor停车对照初始间距一致。下一次动态前车实验需明确同步确认恒速生效并固定起始状态后再录制。
