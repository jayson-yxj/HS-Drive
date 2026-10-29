# HS-Drive · 可复现自动驾驶研究

六摄纯视觉端到端自动驾驶项目。架构沿用 [HS-Drive V1.3 技术基线](docs/HS-Drive_Technical_Plan_v1.0.md)，代码覆盖 CARLA 采集、几何与数据接口、空间证据研究、Oracle 规划研究，以及隔离环境中的 PETR/StreamPETR 评测工具。

开发遵循 [短期功能分支与复现约定](CONTRIBUTING.md)。`main` 作为研究代码主线，合并代码不等于模型质量或驾驶安全通过。当前 [冻结感知对照](reports/m8j11_summary.md) 保留 PETR-A 为参考，StreamPETR 为候选，不改变默认运行模式。历史分支的 [一次性整合清单](reports/branch_consolidation_plan.md) 待用户确认后执行。

以下保留早期基线文档入口；后续研究环境和测试入口见 CONTRIBUTING，历史全量 bootstrap 测试不适用于所有隔离环境。

- [M4 远距离深度、跨地图 Drivable 与路口规则监督](reports/m4_far_rules.md)
- [远场与规则监督接口、实验协议和命令](docs/M4_FAR_ROAD_RULES.md)
- [M4 独立驾驶数据、深度对照与静态 Drivable 实测](reports/m4_expansion.md)
- [增量实现与复现命令](docs/M4_DRIVING_DEPTH_DRIVABLE.md)
- [M4 GPU与小样本实测（完整M4未关闭）](reports/milestone_04_baseline.md)
- [M4模型、训练和复现命令](docs/M4_BASELINE.md)
- [M3工程门禁与实测结果](reports/milestone_03.md)
- [M3数据索引、划分与训练接口](docs/M3_DATASET.md)
- [高画质采集验收结果](reports/imaging_validation.md)
- [训练前成像验证与采集配置](docs/IMAGING_VALIDATION.md)
- [当前M2门禁与证据](reports/milestone_02.md)
- [M2复现实验](docs/M2_ACCEPTANCE.md)
- [首轮历史记录](reports/milestone_01.md)
- [服务器检查](docs/server_env.md)
- [部署、采集、BEV 与验收命令](docs/M0_M2_RUNBOOK.md)
- [几何约定及实施决策](docs/geometry_decisions.md)

```bash
bash scripts/bootstrap_sim.sh
source .venv/bin/activate
python -m tools.record_dataset --help
python -m tools.visualize_oracle_bev --help
```

| 目录 | 责任 |
|---|---|
| `configs/` | 保留交接参数：六摄、CARLA、BEV、后续模型 |
| `simulator/` | 会话生命周期、相机、按 frame_id 同步、快照、事务式录制 |
| `geometry/` | 坐标、深度、反投影、4 层点数 BEV、Box 可视化 |
| `datasets/` | 数据契约、提交帧回放、校验、推理输入字段边界 |
| `models/` | ResNet18/FPN、Depth/Context、四层Lift-Splat与单帧BEV |
| `training/` | 监督损失、分段指标、检查点与CUDA环境处理 |
| `tools/` | 采集、校验、Oracle 可视化、已知平面比较、合成测试 |
| `tests/` | 几何、同步故障、数据完整性、清理测试 |
| `reports/` | 验收结果；`logs/` 保存原始运行日志 |

M0/M1/M2 几何与同步验收未通过前，不进入 Learned BEV 训练。Depth/Actor GT 仅用于监督与 Oracle 调试。空的点数格不是 Free。已有单帧 Depth/BEV 基线、独立驾驶扩容、路口静态道路与方向标签、交通灯停止义务监督接口；方向和信号预测网络尚未训练。可靠感知质量、完整交通规则 Drivable、Occupancy、Planner 与闭环控制尚未完成。

原始交接包的 `MANIFEST.sha256` 留作原包基准，当前新增/修改源码由运行清单的 `source_sha256` 记录；不要将原包校验清单当作开发后工程清单。
