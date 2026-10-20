# M4.5 工程与复现

研发分支 feat/m45-geometry-foundation，稳定基线0f8eab324302d35d65095d097d63a3962b055481。最终源码、报告与验收由本文件所在Git commit标识。环境记录 reports/m45_environment.json 中 git_status 包含刚写出的 freeze文件；进入任务时旧工作区无修改。main不合并。

## 环境

工作目录 /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0；RTX4090 24564MiB，driver580.126.09，Python3.10.12。原 .venv-gpu 为torch2.8.0+cu128，原 .venv 为CARLA0.9.15。保留两个环境；本轮仅给模拟环境安装networkx3.4.2，见 requirements-m45-sim.txt。不需要自定义CUDA。

先检查本机再运行；以下用于已部署服务器。缺少环境时参照原 docs/server_env.md、docs/M4_BASELINE.md 和 requirements-{sim,train}.lock.txt 创建对应Python3.10 venv。不要为复现替换现有环境。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
git branch --show-current
nvidia-smi
.venv/bin/python -c "import carla; print(carla.__file__)"
.venv-gpu/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
.venv/bin/pip install -r requirements-m45-sim.txt
.venv-gpu/bin/pip install -r requirements-diagnostics.txt
```

GPU初始化统一调用原 training.runtime.prepare_cuda。NVML返回host PID，容器内 os.getpid 不同；完整原实验的process memory null不能当0。追加probe通过唯一新增host PID测进程显存。

## 代码入口

| 模块 | 文件/入口 | 输入输出 |
|---|---|---|
| Splat | diagnostics/splat_confidence.py | 同原HeightSplat，height [B,C,4,160,120]、Mass [B,4,160,120] |
| Ray traversal | geometry/ray_tracing.py | origin/end与四层网格→有序flat ranks |
| Occupancy | datasets/occupancy_labels.py | GT rays→独立三态/语义/障碍/valid/count/conflict |
| Temporal | geometry/temporal_warp.py | previous[B,C,X,Y]、prev/current世界位姿→aligned、valid、forward transform |
| Memory | BEVMemory | hidden、pose、timestamp、scene、valid；align失败返回reset reason |
| Oracle | simulator/oracle_driver.py | privileged state/route→[8,4]轨迹→安全原因→控制 |
| 配对采集 | tools.capture_m45_semantic_pairs | 同位姿6×RGB/Depth/Semantic→immutable raw记录 |

每个生成工具拒绝覆盖现有输出；复现请使用新的 m45_ 路径，以下使用 _repro。新原始数据在 /root/gpufree-data/hs-drive-data/raw/m45_semantic_pairs；标签在 processed/m45_occupancy_*。Git包含代表NPZ/manifest/图，完整新原始数据留在服务器，旧数据/模型不随分支复制。

## GPU诊断和测试

先停止本轮启动的CARLA进程（只操作自己的PID），确保不与GPU诊断同时运行。所有诊断为inference_mode，未创建optimizer。

```bash
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q
.venv-gpu/bin/python -m tools.benchmark_splat_confidence --output reports/m45_splat_comparison_repro.json --artifacts reports/m45_splat_figures_repro
.venv-gpu/bin/python -m tools.m45_splat_probe --output reports/m45_splat_oracle_probe_repro.json
.venv-gpu/bin/python -m tools.validate_temporal_warp --output reports/m45_temporal_validation_repro.json --figures reports/m45_temporal_figures_repro
.venv-gpu/bin/python -m tools.validate_m45_static_surfaces --output reports/m45_temporal_static_surfaces_repro.json
.venv-gpu/bin/python -m tools.summarize_m45 --prefix reports/m45_repro
.venv-gpu/bin/python -m tools.replay_oracle_log --output reports/m45_oracle_replay_repro.json --figure reports/m45_oracle_paths_repro.png
```

汇总工具读取已提交的m45原实验结果；--prefix仅指定新汇总输出。完整重做推理使用上面的benchmark，源码中默认输入路径、配置和哈希均保留，不能把汇总工具输出误当新模型实验。

原全仓GPU133 passed/5 skipped（无CARLA），CPU98 passed/12 skipped（无torch）；新增隔离测试后M45模块29 passed/0 skipped。三份JUnit保存skip原因，数量有重叠不可相加。不是以跳过项冒充测试通过。

## Occupancy生成

```bash
.venv/bin/python -m tools.build_occupancy_oracle --scene /root/gpufree-data/hs-drive-data/raw/m45_semantic_pairs/clear --output /root/gpufree-data/hs-drive-data/processed/m45_occupancy_clear_repro --stride 16 --frames 3
.venv/bin/python -m tools.build_occupancy_oracle --scene /root/gpufree-data/hs-drive-data/raw/m45_semantic_pairs/fog --output /root/gpufree-data/hs-drive-data/processed/m45_occupancy_fog_repro --stride 16 --frames 3
```

旧Town04输入源的精确路径保存在 reports/m45_occupancy_figures/m45_occupancy_legacy_town04_manifest.json。schema、Mask、语义编码和训练限制见 M45_OCCUPANCY_CONTRACT.md。生成CPU工具验证源depth/semantic hash并写label sha256；旧录制没有语义时保持未知。

## CARLA采集、闭环与重跑

先在一个终端以非root用户启动独立2100端口。当前服务器已验证 ubuntu用户与该安装路径可用：

```bash
runuser -u ubuntu -- bash /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

另一个终端运行：

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv/bin/python -m tools.capture_m45_semantic_pairs --port 2100 --frames 8 --output /root/gpufree-data/hs-drive-data/raw/m45_semantic_pairs_repro
.venv/bin/python -m tools.run_oracle_closedloop --output reports/m45_oracle_driving_repro.json --artifacts reports/m45_oracle_driving_artifacts_repro
.venv/bin/python -m tools.run_oracle_closedloop --scenarios invalid_trajectory --replay-routes reports/m45_oracle_driving_artifacts --output reports/m45_oracle_live_replay_repro.json --artifacts reports/m45_oracle_live_replay_artifacts_repro
```

runner校验/加载Town03；读取安装包PythonAPI/carla/agents/navigation/global_route_planner.py（哈希保存在报告）。--replay-routes校验地图、spawn、保存路线，重新执行固定注入场景。不存在逐比特仿真承诺；offline replay可精确验证记录状态上的控制输出。未来失败场景用同一日志/路线机制复现。完成后停止自己启动的2100服务器，释放显存。

## 已保存结果

reports/m45_summary.md 为结论入口；m45_acceptance.json区分Passed/Partial。m45_splat_comparison.json是222帧原始实验；m45_analysis_metrics.json为聚合；m45_splat_oracle_probe.json为6帧补充。Temporal以m45_temporal_validation_v2.json为较完整记录，第一版不删；另有实际表面配准。Oracle原9场景、controller replay、重启后live replay分别保留。

logs/m45_* 含运行输出和保留的失败尝试；关键日志副本随reports/m45_execution_logs提交（排除大体积UE4日志，仍保留其SHA）。新命令不会修改旧M4报告/模型/index。m45_asset_integrity.json核对813项SHA、5170项index JSON、26项raw manifest及65657个原数据文件size/mtime；原raw payload没有全部重算SHA，不夸大为逐字节全量验证。

最后的只读历史资产复核命令：

```bash
.venv-gpu/bin/python -m tools.audit_m45_assets --output reports/m45_asset_integrity_repro.json
```
