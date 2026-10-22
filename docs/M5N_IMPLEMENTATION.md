# M5-N 实施与复现

RTX4090 24GB，CARLA0.9.15 Epic；GPU环境`.venv-gpu/bin/python`，SDK环境`.venv/bin/python`。所有新资产m5n前缀；历史文件/模型/数据只读。工具拒绝同名输出覆盖，复现须新工作副本和新数据根。完整源/资产SHA见`reports/m5n_artifact_manifest.json`，Git最终提交是源码版本。没有训练新的Depth/Occupancy/Temporal，没有车辆控制。

## 顺序执行

```bash
git switch -c feat/m5n-direct-actor-localization 29ad67a
.venv-gpu/bin/python -m tools.m5n_prepare
.venv-gpu/bin/python -m tools.m5n_cache --split train
.venv-gpu/bin/python -m tools.m5n_develop
.venv-gpu/bin/python -m tools.m5n_eval --split val
.venv-gpu/bin/python -m tools.m5n_eval --split regression
.venv-gpu/bin/python -m tools.m5n_reference_audit
.venv-gpu/bin/python -m tools.m5n_freeze
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv-gpu/bin/python -m tools.m5n_sealed
```

`develop`依次缓存Val→支持门禁→四臂(Direct/Prior×101/102)600步→历史Regression缓存。各步日志在`runs/m5n_logs`；`training.log`记录真实损失/梯度/选择，Checkpoint在`runs/m5n_training`。缓存`features.pt`只有RGB特征；GT中心/关联/类别在独立`supervision.json`，匹配发生在监督路径。未匹配候选保留，不当成负样本。

每臂按固定种子保存完整样本顺序；每步按位置均匀抽样32候选。模型层冻结，只有Head优化。Direct40,643参数，geometry_prior3,779参数；后者是RGB候选几何对照，不能称完全无RGB的PositionPrior。Z和偏移回归各自Loss、峰值显存和耗时在训练报告。

## 新封存采集

4个几何分配新位置，距历史配置/原M4位置≥45m；每位置13配置（Empty、5/15/30m车辆、侧后方、近远行人、遮挡、锥桶、边缘与合并、Removed）。不按Test图像挑样本。采集在方法/模型/校准锁定后进行，旧M测试只回归。

新Actor参考使用`world.get_snapshot().find(id)`而不是独立时刻Transform读取，记录帧号和时间戳及局部Box参数；配对ego-removed恢复已保存目标姿态，不二次沉降。`m5n_capture_audit`先验证同步/配对，再生成标准索引与像素sidecar。CARLA参考仍是仿真定义，不是精确Mesh或接触Gold。

## 离线Runtime及报告

CARLA采集/封存GPU推理结束后，停止本轮启动的CARLA进程，再执行无共享渲染负载的离线预算：

```bash
.venv-gpu/bin/python -m tools.m5n_runtime
.venv-gpu/bin/python -m tools.m5n_eval --split train
.venv-gpu/bin/python -m tools.m5n_diagnostics
.venv-gpu/bin/python -m tools.m5n_report
.venv-gpu/bin/python -m tools.m5n_figures
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_m5n_sdk.py tests/test_imaging_rig.py tests/test_snapshot.py tests/test_session.py tests/test_geometry_extended.py
.venv-gpu/bin/python -m tools.m5n_finalize
```

`DirectActorCandidate(enable=True)`仅用于Offline研究，前向只接受已有六RGB输入白名单。ActorGT/DepthGT/过期输入实际拒绝。输出无Throttle/Brake/Steer，源码AST检查无CARLA和控制调用。Runtime测量预加载GPU的完整Mobile+连通域+Head+几何+校准，不含磁盘/H2D/IPC。M5-M Runtime Partial继承，本轮不重复在线静态Shadow。

## 评价解释

全部已知可见Actor-view分母保留漏检与无法定位；误差在有位置子集报告，未定位不记0。逐类别/相机/距离/遮挡代理/坡道分层。固定覆盖P95、错误上限覆盖、同目标比较和校准相关性见`m5n_risk_coverage.json`。Test曲线只作评价，不用于选阈值或继续训练。

可见性为实例关联/投影Box面积等代理，非精确物理遮挡真值。无匹配候选可能是未记录真实物体，物理假目标数标null。误差参考Box中心不等于可见表面中心，Dense摘要不是天然BoxCenter估计器。

当前6位置Val同时用于Checkpoint选择和后校准；虽有留一位置校准分数，仍不能视为完全独立Conformal保证。样本相关，严格独立单位是位置而非Actor-view/像素。

## 失败记录

首次参考汇总使用重复`split`关键字而失败，阻止pretest freeze/新Test启动；修复仅报告字段为`split_candidate_count`后继续。失败日志均保留，训练结果未重跑、预算未增加、旧文件未覆盖。原M5-M两帧配对失败在参考报告中单独讨论，继续从主学习/评价排除；没有为了扩大样本自动使用它们。

补充可复现检查：`tools.m5n_inference_equivalence`验证完整RGB路径与缓存评测位置差小于2mm；`tools.m5n_sealed_reference_detail`验证显式Box局部参数；`tools.m5n_supplement`生成类别/Empty/置信度图。以上均不训练或选择模型。运行`m5n_finalize`前完成这些后处理。
