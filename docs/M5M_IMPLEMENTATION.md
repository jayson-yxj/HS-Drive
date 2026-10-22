# M5-M 实施与复现

环境：RTX4090 24GB，PyTorch2.8/CUDA12.8，CARLA0.9.15 Epic，640×384六摄。GPU程序使用`.venv-gpu/bin/python`，CARLA采集使用`.venv/bin/python`。本轮是静态接触/感知及Shadow资源诊断，不是视觉驾驶闭环。

所有新输出以m5m命名。大数据、完整逐Actor记录、模型缓存和失败日志保留服务器；Git交付源码/配置/精简报告/图/SHA索引。工具默认拒绝覆盖；复现需新工作目录和数据输出根，不删除本轮资产。

## 冻结与历史审计

```bash
git switch -c feat/m5m-selective-object-geometry f2f4e1a
.venv-gpu/bin/python -m tools.m5m_prepare
.venv-gpu/bin/python -m tools.m5m_contact_audit --split train
.venv-gpu/bin/python -m tools.m5m_contact_audit --split val
.venv-gpu/bin/python -m tools.m5m_sequence --phase regression
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5m_collision_audit
.venv-gpu/bin/python -m tools.m5m_reference_summary
.venv-gpu/bin/python -m tools.m5m_selective_eval
.venv-gpu/bin/python -m tools.m5m_residual_gating --split val
.venv-gpu/bin/python -m tools.m5m_gating_decision
.venv-gpu/bin/python -m tools.m5m_residual_gating --split regression
.venv-gpu/bin/python -m tools.m5m_residual_gating --split test
.venv-gpu/bin/python -m tools.m5m_ground_report
.venv-gpu/bin/python -m tools.m5m_dense_object
```

`contact_audit`保留原生关联实例表面，与旧侵蚀可靠mask分母并列。历史M5-L相同公式不改；主指标只在同一RGB候选和底部射线上比较Expected/Mode/Plane。`m5m_selective_complete.json`补齐原生匹配对象的Dense结果，旧初始报告保留作审计。

Ground局部平面`z=ax+by+c`做确定性截尾最小二乘，退化支持拒绝；不可把可见底部端点称精确轮胎接地。`collision_audit`重载静态地图查询历史摆放位置，动态Actor未重建，不是原始渲染Mesh真值。

## 有限接触开发采集与封存

```bash
.venv/bin/python -m tools.m5m_allocate --phase dev
.venv/bin/python -m tools.m5m_capture --protocol configs/m5m_capture_dev.json --output /root/gpufree-data/hs-drive-data/raw/m5m_dev_v3
.venv-gpu/bin/python -m tools.m5m_dev_process
.venv-gpu/bin/python -m tools.m5m_anatomy_audit --phase dev
.venv-gpu/bin/python -m tools.m5m_pretest
.venv-gpu/bin/python -m tools.m5m_sealed_pipeline
```

每组9配置（空场景、小/大车辆、行人、锥桶、遮挡、左右方向及移除），4个开发位置、4个新封存位置，距已有采集/原M4位置至少45m。开发集2 Train位置、2 Val位置，用于接触可行性，不足以仅按数量批准Contact Head。封存位置在模型、阈值和源码冻结后按地图几何分配；不使用RGB测试表现挑位置。

新采集记录Actor姿态、多个道路碰撞探针、轮胎物理位置半径和脚部骨骼；车辆从道路候选处物理沉降后冻结。物理轮胎半径不是渲染轮胎接触斑；骨骼原点不是鞋底，均不自动成为Gold。同步RGB/Depth/Instance及input/监督sidecar隔离沿用已有合同。

首次和第二次接触采集触发CARLA Signal11（失败目录`m5m_dev`、`m5m_dev_retry`）。独立API探针正常，日志定位到冻结车辆后`get_physics_control()`；修复为冻结前缓存。成功采集版本使用`m5m_dev_v3`，失败日志和已生成文件均保留。此操作问题不应混称Town03渲染质量失败。API和参考语义参见[CARLA官方Python API](https://carla.readthedocs.io/en/0.9.15/python_api/)。

## Shadow与测试

CARLA运行时不得同时执行其他GPU实验或其他TickOwner：

```bash
.venv-gpu/bin/python -m tools.m5m_online_shadow
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_imaging_rig.py tests/test_snapshot.py tests/test_session.py tests/test_geometry_extended.py
.venv-gpu/bin/python -m tools.m5m_delivery_reports
.venv-gpu/bin/python -m tools.m5m_figures
```

Worker首次推理单列；模型加载期间主进程继续tick并记录初始化时序。阶段切换的Worker停机等待单列，不计入阶段稳态；不能据此声称驾驶热插拔无停顿。主进程队列调用、输出drain及tick分别计时；后台feeder序列化未被精确单独计量。Worker只收RGB和合法标定，不创建CARLA客户端、不输出控制。250ms过期输出丢弃，Unknown不作Free。

## 指标与数据限制

- Actor-view高度相关，独立支持是位置组；严禁将像素数/帧数作独立场景数。
- 二值前景分类不能直接分Vehicle/Pedestrian/Cone；类别只来自评价标签，模型`class_hint`保持未知。
- 已知可见Actor分母受实例与AABB关联覆盖限制；无匹配候选不必然是假物体。
- False-accept不可信接触缺少Gold二元真值，值为null；另报接受但未验证的比例，不混称真实接触错误率。
- Unknown深度数值可作异常诊断，不能进入可靠误差分母或安全负样本。
- 残差门控是归一化概率混合，原始HeightSplat不变；Oracle类别门控只作上限诊断。
- Empty/Removed仅按manifest空配置评估；ego-removed深度只用于自车监督参考，不当作障碍移除场景。
- RGB局部平面来自旧Depth预测及目标下方非前景区域，未验证为道路法向，不能升级定位状态。

最终`m5m_artifact_manifest.json`、`m5m_integrity.json`记录源码/数据/模型/日志SHA及历史资产复核。源码提交哈希以Git最终提交为准；实验前源文件哈希在pretest与各报告，避免报告自引用哈希。

## 封存配对质检中断

封存36帧有2帧在ego-removed参考重置时车辆姿态差超出0.001矩阵元素容差，最大0.262m量级。捕获完整性检查明确失败，失败日志`capture_integrity_sealed_early.log`保留。数据转换完成后、任何新封存GPU预测前，按预声明配对容差剔除这2帧；不调阈值、不补挑场景。原未过滤pixel索引保存`index_unfiltered_preserved.json`，全部原始图像保留。主封存评价只有34帧，数据验收保持Partial。

因此实际恢复流程为：

```bash
.venv-gpu/bin/python -m tools.m5m_sealed_resume
# 必须等Shadow试验完成再启动GPU推理
.venv-gpu/bin/python -m tools.m5m_new_eval --phase sealed
.venv-gpu/bin/python -m tools.m5m_anatomy_audit --phase sealed
.venv-gpu/bin/python -m tools.m5m_sealed_ground
.venv-gpu/bin/python -m tools.m5m_runtime_diagnostics
.venv-gpu/bin/python -m tools.m5m_delivery_reports
.venv-gpu/bin/python -m tools.m5m_figures
.venv-gpu/bin/python -m tools.m5m_assemble
.venv-gpu/bin/python -m tools.m5m_results_corrections
.venv-gpu/bin/python -m tools.m5m_integrity
```

`m5m_sealed_pipeline`保留首次运行入口；遇到配对门禁失败应使用上述明确的审计/恢复流程，不盲目重跑或覆盖。`m5m_run_finish`适用于无配对失败且尚未运行Shadow的全新复现，不能用于覆盖本次成功输出。

后处理复现：`tools.m5m_historical_audit`、`tools.m5m_residual_attribution`、`tools.m5m_ipc_probe`、`tools.m5m_anatomy_figures`、`tools.m5m_run_finish`、`tools.m5m_results_corrections`、`tools.m5m_integrity`。格式化后旧实验报告源SHA不同于最终文件时，pretest前哈希代表当时源码；格式化不改变数值方法。
