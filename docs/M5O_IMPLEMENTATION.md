# M5-O 实施与复现

基线 `2afe740`，独立分支 `feat/m5o-perception-benchmark-reset`。所有 M5-O 输出以新路径写入；M4–M5-N 模型、数据、源码和报告由 `runs/m5o_logs/historical_freeze.json` 校验。报告状态以最终 `m5o_acceptance.json` 为准。

## 监督合同与评价分母

参考保持 M5-N：八个已变换 CARLA OBB 顶点均值是仿真 Actor Box Center，既不是 Actor 原点，也不是可见表面或 Mesh/接触中心。HS Ego 为 x 前、y 左、z 上；Camera Optical 为 x 右、y 下、z 前，K/T、帧号、时间戳随候选保存。

正式模型输入仍由 `datasets.inputs.INPUT_FIELDS` 白名单限定。缓存 features.pt 只有冻结 Mobile RGB 特征和合法标定/候选形状；监督在独立 supervision.json。Depth、Instance、Actor 盒仅用于监督和诊断。GT 实例区域只通过独立 `--oracle-candidates` 诊断开关生成，不能用作正式检测性能。

RGB 连通域阈值 0.9，至少 6 像素、高度至少 3；与几何关联的渲染实例 Hungarian IoU≥0.1。实例可见支持至少 3 原生像素，深度在 [1.5,80)m。渲染 ID 不假定等于 Actor ID；沿用 AABB±0.1m 内至少 90% 端点的关联代理，多 Actor 争用的渲染 ID 不纳入可靠参考。匹配、分裂、合并分别记录。新定位主评价只接受唯一关联，检测分母保留歧义与漏检，不用零误差填补。

所有方法使用同一参考和候选。Fixed Plane 保留旧条件地面交点估计，不增加 GT 类别或盒尺寸补偿；它与 Box Center 的语义差距明确保留。Dense Expected/Mode 来自相同 RGB 区域的旧 Depth 概率，天然偏向可见表面，不能改称原生目标中心模型。

未匹配候选分为目标语义表面、Static/Dynamic 未知、背景语义代理和混合解释。背景杆、墙也可能是真实物理障碍，所以这些代理都不是完整物理 False Positive 真值。空场景只表示没有新放置 Actor；静态环境物体仍存在。Unknown 永远不能转成安全 Free。

## 分组与预算

初次按地图隔离的 Calibration 在 Town02 只能分配 2 位置，失败记录保留。修订使用 Town02 2 组加 Town04 4 组，后者与 Train 不同 road_id，Ego 间距至少 150m；Train 内位置至少 90m、距已列历史位置至少 45m。训练 20 组（Town01/03/04），Validation 6 组（Town05），Calibration 6 组；新 Sealed 计划 Town10HD 6 组。布局模式家族共享，每组的距离/位置、车型和角度随机化，不宣称 Actor-view 独立同分布。

初次径向布局试采出现多数道路落点缺失。保留原始失败数据，转为采集前的地图道路几何配置；正式数据仍需碰撞落点和沉降审计。只用 Train 首组验证通路，未使用 Validation/Test 图像选择规则。

每种子可靠唯一候选目标 Train≥300，Val/Calibration/Test各≥80；同位置同类别同中心（厘米舍入）同相机的重复帧去重。类别与相机低于20候选不支持独立结论。数量是研究支持门槛，不是模型质量门槛。

网络完全复用 `DirectCenterHead`，Direct=600维（576 RGB+24 几何）、Geometry Prior=24维。两种子101/102；25%/50%/100%嵌套训练位置组，固定600步、batch32、AdamW lr0.001。每个种子/比例的两个模型严格相同采样顺序和监督。Validation位置宏平均XY选择Checkpoint；无预算自动延长。

风险模型只用预测距离、框形状、前景分数、截断代理。Calibration位置与Validation分离，留一Calibration位置拟合误差代理并选择固定阈值；经验90%半径没有分布无关保证。未通过覆盖/尾部/同目标基线门禁时阈值1.1拒绝全部准入，仍保留研究原始位置供诊断。Sealed只在源码、Checkpoint和阈值锁定后运行一次。

## 命令

全部从仓库根执行，CARLA进程以 ubuntu 用户运行。重复实验必须使用新的数据根/副本，不得删除结果后原地重跑。

```bash
git switch -c feat/m5o-perception-benchmark-reset 2afe740
.venv-gpu/bin/python -m tools.m5o_freeze
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5o_allocate --phase dev
# 初版 Town02 容量不足会失败并保留配置；以下独立扩展不缩小距离。
.venv/bin/python -m tools.m5o_extend_allocation
.venv/bin/python -m tools.m5o_road_layouts --phase dev
.venv/bin/python -m tools.m5o_campaign --phase dev
.venv-gpu/bin/python -m tools.m5o_process_dev
```

`campaign` 可按已完成组恢复；不覆盖部分完成目录。`stream_index` 可在后续地图采集期间处理已完成 Train/Val；只能有一个 index writer。`process_dev` 等待像素索引、运行三个 Split 缓存、支持审计、固定训练、Validation和Calibration，失败时阻止训练或封存。

```bash
.venv-gpu/bin/python -m tools.m5o_cache --split regression
.venv-gpu/bin/python -m tools.m5o_cache --split regression --oracle-candidates
.venv-gpu/bin/python -m tools.m5o_candidate_audit --split regression
.venv-gpu/bin/python -m tools.m5o_frozen_diagnostic --split regression
.venv-gpu/bin/python -m tools.m5o_frozen_diagnostic --split regression --oracle-candidates
.venv-gpu/bin/python -m tools.m5o_regression_report
.venv-gpu/bin/python -m pytest -q
```

独立风险/模型评价源码为 `tools/m5o_evaluate.py`；`--split sealed` 强制读取预冻结SHA。正式封存命令和实际结果将在完成后记录于summary。

## 系统解耦

`datasets/m5o_interfaces.py` 只提供 Shadow envelope 与拒绝预测合同，无控制连接。Oracle 环境状态、RGB BEV Evidence、RGB目标候选的 provenance 显式不同。时间戳来自源帧，valid_until不能替代原始时间；过期、未来或坐标不符输入拒绝。Planner未来需要对象/空间证据/有效期/Unknown，但本轮不将视觉估计接入现有 Oracle Planner。

BEV保留 `[4,160,120]` Spatial Evidence V2。Free-Ray Evidence只证明射线穿越，Mixed Cell允许 Surface 与 Free-Ray共存，不等同整格可通行。M5-M Runtime Partial继承；不重跑长期压力矩阵，不将Oracle成绩认定为RGB闭环。
