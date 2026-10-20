# M5-B：Counterfactual Visual Occupancy & Geometry

基于 `a252a04723a2c11f1189b3580a6d763345e1ae04`，分支 `feat/m5b-counterfactual-occupancy`。旧文件、模型、V2生成器、48 LID bins、HeightSplat、默认Controller/Physics不变。本轮代码通过独立入口扩展，不替换正式推理或安全模型。

## 评价目标与数学合同

M5-A 的 Surface AP 衡量有限测量配方的表面事件；Obstacle AP 只在可靠语义表面 Mask 下计算，是条件分类，不能解释成全 BEV 检测。Free-Ray是射线穿越事件，不能解释成整格安全空闲。独立采样单位为位置组，布局、体素和像素不是独立场景。

保持V2原损失，新增独立 `m5b_pair_*.npz`，只从监督路径读取。正式模型仍对单帧六摄RGB输出Surface、Free-Ray、Semantic三个分支，无两帧GT Mask输入。

设单帧用于变化诊断的未校准分数

`q_i(v)=sigmoid(surface_i(v))*sigmoid(obstacle_semantic_i(v))`。

这是两个事件分数的乘积，不是经证明的联合物理占据概率。相邻配置只是成对监督，不是部署时序模型。

`M+`：B帧至少3个可靠障碍表面像素端点落入粗体素v；端点在某已摆放Actor的AABB范围内（仅候选过滤，不将盒内体积填为障碍）；在A帧同相机同像素及3×3邻域，最小首次命中深度大于B端点深度+.5m；A帧对应像素通过去自车参考稳定性和自车Mask检查；该粗格A帧不存在已知其他障碍表面，且无自车/未解冲突。`M-`交换A/B。若两方向同格都成立，屏蔽该格。

这些Mask确认**某些可见表面点的出现/消失**，不是证明整个体素有/无物理占据。AABB与Mesh不同，未被观测的目标背面与体积不监督。参考来自CARLA渲染器，非独立Mesh真值。

`Ms`：两帧都有有效Surface/Obstacle语义监督、Surface与Obstacle事件相同、无Mixed Cell、无M+/M-。只表示这些已观察事件稳定。

变化误检的评价域 `V=M+ OR M- OR Ms`。不将域外Unknown判作负例。输出先独立预测，再使用Mask算Loss/Metric。

变化预测为 `qB-qA > τ`（出现）或 `< -τ`（消失）。报告有方向TP/FP/FN、P/R/IoU，方向正确比例、幅度、消失未下降比例、固定.5残留分数诊断、可见距离/高度层、Ground/Obstacle/Unknown/Mixed分层。原V2 Unknown中的点若被成对射线支持，仍只取得局部点证据，不变成整格Free。

定位误差是新增可见表面格到预测正变化格的最近XY距离；无预测时单列unmatched，不能把这些失败排除后单独报小MAE。Footprint指标是Actor AABB XY相交区域中的可见点证据召回，不是精确渲染Mesh轮廓或完整碰撞体积召回。Empty→Obstacle→Empty另比较独立采集的两个Empty输入。

## 数据与封存

`configs/m5b_protocol.json` 在采集/训练前固定12新位置组（每图2train/1val/1test），地图Town01/02/03均历史地图。按地图几何最远点和前25m道路存在性选位，不按渲染/模型表现选。每组12配置：empty、5/10/15/25m Tesla、±3m横移、Sprinter大车、前后遮挡、小行人、升高行人、return empty。升高行人是合成高度压力，不代表自然场景。

144帧原始采集，train72/val36/original reserved36。在任何训练前另固定 `m5b_sealed_protocol.json`：只在三个test位置采集偏移布局（5.5/11/17/27m、横向±2.4m等），共36帧，作为正式封存测试；原位置test布局仅保留资产，不参加选择/正式分数。共180帧，12独立位置，正式test仍只有3独立位置。训练/验证全部配对Mask生成后才训练；封存test的配对Mask与模型评测在 `m5b_pretest_decision.json` 固定checkpoint与阈值之后运行。

M5-A所有数据现在是历史研究/回归数据，不再称盲测。本轮不宣称新地图泛化。配置间演员冻结、场景晴天，非真实运动/天气泛化。

试采集只用Town03_b0训练组：初版误要求空场景远背景表面平滑，导致远车/行人证据消失；修正为3×3最小first-hit在目标后方，保留去自车可见性约束，失败报告保留。修正规则未用val/test调参。

`tools/m5b_dataset.py`复用原M5-A生成器得到不变V2标签，再独立生成配对sidecar。复用器底层旧manifest的protocol字段指向M5-A；权威本轮字段在独立 `m5b_manifest.json` 和报告中，未覆盖旧数据。封存侧原始索引同样沿用旧schema，真实协议在 `m5b_test_opening.json`/`m5b_assets.json`溯源。

## 学习设计与控制

A:384,000参数位置先验。B:112,020参数EvidenceHead，Encoder冻结。C:B相同初始化+Context Conv(4,160参数)可训练，其余Encoder/Depth/BEV Encoder冻结。C_pair与C唯一差异是额外变化Loss。所有模型从同一M4 expanded joint Encoder和同种子新Head开始，400步、同一成对样本索引顺序、两帧batch；种子61/62。A lr .03；B/C Head .001；C Context .0003。A参数化和lr不同，容量差异明确披露。

`models/m5b_cached_encoder.py`仅缓存冻结RGB图像特征、预测depth probabilities和标定；无GT。C重新计算Context→原HeightSplat→原BEV Encoder，梯度仅到Context和EvidenceHead。几何FP32、视觉/Head BF16 autocast。FP32等价性和冻结参数无梯度有测试；BF16单点可能有scatter累加/舍入放大，实际对比见 `m5b_bf16_equivalence.json`，不能声称逐bit等价。

原Loss `L_E`保持不变；新增

`L_pair = mean_M+ ReLU(.1-(qB-qA)) + mean_M- ReLU(.1+(qB-qA))`（有值方向取平均）`+ .1*mean_Ms (qB-qA)^2`。

`L_total = L_E + .2*L_pair`。无Mask域为零梯度。C与C_pair训练步数、采样、初始化、lr均一致。报告稳定背景误检，不能只看方向/Recall。

每100步按val场景均值signed-change IoU选checkpoint和τ∈{.025,.05,.1,.2}。同分优先较小τ；相同checkpoint分数保持最早。A对同位姿变化必定恒定，主指标全为0；初版错误选到step0，训练过程仍完整保留，随后**在test打开前**按原协议预声明的event IoU次指标重放相同400步/同序列，保存 `runs/m5b_prior_tiebreak/` 与 corrected报告。这不是给A追加800步。`m5b_pretrain_contract.json`里“所有平局保持最早”的实现描述对A被纠正；纠正遵循更早原protocol，不用test信息。

D条件在原协议固定：两种子C增益均<.05且Oracle投影至少+.10，才有理由另行预声明Depth/Context短预算。本轮未触发，D未训练；没有Confidence-aware Splat替换。

## 几何归因

仅使用val、预指定配置1/2/3/4/7/9。特征中心采样GT Depth及Actor语义，统计GT最近LID bin概率、±3m质量、正确粗体素概率、期望XY投影误差。Oracle保持原48 centers，最近bin one-hot（量化误差单列），不能称精确连续Oracle Depth。

固定RGB，比较Learned Depth / Oracle Bin；再固定Empty Context，只改变Oracle depth。补充固定Empty Depth与正常Context、固定Empty Context与正常Depth。记录图像特征、Context、Height BEV、BEV Encoder变化幅度。不同层L1不能直接比较成因权重；冻结Head遇到Oracle输入可能OOD。诊断不计入正式测试成绩。

## 异步运行时

复用 `simulator/m5a_async.py`独立spawn Worker和60×.1s Safety，不改默认接口。主循环20Hz，提交5Hz；原双时钟.35s有效期、Scene/Frame/时间校验和应用前复核保留。优化仅缓存车身尺寸、崩溃注入terminate移到helper线程且不join、最后版本一次获取snapshot后同时生成状态和测量；`simulator/m5b_runtime.py`不调用world/tick。

记录get_snapshot、get_control、状态解析、snapshot dict、queue submit、poll/反序列化/验证/重采样、道路查询、Actor guard、Controller、apply_control、world.tick、内存日志耗时；磁盘日志在loop结束写入。独立微基准测deepcopy/pickle/unpickle，真实队列调度仍以闭环age为准。

三次运行全部保留。`m5b_async_runtime.json`与`m5b_async_isolated_runtime.json`是前两次，前者与标签处理并行，后者一部分与有限GPU研究重叠，不能严格称完全隔离。**最终以 `m5b_async_final_runtime.json` 的单独CARLA运行和同源码快照为准**；名称“isolated”不抹除实际负载说明。完整九场景含直道/弯道/路口、慢Worker、行驶卡顿、崩溃、坏输入、紧急命令、静态前车。

Oracle Actor/地图/状态，不是纯视觉闭环。持续时间短、无密集动态交通或交通规则覆盖；预测矩形和道路角点不等于精确连续Mesh碰撞。平均20Hz、P95/P99/Max、tick deadline、应用间隔、计划age分别验收。

## 可复现命令

在仓库根目录运行。既有输出不会覆盖；下列是实际使用路径，重做需新工作目录/输出参数。GPU与CARLA依赖见M5-A锁定清单，本轮无新增运行依赖。

```bash
.venv/bin/python -m tools.m5b_protocol
.venv/bin/python -m tools.m5b_capture --groups Town03_b0 --output /root/gpufree-data/hs-drive-data/raw/m5b_pilot
.venv/bin/python -m tools.m5b_capture --groups Town03_b1 Town03_b2 Town03_b3 Town02_b0 Town02_b1 Town02_b2 Town02_b3 Town01_b0 Town01_b1 Town01_b2 Town01_b3 --output /root/gpufree-data/hs-drive-data/raw/m5b_capture
.venv/bin/python -m tools.m5b_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5b_pilot /root/gpufree-data/hs-drive-data/raw/m5b_capture --output /root/gpufree-data/hs-drive-data/processed/m5b_index --report reports/m5b_dataset_quality.json
.venv/bin/python -m tools.m5b_capture --protocol configs/m5b_sealed_protocol.json --groups Town03_b3_sealed Town02_b3_sealed Town01_b3_sealed --output /root/gpufree-data/hs-drive-data/raw/m5b_sealed
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5b_sealed --output /root/gpufree-data/hs-drive-data/processed/m5b_sealed_index --report reports/m5b_sealed_generation.json
.venv/bin/python -m tools.m5b_data_analysis
.venv-gpu/bin/python -m tools.m5b_learning
.venv-gpu/bin/python -m tools.m5b_prior_tiebreak
.venv-gpu/bin/python -m tools.m5b_geometry
.venv-gpu/bin/python -m tools.m5b_extra_geometry
# 仅上面产生pretest_decision后才开放：
.venv/bin/python -m tools.m5b_seal
.venv-gpu/bin/python -m tools.m5b_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5b_sealed_index
.venv-gpu/bin/python -m tools.m5b_visualize
.venv-gpu/bin/python -m tools.m5b_bf16_audit
.venv/bin/python -m tools.m5b_async_drive --output reports/m5b_async_final_runtime.json --artifacts reports/m5b_async_final_artifacts
.venv/bin/python -m tools.m5b_runtime_analysis
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_geometry_extended.py tests/test_imaging_rig.py tests/test_session.py tests/test_snapshot.py -q
.venv/bin/python -m tools.m5b_assets
```

全日志 `reports/m5b_execution_logs/`，数据/模型/SHA清单 `m5b_assets.json`，训练源码快照 `runs/m5b_learning/source/`，每次CARLA运行都有source目录。测试单测/离线诊断/实际GPU/CARLA实测分开标注。GPU原子操作和重新渲染不保证bitwise复现；复核数值优先使用保留原数据和checkpoint。

交付时必交 `m5b_context_ablation.json` 指向已纠正先验选择的完整结果，初版移存 `m5b_context_ablation_initial.json`，训练记录未丢弃。必交数据质量报告合并V2生成统计和paired覆盖，初版保留 `m5b_dataset_generation_initial.json`。目标几何正确范围在 `m5b_target_geometry.json`，初版actor语义含ego的统计不作目标归因。
