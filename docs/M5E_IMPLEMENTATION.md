# M5-E — Far-Field Visibility Supervision & Depth Spatial Probability

基础 `33005ef`，独立分支 `feat/m5e-far-visibility-depth`。旧标签/数据/模型/源码/报告保留。默认ResNet18/FPN、48LID、HeightSplat、4×160×120、车辆Physics、Controller/Planner不变。新脚本显式启用；Candidate不作为训练负类。

## Gold、Candidate、Presence-only

`datasets/m5e_witness.py`直接调用原M5-D认证器，Gold定义完全保持。原来3×3深度极差、相同pose/标定、反事实更远首命中、self/语义冲突/密集其他障碍veto、每粗格≥3票均不放松。Gold只是指定表面点沿射线缺失的证据，不认证整个粗体素Free。

`datasets/m5e_visibility.py`使用5×5背景光学深度的倒数平面拟合 `1/z=a*u+b*v+c`。要求25点相同实例和语义、有限且1.5–80m、最大重建深度残差≤0.05m。既保留中心也检查边缘，不用平均残差掩盖不连续。平面/斜坡/斜墙通过；突变、遮挡、相邻不同实例、细杆、行人轮廓、噪声/NaN合成测试拒绝。真实曲面可能被拒，同实例未采样细障碍仍不能排除，因此全部仍为Candidate。

逐射线audit保存UV、camera、present/background depth、5×5 depth/semantic/instance patch、HS三维点、Actor候选关联、coarse rank、层混合、拒绝位标志、拟合残差。原数据审计只用M5-D val，M5-D test为历史回归；新test不参与参数调试。跨实例边界是异质表面风险，不等价已经证实危险遮挡；nonplanar/noise类别不能唯一归因于渲染噪声。同步与标定由采集帧时间戳/矩阵和索引固定pose检查支持，没有多视角三角测量认证。

`tools/m5e_collision_reference.py`在新validation空布局用CARLA `world.cast_ray`补充检查，取每组每段固定最多50个候选。碰撞几何与渲染Mesh不同，缺失/不符不解释为确定空闲。查询结果不自动升级Candidate；不用于网络输入或负监督。

`tools/m5e_supervision.py`构造独立Presence-only：可靠摆放Actor可见Obstacle端点，非留出像素，≥3票，且V2语义可靠无Ground/Other/ego/证据冲突。无对侧Absence时仍保留正表面，但不产生负类。训练基础损失始终含可靠Ground负类与Gold正负配对，Unknown不提供BCE负样本。Presence loss按非空距离段等权，各位置组轮转抽pair。

## 数据与停止条件

第一次Town02自动几何选点因既有位置排除后不足失败；无采集数据，日志保留。正式协议在采集前改用Town03/04/05，12位置、各16布局，6train/3val/3test。与所有M5A/B/C/D历史位置及新位置间隔≥45m。布局含10/15/20/25/30/35m车、20/30m货车、15/25m行人、15/25m交通锥、遮挡、横移、empty/removed。无额外不同尺寸锥资产；车型/高度/坡道覆盖按实际报告，不虚报未采样类别。

Gold完整存在性支持目标：各远场段至少12有效格、3train位置。Presence/Depth研究许可：15–20/20–25/25–35m各≥12可靠正格且≥3位置；不满足则停在诊断。200步×2seed为上限，不自动加步或重复采同位置凑数。test采集在Checkpoint/阈值冻结后；全部变体同组分配。测试是新位置而非未见地图/布局类型。真实同步RGB/Depth/Instance与去自车参考保存在raw，输入allowlist不含GT。

## 受控训练与模型选择

同M4初始化、DirectHead同seed初始化、同pair顺序/200步/数据/S16。A_base：原V2+0.2旧pair+Direct/3+Gold existence+stable。B_presence：A+0.5 Presence-only。C_spatial_frozen：B+0.1 spatial loss，但Depth冻结，作为无梯度负对照。D_depth：相同C目标有限解冻Depth（lr1e-5）；Context lr3e-4，Head lr1e-3，主干/BEV保持冻结。

空间目标在相同可靠UV上投影原48Bin，以同高度层、BEV内、XY距离≤2m为支持，按 `exp(-XY_error²/(2*1m²))` 归一化，和Depth log-softmax交叉熵。无支持Bin的射线屏蔽，不强制精确0.5m格、不监督Unknown/不可见。按径向距离段等权，近场可靠表面一起保留。几何计算显式FP32，视觉/Head BF16。C的spatial项不改变梯度，是检查冻结Loss无虚假收益的对照；D效果来自允许Depth更新，不能唯一归因于spatial项相对纯CE（本轮无该独立对照）。

选择不只用存在性调和分数。每50步、每阈值0.1…0.9报告逐位置Actor exact recall、Gold presence、Gold absence FPR、可靠Ground FPR和Pareto状态。预声明可行条件各组Actor≥.70、Presence≥.80、AbsenceFPR≤.10、GroundFPR≤.05、近场期望空间误差≤初始1.1倍。无可行模型时选最大归一化违反量最小者作研究分析并明确infeasible；平局偏Actor再早Checkpoint。每组每阈值完整记录，不能把选中模型称门禁通过。

## 深度和P8诊断

`m5e_depth_diagnostics.py`在共同UV/Mask/K/T上比较continuousGT、nearestBin、完整learned probability、ExpectedDepth点。报告正确层质量、XY .5/1/2m质量、GTcell、同层3×3邻格、熵、SignedBias、期望绝对XY/XYZ误差、期望深度点误差、投影出BEV及GT在BEV外却进入的质量。最后一项是几何风险统计，不等于证明FalseFree。熵和Bias列始终来自learned分布，即使出现在oracle对照行。`E[error]`不等于`error(E[depth])`，不作可加因果百分比分解。

`m5e_stride_factorization.py`使用已有M5-D C/S8两个seed冻结Head，在新val比较S16/S16、真实P8+插值S16概率、真实P8/P8、P8下采样回16+S16概率。保留射线支持、GT1m质量、Actor表面、Goldabsence、Unknown、GPU实际前向/Splat/BEV+Head时延和显存。仅诊断干预，存在Head分布迁移；P8/S16混合同时改变采样中心和概率重采样，不能声称严格唯一归因。完整前向当前保留两路encoder冗余，是诊断成本而非部署优化延迟。S16默认不变。

## Runtime

原5HzPlanner/20HzController/6sSafety/唯一tickowner/双时钟过期规则保留。`simulator/m5e_affinity.py`只设置新worker进程CPU，不改Planner。新driver只增加显式tick/worker affinity与计时、动态前车场景。资源测试2轮，第二轮反序：默认混合、TickOwner固定核、Worker分核、分核低GPU频率、分时无标签/推理负载。每轮1200tick；100m路线运动后驻车分别统计。CPU集合按实际允许CPU生成，OS affinity不代表宿主独占核；CARLA Server仍原调度。不得以某次最好值关闭门禁。

GPU压力重复保存六摄RGB推理，CPU压力生成历史V2，不是实时18摄像头传感器渲染/传输负载。静态障碍、动态前车、Worker崩溃/停滞/超时/坏输入、紧急停车另实测。碰撞0不认证自然交通全场景、连续Mesh碰撞或舒适动力学。资源日志包括GPU/进程CPU/schedstat；world.tick阻塞与唤醒迟延分开，未获得Unreal内部GPU profiler。

## 可复现执行

所有脚本从仓库根执行，CARLA `.venv`、GPU `.venv-gpu`，沿用已有锁定依赖。主要新输出存在时拒绝覆盖，完整复现使用新checkout/独立数据目录及相应参数；固定路径研究脚本需显式更新路径并保留新哈希，勿在原目录盲重跑。

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv-gpu/bin/python -m tools.m5e_ray_audit
.venv-gpu/bin/python -m pytest -q tests/test_m5e_geometry.py tests/test_m5e_selection.py
# 复用已冻结configs/m5e_protocol.json
.venv/bin/python -m tools.m5e_capture --output /root/gpufree-data/hs-drive-data/raw/m5e_capture --groups Town03_e0 Town03_e1 Town03_e2 Town04_e0 Town04_e1 Town04_e2 Town05_e0 Town05_e1 Town05_e2
.venv-gpu/bin/python -m tools.m5e_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5e_capture --output /root/gpufree-data/hs-drive-data/processed/m5e_index --report reports/m5e_dataset_generation.json
.venv-gpu/bin/python -m tools.m5e_surface_index --index /root/gpufree-data/hs-drive-data/processed/m5e_index
.venv-gpu/bin/python -m tools.m5e_supervision --index /root/gpufree-data/hs-drive-data/processed/m5e_index
.venv-gpu/bin/python -m tools.m5e_ray_audit --index /root/gpufree-data/hs-drive-data/processed/m5e_index --prefix m5e_new_visibility
.venv/bin/python -m tools.m5e_collision_reference
.venv-gpu/bin/python -m tools.m5e_depth_diagnostics --output reports/m5e_depth_baseline.json
.venv-gpu/bin/python -m tools.m5e_quality --indices /root/gpufree-data/hs-drive-data/processed/m5e_index --output reports/m5e_pretrain_quality.json
.venv-gpu/bin/python -m tools.m5e_learning
.venv-gpu/bin/python -m tools.m5e_depth_diagnostics --models
.venv-gpu/bin/python -m tools.m5e_stride_factorization
.venv-gpu/bin/python -m tools.m5e_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5e_index --split val
# 此后冻结模型阈值，再采sealedtest
.venv/bin/python -m tools.m5e_capture --output /root/gpufree-data/hs-drive-data/raw/m5e_sealed --groups Town03_e3 Town04_e3 Town05_e3
.venv-gpu/bin/python -m tools.m5e_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5e_sealed --output /root/gpufree-data/hs-drive-data/processed/m5e_test_index --report reports/m5e_test_generation.json --splits test
.venv-gpu/bin/python -m tools.m5e_surface_index --index /root/gpufree-data/hs-drive-data/processed/m5e_test_index
.venv-gpu/bin/python -m tools.m5e_supervision --index /root/gpufree-data/hs-drive-data/processed/m5e_test_index
.venv-gpu/bin/python -m tools.m5e_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5e_test_index --split test
.venv-gpu/bin/python -m tools.m5e_quality --indices /root/gpufree-data/hs-drive-data/processed/m5e_index /root/gpufree-data/hs-drive-data/processed/m5e_test_index
.venv-gpu/bin/python -m tools.m5e_consolidate
.venv-gpu/bin/python -m tools.m5e_visualize
# 其他GPU/CPU实验结束后再运行，独占CARLA tick owner
.venv-gpu/bin/python -m tools.m5e_runtime_campaign
.venv/bin/python -m tools.m5e_async_drive --output reports/m5e_runtime_faults.json --artifacts reports/m5e_runtime_faults --scenarios dynamic_lead slow_worker worker_crash emergency static_obstacle stalled_worker invalid_input --load-description 'isolated M5E fault and moving actor regression'
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5e_tests.xml
```

原始数据、模型、预测数组在服务器并逐文件SHA封存；Git交付源码、报告、图、测试与完整日志。最终实际结果及任何不满足条件/失败记录见 `reports/m5e_summary.md`。

## 补充诊断与复现边界

```bash
.venv-gpu/bin/python -m tools.m5e_gpu_contracts
.venv-gpu/bin/python -m tools.m5e_gold_funnel_review
.venv-gpu/bin/python -m tools.m5e_metric_review
.venv-gpu/bin/python -m tools.m5e_factorization_review
.venv-gpu/bin/python -m tools.m5e_component_timing
.venv-gpu/bin/python -m tools.m5e_distribution_plot
.venv-gpu/bin/python -m tools.m5e_localization
.venv-gpu/bin/python -m tools.m5e_runtime_analysis
.venv/bin/python -m pytest -q tests/test_geometry_extended.py tests/test_imaging_rig.py tests/test_session.py tests/test_snapshot.py --junitxml=reports/m5e_carla_api_tests.xml
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5e_final_tests.xml
```

Restricted Checkpoint loader仅显式允许NumPy float64元数据所需类型，不修改模型权重。原诊断加载失败日志保留；本次没有以重训处理序列化问题。训练源码快照在`runs/m5e_learning/source`，test前冻结在`m5e_pretest_decision.json`；冻结后新增汇总和画图脚本不改变阈值、标签或模型选择。

`m5e_gold_funnel_review.json`提供与旧663→638→6漏斗严格同口径的复核及6条幸存射线的投票数。`m5e_witness_attrition.json`的独立local-reject计数没有先限制stable/behind，二者分母不同，不能直接当同一阶段减法。

`m5e_depth_metric_review.json`明确一个指标命名限制：主空间报告概率行的`correct_layer_mass`实际为“正确高度层且在BEV内”，补充报告另外给出纯高度层质量以及范围外GT的条件概率分母，保留原报告，不掩盖差异。`m5e_component_timing.json`另拆BEV Encoder和Head耗时；该测量期间CPU标签处理仍在运行，不能称独占主机基准。原P8完整前向按诊断双路实现计时，不等于优化后部署成本。

训练设置了相同随机种子、初始化、采样顺序和预算，但原CUDA HeightSplat使用`index_add`原子聚合，没有启用确定性CUDA算法。B和C空间常数负对照的微小差异不能视为研究收益；梯度合成测试证明冻结spatial项不会改变Context梯度。两个种子是研究重复，不构成统计显著性或全场景泛化证明。

新增位置避免与旧位置重合，但同地图道路背景仍相关；新测试只含3位置、既定障碍布局家族。真实斜坡/车型/天气的完备覆盖未达到，合成斜坡不能替代实景证明；交通锥仅一个型号，不宣称不同尺寸覆盖。GT参考是渲染表面及单独碰撞查询，均不是完整独立精确Mesh真值。

资源实验不是所有处理之间的单变量因果对照：default_mixed下压力进程未绑核，其他带tick affinity的处理同时将压力任务放到4–11核；tick_reserved的Worker继承0–1核，split_affinity改为Worker2–3核。默认→分核的改善不能只归因TickOwner。两轮反序减轻时间漂移但不消除；未执行宿主exclusive CPU或CARLA Server进程绑核。

最终汇总命令（须等全部实验/日志结束）：

```bash
.venv-gpu/bin/python -m tools.m5e_data_analysis --indices /root/gpufree-data/hs-drive-data/processed/m5e_index /root/gpufree-data/hs-drive-data/processed/m5e_test_index --report reports/m5e_supervision_isolation.json
.venv-gpu/bin/python -m tools.m5e_scene_figure
.venv-gpu/bin/python -m tools.m5e_assets
.venv-gpu/bin/python -m tools.m5e_acceptance
.venv-gpu/bin/python -m tools.m5e_report
```

`m5e_supervision_isolation.json`沿用通用诊断代码的一句“raised-pedestrian stress”只是历史说明；本轮布局没有抬升行人，实际仅地面15/25m行人。权威布局以`configs/m5e_protocol.json`和raw manifest为准。

新增Runtime包装器单测（主回归244通过之后新增，单独JUnit保留）：

```bash
.venv-gpu/bin/python -m pytest -q tests/test_m5e_runtime.py --junitxml=reports/m5e_runtime_unit_tests.xml
```
