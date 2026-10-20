# M5-C — Reliable Single-Frame Obstacle Localization & Spatial Geometry Resolution

基础提交 `dbb70ed`，分支 `feat/m5c-obstacle-localization`。所有新输出使用 M5-C 路径；旧模型、V2 标签生成器、48 LID Bin、HeightSplat、车辆物理和默认 Controller 保留。`m5c_baseline_freeze.json` 冻结旧源码/Checkpoint；最终验收逐文件复核。

## 指标合同

A：Conditional Semantic Classification 只在可靠表面上评价语义；B：Signed Change 检验两个布局的分数变化；C：Absolute Visible Surface Localization 在单帧六摄 RGB 预测中检验可见障碍表面位置。本轮 C 为主，A/B 保留。三者都不能当作未知空间的可通行保证。

`datasets/m5c_surface.py` 从原生渲染深度/实例和同标定去自车参考取得可靠首命中端点，留出像素 `(u+2*v)%4==1`，与 V2 表面训练端点不重合。每粗格至少 3 个留出 Ground 或 Obstacle 端点才有效；任意 Ground/Obstacle/未分类语义冲突和自车格排除。Free-ray-only 和未观测格不当负样本。此参考与训练仍同源 CARLA 渲染器，不是独立 Mesh 真值。

几何匹配采用 **同高度层、XY 欧氏距离、0.5m 和 1m 容差、双向最近邻**。这是多对一表面覆盖，不是一对一实例匹配。分别记录 pred→GT、GT→pred 的距离、没有同层候选数量、容差外逐 GT miss 和额外预测。距离均值仅对存在同层候选的点计算，未匹配数不能遗漏。已知支持域内算 Precision/Recall；Unknown 中预测数单独报告，既不计已认证 FP，也不默认为安全。

Actor AABB 只关联实际可见渲染端点，不将盒体积作为占据标签。≥3 个有效表面格的 Actor 才可评价；≥3 格且≥20% 表面在同层 1m 内匹配，记录弱检测事件。此门槛不代表完整轮廓已准确恢复。还单列目标 Actor 的 exact-cell recall，避免建筑/墙体表面掩盖放置目标的漏检。规划走廊为 x∈(0,35)m、|y|≤1.5m，仅对可确认表面统计。

## 数据与冻结

`configs/m5c_protocol.json` 在新采集前固定 16 位置、4 地图、每组 16 布局。8 train、4 val、4 test；组内布局不跨集合，新组距同地图已使用位置至少45m。包含5/10/15/20/25/30m轿车、横移、货车、遮挡、15/25m行人、交通锥及抬高行人压力布局；ClearNoon/CloudyNoon/ClearSunset。训练和测试使用预声明同一布局集合，所以新位置盲测不等于未见布局类别或新地图盲测。

Actor 是静态摆放，抬高行人是合成高度压力。道路弯曲/坡度和背景遮挡会限制真实可见支持，零支持目标保留，不根据验证或测试可见性更换位置。旧 M5-B test 仅历史回归。

采集期间首次切图服务退出和 Town04 临时 Actor 停放碰撞的失败目录/日志均保留。重试将非活动 Actor 放置到远离相机的高空；正式布局未改。成功开发组来自 `raw/m5c_capture_retry1` 和 `raw/m5c_capture_retry2`。Sealed Test 只在 checkpoint/阈值/协议冻结后采集并首次评价。

## Loss 和受控训练

P：PositionPrior；A：Frozen RGB-BEV + EvidenceHead；B：Context 微调 + 原证据损失；C：B + 原配对损失；D：C + 独立 Visible Obstacle Head 和辅助 BCE。每组种子71/72、400步、相同配对索引顺序，从同一 M4 Encoder 和同种子随机共享 Head 出发。没有在旧最佳 C 上增加步数冒充单变量收益。旧 C_pair checkpoint 保留，新增训练是匹配重训研究。

设 S 为可靠且无自车/证据/语义冲突的表面监督，G/O/T 为 Ground/Obstacle/Other。独立正 Mask `P=S∩O∩¬G∩¬T`，负 Mask `N=S∩G∩¬O∩¬T`。`L_direct=(mean_P softplus(-l)+mean_N softplus(l))/2`（空类别不产生项）。其他格零梯度。Ground+Free 的 mixed 格可有可靠 Ground 语义，但 free-ray 本身不提供负标签；Ground+Obstacle 不作为独立二分类负样本。

D 总损失为原 `L_evidence + 0.2 L_pair + (1/3) L_direct`；C 不含最后一项。原输出与原 q=`sigmoid(surface)*sigmoid(obstacle)`保留。D 主要输出为独立 sigmoid，仍是可见表面证据分数，不是物理体积占据概率。C/D 差异是一个辅助任务（Head+loss），不能进一步把收益唯一分解为网络容量或监督作用。

Head lr=.001、Context=.0003、PositionPrior=.03。只解冻 Context，不动 Depth、Image Backbone、BEV Encoder。相同 BF16 RGB/Depth 缓存；FP32 几何。每100步按 validation 场景均值 absolute surface exact IoU 选 checkpoint；阈值 [.1,.2,...,.9]、平局保留较低阈值和更早步数。配对变化阈值独立在 val [.025,.05,.1,.2]选择。每个模型都报告，不靠 test 挑选最佳。

## 几何诊断

同一整数 UV、有效 Mask、K/T 比较 continuous GT、nearest LID center、完整 learned probability 和 expected depth。概率仍取原半像素 feature center；四个投影共同使用舍入 UV，消除 oracle-only 半像素移动的混杂。整数采样的量化研究与原生 Splat 推理应分别解释。各距离段必须保留样本数和位置数；实际表面距离与 Actor 中心/名义摆放距离不同。

Stride8 首先只改变采样网格；资源测试用插值 Context/Probability 计量原 Splat 的工作量，不宣称高分辨率特征可学习性。Oracle 替换、固定空场景 Context/Depth、RGB 局部遮蔽仅为验证集诊断，不是正式性能或重新训练后的上限。不同层 Feature L1 不作因果排名。

## Runtime 独立合同

复用唯一 tick owner、独立 Planner Worker、5Hz Planner、20Hz Controller、完整6秒 Safety、双时钟/scene/frame检查和故障停车。M5-C 只增加测量与持续运行开关，不升级 Controller/Physics 默认。

预声明周期门槛：各运行 P99≤55ms、Max≤100ms、tick deadline miss≤0.1%、过期规划正常控制次数0。平均20Hz不能替代门槛。第一轮名为 isolated1 的目录实际与 CPU 标签处理并行，最终明确标为 CPU 负载。五分钟压力运行到达路线终点后驻车，不能称五分钟持续运动；另外保留运动路线重复。GPU 压力为真实六摄 RGB Encoder 推理，但输出不送 Oracle 驾驶器，仍不是纯视觉闭环。

故障延迟起点是注入 tick65，终点为主进程调用制动命令；物理响应发生后续仿真 tick。保留全部帧日志、每分钟统计、控制间隔、world.tick 阻塞、IPC/地图/控制/日志耗时及尾部帧。

## 执行命令

以下命令在仓库根目录执行；输出路径存在时程序拒绝覆盖。完整重跑请使用独立输出目录并同步传入 index/run 参数。CARLA `.venv` 和 GPU `.venv-gpu` 分离。

```bash
.venv-gpu/bin/python -m pip install -r m5c_requirements.txt
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5c_protocol
.venv/bin/python -m tools.m5c_capture --output /root/gpufree-data/hs-drive-data/raw/m5c_capture --groups Town01_c0 Town01_c1 Town01_c2 Town03_c0 Town03_c1 Town03_c2 Town04_c0 Town04_c1 Town04_c2 Town05_c0 Town05_c1 Town05_c2
.venv-gpu/bin/python -m tools.m5c_geometry_resolution
.venv-gpu/bin/python -m tools.m5c_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5c_capture_retry1 /root/gpufree-data/hs-drive-data/raw/m5c_capture_retry2 --output /root/gpufree-data/hs-drive-data/processed/m5c_index --report reports/m5c_dataset_generation.json
.venv-gpu/bin/python -m tools.m5c_surface_index --index /root/gpufree-data/hs-drive-data/processed/m5c_index
.venv-gpu/bin/python -m tools.m5c_data_analysis --indices /root/gpufree-data/hs-drive-data/processed/m5c_index --report reports/m5c_pretrain_quality.json
.venv-gpu/bin/python -m tools.m5c_geometry_resolution --index /root/gpufree-data/hs-drive-data/processed/m5c_index --prefix m5c_new
.venv-gpu/bin/python -m tools.m5c_learning
.venv-gpu/bin/python -m tools.m5c_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5c_index --split val
.venv-gpu/bin/python -m tools.m5c_interventions
```

冻结之后才运行 sealed capture/index/evaluate；具体路径、最终门禁、测试结果及所有结果类型见 `reports/m5c_summary.md` 和 `reports/m5c_acceptance.json`。源码快照、SHA 与原始执行日志保存在报告和 runs 中，失败尝试不删除。

```bash
# 以下命令仅在 m5c_pretest_decision.json 已生成且哈希核对后执行
.venv/bin/python -m tools.m5c_capture --output /root/gpufree-data/hs-drive-data/raw/m5c_sealed_capture --groups Town01_c3 Town03_c3 Town04_c3 Town05_c3
.venv-gpu/bin/python -m tools.m5c_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5c_sealed_capture --output /root/gpufree-data/hs-drive-data/processed/m5c_test_index --report reports/m5c_test_generation.json --splits test
.venv-gpu/bin/python -m tools.m5c_surface_index --index /root/gpufree-data/hs-drive-data/processed/m5c_test_index
.venv-gpu/bin/python -m tools.m5c_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5c_test_index --split test
.venv-gpu/bin/python -m tools.m5c_data_analysis --indices /root/gpufree-data/hs-drive-data/processed/m5c_index /root/gpufree-data/hs-drive-data/processed/m5c_test_index
.venv-gpu/bin/python -m tools.m5c_local_witness_analysis
.venv-gpu/bin/python -m tools.m5c_geometry_semantic_corrected --index /root/gpufree-data/hs-drive-data/processed/m5c_index --prefix m5c_corrected
.venv-gpu/bin/python -m tools.m5c_baseline_review_corrected
.venv-gpu/bin/python -m tools.m5c_bf16_audit
.venv-gpu/bin/python -m tools.m5c_historical_checkpoint
.venv-gpu/bin/python -m tools.m5c_quantization_lattice
```

```bash
# 独占 CARLA 实例，一次只运行一个驾驶/采集进程。
.venv/bin/python -m tools.m5c_async_drive --output reports/m5c_runtime_isolated2.json --artifacts reports/m5c_runtime_isolated2 --scenarios straight mild_curve junction_turn --load-description 'isolated Oracle only'
.venv/bin/python -m tools.m5c_async_drive --output reports/m5c_runtime_isolated3.json --artifacts reports/m5c_runtime_isolated3 --scenarios straight mild_curve junction_turn worker_crash invalid_input emergency --load-description 'second isolated Oracle repeat'
# 压力进程在另一终端启动，随后启动下面6000tick驾驶；GPU进程不连接CARLA。
.venv-gpu/bin/python -m tools.m5c_gpu_load --duration 340 --hz 10 --report reports/m5c_gpu_pressure.json
.venv/bin/python -m tools.m5c_async_drive --output reports/m5c_runtime_pressure_trial.json --artifacts reports/m5c_runtime_pressure_artifacts --scenarios straight --ticks 6000 --hold-after-route --load-description 'RGB inference pressure; Oracle control'
.venv-gpu/bin/python -m tools.m5c_runtime_analysis
.venv-gpu/bin/python -m tools.m5c_visualize
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5c_full_tests.xml
.venv/bin/python -m pytest -q tests/test_geometry_extended.py tests/test_imaging_rig.py tests/test_session.py tests/test_snapshot.py --junitxml=reports/m5c_skipped_api_coverage.xml
.venv-gpu/bin/python -m tools.m5c_assets
```

主报告之外的修正入口：`m5c_diagnostic_corrections.json`。旧语义硬编码几何统计遗漏 Dynamic21 交通锥，权威新验证几何结果是 `m5c_corrected_geometry_resolution.json` / `m5c_corrected_feature_sampling.json`。历史指标重算权威结果为 `m5c_spatial_localization_historical_corrected.json`；初次空记录保存为 `m5c_spatial_localization_initial.json`；主 `m5c_spatial_localization.json` 已填充修正指标，旧结果仍保留。所有修正都未改变训练标签、网络、阈值或封存测试选择。

`m5c_local_witness_scores.json` 是在测试评价前声明的二级诊断：将冻结单帧阈值用于同射线物理点的出现/消失监督。这里的 absent 只表示原物理表面点在对照观测中没有被占据，不认证整个0.5m粗体素为空闲。因此该诊断不能被用来替代正式全体积 Precision，也不能将这些格直接给 Planner 作为 Free。
