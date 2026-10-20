# M5-D — Visibility-Conditioned Obstacle Existence & Single-Frame Geometry

基于 `feat/m5c-obstacle-localization@2aa91f6`，在 `feat/m5d-visible-existence` 开发。保留全部旧源码、模型和结果。默认48 LID、HeightSplat、ResNet18/FPN、V2、四层160×120 BEV、Controller和Physics均未更改。新代码只在显式M5-D入口启用。最终源码/数据/Checkpoint哈希与提交关系见 `reports/m5d_assets.json`、`reports/m5d_acceptance.json`。

## 监督审计与数学合同

`tools/m5d_supervision_audit.py` 对旧train/val做实际梯度探针：旧pair loss读取surface和semantic，`visible_obstacle.weight/bias.grad is None`；共享trunk有梯度，不能说整个网络完全不受pair loss影响。Direct BCE只对可靠Obstacle/Ground表面生效。2840个旧absence物理点探针中仅355个有直接Ground负监督，2485个没有表面endpoint；无endpoint不等于物理absence。按层/距离/事件分解和背景原始语义分布分别在 `m5d_supervision_audit.json` / `m5d_background_supervision.json`。

新 `datasets/m5d_witness.py` 保留旧pair sidecar作对照。设B为出现、A为空场景，对可靠B首命中端点要求：

1. 相同固定ego pose、相机标定、地图、天气（采集和索引检查）。目标端点可靠、语义为Obstacle/Vehicle；AABB只关联已渲染端点。
2. A相同UV为可靠、有限、1.5–80m首命中；同去自车参考稳定，无自车silhouette；3×3最小深度比B深至少0.5m；实例ID改变。
3. 两侧无自车/证据/语义冲突；对侧任何可靠密集Obstacle端点落入同粗格都否决，不限V2训练像素。
4. 每格至少3条支持，出现与消失冲突排除，并取旧合法Witness子集。消失方向交换A/B。

这里的负监督只认证**指定目标表面点沿对应射线未出现**，不认证整个粗格无物体。Unknown体积格可以含已认证点absence；这不允许将其传给Planner为Free。Ground和Free共存的Mixed格不是自动坏标签；语义/证据冲突仍排除。未认证位置无额外BCE梯度。

`training/m5d_existence_loss.py` 对出现Mask M：`0.5*(mean_M softplus(-lB)+mean_M softplus(lA))`；消失Mask反向。按方向和非空径向距离段等权平均，空段不造负样本。加 `0.1*relu(0.1-(p_present-p_absent))`，C另加 `0.2*mean_stable (pB-pA)^2`。stable必须两侧可靠同任务表面，无混合/冲突，不拿Unknown做稳定负标签。训练按位置组轮转，再组内抽pair；零事件pair保留，仍有基础损失。

Loss不认证完整Obstacle存在概率；它学习条件点证据。新数据train/val的20–35m认证absence为0，test只有6个事件（可含出现/消失重复），明确未满足远场验证。`m5d_witness_attrition.json`记录对侧局部深度平滑性规则的覆盖损失，未据此放松本轮Mask。

## 模型与公平对照

历史A：直接加载M5-C D_direct两个Checkpoint，冻结旧阈值做回归；不混同匹配重训。新A_matched/B_existence/C_stable/S8_stable同M4 Encoder初始化、同seed随机Direct Head、同数据、同pair顺序、400步；只解冻Context和Head。A为旧损失 `L_V2+0.2 L_oldpair+L_direct/3`；B增加存在性BCE+ranking；C仅再增stable项。B内部BCE和ranking作为一个监督包，不能进一步唯一归因于其中一项。

种子81/82，Head lr=.001、Context=.0003、PositionPrior=.03。每100步及step0评val，按位置组Presence Recall与Absence Specificity调和均值选步/阈值，阈值0.1…0.9，平局低阈值/早Checkpoint。A种子82选择step0，必须保留披露；所有模型仍执行400次更新，不以相同选中步数冒充对照。表面AP/IoU不作唯一模型选择。训练过程使用BF16特征和FP32几何，冻结ResNet/Depth/BEV参数无梯度，真实显存、耗时、各loss、最大梯度记录在 `m5d_model_ablation.json`。缓存训练时间不是完整端到端训练吞吐。

S8为真实ResNet layer2特征，经原lateral融合上采样layer3/4、原output卷积；不是插值最终Stride16 Context。复用原Depth权重会发生分布迁移，S8对比包括特征融合分辨率和Depth响应变化，不能唯一归因于射线密度。所有模型保留默认HeightSplat。`models/m5d_inference.py`提供单帧wrapper，只接收既有RGB/标定/状态allowlist；不接收pair、GT、Actor或Mask。

P8投影遵循原均匀图像格中心 `(i+.5)*image_size/feature_size-.5`，不是声称与卷积理论感受野中心精确重合。几何诊断四路同用舍入后的UV和相同Mask，概率来自对应原特征中心；这属于明确的采样约定误差，默认投影未变。`m5d_gpu_contracts.json`实测直接与缓存推理误差及GT输入拒绝。

## 数据、评价和封存

固定12个新位置、Town02/04/05、三天气、16布局，6 train / 3 val / 3 test。共192帧，所有六摄RGB/Depth/Instance同步；每布局另有同标定去自车参考。包含空、5–30m车、横移、车型、遮挡、行人、锥、抬高目标、返回空场景。位置距历史/新位置≥45m。Actor静态摆放，升高行人为合成高度压力；不代表运动泛化。三test位置在模型/阈值冻结后才采集，首次评测后无训练和阈值改动。新位置不是未见地图/布局类别。

主指标使用认证point witnesses，按位置组报告，并输出41点阈值曲线（test曲线仅描述，不重选阈值）。Presence/absence是对同一物理点的一正一负配对，因此PR为条件且类平衡，不能当自然场景全BEV Precision。出现/消失重复不算独立事件。单独记录层、距离、分数差方向；绝对定位复用M5-C修正后的同层XY 0.5/1m双向多对一表面覆盖、miss/extra和Actor弱检测；另报告粗体素中心3D距离，不能冒充精确Mesh距离。

绝对参考为留出像素 `(u+2v)%4==1`，每格≥3可靠同语义表面，排除自车/冲突；这些与V2表面训练端点分离，但都是同一渲染器，不是独立Mesh真值。Actor域只有正表面，Precision=1没有检测含义；必须看Recall/漏检。Unknown高分单列，既不当已认证FP，也不当安全Free。返回空场景统计共同可靠表面及ground分数；整格未观测不制造负标签。零支持Actor一直保留。

历史72.5%–80.2%残留比较使用完全相同M5-C原帧/原Mask/旧冻结阈值；`m5d_historical_same_witness.json`复算旧模型，`m5d_historical_new_models.json`对新模型回归。不能把新认证Mask覆盖收缩称为性能改善，也不能把跨轮数据/选步变化称严格单变量消融。新test同时保留legacy Mask指标。

## 几何与Runtime研究范围

`m5d_stride8_analysis.json`实测真实P8射线数、概率和、Splat点数、BEV变化和GPU时延/显存。`m5d_geometry_resolution.json`同射线比较continuous GT、nearest LID、完整learned probability、expected depth和Oracle-bin软XY分配。连续GT自一致100%不是独立准确率。完整分布的期望绝对空间误差不等于Expected Depth点的误差；不可将两者相减解释因果百分比。

`m5d_native_quantization.json`补充657条20–35m原生可靠射线，来自3个val位置；它们是相关像素，不增加独立场景数，也不是Stride16特征支持。子Bin学习/新几何辅助D没有训练；本轮证据优先支持修复监督覆盖与已有分布，未授权由test驱动新实验。

Runtime仍5Hz Planner / 20Hz Controller / 6秒Safety、唯一tick owner、原始时间戳判断0.35s过期；旧默认控制不变。M5-D增加get_snapshot、状态、序列化、IPC、道路、控制、apply、tick、日志分段计时。日志缓冲在循环内、磁盘写入在循环后，不能称消除所有内存或序列化开销。

主campaign顺序：CARLA alone、CPU标签、GPU六摄10Hz、混合、五分钟终点驻车混合、alone重复；运动路线扩为250m并单列实际运动tick/距离/完成率。GPU进程重复推理保存的单个六摄样本，CPU进程循环历史帧生成V2标签；驾驶进程没有实时18摄像头传感器渲染/传输流，因此不是完整在线感知负载认证。监控 `/proc` CPU、schedstat、GPU利用率/显存；控制间隔与deadline分别统计。P99≤55ms、Max≤100ms、deadline≤0.1%、过期正常控制0，各运行独立验收。server snapshot平台时间是帧推进代理，world.tick是客户端阻塞，未获得Unreal内部渲染/GPU profiler，不作更细因果断言。一次顺序因子试验不能证明所有资源争用因果；两次alone保留时间漂移影响。

## 复现命令

在仓库根执行。`.venv`为CARLA Python，`.venv-gpu`为RTX4090环境，依赖沿用原锁定文件及 `m5c_requirements.txt`。输出存在时主要实验拒绝覆盖。完整复现建议新checkout加独立数据根，按入口参数替换路径；固定路径研究脚本需在新checkout同步改路径并记录新源码哈希，不能在原资产上直接重跑。

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv-gpu/bin/python -m tools.m5d_supervision_audit
.venv-gpu/bin/python -m tools.m5d_background_audit
.venv-gpu/bin/python -m tools.m5d_historical_regression
# configs/m5d_protocol.json 为本轮预声明协议；重生成会重新查询地图，不能覆盖原协议
.venv/bin/python -m tools.m5d_capture --output /root/gpufree-data/hs-drive-data/raw/m5d_capture --groups Town02_d0 Town02_d1 Town02_d2 Town04_d0 Town04_d1 Town04_d2 Town05_d0 Town05_d1 Town05_d2
.venv-gpu/bin/python -m tools.m5d_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5d_capture --output /root/gpufree-data/hs-drive-data/processed/m5d_index --report reports/m5d_dataset_generation.json
.venv-gpu/bin/python -m tools.m5d_surface_index --index /root/gpufree-data/hs-drive-data/processed/m5d_index
.venv-gpu/bin/python -m tools.m5d_witness_quality --index /root/gpufree-data/hs-drive-data/processed/m5d_index --output reports/m5d_witness_quality_trainval.json
.venv-gpu/bin/python -m tools.m5d_stride8_analysis
.venv-gpu/bin/python -m tools.m5d_learning
.venv-gpu/bin/python -m tools.m5d_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5d_index --split val
.venv-gpu/bin/python -m tools.m5d_gpu_contracts
.venv-gpu/bin/python -m tools.m5d_witness_attrition
.venv-gpu/bin/python -m tools.m5d_native_quantization
# 必须已有 m5d_pretest_decision.json，核对模型/协议哈希后才开启test
.venv/bin/python -m tools.m5d_capture --output /root/gpufree-data/hs-drive-data/raw/m5d_sealed --groups Town02_d3 Town04_d3 Town05_d3
.venv-gpu/bin/python -m tools.m5d_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5d_sealed --output /root/gpufree-data/hs-drive-data/processed/m5d_test_index --report reports/m5d_dataset_test_generation.json --splits test
.venv-gpu/bin/python -m tools.m5d_surface_index --index /root/gpufree-data/hs-drive-data/processed/m5d_test_index
.venv-gpu/bin/python -m tools.m5d_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5d_test_index --split test
.venv-gpu/bin/python -m tools.m5d_historical_models
.venv-gpu/bin/python -m tools.m5d_data_analysis --indices /root/gpufree-data/hs-drive-data/processed/m5d_index /root/gpufree-data/hs-drive-data/processed/m5d_test_index
.venv-gpu/bin/python -m tools.m5d_visualize
```

```bash
# 结束其余GPU/标签任务后执行，独占同一个CARLA实例，不并发tick owner
.venv-gpu/bin/python -m tools.m5d_runtime_campaign
.venv/bin/python -m tools.m5d_async_drive --output reports/m5d_runtime_faults.json --artifacts reports/m5d_runtime_faults --scenarios mild_curve junction_turn slow_worker worker_crash emergency static_obstacle stalled_worker invalid_input --load-description 'isolated fault and route regression'
.venv-gpu/bin/python -m tools.m5d_runtime_analysis
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5d_tests.xml
.venv/bin/python -m pytest -q tests/test_geometry_extended.py tests/test_imaging_rig.py tests/test_session.py tests/test_snapshot.py --junitxml=reports/m5d_carla_api_tests.xml
```

本轮初次合成Witness测试的整数depth dtype、初次Runtime smoke日志字段引用未初始化变量均修复；失败log和source目录保留。后续验收不能删除失败或把所有场景Passed传播为严格实时/纯视觉安全通过。所有记录以 `m5d_summary.md` 最终结论为准。
