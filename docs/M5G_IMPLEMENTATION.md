# M5-G — Visual Representation Benchmark & Depth Projection Research

基线`5e44a77`，独立分支`feat/m5g-visual-representation-depth`。原Depth/Splat/48LID/BEV/Runtime默认行为不变。新代码只通过显式`tools.m5g_*`启用，GT不进入正式推理。历史资产SHA在`m5g_baseline_freeze.json`先封存。

## 数据与预算

旧M5F Train192帧/12位置、Val96帧/6位置，最近同地图Train/Val距离84.11m，无跨集合相同输入SHA。位置不同但重复车辆资产和布局模板，不能宣称布局完全独立。历史F Test仅用于采样覆盖回归。所有Pixel监督沿用M5F首可见目标表面语义，不代表3D Free。

读取真实400步schedule逐camera计数，正负视角、语义、距离、大小逐项保存。seed101/102仅293/314个独特视角，覆盖25.43%/27.26%；859/838视角未见。前视正样本更新180/155次，右后为0（训练集本身0正视角）。微观计数不是独立场景样本数。

`m5g_protocol.json`训练前固定规则。10个1200步实验：两seed×fpn_bce/fpn_hard/fpn_weighted/head_bce/high_bce；原模型、Head初始化与前400schedule完全一致，随后同generator扩展，不更改相机分布以免混淆预算归因。head_bce冻结Backbone/FPN；fpn_*冻结Backbone并训练FPN；high_bce仅解冻layer3/4，layer2缓存仍是RGB特征，BN统计固定。Head LR1e-3、FPN1e-4、高层1e-5。单camera/步，全六摄评测。

BCE对可靠正负像素分别均值后等权。Hard负项为0.5全可靠负均值+0.5困难负均值；困难集合=当前可靠负logit最高10% ∪ 同组Empty/Removed中其他布局目标位置的可靠负像素。GT配对只定义监督，不输入推理。Weighted对照正负权重1:2，除此相同。Unknown梯度严格0。

逐200步完整Train/Val BCE、AP、Precision、Recall、EmptyFPR。旧M5F没有保存历史ValLoss/TrainAP，不能事后编造；旧日志与本轮匹配重跑分开。旧M5F PR图正负hist行颠倒，但JSON指标函数使用正确顺序；本轮图修正、旧图原样保留并明确缺陷。

模型/阈值仍用逐场景Precision≥.8、Recall≥.8、EmptyFPR≤.01联合约束。不可行时minimax violation→macroIoU→较早step选研究候选，不授予质量通过。低FPR曲线可解释Precision/Recall取舍，Test上仅描述性，绝不回选阈值。

仅当两seed fpnbce的ValAP1200均≥800且均值增益≥.01，允许额外1200步至2400。本轮满足，独立报告`m5g_budget_extension.json`，从真实1200状态和Adam优化器继续。它不能与1200结构实验混称单变量结构收益。

## 成熟视觉参考

Torchvision官方Mask R-CNN ResNet50 FPN v2 COCO_V1，46,359,409参数，来源与SHA见`m5g_pretrained_provenance.json`。代码[BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)；[官方模型页面](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)；[预训练权重条款说明](https://docs.pytorch.org/vision/stable/models.html)要求同时考虑训练数据派生条款，不能把代码许可当作权重无限制商业授权。本轮外部研究参考，未重新分发权重。

原图RGB640×384，模型内部短边800、长边最大1333；FP32逐camera推理。实例confidence×pixel mask probability，按支持类别取max。COCO person1，vehicle2/3/4/6/7/8；cone无对应类别，Ignore且单列未支持。Vehicle与Ped分别评价，二者合并Mask与HS模型对齐。HS单binary Head的分类型指标只是条件分类覆盖，不等于多类别分类器。成熟模型不同预训练数据、结构、输入分辨率与计算量，绝非严格单变量对照。没有CARLA微调。

官方权重单连接较慢后采用HTTP Range并行下载，逐段Content-Range/长度及整体官方SHA前缀验证；失败/部分下载记录保留。

## Oracle采样

原像素全采样、stride4/8/16，phase0/中心/末端，GT连续光轴深度与相同K/T投影到四层160×120。参考用全分辨率≥3端点且正负混合格忽略，采样预测只需>=1有效目标射线。原相位stride8=4和stride16=8与M5F完全一致。完整DenseOracle应精确覆盖全部参考表面，失败即停止几何结论。

记录native像素、sampled rays、覆盖/未覆盖格、0.5/1m同层Recall、距离/Actor大小/相机与相位。Stride4只是采样实验，没有虚构Stride4神经特征。共享CARLA渲染参考，仅自一致性；AABB关联不等于独立Mesh。

## Depth

固定原M4视觉Encoder与cameraMLP，真实P8 RGB条件特征；仅DepthHead训练。原48 LID不变。监督沿用可确认当前表面、自车过滤、语义/身份边界mask，再加3×3深度spread≤.5m、有限1.5–80m。Mask仅训练/eval，正式forward不接收GT。此mask偏向可靠内表面，对斜面/边缘覆盖有选择偏差。

原Head与CE、CE+0.2空间NLL，两seed各1200同序列。光轴距离1.5–10/10–20/20–35/35–60/60–80，各支持段等权，段内Actor/背景各支持侧等权。空间项为`-log Σ p(bin)`，仅同高度层且XY距离≤1m的bin；没有满足邻域的bin时跳过空间项但保留CE，不更改Bin补救。此等权采样改变训练风险分布，可能牺牲近场密集道路性能，必须独立验收。

Val选择先限制全可靠近场ExpectedDepth MAE≤原Head×1.05，再最大化10–35m目标空间质量，失败时step0可被选中。本轮所有训练候选都被该约束拒绝，合格候选为原Depth。另保留全部最终1200模型的诊断评测，Shadow选择其中Val空间质量较高者并明确`depth_exploratory=True`；不把失败候选变成发布候选。Test不得纠正训练。

Depth metrics：Expected/Mode MAE、Bias、GT Bin及±1Bin概率、正确层、同层0.5/1/2m质量、完整分布期望3D距离、ExpectedDepth点3D距离。层/BEV概率限定GT点落入BEV，另报告其分母；Depth MAE统计全可靠深度。用Oracle2D固定输入测投影Recall，排除视觉误差。新测试报告全部训练最终候选及原Head；通过近场门禁的选择报告与最终失败诊断严格区分。

## 新测试与Shadow

模型、阈值、监督、预算与checkpoint封存后采Town02/03/05各1新位置，每位置12布局：10/25/32m前车、侧前/侧后/后车、25m行人、30m交通锥、遮挡、Empty/Removed。距既往M5A–F位置≥45m，支持数实际审计；侧方静态配置是感知实验，不能当驾驶路线。独立位置只有3个，不能以像素量宣称显著泛化。

推理五组：旧Pixel/旧Depth，新Pixel/旧Depth，Oracle2D/新Depth，新Pixel/新Depth，Oracle2D/OracleDepth。共同P8网格，原Depth固定原Encoder输入，新Pixel独立副本；Oracle组仅诊断，正式学习组无GTMask。固定事件阈值0.2，不重新标定Test。事件mass非认证Occupancy，Unknown高分只统计分布，旧目标格在Empty的输出仅事件残留，不是物理Absence FPR。

正式二维消失FPR在Present目标且Empty/Removed可靠非目标的相同像素Mask上计算。全部对象已消失布局可含背景停放车辆，背景车辆不当负类。

Runtime不修改，无视觉控制指令。六摄独立计时在CARLA/其他GPU实验停止后进行；不含传输/render/tick/control，不能关闭Runtime或纯视觉驾驶安全门禁。

## 复现顺序

在相同基线资产可用的独立checkout执行，已存在目录/报告拒绝覆盖，避免改写本轮或历史结果。

```bash
.venv-gpu/bin/python -m tools.m5g_exposure
.venv-gpu/bin/python -m tools.m5g_sampling
.venv-gpu/bin/python -m tools.m5g_sampling_strata
.venv-gpu/bin/python -m tools.m5g_pixel_training
.venv-gpu/bin/python -m tools.m5g_extension
.venv-gpu/bin/python -m tools.m5g_depth_data
.venv-gpu/bin/python -m tools.m5g_depth_training
.venv-gpu/bin/python -m tools.m5g_pixel_eval
.venv-gpu/bin/python -m tools.m5g_depth_eval
.venv-gpu/bin/python -m tools.m5g_depth_eval --final
.venv-gpu/bin/python -m tools.m5g_download
.venv-gpu/bin/python -m tools.m5g_pretrained
.venv-gpu/bin/python -m tools.m5g_freeze
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5g_test_protocol
.venv/bin/python -m tools.m5f_capture --protocol configs/m5g_capture_protocol.json --output /root/gpufree-data/hs-drive-data/raw/m5g_sealed
.venv-gpu/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5g_sealed --output /root/gpufree-data/hs-drive-data/processed/m5g_test_index --report reports/m5g_test_standardize.json
.venv-gpu/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5g_test_index --output /root/gpufree-data/hs-drive-data/processed/m5g_test_pixels --splits test
.venv-gpu/bin/python -m tools.m5g_pixel_eval --root /root/gpufree-data/hs-drive-data/processed/m5g_test_pixels --split test
.venv-gpu/bin/python -m tools.m5g_pretrained --root /root/gpufree-data/hs-drive-data/processed/m5g_test_pixels --split test
.venv-gpu/bin/python -m tools.m5g_depth_eval --root /root/gpufree-data/hs-drive-data/processed/m5g_test_pixels --split test --final
.venv-gpu/bin/python -m tools.m5g_sampling --root /root/gpufree-data/hs-drive-data/processed/m5g_test_pixels --report reports/m5g_oracle_test_sampling.json
.venv-gpu/bin/python -m tools.m5g_shadow
.venv-gpu/bin/python -m tools.m5g_figures
.venv-gpu/bin/python -m tools.m5g_digest
.venv-gpu/bin/python -m tools.m5g_post_audit
# 停止本轮CARLA进程和其他GPU作业后单独计时
.venv-gpu/bin/python -m tools.m5g_latency
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5g_tests.xml
.venv/bin/python -m pytest -q tests/test_session.py tests/test_snapshot.py tests/test_imaging_rig.py tests/test_geometry_extended.py --junitxml=reports/m5g_carla_tests.xml
```

补充命令：

```bash
.venv-gpu/bin/python -m tools.m5g_gpu_contract
.venv-gpu/bin/python -m tools.m5g_depth_figures
.venv-gpu/bin/python -m tools.m5g_artifacts
```

已知失败与复现限制：Oracle采样首次把flatten端点rank与二维mask直接相与，shape错误，修正为原图形状后Dense100%通过；2400延长决策首次numpy.bool无法JSON序列化，转Python bool后按原规则执行；预训练推理首次早于权重下载完成而FileNotFound，下载依赖就绪后重试；验证调度误重复调用已有Depth评测触发FileExistsError并拒绝覆盖。以上失败日志保留，无测试阈值调优。

标准化/采集直接调用已有M5A/M5F工具，新的输出目录和报告使用m5g前缀；原始manifest内部已有schema名继续兼容，不代表旧资产改写。手动复现启动CARLA是常驻服务，应在独立终端运行。

封存后仅新增了一条单测，验证显式传入阈值时评测函数不重新选择工作点；`m5g_test_audit.json`中该测试文件SHA变化如实为false，训练、模型、标签与评测科学源码均不变。完整测试原261通过、5跳过；新增后M5G局部6通过，最终全套再次记录。CARLA环境相关14通过用于覆盖缺失CARLA包的用例，不与全套计数直接相加。

`tools.m5g_background_audit`对冻结测试预测按确认道路地面/人行道/建筑墙面/护栏/植被/道路设施分解误报。它只做诊断，不使用Test做困难样本训练。缺少精确“车辆形状静态物体”及透明反射内表面的独立标注，不能将语义类别代理当完整难例覆盖。

`tools.m5g_lowfpr`提供一个仅从保存Validation直方图选出的补充FPR≤0.001工作点，再描述Test结果。该补充是在主测试后整理，不是预声明验收条件，绝不用于重新选模型或替换主阈值。

外部Mask R-CNN六摄推理成本与HS轻量模型差异很大；Standalone记录含默认800短边FP32逐摄像头实例分割，不能把该方法直接接入20Hz控制。它的用途是视觉能力参考及下一阶段蒸馏/强预训练研究依据。

```bash
.venv-gpu/bin/python -m tools.m5g_background_audit
.venv-gpu/bin/python -m tools.m5g_lowfpr
```

最终全套262 passed、5 skipped；M5G局部6 passed；CARLA环境14 passed。再次调度已存在的Depth/PostAudit报告触发拒绝覆盖，旧报告未被改写。源码diff检查与compileall完成；原失败日志及下载部分文件保留。不同研究作业的训练耗时可能含GPU并发影响，独立推理计时在所有GPU实验停止后重新测量，不能将并发训练时间与独立推理时延混为同一性能条件。
