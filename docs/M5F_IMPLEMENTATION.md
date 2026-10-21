# M5-F — Image-Space Grounding & BEV Geometry Factorization

基线`bf989b4`，分支`feat/m5f-image-grounding`。原DepthHead、48LID、HeightSplat、四层160×120、控制与Safety不改。所有新接口显式调用。GT进入label/diagnostics路径，不进入`ForegroundModel.forward`或正式foreground lift。

## 像素语义合同

`hsd-m5f-visible-first-surface-v1`：正类=当前可靠可见首表面目标Actor；负类=确认非目标首表面；Ignore=未知/无效/自车/边界。建筑墙体为这项**图像目标分类**负类，不是Free，不排除射线后方车辆。原SpatialEvidenceV2、GoldWitness保持不变。

本机CARLA0.9.15 `CityObjectLabel`、原manifest与实采联合审计，Pedestrian12/Rider13，Car14/Truck15/Bus16/Train17/Motorcycle18/Bicycle19。TrafficCone01实采tag21。Dynamic21/Static20/Other22绝不全量当负或正；交通锥仅在已知cone Actor AABB±0.10m内的端点与renderID关联时接受。每renderidentity至少3像素、90%相关端点落在单一候选Box，重叠候选拒绝。RenderID不是ActorID，仍只是保守几何关联，不等于独立Mesh身份真值。

负tag白名单1–10（其中不含Sky11）及24–28；Water23、Sky11、Unknown0、未确认20/21/22保持Ignore。标签深度1.5–80m，仅用于当前表面可靠性与self过滤。相同3×3 semantic+instance内点保留，1像素轮廓带忽略。ego-removed参考沿用已验证自车mask，没有采用粗射线bbox整段过滤。

旧RGB/Depth/Instance采集代码对frame/timestamp有assert，落盘旧格式只存共享frame时间，不能声称逐传感器时间元数据已全部保留。新测试额外存每传感器frame/time。对齐检查包括SHA、图像尺寸、K/T、RGB边缘与语义轮廓近邻的诊断；该指标不是精确像素真值。透明玻璃/反射内表面尚无独立mask，只能列限制。

## 网络、采样、训练

PixelHead=64→32的3×3Conv+GroupNorm+ReLU+1×1前景logit，双线性映射640×384，`align_corners=False`；网格中心`(j+.5)*stride-.5`。原ResNet18 RGB backbone冻结。S16/P8均真实既有FPN路径，P8使用layer2，不是S16输出插值。

三组：S16冻结、P8冻结、P8的lateral/output有限微调。第三组是独立encoder副本，原Depth及Context继续原冻结特征，避免间接改变Depth输入分布。两个seed101/102，各400步。训练复用M5D/E的12train位置192帧、6val位置96帧。相同group轮转与frame/camera采样：一半目标支持camera、一半六摄均匀。一个camera/步，完整六摄评测。冻结RGB backbone缓存；GT监督独立读取。

可靠正像素BCE均值与可靠负像素BCE均值等权；某侧为空只用另一侧。无pairloss，不把网络结构变化与额外pairloss捆绑。Unknown梯度0。全部预算固定，不根据Test追加步数。

每100步验证，阈值0.1…0.9。逐场景Precision≥.8、Recall≥.8、空场景FPR≤.01；不可行时按最大归一化违反量最小→macroIoU高→较早step选择研究结果，标Partial。AP用固定256bin直方图近似，名称`histogram_ap`；阈值落到floor(t*256)/256，保存u8概率误差<1/256。不能用总体accuracy冒充识别成功。

Actor级统计按几何关联renderidentity，>=3有效像素eligible；>=3预测像素且可见像素recall>=50%才检出。大小<=32/33–128/>128像素；零支持保留。摄像头Actor观测与跨六摄Actor配置不是同一个独立样本，报告分别聚合。成对评测使用当前目标像素与Empty/Removed确认非目标像素交集，不依赖三维GoldAbsence。配对只用于评价，单帧推理无A/B拼接。

## 四组几何与独立Lift

A Oracle2D+continuousGTDepth；B Oracle2D+原Learned48Bin；C Learned2D+GTDepth；D Learned2D+LearnedDepth。同一模型网格内四组UV、有效mask、K/T、BEV和matching严格相同。nativeS16/P8之间网格变化必须披露，另给commonS16诊断。可靠参考使用native像素≥3端点/格，正负混合格忽略，故A也受featuregrid稀疏限制。A是同渲染器自一致性，不是独立精确物理真值。

`models/m5f_lift.py`输出foregroundmass、totaldepthmass、ray-bin supportcount和`1-exp(-foregroundmass)`事件分数；没有除以微小totalmass，不称物理Occupancy概率。probability分布的supportcount是ray-bin落点数，不一定是独立射线数；四层接口保留。正式D不接受GTmask，masked四组只为公平诊断，另存unmaskedD。

旧BEV Evidence作为独立baseline，不能拿旧Head接Oracle分布后的结果称重新训练上限。候选融合为oldlogit+log1p(foregroundmass)+log1p(totalmass)的标量线性logit；与old-only重新标定匹配100步/两seed/训练frame顺序。只训练可靠当前表面，Unknown不负监督；不替换默认BEVEncoder。

## 新封存测试与Shadow

在全部pixel模型、阈值、几何评价阈值和小融合训练冻结后采3个新位置（Town03/04/05，离所有M5A–E位置45m以上），每组12配置。车辆7/12/18/25/32m、大车、行人、cone、遮挡、横移、empty/removed。更换WetCloudyNoon/CloudySunset/SoftRainSunset。新位置同地图，不能宣称未见地图或所有坡面天气覆盖。

OfflineShadow读取保存六RGB，输出foregroundBEV事件，关联参考状态/表面做评价，完全不产生CARLA控制。Runtime没有改动；保留M5E Partial及617ms历史长尾，不重跑无新增信息的长压力矩阵。推理时延单独GPU同步计时，不包含在线传感器传输/控制，也不授予闭环Passed。

## 命令

```bash
.venv-gpu/bin/python -m tools.m5f_dataset
.venv-gpu/bin/python -m tools.m5f_cache
.venv-gpu/bin/python -m tools.m5f_learning
.venv-gpu/bin/python -m tools.m5f_evaluate
.venv-gpu/bin/python -m tools.m5f_factorization
.venv-gpu/bin/python -m tools.m5f_posttrain
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5f_test_protocol
.venv/bin/python -m tools.m5f_capture --output /root/gpufree-data/hs-drive-data/raw/m5f_sealed
.venv-gpu/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5f_sealed --output /root/gpufree-data/hs-drive-data/processed/m5f_test_index --report reports/m5f_test_standardize.json
.venv-gpu/bin/python -m tools.m5f_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5f_test_index --output /root/gpufree-data/hs-drive-data/processed/m5f_test_pixels --splits test
.venv-gpu/bin/python -m tools.m5f_evaluate --index /root/gpufree-data/hs-drive-data/processed/m5f_test_pixels --split test
.venv-gpu/bin/python -m tools.m5f_factorization --index /root/gpufree-data/hs-drive-data/processed/m5f_test_pixels --split test
.venv-gpu/bin/python -m tools.m5f_bev_features --index /root/gpufree-data/hs-drive-data/processed/m5f_test_pixels --split test --output runs/m5f_bev_test_features
.venv-gpu/bin/python -m tools.m5f_fusion --test
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5f_tests.xml
```

输出存在拒绝覆盖；重新实验须独立checkout/新目录与新hash。data/checkpoints保留服务器并SHA封存；Git交付报告、代码、图、索引与日志。CARLA0.9.15官方语义参考：https://carla.readthedocs.io/en/0.9.15/ref_sensors/#semantic-segmentation-camera；实际映射以本机enum与`m5f_semantic_version.json`为准。

## 完整复核与补充命令

以下命令均在本轮执行，训练/封存Test协议不变。补充分析脚本只读取冻结预测或运行原冻结网络，不能据此重选模型。

```bash
.venv-gpu/bin/python -m tools.m5f_feature_retention
.venv-gpu/bin/python -m tools.m5f_gpu_contract
.venv-gpu/bin/python -m tools.m5f_label_audit
.venv-gpu/bin/python -m tools.m5f_shadow
.venv-gpu/bin/python -m tools.m5f_digest
.venv-gpu/bin/python -m tools.m5f_geometry_supplement
.venv-gpu/bin/python -m tools.m5f_surface_strata
.venv-gpu/bin/python -m tools.m5f_pair_scene_audit
.venv-gpu/bin/python -m tools.m5f_depth_at_targets
.venv-gpu/bin/python -m tools.m5f_bev_baselines
.venv-gpu/bin/python -m tools.m5f_visualize
# 专门延迟实验需 CARLA 与其他 GPU 作业均停止；脚本保留 nvidia-smi 进程快照。
.venv-gpu/bin/python -m tools.m5f_latency_isolated
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m5f_tests_final.xml
.venv/bin/python -m pytest -q tests/test_session.py tests/test_snapshot.py tests/test_imaging_rig.py tests/test_geometry_extended.py --junitxml=reports/m5f_carla_contract_tests.xml
```

GPU环境全套256 passed、5 skipped（此虚拟环境未安装CARLA Python包）；`.venv`补跑相关14 passed，包括5个CARLA API用例。二者存在重复用例，不应相加成270个独立测试。另有实际RTX4090六摄合同检查，核对BF16 S16辅助特征与旧Encoder完全一致、监督数据变动不影响推理、拒绝6类特权字段。

第一次新增lift单测把训练manifest当作BEV几何配置，产生KeyError；修正测试fixture为`configs/bev.yaml`后通过，未修改模型或训练。失败XML/log保留。旧数据/报告/Checkpoint的原SHA及最终校验、当前来源文件SHA、模型SHA、协议、原始数据和报告SHA见`reports/m5f_artifact_manifest.json`。

`m5f_evaluate`历史字段名`strata/ground`实际是全部确认非目标像素，不能解释为Ground语义。本轮不改已冻结评测值，新增`m5f_surface_strata.json`明确道路地面tag1/2/10/24/25、目标Actor与其他非目标、Ignore。`empty`原字段只统计整帧无任何目标的场景；`m5f_pair_scene_audit.json`覆盖全部3组Empty/Removed布局，排除背景车辆及Unknown，仅以确认非目标表面计算FPR。

四组正式阈值来自Val且各模式不同，不能直接把它们的Recall差当纯几何因果效应。`m5f_geometry_supplement.json`固定所有模式事件阈值0.2，提供共同工作点及距离/高度层/场景分解。BEV Empty/Removed在原目标格的输出只叫事件残留，不称精确Absence FPR，因为当前射线可能观测不到该格。精确二维配对FPR仍使用两帧当前表面可确认的同像素。

Stride8直接使用冻结S16 Depth Head于原P8特征，属于分辨率/输入分布诊断，未训练P8Depth；不能据此推出训练后的P8Depth能力。`m5f_shadow_evaluation.json/commonS16`和融合对照都使用共同S16 Depth/UV，以辅助区分Pixel特征和投影网格变化。三维误差使用体素中心和最近邻，不是精确Mesh表面定位误差。

封存测试三个位置分别是直道、路口、直道。没有新增坡道质量覆盖，未覆盖未见地图；目标配置均主要在前视摄像头，六摄背景统计不能替代环视障碍事件泛化。行人与交通锥各3个配置、图像尺寸44/101有效像素；严格≤32像素小目标无支持，不授予小目标通过。33个Actor配置中30个有有效像素支持，3个完全遮挡目标保留为零支持。Actor表面覆盖率允许大片误报命中目标，必须与Pixel Precision和局部消失残留一起解释。

保存资产清单命令（总结、验收报告及源码提交就绪后执行）：

```bash
.venv-gpu/bin/python -m tools.m5f_artifacts
```

二维输出有效区域与位置可靠性分别报告。`surface_strata`只有正类子集的AP=1是条件指标的退化结果，没有负例时无判别意义；不能作为性能证据。仅可信完整正负Mask下的PR/AP及配对结果。
