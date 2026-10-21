# M5-I — Reliable Visual Representation Transfer & Depth Qualification

基线`4e50203`，独立分支`feat/m5i-visual-transfer-depth-qualification`。所有新增报告/源码/Checkpoint使用m5i前缀；历史资产只读。默认Encoder、Depth48LID、HeightSplat、4×160×120BEV、控制器与Physics不变。无视觉控制、无大型Occupancy/Temporal/SparseQuery训练。

## 预训练与对照

原ResNet18来自官方ImageNet1K V1 `resnet18-f37072fd.pth`，经M4 CARLA优化后作为所有ResNet组初始状态。SHA与manifest/加载实现验证见`m5i_baseline_review.json`。不能描述为从零训练。

架构候选为本机Torchvision0.23.0实际支持的MobileNetV3-Large `IMAGENET1K_V2`，官方模型5,483,032参数（含分类头）；本研究只使用features，在index6/12/16提取真实S8/S16/S32的40/112/960通道，再投影64通道相加，经3×3/GN/ReLU与原PixelHead输出原图。原分类头不参与推理。采用同ImageNet归一化，保持完整640×384图像，不做分类中心裁剪。Backbone冻结，优化FPN/PixelHead，BN统计固定。它与CARLA适配过的ResNet在架构/预训练/初始化均不同，属于架构对照，不是单变量蒸馏；没有排除完整微调的潜力。

[官方模型与权重](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_large.html)、[v0.23代码BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE)、[预训练权重说明](https://github.com/pytorch/vision/blob/v0.23.0/docs/source/models.rst)已核对。代码许可不自动等于训练数据派生权重的无限制许可；仅本地研究，不分发权重。精确SHA/本机源码/版本在`m5i_pretrained_provenance.json`。

A=`GT BCE`；B=`GT BCE +0.25 outputKD`；C（仅B验证条件失败时）=`GT BCE+0.1 featureKD`；Mobile=`同GT BCE`。两seed101/102、相同图像/相机次序、共享全部Train/Val数据、2400步、每400评测，无自动加步数。GT正/负图内均值分别算后平均，Unknown不参与。A/B初始网络与优化参数一致；C单独训练适配层但推理结构同A。采样复用M5H全视角遍历后逆频率补采；每个最佳Checkpoint的实际step/已见视角另记，不能把最终覆盖归给早期最佳模型。

## Teacher输出与特征合同

Teacher仍是官方Mask R-CNN R50FPNv2 COCO_V1，只输入RGB、离线缓存实例与像素分数。旧M5H Train/Val预测按SHA复用；新增Train/Val另缓存。所有教师预测是参考，不覆盖GT。

设可靠GT有效V、目标Y、类别C，教师支持类别score `q=max(instance_score×mask_probability)`，`q_trusted`仅含实例置信度≥.7。输出KD：

- `M_pos = V & Y & C∈{vehicle,person} & q_trusted_same_class≥.7`。
- `M_neg = V & ~Y & max(q_vehicle,q_person)≤.05`。
- `L_output = mean(nonempty(mean_Mpos BCE(student,q), mean_Mneg BCE(student,q)))`。

教师零分不等于确认背景；这里背景真实性来自可靠GT。GT正类漏检、cone、Unknown、教师高分GT负类冲突均不作KD负样本。主GT不变。边界/遮挡忽略区只能报告Unknown响应，不能宣称教师一定错误。

C选Teacher FPN key`1`（P3，stride8，256通道）。Teacher native384×640经官方transform为799×1333，pad800×1344。Student卷积采样中心native`u=8j,v=8i`（不是几何Lift特征单元中心4::8）；映射`u'=(u+.5)*1333/640-.5`，教师卷积index`u'/8`，grid_sample align_corners=False。原图有效范围映射，padding不作真实图像。教师与学生感受野不同，该映射只对齐中心，不保证相同内容感受野。

缓存映射后`[6,256,48,80]` BF16，存储成本与实测时间在index记录。Student64通道经训练用1×1适配至256；沿通道L2归一化，误差`1−cosine`。可靠目标、可靠背景、可靠目标边界分别平均后等权，Unknown/不支持的目标区域排除；可靠性3×3保守检查，边界5×5形态学环。适配层不在部署路径，无GT或Teacher特征进入正式forward。

## 数据与Depth资格

新增4Train/2Val位置，16配置/位置，包括六方车辆、侧后行人/锥桶、25/32mSprinter和远行人、Empty/Removed。补数据统一用于所有视觉模型。深度候选不训练，原Head/M5H Replay101step1000/102step1200只读评估完整Val和新Test。

新Test6独立位置：Town01两处、Town03/04/05/10HD各一。Town02无满足45m候选，Town06未安装，两次失败几何分配日志保留；最终不降低距离。所有旧位置（含已打开M5HTest）和M4路线纳入避让。协议CARLA坐标，记录与M4 index为HS坐标，比较前统一反射Y。独立位置不等于新Actor资产/布局。

预声明支持：每相机Train至少3正Actor位置、Val至少1位置；各类别Train至少4位置、Val至少2。统计Actor-view而非零散语义像素；距离/面积/类别/天气单列。

Depth固定48LID和FP32几何，Expected MAE为Softmax期望光轴Z；另报Mode、SignedBias、GTBin及±1bin、层质量、同层0.5/1/2m、范围外质量与三维误差。每场景近场≤1.05原Head且原M4S16Val也通过。绝对全局Bias最多比原增0.25m，10–20/20–35各段最多增0.5m。两距离段Actor同层1m概率不得下降、Oracle2D Recall最多下降1百分点。20–35可靠支持要求至少100条sample-camera-UV射线、4位置、2目标类型；这些射线相关，不称统计独立真值。全部条件共同决定研究候选资格；原默认不改。没有温度拟合或全局偏移补偿，避免用校准掩盖本轮原候选资格。

Shadow固定P8中心UV/K/T/网格/HeightSplat/0.2事件阈值；正式学生不使用GTMask，教师/Oracle只作离线诊断。六组合为M5GPixel/新Student × 原Depth/Replay，以及Teacher+OracleDepth、Oracle2D+OracleDepth。若Student没有通过联合门禁，只能叫最佳诊断候选。可见背景表面不是安全Free，Unknown不作负样本，粗体素中心误差不是精确Actor误差。

## 复现

`.venv`为CARLA0.9.15；`.venv-gpu`为PyTorch2.8.0 CUDA12.8 Torchvision0.23.0，实测RTX4090。已有目标拒绝覆盖；重跑应使用独立工作树与新的输出命名，保留失败日志。以下顺序中每一步完成再执行依赖步骤。

```bash
.venv-gpu/bin/python -m tools.m5i_prepare
.venv-gpu/bin/python -m tools.m5i_lineage
.venv-gpu/bin/python -m tools.m5i_weights
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5i_capture_protocol
.venv/bin/python -m tools.m5f_capture --protocol configs/m5i_capture_train.json --output /root/gpufree-data/hs-drive-data/raw/m5i_added_train
.venv/bin/python -m tools.m5f_capture --protocol configs/m5i_capture_val.json --output /root/gpufree-data/hs-drive-data/raw/m5i_added_val
bash tools/m5i_process_added.sh
.venv-gpu/bin/python -m tools.m5i_coverage --suffix _complete
.venv-gpu/bin/python -m tools.m5i_capture_audit
.venv-gpu/bin/python -m tools.m5i_cache
.venv-gpu/bin/python -m tools.m5i_teacher_audit
.venv-gpu/bin/python -m tools.m5i_teacher_classes
.venv-gpu/bin/python -m tools.m5i_depth_original_task
.venv-gpu/bin/python -m tools.m5i_depth_eval
.venv-gpu/bin/python -m tools.m5i_depth_support
.venv-gpu/bin/python -m tools.m5i_training
# Only if feature_condition_met=true:
.venv-gpu/bin/python -m tools.m5i_teacher_features
.venv-gpu/bin/python -m tools.m5i_training --feature
.venv-gpu/bin/python -m tools.m5i_evaluate
.venv-gpu/bin/python -m tools.m5i_evaluate --teacher
.venv-gpu/bin/python -m tools.m5i_gpu_contract
.venv-gpu/bin/python -m tools.m5i_selected_exposure
.venv-gpu/bin/python -m pytest -q > reports/m5i_logs/full_tests_pretest.log 2>&1
.venv/bin/python -m pytest tests/test_capture_config.py tests/test_imaging.py tests/test_imaging_rig.py tests/test_m5a_contracts.py -q > reports/m5i_logs/carla_python_tests.log 2>&1
.venv-gpu/bin/python -m tools.m5i_record_tests
.venv-gpu/bin/python -m tools.m5i_freeze
```

冻结后一次封存Test流程：

```bash
.venv/bin/python -m tools.m5f_capture --protocol configs/m5i_capture_test.json --output /root/gpufree-data/hs-drive-data/raw/m5i_sealed
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5i_sealed --output /root/gpufree-data/hs-drive-data/processed/m5i_test_index --report reports/m5i_test_evidence_quality.json
.venv/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5i_test_index --output /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --splits test
.venv-gpu/bin/python -m tools.m5i_teacher --root /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --split test
.venv-gpu/bin/python -m tools.m5i_evaluate --root /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --split test --teacher
.venv-gpu/bin/python -m tools.m5i_evaluate --root /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --split test
.venv-gpu/bin/python -m tools.m5i_depth_eval --root /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --split test
.venv-gpu/bin/python -m tools.m5i_coverage --root /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --suffix _test
.venv-gpu/bin/python -m tools.m5i_capture_audit --test
.venv-gpu/bin/python -m tools.m5i_test_split_audit
.venv-gpu/bin/python -m tools.m5i_depth_support --split test
.venv-gpu/bin/python -m tools.m5i_depth_report
.venv-gpu/bin/python -m tools.m5i_shadow
.venv-gpu/bin/python -m tools.m5i_latency
.venv-gpu/bin/python -m tools.m5i_figures
.venv-gpu/bin/python -m tools.m5i_findings
.venv-gpu/bin/python -m tools.m5i_finalize
.venv-gpu/bin/python -m tools.m5i_artifacts
```

独立GPU延迟需先停止本任务CARLA与其他GPU实验；只计GPU推理，不包含控制关键路径。原Runtime继承Partial，不重复长期压力矩阵。测试成功、视觉学习、空间泛化、Depth候选资格、安全与实时性分别报告。

首次官方权重 urllib 下载在约94%长时间停顿，随后从同一官方 URL 使用 curl 成功下载并核验完整 SHA；原 partial 和失败日志保留，未替换为不明来源权重。`m5i_weight_download_recovery.json`记录恢复。
