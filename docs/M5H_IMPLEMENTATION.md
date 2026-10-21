# M5-H — Efficient Visual Distillation & Near-Preserving Depth

基于 `bf0c38d`，独立分支 `feat/m5h-distill-near-depth`。所有工具仅写 `m5h_` 新资产，已存在目标路径拒绝覆盖。旧模型、48 LID、四层 `[4,160,120]`、HeightSplat、默认控制器均保留。所有学习结果均为离线研究，不发送车辆控制。

## 监督与计算合同

Student 保持原 ResNet18 + 真 P8/FPN + M5-F PixelHead，训练仅 FPN/PixelHead。冻结 RGB Backbone 的 layer2/3/4 缓存只包含 RGB 特征；GT、Teacher 与 Depth 位于独立监督路径。BF16 网络计算，概率与几何 FP32。GT 接口拒绝测试见 `m5h_gpu_contract.json`。

GT BCE 保持 M5-G 的图内正/负分别取均值后平均。Teacher 为官方 Mask R-CNN R50FPNv2 COCO_V1，权重与许可来源沿用 `reports/m5g_pretrained_provenance.json`；不重新分发。Teacher 对六摄 RGB 顺序推理，离线缓存实例 labels/scores/boxes/masks、像素最大 `instance_score × mask_probability`，训练不重复调用 Teacher。原640×384输入经官方缩放实际为1333×799，填充1344×800；缓存初版名义尺寸错误由 `m5h_teacher_resize_verification.json` 更正，原记录保留。

对于像素 i，定义可靠GT有效性 V、目标Y、类别C，以及仅包含实例置信度≥0.7的同类教师分数 q。蒸馏 Mask 为：

`M = V & Y & (C in {vehicle,person}) & (q_same_class >= 0.7)`。

`L = L_GT + 0.25 × mean_M BCEWithLogits(student, q)`。

Teacher 所有负预测、低置信度、GT冲突、Unknown和交通锥均不参与蒸馏。锥桶仍由可靠GT主监督。该保守策略不能说明其他形式的特征蒸馏或可靠背景蒸馏必然无效。分数不是物理占据概率。

采样先随机无放回覆盖全部1248训练视角，其余步数按场景、相机、天气、正/背景、类别、距离、目标尺寸、Empty/Removed的逆频率混合权重补采。两个种子101/102，各GT/Distill2400步；相同初始化与顺序。每400步全Train/Val评测；按旧逐场景 P≥.8、R≥.8、EmptyFPR≤.01 的 minimax violation、macroIoU、较早step选择。选中的Checkpoint可能在完整遍历之前，必须结合 `m5h_selected_exposure.json` 阅读，不能把最终训练曝光赋给早期最佳Checkpoint。

深度独立训练复用M5G固定192帧Train/96帧Val及原缓存，不混入新增16帧Pixel训练数据：仅原DepthHead，LR2e-5，1200步、每200评测。新数据采用所有可靠射线均值CE，取消原五距离段/目标均衡。Replay来自M4原始manifest实际训练ID，原S16与P8近场1.5–10m；训练集不包含原Val。B=`CE_new + CE_replay`；C=B+10×平均近场KL(old||new)，KL覆盖新样本及Replay可靠近场。原分布约束并非GT。所有Deep参数从同一原Head初始化，B/C两seed同序列。当前P8 Val与原M4 S16 Val均须满足近场MAE≤1.05×原值；合格候选按Val 10–35m Actor同层1m概率质量选。原Depth始终保留，测试前不替换默认。空间D为条件允许的可选研究，本轮优先检验B/C独立泛化，明确延期，不自动扩大训练。

## 评价定义

像素正/负仅限可靠首可见表面。Unknown不参与负样本，建筑/道路负标签只表示不是目标Actor，不表示可通行。主阈值与低FPR工作点只由Val确定；Test PR曲线仅描述。新增全确认非目标FPR、每百万有效像素FP、FPR≤1%/0.1%/0.01%的Val最大Recall及锁定测试结果、局部Removed FPR、Actor-view漏检和≥3像素连通组件误检。连通组件不是物理Actor级Precision；同Actor跨相机视角相关。Teacher/Student的vehicle/person比较使用完全相同Mask，cone单列。

Depth主MAE使用Softmax概率的期望光轴深度；另报告Mode、Bias、GTBin及±1Bin、同层0.5/1/2m质量、范围外质量、概率期望三维误差。空间质量只在GT落入BEV的有效射线计算，分母单列。Oracle2D投影诊断与正式RGB路径严格区分。

Shadow使用固定P8 UV、K/T、HeightSplat和0.2 event阈值；学习路径不使用GTMask。旧Pixel为M5-G匹配1200步FPN基线；新Pixel由本轮Val选，另单列Distill+原Depth。Oracle2D/OracleDepth/Teacher+OracleDepth只作诊断。输出同层空间Recall、可见表面Precision/FPR、3D格中心匹配、Actor指标和Unknown分布。不能将粗高度层中心误差当真实物体精确三维误差。

ROI研究仅离线：用冻结学生概率/阈值选S4射线，GT只用于后续OracleDepth诊断。比较射线数、CPU投影成本、ROI目标漏检与完整S4/P8 Oracle上限；没有训练S4Depth，也没有改默认Splat。GPU六摄延迟单独停CARLA及其他GPU作业测量，5次warmup+20次，不含I/O、tick或Controller。

## 可复现命令

工作目录为仓库根目录。`.venv` 为CARLA/CPU环境，`.venv-gpu` 为PyTorch2.8.0+CUDA12.8环境，实测RTX4090。完整控制台记录在 `reports/m5h_logs/`。新跑需独立工作树和新输出路径，禁止删除已有失败资产后重用路径。

```bash
.venv/bin/python -m tools.m5h_prepare
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5h_capture_protocol
.venv/bin/python -m tools.m5f_capture --protocol configs/m5h_capture_train.json --output /root/gpufree-data/hs-drive-data/raw/m5h_added
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5h_added --output /root/gpufree-data/hs-drive-data/processed/m5h_added_index --report reports/m5h_added_evidence_quality.json
.venv/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5h_added_index --output /root/gpufree-data/hs-drive-data/processed/m5h_added_pixels --splits train
.venv-gpu/bin/python -m tools.m5h_data
.venv-gpu/bin/python -m tools.m5f_cache --index /root/gpufree-data/hs-drive-data/processed/m5h_added_pixels --output runs/m5h_added_cache
.venv-gpu/bin/python -m tools.m5h_teacher
.venv-gpu/bin/python -m tools.m5h_teacher_audit
.venv-gpu/bin/python -m tools.m5h_teacher_geometry_metadata
.venv-gpu/bin/python -m tools.m5h_replay
.venv-gpu/bin/python -m tools.m5h_depth_training
.venv-gpu/bin/python -m tools.m5h_depth_eval
.venv-gpu/bin/python -m tools.m5h_pixel_training
.venv-gpu/bin/python -m tools.m5h_evaluate --teacher --split val
.venv-gpu/bin/python -m tools.m5h_evaluate --teacher --split train --suffix _locked
.venv-gpu/bin/python -m tools.m5h_evaluate
.venv-gpu/bin/python -m tools.m5h_gpu_contract
.venv-gpu/bin/python -m tools.m5h_selection_coverage
.venv-gpu/bin/python -m tools.m5h_split_audit
.venv-gpu/bin/python -m tools.m5h_decisions
.venv-gpu/bin/python -m pytest -q
.venv-gpu/bin/python -m tools.m5h_freeze
```

必须完成以上冻结才打开新Test：

```bash
.venv/bin/python -m tools.m5f_capture --protocol configs/m5h_capture_test.json --output /root/gpufree-data/hs-drive-data/raw/m5h_sealed
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5h_sealed --output /root/gpufree-data/hs-drive-data/processed/m5h_test_index --report reports/m5h_test_evidence_quality.json
.venv/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5h_test_index --output /root/gpufree-data/hs-drive-data/processed/m5h_test_pixels --splits test
.venv-gpu/bin/python -m tools.m5h_teacher --root /root/gpufree-data/hs-drive-data/processed/m5h_test_pixels --split test
.venv-gpu/bin/python -m tools.m5h_evaluate --root /root/gpufree-data/hs-drive-data/processed/m5h_test_pixels --split test --teacher
.venv-gpu/bin/python -m tools.m5h_evaluate --root /root/gpufree-data/hs-drive-data/processed/m5h_test_pixels --split test
.venv-gpu/bin/python -m tools.m5h_depth_eval --root /root/gpufree-data/hs-drive-data/processed/m5h_test_pixels --split test
.venv-gpu/bin/python -m tools.m5h_dataset_quality
.venv-gpu/bin/python -m tools.m5h_shadow
.venv-gpu/bin/python -m tools.m5h_roi
.venv-gpu/bin/python -m tools.m5h_latency
.venv-gpu/bin/python -m tools.m5h_figures
.venv-gpu/bin/python -m tools.m5h_finalize
.venv-gpu/bin/python -m tools.m5h_interpretation
.venv-gpu/bin/python -m tools.m5h_artifacts
```

## 失败与局限的保留

初版地图审计漏规范化地图名，第二次又混用HS/CARLA世界Y轴；`m5h_split_audit_final.json` 统一当前Pixel数据坐标；随后复核发现原M4索引positions_xy也为HS坐标，`m5h_replay_coordinate_audit.json` 再次独立更正Replay距离。所有最终距离仍满足45m隔离，未改模型或测试位置；旧错误审计保留，不作为隔离证据。Teacher训练域首轮指标自行选了Train阈值，仅作探索诊断；`train_locked` 使用Val锁定阈值，才是与Val/Test一致的报告。首次CARLA合同测试文件名错误保留日志，正确文件重跑。首次Test Teacher作业在像素索引尚未完成时退出，未产生预测；空输出目录改名为`runs/m5h_teacher_test_failed_preindex`保留，索引完成后原命令重跑。

M5-G三个已开测试位置只作历史结果参考，本轮六位置为独立封存评估，但仍共享Actor资产和配置模板。没有新车型泛化、真实世界或独立Mesh真值。每组12变体不能作为12独立场景。Runtime未改，继承Partial；Oracle安全未重验，纯视觉安全未验证。不得自动进入大型Occupancy/Temporal/SparseQuery训练。
