# M8-J8 有限残差与独立场景评估

基于f50aead，分支feat/m8j8-residual-independent-eval。无optimizer实例、无backward、无更新；不改旧文件、不联网下载或启动CARLA。

## 冻结范围

A为M8-J7相同原始安全tensor归档，B为最终400步checkpoint，路径/SHA已冻结于configs/m8j8_protocol.json。只加载本地已验证来源或本项目自生成且SHA匹配的checkpoint，隔离环境禁网。仍用原PyTorch/MMCV/MMDet3D、FP32/eval/bs1、十类512×1408、正常NMSFree top300、score≥0.2、strict XY范围及同类别score顺序2m/1m/0.5m匹配；不新增NMS或改变PR阈值。

reports/m8j8_evaluation_manifest.json在任何预测前选择官方mini_val全部81关键帧：scene-0103 40帧、scene-0916 41帧；场景/sample与M8-J7无交集。固定顺序scene名、timestamp/token。只使用现有官方Mini，本轮不增加数据源。两个场景作为开发评价，用后不得再声称封存最终集。

原始PETR官方R50-C5配置的ann_file为nuscenes_infos_train.pkl、24epoch；README将其列在nuScenes val成绩表。mini_val两个场景在冻结devkit full val，均不在full train。这支持“按官方配置未用于检测训练”，不独立证明checkpoint完整实际数据历史/模型选择无泄漏。M8-J7更新未见则由本地固定清单直接验证。

## 数据和指标

复用M8-J6官方converter、CustomNuScenesDataset和固定resize/crop/normalize/pad。GT使用原点数valid_flag、XY范围和类别规则；每条保留/排除理由私有保存。逐相机pose/timestamp照实保留；点数只用于标签过滤，无点云/Radar推理输入。GT与RGB/calibration分离为NPZ；输入白名单及监督访问canary复用旧实现。数据转换核对底面/中心和完整尺寸、yaw角点及camera坐标链，不新增监督资格门禁。

项目指标复用training/m8j7_metrics.py不改；新增Δz/Δh及尺寸分项和共同匹配配对误差是诊断，不改变选择或匹配。全部GT数量按场景/类别分开，重复帧记录与独立instance数量分开。无GT类Recall/AP为空，仍报告FP；少量支持如实标明。此结果不是官方mAP/NDS/mATE或完整nuScenes val。

已有官方eval入口可用，但需要额外global格式/attribute及官方类别范围契约适配，本轮不新增该链路；固定项目诊断为主。其未执行单独标记。

## 残差与只读前向

J7缓存缺少query provenance和decoder前输出，因此对旧八帧补A/B固定只读前向；逐框与原J7缓存数值核对atol1e-6/rtol1e-5。保存final decoder logits/box编码和top300映射，验证官方decode输出一致；query仅当前前向内部索引，不当物体身份。

重点3例大Z误差、4例FN、56个duplicate，按annotation_token及instance_token去重。底面z → gravity_center → normalized target的cz/log(h) → 预测cz/log(h) → exp(h) → get_bboxes减h/2 → box.corners → 显示逐项追溯。若数值链一致但预测异常，不裁剪/改GT；若发现合同错误，A/B统一修复重算，旧结果保留。

每帧及最终比较全模型state独立CPU快照，确认参数/buffer精确无变化、模块全eval、梯度None及共享别名关系。新评估仅运行A/B两份固定权重，不搜索模型。私有可视化：旧八帧全部、新每场景首/中/末帧，不按最好表现选帧。

## 复现

冻结代码、配置、manifest及测试提交后，干净工作区一次执行：

```bash
.venv/bin/python -m tools.m8j8_run
```

顺序为prepare（CPU）、A、B。各进程禁网，1200秒上限；目录存在拒绝覆盖，异常立即停止保留记录，不自动重试。训练与更新次数固定0。私有输出在/root/gpufree-data/hs-drive-private/m8j8_20261027_ab；公开仅聚合指标、manifest标识、源码和SHA，不含逐对象GT/预测、原图或权重。
