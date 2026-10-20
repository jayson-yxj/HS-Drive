# M4 GPU与小样本基线实测

2026-10-19 16:28 UTC。**已完成本轮请求的GPU环境验证与小样本Depth/BEV基线，工程门禁14项通过。完整M4质量门禁保持开放。** 网络架构沿用V1.3；没有开始Temporal/Sparse/Planner或闭环。

## 最重要的结果

网络可从RGB训练并输出Depth、四层height_bev与128ch bev_current，损失降低、梯度有效、检查点和续训可用。但当前网络的独立场景深度误差**尚不如逐摄像素训练均值先验**，不能宣称已取得视觉学习收益或完成可靠BEV感知。

| 深度MAE（m，越低越好） | Town03完整val 61帧 | Town10完整test 61帧 |
|---|---:|---:|
| 全局固定深度，仅统计24帧train | 10.370 | 11.053 |
| 每摄每特征像素均值，仅统计24帧train | 2.735 | 5.200 |
| 当前RGB Depth/BEV网络 | 5.119 | 6.756 |

当前test AbsRel=0.897，delta1=0.205；60–80m MAE=41.792m，远距离明显不足。指标只覆盖M3有效平滑表面像素，不覆盖天空、无效/深度边缘，也不是全图或行驶安全指标。完整分段数据见[val结果](m4_val_final/metrics.json)、[test结果](m4_test_final/metrics.json)。test未用于选模或后续调参，本轮权重保持固定。

## 已完成的实施与验证

- 独立`.venv-gpu`：PyTorch2.8.0+cu128 / torchvision0.23.0+cu128 / runtime12.8 / RTX4090 24GB / driver580.126.09。BF16卷积前后向、矩阵求逆、scatter前后向通过；依赖锁`requirements-train.lock.txt`。CPU数据与CARLA环境保留。
- Torchvision共享ImageNet ResNet-18 + 轻量stride16/64ch FPN + 标定调制Depth Head + 48 LID bins + 64ch Context + 四层FP32 chunked scatter Lift-Splat + 128ch BEV。没有构造完整6×48×64×24×40长期特征体。
- 输出[B,6,48,24,40] Depth logits、[B,64,4,160,120] height_bev、[B,128,160,120] bev_current。仅八个M3白名单张量可进入forward，无Depth/Actor/GT Map输入参数。
- Depth CE使用原生采样准确UV和有效mask；BEV辅助监督是采样可见表面的log点数，正/零计数格平衡MSE，权重0.2。零计数不等于Free，不宣称完成Drivable或Occupancy。
- 单样本80步：Depth CE 3.949→1.363（下降65.5%），surface MSE 0.826→0.048（下降94.2%）；通过训练前固定的20%/10%下降门槛。
- 小样本训练24帧、240步、batch1/BF16、AdamW lr2e-4；val独立8帧每40步选模，选中step 240。train CE 3.940→0.577，val子集CE 3.911→1.554。没有把相邻帧随机拆到train/val。
- 实测PyTorch peak allocated 0.732GiB、reserved 0.822GiB。240步运行墙钟72.11s；step_compute P50=0.0250s/P95=0.0271s，排除DataLoader等待，不是完整系统或闭环FPS。后续联合训练显存不能从此直接外推。
- 检查点state_dict精确重载。8步不同样本断点试验4+4：样本顺序和AdamW步数恢复，最大参数绝对差0.001892（阈值0.005）；CUDA scatter/grid_sample不保证位级确定，未声称逐bit一致。
- GPU环境pytest 78项通过、5项CARLA测试跳过；仿真环境68项通过、2个训练测试模块跳过。两环境覆盖83个用例，无失败；另新增门禁拒绝失败续训/篡改checkpoint的负测通过。pip check、compileall通过。

## 产物

- [工程门禁](m4_baseline_gate.json)、[GPU检查](m4_gpu.json)、[断点试验](m4_resume_final.json)。
- 最佳权重：`runs/m4_baseline_v1/best.pt`，SHA256 `8faf346a5de79f22893e5ea5324a6dacaec0a8f2113731085f9ba442d94f855f`；last.pt同时保存优化器/RNG/step；manifest、config、env.lock、metrics.jsonl完整保留。
- [test六摄深度对照](m4_test_final/test_depth.png)、[test四层BEV对照](m4_test_final/test_bev.png)。GT和预测仅在可视化/评测阶段并列，黑色是GT评估掩码，不是模型无效预测。
- 新增`models/`、`training/`、`tools/train_bev.py`、`eval_bev.py`、`check_training_gpu.py`、`check_training_resume.py`、`validate_m4_baseline.py`。模型与训练实现可运行，没有模型占位伪代码。

## 失败与边界

第一次完整模型在CUDA矩阵逆处找不到libtorch_cuda_linalg.so，保留`runs/m4_overfit_v1` failed manifest及日志。库和依赖齐全，改为从本torch安装目录绝对路径加载后通过；这是容器动态库搜索问题的实测规避，不是修改驱动或宣称修复PyTorch内部问题。

单样本拟合仅确认代码可学习；24帧训练来自一个路线组、静态片段比例高。BEV辅助任务可过拟合相机/道路位置先验，现有误差与视觉对照不足以证明通用场景理解。今后先扩大独立驾驶场景并完善Depth实验，再补Drivable标签与对应头/损失；Occupancy/Free/Unknown仍保持独立语义，不能用本轮点数冒充。

## 服务器复现

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_baseline.yaml \
  --output runs/m4_baseline_repeat
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_baseline_repeat \
  --split val --output reports/m4_val_repeat
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_baseline_repeat \
  --split test --output reports/m4_test_repeat
```

部署、接口、损失定义与断点命令见[M4执行说明](../docs/M4_BASELINE.md)。输出不可覆盖历史目录。训练与评估进程均已退出，CARLA未启动，GPU释放。
