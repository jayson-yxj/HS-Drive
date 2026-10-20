# M4 GPU与小样本Depth/BEV基线

继承HS-Drive V1.3、M3数据合同与固定划分。本轮实现单帧学习链路并实际训练；完整M4（含可靠深度/BEV质量、Drivable和更多场景）保持开放。

## 环境与复现

`.venv-gpu`为独立GPU训练环境；`.venv`仿真与`.venv-train` CPU数据环境保留。PyTorch 2.8.0+cu128、torchvision0.23.0+cu128、CUDA runtime12.8；驱动支持上限与runtime不同。精确包锁在`requirements-train.lock.txt`。CARLA停止后训练。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
bash scripts/bootstrap_train.sh
mkdir -p /root/gpufree-data/hs-drive-cache/weights
curl -fL --retry 2 https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o /root/gpufree-data/hs-drive-cache/weights/resnet18-f37072fd.pth
.venv-gpu/bin/python -m pytest -q
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_baseline.yaml \
  --overfit --steps 80 --output runs/m4_overfit_repeat
.venv-gpu/bin/python -m tools.train_bev --config configs/m4_baseline.yaml \
  --output runs/m4_baseline_repeat
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_baseline_repeat \
  --split val --output reports/m4_val_repeat
.venv-gpu/bin/python -m tools.eval_bev --run runs/m4_baseline_repeat \
  --split test --output reports/m4_test_repeat
```

输出目录不得已存在；原始M3索引路径由配置指定。下载权重在加载前核验官方文件名对应的SHA256前缀，运行manifest记录完整哈希。相机/BEV配置、数据index哈希、源码哈希、训练样本ID、依赖与种子均记录。

本容器首轮CUDA卷积通过，但`torch.linalg.inv`动态加载`libtorch_cuda_linalg.so`失败。实际库与依赖齐全；`training.runtime.prepare_cuda`从本环境torch安装目录绝对路径加载其库后成功。GPU检查包含矩阵逆、BF16卷积前后向、scatter梯度；没有更改系统驱动、绕过设备权限或重编CUDA。失败记录在`runs/m4_overfit_v1`及`logs/m4_overfit_v1.log`。

依赖参考：[PyTorch固定版本安装](https://pytorch.org/get-started/previous-versions/#v280)、[Torchvision ResNet18](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet18.html)、[PyTorch AMP](https://docs.pytorch.org/docs/2.8/amp.html)。骨干直接复用torchvision及其ImageNet权重，不复制第三方仓库。

## 模型与监督边界

- `models/image_encoder.py`：共享ResNet-18，取stride8/16/32特征，以轻量横向投影和上/下采样形成stride16、64ch特征。输入沿M3 RGB /255，在模型中进行ImageNet mean/std归一化。BatchNorm统计冻结，仿射参数与骨干权重可训练，适配单场景小批次。
- `models/depth_bev.py`：归一化K和安装外参组成16维camera encoding，MLP调制Depth Head；48个LID深度Bin和64ch Context Head。不是BEVDepth官方权重复现。
- `models/view_transform.py`：按M3特征单元中心约定构造光线，FP32 K求逆/反投影，按深度Bin中心落入四层高度BEV。`index_add_`分块累加，禁止物化完整[B,N,D,C,H,W]特征体。保留加权平均context与独立概率质量mass，压缩成[B,128,160,120]。
- 模型forward严格只接收既有八个合规输入字段；单帧阶段不消费ego/route信息，缺失摄像头直接拒绝。网络没有Depth、Actor或GT Map参数。
- 输出depth_logits/probs=[B,6,48,24,40]、height_bev=[B,64,4,160,120]、bev_current=[B,128,160,120]，另有surface_log_counts和splat_mass诊断。

FP32 geometry在autocast外执行；卷积用BF16。scatter累加用FP32。未启用torch.compile、BEVPoolv2、Temporal、Sparse、Occupancy、Planner。

Depth loss在M3返回的准确depth_uv处对logits做双线性采样，与原生最近像素的监督射线匹配；valid mask和ignore_index=-100处理无效深度。48个LID区间中心用于Lift-Splat；连续深度评估用概率期望，不把argmax或GT深度喂入网络。

BEV监督由同一组有效Depth样本在真实距离处投影形成四层可见表面点数。`surface_head`预测log(1+count)，在有点/无点格分别取MSE再各占一半；总损失=Depth CE+0.2×surface MSE。**无点格只代表本次采样计数为0，绝不标为Free。** 此辅助任务检验Context/BEV梯度链路，不是Drivable或Occupancy，不提供交通语义。

`depth_uv`特征单元中心是显式的工程采样约定，不宣称ResNet感受野严格中心；后续视角变换/resize必须使用同一规则。高度边界和XY右开区间与Oracle一致，有NumPy参考、CPU/CUDA和梯度测试。

## 实验协议

配置`configs/m4_baseline.yaml`固定seed7、batch1、BF16、AdamW lr2e-4/wd1e-4、裁剪norm5、240步。三个train录制各均匀取8帧，共24帧，静止片段占比较高；独立val录制均匀取8帧，每40步评估，按val总损失选best。测试地图不参与选模或调参。初始网络使用ImageNet骨干，其余head随机初始化。

单样本80步试验用于排查学习链路错误。阈值在训练前写入`configs/m4_acceptance.yaml`：depth CE至少下降20%，surface MSE至少下降10%，显存allocated<20GiB、梯度有限、检查点及恢复可用。该门禁是工程基线，完整M4质量门禁不能据此关闭。

最终选定权重在完整61帧val和61帧Town10 test上评估：深度MAE/AbsRel/RMSE/delta1按1.5–10/10–30/30–60/60–80m统计。仅统计M3有效平滑表面，不能外推全图/遮挡边缘。报告另含GT深度分箱后的量化误差下限诊断。

对照均仅拟合实际24帧train子集：①全局Bin中心均值的固定深度；②每摄每特征像素平均深度，未观测位置退回全局均值。后者识别固定相机视角带来的位置先验，不是学习型视觉模型。任何test观察仅报告，不据此延长本轮训练。

## 检查点、恢复与资源

每次评估原子保存last.pt，改善时保存best.pt；含模型、AdamW状态、全局step、配置、index哈希、torch/CUDA RNG。预先生成确定样本顺序，恢复跳过已完成step。resume必须新目录、配置及index一致，沿用原best.pt。

```bash
.venv-gpu/bin/python -m tools.train_bev --steps 8 \
  --stop-after 4 --output runs/resume_demo_part
.venv-gpu/bin/python -m tools.train_bev --steps 8 \
  --resume runs/resume_demo_part/last.pt --output runs/resume_demo_finish
```

可用`.venv-gpu/bin/python -m tools.check_training_resume --output runs/resume_check_repeat`完整重做4+4与连续8步对照。

`--stop-after`保持完整8步计划，只在第4步存盘退出。`checkpointed`表示尚未完成计划。异常写failed manifest，历史产物不覆盖。断点验证包括不同样本的步序和优化器步数，报告恢复与连续运行参数差；CUDA index_add/grid_sample backward不保证位级确定，不宣称可逐bit重放。

step_compute包含数据上GPU、forward/backward、optimizer与同步，排除DataLoader等待；不是端到端推理FPS。peak_allocated与peak_reserved是PyTorch allocator统计，不含驱动/整卡开销，也不代表未来Temporal联合训练显存。没有据此提高默认batch。

## 尚待完成

更多独立路线、车辆与天气；Drivable/Occupancy对应标签；分距离可靠质量；深度离散化/边缘采样及BEV pooling消融；推理延迟专项。当前辅助BEV输出不能用于驾驶控制或完整可行驶空间判断。
