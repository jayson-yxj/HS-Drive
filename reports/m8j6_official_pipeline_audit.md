# M8-J6 官方管线核对

日期：2026-10-27。以下是冻结源码阅读和 CPU 合成测试证据，**不是八帧实测适配结果**。
固定 PETR `f7525f93467a33707ef401c587a52d5e7b34de74`，配置 `petr_r50dcn_gridmask_c5.py`。
各安装源码及 checkpoint SHA 见 `m8j6_source_lock.json`。mmdetection3d 安装源码没有独立 `.git`，以实际文件 SHA 和已安装 0.17.1 版本记录，不编造源码 commit。

| 项目 | 冻结实现及判定 | 已有证据 / 实测缺口 |
|---|---|---|
| 类别 | car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone | 官方配置断言通过；无类别缩减 |
| 相机 | FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT，均为 CAM_ 前缀 | 官方 converter 的插入顺序；不能沿用 CARLA 顺序 |
| 图像 | BGR，均值 103.530/116.280/123.675，std=1，to_rgb=False | 合成 BGR 像素经官方 PIL 裁剪保持通道顺序 |
| 尺寸 | 原图预期 900×1600；固定 resize=.88 到 792×1408，crop=(0,280,1408,792)，最终 512×1408 | 官方固定裁剪合成检查通过；真实 48 张尺寸未读 |
| 增强 | 官方 ResizeCropFlipImage(training=False)；明确不运行随机 GlobalRotScaleTransImage；本次 GridMask 关闭 | 仅拟用于独立 Smoke；原默认配置保留 |
| 标定 | 裁剪矩阵左乘 intrinsic；lidar2img=K×extrinsics.T | 官方实现合成检查通过 |
| 时序 | 相机 sample_data 时间戳、各自 ego_pose 与 reference LIDAR_TOP ego_pose 分别关联 | 旋转和平移合成实例通过；不伪造同帧同步 |
| GT | nuScenes global center / full w,l,h / quaternion → reference LiDAR；旧 SECOND yaw=-yaw−π/2 | 四种旋转八角点集合误差小于 2e-6m（合成） |
| Box 原点 | dataset 接受中心原点 (.5,.5,.5)，内部保存 bottom center；Head.loss 取 gravity_center | 官方类往返测试通过，不重复减半高 |
| 速度 | adjacent annotation 全局有限差分，转 reference LiDAR；官方 vx NaN 时将两个分量置零 | 合成官方 fallback 通过；真实速度未验证。额外非有限值需隔离，不能靠 Loss.nan_to_num 隐藏 |
| 点数 | use_valid_flag=True，num_lidar_pts+num_radar_pts>0 | 属于监督支持过滤，不进入视觉特征；不读点云内容 |
| 范围 | ObjectRangeFilter 对 LiDAR Box 是严格 XY center 边界，不过滤 Z | 测试保留 z=100 的合成 Box 以明确实际代码行为；绝非真实异常 Box 自动合格 |
| 类别过滤 | NameMapping→十类，未知类 label=-1 经 ObjectNameFilter 排除 | 与 point-valid 和 XY 范围分别追踪；没有自加可见性门禁 |
| 输出 | 6层×1batch×900query×10类；回归编码10维，含速度 | 源码/配置事实，真实前向 NotRun |

## 官方 converter 的有限使用方式

复用 `tools/data_converter/nuscenes_converter.py::_fill_trainval_infos`，对 NuScenes 对象的浅副本只设置已冻结八条 `.sample`，`max_sweeps=0`，不对整个 Mini 生成训练 infos。
`.get()` 仍能访问原始 metadata，以保留相机 pose、calibration 与官方 box_velocity 所需相邻 annotation。
相邻 metadata 只用于速度标签，不把额外图像加入训练。
converter 检查 reference LiDAR 文件存在，但不读其点云；监督的点数来自 annotation 表。
该八帧实际转换尚未运行。

## 模型输入和监督分离

实际模型调用应仅为 `extract_feat(img, img_metas)`、`pts_bbox_head(features, img_metas)`，元数据白名单只包含投影、图像形状和必要格式信息。`ann_info`、点数、可见性、GT、点云/Radar 路径均不传入模型；GT 在 `head.loss` 路径单独输入。`pts_filename` 出现在官方 dataset 中不代表模型用了点云，必须检查实际读取及调用边界。

当前未创建真实数据 bundle，尚无本轮真实 I/O 隔离证据，不能声称已经完成泄漏验收。

## 官方监督和统计口径

`PETRHead._get_target_single` 每层做 Hungarian assignment，PseudoSampler 取得正负 Query。Positive 使用真实类别和回归目标；未匹配 Query 使用背景标签 `num_classes=10`。FocalLoss 是 sigmoid 十类，背景标签对应所有类的负目标。

无具体缺标异常、任务过滤核对通过的样本不启用 Bench2Drive 的 Ignore。此时 I=0，P+N=900；实际 P/N 仍必须逐帧逐层测量，不能提前代填。

当前 sigmoid Focal 配置 `bg_cls_weight=0` 作用于**归一化分母**，不是删除负样本梯度：classification 分母 `max(P+N×0,1)`，每个 Query 的 label weight 为1；regression 分母 `max(P,1)`，只对 positive 给非零 bbox weight，速度两个分量 code weight=.2，其余1。每层 loss 权重 cls=2、bbox=.25。

六层辅助 Loss 全部走同一 `loss_single`。真实记录须保存 `(sample_token, decoder_layer)` 的 P/I/N、分类有效权重和分母、正负分类贡献、回归 Loss。J5B 的四组数字是四帧，不能挪作本轮四层结果。

官方 forward 和 loss 含 `nan_to_num`。后续 Smoke 必须在清理前的 decoder 输出、预测、未规约损失及梯度检查非有限值，不能只看最终 loss 有限便通过。尚无本轮真实运行，保留此检查为必要前置。

## 测试结果与边界

- 主工程 metadata selector：13 passed。
- 旧 PETR 环境官方合同：12 passed；CPU、无模型初始化、禁网络和 GPU。
- 第一次直接 pytest 调用了主项目 Python3.10 conftest，在 Python3.8 下失败；保留日志。显式 `--noconftest` 仅隔离不相关主项目 fixture，未修改环境或弱化这些独立合同断言。
- 未执行真实图像查看、GT 投影、baseline、loss、backward、optimizer step、checkpoint 恢复。不新增人工签核要求，也不虚构已查看记录。
