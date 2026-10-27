# M8-J6 实际数据续跑：八帧基线通过，三步更新首步后停止

日期：2026-10-27；基于 `b88d6a6` 继续原独立分支。此前数据缺失报告保留不改；本文件及 `m8j6_actual_acceptance.json` 是数据到位后的最新状态。

**总体 Partial。Mini 已安全接入，八帧 FP32 前向/Loss/反向通过；实际仅完成 1 次 optimizer step，三步 Smoke 未通过，未进入过拟合。**

## 数据和监督

用户在官方页面下载并提供 `/root/Downloads/v1.0-mini (1).tgz`。文件 4,168,148,189 bytes，本地 SHA256：

```text
e5d9c2b5ced29f9e3d39651e2093d27e892d0571f10962409974b21033a254ee
```

没有取得官方预期校验值或浏览器原始下载 URL；这个 SHA 只用于复现，不称官方真实性认证。gzip CRC/trailer 和 tar 检查通过，31,253 条目，无链接、特殊设备或越界路径；解压 5,374,922,749 bytes 到私有目录。没有更改原压缩包。

按原协议选取 `scene-0061`、`scene-0655` 各最早四个 sample，不看预测挑帧。48 张图像、真实 sample token、相机文件 SHA、sample_data、各相机时间戳及 pose/calibration 关联已固定。没有使用 mini_val 或 full val 更新。

八帧包含 **484 条原始 annotation 记录，335 条官方任务训练 GT，142 个原始 instance**。335 不是独立目标数量，连续帧和相机视野存在相关性。149 条记录被官方过滤；原因可重叠：XY范围119、LiDAR/Radar点数均零58、非十类4。点数只用于监督筛选，不读取点云感知输入。没有移植 Bench2Drive 的 N=0 策略，也不把 detection Negative 解释为空空间。

保留类别：car112、pedestrian87、barrier97、truck28、traffic_cone9、motorcycle1、bicycle1（均为跨帧记录数）。bus、trailer、construction_vehicle 本次没有正样本；保持十类输出，不将未覆盖类别标为质量通过。

官方 yaw-only Box 与转换后的八角点误差最大 `1.073e-5 m`；reference LiDAR→各 camera 与 devkit 独立坐标链误差最大 `3.686e-13 m`。前者验证官方 yaw 约定，不意味着保留了 roll/pitch 完整姿态。相机时间相对 sample 为 −45,513 至 +70 μs；保留异步相机 pose，没有假造同步帧。

逐 annotation 原始字段、类别、尺寸、Yaw、过滤原因均在私有 sidecar。公开报告仅含按帧汇总与文件哈希。私有查看页面：

```text
/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared/review/index.html
```

实际查看者为 **Codex AI assistant**：查看八帧六摄全原始框拼图、八帧BEV，以及近距离行人/车辆等原分辨率局部；没有发现需隔离的具体系统性错位或遗漏案例。该记录不是人工 Golden GT 签核，也不证明全数据集标注完备。另生成从实际训练 NPZ 和实际模型裁剪输入绘制的青色框对照，避免只看原始框而未看真实训练框；其中0/3/4/7四帧拼图已实际查看。

## RTX4090 FP32 冻结基线

原PETR十类、900Query、六层、batch1、六摄512×1408。AMP与TF32关闭，FP32参数/输入/特征/输出逐项断言。图像 Backbone 在 eval 和 no_grad 下冻结；仅检测头建立反向图，无 optimizer。输入仅 RGB 和标定、真实相机时间元数据；GT单独进入官方 Loss。Python文件访问 canary 拒绝原始传感器数据和感知调用中的监督访问；这不是对恶意 native 扩展的沙箱认证。

八帧全部完成前向、官方 Loss、六层直接梯度检查；原模型全部参数和buffer精确不变。正分类、负分类、正回归梯度均非零且有限，Negative回归梯度为零。官方 `nan_to_num` 前增加有限性断言，没有靠清理 NaN 通过。

| 帧索引 | 场景 | P | I | N | 总 Loss（六层之和） |
|---|---|---:|---:|---:|---:|
|0|0061|50|0|850|5.058887|
|1|0061|51|0|849|5.324497|
|2|0061|56|0|844|5.442827|
|3|0061|61|0|839|5.353627|
|4|0655|28|0|872|4.153164|
|5|0655|28|0|872|3.411180|
|6|0655|29|0|871|4.013455|
|7|0655|32|0|868|3.683620|

每帧六层计数相同，上表不是八个 Decoder 层。48条 frame×layer 完整分类/回归Loss、正负贡献、有效权重900、实际归一化因子及梯度见 `m8j6_actual_decoder_statistics.json` 和 `m8j6_sources/actual_baseline/`。

可训练检测头11,550,624参数，冻结24,254,781参数。baseline峰值allocated **1,805,817,344 bytes（1.682GiB）**，仅代表本次冻结视觉骨干、训练检测头配置。每帧读取/前向/Loss/反向合计约0.944–2.489s，首帧含冷启动；不能称纯推理延迟或实时20Hz。具体耗时与reserved显存均存原始报告。

## 三步更新实际失败及归因

保持预声明 seed875606、batch1、AdamW lr1e-5、weight_decay0、clip35、仅Head训练、原始图像大小和类别。首帧index0反向有限性和P/N检查通过，执行一次 `optimizer.step()` 后冻结检查抛错：

```text
FrozenChanged:pts_bbox_head.cls_branches.1.0.weight
```

立即停止，未执行第2、3步。没有OOM、没有AMP回退、没有更换样本。

根因是**本轮新检查器实现错误**：官方PETR分类/回归分支在六层间共享对象；`named_parameters()`去重，而`state_dict()`保留别名。检查器误将80个可训练别名归为冻结参数，首个报错就是 `.cls_branches.1.0.weight`。官方结构的CPU复现证实该权重与第0层是同一 Parameter。

修复后按Parameter identity取得全部221个可训练state名称，对剩余337项参数/buffer保持精确检查；合成修改共享Head不会误报，修改BN buffer仍会拒绝。**这是修复验证，不是已失败真实运行重新验收。**

还发现运行器证据保存缺陷：首步详细梯度/变化指标在末尾才落盘，异常发生时尚未写出；失败时也未保存实验checkpoint，内存状态已经释放。因此不能补报首步精确梯度值、参数变化量、完整冻结对照或恢复一致性。代码已补上 step 前落盘和失败checkpoint保存；本次缺失证据如实保留，不重建假记录。

原始预训练checkpoint和safe tensor哈希未变。后续未再执行真实optimizer step；总计仍为1，不能从丢失的状态续做两步，也没有从原checkpoint偷偷重跑三步。

## 修复与回归

- 20项主工程选帧/安全路径测试通过。
- 16项旧PETR环境官方合同/别名/隔离CPU测试通过；包括实际模型结构构造，无预训练权重、无真实数据、无GPU训练。
- 8帧真实GPU基线，48条Decoder记录通过。
- 1次真实Smoke尝试，1次step后失败，完整失败报告未覆盖。
- 两个前处理/展示脚本问题也保留失败日志：浅拷贝NuScenes破坏sample索引，改为代理对象将get绑定原表；OpenCV绘图数组非连续，改为连续副本。均未更改官方数据或模型。

## 决策

- 数据适配：**Go，仅这八帧的官方任务工程范围**。
- FP32真实前向/Loss/反向：**Passed，仅冻结Backbone/Neck+Head反向范围**。
- 三步更新/保存恢复：**Failed / NotRun，整体Partial**。
- 小集合过拟合：**暂不进入**，先完成修复后独立三步及保存恢复验证。

最小后续动作是确认一次新的、固定三步重试；使用同八帧、同权重、同超参数，保留本次失败为独立attempt。因为本轮要求遇到异常立即停止，且已消耗1次step，未自动追加更新。下一次三步会使历史总step为4，报告必须分别记录，不能声称历史只更新过三次。

两个场景均属于官方full train，与预训练分布存在重叠；不宣称泛化。没有启动CARLA、大规模训练或视觉控制；私有图像、逐对象标签和模型均未提交。
