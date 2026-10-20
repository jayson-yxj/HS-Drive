# 训练前成像验证与复现

## 范围

本轮是M3数据采集前置成像验证，不是模型训练、完整M3数据集或真实摄像头标定。沿用V1.3网络输入640×384与六摄安装；新增独立profile可选择更高采集分辨率，RGB与Depth始终同尺寸、同FOV、同位姿。旧配置及M2数据不回写。

CARLA官方参数依据：
- https://carla.readthedocs.io/en/0.9.15/ref_sensors/#rgb-camera
- https://carla.readthedocs.io/en/0.9.15/adv_rendering_options/

Epic是官方最高画质预设，但RGB后处理仍可单独关闭。当前几何配置原先关闭了后处理，导致与训练候选显著不同的颜色/亮度分布。新配置的后处理只影响RGB，Depth仍为原始数值解码。

## 配置

`configs/cameras.yaml`：原几何基准，未指定profile时保持后处理关闭。

`configs/imaging/*.yaml`：成像实验配置，只允许尺寸和RGB成像参数，不允许修改FOV、安装、sensor_tick或镜头几何。`training_960_v4.yaml`为−0.5EV候选；真实默认选择以 `configs/data_capture.yaml` 与 `reports/imaging_gate.json` 为准。

- v1：后处理开启，曝光补偿0EV，histogram范围[10,12]。
- v2：范围改为[1,16]且−1EV；晴天严重泛白，**拒绝**。
- v3：恢复[10,12]、−1EV；日落后右摄黑色比例略超筛选线，**不选作白天/日落统一默认**。
- v4：范围[10,12]、−0.5EV，所有六摄重新测量。
- 候选均关闭motion_blur、bloom、lens_flare，fstop=16减弱景深；gamma=2.2，零畸变。

参数强度是实验选择，不代表现实摄像头标定结果。CARLA传感器实际attributes保存在每场景calibration.json中。相机间独立自动曝光可能造成跨摄亮度差；未来进行光度损失或视觉匹配时需要明确处理。

## 复现命令

```bash
source .venv/bin/activate
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2000 Epic
```

另一个终端执行（必须唯一tick客户端；输出目录不可已存在）：

```bash
python -m tools.compare_imaging \
  --profiles configs/imaging/training_960_v4.yaml \
  --render-every-tick --tick-delay 0.5 --weather noon --frames 11 \
  --output /root/gpufree-data/hs-drive-data/imaging/noon_repeat
python -m tools.analyze_imaging \
  --experiment /root/gpufree-data/hs-drive-data/imaging/noon_repeat \
  --output reports/noon_repeat
```

`--weather sunset`和`--weather night`分别使用8°和−15°太阳高度；noon为50°，太阳方位30°。固定同一个自车、25m目标车及位姿，所有配置逐一重新预热40物理ticks。世界原天气与同步设置在退出时恢复。所有旧版候选保持为可复现实验，不能混入最终版本数据。

长期录制：

```bash
python -m tools.compare_imaging --profiles configs/imaging/training_960_v4.yaml \
  --render-every-tick --tick-delay 0.5 --frames 301 --output /root/gpufree-data/hs-drive-data/imaging/stability_repeat
```

一般采集器支持默认清单（显式CLI参数优先，文件内路径相对于清单目录解析）：

```bash
python -m tools.record_dataset --capture-config configs/data_capture.yaml \
  --frames 61 --motion straight --output /root/gpufree-data/hs-drive-data/raw \
  --run-id imaging_motion_repeat
```

默认清单不更改天气或服务端画质，只选择相机与采集参数；应确认服务端以Epic启动，并检查录制manifest中的实际天气。清单中的network_input是后续loader约定，不会在录制时缩小原始RGB/Depth。旧命令不带`--capture-config`或`--camera-profile`时仍保留640几何基准。

最终门禁复核（需替换场景与报告路径为自己的重测产物）：

```bash
python -m tools.validate_imaging_gate --profile-id training-960-v4 \
  --reports reports/imaging_noon_v4_everytick/report.json reports/imaging_sunset_v4_everytick/report.json \
  --stability-scene /root/gpufree-data/hs-drive-data/imaging/stability_960_v4_everytick/training-960-v4 \
  --geometry-report reports/imaging_stability_geometry_v4_everytick.json \
  --motion-report reports/imaging_motion_geometry_v4_everytick.json \
  --depth-report reports/imaging_depth_photometry_v4_everytick.json \
  --output reports/imaging_gate_recheck.json
python -m pytest -q
```

## 指标与限制

全部图像先缩放到共同640×384进行亮度诊断，另保存原生分辨率目标裁剪。近黑定义8位灰度≤5，近白≥250；这是编码值诊断，不是物理传感器过曝测量。Depth的[1.5,80)m掩码排除天空和远处背景。所有六摄分别计算，不能只用前摄决定通过。报告中每摄亮度比例是该片段各帧指标的均值，不能理解为每帧最坏值。

静态闪烁指标是固定相机/目标场景相邻RGB的8位MAE，也会受树叶等场景变化影响；它不是纯曝光稳定性测量。夜间测试没有专门启动车头灯，只是诊断保持白天参数的退化，不代表夜间训练配置。

门槛见 `configs/imaging/acceptance.yaml`：白天各摄近黑/近白均≤5%，静态相邻帧MAE≤2，完整录制301帧、重叠几何通过。它们为工程筛选线，不是模型精度保证。宽曝光v2和日落失败v3数据保留。

GPU通过nvidia-smi约0.5秒轮询，是整卡采样最大值（包含服务端及缓存），不是CUDA allocator精确峰值。不同顺序/服务端热身导致缓存差异，不能将显存差值直接归因于profile。短测吞吐包括0.3秒/tick等待，最终稳定性重测包括0.5秒/tick等待、PNG/NPZ压缩和磁盘写入，不能当作纯渲染FPS。

960的v4首轮长测也发生过中途部分传感器落在相邻帧的问题；虽然写满301帧，完整性门禁拒绝。失败场景`stability_960_v4`、诊断及`reports/imaging_gate_v4_unpaced_failed.json`保留；0.5秒等待的重测`stability_960_v4_paced`仍失败，说明仅加大墙钟等待不足以解决问题。最终方案使用逐物理tick渲染加显式统一选帧，绝不修改帧号或放宽时间戳容差。后续每次采集都必须独立做完整性检查。

960原生数据每帧约9MB（六摄RGB+Depth），存储约为640的2.2倍。1280候选短测曾有同步丢帧，失败保留；未来优化后可复测，不能以不完整片段宣布可靠吞吐。

## 模型接口兼容

训练RGB可通过 `geometry.cameras.resize_rgb_and_intrinsics` 缩为640×384。OpenCV像素中心映射为 `u_new=(u_old+0.5)*sx-0.5`，K同步变换。此辅助函数仅处理RGB；**禁止直接用面积插值平均Depth**，否则跨物体边缘产生虚假距离。M3需设计原分辨率深度监督采样/有效掩码，不能把这个函数当作完整训练loader。

输入分辨率和网络结构没有变化，也没有训练结果证明960采集一定优于640。当前依据是画面、几何、稳定性和资源开销，后续应通过相同数据量/划分的训练消融确认收益。

## 传感器调度与输出频率

`--render-every-tick` 将12路CARLA传感器的实际 `sensor_tick` 设为0（每物理步输出），相机配置的 `sensor_tick_s=0.1` 仍表示数据集输出间隔，另记录 `render_sensor_tick_s=0.0`；calibration保存真实blueprint属性。同步器在预热结束时显式选定first_capture_frame，每两个物理帧仅接纳同一预定frame_id的12路数据，其他相位计入 `subsampled_sensor`。它不做最近帧匹配，也不改时间戳，缺失选定帧照样计为失败。

理由：CARLA 0.9.15 [Sensor.cpp](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/Sensor.cpp#L32-L42) 将非零sensor_tick交给每个Actor的浮点TickInterval调度；[官方历史问题3653](https://github.com/carla-simulator/carla/issues/3653)记录过sensor_tick与固定步长不一致。当前故障与此相符，但未通过UE4内部跟踪证明本机具体根因。逐tick输出含义见[官方传感器文档](https://carla.readthedocs.io/en/0.9.15/ref_sensors/)。

代价是传感器渲染/传输20Hz、数据保存10Hz，GPU渲染工作增加；并非免费提速。实际吞吐以该模式自己的benchmark为准。原M2配置与历史证据保持原有sensor_tick=.1语义，新模式显式启用。

## 逐项复核工具

```bash
python -m tools.validate_geometry --scene /path/to/scene --frame-step 5 --output reports/geometry.json
python -m tools.check_depth_photometry --first /path/to/same_pose_geometry_scene \
  --second /path/to/same_pose_training_scene --output reports/depth_comparison.json
python -m tools.visualize_oracle_bev --scene /path/to/scene --index 30 \
  --depth-convention optical_z --output runs/oracle_preview
```

Depth对照工具要求来自compare_imaging的冻结同位姿场景；不能将随机位置两段数据直接比较。六摄geometry输出的motion_passed对静止场景为false是预期；最终门禁分别要求静态cross_view通过及独立运动场景motion通过。
