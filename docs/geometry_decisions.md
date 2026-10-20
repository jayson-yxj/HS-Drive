# M0–M2 几何与实施决策（2026-10-19）

基线仍为 `HS-Drive_Technical_Plan_v1.0.md` / V1.3。未修改网络架构。

- 复用 `geometry/reference.py` 作为唯一基础数学实现，新文件提供明确职责的接口。
- 像素中心沿用交接代码的整数 `(u,v)`、主点 `((W-1)/2,(H-1)/2)`；不得与主点 W/2 混用。仍须检查渲染器半像素偏差。
- `T_hs_ego_from_opt = S @ mount @ C`。HS world 同样右手化，`T_hs_world_from_hs_ego = S @ T_carla_world_from_carla_ego @ S`。
- 线速度是极向量，用 S；角速度是轴向量，反射时用 det(S)S，并将 deg/s 转 rad/s。
- 原始 BGRA 深度直接解码 float32 米数；默认 Oracle 仅使用 [1.5,80)m，超界排除而不截断为 80m。
- BEV 存储 `[height=4, X=160, Y=120]`；可视化单独翻转，使 +X 向上、+Y 向左。空格未被标注为 Free。
- 相机使用 Rigid attachment，RGB/Depth 同位姿、FOV、尺寸、sensor_tick，lens_k/lens_kcube=0；保存实际 sensor attributes。
- 同帧外参误差上限暂用矩阵元素绝对误差 2e-3（含米制平移和旋转元素），为工程初值，后续应拆成角度/平移阈值。
- M0–M2 默认静止制动车辆，避免加入尚未验证的控制器；可另放置静止目标车辆用于 3D Box 重投影。未启用 Traffic Manager，因此不改变外部 TM 同步状态。0.9.15 Python API 没有获取 TM 同步状态的 getter，不能伪称恢复未知状态。后续行驶采集需单独拥有 TM 生命周期。
- 数据帧以独立 `frames/*.json` 作为最后提交标记；它替代容易出现半行的主帧 JSONL，保留基线 RGB/Depth/labels 路径。异常后可读取已提交数据；重启仿真使用新 run_id，不混接旧场景。
- 只写入已生成的模块，不为 M4–M8 放置看似可用的模型占位代码。

深度数值定义仍需独立已知几何实验。`--depth-convention optical_z` 是显式假设，不是门禁证明。`tools.check_depth_plane` 接受独立已知平面 `n·p=offset`、人工审查的 bool 掩码，分别报告中心/边缘的两种假设 MAE。不可从待检深度自身拟合平面再宣称验证成功。

正式 learned_vision loader 尚属 M3，当前提供输入字段白名单验证器，未声称实现完整递归数据泄漏审计或模型推理。

API 核对来源：
- https://carla.readthedocs.io/en/0.9.15/ref_sensors/
- https://carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep/
- https://carla.readthedocs.io/en/0.9.15/tuto_G_bounding_boxes/

## 实测引出的调整

- 首次低延迟（每物理 tick 仅 sleep 10ms）测试出现相机 frame_id/timestamp 不一致；未放宽校验。增加 20 ticks 预热和默认 100ms 墙钟间隔，仿真时间步仍为 0.05s、传感器 0.1s。这样以吞吐换取可验证的基础采集，不能宣称实时 10Hz 墙钟性能。`--tick-wall-delay` 供后续性能实验调节。
- 默认 Town03 切换触发 UE4 Vulkan 崩溃；首轮在 Town10HD_Opt 独立配置完成验证。原 `configs/carla.yaml` 保持 Town03，替代配置 `configs/carla_smoke_town10.yaml` 不代表基线地图已验收。
- 服务端启动移除继承的远程桌面 DISPLAY，并显式选择 NVIDIA ICD。此组合改善了启动，尚不能断言 DISPLAY 是唯一根因。

## 已知平面实测结论

`reports/depth_semantics_probe_2.json`：10m 向下相机，独立道路碰撞射线建立平面，955 个道路点（中央 77、边缘 471）。optical_z 的整体/中央/边缘 MAE 约 0.429/0.426/0.433mm；ray_range 边缘 MAE 3.687m。对于本次 CARLA 0.9.15、640×384、100°、零畸变配置，采用 **optical_z**。

首处道路实验保留了 15/631 个 >0.1m 异常点，可能来自道路碰撞网格与可见物体差异，不能用筛掉异常后的统计宣称全场景毫米精度。第二处独立位置无 >0.1m 异常。证据仅确认本配置的深度定义，不外推到所有地形、遮挡和参数。

录制标定仍写 `unverified`，因为采集发生在上述实验之前；不回写或篡改已哈希的 calibration。后续 Oracle 使用报告 `reports/depth_convention.json` 作为独立解释证据。

## 第二轮实测更新

Town03改Epic后稳定，默认tick墙钟等待调整为0.3秒。M2工程门禁在当前配置关闭，证据见 `../reports/milestone_02.md`。后摄ego AABB命中最高8.23%，确认存在车身遮挡；保持安装参数但记录有效近场限制。六摄深度/时序验收阈值见 `../configs/m2_acceptance.yaml`。

## M3增量

已实现独立input/target派生索引、整场景/路线组拆分、RGB-only推理读取及受审计张量边界；详见[M3数据接口](M3_DATASET.md)。旧白名单是历史起点，现额外拒绝嵌套/非数值载荷。
