# M2 工程门禁与复现

沿用 V1.3 架构。本轮只处理 CARLA 渲染、Oracle 几何、运动与遮挡，不实现 Learned BEV 或规划。

## Town03 启动修正

本服务器 CARLA 0.9.15 + RTX4090 + driver 580.126.09 在 Low 模式切换 Town03 会在 Vulkan pipeline creation 崩溃；同包同驱动改 Epic 后成功，包含重复加载 Town03。这是**经实测的运行配置规避**，并非已修复 UE4 源码或确定驱动根因。启动脚本默认改 Epic，保留第三参数 Low 用于复现旧问题。

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2000 Epic
```

普通用户直接运行脚本。仍使用 `-RenderOffScreen`，不能启用 no_rendering_mode。新场景 manifest 记录声明的画质；它是操作者声明，CARLA API 未提供实际画质读回验证。服务端启动日志是画质的原始依据。

## 运动与同步

默认墙钟 tick 间隔由0.1秒改为0.3秒；仿真步长0.05秒、相机sensor_tick=0.1秒保持原定值。Epic下0.1秒曾出现frame/timestamp不一致，失败数据保留；没有放宽同帧/外参阈值。允许更慢的离线采集，不宣称实时性能。

```bash
source .venv/bin/activate
python -m tools.record_dataset --load-map --frames 61 --motion straight \
  --output /root/gpufree-data/hs-drive-data/raw --run-id motion_new
python -m tools.validate_geometry --scene /root/gpufree-data/hs-drive-data/raw/motion_new \
  --frame-step 5 --output reports/motion_new.json
```

低速直行仅用于仿真几何实验，保持约2m/s，不是导航控制器。目标运动单独使用 `--target-distance 14 --target-lateral-speed 0.5`，通过CARLA Transform逐tick驱动目标；报告明确标记运动学运动，不声称真实车辆动力学验证。

## 重叠、时序与遮挡指标

`configs/m2_acceptance.yaml` 在本轮实验前确定阈值：相邻六组双向共12方向，每方向至少100个可比较点，median绝对深度误差≤0.10m、P95≤0.50m、0.50m内比例≥80%。采样距离2–40m，剔除目标3×3深度patch跨度≥0.20m的边缘；投影落在近处表面之后的点记为遮挡，单独计数，不能当成空闲空间。

时序测试使用同一相机历史深度，按独立ego位姿转换到当前时刻，对比当前深度。报告附**不做自车补偿的负对照**；两者差别说明实际使用了位姿。范围限制为静态平滑表面；动态参与者另测。全部场景先逐帧检查文件校验、时间戳和外参，完整性不合格时几何门禁也不得通过。

```bash
python -m tools.occlusion_probe --output /root/gpufree-data/hs-drive-data/raw/occlusion_new
python -m tools.validate_actor_surfaces --scene /path/to/dynamic_scene --output reports/dynamic_new.json
```

遮挡场景：前方卡车完全遮住更远目标车，移除卡车后显露；按独立GT目标包围盒选出显露表面，检查遮挡时深度确实更近；另检查目标中心投影区域显露。当前为前摄受控场景，不声称所有遮挡类别覆盖。遮挡阶段和显露阶段之间有预热，故两帧不是连续采集样本，不用于M1时间连续性门禁。

车身审查统计每相机深度点落在ego AABB的比例（阈值10%为诊断初值），结合六摄截图查看近场遮挡；不能据此宣称无盲区。

## 门禁汇总

```bash
python -m pytest -q
python -m tools.validate_m2_gate --output reports/m2_gate.json
```

汇总器读取指定证据报告、记录SHA256；任一必要检查失败则返回非零状态。结论仅覆盖当前版本、画质、地图和相机安装配置。后续更改相机/FOV/分辨率/深度语义/采样时钟需重新验证。完整Occupancy、Free/Unknown、学习模型、天气泛化与道路安全均不在M2通过声明之内。
