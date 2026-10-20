# M0–M2 可执行手册

更新：当前Town03已通过Epic模式M2工程门禁；默认墙钟间隔0.3秒。第二轮命令与证据优先参阅 `M2_ACCEPTANCE.md`，下文Town10/0.1秒内容保留为首轮历史。

从仓库根目录执行。技术基线是 `docs/HS-Drive_Technical_Plan_v1.0.md`，实际状态见 `reports/milestone_01.md`。所有录制和 BEV 均为 Oracle/Privileged，不是 Learned 视觉驾驶。

## 环境和依赖

```bash
bash scripts/check_server.sh
bash scripts/bootstrap_sim.sh
source .venv/bin/activate
python -m pip check
python -m pytest -q
```

`requirements-sim.in` 是依赖入口，`requirements-sim.lock.txt` 是本次 Python 3.10/Linux 实装锁定结果。M0–M2 不安装 Torch/CUDA toolkit，不改变驱动；训练环境留待 M4 单独建立。`nvidia-smi` 中 CUDA Version 仅为驱动支持能力，不能当作已安装 Toolkit 版本。

## 服务端下载与启动

包来自官方 Release：https://github.com/carla-simulator/carla/releases/tag/0.9.15 。使用数据盘，压缩包约 8.39GB；解压需要额外空间。

```bash
bash scripts/download_carla.sh /root/gpufree-data/simulators
```

UE4 需以普通用户启动。本容器已有 `ubuntu` 用户。若包由 root 解压，仅将这个新安装目录授予该用户：

```bash
chown -R ubuntu:ubuntu /root/gpufree-data/simulators/CARLA_0.9.15
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15
```

普通用户环境直接运行 `bash scripts/start_carla.sh /path/to/CARLA_0.9.15`。服务端日志建议保存到 `logs/`。只在独占仿真实例使用录制程序，不能与另一个 tick 客户端并行。`--load-map` 会切换 Town03；没有该标志时地图不一致会报错。

## M0 前摄和 M1 六摄

```bash
python scripts/check_carla.py
python -m tools.record_dataset --front-only --frames 10 --load-map \
  --output /root/gpufree-data/hs-drive-data/raw --run-id m0_front
python -m tools.validate_recording \
  --scene /root/gpufree-data/hs-drive-data/raw/m0_front \
  --front-only --output reports/m0_front.json
python -m tools.record_dataset --frames 301 --target-distance 12 \
  --output /root/gpufree-data/hs-drive-data/raw --run-id m1_sixcam
python -m tools.validate_recording \
  --scene /root/gpufree-data/hs-drive-data/raw/m1_sixcam \
  --output reports/m1_sixcam.json
```

301 帧覆盖首末时间差 30 秒。当前采集静止制动车辆，不涉及规划或驾驶控制。目标位置被占用时 spawn 会明确失败，可换 `--spawn-index` 或不放目标。run_id 必须唯一；禁止覆盖已有场景。失败后保留已提交帧，使用新 run_id 重新采集。`--max-ticks` 限制缺帧等待；SIGINT/SIGTERM 会清理自有 Actor 并恢复 world settings。SIGKILL/服务端崩溃不能保证清理，需要重启独占服务器。

每场景保存 calibration、配置、天气、地图、种子、依赖、Git SHA（无 Git 仓库时为 null）、原始数值深度、每路 image.transform、拍摄时刻 ego pose、同快照 Actor boxes、SHA256 和 diagnostics。M1 验收检查全部帧文件、时间连续、同帧外参、RGB 非空、同步异常。开头尚未识别传感器节拍的时段与停止时尾帧不计入录制区间完整率，尾帧单列 shutdown_incomplete。

## M2 Oracle BEV

先分别生成两种假设，比较已知物体与边缘几何：

```bash
python -m tools.visualize_oracle_bev \
  --scene /root/gpufree-data/hs-drive-data/raw/m1_sixcam \
  --depth-convention optical_z --stride 1 --output runs/oracle_z
python -m tools.visualize_oracle_bev \
  --scene /root/gpufree-data/hs-drive-data/raw/m1_sixcam \
  --depth-convention ray_range --stride 1 --output runs/oracle_range
```

输出 `oracle_bev.npz`（总点数及每相机分层点数）、`bev.png`、`height_layers.png`、`cameras_boxes.png` 和 report。框为几何投影，不做可见性剔除，遮挡物后的框也可能显示。

已知平面实验命令（参数必须取自独立几何，不能照抄示例值冒充测量）：

```bash
python -m tools.check_depth_plane --scene /path/to/known_plane_scene \
  --camera front --mask /path/to/plane_mask.npy \
  --plane-normal-opt 0 0 1 --plane-offset-m 10 \
  --output reports/depth_plane.json
```

这里的示例平面是 optical Z=10m。掩码须覆盖实际平面中央和图像边缘，各至少 20 像素；几何位置必须独立确认。审查遮挡、左右镜像、相机重叠区域、已知距离后记录结论，不由 CLI 自动把 M2 标为通过。

## 无服务器集成验收

```bash
python -m tools.synthetic_demo --output runs/synthetic_room_new
python -m tools.visualize_oracle_bev --scene runs/synthetic_room_new \
  --depth-convention optical_z --output runs/synthetic_oracle_new
python -m tools.validate_recording --scene runs/synthetic_room_new \
  --output reports/synthetic_recording.json
```

合成房间测试仅验证数值链路和磁盘回放，不能替代 CARLA 渲染、同步吞吐或 M1/M2 实景门禁。

## 当前服务器的实测配置

Town03 地图切换出现 UE4 Vulkan 崩溃，本轮稳定实验使用 `configs/carla_smoke_town10.yaml`（Town10HD_Opt）。保持原技术基线 Town03 参数，复现实测需给录制命令添加：

```bash
--config configs/carla_smoke_town10.yaml
```

默认 `--warmup-ticks 20 --tick-wall-delay 0.1` 已包含在程序中；它们使物理时间/采样时间保持 20Hz/10Hz，但墙钟采集速度较慢。不要把仿真时间 10Hz 当作实际实时吞吐。

可运行的独立道路平面实验（场景需由本程序独占，不与采集同时运行）：

```bash
python -m tools.carla_depth_probe --config configs/carla_smoke_town10.yaml \
  --output runs/depth_plane_probe
```

该工具将独立 `world.cast_ray` 的道路命中点拟合为平面，仅保留共面道路像素，再比较深度。碰撞几何和渲染网格可能不同，报告保留每个采样点及误差供复核。
