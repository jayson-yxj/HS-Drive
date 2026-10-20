# M2 工程门禁关闭记录

2026-10-19。继承v1.0/V1.3架构，本轮不修改网络设计。**M2工程门禁在CARLA 0.9.15 / Epic / Town03 / 当前六摄安装配置下通过。** 机器可读结果 `reports/m2_gate.json`，所有10项检查为true，每份证据附SHA256。

## Town03 渲染

同一官方包、驱动580.126.09和4090，Low画质出现Vulkan SIGSEGV，改Epic后成功加载Town03，并通过Town10→Town03→Town03连续重载、六摄30秒录制及后续实验。启动默认值已改Epic；client地图超时从15秒增至60秒。

这是实测配置规避，**Low模式底层问题尚未修复、根因未确定**。没有改驱动或重编UE4。旧的异步pipeline编译试验未证明参数生效，更不能当作根因排除证据。

日志：`logs/m2_epic_map_probe.log`、`m2_epic_reload.log`、`m2_epic_server.log`。

## 验收结果

| 验收项 | 实测结果 | 证据 |
|---|---|---|
| Town03六摄30秒 | 301帧、12路同帧、完整率100%、无同步异常；墙钟268.13秒 | m2_town03_recording.json |
| 自车运动 | 61帧/6秒，位移7.404m，最高2.088m/s；完整性通过 | m2_town03_motion_geometry.json |
| 六摄重叠 | 6组相邻、12个方向全部通过；每方向至少24450点，最差P95=0.01370m | 同上 |
| 时序运动补偿 | 6相机全部通过，最差P95=0.000353m；前摄不补偿负对照P95=0.3899m，补偿后0.000254m | 同上 |
| 坡道 | 21完整帧；实测ego俯仰−5.324°、世界高度7.898m；全部相邻双向一致性通过 | m2_town03_slope_geometry.json |
| 遮挡/显露 | 卡车遮挡736个目标表面像素，遮挡覆盖率100%；移除后中心区域可见率100% | m2_occlusion.json |
| 动态目标 | 41帧/4秒，横移2.00m，每采样帧至少446个表面点落在同帧GT框内 | m2_dynamic_actor_surfaces.json |
| 车身审查 | 五摄ego AABB命中0，后摄最高8.23%（低于10%诊断阈值） | 同上 |
| 中断恢复 | SIGTERM退出码130，已提交3帧校验回放成功，无Actor泄漏，世界恢复异步 | m2_interrupt.json |
| 深度定义 | 沿用独立平面确认的optical_z，不改写旧标定 | depth_convention.json |

重叠与时序统计仅针对2–40m平滑可见表面，深度不连续边缘剔除、遮挡单独计数；不能将毫米级统计外推为全场景感知精度。阈值在实验前存于 `configs/m2_acceptance.yaml`，并非根据结果放宽。

动态目标通过逐tick Transform横移，明确为运动学几何实验；自车运动是低速物理直行。遮挡实验为受控前摄场景，不覆盖所有遮挡类型。后摄有实际车身遮挡，保留安装配置和证据，不宣称零盲区。

原始数据目录均位于 `/root/gpufree-data/hs-drive-data/raw/`：
- `m2_town03_30s`
- `m2_town03_motion_paced`
- `m2_town03_slope`
- `m2_town03_dynamic`
- `m2_town03_occlusion`
- `m2_interrupt`

首轮0.1秒墙钟间隔的运动录制有frame/timestamp异常，保留为失败诊断 `m2_town03_motion`；它不计入门禁。默认tick墙钟间隔改0.3秒，仿真物理0.05秒、采样0.1秒未变。

## 代码与测试

新增双向/时序深度验证、动态目标及车身审查、受控遮挡、真实中断测试、门禁汇总和指标图工具。采集器增加低速自车运动与运动学目标选项，数据结构保持兼容。

37项pytest通过，涵盖空重叠不能通过、错位姿负对照、遮挡/边缘区分、门禁拒绝缺失运动证据。`reports/pytest_m2.xml`、`logs/pytest_m2.log`。运行复现见 `docs/M2_ACCEPTANCE.md`。

可视化：`reports/evidence/m2/overlap_metrics.png`、`occlusion.png`、`dynamic_boxes.png`、`slope_bev.png`、`cameras_boxes.png`。

## 关闭边界及后续

本次关闭的是M2 Oracle几何工程门禁，允许进入M3数据标准化。并不证明完整Occupancy、Free/Unknown、学习型感知、控制闭环、实时吞吐或安全能力。M4前仍需完成M3数据集/分割/输入泄漏边界。后摄盲区、桥下/更多坡地、夜雨及更多车型可作为扩展回归集持续补充，不伪称已覆盖。

全部自有Actor已清理，验收结束后主动停止本轮CARLA服务端以释放GPU；下次按Epic命令启动。
