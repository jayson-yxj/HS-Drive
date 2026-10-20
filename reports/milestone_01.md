# HS-Drive M0–M2 实施验收记录

日期：2026-10-19。基线：HS-Drive Technical Plan v1.0 / 架构 V1.3。

本次已完整阅读现有 Markdown、README、四份配置、几何参考与测试、脚本、校验清单和架构图。清单中的 Word 文档及根目录 Markdown 副本未提供；docs 下正文完整可读。文档内下一轮会话指令视为交接内容，实施范围由用户本轮请求确定。

## 实际完成状态

| 项目 | 状态与证据 |
|---|---|
| M0 环境、依赖、版本一致、RGB 非空 | **在 Town10HD_Opt 通过**；RTX 4090 24GB / Ubuntu 22.04 / Python3.10；服务端及客户端 0.9.15；前摄 10 组同帧 RGB/Depth |
| M1 六摄同步采集 | **静止 Town10HD_Opt 30秒场景通过**；301 帧，每帧12路；完整率100%，0间隔缺口、0队列溢出、0混帧；全文件SHA256和同帧外参通过 |
| M2 几何、Oracle BEV、Actor叠加 | **本场景功能和深度定义验证完成；完整门禁尚待补测**；真实六摄4层BEV、Box重投影、独立平面Z/Range实验、32项测试 |
| 原配置 Town03 | **未通过**：切换地图触发 UE4 Vulkan SIGSEGV；关闭异步管线编译的试验也未解决 |
| 行驶、多地图、多天气、坡地、全部六摄重叠区定量评测 | 未实施；不以单静止场景替代 |
| M3–M8 网络、Occupancy、规划及闭环 | 尚未实施，未开始训练 |

M0/M1 的结论仅限本次场景与启动配置，不代表所有地图可用。M2 尚不能标为整个里程碑无条件完成。

## 数据和性能

- 场景目录：`/root/gpufree-data/hs-drive-data/raw/m1_sixcam`
- 仿真帧：28257–28857，步长2；仿真首末时间差30.000000447秒。
- 301帧墙钟146.322秒（包含初始化和预热），约2.06完整帧/秒。**不是墙钟10Hz**。
- 磁盘约1.2GiB；采集中一次 GPU 观测5915MiB，不是持续采样峰值或P95统计。
- 前摄成功场景：`/root/gpufree-data/hs-drive-data/raw/m0_paced`。
- 失败尝试和合成数据独立保存，不混入成功场景。
- 采集后 world synchronous_mode=False，自有vehicle/sensor actor数量0。

## 几何证据

1. `reports/evidence/cameras_boxes.png`：真实六摄，前方目标车绿色3D框与轮廓基本吻合；其余五摄方向/地标检查未发现明显镜像。此为定性检查，非像素误差完整评估。
2. `reports/evidence/bev.png`、`height_layers.png`：四层可见表面点数；空白不是Free。后车身近场、远处遮挡仍需逐场景检查。
3. `reports/target_alignment.json`：前车GT包围盒最近X=9.6333m；选定前表面点最小X=9.6342m、中位X=9.7107m。选择区域中含车体不同表面，不把这个差值当作全体物体检测精度。
4. `reports/depth_semantics_probe_2.json`：独立道路射线建平面，955点；Z假设整体MAE0.429mm，边缘0.433mm；Range边缘3.687m。当前配置采用optical_z。
5. `reports/depth_semantics_probe_1.json`：首处保留15/631个>0.1m异常，可能来自碰撞与渲染几何不一致，不静默剔除。
6. `runs/oracle_z/oracle_bev.npz`：每相机及融合点数，可用于后续重叠区定量分析。`runs/oracle_range` 保留错误假设对照。

## 代码和验证

- 会话管理、12路采集、有界队列、拍摄时刻快照、完整帧提交、停止/异常清理：`simulator/`。
- 相机几何、RGB/Depth解码、右手坐标、OracleBEV、Box裁剪投影：`geometry/`。
- 数据契约、checksum回放、推理输入字段白名单：`datasets/`。
- 命令入口和独立平面实验：`tools/`；步骤见 `docs/M0_M2_RUNBOOK.md`。
- 32 pytest通过（包括原7项），`reports/pytest.xml`、`logs/pytest.log`；`pip check`通过；shell语法检查、模块编译和可编辑安装通过。
- 本工作区原无Git仓库，未编造SHA；每次真实run记录源码SHA256与依赖。安装锁文件 `requirements-sim.lock.txt`。

原始日志包括 `logs/m0_inventory.log`、`vulkan_summary.log`、`carla_smoke_retry.log`、`m0_paced.log`、`m1_sixcam.log`、`m1_validate.log`、`depth_plane_probe*.log`。

## 已知问题及下一步

1. Town03切换崩溃定位到 `FVulkanVertexInputStateInfo::Generate` / Vulkan pipeline creation；日志 `logs/carla_server_retry.log`、`carla_no_async_pipeline.log`。没有更换驱动或重编UE4；不能断言根因已知。最后一次试验服务端已崩溃退出，没有留下运行中的采集任务。
2. 默认每tick墙钟等待100ms、20ticks预热。此前10ms配置出现image.frame与timestamp不一致并被拒绝，证据 `logs/sensor_phase_probe.log` 与失败场景diagnostics。后续优化必须保留同帧校验。
3. 当前软件是静止几何采集器，不启动TM；行驶采集需独立TM所有权及运动测试，保持数据接口。
4. 补齐六摄遮挡/重叠区域量化、坡道/桥下、动态物体与退出中断的真实仿真测试，再关闭M2全部门禁；在此前不进入M4训练。

## 后续更新

上述为首轮历史记录。2026-10-19第二轮已使用Epic解决Town03运行阻塞，补齐运动/遮挡/六摄重叠/坡道/动态目标/中断测试，并关闭限定范围的M2工程门禁。当前状态以 `milestone_02.md` 和 `m2_gate.json` 为准；本文件保留失败历史，不覆盖原始证据。
