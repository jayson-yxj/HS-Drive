# M8-I1：Traffic Manager 有限真实交通数据源 Pilot

**结论：`PilotNoGo / TrainingReadinessNoGo`。唯一真实启动尝试在地图身份检查处停止，尚未验证 TM 能否提供合格六摄车辆／行人数据。** 没有把启动失败解释成 TM、导航或物理机制不可行，也没有重启、换图、换种子或补采。

基线 `502f0701025f2de5a8f2cc1ecbdcd4b3cacd2b6a`；分支 `feat/m8i1-traffic-manager-development-pilot`；检查日期 2026-10-26。实际运行源码已预先提交为 `2a5edce0a78fd0e8d08fd96dd6c27015a8dd5490`，与事后报告、测试新增分别封存。

## 实际执行结果

| 项目 | 实测或状态 |
|---|---|
| 自有 CARLA 启动 | 1 次；Client/Server 0.9.15 版本检查通过 |
| 停止原因 | `RuntimeError: MapMismatch_NoReload`，发生在 World/TM 设置和 Actor 创建之前 |
| 期望地图 | Town02；命令传入 `/Game/Carla/Maps/Town02` |
| 实际地图、Episode、源 Frame | 失败路径未保存，不能推测实际地图或伪造帧号 |
| Owner Tick／交通观测时间 | 0／0s；服务器异步启动阶段推进未测量，不能称总仿真推进为 0 |
| Actor／Sensor／RGB／GT 载荷 | 全部为 0；无 Ego、导航返回或 TM 注册 |
| 完整保存时刻 | 0/12；同步 NotRun，不能解释为 12 次传感器丢帧 |
| 墙钟 | 启动至监督进程结束 23.7454s，预算 1800s |
| GPU | 采样最高 5840MiB；不是完整峰值测量，没有模型前向 |
| 数据 | 报告生成时外部记录 8511 bytes，未产生图像；最终完整 SHA 清单另行封存 |
| 清理 | 首次探针尚有进程组；随后 OS 探针确认自有进程组消失，2100–2102、8100 关闭 |

原始失败、最初清理不完整的瞬时记录和事后复核均保留。World、天气、TM、Physics 尚未修改，恢复这些设置的分支未执行，不能宣称测试了恢复路径。实际执行源和命令见 [启动与 Tick 审计](m8i1_session_and_tick_audit.json)、[资源与清理](m8i1_resource_and_cleanup.json)。

## 八项独立门禁

| 门禁 | 结论 |
|---|---|
| A TrafficGenerated | NotRun：没有创建车辆或行人 |
| B SensorSyncGo | NotRun，工程目标 0/12，保留 SensorSyncPartial |
| C ActorIdentityGo | NotRun |
| D GeometryEvaluable | NotRun |
| E Vehicle20To40mSupport | NotRun，支持 0 |
| F Walker20To40mSupport | NotRun，支持 0；SourceSupportNoGo |
| G NaturalSceneReview | NaturalScenePartial / UnknownBackground / ExactPhysicalContactNotCertified |
| H ResourceAndCleanup | 本次失败路径预算与后续 OS 清理复核完成；有 Actor 时的清理仍未验证 |

## 已完成工程与证据边界

新增独立 `TrafficDevelopmentSessionOwner`、一次性启动监督器、原生来源生命周期记录、24 路 FrameSynchronizer 接入、RGB 与 GT 分目录导出代码和 RGB 输入白名单。保持原 640×384、FOV100、0.1s 六摄 Rig，固定世界 dt0.05s；预声明最多 8 辆车、12 行人、680 个研究 Tick（上限 700），未改默认驾驶模块。

上述 Actor、传感器与导出路径只有源码／局部合同检查，**尚未真实运行，不是完整采集系统验收通过**。渲染身份规则已预声明，但被动身份关联、20–40m 统计和自然场景可视化的真实评估未完成；本次报告记录 NotRun，不能以空数组冒充评价器已验收。

实际文件读取隔离探针使用合成 PNG 和虚构 GT：正常读取六张 RGB；GT 字段、路径越界、绝对路径、符号链接、嵌套 GT 和混帧时间均拒绝；独立低权限进程可读 RGB、不能读测试侧车目录。它证明限定加载器及该测试权限配置的行为，不证明真实采集已隔离，也不是对 root 任意代码的沙箱。详见 [隔离探针](m8i1_rgb_input_isolation.json)。

38 项不同测试通过：新增合同 9 项、历史同步／坐标 17 项、M8-H1 证据 12 项；另有历史足迹 5 条离线回归，包括 3.635626m、4.595351m。预运行的 8 项重复检查不累计为新增测试。最初 Pillow 导入失败保留；新加载器改用环境已有 OpenCV 后通过，没有改 GPU 环境。报告生成器首次重复键错误也已修正，仅影响离线归档，未再次运行服务器。

## 尚未闭合的问题与下一步决策

1. 启动命令包含 Town02，但实际 `get_map().name` 未符合合同。失败分支没有落盘实际名称、Episode 或 XODR；现有日志也未给出加载地图。启动参数、启动完成时机等只能作为后续排查假设，根因未确定。
2. Runner 捕获错误后返回 0，Supervisor 的 `failure=null` 仅表明没有监督器级异常。必须读取 Session failure；不能用 shell 成功当 Pilot 成功。
3. 运行后来源记录器曾把一个未执行且随后删除的 `datasets/m8i1_evidence.py` 草稿哈希纳入 OS 探针的目录快照。该哈希不是导入／执行证明，草稿未进入实际运行提交；真实运行 8 项源码 SHA 已逐项对提交验证。保留此不完整草稿记录，不补造源码。

**优先建议仅排查启动地图身份与失败日志／退出码，暂不扩大交通或另建放置器。** 本任务书规定一个 Session 且地图身份不符立即停止，因此本轮到此结束；新的服务器尝试须另行授权。即使将来启动通过，还须完成真实身份评价器和单次受限采集验收，不能直接进入训练。

历史 16562 条空间索引及 260 项 G/H0/H1 封存记录不变。新增 Validation／Calibration／Final、独立位置组均为 0。PETR 前向、训练、视觉控制均为 0；M8-H1 NoGo、PETR 质量／时延、Occupancy 和视觉安全的既有状态均未升级。没有合并 main。

详细交付见 [验收 JSON](m8i1_acceptance.json)、[实现及复现](../docs/M8I1_IMPLEMENTATION.md)、[架构决策](../docs/M8I1_ARCHITECTURE_DECISION.md)。
