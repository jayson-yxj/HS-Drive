# M8-I1b 架构决策：结束无证据的启动修补路线

**B：`StartupRootCauseUnresolved`。** 基础提交 `5c14c347de2b183d59d7bd3f604ec45d0b602da6`；分支 `feat/m8i1b-offline-startup-forensics`。本轮仅本地文件、Git及已有OS日志审计，CARLA启动、RPC连接、Tick、Actor、Sensor、TM、采集、模型前向、训练和控制全部0。

原M8-I1的版本RPC成功／地图拒绝与M8-I1a的31次版本RPC超时是不同证据层级，不能合并归因。H1最终World和XODR可用，但预检允许加载Town02；它不构成同命令、同CWD、同就绪策略的受控成功基线。

本轮找到了可核查差异，但没有错误来源证据：无可归属的新UE启动日志、无对应窗口GPU/OOM/权限错误、无RPC内部轨迹。默认地图、2s超时、Client重用、GPU99%和EOF消息都不足以各自确定原因。因此不选择A或C，不申请新启动验证，不修改启动器、等待时间或地图参数。

唯一建议的下一方向是Bench2Drive单片段许可与数据访问条件评审。先复用M8-I0现成接口与许可差异清单，明确一个官方片段的来源、许可适用范围和可访问性；数据未取得前不宣称Ready。此建议不授权下载、220路线评测、视觉模型训练或车辆控制，不自动开展该评审。

授权来源按用户本轮提供的外部转录勘误：外部user申请、外部assistant批准、user向Codex转述执行。原凭据只描述本地接收角色，保留原文SHA并追加来源层次，外部平台ID/时间/导出未获独立验证。已发生的一次执行和已消耗额度不变，新授权为0。

交付见[总结](../reports/m8i1b_summary.md)、[启动路径比较](../reports/m8i1b_success_failure_launch_diff.json)、[状态链](../reports/m8i1b_rpc_readiness_evidence.json)、[根因假设](../reports/m8i1b_root_cause_hypotheses.md)及[授权勘误](../reports/m8i1b_authorization_record_erratum.md)。历史模型、失败报告、默认驾驶系统均未修改，不合并main。
