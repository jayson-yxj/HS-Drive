# M8-I1a 获批单次执行结果

**`StartupDiagnosticNoGo / StartupIdentityUnknown / StatusPropagationGo / CleanupGo`。** 已执行本次对话批准的唯一一次只读启动；连接窗口内版本 RPC 始终超时，未到达 World/Map 身份读取。没有修改参数、重新启动或进入 Traffic Manager Pilot。

执行提交 `2cc22439587976d87550bf411e22f625d30508d0`，分支 `feat/m8i1a-startup-identity-diagnostic`；run_id `b1d40dfe-3691-46e5-9377-96090a8d0cce`。授权记录保存本次用户“批准了”及五项要求原文，以对话相对位置引用；消息 ID 不可见，明确为 null，没有编造。

本报告为**最新实测状态**。原 `m8i1a_summary.md`、`m8i1a_acceptance.json` 等 NotRun 文件是批准前已封存快照，保持只读；本次采用 `m8i1a_execution_` 追加结果，不能以旧 NotRun 覆盖已发生的启动，也不覆盖原 M8-I1 失败。

| 检查 | 实际结果 |
|---|---|
| 启动前版本与分支 | `2cc2243`，目标分支，工作区干净 |
| 冻结与离线门禁 | 43 项启动输入 SHA 一致；25 项已通过离线测试及 87 项封存文件核验 |
| 启动前端口 | 2100、2101、2102、8100 均关闭；无 CarlaUE4 Shipping 进程 |
| 自有服务器启动 | 1 次，保持原 argv、资源和显式仓库 CWD；无第二次启动 |
| 连接 | 31 次 `get_server_version()` 均为 2000ms 超时；成功次数 0 |
| 真实地图／Episode／Frame／版本RPC／XODR | 全部 Unknown，未到达读取；没有用期望值填充 |
| Runner | `ClientConnectionFailed`，实际退出码 **10** |
| Supervisor | `StartupDiagnosticNoGo`，实际退出码 **1**，无状态矛盾 |
| 总墙钟／日志及记录 | **94.154s**／runtime目录 **29,508 bytes**，低于180s/64MiB |
| 清理 | 首次仍有服务器进程与端口；21次探针保留过程，最终及独立事后 OS 探针均通过 |
| 本任务创建 Actor／Sensor／TM、Owner Tick | 全部 **0**；无图像、GT、模型输出或训练 |

90s 连接边界在失败 RPC 返回后检查，最后一次 2s RPC 可以跨过该连接边界；并未延长协议的180s总墙钟预算。操作系统级进程和 TCP 探针没有触发 CARLA RPC、World Tick 或 Actor 创建。服务器自身异步启动推进未测量，不能声称引擎总 Frame 数为0。

## 新取得的真实证据

已在启动早期持久化完整进程参数及 CWD。包装脚本 CWD 为仓库目录；Shipping binary 实际 CWD 为 `/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4/Binaries/Linux`。其命令包含 `CarlaUE4 /Game/Carla/Maps/Town02` 及原渲染、质量、端口参数。不同进程 CWD 不同，不能把 Supervisor 请求 CWD 当成二进制实际 CWD。

运行中 OS 探针发现2100–2102可建立TCP连接，GPU采样6004MiB、99%利用率；这没有提供成功的CARLA应用层RPC。退出后GPU为78MiB、0%，无匹配服务器进程，四个端口全关闭。

`server.log` 未给出已加载地图，退出阶段有 EOF/stream/Operation canceled 文字。日志没有逐行时间戳，TCP探针与进程关闭也可能影响这些记录，不能据此归因为地图错误、渲染崩溃或RPC内部故障。根因仍为 **Unresolved**；本次也不能倒推旧M8-I1实际地图。

## A–G 与后续边界

- A：实际地图尚未取得，原因是连接失败。
- B：身份先持久化再判断的逻辑已离线验证；真实运行未进入该阶段，没有作地图判断。
- C：本次真实连接失败已正确传播 `10 → 1 → NoGo`；真实地图拒绝路径本次未触发，仍只有离线证据。
- D：实际 argv/CWD 已有证据，地图启动／RPC失败根因尚未确定。
- E：完成一次真实只读启动尝试，身份确认目标未实现。
- F：没有创建交通、Sensor、TM，没有Owner Tick、World或天气修改、模型前向。
- G：自有进程与端口最终完整清理；瞬时失败记录保留，状态 `CleanupGo`。

本次额度已消耗。地图身份及全链成功门禁未通过，**不能提出 TM Pilot 放行，不自动重试、不训练**。如继续研究，应先离线审查启动/RPC可观测性与已有日志，再另行确定有界实验；本轮不调整参数猜测根因。

交付：[验收](m8i1a_execution_acceptance.json)、[身份及进程](m8i1a_execution_map_identity.json)、[就绪失败](m8i1a_execution_startup_readiness.json)、[退出传播](m8i1a_execution_exit_status.json)、[清理](m8i1a_execution_cleanup.json)、[批准与前检](m8i1a_execution_preflight.json)、[复现与证据边界](../docs/M8I1A_EXECUTION_RECORD.md)。
