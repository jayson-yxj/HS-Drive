# M8-I1a：启动地图身份与失败传播诊断

**离线门禁 `OfflineInstrumentationPassed / StatusPropagationGo`；真实启动 `NotRun`，实际地图 `StartupIdentityUnknown`。** 本轮先完成独立诊断入口和可复核故障测试，尚未取得任务书要求的第二项、单独运行授权，没有启动 CARLA。

基线：`916ba75072259f17da725ddbc9c76f2f5fbeea13`；分支：`feat/m8i1a-startup-identity-diagnostic`。全部旧 M8-I1 源码、报告与外部日志逐项 SHA 验证通过，旧结论仍为 `PilotNoGo / TrainingReadinessNoGo`。新实现不修改旧入口，也不将旧 Session 追认为通过。

## 实现及实际测试

新只读 Runner 先读取实际 Map、Episode、Snapshot、版本、设置、天气及 XODR 哈希，使用排他创建、文件 fsync、目录 fsync 持久化身份，然后判断版本、地图名与 XODR。Map RPC 失败保存 null 实际身份和异常，不代填 Town02；不调用加载地图或 Reload，不创建 Actor、Sensor、TM，不 Tick、不改 World。

业务失败在 finally 保存 Cleanup 和 Session Result 后返回非零退出码。Supervisor 组合检查退出码、同一 run_id 的业务状态和清理证据；矛盾状态明确标记 `StatusPropagationNoGo`。真实进程组清理保留瞬时检查，最终 OS 探针不通过则不能成功。

**25 项离线测试通过**：10 个真实 Python 子进程场景，含成功、地图不符、XODR 不符、Map RPC 失败、普通异常、清理失败、SIGTERM、SIGKILL、连接失败及版本不符；另覆盖 6 类状态矛盾、监督器故障、身份写入失败、排他保护、授权拒绝和只读调用检查。测试中的 World/Map/Frame 是 Mock，未连接 CARLA。SIGKILL 无法执行 finally，保存先前身份并报告缺失结果，不伪造清理成功。详见 [故障注入报告](m8i1a_failure_injection_tests.json)。

## 静态检查发现

安装包 Town02 资源存在且与冻结 SHA 一致，XODR 为 `953c05f17def231239ffcadba3307628a82d0d335b67b4e29c3098f7aed1dd7d`。旧完整 argv 确实包含 `/Game/Carla/Maps/Town02`，位于 CarlaUE4.sh 之后；包装脚本向 Shipping binary 传入 `CarlaUE4` 和余下参数，参数解析行为未实测。

包内 `DefaultEngine.ini` 的默认地图字段指向 Town10HD_Opt，**不能据此推断上次实际加载 Town10HD_Opt**。旧 `scripts/start_carla.sh` 先进入安装目录，而 M8-I1 Popen 未指定 cwd；旧真实进程 CWD 未保存，无法恢复。服务器原日志没有地图加载记录。以上只能形成待证假设，不证明 CWD 或参数是根因。

## 退出问题 A–G

| 项目 | 结论及证据层级 |
|---|---|
| A 实际启动地图是否记录 | 尚未，新启动 NotRun；旧实际地图继续 Unknown |
| B 地图检查是否在身份落盘后 | 离线通过；测试在校验入口读取已持久化 JSON 并核对内容 |
| C 地图拒绝是否非零传播 | 离线通过，真实 Python Runner rc=12，监督合同 NoGo；CARLA 全链待运行 |
| D 根因是否确定 | 未确定；资源完整性有证据，参数／CWD／就绪原因均未证实 |
| E 是否完成真实只读启动 | 未执行，待单独授权 |
| F 是否未生成交通／Sensor／Owner Tick | 是，本轮均为 0；模型前向和训练同为 0 |
| G 是否完整清理自有服务器 | 本轮没有自有服务器，NotRun；Python 测试进程已回收，旧清理证据保持封存 |

## 可审阅的下一次运行

范围固定为一次自有服务器、Town02、RPC2100、原 Epic／RenderOffScreen argv、当前仓库目录作为显式 CWD，≤180s 墙钟、≤64MiB 日志。CWD 只与旧代码的继承意图一致，不能声称与未记录的旧实际 CWD 已严格匹配。唯一研究变量是被动身份观测：匹配时最多两次读数，地图不符时先留证后立即拒绝；不换图、不改资源、不重启。

成功只允许 `StartupIdentityGo_NoTraffic`。身份及传播门禁未同时实测通过前，不提出下一次交通采集 Pilot 放行，不批准训练、模型替换、视觉控制或 Final。

交付：[验收](m8i1a_acceptance.json)、[静态审计](m8i1a_static_launch_audit.json)、[根因评估](m8i1a_root_cause_assessment.md)、[实现与复现](../docs/M8I1A_IMPLEMENTATION.md)、[架构决策](../docs/M8I1A_ARCHITECTURE_DECISION.md)。
