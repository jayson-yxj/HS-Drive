# M8-I0：现成交通场景与数据源复用评审

**选择原生 CARLA 0.9.15 Traffic Manager 作为唯一优先路线，Bench2Drive 单片段离线审核作为条件备用。只形成下一项 Pilot 提案，没有执行 Pilot。** 当前没有新数据源获得自然场景或训练资格；`SourceReviewCompleted / DevelopmentPilotProposalEligible` 与 `NoSourceReady`（已验证数据层）同时成立。

依据提交 `69ce3158ab692387e615a781deb41aae0fb61499`，独立分支 `feat/m8i0-existing-scenario-data-review`。检查日期 2026-10-26。证据为本机文件/SHA 审计、固定版本官方源码/XML/Schema 和公开网页；本轮 CARLA 连接、Actor、传感器、RGB、GT 采集、PETR 前向、训练均为 0。源码下载后只作文本/AST 解析，未安装或执行外部框架。

## 历史结论封存

M8-G/H0/H1 的 260 项 manifest 记录逐项哈希通过。M8-H1 两处有效 Ego 均真实生成并在 40 Tick 达到静止标准；每处 15 点支撑均无命中，目标与传感器 NotRun、新 RGB=0。另有两次首 Tick 前读取缓存导致的无效 Ego 测量，不能遗漏。保留 `DevelopmentSmokeNoGo / TrainingReadinessNoGo`。

H1 中 OBB 底部约 0.213643m，射线起点约 0.208643m，先前道路表面约 0.22m。向下射线的起点已低于先前表面约 11.36mm；这解释了检查的适用性风险，**不证明当前接触、悬空或物理生成不可能**。未来采用独立证据合同，不能追认 H1 Passed，也不继续定制放置器和 M8-H2 阈值补丁。

M8-B 冻结 PETR 192 帧推理成功，但既有标注 Recall 45/155、20–40m 仅 1/83、代理关联 XY P95 18.57m；端到端 P95 337.94ms。11,045 个未匹配候选不是全场物理误检。零样本质量、在线实时性、Occupancy、视觉安全均保持原 NoGo/Partial。

## 决策依据

| 路线 | 实际检查发现 | 本轮状态 |
|---|---|---|
| A 原生 TM | 已安装生成示例与官方 0.9.15 文件 SHA 完全相同；车辆原生 spawn、行人导航+AI controller，来源及动态 Box 可经 API 获取；需一个 Tick Owner 和独立采集审核 | 优先提案；真实来源质量 NotRun |
| B ScenarioRunner | 固定 v0.9.15，审核三类场景、26 个 XML 实例；有行为触发与清理，但行人隐藏/替换、额外 Z 偏移、恒速干预和旧依赖增加自然状态与集成成本 | `ScenarioExecutionCandidate_GTPartial`，本次不列备用 |
| C Bench2Drive | 六摄和丰富标注有公开源码；旧 Mini 清单 10 段、2,812,733,742 字节；原始帧时间证据不足、静态匹配风险、许可声明差异；服务器未取得数据载荷 | 条件备用；单片段申请前先澄清许可/访问 |

以上是适配成本和证据潜力比较，不是运行成功率比较。TM 没有保证行人和车辆同时在 20–40m 可见，也没有提供现成六摄标注数据包。

## 强制回答 A–G

A：**有现成场景来源能力，尚无本轮实际场景合格证据。** TM 控制车辆，行人由独立 `controller.ai.walker` 控制；不能说 TM 自己控制行人。

B：**事件更可控，GT 导出与自然状态并不自动更可靠。** v0.9.15 有相应场景；默认 SensorInterface 未验证所有标签具有相同 frame，且普通 scenario 命令不接受 `--agent`。需要额外只读 Evaluator，不能直接挂一个六摄 Agent 即宣称完成。

C：**存在可低成本申请的官方单片段，尚不能标记合法接入 Ready。** GitHub LICENSE 为 CC-BY-NC-ND 4.0；官方 HF 数据卡声明 Apache-2.0，资产范围未澄清。本机 HF 元数据请求失败，网页可读不等于服务器能取得文件。没有下载数据。

D：**已有 Bench2Drive 数据最少依赖自定义放置；TM 是本机最低新增依赖的仿真来源。** 两种成本不能混为一谈。

E：**现成字段丰富度以 Bench2Drive 更强；可追溯原始帧/Actor 生命周期的审查以 TM 新 Sidecar 更直接。** 二者均不具备完整独立 Mesh、全部静态身份和精确接触真值。不存在已确认全背景 GT 的优胜者。

F：**优先 TM。** 复用当前服务器、六摄标定、FrameSynchronizer 和身份审计，不引入旧版 Python 场景依赖、不运行模型。仅写有限编排与 Evaluator，不维护新的位置搜索器。

G：**值得申请一个 TM 有界 Development Pilot。** 一次、一个已有 Town、最多 30s 观测、最多 8 辆车（含观测 Ego）+12 个行人；只用原生推荐来源，不搜索修补，未见支持照实失败。详细提案见 [边界与退出](m8i0_next_pilot_proposal.md)。本阶段不批准或执行。

完整决策见 [架构文档](../docs/M8I0_ARCHITECTURE_DECISION.md)，字段级状态见 [GT 矩阵](m8i0_groundtruth_compatibility_matrix.json)，执行与冻结检查见 [验收](m8i0_acceptance.json) 和 [证据清单](m8i0_evidence_inventory.json)。
