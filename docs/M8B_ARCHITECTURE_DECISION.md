# M8-B Architecture Decision — PilotOnly / ZeroShotDomainGap

依据：base `a3a892219e29c3c16d969e8b0d934173a6f8dbb0`，官方PETR固定 `f7525f93467a33707ef401c587a52d5e7b34de74`，Checkpoint SHA `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`。实际运行、源码、测试、数据和各报告哈希见本轮release_manifest。

| 独立决策 | 状态 | 理由 |
|---|---|---|
| A 六摄Adapter | AdapterContractGo | 白名单、同步/SHA、官方增强、刚体投影与box约定检查通过；坡道upright近似局限显式记录 |
| B 真实输入完整前向 | CARLAForwardGo | 192/192帧、全部57,600候选、无NaN/OOM、558权重张量不变、0训练步 |
| C 自然检测 | CandidateNoGo / ZeroShotDomainGap | score0.1中心2m召回45/155；20–40m仅1/83；不完整标注下11,045未匹配候选不能认证Precision |
| D 条件定位 | QualityNotValidated | 无中心截止的投影关联P95 18.57m、>5m56.52%；不是全体目标风险，且关联仍为代理 |
| E 风险覆盖 | NoGo_NoQualifiedRiskCoverage | Calibration支持与错误率同时不合格；不使用旧Sealed反选阈值 |
| F 离线资源 | LimitedOfflineResearchGo | 24GB可运行；E2E P95约338ms，在线周期与共享GPU安全不通过 |
| G 后续训练 | NotApproved / ProposalOnly | 未证实几何参考分布与外观贡献；数据右侧/远场/完整背景缺口、新盲测均未补齐 |

保留PETR作为冻结、只读Offline Shadow研究工具，不能替代当前默认Encoder、Depth、BEV或任何安全关键Oracle输入。官方预训练能够运行和部分近距离目标识别，不等于CARLA域可靠感知。Mobile/DirectCenterHead旧路线不恢复扩训；本次也不能批准正式Occupancy或视觉驾驶。

下一主研究请求可以是**有界的PETR参考分布与标签支持诊断**，先明确nuScenes模型参考原点高度、CARLA rig、尺度和裁剪语义差异；每个变化独立版本和预声明对照。其后是否申请固定预算域适配需要单独决策，不在M8-B自动执行。不能根据当前已打开的Sealed选新参数；任何训练后需新独立最终位置。

Oracle规划工程仍可在其已有授权和特权标记下独立推进。本轮没有车辆实验；M7在线NoGo、M5感知/Occupancy开放门禁、共享GPU严格20Hz未验证状态原样保留。没有视觉到控制连接，没有新时效放宽。
