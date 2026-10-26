# M8-I1a 架构决策

**接受独立只读启动诊断的离线实现，保持真实身份与根因门禁未通过。** 状态 `OfflineInstrumentationPassed / StatusPropagationGo（离线） / StartupIdentityUnknown（真实）`。

采用新 Runner/Supervisor 而不修改旧 M8-I1，是为了永久保留发生过失败的执行源码及 SHA。它只实现身份与失败传播修复，不导入旧 TrafficDevelopmentSessionOwner，不建立另一套交通或放置器。未来交通模块可以在另行批准后复用已实测的启动合同，本轮不接入默认驾驶。

25 项 Mock/OS 子进程测试证明身份先落盘、异常与清理拒绝、非零退出传播和矛盾闭锁。没有新的 CARLA 连接。进程实际地图、就绪时间、真实清理与服务器预算尚未验证，不能升级为 `StartupIdentityGo_NoTraffic` 或 `CleanupGo`。

原生 TM 路线的可行性仍未回答。包内默认地图字段、旧脚本 cd 和原 argv 提供排查线索，不提供因果结论。下一项唯一建议是本任务书限定的一次只读启动，维持原 argv、资源和预声明 CWD，增加被动身份观察。禁止同时换工作目录到安装根、改变地图参数和延长等待来获取成功。

任务书第六节要求新的独立运行授权，因此本次先交付代码、25 项测试、静态审计、预运行 SHA 清单及实际启动 NotRun 状态，再提交具体运行请求。未获授权不启动；获批仍只能一次、180s、64MiB、无TM/Actor/Sensor/Owner Tick。实际地图不符先留证后停止，不换图、不Reload、不自动重试。

下一次 TM Pilot 申请资格当前为 false，须地图身份和失败传播同时通过才可提出。任何本阶段结果均不批准训练、模型替换、视觉控制、正式 Occupancy 或独立 Final。旧 `PilotNoGo / TrainingReadinessNoGo` 不变，不合并 main。
