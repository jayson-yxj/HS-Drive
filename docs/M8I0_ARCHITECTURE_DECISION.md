# M8-I0 架构决策：停止放置器扩张，先审查原生交通来源

状态：`SourceReviewCompleted / DevelopmentPilotProposalEligible`。
依据：`69ce3158ab692387e615a781deb41aae0fb61499`；检查2026-10-26；研发分支 `feat/m8i0-existing-scenario-data-review`。

**唯一优先路线：原生 CARLA0.9.15 Traffic Manager + Walker navigation/controller 的单次 Development 来源审核。唯一条件备用：Bench2Drive旧Mini单片段只读Schema审核。** ScenarioRunner保留三类源码评审，不部署为第二条并行工程。nuScenes只作原域参考，不加第三路线。

决策原因：本机TM示例与官方固定提交一致，能够保持原HS标定并直接记录真实帧与Actor生命周期，新增依赖最少。它并未验证自然放置/渲染身份；所需工作应集中于唯一Tick Owner和可拒绝Evaluator，不能重建几何候选搜索器。允许提出Pilot不等于已有合格数据源：目前数据层 `NoSourceReady`。

不优先ScenarioRunner：v0.9.15提供可控事件，但旧Python依赖、Actor隐藏替换、恒速/灯控、默认传感器帧匹配不足增加维护负担。成熟场景仍需独立几何/背景审查。普通scenario CLI也不能直接挂agent。

Bench2Drive对定制放置器依赖最少，字段丰富，但GitHub与HF许可声明不一致，本机未取得文件，静态Box匹配与源时间需审查。若许可/访问闭合，可单独申请一个官方280,176,944字节片段，先看一帧，不自动取全Mini/新版Depth。保留离线数据来源与未来闭环Benchmark的独立门禁。

旧M8-H1结论不变：两处Ego真实落稳，支撑证据失败；未证明悬空或生成不可能。15点向下查询起点可能已低于路面，说明方法适用性问题，不构成接触证明。禁止恢复M8-H2射线阈值修补。新来源必须采用事先声明的多证据审核，未知不放行、不当Free，不能悄悄降低旧标准。

下一项仅为 [有限Pilot提案](../reports/m8i0_next_pilot_proposal.md)，**须用户单独批准**。本任务书第九、十节明确“只负责设计下一项Pilot，不实际执行”“下一项Pilot必须单独批准”；因此本轮完成评审后停止，不能据此采集或训练。这是用户规定的阶段边界，并非额外审批流程。

不改默认模型、Planner、Safety、Controller、Physics；不执行CARLA场景、Actor生成、RGB/GT采集、PETR前向、训练或视觉控制；不合并main。既有PETR质量/时延NoGo、Occupancy/Runtime/视觉安全门禁保持开放。架构选择、场景自然真实性、数据泛化、训练资格、驾驶安全分别记账。

若获批Pilot仍不能在固定预算中取得可靠车辆/行人身份与几何支持，保留失败并结束，不自动扩大随机交通或转ScenarioRunner补位置。后续是否用条件备用必须再次决策。
