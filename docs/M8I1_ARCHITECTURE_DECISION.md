# M8-I1 架构决策：保留 TM 来源假设，停止本次失败 Session

状态：`PilotNoGo / SourceSupportNoGo / TrainingReadinessNoGo`。
基线 `502f0701025f2de5a8f2cc1ecbdcd4b3cacd2b6a`；执行源码 `2a5edce0a78fd0e8d08fd96dd6c27015a8dd5490`。

**本轮没有证据证明 Traffic Manager 已提供合格数据，也没有证据证明该来源不可行。唯一启动在 Town02 身份检查处被拒绝，交通机制尚未被测试。** 当前应优先解决可观测的启动配置问题，不能由此转向新放置器、增加交通搜索或训练模型。

已证实：服务器响应并通过 0.9.15 Client/Server 检查；地图名称不满足预声明合同；拒绝发生在 World 设置、Actor 和 Sensor 创建之前。自有服务器已退出，后续 OS 探针确认进程组和端口清理。合成 RGB 白名单／权限、帧同步合同与历史回归通过。

未证实：实际加载地图名称与启动失败根因；原生交通的落地、运动、图像同步、车辆／行人 20–40m 支持、多视角身份与 Box 闭合；真实采集输入隔离；有 Actor 时的异常清理和设置恢复。实际版本名称检查不是完整服务器二进制行为验证；采样 GPU 使用量不是峰值或实时性验收。

本轮任务书第三节限定“1个 CARLA 研究 Session”，第十一节要求“当前地图、服务器或必要配置身份不符”立即停止。因此保留此次失败，不加载另一地图、不换种子、不追加启动。下一项仅建议单独批准一次有界启动地图诊断，记录实际地图／Episode／frame／版本，再检查官方启动参数与就绪条件；当前没有批准第二次启动。

未来修复的必要条件包括：异常分支先保存身份再拒绝；Session 业务失败向 Supervisor 和进程退出码传播；原始异步启动推进与研究 Tick 分开统计；退出后可靠等待自有进程组消失。完成启动修复后，仍需实现并独立测试被动渲染身份评价器，随后按新批准预算验证单次实际数据来源，不能用本轮空结果替代。

M8-I0 的 TM 优先来源建议暂不推翻。ScenarioRunner 和 Bench2Drive 不自动启动；M8-F/G 放置器继续封存，旧 15 点支撑结果不被追认为通过。不修改默认 Encoder、Depth、BEV、Planner、Safety、Controller、Physics，不运行 PETR，不进行训练或视觉控制。

数据源功能、自然场景资格、独立数据支持、模型质量和驾驶安全继续分账。新增数据、Validation、Calibration、Final 均为 0，所有产物仅用于 Development 失败审计。已发布的 PETR、Occupancy、Runtime 与安全门禁保持原状态。

完整证据见 [总结](../reports/m8i1_summary.md) 与 [独立验收](../reports/m8i1_acceptance.json)。
