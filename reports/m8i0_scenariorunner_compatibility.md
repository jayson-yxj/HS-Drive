# 路线 B：ScenarioRunner 0.9.15

2026-10-26 官方源码审查；未 clone/install/import/run 框架。固定 `v0.9.15` → `d12d8bb788201ebf6f03e55b9c5169f9dcb2ea87`，MIT。[版本 README](https://github.com/carla-simulator/scenario_runner/blob/d12d8bb788201ebf6f03e55b9c5169f9dcb2ea87/README.md) 声明与 CARLA 0.9.15 兼容。不能跟随 master。

该 tag 的 `CARLA_VER` 仍写 0.9.13，Getting Started 仍提 Ubuntu16.04/Python2.7/3.5；这些是仓库内陈旧信息，不能照抄为安装方案。`requirements.txt` 固定 py-trees0.8.3、numpy1.18.4、networkx2.2、Shapely1.7.1、opencv4.2.0.32、xmlschema1.0.18，另有 antlr4-runtime4.10。与现代 HS Python/GPU 栈存在冲突可能；只可另建与 CARLA 0.9.15 wheel/egg 匹配的 Python3.7/3.8 研究环境，实际 resolver/运行尚未测试。不得在 HS 环境降级包。

## 三类源码检查

详细 XML 精确 Ego 坐标、Map 和数量见 [机器目录](m8i0_scenariorunner_scenario_catalog.json)，只审核以下三类。

| 类 | 官方行为与可用性 | 不能省略的风险 |
|---|---|---|
| FollowLeadingVehicle（11 个配置） | Patrol 在道路前方25m；WaypointFollower、路口前停车、距离/静止终止 | 场景加0.5m高度，provider再加0.2m；自然落稳未被成功标准检查；六摄导出无内建支持 |
| DynamicObjectCrossing（9 个配置） | 行人+遮挡物；时间到达/距离触发，横穿后销毁 | 最多6次位置尝试、行人+1.2m、先隐藏到地下、销毁替换和physics切换；遮挡物关physics；不能把触发成功当自然落地 |
| SignalizedJunctionLeftTurn（6 个配置） | 路口来车流，ActorFlow+TM、灯控、终点与超时 | 持续Actor创建销毁；ActorFlow还启用constant velocity及忽略灯/标志，碰撞时改变恒速；需逐帧记录干预 |

[场景目录](https://github.com/carla-simulator/scenario_runner/tree/d12d8bb788201ebf6f03e55b9c5169f9dcb2ea87/srunner/scenarios)、[原子行为](https://github.com/carla-simulator/scenario_runner/blob/d12d8bb788201ebf6f03e55b9c5169f9dcb2ea87/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py) 是上述判断依据。没有因为成熟框架而豁免物理/标注审查。

## 接口与同步

Python `BasicScenario` 包含初始化、行为树、criteria、cleanup。World Snapshot timestamp 驱动 ScenarioManager、GameTime 与 CarlaDataProvider；provider 中非 prop 生成会加 Z，并可能调用 tick。Manager、runner、provider 的多个 tick 调用必须属于同一个统一生命周期，不能再加 HS tick 客户端。

普通 `--scenario`/`--openscenario` 与 `--agent` 在 CLI 中明确互斥；route-based agent 另有集成路径。不能把现成场景加六摄 agent 当作无工作量接入。独立 Evaluator 可通过 actor registry、world/sensors 获取源 Box、K/T、图像和事件，但需要新增被动导出器。原 `srunner/autoagents/sensor_interface.py:get_data()` 等待收齐各 tag，却不强制所有 frame 相等，也不保存原图像 timestamp；HS 必须独立同帧核验。

事件时刻应记录行为树状态、Actor 创建/变换/physics enable/销毁，以及对应原 frame/timestamp。注册 ID 不能延用到替代 Walker；停用/隐藏 Actor 不能成为普通可见负样本。默认场景 criteria 是驾驶事件评价，未导出完整六摄 GT，不等同数据合格。

## OpenSCENARIO 与资源

[1.0 支持表](https://github.com/carla-simulator/scenario_runner/blob/d12d8bb788201ebf6f03e55b9c5169f9dcb2ea87/Docs/openscenario_support.md) 仅部分 Action/Condition；SceneGraphFile 和若干项不支持，Vehicle/Pedestrian 的 XML BoundingBox 等字段被忽略，因此只能读取实际 CARLA Box。0.9.x 有转换工具，不代表任意标准文件可运行。tag 还含 OSC2 parser/CLI，实验性质和语法/功能覆盖必须逐项检查，本轮不采用。

选择目录涉及 Town01–05 中的子集（每类见JSON），准确地图包与 XODR 仍须绑定；Town名字相同不证明资产一致。没有安装附加地图。TM 为 ActorFlow 所需，前车场景部分使用 WaypointFollower。失败恢复需停止树/agent/sensors、销毁本次对象、恢复 settings，不能批量清空共享世界。

结论：事件可控性更强，依赖隔离与生命周期审计成本高于 TM；`ScenarioExecutionCandidate_GTPartial`。现阶段不列优先或备用，不启动安装。
