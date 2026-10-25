# M8-G — Grounded Placement Feasibility & Historical Footprint Preflight

结论：**Partial / PlacementFeasibilityNoGo / TrainingReadinessNoGo**。在 80 个预声明候选内，不能在两处 Development 同时得到车辆与行人。已停止本版混合放置器自动扩展；未执行 RGB Smoke，没有新 Actor、Final、PETR 前向或训练。默认控制、Physics 和模型保持原样。

基础 `8cde984ee606cc546c9d6c83962a7aaa95a08af8`；分支 `feat/m8g-placement-feasibility-diagnostic`。执行父提交与逐文件源码 SHA 在每份 JSON 和日志中，最终 Git 提交封存 release manifest。外部原始资产目录 `/root/gpufree-data/hs-drive-experiments/m8g-placement-feasibility`。

## 原 7 个拒绝的归因

严格按“整个请求已证实”的互斥口径：**确认物理非法 0；已证明可安全接受的保守误判 0；仍有未闭合条件 7**。这不是认定 7 个都合法，也不是无证据地认定全都真实碰撞。另有 1 个历史行人保持 GeometryProxySupported。

| 冻结请求 | 多重原因与证据 | 请求级结果 |
|---|---|---|
| 00 Vehicle | NONE 与限速触发体顶面吻合，历史 Depth 不支持可见实体，缺少 hit component 身份 | UnknownCollisionSource |
| 00 Person | 树冠命中明显高于人体；多个树木大 OBB 包含点，无法证明完整体积净空 | UnknownCollisionSource |
| 00 Cone | Driving 内部点被 Any 查询拒绝；植被大 OBB 交叠，无目标体内可见表面证据 | MapQueryAmbiguity + ConservativeRejectionUnresolved |
| 00 Barrier | 同上，静态 OBB 冲突发生于 Shifted | MapQueryAmbiguity + ConservativeRejectionUnresolved |
| 01 Vehicle | 交通灯触发体 NONE 与 Any 车道查询歧义并存 | UnknownCollisionSource + MapQueryAmbiguity |
| 01 Cone | 车道类型查询歧义，未改规则放行 | MapQueryAmbiguity |
| 01 Barrier | 同上 | MapQueryAmbiguity |

真实 CARLA 0.9.15 重放全部 8 请求 × 2 phase。16/16 的状态、全部拒绝原因、射线、所选地面、车道判定、静态竞争源及计划 OBB 与冻结记录一致。B 是诊断与历史预检包装，未引入接受覆盖；新接受 0。实际最终支撑继承历史参考，本轮没有伪造新 Actor 支撑结果。

## 几何归因的边界

21 个 NONE 命中均与 traffic Actor 包围盒顶面吻合（最大残差约 0.000001m）；同位置 `project_point` 相机通道全部命中 Road。结合官方源码，强烈支持触发体来源假设，但 `cast_ray` 不返回组件身份，所以不升级为物理碰撞/安全净空证明。原始严格分类仍为 17 UnidentifiedRayHit、4 ConflictingGeometry。后续证据单独保存在 `m8g_source_geometry_followup.json`，没有覆盖首次诊断。

30 个树冠命中：15 ConflictingGeometry、15 UnidentifiedRayHit；全部维持 C（不能判定完整空间安全）。高于人体的命中点不等于树木完整几何高于人体。两次静态冲突均指向约 12.75×12.39×20.21m 植被 OBB；历史六摄 Depth 中目标体积内部没有可见表面样本，仅说明缺乏正向碰撞证据，不能证明空间空闲。

240 个足迹点全部落在独立计算的声明车道 0.10m 内缩边界内；其中 130 个与原 `Any/project_to_road=False` 冲突，110 个一致。声明车道中心与 API 闭合误差小于 0.00003m。130 个点的只读 Driving 类型查询全部返回车道，支持“最近窄车道被选中后宽度检查失败”的机制。该算术与 CARLA 共用 OpenDRIVE，渲染地面也来自同一模拟器；不是独立精确 Mesh 真值。原验证器、inset 和 Unknown 拒绝策略均未改变。

[官方 Map 查询实现](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/LibCarla/source/carla/road/Map.cpp)、[RayTracer 通道实现](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/RayTracer.cpp)、[TrafficSign 包围盒来源](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/BoundingBoxCalculator.cpp) 支持上述机制解释；具体已运行二进制未做组件插桩，故因果归因仍有边界。

## 有界扫描与历史预检

在新位置分配前合并 13,463 + 3,099 = **16,562 条**记录，校验 **6,640 个**历史源文件。后者来自 6,190 原始帧；并非 6,190 个独立场景。5 个真实足迹回归均通过，包括 **3.635626m、4.595351m**。已恢复 OBB 使用精确 XY 多边形距离；仅有圆包络的记录输出保守下界，不冒充精确 OBB。历史地图版本仍有缺口，全部保持 HistoricalCoverageIncomplete、FinalEligible=false。

扫描在同两处 Development 道路内，4 类各 20 个，总 80 个，先记录完整足迹及历史关系，再选择组合；所有失败保留。车辆/行人使用类型对应车道，设施改为明确的临时车道中心施工场景，不再统一贴边；几何阈值不变。该语义调整不是旧候选同位置消融，不宣称单变量总体改善。

| 类别 | 两 phase 几何通过 / 候选 | Development00 | Development01 |
|---|---:|---:|---:|
| Vehicle | 3/20 | 0/10 | 3/10 |
| Person | 10/20 | 2/10 | 8/10 |
| Cone | 8/20 | 2/10 | 6/10 |
| Barrier | 7/20 | 2/10 | 5/10 |

28/80 仅通过射线/车道/OBB 代理，不是最终 Actor、身份、自然场景或训练合格率。剩余 52 个失败保留多原因：Covered 43、StaticOverlap 9、Footprint 5，原因有重叠。第一处车辆 8 个 Covered、2 个静态 OBB 冲突；没有在失败后增加随机搜索。

## 验收与下一步

真实 CARLA 的车身越界、建筑、无命中、水面反例继续拒绝；4 类正向几何复测通过；合计 10 项真实对照通过。合成测试单独验证高空点不等于净空、竞争源、错误高度、端点越界、弧线坐标和历史足迹下界。完整结果见 `m8g_test_results.json`，历史资产 SHA 重新验证。

Development Smoke：**NotRun_PreconditionFailed**。未满足“两处均有车辆+行人”，所以未开始唯一一次 RGB 采集。实际最终 OBB 支撑、渲染身份及 Empty/Present/Shifted/Removed 闭合均 NotRun，不借用历史行人充数。没有新的独立训练支持。

回答 A–G：A 为 0/0/7 的严格归因；B 有可复核的触发体与高空树冠证据，但完整物理归因未闭合；C 车道算术与渲染支持已闭合、Any 查询仍冲突，不能全体批准；D 只一处满足预检核心；E 已在全部候选选择前检查实际类别尺寸足迹；F 未通过且未执行；G **不恢复当前放置器的数据扩张**。

建议下一技术决策比较原生车辆 Spawn Point、行人导航点、人工审核场景模板，并分别验证最终几何和身份。不是自动启动 M8-H 阈值修补。训练与视觉控制不获批准。

图像：`m8g_figures/m8g_ray_body_3d.png`、`m8g_lane_boundary.png`、`m8g_scan_matrix.png`、`m8g_none_historical_rgb.png`、`m8g_history_footprints.png`。图像均由实测/历史证据生成，不是生成式示意。复现步骤见 `docs/M8G_IMPLEMENTATION.md`。
