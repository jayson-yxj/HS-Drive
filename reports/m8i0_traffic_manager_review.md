# 路线 A：CARLA 原生 Traffic Manager

2026-10-26，只读源码/安装文件检查。固定官方 CARLA 0.9.15 commit `d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18`。[generate_traffic.py](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/PythonAPI/examples/generate_traffic.py)、[TM 文档](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/Docs/adv_traffic_manager.md)、[同步合同](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/Docs/adv_synchrony_timestep.md)。CARLA 示例 MIT；未由此推导全部地图资产的再分发授权。

本机 `/root/gpufree-data/simulators/CARLA_0.9.15/PythonAPI/examples/generate_traffic.py` 与固定官方文件逐字节一致，SHA256 `2e9ec1a0a578a6269d93749bc8ba8ca496bbb8b94ae45d524abe9be507dc7d2b`。这证明安装示例来源一致，不证明服务器本轮运行或场景成功。

## 来源与生命周期

车辆使用 `get_spawn_points()`，按 Blueprint 筛选后 `SpawnActor.then(SetAutopilot)`，返回 Actor ID；TM 负责交通行为。行人通过随机 navigation 点、Walker Blueprint、附属 `controller.ai.walker` 和导航目的地产生；行人不受车辆 TM 直接控制。应保存每次实际导航返回和失败、Blueprint/属性、spawn ID、最终状态、controller ID、销毁时刻与地图版本。`seedw=0` 在示例的 `if args.seedw` 分支中不会设置服务器行人种子；未来不能用默认 0 声称已经固定行人随机性。

20–40m 与左右/后向目标只有交通流提供的机会，不是种子或车辆数保证的结果。道路侧停车辆、标志、建筑、树木等不能仅由本轮生成 ID 列表覆盖；须独立静态库存、语义/实例支持，未关联区域 Unknown。不要把没有自建目标的背景帧称为 Empty。

## Tick 与传感器

脚本检测 synchronous master 并在主循环 `world.tick()`；**两处行人初始化 `apply_batch_sync(batch, True)` 无条件要求推进**，不能与已有采集 Tick Owner 并排直接启动。唯一 owner 必须覆盖初始化、行人控制器建立、稳态、清理全过程；只读 Evaluator 不调用 tick/apply_batch_sync(...,True)/reload_world。所有状态在对应 frame 的 Snapshot 固定，动态 Pose/速度不混用晚到 RPC。

World 与一个 TM 同步、dt=0.05s；重新加载世界后重新设置随机种子，固定车辆顺序、Blueprint、天气、地图/XODR/包 SHA 与生成失败清单。官方“同条件确定性”不是所有平台、导航、渲染 bitwise 保证。禁止 hybrid physics、respawn teleport、忽略交通规则等选项作为自然场景证据。渲染必须开启；GPU 图像可能晚于 Tick 返回，按真实 `image.frame/timestamp` Join，不用回调到达顺序。

ActorSnapshot 可提供逐帧 Pose/速度，Actor 提供 local BoundingBox（含 rotation），计算 `T_world_actor @ T_actor_box`；尺寸为 `2*extent`。这是 CARLA 源 OBB，不是 Mesh 或精确接触。传感器各自的 K/T 和实际 image transform 必须闭合；Instance 16-bit ID 与完整 Actor ID 只在无碰撞并有投影/语义支持时关联。多 Actor 重用/销毁和 modulo ID 歧义均保留 Unknown。

现有 `simulator/synchronizer.py` 可复用帧号 Join、队列上限、超时和 tick gap 检查；原 collector 是否支持六摄四模态、动态 Actor 状态与输入隔离，仍需 Pilot 前静态接口测试，不能直接假设。官方示例不是数据导出器。未来须自有有限 Session 编排及 Sidecar，而非新放置算法。

## 独立状态

| 状态 | 实际含义 | 本轮 |
|---|---|---|
| TrafficGenerated | 本次真实运行生成交通 | NotRun |
| ActorIdentityAvailable | 本次 ID、来源及生命周期记录可用 | API 能力已审查；实物 NotRun |
| GeometryEvaluable | 对应图像可评价的源 Box/实例/时间闭合 | NotRun |
| NaturalSceneQualified | 声明范围内自然轨迹、区域、环境及身份审核成立 | NotRun |

清理必须只操作本次自有对象，先停传感器和行人 controller，再销毁 Actor，并恢复原 World/TM 设置；不能照搬会修改共享设置的示例 finally。若身份/几何证据不可取，结束并报告，不再扩大密度或导航搜索。
