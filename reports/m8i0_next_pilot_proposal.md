# 下一项提案：原生 TM 单次 Development 来源审核

**ProposalOnly / AwaitingSeparateApproval。本文不是执行授权，本轮不实现或运行。** 唯一来源为当前 CARLA0.9.15 官方 Traffic Manager + 官方 Walker navigation/controller；观测器读取源状态，绝不以 HS 视觉结果控制、筛选场景或定义真值。运行交通本身会改变世界，因此“只读”专指 Evaluator，不虚称整个仿真会话只读。

## 固定范围

- 一个既有 Town02、一个固定天气、一个Session、一次最多30s观测（另给不超过5s初始化），总World推进≤700 Tick、dt0.05s。Town02已观察，严格Development。开跑前冻结实际地图包/XODR SHA、官方示例SHA、HS rig、Blueprint/Physics配置和seed。
- 最多8辆车（含一辆带观测Rig的TM vehicle）、最多12行人；各Actor最多一次原生候选尝试，无重试找位置。导航spawn最多12次、目的地最多12次；返回None完整记录，不补采。示例种子选非零固定值，保存实际所有Transform而非仅seed。不开hybrid/respawn，不改物理、不ignore规则。
- Ego使用官方生成集合中预声明第一辆成功vehicle，不择优看图。稳定阶段先暂停其autopilot命令并保留原Physics，记录自然下落，随后交由TM；失败停止，不能改用另一Ego。其他车辆启动/接管阶段同样记录，TM行为与HS驾驶成绩分开。
- 六摄原HS安装；RGB/Depth/Instance/Semantic的K/T配对一致。原生960×576、render_every_tick；24路同帧Join但仅保留预先确定的12个时刻（不按可见目标或模型输出挑帧），其余逐Tick保存Actor/事件摘要与队列诊断。磁盘≤1GiB、墙钟≤30min，越界即有界停止。实际GPU/CPU/磁盘统计保存，不承诺实时。
- 不安排Empty/Present/Shifted/Removed，不生成Cone/Barrier，不扩地图或位置；原有背景对象必须独立列入未知/可评价状态。无PETR前向、训练、视觉控制及官方220路线。

## 唯一Tick与输入隔离

Session owner负责所有初始化、batch、传感器接入、World/TM同步与退出。不能同时启动原版 `generate_traffic.py` 和既有HS recorder；其中walker初始化的tick请求也必须由owner执行。Evaluator被动接收不可变Snapshot缓存与image frame；不推进、reload、移动Actor、修Physics或重设状态。

正式RGB包白名单：RGB、每摄K/T（含确定性图像变换）、frame_id、timestamp、schema版本/来源。Evaluator Sidecar是独立路径，存GT/Actor/Depth/Instance/Semantic/环境库存。文件权限/loader依赖测试确认不能从Sidecar补全模型输入；本提案不运行模型。源frame/image timestamp缺失、跨帧、重复、超时均拒绝，而非复用上一帧。

## 新的几何证据合同（待单独批准，不改旧验证器）

自然状态、源Box正确性、可见渲染身份、精确接触是四项不同命题。旧15点射线不再是唯一接触判据，也不会将其Unknown改成Passed。

| 命题 | 下一项需要的独立证据 | 失败与限制 |
|---|---|---|
| 物理状态可观察 | 初始到实际落稳/运动原Pose/速度/姿态、原Physics配置与所有干预；有效Snapshot后测量 | 不能以关physics或固定Z静止证明；Walker脚部/骨骼与车辆轮胎有不同合同 |
| 合法区域/地面来源 | 原生spawn/nav身份、地图版本、完整OBB角点与道路/人行道区域；语义/深度道路观测、静态层及环境竞争 | OBB不是轮胎/人体接触面；路面代理与坡道多层不一致必须Unknown；不自动生成新射线规则 |
| 源几何可评价 | T_world_actor×T_actor_box、八角点、2×extent、跨摄正深度投影、可见实例对应、无遮挡表面Depth | Box内部深度非精确Mesh；背景同类冲突、截断或低像素支持分层，不把未观测当负例 |
| 自然场景审核 | 上述原轨迹、合法区域、无证实穿透/悬空、实例/背景证据和人工复核共同支持 | 仅声明审核范围内的`NaturalSceneQualified_ReviewScope`；若证据缺失，保持Partial |
| 精确物理接触 | 独立可用接触/轮胎碰撞证据 | 现有公开源未提供完整证明，`NotAvailable`，不输出PhysicalContactCertified |

几何代理发现真实异常要拒绝；仅缺支持证据则保留Unknown与原因。不得删掉建筑/水面/多层/遮挡等历史反例。新合同须在Pilot启动前固定，并用历史失败回放区分“方法不适用”和“物理不合法”；不会重新标定H1或增加接受容差。

## 有限成功条件与停止规则

工程成功：12个预定时刻完整24路、同frame/timestamp与Actor缓存匹配；无GT泄漏；全部生成/失败/销毁、所有异常和cleanup可审计。任何关键同步缺失则工程Partial，不通过后处理伪造时序。

来源最低支持：至少一个实际Vehicle和一个实际Walker有可靠可见实例、独立来源Box与当前帧闭合；至少一个Vehicle和一个Walker在20–40m范围得到可评价支持。逐摄、距离、类别、遮挡、静态/动态分别计数，单Actor多帧/多摄不算独立样本。数量只证明管线可检查，不证明训练泛化。

Actor/地面/背景无法闭合时记录来源状态和不可评价比例；不扩大对象数、延长时长、换seed或生成另一个模板。缺少目标也不能把NotRun/未观察算作检测失败。若有限运行仍不足，提交 `TrafficGenerated / IdentityPartial / GeometryPartial / NaturalSceneNotQualified` 等实际状态后停止，另行决策，不能自动转备用下载。

历史独立性：复用M8-G16562条及完整来源清单检查Ego/完整目标足迹、道路区域、模板与视野包络；缺口仍明确。M8-F/G/H0/H1、M5/M8旧Sealed与所有已观察样本均Development/Regression，新Final=0。单次流量录制不是12个独立位置组。

输出拟包含：Session配置及源码SHA、真实frame日志、动态/静态来源清单、同帧RGB与隔离Sidecar、逐对象几何/身份/未知表、20–40m支持、失败截图与清理结果。不以零碰撞或场景跑完替代以上验收。Pilot成功只允许下一次数据源扩展决策，不能自动获得训练或视觉安全资格。
