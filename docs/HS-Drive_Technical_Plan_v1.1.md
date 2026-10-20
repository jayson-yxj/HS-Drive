# HS-Drive 技术基线 v1.1 实施附录

日期：2026-10-19。本文为v1.0的增量记录，不重新设计V1.3网络架构。

- 架构、训练顺序、4090约束、纯视觉边界保持v1.0。
- 实装环境与依赖见 `server_env.md` 和根目录 `requirements-sim.lock.txt`。
- M0/M1在Town10HD_Opt静止场景实测通过；Town03 Vulkan崩溃未解决。
- M2点云/四层BEV/Box叠加已实现，独立平面实验确认本配置optical_z；完整跨场景门禁仍待补测。
- 同步增加预热和墙钟推进节奏；数据提交改为独立frames JSON，保证异常后的已提交帧可回放。
- 所有与原基线的实现细节差异见 `geometry_decisions.md`，证据及未完成项见 `../reports/milestone_01.md`。

未经上述门禁复核，不开始M4或将当前Oracle链路称为已完成的纯视觉自动驾驶。

第二轮更新：Epic / Town03完成M2工程门禁，状态及范围以 `../reports/milestone_02.md` 为准。网络架构保持不变；先进入M3，尚未启动M4训练。

第三轮更新：M3标准化索引、场景/路线组拆分、PyTorch加载器和监督字段隔离通过工程门禁。495帧小型工程集、未见地图Town10HD_Opt；详见 `../reports/milestone_03.md`。数据环境为CPU PyTorch，尚未验证CUDA训练或完成M4。

第四轮更新：独立CUDA12.8/PyTorch2.8.0环境、ResNet18+Depth+四层Lift-Splat BEV及小样本训练已实测，工程基线通过；网络未优于逐摄像素深度先验，完整M4保持开放。范围及失败记录见 `../reports/milestone_04_baseline.md`。

第五轮更新：新增六条通过运动/完整性验收的 TM 驾驶片段，最终索引 1461 帧/11场景；静态道路支持标签、Unknown 掩码和可选 Drivable head 已实现，CE/log-depth/joint 三组 1200 步实际训练完成。深度相较旧基线改善，但完整质量门禁保持开放。失败采集与排除记录、先验对照和冻结测试见 `../reports/m4_expansion.md`；语义与复现见 `M4_DRIVING_DEPTH_DRIVABLE.md`。

第六轮更新：远距离均衡深度损失与路口道路v2监督已完成三组1200步实测；增加路口条带、方向及歧义掩码，以可选overlay保持原索引和RGB输入兼容。配对实验按预设协议保留1×1道路头，上下文头未显示优势；远场深度改善伴随近场退化，跨地图Drivable尚未超过先验。新增Town01/04/02三场景、123帧受控信号诊断及停止义务监督，不声称交通规则网络或闭环已训练通过。完整M4质量门禁继续开放；结果见 `../reports/m4_far_rules.md`，字段合同与执行命令见 `M4_FAR_ROAD_RULES.md`。
