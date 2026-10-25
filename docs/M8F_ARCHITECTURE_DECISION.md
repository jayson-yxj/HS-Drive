# M8-F 架构决策

**保留拒绝型质量合同；暂停数据扩张与训练申请。** 实测两个Development位置、8帧没有满足冻结的Smoke门禁。下一阶段应限定为放置器可行性诊断，不应继续盲目采集或调整模型。

采用独立证据等级：渲染身份、支撑表面、自然放置、标注来源、Split独立性和输入完整性。所有条件必须同时满足才能获得EligibleForProposalOnly，仍不授权训练。现有目标独立训练资格为0。

新生成器使用车道/人行道候选；15点足迹记录全部射线命中、合法类别区域及静态Box竞争。请求失败在spawn之前拒绝，实际Actor姿态在保存帧再次核验。不存在固定Z回退，继续保留未知与拒绝原因。

限制仍然明确：完整足迹的有限采样不是连续支撑证明；环境OBB不是Mesh；关闭物理后静止不是接触证据；高树冠触发保守拒绝不代表身体碰撞；NoLaneAtFootprint不等于已证实真实路面越界。多个同时接受目标的相互作用尚未实测，不批准多Actor自然训练采集。

历史隔离使用足迹与场景包络，保持缺失来源显式不完整。后补原始M5索引不能追认采集前独立性；未来必须先加载完整补充索引。两个已看Development和旧Held-out均不得转为新Final。全部历史Town、资产和模板复用，不认证跨地图泛化。

静态对象仅在实际源证据支持的像素区域具有有限身份评价资格。其余区域保持Unknown，不能计算全场物理Precision或安全Free。匿名几何不强配InstanceID。模型输入合同维持RGB、K/T、Frame/Timestamp。

决策状态：GroundPlacement有限实现通过；NaturalScene SmokeNoGo；Identity/Background Partial；FarField不足；HeldoutIndependenceNoGo；TrainingReadinessNoGo。Perception、Occupancy、Runtime和Safety不变。新PETR前向、训练、闭环、默认模型/控制/Physics修改均未执行。
