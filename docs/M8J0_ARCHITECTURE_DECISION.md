# M8-J0 Architecture Decision

日期：2026-10-26。状态：`LicenseAndAccessReviewCompleted / LicenseScopePartial / AccessOrLicenseNoGo`。

选择保留Bench2Drive旧Mini唯一片段作为**有条件的Development审核对象**，停止当前获取推进。固定清单确认文件名、大小、SHA及官方仓库路径；没有充分证据绑定当前可下载对象的修订和许可，也没有本服务器HF元数据可达证据。范围可控的外部数据研究仍比无证据继续CARLA启动修补更合理，但这不是本轮批准下载。

许可声明存在真实冲突，而非仅HF元数据标签差异。非商业内部复制/制作有条件支持；不能将此扩展到所有训练、衍生数据和公开权重用途。优先通过资产级维护者说明厘清，不选择对项目更宽松的一条声明当作事实。

未来若条件闭合，唯一下一候选是独立授权的“单包完整性 + 一帧Schema”审核，建立外部数据的单独合同；六摄源安装参数必须审查，原始时间不存在时保持null，GT严格隔离。禁止为适配旧包伪造时间或放宽HS Adapter。旧包不是新Final，单帧可读不等于训练可用。

本轮没有创建新模型、放置器、采集或启动方案。不恢复Traffic Manager，不启动CARLA，不训练/PETR前向，不动默认Planner/Safety/Controller/Physics，不合并main。原M8-I1b根因未明及已消耗授权不变。感知、Occupancy、Runtime、视觉安全均无门禁升级。

下一动作依赖：[许可用途矩阵与询问草稿](../reports/m8j0_license_scope_matrix.md)、[访问和身份缺口](../reports/m8j0_next_action_decision.md)、[未来资源上限](../reports/m8j0_single_clip_resource_plan.md)。询问未发出，下载未申请，任何之后的实际读取需独立任务授权。
