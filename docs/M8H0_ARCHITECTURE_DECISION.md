# M8-H0 Architecture Decision

选择 **原生候选来源 + 有限审核固定模板（C）**，作为下一项Development验证的候选来源。可提出 `DevelopmentSmokeProposalEligible`；本轮不批准、不执行真实Actor Smoke。

来源审核完成：101个原生车辆点中42个得到几何代理支持，60次导航返回中28个得到支持。已冻结四组20–40m模板，保留完整尺寸、姿态、来源ID、地图版本、目标间足迹、历史关系及静态未知；原生点的初始Z没有被冒充为最终落地姿态。

完整几何状态仍为 `TemplateGeometryPartial`：四组都存在至少一条视线的路灯/站亭保守OBB警告，没有真实Actor落稳或渲染身份。`CuratedTemplateCandidateGo`仅表示可审核候选成立，不能用于自然训练资格。M8-G继续保持原失败状态。

不选A继续扩张：M8-G固定预算无法完成两处核心要求，NONE触发体/树冠/Any地图查询都有未闭合问题，本轮没有修改这些规则。不选B直接自动采集：原生来源未保证姿态/碰撞/观察/身份，行人导航位置尤其不能直接作为Actor中心。C让这些条件成为有限、可复核的模板义务，而不是自动接受或持续搜索。

实际证据不支持宣称“原生来源已更稳定”：范围、候选分布及phase不同，旧Development00附近仍0/5；只支持当前Town02存在可用候选池。单地图、单车型/单行人、重复地图道路和历史近邻限制外推。模板01几乎复用旧Development01区域，明确不是新盲测。

建议下一份任务书只申请模板00和02的两处Smoke。二者Ego距离123.24m，行人方向没有当前OBB中心视线警告；这只是调试优先级，不形成独立分布。两组车辆都有路灯OBB警告，真实可见性和目标身份仍是必须通过的门禁。Shifted位置须预先限定、身份明确；本轮没有批准自动移位采集。

未来若真实生成/落稳/身份不能通过，不自动重采更多点，不再补地面射线阈值。转为技术评估CARLA已有交通场景、ScenarioRunner或现成标注数据；该替代路线本轮未部署、未验证。

独立状态：来源 `NativeCandidateSourceReviewed`；模板候选 `CuratedTemplateCandidateGo`；完整几何 `TemplateGeometryPartial`；历史 `HistoricalCoverageIncomplete`；真实Smoke/渲染身份/接触 `NotRun`；训练及视觉控制 `NotAuthorized`。不合并main。
