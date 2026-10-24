# M8-B — Frozen PETR CARLA Adapter & Zero-Shot Quality Pilot

**完成实际代码、192帧RTX4090冻结推理与离线评价；结论为 `AdapterContractGo / CARLAForwardGo / ZeroShotDomainGap / CandidateNoGo / PilotOnly`。未训练，未启动车辆闭环。**

基于 `a3a892219e29c3c16d969e8b0d934173a6f8dbb0`，独立分支 `feat/m8b-petr-carla-zero-shot`。官方PETR-R50-DCNv2-C5、900Query/6Decoder/10类、FP32、六摄512×1408及权重SHA全部固定。M8-A207份封存资产重新哈希无变化。执行时源码SHA与命令在 `m8b_logs/*.json`；本轮文件和外部数据SHA见 `m8b_release_manifest.json`。

| 问题 | 实测结论 |
|---|---|
| A Adapter正确吗 | 合同通过。192帧1152图同步、SHA、K/T、官方裁剪与box转换验证；输入只有RGB+K/T+原时间，GT仅Evaluator读取。坡道upright近似最大角点差0.316m，非精确mesh。 |
| B 真实六摄能完整前向吗 | 192/192成功，共57,600原始topK，所有输出有限，558张量前后逐位一致，训练0步。 |
| C 自然检测如何 | 固定score≥0.1、同粗类XY≤2m且|Z|≤2m：45/155 TP，110漏检，Recall29.03%；11,090候选中11,045对记录标注未匹配。0.406%为不完整标注下annotation Precision，不能当物理误检率。 |
| D 定位尾部是否改善 | 92个独立投影代理关联：mean7.39m、median6.22m、P95 18.57m、>5m56.52%。与旧基线同目标对比未见稳定优势；自然2m子集P95=1.85m已被门限截断，不能代表真实尾部。 |
| E 风险覆盖有效吗 | 否。Calibration24目标、raw关联最多18；固定score网格无合格阈值。Score不等于安全概率。 |
| F 资源与时延 | 实际峰值allocated1.03GiB/reserved1.25GiB（非整个设备上下文）。GPU Forward P50/P95=52.96/53.51ms；E2E=330.44/337.94ms，稳态Max441.62ms，首帧718.69ms。离线可行，5Hz/严格20Hz未通过。 |
| G 是否值得后续训练 | 仅值得提出独立、有界的参考分布与数据支持诊断。当前不批准直接扩训；新盲测、右侧/远场/背景完整标注和几何分布因果仍缺失。 |

全40份M7-A先完成，其38空记录目标帧导致支持不足；推理前冻结各38位置组分位采样，最终Train102、Val30、Calibration30、旧Sealed30。155目标实例观测对应119个组内Actor身份、265相关视角，不能当155独立场景。Vehicle105、Person30、Cone20；右前12、右后4，>=40m无支持；仅1帧多Actor。历史Sealed只是回归，没有新的盲测，所以保持PilotOnly。

车辆/行人/Cone自然Recall分别27.62%/30.0%/35.0%。距离0–10m为23/25、10–20m21/47、20–40m1/83；不同层的类和布局混杂，不能宣称严格距离因果。48/265视角部分裁剪，支持像素损失14.26%，0目标完全裁失。冻结官方裁剪没有被临时修改。

可视化：

- [质量、风险与时延](m8b_figures/quality_risk_latency.png)
- [高分错位例六摄与裁剪](m8b_figures/s107_six_camera_crop.png)
- [同例BEV空间误差](m8b_figures/s107_bev.png)
- [类别召回](m8b_figures/class_recall.png)

验证：18项新增合同/几何/匹配测试通过，42项历史资产/Shadow/Benchmark接口回归通过；失败日志保留（评价环境缺包、测试发现路径、序列化list算术、报告字段重名），修复未改模型或评价协议。实际GPU环境未重新安装，原默认模型、Planner/Safety/Controller/Physics均未修改。Python文件审计不是恶意native代码文件系统沙箱，约束与可复现方法见 [实施文档](../docs/M8B_IMPLEMENTATION.md)。

完整解释见 [域偏移分析](m8b_domain_gap_analysis.md)，独立状态见 [验收JSON](m8b_acceptance.json)，后续选择见 [架构决策](../docs/M8B_ARCHITECTURE_DECISION.md)。无输出为Unknown，绝不导出Free、Oracle Actor或控制输入。M5/M7与共享GPU运行时历史未通过门禁不变。
