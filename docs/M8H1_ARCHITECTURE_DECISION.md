# M8-H1 Architecture Decision

结果：**DevelopmentSmokeNoGo / TrainingReadinessNoGo**。执行基础是M8-H0固定00/02模板，资源SHA一致，唯一Shifted候选预检通过，实际Ego均能自然静止。失败发生在实际支撑证据门禁，目标Actor与传感器均未生成。

停止定制自动放置器的继续扩展；不自动启动M8-H2阈值修补。下一技术工作应独立比较CARLA已有交通场景、ScenarioRunner或成熟标注数据源。它们本轮没有部署或验证，不能预先宣布可用。

理由：自然落稳的车辆OBB底部略低于先前道路射线表面，原“从底部下方再向下查地面”的证据方法不能得到15个有效命中。虽然车辆速度和姿态稳定，当前证据不足以证明完整几何支撑。不能为了采到数据而上移射线、忽略Unknown或固定车辆到假设Z。这个失败表明当前验证合同与自然落稳的可观测性存在不匹配，**不证明原生车辆物理上悬空或场景本身不可行**。

原生来源仍有资产身份和复现价值；H0的地图/坐标/模板/历史索引、H1的Shifted合同和实际稳定轨迹可以作为下一来源评估的参考。当前RGB空间感知、训练、真实驾驶安全和独立Final门禁全部保持原状态。

首次Spawn立即读缓存导致的无效测量及其同点重测完整披露；没有将软件测量故障归因于车辆物理，也没有在真正支撑失败后继续重试。成功计数仅2个实际稳定Ego，不能当作2个目标车辆或自然训练对象。

最终独立状态：Ego Spawn成功；原始动态静止观察成功；Ego完整支撑Unknown/NoGo；目标车辆/Walker NotRun；实际遮挡与身份NotRun；相机位姿闭合及24路同步NotRun；四阶段反事实NotRun；DevelopmentSmokeNoGo；HistoricalCoverageIncomplete；FinalEligible=false。
