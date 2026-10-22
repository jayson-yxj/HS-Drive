# M5-O 架构决策记录

状态：**停止当前冻结Mobile+DirectCenter Head路线；M5-O总体Partial；不批准Offline Object-Center Pilot。**

实际完成38位置、1223个六摄配置，Train/Val/Calibration/Sealed支持分别642–658 / 198–202 / 199–205 / 269–279可靠候选。两个种子的固定预算结果均未稳定优于同候选Geometry Prior；封存Direct P95 12.33/10.66m，严重错误35.4%/32.8%。独立Calibration无合格阈值，全部准入拒绝。新增数据改善均值，但未解决尾部；不再自动增加层数、Loss或步数。

工程合同与有限训练管线通过，定位泛化与选择性风险未通过，纯视觉安全没有验证。未来可复用此Benchmark预声明研究其他成熟3D表示，但本轮没有验证替代方案，不能直接批准位置型Pilot。Oracle Planner/仿真工程可独立推进。

本轮只重建目标级定位研究基础。Mobile、ResNet Depth、48 LID、HeightSplat、Spatial Evidence V2 和 Oracle 控制保持原状，不引入新 Head、Loss 或视觉控制。

## 已确认的问题与处理

历史 M5-N 只有约64个可靠训练候选/种子。旧 Validation 同时选择 Checkpoint 和风险阈值，无法回答独立校准是否可靠。新 Benchmark 将 Train / Validation / Calibration / Sealed 分开；固定两种子、600步、三档独立位置支持，使用相同 RGB 候选与样本顺序比较 Direct 和 Geometry Prior。

旧测试大量未匹配区域不能统一标为物理 False Positive。逐场景六摄可视化和渲染语义审计将“记录 Actor 关联失败”“未知静态物体”“背景语义代理”分开。GT 实例候选只回答诊断问题，绝不作为正式 RGB 检测输入。Oracle 区域改变 Head 的输入形状分布，成绩不是简单的检测器能力上界。

定位误差统一对仿真 OBB Center；不再混称接触位置、可见表面中心或 Mesh 质心。Fixed Plane 条件接地点与 Dense 表面深度仍作为冻结对照，并说明语义差异；不加入 GT 尺寸补偿让它们成为作弊中心估计。

## 分阶段数据决策

第一版径向放置虽然覆盖六方向，但道路支持不足；停止复制这种数据，保留失败记录。采用预渲染道路几何放置，仍记录碰撞查询支持，不能假定地图点必然对应精确渲染/物理接触。

Town02 无法在历史隔离与90m组间距下容纳6组，未降低间距凑数。Calibration 追加 Town04 的独立 road_id 和距Train≥150m片区。新 Sealed 用 Town10HD；Encoder 的历史地理覆盖另外披露。所有布局共享道路位置选择模式家族，统计独立单位为位置/道路片区而非 Actor-view。

## 停止或继续规则

如果可靠独立数据支持不足，本轮可交付 Benchmark 和失败诊断，但不允许解释为“充分数据下 Head 失败”。训练停止，不放宽质量来凑数。

如果支持充分，固定预算的 Direct 在跨位置 Validation 和独立 Calibration/Sealed 上不能稳定超越 Geometry Prior/Fixed Plane，并同时满足≥30%覆盖、≥20匹配、≥3位置、P95≤5m、>5m比例≤5%，停止现有 Direct Center Head 路线的自动扩展。不会通过增加层数、调Loss或反复看Test寻找偶然通过。数据量曲线可区分增加场景的收益与尾部残留，但不足以证明某个全新架构优越。

只有两个种子方向一致、独立风险校准与全部候选负担通过，才允许提出限定 Offline Object-Center Pilot。即使通过，该许可也不涵盖 Contact Gold、Occupancy、Sparse Query正式训练、规划输入或纯视觉驾驶。

如需转向其他3D目标表示，先在本统一Benchmark上评估已成熟实现的候选/目标表示；本轮没有验证任何替代架构，不能以建议代替证据。

## 主项目解耦

感知研究输出仅 Shadow。Oracle 环境状态可继续支持独立 Planner/仿真工程验证；BEV 证据与目标候选有独立 schema、坐标、源时间戳、有效期及 Unknown 语义。现有 Oracle 驾驶成绩不转移为 RGB 闭环能力。M5-M Runtime Partial 继续保持。
