# M8-J3：单片段已下载并完成真实数据初验

检查日期：2026-10-26。基线 c4679352669e3a2b3add0ccb0fcc05282e9285d6。分支 feat/m8j3-bench2drive-single-clip-validation。**SingleClipLocalAuditGo / Partial / TrainingReadinessNoGo**。

用户在本轮明确授权个人、非商业的指定片段下载和本地审核，取代此前不下载的执行边界；许可证据、维护者询问信仍保留，但不再作为本次下载前置条件。没有对外联系，LicenseConflictUnresolved不变。J0/J1/J2原始证据和历史PayloadNotVerified不回写；本轮对唯一已下载片段建立新的实际验证状态。

## 下载与完整性

官方HF仓库 rethinklab/Bench2Drive，revision ce56cfbc26edf7796586e26fd44644f0ac2f6bcc。唯一文件 DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz。

- 通过已验证HTTP代理7897，官方huggingface_hub 0.26.5下载，TLS校验开启、无登录令牌。
- 首次HEAD302→官方存储GET200成功，22.86s；没有实际重试/续传，工具支持有界续传。
- 实测280176944 bytes，SHA256 b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2，与官方清单和J2冻结LFS一致。
- 完整gzip CRC及tar路径/类型检查通过，6237条目、6206普通文件，实际文件总字节329374497；无越界路径、链接、设备或重复项。
- 全部提取文件另建SHA清单并复核。原始归档、解压内容、三个RGB-only样本和可视化只存于 /root/gpufree-data/hs-drive-private/m8j3-bench2drive-single-clip。

详见[下载校验](m8j3_download_integrity.json)、[安全解压](m8j3_archive_audit.json)及[复现说明](../docs/M8J3_IMPLEMENTATION.md)。没有下载其他片段。

## 实际内容与接口验证

| 内容 | 实际观察 |
|---|---|
| 主序列 | anno与六摄图像均214组，ordinal00000…00213；一个高度相关路线片段 |
| 六摄RGB | 1284张1600×900 JPEG，全可解码；另有214张top-down，禁止作为六摄输入 |
| 深度 | 1284张900×1600单通道uint8 PNG，值1…255；不是原始24bit浮点深度 |
| Instance/Semantic | 各1284张；Instance四通道，Semantic单通道，语义值0…28 |
| 标注 | K、cam2ego、world2cam、Box center/extent/corners、Actor ID/类型、控制量、天气等 |
| 其他 | LiDAR LAZ、Radar H5、expert_assessment NPZ各214份；不进入纯视觉输入 |
| 原始时间 | 扫描全部214份标注未发现Frame/Timestamp/Episode/Version键；保留null，不从序号推造 |

5136张六摄四模态图像全部解码通过。K与各自FOV精确一致；cam2ego×world2cam相对world2ego的最大元素残差0.00010027，静态标定全序列无变化。Box center与八角点均值最大差8.53e-6m。上述是内部几何一致性，不是精确渲染Mesh或同步认证。

独立RGB-only加载器实际读取00000、00107、00213，输出[6,900,1600,3] RGB、[6,3,3] K与[6,4,4] optical→HS ego。包内后摄FOV110，其他70；使用包内K/T及明确映射，不套HS相机安装参数、不修改M8-B或默认网络。GT、世界位姿、Depth、Instance、top-down和有效mask不进入输入。

三个私有投影叠图已生成，人工查看首帧与中间帧：六摄方向和可见行人/车辆的投影位置具有基本一致性。图中的框是GT投影，未完成逐对象像素关联，不能作为检测准确率。图片与逐对象诊断不推送GitHub。

## 发现的问题及实际失败记录

1. **专家评估序号不一致。** expert_assessment为-0001…00212，图像/anno为00000…00213，首次全组对齐断言失败并保存原日志。官方固定源码以frame-1保存，与现象一致；尚未审核内容对应，不自动平移，不用作轨迹标签。
2. **时间信息缺口。** 文件序号匹配不证明原始逐传感器同步；无法认证速度/加速度/轨迹时间基准。包内speed等字段可以保留来源，但不得由ordinal伪造仿真时间。
3. **Depth精度不足。** 5.3772%的深度像素为255，源代码将米制浮点写PNG与量化/饱和解释相符，非确证每个255是255m表面。禁止直接复用HS raw-BGRA解码或当高精度Oracle Depth；0.0.4的新NPZ包并不在本次资产内。
4. **隐藏Actor。** 124条walker记录中有23条中心低于ego20m以上，首例约-100m；诊断发现后保留原文，没有阈值改写成最终标签。模型GT须有可见性/有效范围与隐藏状态审核。全片唯一非ego车辆ID10个、walker ID1个，不能把重复帧数当独立目标支持。
5. **实例关联未闭合。** 名义Instance语义通道与单独Semantic像素一致率87.3776%，原始数值ID不能直接等同Actor ID。该比例不是标注错误率或同步失败归因；仍需核实编码、渲染语义与背景身份，不能宣称可靠可见性。
6. **质量与分组。** 源码JPEG质量参数20，实际JPEG存在有损压缩；远场小目标须单独审查。只有一个场景组，不能按相邻帧切分train/val/test并声明泛化。静态设施、遮挡与未标注背景也不能当安全负样本。

[载荷Schema](m8j3_payload_schema.json)、[数值检查](m8j3_numeric_checks.json)、[验收](m8j3_acceptance.json)保存聚合证据。独立风险、正式Occupancy、时序、安全门禁未改变。

## 可进入的下一步

现在可以实施**限定的私有单帧适配与可见目标审核**：使用已验证RGB/K/T，独立sidecar转换Actor Box Center、尺寸和朝向，先为少量代表帧完成投影/可见性/实例关联并审核隐藏及静态对象；不依赖精确Depth，不输出Occupancy Free，不把GT输入模型。可在后续明确预算下进行冻结模型单帧研究，当前尚未执行，不批准扩大数据或训练。

正式训练还缺：可用目标mask及身份合同、类别映射和Yaw/尺寸验收、深度精度与用途限定、同路线样本相关性处理及独立位置组、训练/验证协议和显存预算。时间字段缺失时仅允许单帧研究，不能开启时序训练或驾驶闭环。

早先独立CARLA路线的版本/许可/接口评审已保留于[独立计划](../docs/M8J3_INDEPENDENT_CARLA_PLAN.md)：离线合同工程可推进，但历史启动与自然支撑问题未解决，没有新启动授权。

## 验证与发布边界

79项历史离线合同测试+20项新增测试通过（99项唯一测试；额外11项归档子集为重复运行，不累计），1项梯度测试刻意排除。实际下载/解压/图像读取为服务器CPU实验；安全拒绝和GT隔离是合成测试。GPU前向、训练、CARLA、视觉控制均0次。

本轮只提交审核源码、统计报告、官方许可证据和脱敏日志。完整论文与数据载荷在私有目录，图像/派生标注/权重不提交；不合并main。提交SHA可用 git log -1 --format=%H -- reports/m8j3_release_manifest.json 获取，文件SHA由该提交内的manifest固定。
