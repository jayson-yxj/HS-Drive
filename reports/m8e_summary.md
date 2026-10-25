# M8-E — Perception Ground Truth & Data Support Foundation

结论：**DataContractGo / IdentitySupportPartial / AnnotationCompletenessPartial / FarFieldSupportInsufficient / TrainingReadinessNoGo**。本轮完成有限 CARLA 静态采集、身份与类别合同、历史候选重审及可复现审计；完整新评价基准仍未就绪。没有模型训练、新 PETR 前向或驾驶闭环。

基线提交为 037ddba29d6eaaa292e8ad74db58bd94e5f42b50，研发分支为 feat/m8e-perception-data-groundtruth。执行时的父提交及各文件 SHA 记录在每个运行 JSON；最终提交封存 release manifest。旧报告、数据和默认代码不变。

## 实际执行及七项结论

| 问题 | 实测结果 | 决策 |
|---|---|---|
| A 采集完整可复现？ | 8 个位置、32 个关键快照、192 张 RGB；768 个传感器载荷帧号、时间戳及 SHA 一致 | 数据传输合同通过；场景质量 Partial |
| B 身份与标注可信？ | 80 个独立受控 Present 目标中，28 单视角、18 多视角、34 Unknown；46 个接受目标全部通过实例编码出现/移除检查 | 有限渲染身份支持；静态背景和物理接触未闭合 |
| C 远场支持足够？ | 20–40m：21 个去重目标、8 个位置；目标为至少30个 | InsufficientSupport；不能用 Shifted 帧或重叠视角补数 |
| D 分母明确？ | 新数据单/多视角 28/18；历史155目标为6单视角、6多视角、80代理、63未知 | 明确；新数据身份计数不是模型 Recall |
| E Barrier 类别正确？ | 明确资产 streetbarrier 可映射；GuardRail/墙/栅栏不自动等同 Barrier；历史缺标 Barrier 保持 Unknown | 不再把未标注背景统称物理误检 |
| F 真正独立？ | Split 地图互不相同，但全部地图和布局家族历史复用；预留最终目标距旧 M5-O 目标仅9.9007m | **无合格的新独立最终组** |
| G 可以申请域适配？ | 远场、自然场景质量、静态标注及最终集独立性均不足 | **TrainingReadinessNoGo** |

实际分组：Development Town04 3组，Validation Town03 2组，Calibration Town05 2组，预留 Held-out Town01 1组。原先几何候选请求10组，在冻结的间距约束下只能分配8组，没有降低间距。每组 Empty/Present/Shifted/Removed 四帧。因支持不足，在32帧停止，没有用剩余上限挑选成功场景补齐。

这8组是不同位置，不意味着8个独立同分布样本。车型、相对布局和渲染器高度相关。Held-out 只看标注，从未运行模型；原 Split 字段保留，独立性例外报告将它限定为支持性资料，不能继续作为新最终集。

## 标签与场景质量

Actor、EnvironmentObject 和匿名 level bounding boxes 均实际查询并保存。对象记录包含完整 Box 变换、尺寸、坐标、原始旋转、Upright 近似残差、源帧/时间以及逐摄像头实例、语义、深度证据。实例编码从未当作 Python Actor ID 或 EnvironmentObject ID。

来源资格要求：唯一兼容 OBB、深度点包含比例至少0.9、0.25m几何容差、独立语义相机一致率至少0.95、至少3个保留像素。预测关联另使用冻结的投影/实例 Mask 条件，不用3D中心误差决定身份。单视角可以成立；多视角必须来自不同相机，不能把同相机多个组件算成多视角。

| 类别 | Present 总数 | 单视角 | 多视角 | Unknown | 20–40m接受目标 |
|---|---:|---:|---:|---:|---:|
| Vehicle | 32 | 10 | 3 | 19 | 8 |
| Person | 16 | 6 | 7 | 3 | 6 |
| Cone | 16 | 8 | 0 | 8 | 3 |
| Barrier | 16 | 4 | 8 | 4 | 4 |

远场接受视角：Front 5、Front Left 0、Front Right 3、Rear 8、Rear Left 4、Rear Right 2，共22视角对应21目标。超过40m没有计划样本，必须记为缺口。

**图像复核暴露了采集器的场景有效性问题。** Development_02 的远车/锥桶出现在水面或道路外，Calibration_00 的锥桶也呈悬空外观。当前圆周布置和射线落地 fallback 没有逐目标接触记录，不能保证实际路面支撑。21个只是“渲染身份支持”样本，不能被称为21个自然道路有效训练目标。本轮没有认证任何自然场景训练子集；原始失败样本全部保留。

独立语义相机与实例通道的全图语义不一致比例均值7.50%、P95 20.23%。这些数值包含背景像素，不能直接解释成目标标注错误率。原因未分离；若目标区域达不到0.95原阈值就保留 Unknown，没有为提高覆盖放宽规则。

OBB 包含渲染表面不是精确 Mesh 真值；静态环境存在多重 Box 竞争和未链接区域。46/46出现/移除检查支持本 episode 的受控对象身份，不能认证接触、可通行空间或安全。12张图均进行了 AssistantVisualReview，没有独立人工审核，不使用 HumanDoubleVerified。

## 冻结历史预测重审

历史 A/B 主匹配45/155、远场1/83、严格多视角身份6/5均保留。A仍为默认 Offline 顺序，官方训练 PKL 顺序未核实，Calibration仍NoGo。

对固定154个背景诊断候选：148个 UnlabeledOrUnknown、5个 DuplicateHypothesis、1个获得渲染支持的已记录 Actor。**0个被证实为物理误检**，这不表示没有误检。无法用新 episode 的环境 ID 补造旧帧标注。历史原始57,600候选及阈值后11,090候选负担保留；154条是诊断子集，不能代表全体Precision。

新身份口径下，历史155个目标有12个允许定位统计，143个不具备这一资格。12个的 XY 均值5.377m、P95 8.918m，>5m比例41.67%；6单视角 P95 8.434m，6多视角 P95 7.931m。远场仅1个，误差4.649m，不能推断远场总体质量。XYZ、光学深度、横向、径向、高度、L/W/H、Yaw、Upright IoU及分层分母见 JSON。

这是评价资格变化，**没有模型进步证据**。静态背景不完整，物理 Precision 与完整类别混淆矩阵仍不可得。新数据冻结 PETR 诊断为 NotRun_AnnotationSupportGateNotMet。

## 验证、隔离与资产

18个本轮合同/实证测试、16个M8-D回归、18个M8-B回归，共 **52项通过**。另外32个新RGB包通过原适配器校验，192图SHA一致；32次GT字段注入、3个评价文件访问探针被拒绝。独立进程使用网络 namespace 和 Python文件访问守卫；这属于误用防护，不是对恶意原生代码的安全沙箱。

全部 Depth/Instance/Semantic/Actor/Environment 数据位于 evaluator 侧。RGB包只含RGB、K/T及原始帧号时间。没有加载模型，也没有为了测试隔离打开新最终集模型成绩。CARLA以Epic离屏运行；采集结束恢复异步世界、销毁传感器/车辆/行人并停止本轮服务器。VehiclePhysicsControl未改变；静态试验用 set_simulate_physics(False) 固定 Actor，不能当作动力学实验。

原始数据、完整环境清单和审核记录位于 /root/gpufree-data/hs-drive-experiments/m8e-groundtruth；Git仅包含源码、协议、报告、日志和12张诊断图。所有新资产及历史复验通过 release manifest 索引；原失败运行及执行源码版本保留。

复现入口见 [实施文档](../docs/M8E_IMPLEMENTATION.md)，决策见 [架构决策](../docs/M8E_ARCHITECTURE_DECISION.md)。主要机器结果见 [验收](m8e_acceptance.json)、[远场支持](m8e_far_field_support.json)、[审核](m8e_annotation_review.json)、[独立性例外](m8e_heldout_independence_exception.json)。

下一步应先提出小范围场景生成与身份质量修复：逐目标记录落地点/支撑，避免水面和悬空，按历史目标覆盖范围而非仅Ego位置排除近重复，并在模型无关条件下预留新最终区域。当前不具备提出PETR域适配训练的前提。
