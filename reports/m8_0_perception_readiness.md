# M8-0 六摄感知就绪度与独立 P0 提案

**建议优先补齐一个成熟单帧 Camera-only 3D 框架的实际可行性证据。当前没有框架获批；PETR 仍为候选，部署/前向/质量均未通过。** 本轮基于 `b3dab19` 只做盘点及决策，不安装环境、不下载权重、不执行 GPU 前向或训练。

## 已有资产及有效支持

| 资产 | 已有证据 | 使用边界 |
|---|---|---|
| 六摄 RGB / K / T / 同步 | M7-A 40 packets、240 张图、38 位置组；本轮重新核验全部图像 SHA | 原始图像 640×384；固定相机顺序；不是新模型已适配 |
| M5-O 定位基准 | 38 位置组、1223 帧配置；Train20 / Val6 / Calibration6 / 原Sealed6 | 各集合已有历史使用/评测；原 Sealed 只作回归，不能作为新盲测 |
| 可靠候选支持 | 两种子 Train642/658、Val198/202、Cal199/205、原Sealed269/279 | Actor-view/重叠相机/同位置布局相关；不能以候选数替代独立场景数 |
| 3D 监督字段 | M6-A 审计 Train634+Val203帧、5022 Actor 配置；同步、K/T、盒中心闭合 | 参考为 CARLA Actor OBB，非 Mesh 质心、接触点或完整未知静态物体真值 |
| BEV 表征 | 原 HeightSplat、48 LID、四层 `[4,160,120]`、Spatial Evidence V2 保留 | Surface 与 Free-Ray 可共存；不认证物理 Occupancy 或安全 Free |
| Shadow 接口 | RGB/K/T 进入模型，GT sidecar 独立；原时间、有效期与来源可核查 | 视觉输出不能转换为 Oracle 控制输入；类型标记不替代文件访问隔离 |

盘点来源：[M5-O 总结](m5o_summary.md)、[数据支持](m5o_dataset_support.json)、[M6-A 字段审查](m6a_perception_framework_feasibility.json)、[六摄合同](m7a_six_camera_shadow_contract.json)、[访问审计后的前向](m7a_shadow_rgb_forward_audited.json)。

固定相机顺序为 `front, front_left, front_right, rear_left, rear_right, rear`。HS Ego 为前/左/上，光学相机为右/下/前，T 为 cam-to-ego。RGB 模型无需 GT Actor Transform；packet 中特权 Ego world pose 仅用于隔离的世界坐标诊断，不是正式视觉特征。原始帧号、timestamp、K/T 与图像增强变换必须共同保存。

本轮只读 `nvidia-smi`：RTX4090、24564MiB、驱动 580.126.09。这是设备存在性，不是 PETR 兼容性或显存通过证据。历史 Mobile 的两帧前向冷/热约 722.63/14.72ms、分配峰值 140.4MiB，只是接口 smoke；不能外推六摄检测框架、共享 CARLA 资源或时延分布。

本轮重新核验的代表模型：

| Checkpoint | SHA256 |
|---|---|
| `runs/m5i_learning/m5i_mobile_gt_101_step2400.pt` | `e2f0d0d85a03544186cd8c92095c95a3e2baf256fee0d9ab8f591e1c715d5c33` |
| `runs/m5o_training/m5o_direct_101_f100_step100.pt` | `9d407af4c0eefb31e85fd7b2a0d9d2bb82948d9a320447ce8df3b49145174ebb` |

全部历史模型和失败结果继续保留；未重跑这些模型。

## 为什么停止旧 Direct 路线，为什么不能立即扩大 Occupancy

M5-O 将独立训练位置从 5 增至 20 后，Direct Validation 平均 XY 从 4.95→3.61m、4.31→3.66m，说明数据覆盖确有作用；但同目标上两种子仍劣于 Geometry Prior。历史最终 Direct P95 为 12.33/10.66m，>5m 比例 35.4%/32.8%；Prior 为 8.69/8.28m、14.4%/15.0%。独立 Calibration 未找到合格准入阈值，Direct selected coverage 为 0。[同目标分析](m5o_paired_method_analysis.json)、[风险覆盖](m5o_risk_coverage.json)。

已知可见目标中候选/关联约造成 23.7%/20.9% 无法可靠定位，另外存在 3963/2975 个未匹配候选；后者可能包含未记录静态物体，不能全部当物理误检。相机/类别支持也不均衡：历史最终右前 7/7、右后 3/4 个可靠候选且各仅一位置；Calibration Cone17/13；40–80m 缺少泛化支持。当前数据能复用作开发基准，不能认证所有类别/六摄方向。[候选质量](m5o_candidate_quality.json)。

由此保留 `CandidateNoGo`，不再自动扩训 DirectCenterHead、加 Head/Loss 或建立同路线新阶段。成熟 3D 框架的识别/几何联合输出是否改善这些问题尚无本地证据；单帧基础不合格时增加 Temporal 或 Sparse 的复杂度也不能自动解决根因。正式 Occupancy、Temporal 与视觉控制继续独立开放。

## 唯一候选：原始 PETR 单帧 R50-DCNv2-C5

本次只提出该候选的有界 P0，复用 M6-A 已审查的官方源码；不同时移植 PETRv2、StreamPETR 或其他框架。它不是新的默认 Encoder/Depth/BEV，也不是对 HS 长期混合架构的替换决定。

官方代码固定 `f7525f93467a33707ef401c587a52d5e7b34de74`。2026-10-24 重新查阅官方 README、安装说明、许可证及配置：官方仍列出原始 PETR 模型与下载入口，但网页存在不证明服务器能取得完整权重。[官方仓库](https://github.com/megvii-research/PETR)、[固定版本安装说明](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/install.md)、[固定版本许可证](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/LICENSE)。

| 必需条件 | 当前实际证据 | 当前状态 |
|---|---|---|
| 官方代码、合法可用权重及 SHA | 代码 Apache-2.0；官方权重链接存在，但 M6-A 未取得模型字节/权重 SHA，也未核实单独权重适用条款 | **NotReady**；代码许可不能代替权重来源核查 |
| 独立环境可部署 | M6-A 独立 Python3.8.20、MMCV wheel 已取得；Torch 下载在资产失败后终止 | **Partial**；完整框架 import/CUDA ABI 未验证 |
| RTX4090 实际六摄前向 | 官方模型 forward、延迟、峰值显存均无记录 | **NotRun**；不使用随机权重前向冒充复现 |
| 官方数据合同 | 已做源码级输入/输出审查 | **ReviewedOnly**；未取得可运行官方样本结果 |
| CARLA 数据与坐标适配 | 837 帧字段/盒中心检查，最大 OBB 顶点闭合残差约 0.00013645m | **Partial**；不是 PETR adapter 或 yaw/尺寸转换通过 |
| GT 隔离、全候选/风险评测 | HS 已有合同与基准可复用 | **接口就绪**；新模型接线及访问审计待验证 |

M6-A 的事实是 Google 两次失败（网络不可达/超时），Baidu 官方列表可见但下载返回 errno=2，未获取 checkpoint；故退出 `NoGo_P0_AssetAccess`。不能据此断言 PETR 架构质量差、永远无法取得或 RTX4090 不兼容。本轮没有重试下载。[原失败证据](m6a_perception_framework_feasibility.json)，SHA256 `92cc2894dd860f1084d378418cd756207db5dfec17d330347651612058e1196f`。

原依赖为 Torch1.9 / MMCV-full1.4 / MMDetection2.24.1 / MMSeg0.20.2 / MMDetection3D0.17.1；官方 README 环境及 wheel CUDA 版本也不同。旧 CUDA wheel、DCNv2 和当前驱动/显卡能否共同运行须实测，不能仅凭可 import PyTorch 通过。不能直接升级到新版 MMDetection3D 后假定 box/yaw 合同不变。任何受控补丁必须单独记录 diff、构建环境及算子一致性，不修改 HS `.venv` 或 `.venv-gpu`。

所选官方配置是六摄、`[B,6,3,512,1408]`、BGR/caffe 归一化、R50-DCNv2、900 query/6 decoder、十类、最多 300 个输出盒。单帧不需要 temporal Ego Motion；相机到参考系投影仍必需。配置中的 64 个位置深度采样不是 HS 的 48 个 Depth Bin。[固定配置](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/configs/petr/petr_r50dcn_gridmask_c5.py)。

## 待单独批准的 P0 证据获取顺序与退出

**当前不具备“已可部署并前向”的申请材料。下一申请应明确是补齐这些证据的有限可行性任务，不是模型采用、训练或迁移申请。只有以下证据全部取得，才可提出后续适配/质量里程碑。**

建议总预算上限 5 人日，其中资产核查最多 1 人日；这是规划估计和止损建议，并非已授权工时。只接受可追溯官方来源，不绕过访问限制，不采用未知第三方 checkpoint。沿已知官方入口进行有限重试，若仍无字节与可确认来源则立即结束，保留 NoGo，不自动换第二框架。

1. **资产门禁。** 固定代码、配置、全部所需模型/初始化权重，确认适用来源/使用条款、大小、SHA、可加载 state_dict。不能把原 R50 backbone 初始化权重误作训练完成的 PETR 检测权重。缺少资产则不继续长时间安装。
2. **环境与官方合同门禁。** 新隔离目录/环境与锁文件；先 CUDA/DCNv2 等算子，再原始官方样本的合法输入和输出。最多一次有记录的兼容性补丁路线；预算耗尽停止。共享目录只读，原 GPU 环境及 CARLA 不改。
3. **4090 前向门禁。** 先官方 batch=1、六摄与原分辨率；保存全部模型加载日志、实际输出、冷/热延迟、CUDA 同步计时、allocated/reserved 与进程显存。建议 3 次预热、20 次测量，注明样本和环境，不用少量样本宣称可靠 P99。未满足原规格便报告 ResourceNoGo，不偷偷缩图/减少相机/切换模型后称原配置通过；任何降配属于新方案。
4. **坐标合同审查。** 官方合同通过后才做最小只读 CARLA 适配验证；若需写 adapter，必须在该后续里程碑授权范围内。原 RGB/K/T 经显式增强矩阵进入模型；GT sidecar 不进入 batch 的推理字段。完成下表的闭合及访问测试，才可申请冻结 checkpoint 的 CARLA 域诊断。

阶段退出可以是 AssetNoGo、DependencyNoGo、ResourceNoGo 或 ContractNoGo；不能把缺失证据标为 ArchitectureQualityFail，也不能因能 forward 就认定 PerceptionGo。不得自动开始微调、替换默认网络或 Direct 复活；后续有限适配仍需单独预算及批准。

## CARLA → 框架的必要合同检查

| 项目 | 必须验证的内容 |
|---|---|
| 图像与内参 | 原 640×384 与官方 1408×512 宽高比不同；明确 resize/crop/pad/letterbox，`K' = A K`，投影元数据同步。放大不创造远场目标像素 |
| 六摄/参考系 | 显式列出 HS 名称到框架相机顺序，不按字符串排序；`ego2img = K × inverse(cam_to_ego)` 的齐次扩展、投影深度符号逐点验证；名为 lidar2img 的矩阵可用固定参考系标定，不输入 LiDAR 数据 |
| 中心与尺寸 | Actor Box Center 参考为八顶点均值；区分 PETR 内部 gravity center 与导出 bottom-center，按固定版本的 box API 转换，核查 L/W/H 顺序 |
| Yaw 与反射 | CARLA 左手世界到 HS 右手 Ego，旧 MMDetection3D yaw/轴约定分别检查；非方盒 0/90/180°、六摄正前方向、盒角投影与逆变换 |
| 坡道与遮挡 | upright yaw 盒不能完整表达 roll/pitch；报告坡度误差/支持限制，不能改 GT 假装一致。可见性只用于监督/评价，未观测空间不作负例 |
| 类别与全候选 | Vehicle 到 car/truck/bus 等需明确映射，Person 与 pedestrian、Cone 与 traffic_cone；不支持类与静态未知候选单列，不静默消失 |
| 输入隔离 | RGB/K/T/增强形状为允许字段；Depth、Instance、Actor Center/Transform、可见 Mask、GT 有效区不能作为正式推理特征。全局 Ego pose 只在隔离的评测变换中使用 |
| 时空完整性 | 同帧六摄、原 timestamp/有效期、坐标版本和来源；模型输出仅 DetectedObjectHypothesis/OfflineShadow，不接控制 API |

不能用一个同名 box tensor 直接比较新旧方法。初期用解析盒/投影闭合及 CARLA 已有 OBB 做两层测试，未获独立 Mesh 的地方明确为代理参考。

## 后续质量评测提案，不在 M8-0 执行

冻结 checkpoint 的 CARLA 结果只是域偏移诊断；nuScenes 官方成绩不代表 CARLA 成绩。先报告所有输出，再按共同可评类别/距离/可见性与 M5-O Geometry Prior、Direct 及 Fixed Plane 做配对。新框架可能产生不同候选，必须同时有全目标自然检测分母和同目标定位分母，不能只取两者都成功的样本来证明整体提升。

必须包含：每帧候选数及未知负担、3D 一对一匹配 PR/Recall、漏检/重复/合并、Actor-view 与跨摄去重目标支持、XY Mean/P50/P90/P95/Max、>5m 比例、尺寸/Yaw 误差、按 Vehicle/Person/Cone/相机/距离/位置分层、拒绝后的 Coverage/尾部及校准风险。GT 可见/有效 Mask 只限制评价范围，不能用于模型输入或事后删掉误检。

Checkpoint 和检测阈值只在预声明 Validation 规则中选择；风险阈值用独立 Calibration 位置。旧 M5-O Sealed 全部作为历史回归，任何模型或规则选择完成后还需新的独立最终位置，提前冻结协议及模型。按位置组报告不确定性，保留相关布局披露。未来如有训练必须另批固定预算和两种子；P0 原模型 smoke 不要求通过训练质量门禁。

即使将来达到 M5-O 的离线研究门槛（Coverage≥30%、匹配≥20、位置≥3、XY P95≤5m、>5m≤5%，同目标优于冻结基线并有独立风险校准），也只允许讨论后续感知研究，不认证接触、Occupancy、交通规则或驾驶安全。

## Bench2Drive：仅保留接入计划

沿用 [M7-A 官方接口审查](m7a_bench2drive_feasibility.json)：版本 0.0.4，源码 `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`。官方仍说明 CARLA0.9.15 及附加地图；本轮网页查询不等于本机兼容验证。[官方说明](https://github.com/Thinklab-SJTU/Bench2Drive)、[固定源码](https://github.com/Thinklab-SJTU/Bench2Drive/tree/7ec25d1c9f7522d923ce5f3420986cef1cb2d956)。

| 接入顺序 | 计划及准入条件 | 现状 |
|---|---|---|
| 独立环境 | 锁定 evaluator/scenario_runner/依赖/SDK/服务端；验证完整 import。README0.9.15 与 vendored CARLA_VER0.9.13、最低版本检查不是等价兼容证明 | 历史只通过 4 个提取函数 smoke；完整环境 NotRun |
| 地图 | 单独安装官方匹配版本附加地图，核验 SHA 与 load；审计实际路线 XML、唯一 ID、所需 Town | 本机历史缺 Town11/12/13/15；v0.0.4 路线文件当时传输不全，完整 220 路线/地图清单未核定 |
| Agent | `sensors()` 和 `run_step(input_data,timestamp) → VehicleControl`；BGRA→RGB、准确帧关联、标定和来源审计 | 六摄符合 SENSORS max8/radius3m 的提取规则；qualifier max4 不支持六摄 |
| Tick Owner | evaluator 的 scenario_manager 唯一 tick/apply_control；不能同时启动 HS 自有 tick/control loop | 接口审查完成，真实集成 NotRun |
| 时间接口 | 历史 HS 10Hz RGB 与 evaluator20Hz 精确帧采集不一致，需明确频率/缺帧策略与验证；不能重用旧图冒充新帧 | 未实施适配 |
| 小范围验证 | 后续单独批准环境 smoke，再开发路线；源数据允许的 GNSS/IMU/导航另审，禁用 CARLA Actor/地图特权作视觉特征 | 本轮不启动 CARLA 或路线 |
| 独立官方评价 | 合格视觉 Agent 冻结后才申请；220 唯一预期 ID、失败保留、原评分及 infractions，不用自己选成功路线替代 | 官方 220 NotRun，官方分数为空 |

历史 dev10 涉及 Town03/04/05/11/12/13/15，但未经路线 ID 比对不能称 v0.0.4 的标准子集。原 Python3.7/3.8 与 numpy1.18.4 等依赖须独立处理，不能降级当前 HS 环境。附加地图、Town12/13 和 GPU 渲染与模型的共存显存也未经实测，单模型前向成功不能代替此项。

证据层级固定分开：**Oracle 离线规划 → RGB Offline Shadow → 实际车辆执行 → 官方视觉闭环评分**。它们不是自动升级链；每层需要自身的输入资格与评价。Oracle 可用于工程诊断，但其路线成绩永远不能作为官方 camera-only 分数。Bench2Drive 接入保持次级准备事项，不与视觉 P0、规划 B 同时全面开工。
