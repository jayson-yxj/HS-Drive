# M8-J12：SparseDriveV2 Bench2Drive 最小链路

2026-11-01 UTC 实测。已完成 **真实六摄 → 原生规划 → HS 三秒轨迹 → 原配 PID 离线干跑**。这是接口接入通过，不是驾驶质量、实时性或安全认证。PETR-A 仍为参考，默认系统没有改动。

| 独立状态 | 结果与证据 |
|---|---|
| AssetAndEnvironment | 冻结推理范围通过；严格加载缺失/多余键均为 0，实际 FP32 |
| InputBoundary | 通过；六摄 RGB、实测时间、Oracle Ego State 和固定导航，不输入环境 GT |
| PlanningReplay | 通过；64 个独立观测各两次，共 128 次成功前向，预算 160 |
| TrajectoryAndControlAdapter | 通过；保留原生 6×0.5s 轨迹，控制只返回数值 |
| ClosedLoopDriving / Training | NotRun / NotRun |
| ModelControlApplied | 0 |

## 固定资源与实际差异

使用 [官方 bench2drive 源码](https://github.com/swc-17/SparseDriveV2/tree/e42ea59dd4946238dc65097495a9aa0708121fae)，commit `e42ea59dd4946238dc65097495a9aa0708121fae`，配置 `projects/configs/sparsedrive_stage2.py`。官方源码 LICENSE 为 MIT；来源和 SHA 保留，不引申为所有第三方资产均可任意再分发。

权重来自官方项目链接的 [HF 固定 revision](https://huggingface.co/wenchaosun/SparseDriveV2/tree/42c15531aa6cb18c7360dd0377cfc1c221280597)：`sparsedrive_small_b2d_stage2.pth`，887,103,732 bytes，SHA256 `f15c889b759734894e6ac906403df55a9973d7e60e3101da3ac71e9091f7b130`，与该 revision 的 LFS 清单一致。只取得一个完整 Stage2 checkpoint、六个必要模板，以及任务指定的另一轨迹模板用于内容核对。未获取 Stage1、聚类数据、训练 infos 或整套 HF 仓库。

`trajectory_1024_256.npz` 实际为 `[1024,256,8,3]`；`trajectory_1024_256_b2d.npz` 为 `[1024,256,6,2]`。不能改名互换。固定源码使用后者，其轨迹及 mask 与 checkpoint 张量逐值一致；checkpoint 内嵌的旧文件名不能证明当前无后缀文件兼容。其余结构一致，内嵌的 `rescore=True/rescore_thresh=0.5` 等于源码默认值。详情见 [配置解释](m8j12_sources/config_resolution.json) 与 [完整性](m8j12_sources/checkpoint_integrity.json)。

隔离环境：Python 3.8.20、Torch 2.0.1+cu118、MMCV-full 1.7.1、MMDetection 2.28.2，RTX 4090。从固定源码编译原 CUDA 扩展，没有替换注意力/聚合实现；运行副本 Python 源码与官方一致。94,981,823 个参数全部 FP32。没有执行 `init_weights`、随机骨干替代或忽略关键权重。该环境仅验证冻结推理；`pip check` 所列 Jupyter、TensorBoard 服务依赖及 Triton 未安装，不能宣称完整训练环境通过。

## 真实采集及失败记录

最终采集使用既有 CARLA 0.9.15、Town03、ClearNoon、Lincoln MKZ 2020，以及既有 PurePursuitPID 采集控制器。复用历史固定路线；S0 从直路第 10 个路径点、S1 从缓弯第 46 个路径点开始。固定 20 tick 静置、30 tick 行驶预热，然后取 32 个连续测量帧；不依据模型输出选择数据。

| 片段 | 实测帧 | 窗口跨度 | 总模拟时间 | 实际前向速度 | 窗口航向变化 |
|---|---|---:|---:|---:|---:|
| S0 直路 | 96–127，32 帧 | 3.1s | 8.3s | 1.5420–1.5422m/s | −0.00020rad |
| S1 缓弯 | 204–235，32 帧 | 3.1s | 8.3s | 1.5754–1.5784m/s | 0.04279rad |

先后有三次失败启动/采集尝试：解释器 `resolve()` 跟随软链接而丢失虚拟环境；原 RoutePlanner 调试绘图缺 Pillow；新 Actor 尚未进入上一帧 WorldSnapshot。均在取得有效输入序列前失败，日志未覆盖；修复、预检与回归后第四次启动完成两段采集。不是地图加载或模型失败。四次均完成自有进程组和端口清理；最终 2300–2302 无监听。失败路径和源码提交见 `m8j12_sources/capture_*`。

物理步长/六摄采集/模型观测间隔分别为 0.1/0.1/0.1s；IMU 配置 sensor_tick=0.05s，在该仿真步长下使用同帧实际回调。轨迹间隔另为 0.5s。六摄采集 704×396，官方 crop 后 704×384，JPEG quality=20 与官方归一化。没有套用 HS/PETR 旧标定。全部测量外参相对官方矩阵最大绝对差 `9.46495e-6`。

原 Agent 使用 LiDAR Actor **位姿**构建坐标变换，不读点云特征；BEV 图仅保存/展示。离线入口关闭两种订阅，以当前自车位姿和固定虚拟外参重建同一变换。使用 IMU、前向有符号速度与已知导航，不读取其他 Actor、道路 GT、交通真值或未来行驶轨迹。`gt_ego_fut_cmd` 是官方命名的导航 one-hot，不是未来 GT。自车状态明确是 **Oracle Ego State**，不是视觉里程计。

## 回放、轨迹与控制

执行源码 `6e6eee7732b56c15f5b2ee44940968712a93fb50`，run `m8j12-integration-20261101/replay_first`。CARLA 关闭后在禁网隔离进程中运行。128/128 成功，0 次失败整模型前向，未使用剩余 32 次预算。没有预热整模型调用或训练梯度。

S0/S1、两次重放之间重置检测/map instance bank、motion/planning queue，并新建原配 PID；导航在采集阶段按序生成，回放读取同一冻结导航流。历史时间严格早于当前测量，不跨序列。64 组轨迹、路径、速度及最终控制重复差均为 0，预声明容差仍为 atol/rtol=1e-5。各段前后完整 state_dict 哈希一致，运行内参数版本未变。

输出保留原生预测检测、地图、运动、15 个等空间间隔路径点、6 个速度窗及最终 6 点轨迹。原解码器在 1024 个过滤后组合候选中进行原生重评分/选择；只读 profiler 保存实际候选索引、评分与候选集，没有重写选择算法。最终结果逐点与选中原生候选一致。

原生规划是当前自车原点、x 向右/y 向前，HS 映射为 x 向前/y 向左。第一个预测点为 t+0.5s，最后 t+3.0s；不追加尾部或安全分数。速度是原生半秒窗口平均速度，HS yaw 是相邻轨迹段派生值，均有来源标记。

原配 PID 与 Agent 后处理直接复用。64 个独立观测中，S0 满油门 27/32、满制动 5/32；S1 为 28/32、4/32。相邻油门/制动最大变化均为 1.0。原生速度模板输出范围约 0.334–8.254m/s，而采集车速约 1.5–1.6m/s。异常和饱和未平滑，原 PID 内置限幅及 Agent 原限幅明确保留；这不证明控制舒适、可执行或安全。

## 资源和可视化

峰值 allocated **1,000,976,384 bytes（0.932GiB）**，reserved **1,281,359,872 bytes（1.193GiB）**，不含此前 CARLA 渲染。模型含解码后续帧 P50/P95/Max 为 **87.61/98.92/112.44ms**；进程首帧 **1164.04ms**。解码计时有 CUDA 同步；非严格 GPU kernel-only 时延。

读取、JPEG、预处理、模型及控制串行耗时 P50：S0 **235.68ms**，S1 **234.17ms**，尚不含缓存写入/审计。不能用模型约 99ms P95 宣称端到端实时，亦未验证共享渲染或其他模型负载。

六组最终私有图位于 `/root/gpufree-data/hs-drive-private/m8j12-integration-20261101/visualizations/`，每段固定索引 0/16/31。六摄图及 BEV 均标记 **Planning suggestion on recorded data — NOT EXECUTED BY VEHICLE**；相机投影假设当前自车 z=0，仅供显示，不认证真实地面。助手查看了代表性生成图，无人工标签或驾驶质量签核。原图、观测、逐帧输出/记忆、候选集、权重均未入 Git。

## 验收与收束

相关 16 项合成/合同测试在 HS 和新隔离环境各通过一次，0 fail/skip；另有真实 GPU 算子检查、128 次真实回放、64 组重复比较和全帧外参核对。全部执行命令、退出码与 SHA 见 [验收](m8j12_acceptance.json)、`m8j12_sources/*_execution.json`；[复现说明](../docs/M8J12_IMPLEMENTATION.md)。没有重新运行历史训练或全面驾驶测试。

本轮接入可收束。下一步建议仅做基于现有回放的规划/速度与控制饱和归因，先核对可执行性，再单独决定是否申请有限闭环；本轮不自动执行。

代码具备合并条件：仅新增 opt-in 工具，默认行为未改，必要回归通过。2026-11-01 接手续核确认远程仅有 `main`，指向基线 `eb69114636ab60cd48fdba3708067b751fd4038c`；本功能分支尚未推送。申请一次确认后，将 `feat/sparsedrivev2-b2d-integration` 最终报告提交快进合入 main，正常推送并核验远程包含，再仅删除本地该功能分支；当前没有同名远程分支可删除。若执行前远程状态变化，先报告差异。最终完整 SHA 在交付消息给出；不新增 tag、不改参考权重、不重跑实验。

## 会话交接复核

原会话中断时上述实验已结束，遗留的是 19 个尚未提交的报告、证据和缓存审计工具文件。接手仅复核现存文件及日志：68 个观测包相关文件的大小/哈希、6 张最终图的哈希、128 条前向预算记录及相关执行日志哈希均一致；当前采集端口无监听。原有两个环境各 16 项通过的记录复用，没有新增测试运行、模型前向或采集。

回归记录与最终源码的唯一差异是缓存审计工具的图表坐标范围及图例字号；反向还原该行可精确匹配回归时源码哈希，最终缓存审计记录则匹配当前源码且退出码为 0。原 `repository_boundary.json` 的 67 个文件是当时已跟踪新增文件数，不包含上述 19 个遗留文件；接手复核另新增一份聚合记录，最终交付范围共 87 个新增文本文件。历史记录不覆盖。详见 [交接复核](m8j12_sources/handoff_verification.json)。
