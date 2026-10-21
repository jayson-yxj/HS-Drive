# M5-K — 预声明架构与研究决策

基线 `db03730`，独立分支 `feat/m5k-target-depth-geometry`。原 MobileNet 101/102、ResNet18、48 LID、HeightSplat、四层网格和所有 Runtime/Physics 默认行为保持不变。

## 定义

A 为可见非目标来源，不能解释为安全 Free。B 为目标射线的概率落点不在 GT 端点粗格，不能据此断定物理空体积。E 来源 Unknown 单列。低 q 贡献和多射线累积是不同描述。`ray_bin_count` 与按 camera/pixel 去重的射线支持不同；去重后的射线也存在相关性。

所有深度分布对照使用同一 P8 中心、可靠深度 Mask、GT 在 BEV 内的共同分母。光轴 Z 与像素 K/T 反投影合同不变。原生 S16 对照独立标记，不借不同采样分母宣称深度改善。表面 Precision 是可靠可见首表面条件指标，非完整空间障碍检测或可行驶认证。

## 决策顺序

1. 原 Head P8 对比原生 S16 概率双线性重采样。`align_corners=False`、边缘复制、FP32 逐射线归一化。中心 8+16i 映射到 4+8j；不插值 logits。历史 Replay 仅诊断。
2. 新 Train 和 Val 各至少四个位置、100 条 20–35m 严格关联且在 BEV 内的 Actor P8 射线，至少两类各 10 条。任一失败不训练，不扩大 Unknown 或放宽 Mask。
3. 仅对齐未解决问题且支持通过时，执行零初始化轻量残差；B/C 同容量、相同初始化、样本顺序、600 步预算和 Loss。仅 Mobile 条件通道有无不同。不存在新 S4 Depth 网络。
4. 自适应额外 S4 射线只由冻结 Mobile q≥0.9 选择；均匀对照逐帧逐种子预算精确相等。召回分母包含未选 ROI 的所有可靠目标。
5. 阈值固定 0.2。BEV 资格要求两个种子 Val Precision 至少 +1pp、同层 1m Recall 至少 +5pp、背景 FPR 增量≤0.1pp、Actor 检出不降，并保留近场与 Bias 门禁。Test 不能改变方法、Loss、阈值。
6. 计算输入通路只测离线完整延迟和实际 CARLA BGRA buffer 回放。不得将各模块 P95 相加。无视觉控制、无 Runtime 安全升级。

结果决策在实验完成后追加；原预声明协议保存在 `configs/m5k_protocol.json`。
