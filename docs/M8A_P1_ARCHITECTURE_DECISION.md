# M8-A P1 技术决策

结论：**FeasibilityPassed_QualityNotValidated**。

官方原始PETR-R50-DCNv2-C5已在RTX4090 24GB上完成完整六摄GPU前向。558张量严格匹配、900Query/6Decoder/10类及9个DCNv2不变；24次输出有限且一致。由此可以结束“官方检测权重无法取得/旧框架在4090上是否能运行”的当前条件性阻塞，但不能改写历史失败。

本次采用Torch2.0/cu118+MMCV1.7受控兼容路线，旧MMDetection接口保留，PETR源码不改。补丁包括版本上限、构建头文件/C++17、稀疏注册和Numba导入，不是官方Torch1.9环境的逐比特复现，也不包含训练兼容性证明。完整补丁和二进制身份已经封存。

| 决策对象 | 状态 | 理由 |
|---|---|---|
| 原规格离线推理工程可行性 | Passed | 真实4090、CUDA DCNv2、严格完整权重、六摄完整分辨率 |
| 最小CARLA输入/坐标合同 | Passed，范围受限 | M7-A真实RGB/K/T及已知非方形盒检查；无完整Adapter |
| PETR在CARLA上的感知质量 | NotValidated | 本轮模型前向为合成输入 |
| 训练/微调可行性与预算 | NotRun | 未做backward、优化器或训练显存测试 |
| 默认模型采用、视觉控制 | NotApproved | 没有定位误差、风险覆盖和安全闭环证据 |
| 共享GPU严格20Hz | NotValidated | 单模型P50约53.72ms，且20次样本不足以评价长尾 |

建议下一阶段**单独批准一个只读CARLA Adapter和冻结权重零样本诊断Pilot**。先完成六摄图像增强/坐标/类别适配，审查裁剪丢失和域偏移，在预声明目标匹配及全候选负担协议下测量；不得将公开nuScenes分数搬作CARLA成绩。该建议不是本轮自动实施授权。

需要独立审查的合同包括真实Actor Box局部中心/完整姿态、相机排列和nuScenes参考帧差异、可见性与一对一关联、类别不支持情况、分距离XY误差/严重错误/风险覆盖、全候选负担，以及RGB-only推理与Evaluator GT隔离。现有历史测试只用于回归；如后续训练或选择，须新独立最终评价。

当前显存余量只支持“推理容量可容纳”的判断，不能据此批准联合训练或共享GPU同时运行CARLA。默认Encoder、Depth Bin、HeightSplat、Planner、Safety、Controller、Physics保持原状。M5 CandidateNoGo和旧Runtime Partial不因本轮成功被关闭。
