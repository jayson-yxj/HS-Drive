# M8-A P1：PETR六摄GPU可行性结果

**核心验收通过：FeasibilityPassed_QualityNotValidated。** 官方训练后原始PETR-R50-DCNv2-C5已在实际RTX4090 24GB上，使用完整六摄 `[1,6,3,512,1408]`、FP32、原900Query/6Decoder/10类和真实DCNv2完成前向。不是CARLA感知质量通过。

| 门禁 | 实际结果 |
|---|---|
| AssetAccess | 原435,218,261字节权重重新核验SHA一致，没有重新下载 |
| Dependency | 独立Python3.8/Torch2.0+cu118/MMCV1.7.2，保留旧MMDetection3D0.17.1；受控补丁后pip check通过 |
| Operator | 4090实际DCNv2 kernel，重复有限且一致；11个旧框架扩展编译完成，未宣称所有未使用LiDAR算子均实测 |
| ModelLoad | 35,805,405参数，558/558张量Key/Shape匹配，strict=True，加载后全部逐元素相等 |
| Forward | 冷启动1+预热3+同步测量20；完整网络9个DCNv2调用，输出300×9 Box、scores、labels，无NaN/Inf |
| Resource | 峰值allocated1.029GiB、reserved1.252GiB；NVML时间关联采样最大1752MiB，适配24GB推理 |
| Contract | M7-A已记录六摄RGB/K/T只读检查通过，含增强K/深度符号/非方形盒/Yaw；无完整CARLA Adapter |

实际同步测量 **P50 53.72ms、P95 53.99ms、Max54.08ms**；首个完整模型前向400.92ms，CUDA初始化183.10ms。首轮之前已有Torch/DCN探针，不能把400.92ms叫清空缓存后的纯冷启动。稳态计时不含预处理/H2D/JSON，20次样本不认证P99或最坏时间。单模型已经超过50ms，不关闭共享GPU严格20Hz门禁。

模型输入明确为 **SyntheticInputSmoke**：六摄标定语义一致，RGB为合成随机像素。24次结果完全一致，但300个输出是极低分top-k假设（0.00110–0.00309），不是物理障碍检测结果。真实nuScenes/CARLA质量、训练、视觉闭环均NotRun。

相机合同使用已有M7-A一帧六摄RGB。640×384按官方规则缩放裁剪至1408×512，会裁去较多上部图像，质量影响仍未知。旧版LiDAR Box使用bottom center、width/length/height及特殊Yaw约定，已用非方形盒验证；角点最大差3.91e-7m，投影逆变换差8.88e-15m。仅证明基本适配合同，不证明完整Actor姿态、动态同步或感知质量。

保留了下载截断、Git超时、沙箱/proc、缺失构建头文件、旧API注册及导入问题的失败证据。原始checkpoint只在无网络/只读文件沙箱中安全读取；GPU进程使用数值NPZ、禁止Pickle，独立环境无网络。PETR数学源码没有改变；所有兼容补丁与二进制SHA已记录。主项目两个虚拟环境包清单不变，历史报告清单复核一致，无默认模块修改。

新增与历史资产工具回归 **19 passed**。源码执行提交 `09c14d5a7a457d642bb16533adb76757f118359b`，精确运行脚本SHA和命令见执行日志，最终文档提交另由Git记录。

- [独立验收状态](m8a_p1_acceptance.json)
- [环境、二进制和资产身份](m8a_p1_environment_lock.json)
- [逐Key权重审计](m8a_p1_checkpoint_key_audit.json)
- [完整前向结果](m8a_p1_forward_results.json) / [全部24次输出](m8a_p1_results/forward_results.json)
- [显存与计时口径](m8a_p1_gpu_memory_latency.json)
- [相机合同](m8a_p1_camera_contract.json) / [失败账本](m8a_p1_failure_ledger.json)
- [复现与实现](../docs/M8A_P1_IMPLEMENTATION.md) / [技术决策](../docs/M8A_P1_ARCHITECTURE_DECISION.md)

下一步建议单独批准只读CARLA适配与冻结权重质量诊断Pilot。本轮不自动执行训练、默认模型替换、视觉控制或新闭环；感知质量、Occupancy安全Free、纯视觉驾驶安全和历史Runtime Partial继续开放。
