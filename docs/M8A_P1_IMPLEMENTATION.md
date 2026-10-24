# M8-A P1：实际实现与复现

基础提交 `f4c49a5dbf20f5ee4870f78c93eaf0f87ae2434e`；独立分支 `feat/m8a-petr-forward-compat`。核心成功实验 `probe13` 与只读合同实验 `camera01` 的执行提交均为 `09c14d5a7a457d642bb16533adb76757f118359b`，精确脚本 SHA 见各执行 JSON。执行时的工作区文件身份由 SHA 补充，不把最后的文档提交冒充实验提交。

## 实际范围

官方原始 PETR-R50-DCNv2-C5，源码 `f7525f93467a33707ef401c587a52d5e7b34de74`，原配置 `projects/configs/petr/petr_r50dcn_gridmask_c5.py`。官方检测权重 435,218,261 字节，SHA256 `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`，本轮没有重新下载。

独立目录 `/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime`。所有下载、编译、模型执行均实际在这台 RTX4090 上进行。没有启动 CARLA、模型训练或驾驶控制。图像前向是 **SyntheticInputSmoke**，不是 nuScenes 或 CARLA 感知评测。

## 独立版本和兼容补丁

Python3.8.20、Torch2.0.0+cu118、torchvision0.15.1+cu118、MMCV-full1.7.2、MMDetection2.24.1、MMSegmentation0.20.2、MMDetection3D0.17.1、NumPy1.19.5、Numba0.53.1、einops0.4.1。完整依赖、wheel、CUDA 扩展、源码 SHA 见 [environment lock](../reports/m8a_p1_environment_lock.json)。系统 CUDA12.8/驱动及 HS 两个虚拟环境没有变更。

MMCV 使用官方 cu118/torch2.0 预编译包。旧 MMDetection3D 的 11 个扩展用隔离 CUDA11.8.89/GCC11.4 编译为 sm89。MMCV 二进制包含 sm86，并在 sm89 上实际执行 DCNv2；没有声称预编译包本身含 sm89，也没有声称直接观测了驱动最终选择的 cubin。

以下改动仅作用于新隔离环境，完整差异在 `reports/m8a_p1_logs/*.patch`：

- 三个旧框架的 MMCV 最大版本显式设为1.7.2；保留真实版本检查。
- 移除6处旧 THC 头文件/未使用声明，其中3处补充 ATen CUDAContext 显式声明，保留原 CUDA Stream 调用。
- 稀疏扩展构建标准 C++14→C++17，以满足 Torch2.0 头文件。
- 旧稀疏卷积注册设 `force=True`，明确选择旧 MMDetection3D 实现，解决 MMCV1.7 重名注册。PETR 本模型不使用稀疏卷积。
- NumPy不变；Numba0.53.1 对应警告类从 `numba.core.errors` 导入。相关训练增强没有运行。
- Ninja1.11.1.1 的 WHEEL 元数据在 Tag 前多一个空行；去掉空行后 `pip check` 通过，Ninja 二进制不变。

PETR 项目源码和数学结构没有修改。唯一运行时配置差异是 `img_backbone.pretrained=None`，跳过冗余 MSRA 初始化文件；完整558个最终张量严格加载后逐张量 `torch.equal`，没有新初始化值残留。900 Query、6 Decoder、10类别、9个DCNv2和原归一化全部保留。

Git 完整克隆和大文件下载出现截断；损坏文件被拒绝。PyTorch 分段下载最终匹配官方索引哈希。MMDetection3D 官方 PyPI sdist 缺少头文件及部分kernel，从同标签官方稀疏 Git 检出补齐，提交 `f1107977dfd26155fc1f83779ee6535d2468f449`。没有更换版本路线。

## 权重读取隔离

原始 `.pth` 仅在 bwrap 无网络、只读系统/模型/隔离运行库沙箱内，以 CPU `torch.load(weights_only=True)` 读取。输出558个纯数值数组 NPZ，SHA `cf33dccb1d549c425a89cf346d6f05401c49f6cccbd5a579cb7d8937b5a6d591`，不导出 optimizer。

宿主禁止挂载新 `/proc`，空 `/proc` 又使 CUDA 返回304。最终 GPU 进程使用新 venv、清理继承环境、`unshare --net`，读取哈希固定的 NPZ 且 `allow_pickle=False`。GPU进程保留宿主文件系统可见性，**不是文件系统沙箱**；原始 Pickle 没有在该进程反序列化。两层隔离不能混称。

## 已执行和可复现命令

以下命令在仓库根目录执行。所有报告/实验目录排他创建；复跑必须使用新的 run-id。旧结果不删除、不覆盖。

```bash
# 只读重新核验唯一官方检测权重
sha256sum /root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/assets/epoch_24.pth

# 工程合同和旧资产工具回归
.venv/bin/python -m pytest -q tests/test_m8a_p1.py tests/test_m8a_assets.py tests/test_m8a_proxy.py

# 现成隔离环境中的原规格执行，结果在 artifacts/reproduce01
python3 -m tools.m8a_p1_isolate --run-id reproduce01

# 只有该次完整前向 Passed 才运行最小只读相机检查
python3 -m tools.m8a_p1_camera_launch --forward-run reproduce01 --run-id camera_reproduce01

# 单独复查安全 CPU 提取，不替换原 safe_extract 输出
python3 -m tools.m8a_p1_isolate --run-id extract_reproduce01 --extract-only

# 如确需重建相同扩展：已验证工具链必须存在
python3 -m tools.m8a_p1_build --attempt reproduce01
```

首次安装的实际全部 subprocess 命令、cwd、超时和耗时保存于 `reports/m8a_p1_logs/*.json`，下载发生时代理显式传入子进程。`PIP_CONFIG_FILE=/dev/null` 防止全局 pip 配置进入后续安装。早期 torch本地安装日志打印过系统默认索引，但安装目标是已哈希校验的本地wheel且 `--no-deps`，没有从该索引取得Torch。

从零重建可先核验环境锁中的官方 wheel 和源码 SHA，再用锁定 wheel 离线安装、恢复已封存补丁源码、使用隔离 CUDA11.8 构建。不要盲目再次运行 `--step bootstrap` 或旧 `--step sources`：它们是首次尝试入口，现有目录会拒绝，且实际成功路径包含留痕的分段下载/源码补齐。最终完整已打补丁源码包及 SHA 在 `m8a_p1_logs/patched_source_archive.json`；不依赖不可复核的隐式手工改动。没有提供会自动开始训练的命令。

## 测量口径

固定输入 `[1,6,3,512,1408]`、FP32、关闭 autocast/TF32。原900×1600合成BGR像素通过官方 resize/crop/normalize/pad；相机是右手系环视rig，具有非零安装位移和不同朝向，投影矩阵不是六个单位阵。合成数据和标定不是nuScenes真实样本。

冷前向1次、预热3次、CUDA同步测量20次。预处理、H2D和JSON序列化在稳态计时外；完整backbone/head/decoder/box解码在内。首轮额外检查各DCNv2输出和Head有限性，之后移除hook。冷前向前已运行基础Torch/DCN探针，因此400.92ms是本进程首个完整模型前向，不是清空驱动缓存的纯冷启动。CUDA初始化183.10ms单独报告。

PyTorch峰值allocated/reserved是进程内实际记录；NVML每250ms采样，宿主PID与容器PID不同。1752MiB是同期唯一新增GPU进程的**时间关联采样最大值**，不是直接PID命名空间映射证明或瞬时精确峰值。原有436MiB进程仍存在，没有声称独占GPU。

全部24次300×9 Box、score、label均保存于 `reports/m8a_p1_results/forward_results.json`。随机图像输出分数仅0.00110–0.00309，均为类别8；这是top-k候选输出合同，不是300个真实障碍物。不能拿合成输出讨论类别坍缩或感知精度。

## CARLA最小合同

使用M7-A已有 `Town01_m5o_train_00/00074004` 的六张RGB及K/T。固定HS顺序；模型没有消费ego_pose、Depth、Actor或GT字段。检查执行官方图像变换及独立已知非方形盒几何，未运行CARLA模型前向。

384×640→缩放2.2得到844×1408→裁剪 `(0,332,1408,844)`；K用同一增强矩阵更新。官方整数resize与名义scale有0.8像素高度舍入，保留其约定并披露，不假装像素采样完全无误差。前端裁剪影响需后续质量评价。

旧LiDAR盒是bottom center；内部PETR回归gravity center后减h/2。尺寸为width/length/height；局部平面Yaw转换 `yaw_HS=-yaw_CARLA`，`yaw_old_lidar=-yaw_HS-pi/2`。非方形盒角点最大差3.91e-7m，投影逆变换差8.88e-15m。任意坡度/roll/pitch Actor Box及动态同步尚不在本轮验证范围。

本阶段不批准训练、默认PETR替换、Occupancy安全Free、视觉控制、共享GPU严格20Hz。历史AssetNoGo和后续AssetAccessGo保持原文原SHA。
