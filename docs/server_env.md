# 实际环境检查（2026-10-19，Asia/Shanghai）

本次命令确实在开发容器内执行。工作区 `/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0`；服务端与大数据放 `/root/gpufree-data`。

| 项目 | 实测 |
|---|---|
| OS | Ubuntu 22.04.5 LTS / x86_64 |
| Python | 3.10.12；项目 `.venv` |
| GPU | NVIDIA RTX 4090，24564 MiB |
| Driver | 580.126.09 |
| Vulkan | `vulkaninfo --summary` 成功枚举 NVIDIA 专有驱动及 RTX 4090 |
| 容器 CPU | cgroup quota 1400000/100000 = 14 核配额 |
| 容器内存 | cgroup memory.max = 53687091200 = 50GiB |
| /dev/shm | 16GiB |
| 数据盘 | 初查约 344GiB 可用；下载/采集后会减少 |
| 根 overlay | 初查约 26GiB 可用，不用于大型仿真包 |
| CARLA Python API | 0.9.15 已安装、导入、Transform 数学测试通过 |
| 初始 CARLA 连接 | 127.0.0.1:2000 超时；初始未安装服务端 |

`free -h` 显示宿主机 503GiB，不能当作容器限额。`nvidia-smi` 的 CUDA 13.0 是驱动支持上限，本次未安装或验证训练环境。

依赖通过环境默认的清华 PyPI 镜像安装，实际版本锁定在 `requirements-sim.lock.txt`，`pip check` 无冲突。CARLA 大包使用官方 GitHub Release 指向的 HTTPS 地址；本地 SHA256 仅用于后续重放校验，并非官方独立签名。

原始日志：`logs/m0_inventory.log`、`logs/vulkan_summary.log`、`logs/carla_initial_smoke.log`。运行过程使用 UTC 元数据；宿主 shell 原有日志显示 UTC+8，不代表用户时区。

交接包校验：所有实际存在的原始文件 SHA256 与 MANIFEST 一致；清单额外列出的根目录 Markdown 副本、Word 版本未随当前工作区提供。完整 Markdown 在 docs 下，已阅读全文及架构图。没有发现 AGENTS.md。原始包不是 Git 仓库，因此本次初始 run manifest 的 git_sha 为 null；不可编造 commit。

## 最终实测补充

官方CARLA包下载、解压成功，安装目录 `/root/gpufree-data/simulators/CARLA_0.9.15`，约19GiB；本地压缩包SHA256见 `reports/carla_package.local.sha256`。使用已有ubuntu用户运行。

Town10HD_Opt离屏启动、RPC版本、真实RGB/Depth、六摄30秒录制均成功。需移除DISPLAY并选择NVIDIA ICD（见启动脚本）。Town03地图切换仍会在Vulkan pipeline creation阶段SIGSEGV；关闭异步管线编译的实验也失败。cgroup OOM计数为0。最后一次试验进程已崩溃退出。

第二轮：Epic画质成功加载Town03并重复重载，完成全部M2验证；Low仍不受支持。启动脚本现默认Epic。验收后主动停止本轮服务端释放GPU。见 `../reports/milestone_02.md`。
