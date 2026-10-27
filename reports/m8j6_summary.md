# M8-J6 — 八帧接入准备完成，真实 Smoke 被数据访问阻塞

检查日期：2026-10-27；机器证据使用显式 UTC 时间。
基础提交：`ffa9b0e6b73376e0bbfb23826290041264a70dd2`。
分支：`feat/m8j6-nuscenes-mini-fp32-smoke`。

**结论：Partial。没有执行真实 nuScenes 前向、反向或 optimizer step。**
阻塞为本机没有找到 `v1.0-mini`，也尚未取得已登录且确认条款的官方获取方式。
这不是模型、显存或监督质量的失败，不能据此判它们通过或失败。

## 已实际完成

- 对五个数据根目录进行本地检索，扫描 11,232 个目录，未找到 Mini 压缩包或 `sample.json` / `scene.json`。可用磁盘约 159 GiB；该结果不是下载或解压完整性证明。
- 创建独立分支，冻结 M8-J5C 资产和已有权重哈希；历史 A=Partial、B=NoGo、C=NotRun 保持原样。Bench2Drive 仅保留为缺标、Ignore 和适配回归集。
- 固定官方十类顺序、六摄顺序、900 Query、六层 Decoder、原始几何范围和固定预处理。没有升级依赖或修改默认模型。
- 实现 metadata-only 选帧与关联预检；13 项合成测试通过。
- 在原 PETR Python 3.8 环境中，禁网、禁 GPU 运行官方实现的 12 项 CPU 合成合同测试，通过中心、尺寸、Yaw 八角点、速度缺失处理、异步相机坐标转换与裁剪后标定检查。
- 真实数据预检非零退出，具体缺少 `data/v1.0-mini/scene.json`；未创建虚假的八帧清单或训练结果。

## 重要适配发现

`mini_train` 不等于完整版 `train`：`scene-0553`、`scene-0796` 属于完整版验证集。
预声明规则取 `mini_train ∩ train` 按名称排序的前两个场景，即 `scene-0061`、`scene-0655`，每场景按 `(timestamp, token)` 取最早四个关键帧；异常样本隔离，不替换凑数。
真实 token、48 张相机图像、原始标注关联仍待数据，不能用场景名称冒充八帧已冻结。

该选择与官方预训练的 nuScenes train 分布存在预期重叠；没有逐样本训练日志，不能断言每个 token 被该 checkpoint 训练过。后续 Smoke 只验证工程更新链路，不能用于泛化评价。

官方检测负样本遵循 Hungarian 未匹配 Query 规则，不要求 Occupancy 空空间证明。不会搬用 Bench2Drive 的 N=0 或整帧 Ignore。发现具体未解决的缺标或几何异常时，隔离对应样本。

## 访问状态及必要操作

官方下载页面直连曾返回 HTTP 200，随后一次请求在 TLS 握手超时；旧代理 `127.0.0.1:7897` 连接拒绝。这仅是网页访问证据，不是数据下载成功。没有请求 Mini 载荷，没有第三方镜像或访问绕过。

[官方获取说明](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup)要求登录账号并确认使用条款。
已经向用户请求现有合法 Mini 路径或官方访问方式，尚未收到回答。需要以下任一项：

1. 已合法取得并解压的 Mini 根目录，含 `v1.0-mini/`、`samples/` 等；或
2. 用户在[官方页面](https://www.nuscenes.org/download)完成账号登录与条款确认后提供 Mini 获取方式。无需提供账号密码；不把链接中的临时凭据提交仓库。

当前任务已经授权条件满足后的三步更新，不需要重复申请训练权限；尚缺的是实际数据访问条件。

## 真实实验状态

| 项目 | 本轮结果 |
|---|---|
| 八帧真实文件、标注及时间戳核验 | NotRun；token 尚不可得 |
| 六摄全目标与 BEV 可视化、实际查看记录 | NotRun；无图片，不伪造签核 |
| FP32 冻结前向与 Loss | NotRun |
| frame × decoder_layer 的 P/I/N、Loss、归一化 | 无真实记录；不填推测数值 |
| 分类/回归梯度、参数与缓冲区变化 | NotRun |
| 显存、推理/反向耗时 | 未测量 |
| 三次 optimizer step、保存恢复 | 0 步；NotRun |
| 下一轮小集合过拟合 | 暂不能进入；缺真实八帧与三步证据 |

最小续跑入口、官方代码依据及三步固定配置见 [实施说明](../docs/M8J6_IMPLEMENTATION.md)。
当前只实现数据预检和合同测试入口，未将尚无真实数据验证的训练脚本包装成可用链路。

## 证据入口

- [验收](m8j6_acceptance.json)、[八帧选择状态](m8j6_eight_frame_manifest.json)
- [官方适配合同](m8j6_official_pipeline_audit.md)、[源码与权重哈希](m8j6_source_lock.json)
- [数据预检实际错误](m8j6_sources/preflight.json)、[访问记录](m8j6_access_status.json)
- [FP32 基线状态](m8j6_fp32_baseline.json)、[三步更新状态](m8j6_three_step_smoke.json)

没有启动 CARLA、训练或 GPU 模型任务；没有新增原图、逐对象标签或权重到仓库；没有合并 main。
