# M8-A — PETR Official Checkpoint & RTX4090 Inference Feasibility

**退出：AssetNoGo。未取得官方训练后检测权重，后续依赖、算子、模型加载、GPU 前向及 CARLA 合同工作包全部 NotRun。** 这是当前服务器、固定官方入口与有限尝试下的资产可达性结论，不是 PETR 模型质量、Ada 兼容性或 24GB 容量的否定结论。

基线 `478070b51f03daf7e4ff2e7cb0b0a96e7507d9ba`；独立分支 `feat/m8a-petr-p0-feasibility`。本轮不训练、不启动 CARLA、不接视觉控制，不修改旧环境、默认模型、Depth/HeightSplat 或控制路径。

## 实际执行和退出依据

先复核 [M6-A 失败记录](m6a_perception_framework_feasibility.json)及固定官方源码。官网 README 的原始 PETR-r50-c5-1408x512 行确实将所选配置对应到 Google 权重入口，并另列百度官方分享。已核验本地源码提交与 README/config/LICENSE SHA。[固定官方 README](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/README.md)。

下载协议与执行源码在 **`242487ef96038b67670ed3f720c39c892df3e136`** 冻结后执行，实际访问过程约 **18.14s**：

| 请求 | 实际结果 | 能证明什么 |
|---|---|---|
| Google 官方文件页 | curl exit28，连接超时，约7.71s，HTTP000/0字节 | 此次服务器未连通文件页 |
| Google 对应公开下载 | curl exit28，连接超时，约7.71s，HTTP000/0字节 | 此次未取得 checkpoint 字节 |
| 百度官方分享页 | HTTP200 | 分享页面可访问 |
| 官方公开提取码 `petr` | HTTP200，errno=0 | 本次提取码验证成功 |
| 认证后的分享页 | HTTP200 | 页面响应存在，不等于权重下载 |
| R50-C5 目标目录 | HTTP200，但业务 errno=-7、空文件列表 | 本次未取得唯一目标条目或下载链接 |

没有继续请求百度文件下载，没有第三方镜像、登录绕过、验证码规避或无界重试。百度 `-7` 未被臆断为“文件永久删除”：服务端响应、客户端会话/参数兼容和入口变化尚不能区分。本次脚本的公开页面字段解析没有获得独立成功对照，这也是访问结论的限制。Google 失败也不能外推所有网络均不可访问。

历史百度记录中的 `epoch_24.pth` **435218261 bytes** 只是 M6-A 目录值；本轮没有该文件，不能产生 checkpoint SHA、读取结果或 Tensor 形状。[本轮资产报告](m8a_official_assets.json)、[冻结与逐请求记录](m8a_execution_logs/official_attempt_01/freeze.json)、[实际结果](m8a_execution_logs/official_attempt_01/outcome.json)。

Apache-2.0 代码许可证已核验，作者提供模型用于评价/研究的入口也已确认；没有获得文件内声明或独立 checkpoint 使用条款，不声称拥有额外再分发/商业权利。本轮没有分发权重。[固定许可证](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/LICENSE)。

## 六项独立回答

| 门禁 | 结论 | 实际证据 |
|---|---|---|
| A AssetAccess | **AssetNoGo** | 官方入口确认，但完整训练后 checkpoint 未取得 |
| B Dependency | **NotRun** | 按上游失败停止；没有安装新 Python/Torch/MMCV 或尝试 CUDA 兼容修复 |
| C ModelLoad | **NotRun** | 无模型字节；Head覆盖、Missing/Unexpected Keys、参数量均未知 |
| D Forward | **NotRun** | `[1,6,3,512,1408]` 原规格、SyntheticInputSmoke 均未执行 |
| E Resource | **NotRun** | 未测模型峰值显存/时延；设备存在不能证明模型可运行 |
| F Contract | **NotRun** | 按条件门禁未新增 CARLA/PETR 投影或非方盒测试，未开发完整 Adapter |

主机只读 inventory 为 RTX4090、24564MiB、驱动580.126.09；没有调用 PETR CUDA 或使用现有 Torch 环境加载外部 pickle。输入、输出、显存曲线、OOM 与 CUDA 错误字段为空，不能用零值替代未测。

## 代码、测试与保全

新增 [协议](../configs/m8a/protocol.json)、[官方入口采集工具](../tools/m8a_assets.py)、[报告与条件门禁工具](../tools/m8a_report.py)及[测试](../tests/test_m8a_assets.py)。文件使用排他创建，历史结果不能被覆盖；HTML、目录大小或仅有 ZIP 文件头都不能触发后续资产通过。

**11 项合成单元测试通过**，最终记录0.26s；覆盖伪 checkpoint、逐项资产前置条件、访问限制停止及旧证据保护。[实际测试日志](m8a_execution_logs/asset_gate_tests_final.txt)。它们验证工具行为，不能作为 PETR 算子或模型测试。

实际保全检查：基础提交的已跟踪文件无修改/删除；M7-E清单525/525文件大小及SHA一致；代表Mobile与Direct模型SHA一致；原 `.venv` 和 `.venv-gpu` Python/软件包版本清单相同；M6-A失败报告及历史manifest SHA一致。[保全记录](m8a_execution_logs/preservation.json)。没有重算全量外部训练数据，也不声称逐字节审计每个安装包。

报告生成代码另在 `e0bfa7d` 冻结后执行；每个 JSON 保存实际生成提交、源码 SHA 和协议 SHA。最终交付 manifest 对报告、失败记录、文档及工具封存，见[交付清单](m8a_release_manifest.json)。没有环境锁文件、输入输出数组或显存曲线，是因为相应条件实验未运行，不是遗漏了成功实验的记录。

M5-O `ResearchCompleted / CandidateNoGo`、M7 `ResearchCompleted / OfflinePlannerResearchGo / OnlinePlannerNoGo`、ControlNoGo、RuntimePartial、Occupancy NotPassed 和 SafetyNotValidated 全部继承，不自动替换框架或扩展研究。

复现和条件范围见[实现文档](../docs/M8A_IMPLEMENTATION.md)；退出风险见[风险决策](m8a_risk_and_exit_decision.md)及[架构决策](../docs/M8A_ARCHITECTURE_DECISION.md)。
